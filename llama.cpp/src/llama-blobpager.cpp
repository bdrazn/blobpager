// llama-blobpager.cpp — E7b hybrid hit/miss execution module (blobpager Phase-1 PoC).
//
// See llama-blobpager.h for the design. One-sentence summary: GPU pools serve
// the pinned experts in-graph; a scheduler eval-callback computes exactly the
// missed experts on the CPU and injects their weighted contribution through a
// GPU correction tensor that build_moe_ffn adds to the MoE output.

#include "llama-blobpager.h"

#include "llama-model.h"
#include "llama-hparams.h"
#include "llama-impl.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

// ------------------------------------------------------------------ state

namespace {

struct bp_layer {
    bool has_pins = false;
    std::vector<int32_t> pins;      // slot -> expert id
    std::vector<int32_t> slot_of;   // expert id -> slot, or -1 if miss
    ggml_tensor * pool[3]  = { nullptr, nullptr, nullptr }; // up, gate, down (npins+1 slots)
    ggml_tensor * slots    = nullptr;  // [n_expert] I32 GPU
    ggml_tensor * corr     = nullptr;  // [n_embd, maxt] F32 GPU
    const ggml_tensor * cpu[3] = { nullptr, nullptr, nullptr }; // CPU-resident full expert tensors
    std::vector<float> corr_staging;   // n_embd floats, reused per token
    bool corr_dirty = false;
};

struct bp_stats {
    int64_t pass_invocations = 0;
    int64_t miss_draws       = 0;
    int64_t bytes_read       = 0;
    double  pass_us          = 0.0;
    double  graph_us         = 0.0;
    double  io_us            = 0.0;
};

struct bp_state {
    bool attempted = false;
    bool active    = false;
    bool broken    = false;

    int  n_serving  = 0;
    int  n_expert   = 0;
    int  n_embd     = 0;
    int  n_ff       = 0;
    int  maxt       = 8;
    bool norm_w     = false;
    float w_scale   = 1.0f;

    bp_layer layers[LLAMA_MAX_LAYERS];
    ggml_tensor * moe_in_stash[LLAMA_MAX_LAYERS] = {};

    ggml_backend_t cpu_backend = nullptr;

    // user callback chaining (accounting observers such as blobpager-pager)
    void * user_cb = nullptr;   // ggml_backend_sched_eval_callback
    void * user_ud = nullptr;

    bp_stats stats;
};

static bp_state g;

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ------------------------------------------------------------ pins parsing

struct bp_pins {
    int layer = -1;
    std::vector<int32_t> pins;
};

// same textual format as blobpager-pager's load_pins: the TENSOR lines are
// consumed (they carry GGUF offsets, which this module does not need — the
// expert bytes are read from the model's own CPU-resident tensors)
static bool parse_pins(const char * path, std::vector<bp_pins> & out) {
    std::ifstream f(path);
    if (!f) {
        return false;
    }
    std::string tag;
    int nlayers = 0;
    if (!(f >> tag >> nlayers) || tag != "LAYERS") {
        return false;
    }
    for (int i = 0; i < nlayers; ++i) {
        bp_pins bp;
        int npins = 0;
        int64_t pool_bytes = 0;
        if (!(f >> tag >> bp.layer >> npins >> pool_bytes) || tag != "LAYER") {
            return false;
        }
        for (int t = 0; t < 3; ++t) {
            std::string name;
            uint64_t offset = 0;
            int64_t ebytes = 0;
            if (!(f >> tag >> name >> offset >> ebytes) || tag != "TENSOR") {
                return false;
            }
        }
        if (!(f >> tag) || tag != "PINS") {
            return false;
        }
        bp.pins.resize(npins);
        for (int p = 0; p < npins; ++p) {
            if (!(f >> bp.pins[p])) {
                return false;
            }
        }
        out.push_back(std::move(bp));
    }
    return !out.empty();
}

// ---------------------------------------------------------------- stats

static void write_stats() {
    const char * path = getenv("BLOBPAGER_STATS");
    if (path == nullptr || !g.active) {
        return;
    }
    std::ofstream f(path);
    if (!f) {
        return;
    }
    f << "{\n"
      << "  \"active\": " << (g.active && !g.broken) << ",\n"
      << "  \"broken\": " << g.broken << ",\n"
      << "  \"n_serving\": " << g.n_serving << ",\n"
      << "  \"maxt\": " << g.maxt << ",\n"
      << "  \"pass_invocations\": " << g.stats.pass_invocations << ",\n"
      << "  \"miss_draws\": " << g.stats.miss_draws << ",\n"
      << "  \"bytes_read\": " << g.stats.bytes_read << ",\n"
      << "  \"pass_us_total\": " << g.stats.pass_us << ",\n"
      << "  \"cpu_graph_us_total\": " << g.stats.graph_us << ",\n"
      << "  \"gpu_io_us_total\": " << g.stats.io_us << "\n"
      << "}\n";
}

// ------------------------------------------------------------------ setup

} // namespace (blobpager internals)

void llama_blobpager_setup(struct llama_model & model) {
    if (g.attempted) {
        return;
    }
    g.attempted = true;

    const char * pins_path = getenv("BLOBPAGER_PINS");
    if (pins_path == nullptr) {
        return;
    }
    if (const char * mt = getenv("BLOBPAGER_MAXT")) {
        g.maxt = atoi(mt);
        if (g.maxt < 1) {
            g.maxt = 1;
        }
    }

    const llama_hparams & hp = model.hparams;
    g.n_expert = (int) hp.n_expert;
    g.n_embd   = (int) hp.n_embd;
    g.n_ff     = (int) hp.n_ff();

    std::vector<bp_pins> lps;
    if (!parse_pins(pins_path, lps)) {
        LLAMA_LOG_ERROR("%s: cannot parse BLOBPAGER_PINS '%s'\n", __func__, pins_path);
        return;
    }

    // GPU buffer type from the model's own device list (weight tensors may
    // still be lazily materialized at this point, so do not anchor on one)
    if (model.devices.empty()) {
        LLAMA_LOG_ERROR("%s: model has no GPU devices; blobpager hybrid disabled\n", __func__);
        return;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(model.devices[0].dev);

    // per-layer bookkeeping
    for (const auto & bp : lps) {
        if (bp.layer < 0 || bp.layer >= LLAMA_MAX_LAYERS) {
            continue;
        }
        bp_layer & L = g.layers[bp.layer];
        L.has_pins = true;
        L.pins     = bp.pins;
        L.slot_of.assign(g.n_expert, -1);
        for (size_t s = 0; s < L.pins.size(); ++s) {
            const int32_t e = L.pins[s];
            if (e < 0 || e >= g.n_expert) {
                LLAMA_LOG_ERROR("%s: pin id %d out of range\n", __func__, e);
                return;
            }
            L.slot_of[e] = (int32_t) s;
        }
        g.n_serving++;
    }
    if (g.n_serving == 0) {
        return;
    }

    // CPU expert tensors (the -ncmoe placement keeps them host-resident)
    for (int il = 0; il < LLAMA_MAX_LAYERS; ++il) {
        bp_layer & L = g.layers[il];
        if (!L.has_pins) {
            continue;
        }
        const auto & ml = model.layers[il];
        const ggml_tensor * src[3] = { ml.ffn_up_exps, ml.ffn_gate_exps, ml.ffn_down_exps };
        for (int k = 0; k < 3; ++k) {
            if (src[k] == nullptr || src[k]->data == nullptr || !ggml_backend_buffer_is_host(src[k]->buffer)) {
                LLAMA_LOG_ERROR("%s: layer %d expert tensor %d is missing or not host-resident\n", __func__, il, k);
                return;
            }
            L.cpu[k] = src[k];
        }
    }

    // create the pools, slot tables and correction tensors in one GPU buffer
    const size_t n_tens = (size_t) g.n_serving * 5;
    ggml_init_params ip = { ggml_tensor_overhead()*n_tens + 1024, nullptr, true /* no_alloc */ };
    ggml_context * gctx = ggml_init(ip);
    if (gctx == nullptr) {
        LLAMA_LOG_ERROR("%s: ggml_init failed\n", __func__);
        return;
    }

    for (int il = 0; il < LLAMA_MAX_LAYERS; ++il) {
        bp_layer & L = g.layers[il];
        if (!L.has_pins) {
            continue;
        }
        const int64_t nslots = (int64_t) L.pins.size() + 1;
        for (int k = 0; k < 3; ++k) {
            const ggml_tensor * cpu = L.cpu[k];
            L.pool[k] = ggml_new_tensor_3d(gctx, cpu->type, cpu->ne[0], cpu->ne[1], nslots);
        }
        L.slots = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, g.n_expert);
        L.corr  = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, g.n_embd, g.maxt);
        L.corr_staging.resize((size_t) g.n_embd, 0.0f);
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(gctx, buft);
    if (buf == nullptr) {
        LLAMA_LOG_ERROR("%s: pool buffer allocation failed\n", __func__);
        ggml_free(gctx);
        return;
    }

    // fill: slot s gets pinned expert pins[s]'s full weight runs; the last
    // slot (dummy) is zeroed so miss draws contribute exact 0 on the GPU arm
    size_t max_ebytes = 0;
    for (int il = 0; il < LLAMA_MAX_LAYERS; ++il) {
        const bp_layer & L = g.layers[il];
        if (!L.has_pins) {
            continue;
        }
        for (int k = 0; k < 3; ++k) {
            max_ebytes = std::max(max_ebytes, (size_t) L.cpu[k]->nb[2]);
        }
    }
    std::vector<uint8_t> zeros(max_ebytes, 0);

    const double t0 = now_us();
    int64_t fill_bytes = 0;
    for (int il = 0; il < LLAMA_MAX_LAYERS; ++il) {
        bp_layer & L = g.layers[il];
        if (!L.has_pins) {
            continue;
        }
        const int64_t npins = (int64_t) L.pins.size();
        for (int s = 0; s <= (int) npins; ++s) {
            const bool dummy = (s == (int) npins);
            const int32_t e  = dummy ? -1 : L.pins[s];
            for (int k = 0; k < 3; ++k) {
                const ggml_tensor * cpu  = L.cpu[k];
                ggml_tensor       * pool = L.pool[k];
                GGML_ASSERT(cpu->type == pool->type && cpu->ne[0] == pool->ne[0] && cpu->ne[1] == pool->ne[1]);
                GGML_ASSERT(cpu->nb[2] == pool->nb[2]);
                const uint8_t * src = dummy ? zeros.data()
                                            : (const uint8_t *) cpu->data + (size_t) e * cpu->nb[2];
                ggml_backend_tensor_set(pool, src, (size_t) s * pool->nb[2], (size_t) cpu->nb[2]);
            }
            fill_bytes += 3 * (int64_t) max_ebytes;
        }
        // slot table: expert -> slot, miss -> dummy
        std::vector<int32_t> st(g.n_expert);
        for (int e = 0; e < g.n_expert; ++e) {
            st[e] = (L.slot_of[e] >= 0) ? L.slot_of[e] : (int32_t) npins;
        }
        ggml_backend_tensor_set(L.slots, st.data(), 0, (size_t) g.n_expert * sizeof(int32_t));
        // correction starts at zero
        ggml_backend_tensor_set(L.corr, zeros.data(), 0, (size_t) ggml_nbytes(L.corr) < zeros.size() ? ggml_nbytes(L.corr) : zeros.size());
    }
    const double t1 = now_us();

    // fill verification: read every slot back and compare against the source
    // expert bytes; a mismatch means the fill path is broken
    {
        int64_t mism = 0;
        std::vector<uint8_t> back(max_ebytes);
        for (int il = 0; il < LLAMA_MAX_LAYERS; ++il) {
            const bp_layer & L = g.layers[il];
            if (!L.has_pins) {
                continue;
            }
            const int64_t npins_v = (int64_t) L.pins.size();
            for (int s = 0; s <= (int) npins_v; ++s) {
                const bool dummy = (s == (int) npins_v);
                const int32_t e  = dummy ? -1 : L.pins[s];
                for (int k = 0; k < 3; ++k) {
                    const ggml_tensor * cpu = L.cpu[k];
                    ggml_tensor       * pool = L.pool[k];
                    ggml_backend_tensor_get(pool, back.data(), (size_t) s * pool->nb[2], (size_t) cpu->nb[2]);
                    const uint8_t * want = dummy ? zeros.data()
                                                 : (const uint8_t *) cpu->data + (size_t) e * cpu->nb[2];
                    if (memcmp(back.data(), want, (size_t) cpu->nb[2]) != 0) {
                        mism++;
                    }
                }
            }
        }
        LLAMA_LOG_INFO("%s: pool fill verification: %lld mismatches\n", __func__, (long long) mism);
    }

    g.active = true;
    g.cpu_backend = ggml_backend_cpu_init();
    if (g.cpu_backend == nullptr) {
        LLAMA_LOG_ERROR("%s: CPU backend init failed — hybrid cannot run miss passes\n", __func__);
        g.active = false;
        ggml_backend_buffer_free(buf);
        ggml_free(gctx);
        return;
    }

    atexit(write_stats);

    LLAMA_LOG_INFO("%s: blobpager hybrid armed: %d serving layers, %d pins + 1 dummy slot, "
            "pool %lld B filled in %.3f s, maxt %d\n",
            __func__, g.n_serving, (int) g.layers[0].pins.size(), (long long) fill_bytes,
            (t1 - t0) / 1e6, g.maxt);

    llama_blobpager_selftest(model);
}

bool llama_blobpager_active() {
    return g.active && !g.broken;
}

// GPU mul_mat_id correctness self-test on the real pool data: run one
// 8-lane call on the GPU against the layer-0 gate pool and compare with a
// CPU recompute from the same expert bytes.
void llama_blobpager_selftest(const struct llama_model & model) {
    if (!g.active || g.n_serving == 0) {
        return;
    }
    const bp_layer & L = g.layers[0];
    const int64_t npins = (int64_t) L.pins.size();
    if (npins < 8) {
        return;
    }

    // fixed ids: use the first 8 pins (definite hits)
    std::vector<int32_t> ids_v(8);
    for (int k = 0; k < 8; ++k) {
        ids_v[k] = L.slot_of[L.pins[k]];   // slot of pin k
    }

    // activation
    std::vector<float> x((size_t) g.n_embd);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = 0.001f * (float) ((int) (i % 97) - 48);
    }

    // ---- GPU arm
    ggml_init_params gip = { 64 * 1024 * 1024, nullptr, true /* no_alloc */ };
    ggml_context * gctx = ggml_init(gip);
    ggml_tensor * ids_g = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, 8, 1);
    ggml_tensor * x_g   = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, g.n_embd, 1);
    ggml_tensor * out_g = ggml_mul_mat_id(gctx, L.pool[1] /* gate */, x_g, ids_g);
    ggml_backend_buffer_t gbuf = ggml_backend_alloc_ctx_tensors_from_buft(gctx, ggml_backend_dev_buffer_type(model.devices[0].dev));
    if (gbuf == nullptr) {
        LLAMA_LOG_ERROR("%s: selftest alloc failed\n", __func__);
        ggml_free(gctx);
        return;
    }
    ggml_backend_tensor_set(ids_g, ids_v.data(), 0, ids_v.size() * sizeof(int32_t));
    ggml_backend_tensor_set(x_g, x.data(), 0, x.size() * sizeof(float));
    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, out_g);
    ggml_backend_t bdev = ggml_backend_dev_init(model.devices[0].dev, nullptr);
    ggml_backend_graph_compute(bdev, gf);
    std::vector<float> gpu_out((size_t) ggml_nbytes(out_g) / sizeof(float));
    ggml_backend_tensor_get(out_g, gpu_out.data(), 0, ggml_nbytes(out_g));

    // ---- CPU arm: same ids, same x, from the CPU expert tensors
    double maxd = 0.0;
    for (int j = 0; j < 8; ++j) {
        const int32_t e = L.pins[j];
        const ggml_tensor * gate = L.cpu[1];
        // gw = expert e rows [ne0, ne1]
        std::vector<float> acc((size_t) gate->ne[1], 0.0f);
        // dequant-free reference: use ggml on the CPU
        ggml_init_params cip = { 64 * 1024 * 1024, nullptr, false };
        ggml_context * c = ggml_init(cip);
        ggml_tensor * gw = ggml_new_tensor_2d(c, gate->type, gate->ne[0], gate->ne[1]);
        gw->data = (uint8_t *) gate->data + (size_t) e * gate->nb[2];
        ggml_tensor * xt = ggml_new_tensor_2d(c, GGML_TYPE_F32, g.n_embd, 1);
        xt->data = x.data();
        ggml_tensor * r = ggml_mul_mat(c, gw, xt);
        ggml_cgraph * cf = ggml_new_graph(c);
        ggml_build_forward_expand(cf, r);
        ggml_backend_t cpu_backend = ggml_backend_cpu_init();
        ggml_backend_graph_compute(cpu_backend, cf);
        std::vector<float> cpu_out((size_t) ggml_nbytes(r) / sizeof(float));
        memcpy(cpu_out.data(), r->data, ggml_nbytes(r));
        for (size_t i = 0; i < cpu_out.size(); ++i) {
            const float d = fabsf(gpu_out[(size_t) j * cpu_out.size() + i] - cpu_out[i]);
            if (d > maxd) {
                maxd = d;
            }
        }
        ggml_backend_free(cpu_backend);
        ggml_free(c);
    }
    LLAMA_LOG_ERROR("%s: GPU-vs-CPU pool mul_mat_id max diff = %.6f\n", __func__, maxd);

    ggml_backend_free(bdev);
    ggml_backend_buffer_free(gbuf);
    ggml_free(gctx);
}

// ---------------------------------------------------------- graph helpers

bool llama_blobpager_hybrid(int il, int64_t n_tokens, bool norm_w, float w_scale) {
    if (!llama_blobpager_active()) {
        return false;
    }
    if (il < 0 || il >= LLAMA_MAX_LAYERS) {
        return false;
    }
    bp_layer & L = g.layers[il];
    if (!L.has_pins) {
        return false;
    }
    if (n_tokens > g.maxt) {
        return false;
    }
    g.norm_w  = norm_w;
    g.w_scale = w_scale;
    return true;
}

ggml_tensor * llama_blobpager_remap(struct ggml_context * ctx, struct ggml_tensor * selected_experts, int il) {
    bp_layer & L = g.layers[il];
    // slots [n_expert] -> [1, n_expert]; get_rows over the top-k ids gives
    // [1, n_used, n_tokens]; reshape back to [n_used, n_tokens]
    ggml_tensor * s2     = ggml_reshape_2d(ctx, L.slots, 1, g.n_expert);
    ggml_tensor * rem    = ggml_get_rows(ctx, s2, selected_experts);
    return ggml_reshape_2d(ctx, rem, selected_experts->ne[0], selected_experts->ne[1]);
}

ggml_tensor * llama_blobpager_pool(int il, int which) {
    return g.layers[il].pool[which];
}

ggml_tensor * llama_blobpager_correction_view(struct ggml_context * ctx, int il, int64_t n_tokens) {
    bp_layer & L = g.layers[il];
    return ggml_view_2d(ctx, L.corr, g.n_embd, n_tokens, L.corr->nb[1], 0);
}

// ------------------------------------------------------------ miss pass

static const char kMoeIn[]  = "blobpager_moe_in-";
static const char kTopk[]   = "ffn_moe_topk-";

static bool tensor_read(const ggml_tensor * t, void * dst, size_t nbytes) {
    if (t == nullptr || t->data == nullptr) {
        return false;
    }
    if (ggml_backend_buffer_is_host(t->buffer)) {
        memcpy(dst, t->data, nbytes);
    } else {
        ggml_backend_tensor_get(t, dst, 0, nbytes);
    }
    return true;
}

static void do_miss_pass(int il, struct ggml_tensor * topk) {
    const double t0 = now_us();
    bp_layer & L = g.layers[il];

    const int64_t n_used = topk->ne[0];
    const int64_t n_tok  = topk->ne[1];

    // 1. ids
    std::vector<int32_t> ids((size_t)(n_used * n_tok));
    if (!tensor_read(topk, ids.data(), ids.size() * sizeof(int32_t))) {
        LLAMA_LOG_ERROR("%s: cannot read ids tensor for layer %d\n", __func__, il);
        g.broken = true;
        return;
    }

    // 2. probs: walk up from the top-k node to the softmax output
    ggml_tensor * probs = topk;
    for (int hop = 0; hop < 5 && probs != nullptr; ++hop) {
        if (probs->op == GGML_OP_SOFT_MAX) {
            break;
        }
        probs = probs->src[0];
    }
    if (probs == nullptr || probs->op != GGML_OP_SOFT_MAX || probs->ne[0] != g.n_expert || probs->ne[1] != n_tok) {
        LLAMA_LOG_ERROR("%s: cannot locate softmax probs for layer %d (shape mismatch)\n", __func__, il);
        g.broken = true;
        return;
    }
    std::vector<float> pw((size_t)(g.n_expert * n_tok));
    if (!tensor_read(probs, pw.data(), pw.size() * sizeof(float))) {
        LLAMA_LOG_ERROR("%s: cannot read probs for layer %d\n", __func__, il);
        g.broken = true;
        return;
    }

    // 3. FFN input activation
    ggml_tensor * mi = g.moe_in_stash[il];
    if (mi == nullptr || mi->ne[0] != g.n_embd || mi->ne[1] != n_tok) {
        LLAMA_LOG_ERROR("%s: moe_in stash missing or shape mismatch for layer %d\n", __func__, il);
        g.broken = true;
        return;
    }
    const double tA = now_us();
    g.stats.io_us += tA - t0;
    std::vector<float> x((size_t)(g.n_embd * n_tok));
    if (!tensor_read(mi, x.data(), x.size() * sizeof(float))) {
        LLAMA_LOG_ERROR("%s: cannot read moe_in for layer %d\n", __func__, il);
        g.broken = true;
        return;
    }
    const double tB = now_us();
    g.stats.io_us += tB - tA;

    // 4. per-token miss execution on the CPU
    const float clamp_lo = 6.103515625e-5f;   // same clamp as build_moe_ffn norm_w
    for (int64_t t = 0; t < n_tok; ++t) {
        // collect misses (expert id, weight) for this token, and the norm sum
        std::vector<std::pair<int32_t, float>> misses;
        float sum = 0.0f;
        for (int64_t k = 0; k < n_used; ++k) {
            const int32_t e = ids[(size_t) t * n_used + k];
            sum += pw[(size_t) e + (size_t) t * g.n_expert];
            if (L.slot_of[e] < 0) {
                misses.emplace_back(e, 0.0f);   // weight filled below
            }
        }
        if (misses.empty()) {
            if (L.corr_dirty) {
                std::vector<float> zz(g.n_embd, 0.0f);
                ggml_backend_tensor_set(L.corr, zz.data(), (size_t) t * g.n_embd * sizeof(float),
                                        (size_t) g.n_embd * sizeof(float));
                L.corr_dirty = false;
            }
            continue;
        }

        // gating weights: replicate the graph's weights path exactly
        const float denom = std::max(sum, clamp_lo);
        for (auto & m : misses) {
            float w = pw[(size_t) m.first + (size_t) t * g.n_expert];
            if (g.norm_w) {
                w /= denom;
            }
            if (g.w_scale != 0.0f && g.w_scale != 1.0f) {
                w *= g.w_scale;
            }
            m.second = w;
        }

        // one tiny CPU graph: gate -> up -> swiglu -> down per miss, scaled and summed
        const double g0 = now_us();
        ggml_init_params cip = { 64 * 1024 * 1024, nullptr, false /* alloc */ };
        ggml_context * c = ggml_init(cip);
        if (c == nullptr) {
            LLAMA_LOG_ERROR("%s: cpu ctx init failed\n", __func__);
            g.broken = true;
            return;
        }
        ggml_tensor * xt = ggml_new_tensor_2d(c, GGML_TYPE_F32, g.n_embd, 1);
        xt->data = x.data() + (size_t) t * g.n_embd;

        ggml_tensor * tot = nullptr;
        for (const auto & m : misses) {
            const int32_t e = m.first;
            const ggml_tensor * cpu_up   = L.cpu[0];
            const ggml_tensor * cpu_gate = L.cpu[1];
            const ggml_tensor * cpu_down = L.cpu[2];

            ggml_tensor * gw = ggml_new_tensor_2d(c, cpu_gate->type, cpu_gate->ne[0], cpu_gate->ne[1]);
            gw->data = (uint8_t *) cpu_gate->data + (size_t) e * cpu_gate->nb[2];
            ggml_tensor * uw = ggml_new_tensor_2d(c, cpu_up->type, cpu_up->ne[0], cpu_up->ne[1]);
            uw->data = (uint8_t *) cpu_up->data + (size_t) e * cpu_up->nb[2];
            ggml_tensor * dw = ggml_new_tensor_2d(c, cpu_down->type, cpu_down->ne[0], cpu_down->ne[1]);
            dw->data = (uint8_t *) cpu_down->data + (size_t) e * cpu_down->nb[2];

            ggml_tensor * gt = ggml_mul_mat(c, gw, xt);   // [n_ff, 1]
            ggml_tensor * ut = ggml_mul_mat(c, uw, xt);   // [n_ff, 1]
            ggml_tensor * hid = ggml_swiglu_split(c, gt, ut);
            ggml_tensor * o = ggml_mul_mat(c, dw, hid);   // [n_embd, 1]
            ggml_tensor * sc = ggml_scale(c, o, m.second);
            tot = (tot == nullptr) ? sc : ggml_add(c, tot, sc);
        }

        ggml_cgraph * gf = ggml_new_graph(c);
        ggml_build_forward_expand(gf, tot);
        ggml_backend_graph_compute(g.cpu_backend, gf);

        // tot lives in the raw CPU ctx pool: data is host memory, but buffer is
        // NULL (only backend-allocated tensors get buffers) — read data directly
        GGML_ASSERT(tot->data != nullptr);
        memcpy(L.corr_staging.data(), tot->data, (size_t) g.n_embd * sizeof(float));
        ggml_backend_tensor_set(L.corr, L.corr_staging.data(), (size_t) t * g.n_embd * sizeof(float),
                                (size_t) g.n_embd * sizeof(float));
        ggml_free(c);
        const double g1 = now_us();
        g.stats.graph_us += g1 - g0;
        g.stats.miss_draws += (int64_t) misses.size();
        g.stats.bytes_read += (int64_t) misses.size() *
            (L.cpu[0]->nb[2] + L.cpu[1]->nb[2] + L.cpu[2]->nb[2]);
        L.corr_dirty = true;
    }
    const double t1 = now_us();
    g.stats.pass_us += t1 - t0;
}

// -------------------------------------------------------------- callback

bool llama_blobpager_cb_eval(struct ggml_tensor * t, bool ask, void * /*user_data*/) {
    typedef bool (*cb_t)(struct ggml_tensor *, bool, void *);
    const bool active = llama_blobpager_active();

    if (ask) {
        if (active && strncmp(t->name, kMoeIn, sizeof(kMoeIn) - 1) == 0) {
            const int il = atoi(t->name + sizeof(kMoeIn) - 1);
            if (il >= 0 && il < LLAMA_MAX_LAYERS) {
                g.moe_in_stash[il] = t;
            }
            return false;   // read at the topk stop point, no extra sync here
        }
        bool user_need = false;
        if (g.user_cb != nullptr) {
            user_need = ((cb_t) g.user_cb)(t, true, g.user_ud);
        }
        if (!user_need && active && strncmp(t->name, kTopk, sizeof(kTopk) - 1) == 0) {
            const int il = atoi(t->name + sizeof(kTopk) - 1);
            if (il >= 0 && il < LLAMA_MAX_LAYERS && g.layers[il].has_pins && t->ne[1] <= g.maxt) {
                return true;   // our stop point: sync + ask(false) with data ready
            }
        }
        return user_need;
    }

    // ask == false: run the miss pass, then chain to the user's callback
    if (active && strncmp(t->name, kTopk, sizeof(kTopk) - 1) == 0) {
        const int il = atoi(t->name + sizeof(kTopk) - 1);
        if (il >= 0 && il < LLAMA_MAX_LAYERS && g.layers[il].has_pins && t->ne[1] <= g.maxt) {
            do_miss_pass(il, t);
            g.stats.pass_invocations++;
        }
    }
    if (g.user_cb != nullptr) {
        return ((cb_t) g.user_cb)(t, false, g.user_ud);
    }
    return true;
}

void llama_blobpager_set_user_cb(ggml_backend_sched_eval_callback cb, void * user_data) {
    g.user_cb = (void *) cb;
    g.user_ud = user_data;
}