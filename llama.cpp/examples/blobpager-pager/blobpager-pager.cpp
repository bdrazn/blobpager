// blobpager-pager: Phase-1 PoC harness for the blobpager slot-pool pager.
//
// Three things, in one example, with zero llama.cpp source changes:
//   1. PIN TABLE  — loads blobpager/data/pins-qwen.txt (emitted from the
//      blobpack manifest): per-layer top-96 expert ids + the byte extents
//      (base offset + expert_bytes) of the 3 routed-expert tensors.
//   2. DEVICE POOL — allocates one GPU buffer sized to hold every pinned
//      expert blob (pin-major layout: slot s = pins[s]'s gate|up|down runs),
//      fills it with pread() from the GGUF + ggml_backend_tensor_set(), and
//      times the two phases separately (disk read vs PCIe upload). The pool
//      is freed after timing: this increment proves the fill path and its
//      cost, it does not yet execute against the pool.
//   3. LIVE HIT/MISS ACCOUNTING — runs the model exactly like the --cpu-moe
//      baseline (-ngl 99 -ncmoe 48) and observes every ffn_moe_topk-<layer>
//      via params.cb_eval, partitioning each top-8 draw into hits (expert is
//      pinned) vs misses. Runs prefill over a corpus slice, then greedy
//      decode (128 tokens) so DECODE-phase routing — never traced before —
//      gets measured. Also tracks, per ubatch, the union of miss experts
//      (the expert set the CPU side would have to execute) and the union of
//      hit experts (what the GPU pool would serve).
//
// Output: human-readable summary + JSON summary on stdout; per-(req,pos,
// layer) JSONL detail (with a "ph" phase tag) if --out is given.
//
// Usage:
//   llama-blobpager-pager -m model.gguf -ngl 99 -ncmoe 48 -t 20 -c 1024 \
//     -b 512 -ub 512 --corpus corpus.txt --pins pins-qwen.txt \
//     --out pager-run.jsonl --n-prefill 16 --n-gen-lines 4 --n-gen 128

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// ---------------------------------------------------------------- pins file

struct tensor_run {
    std::string name;
    uint64_t    offset = 0;
    int64_t     expert_bytes = 0;
};

struct layer_pins {
    int     layer = -1;
    int64_t pool_bytes = 0;                 // recomputed: sum(ebytes) * npins
    std::vector<int32_t> pins;              // pool slot order
    std::vector<tensor_run> tensors;        // gate/up/down runs (expert-major)
};

static bool load_pins(const std::string & path, std::vector<layer_pins> & out) {
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
        layer_pins lp;
        int npins = 0;
        if (!(f >> tag >> lp.layer >> npins >> lp.pool_bytes) || tag != "LAYER") {
            return false;
        }
        for (int t = 0; t < 3; ++t) {
            tensor_run r;
            if (!(f >> tag >> r.name >> r.offset >> r.expert_bytes) || tag != "TENSOR") {
                return false;
            }
            lp.tensors.push_back(r);
        }
        if (!(f >> tag) || tag != "PINS") {
            return false;
        }
        lp.pins.resize(npins);
        for (int p = 0; p < npins; ++p) {
            if (!(f >> lp.pins[p])) {
                return false;
            }
        }
        out.push_back(lp);
    }
    return !out.empty();
}

// ------------------------------------------------------------- pager state

struct phase_stats {
    int64_t draws = 0;
    int64_t hits  = 0;
    int64_t obs   = 0;
    int64_t aux_skipped = 0;
    int64_t mismatched  = 0;
    std::map<int, std::pair<int64_t, int64_t>> layers;   // layer -> (draws, hits)
    int64_t miss_union_sum = 0, miss_union_max = 0;      // per-observation
    int64_t hit_union_sum  = 0, hit_union_max  = 0;
    int64_t ub_miss = 0, ub_hit = 0;                     // current ubatch
    std::vector<int64_t> ub_miss_hist;                   // per-ubatch totals
    std::vector<int64_t> ub_hit_hist;
};

struct pager_state {
    std::ofstream out;
    // pin lookups, per layer: expert id -> slot (or -1), and pinned flag
    std::map<int, std::vector<int32_t>> slot_of;
    std::map<int, std::vector<uint8_t>> ispin;
    int64_t n_pin = 0;

    int64_t req_idx     = -1;
    int64_t req_len     = 0;   // prompt token count of the in-flight request
    int64_t chunk_pos   = 0;   // prefill chunk start position
    int64_t expected_tok = 0;  // expected ubatch n_tok during prefill
    int64_t decode_idx  = 0;
    bool    in_decode   = false;

    phase_stats prefill, decode;

    phase_stats & cur() {
        return in_decode ? decode : prefill;
    }
};

static const char kPrefix[] = "ffn_moe_topk-";

static bool pager_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (pager_state *) user_data;

    if (ask) {
        return strncmp(t->name, kPrefix, sizeof(kPrefix) - 1) == 0;
    }
    if (strncmp(t->name, kPrefix, sizeof(kPrefix) - 1) != 0) {
        return true;
    }

    const int layer = atoi(t->name + sizeof(kPrefix) - 1);

    if (t->type != GGML_TYPE_I32) {
        LOG_ERR("%s: expected I32 ids tensor, got %s\n", __func__, ggml_type_name(t->type));
        exit(1);
    }

    const int64_t n_used = t->ne[0];
    const int64_t n_tok  = t->ne[1];

    // layers outside the pin table (MTP aux block) are not serving layers
    auto so_it = st->slot_of.find(layer);
    if (so_it == st->slot_of.end()) {
        st->cur().aux_skipped++;
        return true;
    }

    if (!st->in_decode && n_tok != st->expected_tok) {
        if (st->prefill.mismatched < 40) {
            LOG_ERR("%s: MISMATCH req %lld layer %d n_tok %lld != expected %lld — skipped\n",
                    __func__, (long long) st->req_idx, layer, (long long) n_tok, (long long) st->expected_tok);
        }
        st->prefill.mismatched++;
        return true;
    }

    // the topk tensor may be host- (CPU experts) or device-resident (router on GPU)
    std::vector<int32_t> ids((size_t)(n_used * n_tok));
    if (ggml_backend_buffer_is_host(t->buffer)) {
        memcpy(ids.data(), t->data, ids.size() * sizeof(int32_t));
    } else {
        ggml_backend_tensor_get(t, ids.data(), 0, ids.size() * sizeof(int32_t));
    }

    const std::vector<int32_t> & so = so_it->second;
    const std::vector<uint8_t> & ip = st->ispin[layer];

    int64_t hits = 0, misses = 0, hit_union = 0, miss_union = 0;
    uint64_t seenH[2] = {0, 0}, seenM[2] = {0, 0};
    for (int64_t j = 0; j < n_tok; ++j) {
        for (int64_t k = 0; k < n_used; ++k) {
            const int32_t id = ids[j * n_used + k];
            if (ip[id]) {
                hits++;
                if (!(seenH[id >> 6] & (1ull << (id & 63)))) {
                    seenH[id >> 6] |= 1ull << (id & 63);
                    hit_union++;
                }
            } else {
                misses++;
                if (!(seenM[id >> 6] & (1ull << (id & 63)))) {
                    seenM[id >> 6] |= 1ull << (id & 63);
                    miss_union++;
                }
            }
        }
    }

    phase_stats & ps = st->cur();
    ps.draws += n_used * n_tok;
    ps.hits  += hits;
    ps.obs++;
    auto & L = ps.layers[layer];
    L.first  += n_used * n_tok;
    L.second += hits;
    ps.miss_union_sum += miss_union;
    ps.miss_union_max  = std::max(ps.miss_union_max, miss_union);
    ps.hit_union_sum  += hit_union;
    ps.hit_union_max   = std::max(ps.hit_union_max, hit_union);
    ps.ub_miss += miss_union;
    ps.ub_hit  += hit_union;

    if (st->out.is_open()) {
        for (int64_t j = 0; j < n_tok; ++j) {
            const int64_t pos = st->in_decode ? (st->req_len + st->decode_idx) : (st->chunk_pos + j);
            st->out << "{\"ph\":\"" << (st->in_decode ? "d" : "p") << "\",\"req\":" << st->req_idx
                    << ",\"pos\":" << pos << ",\"layer\":" << layer << ",\"exp\":[";
            for (int64_t k = 0; k < n_used; ++k) {
                if (k) {
                    st->out << ",";
                }
                st->out << ids[j * n_used + k];
            }
            st->out << "]}\n";
        }
    }
    return true;
}

// --------------------------------------------------------------- pool fill

struct pool_fill_info {
    bool    ok = false;
    int     layers = 0;
    int     pins_per_layer = 0;
    int64_t bytes = 0;
    double  read_s = 0;
    double  upload_s = 0;
    std::string device;
    std::string note;
};

static pool_fill_info fill_pool(const std::vector<layer_pins> & lps,
                                const std::string & gguf_path,
                                int n_pin_max) {
    pool_fill_info info;

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == nullptr) {
        info.note = "no GPU device available";
        return info;
    }
    info.device = ggml_backend_dev_name(dev);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    std::vector<layer_pins> use = lps;
    int64_t total = 0;
    for (auto & l : use) {
        if (n_pin_max > 0 && (int) l.pins.size() > n_pin_max) {
            l.pins.resize(n_pin_max);
        }
        int64_t per = 0;
        for (const auto & t : l.tensors) {
            per += t.expert_bytes;
        }
        l.pool_bytes = per * (int64_t) l.pins.size();
        total += l.pool_bytes;
    }

    info.layers        = (int) use.size();
    info.pins_per_layer = (int) use[0].pins.size();
    info.bytes = total;

    ggml_init_params ip = { ggml_tensor_overhead() * 8, nullptr, true /* no_alloc */ };
    ggml_context * gctx = ggml_init(ip);
    if (gctx == nullptr) {
        info.note = "ggml_init failed";
        return info;
    }
    ggml_tensor * pool = ggml_new_tensor_1d(gctx, GGML_TYPE_I8, total);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(gctx, buft);
    if (buf == nullptr) {
        info.note = "device buffer alloc failed for " + std::to_string(total) + " B";
        ggml_free(gctx);
        return info;
    }

    const int fd = open(gguf_path.c_str(), O_RDONLY);
    if (fd < 0) {
        info.note = "cannot open gguf for pread";
        ggml_backend_buffer_free(buf);
        ggml_free(gctx);
        return info;
    }

    std::vector<uint8_t> staging;
    size_t off = 0;
    double read_acc = 0, up_acc = 0;
    for (const auto & l : use) {
        staging.resize((size_t) l.pool_bytes);
        auto r0 = std::chrono::steady_clock::now();
        size_t pos = 0;
        for (const auto & pin : l.pins) {
            for (const auto & t : l.tensors) {
                const uint64_t src = t.offset + (uint64_t) pin * (uint64_t) t.expert_bytes;
                size_t got = 0;
                while (got < (size_t) t.expert_bytes) {
                    const ssize_t n = pread(fd, staging.data() + pos + got,
                                            (size_t) t.expert_bytes - got, (off_t) (src + got));
                    if (n <= 0) {
                        LOG_ERR("%s: pread failed at offset %llu\n", __func__, (unsigned long long) (src + got));
                        close(fd);
                        ggml_backend_buffer_free(buf);
                        ggml_free(gctx);
                        info.note = "pread failed";
                        return info;
                    }
                    got += (size_t) n;
                }
                pos += (size_t) t.expert_bytes;
            }
        }
        auto r1 = std::chrono::steady_clock::now();
        ggml_backend_tensor_set(pool, staging.data(), off, (size_t) l.pool_bytes);
        auto r2 = std::chrono::steady_clock::now();
        read_acc   += std::chrono::duration<double>(r1 - r0).count();
        up_acc     += std::chrono::duration<double>(r2 - r1).count();
        off        += (size_t) l.pool_bytes;
    }

    info.read_s   = read_acc;
    info.upload_s = up_acc;
    info.ok = true;
    info.note = std::to_string(use.size()) + " layers x " + std::to_string(info.pins_per_layer) + " pins";

    close(fd);
    ggml_backend_buffer_free(buf);
    ggml_free(gctx);
    return info;
}

// -------------------------------------------------------------------- CLI

struct cli_opts {
    std::string corpus;
    std::string pins;
    std::string out;
    std::string dump;   // --dump-logits: binary dump of sampled token + logits per step
    int n_prefill   = 16;
    int n_gen_lines = 4;
    int n_gen       = 128;
    int n_pin       = 96;
    bool pool_fill  = true;
};

static std::vector<char *> strip_custom_args(int argc, char ** argv, cli_opts & o) {
    std::vector<char *> rest;
    rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has_val = (i + 1 < argc);
        if      (a == "--corpus"      && has_val) o.corpus      = argv[++i];
        else if (a == "--pins"        && has_val) o.pins        = argv[++i];
        else if (a == "--out"         && has_val) o.out         = argv[++i];
        else if (a == "--dump-logits" && has_val) o.dump        = argv[++i];
        else if (a == "--n-prefill"   && has_val) o.n_prefill   = atoi(argv[++i]);
        else if (a == "--n-gen-lines" && has_val) o.n_gen_lines = atoi(argv[++i]);
        else if (a == "--n-gen"       && has_val) o.n_gen       = atoi(argv[++i]);
        else if (a == "--n-pin"       && has_val) o.n_pin       = atoi(argv[++i]);
        else if (a == "--pool-fill"   && has_val) o.pool_fill   = atoi(argv[++i]) != 0;
        else rest.push_back(argv[i]);
    }
    return rest;
}

// ---------------------------------------------------------------- summary

static std::string fmt_stats(const char * name, const phase_stats & ps, int64_t tokens) {
    char buf[512];
    std::string s = std::string("\"") + name + "\": {";
    snprintf(buf, sizeof(buf),
             "\"requests_seen_tokens\": %lld, \"draws\": %lld, \"hits\": %lld, \"hit_rate\": %.6f, "
             "\"obs\": %lld, \"aux_skipped\": %lld, \"mismatched\": %lld, "
             "\"miss_union_mean\": %.3f, \"miss_union_max\": %lld, "
             "\"hit_union_mean\": %.3f, \"hit_union_max\": %lld, "
             "\"ubatch_miss_mean\": %.3f, \"ubatch_miss_max\": %lld, \"ubatches\": %zu",
             (long long) tokens, (long long) ps.draws, (long long) ps.hits,
             ps.draws ? (double) ps.hits / (double) ps.draws : 0.0,
             (long long) ps.obs, (long long) ps.aux_skipped, (long long) ps.mismatched,
             ps.obs ? (double) ps.miss_union_sum / (double) ps.obs : 0.0,
             (long long) ps.miss_union_max,
             ps.obs ? (double) ps.hit_union_sum / (double) ps.obs : 0.0,
             (long long) ps.hit_union_max,
             ps.ub_miss_hist.empty() ? 0.0 : [&]{ int64_t t = 0; for (auto v : ps.ub_miss_hist) t += v; return (double) t / (double) ps.ub_miss_hist.size(); }(),
             [&]{ int64_t m = 0; for (auto v : ps.ub_miss_hist) m = std::max(m, v); return (long long) m; }(),
             ps.ub_miss_hist.size());
    s += buf;
    s += ", \"per_layer\": [";
    bool first = true;
    for (const auto & kv : ps.layers) {
        char lb[128];
        snprintf(lb, sizeof(lb), "%s[%d,%lld,%lld,%.4f]", first ? "" : ",",
                 kv.first, (long long) kv.second.first, (long long) kv.second.second,
                 kv.second.first ? (double) kv.second.second / (double) kv.second.first : 0.0);
        s += lb;
        first = false;
    }
    s += "]}";
    return s;
}

// ------------------------------------------------------------------- main

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    cli_opts o;
    std::vector<char *> rest = strip_custom_args(argc, argv, o);

    if (o.corpus.empty() || o.pins.empty()) {
        LOG_ERR("usage: %s -m model.gguf -ngl 99 -ncmoe 48 --corpus corpus.txt --pins pins.txt "
                "[--out run.jsonl] [--n-prefill N] [--n-gen-lines N] [--n-gen N] [--n-pin N] [--pool-fill 0|1]\n", argv[0]);
        return 1;
    }

    std::vector<layer_pins> lps;
    if (!load_pins(o.pins, lps)) {
        LOG_ERR("%s: cannot parse pins file '%s'\n", __func__, o.pins.c_str());
        return 1;
    }
    LOG_INF("%s: pins loaded: %zu layers, %d pins/layer (limit %d)\n", __func__,
            lps.size(), (int) lps[0].pins.size(), o.n_pin);

    pager_state st;
    if (!o.out.empty()) {
        st.out.open(o.out, std::ios::binary);
        if (!st.out) {
            LOG_ERR("%s: cannot open detail output '%s'\n", __func__, o.out.c_str());
            return 1;
        }
    }

    common_params params;
    common_init();
    if (!common_params_parse(rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    params.warmup            = false;   // warmup forces all experts — pollutes accounting
    params.cb_eval           = pager_cb;
    params.cb_eval_user_data = &st;
    params.embedding         = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const uint32_t n_ctx = llama_n_ctx(ctx);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // logits dump for the E7b validation: per step, {int32 sampled token} + {n_vocab f32 logits}
    std::ofstream dumpf;
    if (!o.dump.empty()) {
        dumpf.open(o.dump, std::ios::binary);
        if (!dumpf) {
            LOG_ERR("%s: cannot open logits dump '%s'\n", __func__, o.dump.c_str());
            return 1;
        }
        const uint32_t magic = 0x424C4731;  // "BLG1"
        dumpf.write((const char *) &magic, 4);
        dumpf.write((const char *) &n_vocab, 4);
    }
    auto dump_step = [&](int32_t sampled_tok, const float * lg) {
        if (!dumpf.is_open()) {
            return;
        }
        dumpf.write((const char *) &sampled_tok, 4);
        dumpf.write((const char *) lg, (size_t) n_vocab * sizeof(float));
    };

    // expert count from model metadata, then build pin lookups
    int32_t expert_count = 128;
    {
        char key[256], val[256];
        const int32_t n_meta = llama_model_meta_count(model);
        for (int32_t i = 0; i < n_meta; ++i) {
            if (llama_model_meta_key_by_index(model, i, key, sizeof(key)) != 0) {
                break;
            }
            if (std::string(key).find(".expert_count") != std::string::npos &&
                llama_model_meta_val_str(model, key, val, sizeof(val)) == 0) {
                expert_count = atoi(val);
                break;
            }
        }
    }
    for (const auto & lp : lps) {
        auto & so = st.slot_of[lp.layer];
        auto & ipv = st.ispin[lp.layer];
        so.assign(expert_count, -1);
        ipv.assign(expert_count, 0);
        for (size_t s = 0; s < lp.pins.size(); ++s) {
            const int32_t id = lp.pins[s];
            if (id < 0 || id >= expert_count) {
                LOG_ERR("%s: pin id %d out of range (expert_count %d)\n", __func__, id, expert_count);
                return 1;
            }
            so[id] = (int32_t) s;
            ipv[id] = 1;
        }
        st.n_pin = (int64_t) lp.pins.size();
    }
    LOG_INF("%s: pin lookups ready: %zu layers, expert_count %d\n", __func__, st.slot_of.size(), expert_count);

    // pool fill (timed, then freed) — degrading pin count if VRAM refuses
    pool_fill_info pf;
    if (o.pool_fill) {
        const int candidates[] = { o.n_pin, 88, 80, 72, 64, 48 };
        for (int ci = 0; ci < 6; ++ci) {
            const int np = candidates[ci];
            if (ci > 0 && np >= o.n_pin) {
                continue;
            }
            if (ci > 0) {
                LOG_WRN("%s: retrying pool fill with %d pins/layer\n", __func__, np);
            }
            pf = fill_pool(lps, params.model.path, np);
            if (pf.ok) {
                break;
            }
        }
        LOG_INF("%s: pool fill %s: %s, %lld B, read %.3f s (%.2f GB/s), upload %.3f s (%.2f GB/s)\n",
                __func__, pf.ok ? "OK" : "FAILED", pf.note.c_str(), (long long) pf.bytes,
                pf.read_s, pf.read_s > 0 ? pf.bytes / pf.read_s / 1e9 : 0.0,
                pf.upload_s, pf.upload_s > 0 ? pf.bytes / pf.upload_s / 1e9 : 0.0);
    }

    // corpus
    std::ifstream corpus(o.corpus);
    if (!corpus) {
        LOG_ERR("%s: cannot open corpus '%s'\n", __func__, o.corpus.c_str());
        return 1;
    }
    std::vector<std::string> lines;
    for (std::string line; std::getline(corpus, line); ) {
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    const size_t n_prefill = std::min((size_t) o.n_prefill, lines.size());
    const size_t n_genl = std::min((size_t) o.n_gen_lines, lines.size() - n_prefill);
    LOG_INF("%s: corpus %zu lines; plan: %zu prefill-only + %zu prefill+generate(%d)\n",
            __func__, lines.size(), n_prefill, n_genl, o.n_gen);

    llama_memory_t mem = llama_get_memory(ctx);

    // explicit batch: batch_get_one() leaves logits unset and this master computes
    // no outputs for it — we need logits on the final prefill token to start generation
    llama_batch batch = llama_batch_init(params.n_batch, 0, 1);
    if (batch.token == nullptr) {
        LOG_ERR("%s: llama_batch_init failed\n", __func__);
        return 1;
    }

    auto flush_ub = [&]() {
        phase_stats & ps = st.in_decode ? st.decode : st.prefill;
        ps.ub_miss_hist.push_back(ps.ub_miss);
        ps.ub_hit_hist.push_back(ps.ub_hit);
        ps.ub_miss = 0;
        ps.ub_hit  = 0;
    };

    auto run_prefill = [&](int64_t req_idx, std::vector<llama_token> & tokens) {
        st.req_idx = req_idx;
        st.req_len = (int64_t) tokens.size();
        st.in_decode = false;
        st.decode_idx = 0;
        st.chunk_pos = 0;
        llama_memory_seq_rm(mem, 0, 0, -1);
        for (size_t off = 0; off < tokens.size(); off += params.n_ubatch) {
            const size_t n = std::min((size_t) params.n_ubatch, tokens.size() - off);
            st.expected_tok = (int64_t) n;
            st.chunk_pos = (int64_t) off;
            batch.n_tokens = (int32_t) n;
            for (size_t i = 0; i < n; ++i) {
                batch.token[i]  = tokens[off + i];
                batch.pos[i]    = (llama_pos)(off + i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = false;
            }
            batch.logits[n - 1] = true;   // last token always gets logits (cheap; needed after the final chunk)
            if (llama_decode(ctx, batch)) {
                LOG_ERR("%s: decode failed on request %lld\n", __func__, (long long) req_idx);
                return false;
            }
            flush_ub();
        }
        return true;
    };

    // pass 1: prefill-only accounting
    for (size_t i = 0; i < n_prefill; ++i) {
        std::vector<llama_token> tokens = common_tokenize(vocab, lines[i], add_bos, true);
        if (tokens.empty()) {
            continue;
        }
        if (tokens.size() > n_ctx || tokens.size() > params.n_batch) {
            tokens.resize(std::min((size_t) n_ctx, (size_t) params.n_batch));
        }
        if (!run_prefill((int64_t) i, tokens)) {
            return 1;
        }
    }
    LOG_INF("%s: prefill-only accounting done over %zu requests\n", __func__, n_prefill);

    // pass 2: prefill + greedy generation with decode-phase accounting
    int64_t req_idx = (int64_t) n_prefill;
    int64_t gen_tokens_total = 0;
    for (size_t g = 0; g < n_genl; ++g) {
        std::vector<llama_token> tokens = common_tokenize(vocab, lines[n_prefill + g], add_bos, true);
        if (tokens.empty()) {
            continue;
        }
        if (tokens.size() > n_ctx || tokens.size() > params.n_batch) {
            tokens.resize(std::min((size_t) n_ctx, (size_t) params.n_batch));
        }
        if (!run_prefill(req_idx, tokens)) {
            return 1;
        }
        float * lg = llama_get_logits_ith(ctx, (int32_t) (st.expected_tok - 1));
        if (lg == nullptr) { LOG_ERR("%s: no logits after prefill of req %lld\n", __func__, (long long) req_idx); return 1; }
        int best = 0;
        for (int32_t k = 1; k < n_vocab; ++k) {
            if (lg[k] > lg[best]) {
                best = k;
            }
        }
        llama_token tok = (llama_token) best;
        dump_step((int32_t) tok, lg);

        st.in_decode = true;
        int produced = 0;
        for (int gen = 0; gen < o.n_gen; ++gen) {
            if (llama_vocab_is_eog(vocab, tok)) {
                break;
            }
            st.decode_idx = gen;
            batch.n_tokens = 1;
            batch.token[0] = tok;
            batch.pos[0] = (llama_pos)(st.req_len + gen);
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;
            if (llama_decode(ctx, batch)) {
                LOG_ERR("%s: decode failed on req %lld gen %d\n", __func__, (long long) req_idx, gen);
                return 1;
            }
            flush_ub();
            produced++;
            lg = llama_get_logits_ith(ctx, 0);
            if (lg == nullptr) { LOG_ERR("%s: no logits after gen %d\n", __func__, gen); return 1; }
            best = 0;
            for (int32_t k = 1; k < n_vocab; ++k) {
                if (lg[k] > lg[best]) {
                    best = k;
                }
            }
            tok = (llama_token) best;
            dump_step((int32_t) tok, lg);
        }
        st.in_decode = false;
        gen_tokens_total += produced;
        LOG_INF("%s: req %lld generated %d tokens\n", __func__, (long long) req_idx, produced);
        ++req_idx;
    }

    // summary
    printf("{\n");
    printf("  \"meta\": {\"model\": \"%s\", \"expert_count\": %d, \"pins_layers\": %zu, \"pins_per_layer\": %lld, \"n_gen\": %d},\n",
           params.model.path.c_str(), expert_count, st.slot_of.size(), (long long) st.n_pin, o.n_gen);
    printf("  \"pool_fill\": {\"ok\": %s, \"device\": \"%s\", \"note\": \"%s\", \"layers\": %d, \"pins_per_layer\": %d, "
           "\"bytes\": %lld, \"read_s\": %.4f, \"upload_s\": %.4f, \"read_gbs\": %.3f, \"upload_gbs\": %.3f},\n",
           pf.ok ? "true" : "false", pf.device.c_str(), pf.note.c_str(), pf.layers, pf.pins_per_layer,
           (long long) pf.bytes, pf.read_s, pf.upload_s,
           pf.read_s   > 0 ? pf.bytes / pf.read_s   / 1e9 : 0.0,
           pf.upload_s > 0 ? pf.bytes / pf.upload_s / 1e9 : 0.0);
    printf("  %s,\n", fmt_stats("prefill", st.prefill, 0).c_str());
    printf("  %s,\n", fmt_stats("decode", st.decode, gen_tokens_total).c_str());
    printf("  \"decode_tokens_generated\": %lld\n", (long long) gen_tokens_total);
    printf("}\n");

    st.out.flush();
    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}