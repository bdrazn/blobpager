// blobpager-bench: E7a microbenchmark
// GPU pool-hit path + CPU miss path, K iterations each, JSON output.
// All tensors and the compute graph are built before a single alloc call.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static bool parse_manifest_gate(const std::string & path,
                                int64_t & n_embd, int64_t & n_ff, int64_t & n_expert,
                                std::string & type_str, int64_t & expert_bytes) {
    std::ifstream f(path); if (!f) return false;
    std::string c((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()); f.close();
    size_t pos = c.find("\"ffn_gate_exps\""); if (pos == std::string::npos) return false;
    size_t eb = c.find("\"expert_bytes\"", pos); if (eb == std::string::npos) return false;
    size_t v1 = c.find(':', eb) + 1; expert_bytes = std::stoll(c.substr(v1, c.find_first_of(",}", v1) - v1));
    size_t tp = c.find("\"type\"", pos); if (tp == std::string::npos) return false;
    size_t vs = c.find('"', c.find(':', tp) + 1) + 1; size_t ve = c.find('"', vs);
    type_str = c.substr(vs, ve - vs);
    size_t ne = c.find("\"ne\"", pos); if (ne == std::string::npos) return false;
    size_t as = c.find('[', ne) + 1, ae = c.find(']', as);
    std::string arr = c.substr(as, ae - as); int idx = 0; size_t p = 0;
    while (p < arr.size() && idx < 3) {
        while (p < arr.size() && (arr[p] == ' ' || arr[p] == ',')) p++;
        if (p >= arr.size()) break;
        size_t e = arr.find_first_of(" ,", p); if (e == std::string::npos) e = arr.size();
        long long val = std::stoll(arr.substr(p, e - p));
        if (idx == 0) n_embd = val; else if (idx == 1) n_ff = val; else n_expert = val;
        idx++; p = e + 1;
    }
    return (idx == 3 && n_embd > 0);
}

static ggml_type type_from_str(const std::string & s) {
    if (s == "Q4_K") return GGML_TYPE_Q4_K; if (s == "Q5_K") return GGML_TYPE_Q5_K;
    if (s == "Q6_K") return GGML_TYPE_Q6_K; return GGML_TYPE_Q4_K;
}

struct tensor_run { std::string name; uint64_t offset = 0; int64_t expert_bytes = 0; };
struct layer_pins { int layer = -1; int64_t pool_bytes = 0; std::vector<int32_t> pins; std::vector<tensor_run> tensors; };

static bool load_pins(const std::string & path, std::vector<layer_pins> & out) {
    std::ifstream f(path); if (!f) return false;
    std::string tag; int nl = 0;
    if (!(f >> tag >> nl) || tag != "LAYERS") return false;
    for (int i = 0; i < nl; ++i) {
        layer_pins lp; int np = 0;
        if (!(f >> tag >> lp.layer >> np >> lp.pool_bytes) || tag != "LAYER") return false;
        for (int t = 0; t < 3; ++t) { tensor_run r; if (!(f >> tag >> r.name >> r.offset >> r.expert_bytes) || tag != "TENSOR") return false; lp.tensors.push_back(r); }
        if (!(f >> tag) || tag != "PINS") return false;
        lp.pins.resize(np); for (int p = 0; p < np; ++p) if (!(f >> lp.pins[p])) return false;
        out.push_back(lp);
    }
    return !out.empty();
}

struct bench_result { std::string label; double us, gbs; int64_t bt; };

int main(int argc, char ** argv) {
    std::string pins_path, manifest_path, gguf_path;
    int K = 1000, n_pin = 96, layer_pick = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i]; const bool hv = (i+1 < argc);
        if      (a == "--pins"     && hv) pins_path     = argv[++i];
        else if (a == "--manifest" && hv) manifest_path = argv[++i];
        else if (a == "-m"         && hv) gguf_path     = argv[++i];
        else if (a == "-K"         && hv) K             = atoi(argv[++i]);
        else if (a == "--n-pin"    && hv) n_pin          = atoi(argv[++i]);
        else if (a == "--layer-pick" && hv) layer_pick  = atoi(argv[++i]);
    }
    if (gguf_path.empty() || pins_path.empty() || manifest_path.empty()) {
        fprintf(stderr, "Usage: blobpager-bench -m model.gguf --manifest manifest.json --pins pins.txt [-K N]\n"); return 1;
    }

    int64_t n_embd = 0, n_ff = 0, n_expert = 0, expert_bytes = 0;
    std::string type_str;
    if (!parse_manifest_gate(manifest_path, n_embd, n_ff, n_expert, type_str, expert_bytes)) { fprintf(stderr, "Bad manifest\n"); return 1; }
    const ggml_type gate_type = type_from_str(type_str);
    const int n_expert_used = 8;
    fprintf(stderr, "Manifest: n_embd=%lld n_ff=%lld n_expert=%lld type=%s ebytes=%lld\n",
            (long long)n_embd, (long long)n_ff, (long long)n_expert, type_str.c_str(), (long long)expert_bytes);

    std::vector<layer_pins> lps;
    if (!load_pins(pins_path, lps)) { fprintf(stderr, "Bad pins\n"); return 1; }
    if (layer_pick < 0 || layer_pick >= (int)lps.size()) { fprintf(stderr, "layer_pick OOR\n"); return 1; }
    layer_pins & lp = lps[layer_pick];
    std::vector<int32_t> pins = lp.pins;
    if ((int)pins.size() > n_pin) pins.resize(n_pin);

    const int N_SLOTS = n_pin + 1, DUMMY = n_pin;
    const int64_t pool_bytes = expert_bytes * N_SLOTS;
    fprintf(stderr, "Pool: %d slots x %lld bytes = %.2f MiB\n", N_SLOTS, (long long)expert_bytes, pool_bytes/(1024.0*1024.0));

    // Backends
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu_dev) { fprintf(stderr, "No GPU\n"); return 1; }
    ggml_backend_buffer_type_t gpu_buft = ggml_backend_dev_buffer_type(gpu_dev);
    ggml_backend_t gpu_be = ggml_backend_dev_init(gpu_dev, nullptr);
    ggml_backend_t cpu_be = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);

    // Pool on GPU
    ggml_context * gctx_p = ggml_init({ggml_tensor_overhead()*4, nullptr, true});
    ggml_tensor * pool97 = ggml_new_tensor_3d(gctx_p, gate_type, n_embd, n_ff, N_SLOTS);
    ggml_backend_buffer_t pool_buf = ggml_backend_alloc_ctx_tensors_from_buft(gctx_p, gpu_buft);

    int fd = open(gguf_path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "No GGUF\n"); return 1; }
    std::vector<uint8_t> staging((size_t)expert_bytes);
    const uint64_t gate_off = lp.tensors[0].offset;
    for (int s = 0; s < (int)pins.size(); ++s) {
        const uint64_t src = gate_off + (uint64_t)pins[s]*(uint64_t)expert_bytes;
        size_t got = 0;
        while (got < (size_t)expert_bytes) { ssize_t n = pread(fd, staging.data()+got, (size_t)expert_bytes-got, (off_t)(src+got)); if (n<=0) return 1; got+=(size_t)n; }
        ggml_backend_tensor_set(pool97, staging.data(), (size_t)(s*expert_bytes), (size_t)expert_bytes);
    }
    std::fill(staging.begin(), staging.end(), 0);
    ggml_backend_tensor_set(pool97, staging.data(), (size_t)(DUMMY*expert_bytes), (size_t)expert_bytes);
    close(fd);
    fprintf(stderr, "Pool filled (%d pins + zero dummy)\n", (int)pins.size());

    std::vector<int32_t> SLOTS(n_expert, DUMMY);
    for (int s = 0; s < (int)pins.size(); ++s) SLOTS[pins[s]] = s;
    std::vector<int32_t> miss_ids;
    for (int e = 0; e < (int)n_expert && (int)miss_ids.size() < 4; ++e) if (SLOTS[e]==DUMMY) miss_ids.push_back(e);
    fprintf(stderr, "Miss ids:"); for (auto m : miss_ids) fprintf(stderr, " %d", m); fprintf(stderr, "\n");

    std::vector<float> x_data(n_embd);
    for (int64_t i = 0; i < n_embd; ++i) x_data[i] = (float)(i%1000)/1000.0f;

    const int N_IDs = n_expert_used;
    struct bench_pattern { int n_miss; std::vector<int32_t> rids; std::string label; };
    std::vector<bench_pattern> patterns;
    { bench_pattern p; p.n_miss=0; for(int i=0;i<8&&i<(int)pins.size();++i) p.rids.push_back(SLOTS[pins[i]]); p.label="0miss_8hit"; patterns.push_back(p); }
    { bench_pattern p; p.n_miss=1; for(int i=0;i<7&&i<(int)pins.size();++i) p.rids.push_back(SLOTS[pins[i]]); p.rids.push_back(DUMMY); p.label="1miss_7hit"; patterns.push_back(p); }
    if (miss_ids.size()>=2) { bench_pattern p; p.n_miss=2; for(int i=0;i<6&&i<(int)pins.size();++i) p.rids.push_back(SLOTS[pins[i]]); p.rids.push_back(DUMMY); p.rids.push_back(DUMMY); p.label="2miss_6hit"; patterns.push_back(p); }

    // ===== GPU BENCH =====
    // x_gpu on GPU (separate context, persists across patterns)
    ggml_context * gctx_x = ggml_init({ggml_tensor_overhead()*4, nullptr, true});
    ggml_tensor * x_gpu = ggml_new_tensor_3d(gctx_x, GGML_TYPE_F32, n_embd, 1, 1);
    ggml_backend_buffer_t x_buf = ggml_backend_alloc_ctx_tensors_from_buft(gctx_x, gpu_buft);
    ggml_backend_tensor_set(x_gpu, x_data.data(), 0, n_embd*sizeof(float));

    std::vector<bench_result> gpu_results;
    for (auto & pat : patterns) {
        // Per-pattern: ids tensor + compute graph, all in one context, single alloc
        ggml_context * gctx = ggml_init({ggml_tensor_overhead()*64 + 262144, nullptr, true});
        ggml_tensor * ids = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, N_IDs, 1);
        ggml_tensor * result = ggml_mul_mat_id(gctx, pool97, x_gpu, ids);
        // Set ids data before alloc
        // Can't set data before alloc (tensor has no buffer). Set ids values after alloc.
        // But we need to set ids before building the graph... Actually, we can set
        // data after alloc. The graph is already built; we just need the ids values.
        // Build the graph
        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, result);
        // Now alloc (this allocates ids and result, pool97 and x_gpu already have buffers)
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, gpu_be);
        // Set ids data
        ggml_backend_tensor_set(ids, pat.rids.data(), 0, N_IDs*sizeof(int32_t));

        // Warmup + timed
        for (int w = 0; w < 10; ++w) ggml_backend_graph_compute(gpu_be, gf);
        ggml_backend_synchronize(gpu_be);
        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < K; ++k) { ggml_backend_graph_compute(gpu_be, gf); ggml_backend_synchronize(gpu_be); }
        auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double,std::micro>(t1-t0).count() / K;
        int64_t bt = expert_bytes * N_IDs;
        gpu_results.push_back({pat.label, us, (double)bt/(us*1e-6)/(1024.0*1024.0*1024.0), bt});
        fprintf(stderr, "GPU [%s]: %.2f us/call, %.2f GB/s\n", pat.label.c_str(), us, gpu_results.back().gbs);

        ggml_backend_buffer_free(buf); ggml_free(gctx);
    }

    // ===== CPU BENCH =====
    // F32 proxy: same dimensions, ggml-managed allocation (avoids buffer lifetime bugs)
    // The ratio per-expert transfers to Q4_K; absolute timing will differ.
    std::vector<bench_result> cpu_results;
    const int64_t f32_ebytes = n_embd * n_ff * sizeof(float);
    fprintf(stderr, "CPU bench: F32 proxy, expert_bytes_F32=%lld\n", (long long)f32_ebytes);
    for (int nm = 1; nm <= 2; ++nm) {
        size_t mem_sz = ggml_tensor_overhead()*16 + f32_ebytes*nm*2 + n_embd*sizeof(float)*4 + nm*sizeof(int32_t)*4 + 16*1024*1024;
        ggml_context * gctx = ggml_init({mem_sz, nullptr, /*no_alloc=*/false});
        if (!gctx) { fprintf(stderr, "CPU ctx alloc failed\n"); return 1; }
        ggml_tensor * x_cpu  = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, n_embd, 1, 1);
        ggml_tensor * mid_t  = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, nm, 1);
        ggml_tensor * mexps  = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, n_embd, n_ff, nm);
        memcpy(x_cpu->data, x_data.data(), n_embd*sizeof(float));
        std::vector<int32_t> ids_cpu(nm);
        for (int m = 0; m < nm; ++m) ids_cpu[m] = m;
        memcpy(mid_t->data, ids_cpu.data(), nm*sizeof(int32_t));
        std::vector<float> ones(n_embd * n_ff, 1.0f);
        for (int m = 0; m < nm; ++m)
            memcpy((char*)mexps->data + m*f32_ebytes, ones.data(), f32_ebytes);

        ggml_tensor * result = ggml_mul_mat_id(gctx, mexps, x_cpu, mid_t);
        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, result);
        // Warmup
        for (int w = 0; w < 20; ++w) ggml_graph_compute_with_ctx(gctx, gf, 20);
        // Timed
        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < K; ++k) ggml_graph_compute_with_ctx(gctx, gf, 20);
        auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double,std::micro>(t1-t0).count() / K;
        int64_t bt = f32_ebytes * nm;
        std::string label = std::to_string(nm) + "miss_CPU_F32";
        cpu_results.push_back({label, us, (double)bt/(us*1e-6)/(1024.0*1024.0*1024.0), bt});
        fprintf(stderr, "CPU [%s]: %.2f us/call, %.2f GB/s\n", label.c_str(), us, cpu_results.back().gbs);
        ggml_free(gctx);
    }

    // JSON
    printf("{\n");
    printf("  \"experiment\": \"E7a_pool_microbench\",\n");
    printf("  \"layer\": %d, \"n_embd\": %lld, \"n_ff\": %lld, \"n_expert\": %lld, \"n_expert_used\": %d,\n",
           lp.layer, (long long)n_embd, (long long)n_ff, (long long)n_expert, n_expert_used);
    printf("  \"n_pins\": %d, \"n_slots\": %d, \"expert_bytes\": %lld, \"type\": \"%s\", \"pool_bytes\": %lld, \"K\": %d,\n",
           n_pin, N_SLOTS, (long long)expert_bytes, type_str.c_str(), (long long)pool_bytes, K);
    printf("  \"gpu_arm\": [\n");
    for (size_t i = 0; i < gpu_results.size(); ++i) {
        auto &r = gpu_results[i];
        printf("    {\"label\":\"%s\",\"us_per_call\":%.2f,\"gb_per_s\":%.2f,\"bytes\":%lld}%s\n",
               r.label.c_str(), r.us, r.gbs, (long long)r.bt, i+1<gpu_results.size()?",":"");
    }
    printf("  ],\n");
    printf("  \"cpu_arm\": [\n");
    for (size_t i = 0; i < cpu_results.size(); ++i) {
        auto &r = cpu_results[i];
        printf("    {\"label\":\"%s\",\"us_per_call\":%.2f,\"gb_per_s\":%.2f,\"bytes\":%lld}%s\n",
               r.label.c_str(), r.us, r.gbs, (long long)r.bt, i+1<cpu_results.size()?",":"");
    }
    printf("  ]\n}\n");

    ggml_backend_buffer_free(x_buf); ggml_free(gctx_x);
    ggml_backend_buffer_free(pool_buf); ggml_free(gctx_p);
    ggml_backend_free(cpu_be); ggml_backend_free(gpu_be);
    return 0;
}
