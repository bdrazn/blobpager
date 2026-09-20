// CPU miss-path microbench: time mul_mat_id with 1-2 experts on CPU
// Standalone: no llama model, just ggml ops with F32 tensors
// Build: g++ -O2 -std=c++17 -I llama.cpp/ggml/include -I llama.cpp/ggml/src -I llama.cpp/include cpu-miss-bench.cpp -L llama.cpp/build-cuda/bin -lggml -lggml-base -lpthread -o cpu-miss-bench

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>

int main(int argc, char ** argv) {
    // Qwen3-30B gate_exps dimensions: [2048, 768, 128] Q4_K
    // For F32 bench: [2048, 768, 128]
    const int64_t n_embd = 2048;
    const int64_t n_ff = 768;
    const int64_t n_expert = 128;
    const int K = argc > 1 ? atoi(argv[1]) : 10000;

    printf("CPU miss-path bench: n_embd=%lld n_ff=%lld n_expert=%lld F32 K=%d\n",
           (long long)n_embd, (long long)n_ff, (long long)n_expert, K);

    // CPU backend
    ggml_backend_dev_t dev_cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t cpu_be = ggml_backend_dev_init(dev_cpu, nullptr);

    for (int nm : {1, 2, 8}) {
        // Context: weights [n_embd, n_ff, nm], activation [n_embd, 1, 1], ids [nm, 1]
        size_t ctx_size = (size_t)n_embd * n_ff * nm * sizeof(float)  // weights
                        + (size_t)n_embd * sizeof(float)               // activation
                        + (size_t)nm * sizeof(int32_t)                 // ids
                        + (size_t)n_ff * nm * sizeof(float)            // result
                        + 4 * 1024 * 1024;                             // overhead
        struct ggml_init_params params = {
            .mem_size = ctx_size,
            .mem_buffer = nullptr,
            .no_alloc = true,
        };
        ggml_context * ctx = ggml_init(params);

        // Full weight set: [n_embd, n_ff, n_expert] but we only need nm experts
        // For miss-path timing, we use nm experts (1, 2, or 8)
        ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, nm);
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, 1);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, nm, 1);
        ggml_tensor * result = ggml_mul_mat_id(ctx, weights, x, ids);

        // Allocate
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu_be);

        // Fill with valid data
        std::vector<float> ones(n_embd * n_ff, 1.0f);
        for (int e = 0; e < nm; e++) {
            ggml_backend_tensor_set(weights, ones.data(),
                                    (size_t)e * n_embd * n_ff * sizeof(float),
                                    n_embd * n_ff * sizeof(float));
        }
        std::vector<float> x_data(n_embd, 0.5f);
        ggml_backend_tensor_set(x, x_data.data(), 0, n_embd * sizeof(float));
        std::vector<int32_t> ids_data(nm, 0); // expert 0, 1, ...
        for (int i = 0; i < nm; i++) ids_data[i] = i;
        ggml_backend_tensor_set(ids, ids_data.data(), 0, nm * sizeof(int32_t));

        // Build compute graph
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, result);

        // Warmup
        for (int w = 0; w < 20; w++) {
            ggml_backend_graph_compute(cpu_be, gf);
        }
        ggml_backend_synchronize(cpu_be);

        // Timed run
        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < K; k++) {
            ggml_backend_graph_compute(cpu_be, gf);
        }
        ggml_backend_synchronize(cpu_be);
        auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / K;
        int64_t bytes = n_embd * n_ff * nm * sizeof(float);
        double gbs = (double)bytes / (us * 1e-6) / (1024.0 * 1024.0 * 1024.0);

        printf("  %d expert(s): %.1f us/call, %.1f GB/s (data: %lld bytes)\n",
               nm, us, gbs, (long long)bytes);

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    ggml_backend_free(cpu_be);
    return 0;
}