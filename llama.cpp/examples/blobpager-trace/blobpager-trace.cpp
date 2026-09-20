// blobpager-trace: MoE expert-routing tracer for the blobpager project.
//
// Runs a corpus of prompts through a MoE GGUF (prefill only — routing depends
// only on the input tokens, no sampling/generation needed) and captures the
// selected expert ids per layer per token via the backend-scheduler eval
// callback. The graph builder names the router top-k tensor "ffn_moe_topk-<il>"
// for every MoE layer (see llama_context::graph_get_cb), which is the only
// hook we need: zero llama.cpp source changes.
//
// Output: JSONL, one record per (request, token position, layer):
//   {"req":i,"pos":j,"layer":L,"exp":[e0,e1,...]}
// preceded by one meta record with model metadata.
//
// Usage:
//   llama-blobpager-trace -m model.gguf --corpus corpus.txt --out trace.jsonl -c 1024 -b 512 -ub 512 -ngl 0
//
// Constraints (checked loudly):
//   * every corpus request must fit in ONE ubatch (tokens <= -b, -ub and -c),
//     so a whole request's routing lands in a single ids tensor
//   * build a CPU-only (or CPU-resident-expert) binary so the ids tensor is
//     in a host buffer — we read t->data directly

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct trace_state {
    std::ofstream out;
    int64_t       req_idx  = -1;   // index of the request currently in flight
    int64_t       req_len  = 0;    // token count of the in-flight request
    int64_t       n_tokens = 0;    // total tokens traced
    int64_t       n_records = 0;   // (req,pos,layer) records written
    int64_t       n_skip    = 0;   // observations skipped (shape mismatch)
    int64_t       n_aux     = 0;   // aux/MTP-head observations skipped (n_tok == 1)
    int64_t       obs_count = 0;   // total topk observations seen
    bool          meta_written = false;
};

static const char kPrefix[] = "ffn_moe_topk-";

static bool trace_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (trace_state *) user_data;

    // ask phase: only observe the router top-k nodes
    if (ask) {
        return strncmp(t->name, kPrefix, sizeof(kPrefix) - 1) == 0;
    }

    if (strncmp(t->name, kPrefix, sizeof(kPrefix) - 1) != 0) {
        return true;
    }

    // layer index is baked into the name: "ffn_moe_topk-<il>"
    const int layer = atoi(t->name + sizeof(kPrefix) - 1);

    if (t->type != GGML_TYPE_I32) {
        LOG_ERR("%s: expected I32 ids tensor, got %s\n", __func__, ggml_type_name(t->type));
        exit(1);
    }

    // [n_expert_used, n_tokens] — one column per token in this ubatch
    const int64_t n_used = t->ne[0];
    const int64_t n_tok  = t->ne[1];

    // debug stream: first observations, so shape anomalies are visible live
    if (st->obs_count < 60) {
        LOG_INF("%s: obs %lld req %lld layer %d n_tok %lld req_len %lld\n", __func__,
                (long long) st->obs_count, (long long) st->req_idx, layer, (long long) n_tok, (long long) st->req_len);
    }
    st->obs_count++;

    if (n_tok != st->req_len) {
        // Auxiliary/MTP prediction heads run as a trailing extra MoE "layer"
        // over exactly one token (the next-token draft). That routing is not
        // part of the request's prefill — count it and skip.
        if (n_tok == 1 && st->req_len > 1) {
            st->n_aux++;
            return true;
        }
        if (st->n_skip < 40) {
            LOG_ERR("%s: MISMATCH req %lld layer %d ubatch n_tok %lld != req_len %lld — skipping (raise -b/-ub, "
                    "lower -c, or shorten the corpus lines)\n",
                    __func__, (long long) st->req_idx, layer, (long long) n_tok, (long long) st->req_len);
        }
        st->n_skip++;
        return true;
    }

    // CPU-resident tensors: data is directly readable
    const int32_t * ids = (const int32_t *) t->data;

    if (!st->meta_written) {
        st->out << "{\"type\":\"meta\",\"expert_used\":" << n_used << "}\n";
        st->meta_written = true;
    }

    std::ostringstream line;
    for (int64_t j = 0; j < n_tok; ++j) {
        line.clear();
        line.str("");
        line << "{\"req\":" << st->req_idx << ",\"pos\":" << j << ",\"layer\":" << layer << ",\"exp\":[";
        for (int64_t k = 0; k < n_used; ++k) {
            if (k) {
                line << ",";
            }
            line << ids[j * n_used + k];
        }
        line << "]}\n";
        st->out << line.str();
    }

    st->n_tokens  += n_tok;
    st->n_records += n_tok;
    return true;
}

// strip "--corpus <path>" / "--out <path>" from argv before common_params_parse
static std::vector<char *> strip_custom_args(int argc, char ** argv, std::string & corpus, std::string & outpath) {
    std::vector<char *> rest;
    rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--corpus") == 0 && i + 1 < argc) {
            corpus = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            outpath = argv[++i];
        } else {
            rest.push_back(argv[i]);
        }
    }
    return rest;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string corpus_path;
    std::string out_path;

    std::vector<char *> rest = strip_custom_args(argc, argv, corpus_path, out_path);

    if (corpus_path.empty() || out_path.empty()) {
        LOG_ERR("usage: %s -m model.gguf --corpus corpus.txt --out trace.jsonl [common params]\n", argv[0]);
        return 1;
    }

    trace_state st;
    st.out.open(out_path, std::ios::binary);
    if (!st.out) {
        LOG_ERR("%s: cannot open output '%s'\n", __func__, out_path.c_str());
        return 1;
    }

    common_params params;
    common_init();

    if (!common_params_parse(rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    params.warmup       = false;   // warmup forces all experts — would pollute the trace
    params.cb_eval      = trace_cb;
    params.cb_eval_user_data = &st;
    params.embedding    = false;

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

    // meta record: dump MoE-relevant model metadata
    {
        char key[256];
        char val[256];
        std::string meta = "{\"type\":\"meta\"";
        const int32_t n_meta = llama_model_meta_count(model);
        for (int32_t i = 0; i < n_meta; ++i) {
            if (llama_model_meta_key_by_index(model, i, key, sizeof(key)) != 0) {
                break;
            }
            const std::string k(key);
            if (k.find("general.architecture") != std::string::npos ||
                k.find(".block_count")          != std::string::npos ||
                k.find(".expert_count")         != std::string::npos ||
                k.find(".expert_used_count")    != std::string::npos ||
                k.find(".leading_dense_block_count") != std::string::npos ||
                k.find(".expert_feed_forward_length") != std::string::npos) {
                if (llama_model_meta_val_str(model, key, val, sizeof(val)) == 0) {
                    meta += std::string(",\"") + key + "\":\"" + val + "\"";
                }
            }
        }
        meta += "}\n";
        st.out << meta;
    }

    // read corpus: one request per line
    std::ifstream corpus(corpus_path);
    if (!corpus) {
        LOG_ERR("%s: cannot open corpus '%s'\n", __func__, corpus_path.c_str());
        return 1;
    }

    llama_memory_t mem = llama_get_memory(ctx);

    std::string line;
    int64_t req_idx = 0;
    while (std::getline(corpus, line)) {
        if (line.empty()) {
            continue;
        }

        std::vector<llama_token> tokens = common_tokenize(vocab, line, add_bos, true);
        if (tokens.empty()) {
            continue;
        }
        if (tokens.size() > n_ctx) {
            tokens.resize(n_ctx);
        }
        if ((int64_t) tokens.size() > (int64_t) params.n_batch) {
            LOG_WRN("%s: request %lld has %zu tokens > n_batch %u — truncating (raise -b/-ub)\n",
                    __func__, (long long) req_idx, tokens.size(), params.n_batch);
            tokens.resize(params.n_batch);
        }

        st.req_idx = req_idx;
        st.req_len = (int64_t) tokens.size();
        LOG_INF("%s: req %lld: %zu tokens (n_batch %u, n_ubatch %u)\n", __func__,
                (long long) req_idx, tokens.size(), params.n_batch, params.n_ubatch);

        // each request must not see the previous request's KV
        llama_memory_seq_rm(mem, 0, 0, -1);

        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size()))) {
            LOG_ERR("%s: decode failed on request %lld\n", __func__, (long long) req_idx);
            return 1;
        }

        if (req_idx % 50 == 0) {
            LOG_INF("%s: request %lld done (%lld tokens traced)\n", __func__, (long long) req_idx, (long long) st.n_tokens);
        }
        ++req_idx;
    }

    st.out.flush();
    LOG_INF("%s: done. %lld requests, %lld tokens, %lld records, %lld aux-head obs skipped -> %s\n",
            __func__, (long long) req_idx, (long long) st.n_tokens, (long long) st.n_records,
            (long long) st.n_aux, out_path.c_str());

    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}