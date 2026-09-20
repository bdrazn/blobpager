#pragma once

// blobpager hybrid execution (E7b) — internal libllama module.
//
// Env-gated: active only when BLOBPAGER_PINS=<pins file> is set at model load.
// At the end of llama_model::load_tensors, allocates per-serving-layer GPU
// pools (top-96 pinned experts + 1 zero dummy slot = 97 slots) filled from the
// CPU-resident expert tensors, a per-layer GPU slot-remap table SLOTS[n_expert]
// (expert -> slot, miss -> dummy), and per-layer GPU correction tensors
// [n_embd, BLOBPAGER_MAXT] F32.
//
// During decode (ubatch n_tokens <= maxt), llm_graph_context::build_moe_ffn
// routes gate/up/down through the pools via a get_rows remap (misses land on
// the zero dummy, contributing exact 0), and adds the correction tensor to the
// MoE output. The eval-callback wrapper (installed in place of the user's
// cb_eval; the user's callback is chained) intercepts each ffn_moe_topk-<il>
// node — the scheduler synchronizes the backend before the callback fires
// (ggml-backend.cpp) — computes the miss experts on the CPU from the same
// quantized expert bytes with the same gating weights, and writes the
// correction into the GPU tensor. Every top-8 draw is computed exactly once:
// hits from the pool, misses from the CPU tensor. Exact inference.
//
// With BLOBPAGER_PINS unset every hook is a no-op and the model runs
// byte-identical to unpatched llama.cpp.

#include "ggml.h"
#include "ggml-backend.h"

struct llama_model;

// true once setup succeeded (BLOBPAGER_PINS set, pools allocated and filled)
bool llama_blobpager_active();
void llama_blobpager_selftest(const struct llama_model & model);

// called once at the end of load_tensors; reads BLOBPAGER_PINS / BLOBPAGER_MAXT
void llama_blobpager_setup(struct llama_model & model);

// install before/instead of the raw cparams.cb_eval on the sched; the wrapper
// chains to this user callback (accounting) and adds the CPU miss pass
void llama_blobpager_set_user_cb(ggml_backend_sched_eval_callback cb, void * user_data);

// the sched eval callback (wraps the user's)
bool llama_blobpager_cb_eval(struct ggml_tensor * t, bool ask, void * user_data);

// graph-builder hooks, called from llm_graph_context::build_moe_ffn
bool          llama_blobpager_hybrid(int il, int64_t n_tokens, bool norm_w, float w_scale);
ggml_tensor * llama_blobpager_remap(struct ggml_context * ctx, struct ggml_tensor * selected_experts, int il);
ggml_tensor * llama_blobpager_pool(int il, int which);   // 0 = up, 1 = gate, 2 = down
ggml_tensor * llama_blobpager_correction_view(struct ggml_context * ctx, int il, int64_t n_tokens);