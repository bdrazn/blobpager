# blobpager Phase 0 — Locality Report (measured, not assumed)

Date: 2026-09-19. Workspace: testlocal. Tracer: `llama.cpp/examples/blobpager-trace/`
(zero-patch, `params.cb_eval` observing `ffn_moe_topk-<layer>` tensors).

## What ran

- Corpus: `blobpager/tools/gen_corpus.py` (seed 20260919) — 480 train + 60 held-out
  seller-desk requests (~55 tokens avg), 85% wrapped in serving shapes
  (reply-draft / classify / summarize / follow-up) with a shared system prefix,
  15% raw fragments. Pattern bank was unavailable (token not configured); the
  scenario library is hand-built.
- Trace format: JSONL per (req, pos, layer) → top-k expert ids.
- Tracer findings along the way: GLM deepseek2 exposes an **MTP head as a
  trailing extra MoE "layer" running on exactly 1 token** (layer 46 of 47
  blocks) — observed once per request, skipped by design (`n_aux` counter).
  Qwen3 shows the same aux-head pattern.

## Models measured

| | GLM-4.7-Flash-REAP-23B-A3B Q4_K_S | Qwen3-30B-A3B Q4_K_M (pilot) |
| --- | --- | --- |
| Arch | deepseek2, MLA | qwen3moe, GQA |
| Expert pool | 48 × 45 MoE layers, top-4 | 128 × 47 layers, top-8 |
| File | 13,328,455,392 B | 18,556,685,824 B |
| Train trace | 480 req, 26,441 tok, 1,189,845 records | 480 req, 27,387 tok, 1,287,189 records |
| Distinct keys | 2,160 = full pool | 6,016 = full pool |

## Finding 1 — routing is near-uniform on both models

Share of all expert draws covered by the top-j experts per layer (train):

| top-j | GLM REAP (uniform j/48) | Qwen3-30B (uniform j/128) |
| --- | --- | --- |
| top-16 | 35.2% (33.3%) | 13.7% (12.5%) |
| top-24 | 52.2% (50.0%) | 20.4% (18.8%) |
| top-32 | 68.8% (66.7%) | 27.1% (25.0%) |
| top-64 | 100% (pool exhausted) | 53.0% (50.0%) |
| top-96 | — | 77.6% (75.0%) |

Concentration beats uniform by only ~2–4 points on both architectures, and the
per-layer spread is razor-thin (Qwen top-64: min 52.8% / med 53.0% / max 53.2%).
Every layer touches every expert across 26k tokens. **The "our domain sticks to
a small expert set" hypothesis is falsified for prefill routing on these two
models.** Notably, GLM's REAP pruning (48 survivors selected by importance) may
actively flatten the distribution; but unpruned Qwen3-30B is near-uniform too —
load-balancing looks like a trained property of modern routers, not an artifact.

## Finding 2 — policy: pins beat LRU by 3×; pins transfer perfectly

Hit rate (%) at budget S slots per layer, seller-desk traffic:

Qwen3-30B (pool 128):

| S/layer | 16 | 32 | 48 | 64 | 96 | 128 |
| --- | --- | --- | --- | --- | --- | --- |
| LRU | 0.5 | 1.7 | 3.9 | 7.8 | 25.1 | 100 |
| freq pins (cal) | 13.7 | 27.1 | 40.2 | 53.0 | 77.6 | 100 |
| pins train→held-out | 13.6 | 27.0 | 40.2 | 53.2 | 77.9 | 100 |

GLM REAP (pool 48): LRU 8.3% at S=24 vs pins 52.1%; identical pattern.

Two robust takeaways:

1. **LRU is pathological on these streams** — re-use gaps (~pool/top-k draws)
   exceed any realistic cache depth, and the stream is anti-recency (draws
   avoid the immediately-previous window). Never use plain LRU for expert
   caching here; frequency pins or pins+LRU hybrid only.
2. **Pin sets transfer 100% to unseen traffic** (0 unseen keys held-out;
   train→held-out hit rates equal in-sample rates to ±0.3). Calibrate once on
   desk traffic; the pins generalize. The ceiling is the concentration, not
   transfer.

## Finding 3 — the decisive number is budget-share, not locality

Feasible-budget hit rates are real but mechanical: hit ≈ S/E plus a small skew
bonus. For Qwen3-30B on the 4080 SUPER (16,376 MiB): attention+KV+shared
overhead ≈ 1.5–2 GB leaves ~14 GB for experts ≈ 96 slots/layer of 128 →
**~77.6% hit rate with pins** — an 18.6 GB model running on a 16 GB card, with
~22% of FFN draws served from elsewhere. But that 77.6% is ~75% slot-share
+ ~2.6 points of actual domain locality. Prediction is not the lever; the
budget fraction is.

This reproduces the llama.cpp #24528 replay-study conclusion on our own data:
prediction is the seasoning, overlap is the meal.

## Gate verdict (Phase 0 decision gate)

**Build overlap-first.** The pager's win must come from the execution model:

- expert-contiguous re-layout (page-fault reduction, per #23324),
- async pipelined fetch of the next layer's experts during compute,
- hybrid hit/miss execution: hits on GPU, misses executed on CPU (no PCIe
  copy on the critical path), per #24528,
- cache policy: static frequency pins (they also serve as the always-resident
  floor) + LRU only in the flexible remainder,
- prediction/prefetch: demote to a small opportunistic layer, not the core.

Phase 1 (next): expert-contiguous GGUF re-layout + a slot-pool pager PoC on the
pilot Qwen3-30B with a 96-slot/layer budget, measuring wall-clock tok/s vs
`--cpu-moe` baseline at identical quant.

## Caveats

- Prefill-only: decode-time routing (generated tokens) unmeasured; chat-template
  prefixes in real serving are longer than the corpus's, which can only help
  concentration, but was not measured here.
- Synthetic corpus (hand-built scenarios). Real reply-desk logs should be traced
  before Phase 2 hardening.
- Aux-head observations (1 per request) excluded; multi-request tracing verified
  exact (records == tokens × layers on both archs).