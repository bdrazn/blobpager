# blobpager Phase 1 — slot-pool pager PoC design

Date: 2026-09-20. Pilot: Qwen3-30B-A3B Q4_K_M (18,556,685,824 B file,
17,553,162,240 B = 16.35 GiB of routed experts across 48 blocks x 128
experts; top-8). Card: RTX 4080 SUPER, 15,942 MiB usable. All expert
tensors are expert-major (gguf_layout.py verdict), so a blob = 3
contiguous runs (gate/up/down), 2,654,208–3,059,712 B per expert.

## What exists now

- `blobpager/data/manifest-qwen.json` — the page table: per (layer,
  expert) byte extents inside the source GGUF + top-96 pin lists per
  layer from the train trace.
- `llama.cpp/build-cuda/` — CUDA 13.4.92 build (user-space wheel
  toolchain), verified on the card.
- Baselines (llama-bench, -t 20, pp512/tg128, build e613ef2):
  - CPU-only: pp512 105.59 tok/s, tg128 17.22 tok/s
  - `--cpu-moe` (ngl 99, n-cpu-moe 48): pp512 631.19 tok/s, tg128 24.00 tok/s
- Phase-0 hit-rate table (workset-qwen.json): pins@96 = 77.6% in-sample,
  77.9% held-out; LRU@96 = 25.1%. Pin transfer to unseen traffic: 100%.

## Budget math (why 96 pins/layer)

Card 15,942 MiB free. Attention+shared+KV+output for -ngl 99 costs
~2.5–3 GiB; leaves ~13–14 GiB for the expert pool. 96 pins/layer of
Q4_K blobs ≈ 96 × 2.66–2.92 MiB × 47 serving layers ≈ 12.9–13.9 GiB —
fits. The remaining 32/128 experts per layer are the dynamic miss set
(22.4% of draws per Phase 0).

## PoC architecture (Phase 1 scope)

**Slot pool per MoE layer, hybrid execution.** Follow the #24528
direction: hits execute on GPU from the pool; misses are NOT copied
across PCIe on the critical path — they execute on CPU (or are served
by an async prefetch that lands before the layer runs, best-effort).

- New llama.cpp example `blobpager-pager` (same zero-patch style as the
  tracer where possible): per-layer device pool tensors of 96 slots,
  prefilled with the manifest's pinned experts at load time (one bulk
  H2D copy per layer, ~2.66–2.92 MiB × 96).
- Routing observation via the same `params.cb_eval` mechanism: on each
  `ffn_moe_topk-<layer>`, partition the top-8 ids into hits (in pool)
  vs misses. Hits run against pool copies via a remapped mul_mat_id
  when feasible; misses run on the CPU-resident copy of the full
  tensor (mmap).
- v0 measurement stance: even a *correctness-only* hybrid path that
  always executes MoE on CPU while attention runs on GPU is exactly the
  `--cpu-moe` baseline (24.00 tok/s). The PoC's first win condition is
  therefore: pool-resident experts executing on GPU for the 77.6% of
  draws that are pins, with misses on CPU, beating 24.00 tok/s at the
  same settings. Second win condition: no regression at pp512.
- Fallback if remapping MUL_MAT_ID per draw proves too invasive for a
  PoC: measure the pure-I/O substrate first (pread + PCIe upload of the
  32-miss set per token, overlapped) to bound what a merged
  implementation could save. This still de-risks the pager's core
  cost model before any kernel work.

## Metrics for the verdict

- tg128 tok/s and pp512 tok/s vs both baselines (CPU-only, --cpu-moe).
- Per-token miss latency distribution when a miss must be served.
- VRAM headroom at -c 1024 to prove the pool actually fits alongside
  attention (nvidia-smi during run).

## Phase-2 door

If hybrid execution beats --cpu-moe: build the async prefetch ring +
LRU remainder (dynamic paging) as Phase 2. If it does not beat it at
77.6% hit, the locality ceiling has spoken: the pool must cover more of
the pool via smaller quant per slot or fewer layers, and the verdict
goes to disk-paging semantics (#23324) instead of VRAM caching.