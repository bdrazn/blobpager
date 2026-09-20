# blobpager Phase 1 — PoC increment results (2026-09-20)

Harness: `llama.cpp/examples/blobpager-pager` (zero-patch, cb_eval-based, CUDA build).
Pilot: Qwen3-30B-A3B-Q4_K_M. Pins: top-96/layer × 47 serving layers from the train trace.
Run: `LLAMA_GRAPH_REUSE_DISABLE=1 llama-blobpager-pager -m <gguf> -ngl 99 -ncmoe 48 -t 20 -c 1024 -b 512 -ub 512 --corpus corpus.txt --pins pins-qwen.txt --n-prefill 16 --n-gen-lines 4 --n-gen 128`

## What ran clean

- Pool fill: **12,871,139,328 B (11.99 GiB)** into VRAM in 1.61 s total —
  pread 0.91 s (14.1 GB/s) + H2D upload 0.70 s (18.5 GB/s). One-time startup tax, page-cache-warm.
- Prefill accounting: 20 ubatches, 436,160 draws, hit rate **77.68%** (Phase-0 predicted 77.6% — pins transfer perfectly to the serving path).
- Decode accounting (NEW data — Phase 0 was prefill-only): 512 greedy tokens,
  192,512 draws, hit rate **82.76%**. Decode concentrates more than prefill:
  per-token miss-union mean 64.83 experts (max 160) across all layers vs 376 draws.
- VRAM peak with pool + attention + KV: **14,391 / 15,942 MiB**. The 96-pin pool fits with room.
- Zero mismatches; 532 aux(MTP)-head observations skipped as designed.

## The union numbers (what the hybrid must schedule)

| Phase | hit union / layer | miss union / layer | per-ubatch totals |
| --- | --- | --- | --- |
| Prefill (512-tok ubatch) | 96.00 (all pins) | 32.00 (all non-pins) | hit 4,512 / miss 1,504 experts |
| Decode (1 tok) | 6.62 mean (max 8) | 1.38 mean (max 7) | hit 311 / miss 64.8 experts |

Prefill: CPU executes exactly the 32-expert complement = 25% of the
--cpu-moe expert work; GPU executes 75%. Decode: CPU executes 1.38/8 = 17%
of per-layer draws — a ~4.8x reduction in CPU expert work vs the 25.23 tok/s baseline.

## ncmoe sweep (llama-bench, -t 20, -r 2, same build)

| CPU-expert layers | pp512 tok/s | tg128 tok/s |
| --- | --- | --- |
| 48 (baseline) | 615.22 ± 13.57 | 25.23 ± 3.47 |
| 44 | 671.26 | 28.87 |
| 40 | 729.20 | 24.97 |
| 36 | 781.67 | 15.73 |
| 32 | 869.09 | 17.30 |
| 28 | 966.22 | 19.81 |
| 24 | 1064.91 | 21.71 |
| 20 | 1191.13 | 27.14 |
| 16 | 1410.29 | 35.67 |
| 12 | 1672.44 | 50.31 |
| 8 | — | fails: context creation OOM |

Two readings:

1. **Prefill scales cleanly and steeply**: every CPU-expert layer moved to
   VRAM buys ~21–24 tok/s of prefill. +172% at ncmoe=12. The pager's
   prefill (25% of expert work on CPU) should land well above the 615 baseline.
2. **Decode is non-monotonic — a mid-zone dead valley.** tg128 dips BELOW
   the all-CPU-expert baseline (25.23) across 32–40 CPU layers (15.73–17.30),
   recovers only at ≤20 CPU layers (27.14) and pays off hard at 12 (50.31,
   +99%). The per-layer GPU↔CPU alternation penalty is real (exactly what
   RFC #24528's failed PRs hit). ncmoe=8 (13.6 GiB of experts + KV) does not fit → the fit
   boundary sits near ncmoe≈12, i.e. ~36 GPU-resident expert layers ≈ 12.6 GiB.

## Verdict-relevant conclusions

- The static-pin pool architecture is validated end-to-end: manifest → pins file
  → device pool → live hit/miss accounting on the real serving path, no source patches.
- Prefill win is near-certain (measured scaling + 77.7% hit rate).
- Decode win requires **avoiding the mid-zone execution pattern**: the pager's
  CPU miss set per layer is tiny (1.38 experts), so per-layer synchronous
  alternation must not be used. The design must either (a) defer all miss
  execution to one consolidated CPU pass per ubatch, or (b) run misses on the
  CPU while the GPU proceeds (true overlap), not layer-by-layer ping-pong.
- The decode hit rate (82.8%) exceeding prefill (77.7%) means the CPU share of
  decode work is even smaller than budgeted — headroom for the pool to win at tg.

## Next increment (the actual hybrid execution)

In-graph remapped MUL_MAT_ID: per layer, a 96-slot pool tensor + a
remap table (expert id → slot, -1 = miss); hits execute against the pool on
GPU, the miss set (mean 1.38 experts) accumulates to a consolidated CPU pass.
Win condition unchanged: beat 25.23 tg128 and 615 pp512 at identical quant/settings.

## Environment gotchas (all solved, see project memory)

1. GGML_CUDA silently flips OFF on reconfigure without -DCUDAToolkit_ROOT — binaries run CPU-only.
2. llama_batch_get_one is deprecated + yields NO logits on this master; use explicit llama_batch with batch.logits[last]=true; get_logits_ith is batch-token-indexed.
3. cb_eval + repeated-shape decode loops crash without LLAMA_GRAPH_REUSE_DISABLE=1.