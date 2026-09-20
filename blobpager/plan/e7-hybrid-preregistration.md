# E7 Pre-registration — Hybrid hit/miss execution

Registered: 2026-09-20, **before implementation**. No post-hoc metric changes;
results reported regardless of outcome.

## Charter context

Win condition (from phase1.md, unchanged): beat the incumbent `--cpu-moe`
baseline at identical quant and settings — **pp512 > 615.22 tok/s** and
**tg128 > 25.23 tok/s** (canonical sweep row: 615.22 ± 13.57 / 25.23 ± 3.47).

Inputs already measured (E5, `blobpager/plan/phase1-poc-notes.md`):
prefill miss union = 32/128 experts per layer (25.0%); decode miss mean =
1.38 of 8 draws per layer (17.25%), per-token miss union 64.83 experts across
47 layers (max 160); pool = 4,512 blobs, 12,871,139,328 B; VRAM peak
14,391 / 15,942 MiB.

## E7a — Pool microbenchmark

**Question.** What does each execution arm cost in isolation, at the measured
miss load?

**Method.** In `blobpager-pager` (bench mode): take one serving layer.
Device arm: 96-slot pool tensor + 1 zero-filled dummy slot = 97-slot device
tensor; slot table `SLOTS[128]` (expert id → slot, miss → 96); synthetic
activation x [n_embd] fp32; ids [8] sampled at the measured decode miss rate
(1.38/8 ≈ 17.25%); graph: `r = get_rows(SLOTS, ids)`;
`out = mul_mat_id(pool97, x_bcast, r)`; time K = 1000 iterations with
synchronize per iteration. Host arm: `mul_mat_id` against the CPU-resident
full expert tensor for that layer, miss-only ids (1 and 2 experts).

**Metrics.** µs per 8-draw layer call; effective GB/s.

**Predictions.** GPU pool path < 20 µs per layer call; host miss path
≈ 0.12–0.15 ms per layer (17.25% of the ~0.7–0.85 ms/layer implied by the
39.6 ms/token baseline at 25.23 tok/s, ~1.0 GB/token expert reads).

## E7b — Hybrid v1 (sequential per-layer)

**Question.** Does GPU-hits + inline-CPU-miss execution beat the incumbent at
decode, or does per-layer synchronization eat the margin (the #24528 Metal
lesson)?

**Method.** Per layer: router runs on GPU; ids remapped in-graph
(`get_rows(SLOTS, ids)`, misses clamped to the zero dummy slot);
`mul_mat_id` against the 97-slot pool on GPU produces hit terms (miss terms
zero); the miss set (mean 1.38 experts/layer) executes on the CPU full tensor
in the same ubatch; contributions combined. Exact inference — no approximation;
no fallback weights. Validate logits against the `-ncmoe 48` run on identical
tokens (agreement threshold to be fixed in code before the run; any mismatch
invalidates the run).

**Primary endpoint.** tg128 tok/s (greedy, 128 tokens, -t 20, identical bench
protocol to the sweep). Secondary: pp512; per-phase breakdown
(GPU hit time / CPU miss time / sync overhead).

**Pre-registered predictions and refutation bands (tg128).**

- Predicted band: **35–65 tok/s** (point estimate ~50): CPU expert work drops
  to ~17.25% of baseline (~6–7 ms/token), leaving ~32 ms/token of budget for
  attention + synchronization before the win disappears.
- **Refuted** if tg128 < 25.23 (sequential hybrid loses to the incumbent →
  per-layer sync dominates; pivot to consolidated-miss-pass v2 or true overlap).
- Marginal win: 25.23 ≤ tg128 < 36 → continue engineering; measure the
  sync-tax decomposition explicitly.
- Strong win: tg128 ≥ 36 (+43%) → opens the Phase-2 async-prefetch door.

**Prefill guard.** Hybrid prefill must not regress below 615.22 pp512
(predicted band 900–1200: CPU executes exactly 25% of expert work per layer,
between the 615.22 all-CPU and 1672.44 ncmoe=12 rows).

**Falsification of the architecture (not just v1).** If both E7b and a
consolidated/overlapped v2 fail to beat 25.23, the VRAM-caching thesis is
weakened for this model/card and the Phase-2 door defaults to #23324-style
disk-paging semantics (which the 3-run contiguous blob layout already
supports).