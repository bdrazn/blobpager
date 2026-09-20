# blobpager — demand paging for MoE expert blobs

**Working name — rename at will.**

**Goal:** consumer hardware runs large MoE models at useful speed by treating routed-expert
weights as pageable blobs. A page table over the blobs, VRAM as a bounded cache, eviction and
insertion mid-generation. The whole model never has to fit.

## Why this is winnable

- OS paging already proves bigger-than-memory works (mmap + page cache) — but it is blind to model
  structure (4 KB pages, no expert semantics) and thrashes when it spills.
- Research proves expert-level paging + prediction works on personal machines: Apple "LLM in a
  flash", Fiddler, MoE-Infinity, Pre-gated MoE, mixtral-offloading.
- llama.cpp upstream has the idea live but unshipped: Discussion **#23324** = working disk-paging
  PoC (Qwen3-30B-A3B-Q6_K with 27 GiB of experts runs on a 16 GB M1 Pro at **13 tok/s**), and
  Discussion **#24528** = open RFC for a VRAM expert cache with hybrid hit/miss execution. Five
  prior PRs died on the same mistake: cache misses on the critical path as synchronous PCIe copies.

## Our bet

1. **Locality is domain-shaped.** The #24528 replay study says a perfect predictor only cuts ~12.8%
   of total I/O in general traffic — but a text-calibrated 10-token pass lifted hit rate from 6% to
   42%, and real-estate seller-desk traffic is a far narrower domain than ShareGPT. Phase 0
   measures OUR number.
2. **Misses never wait.** The win comes from the execution model: hits on GPU, misses computed on
   CPU concurrently, prefetch overlapped with compute, expert-contiguous layout so one blob spans
   few pages (relayout cut page faults 36× — 192 vs 6,912 pages per expert).
3. **Wall-clock is the only metric.** Hit rate is diagnostic, not the goal.

## Modules

- `blobpack` — repacks a GGUF expert-major and emits `manifest.json` with per-(layer, expert)
  byte extents. The table is only as good as the layout it indexes.
- `tracelog` — llama.cpp patch dumping `{req, pos, layer, expert}` JSONL per token, env-gated
  (`BLOBPAGER_TRACE=<path>`), streaming append, negligible overhead.
- `workset` — analyzer: distinct keys, per-token and per-conversation working sets, hit-rate
  curves (LRU / LFU / LFRU / calibrated pins / oracle), prefetch lookahead accuracy. Methodology
  kept comparable to the #24528 replay study so our numbers are comparable to theirs.
- `pager` — the runtime: per-layer slot pool in VRAM, expert→slot table, LRU + calibrated pins,
  async prefetch ring, miss-executes-on-CPU path so nothing ever stalls on PCIe.

## Phases

0. **Trace & measure** our corpus on a pilot MoE GGUF. Decision gate: calibrated-pinning hit rate
   at a feasible cache budget.
1. **Static:** profile corpus → pin top-N experts per layer + expert-contiguous relayout.
2. **Dynamic:** slot-pool pager with async prefetch; miss path off the critical path.
3. **Upstream:** package our measurements as evidence for the open RFCs (#24528, #23324).

## Non-goals

- No weight regeneration. SGD leaves no recoverable recipe inside the blobs; the "formula" we
  exploit is the model predicting its own routing, plus corpus locality.
- No KV-cache paging in v0 — separate frontier, different physics.
- No training/offloading frameworks — this is inference serving on one consumer box.

## Links

- llama.cpp Discussion #24528 — RFC: MoE expert cache (Jun 2026)
- llama.cpp Discussion #23324 — on-demand disk paging PoC (May 2026)
- Fiddler arXiv 2402.07033 · MoE-Infinity arXiv 2401.14361 · Pre-gated MoE arXiv 2308.12066
- Apple LLM-in-a-flash arXiv 2312.11514 · dvmazur/mixtral-offloading