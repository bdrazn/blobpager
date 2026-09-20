# BlobPager

**Demand-paged Mixture-of-Experts inference on a single consumer GPU.**

BlobPager treats routed-expert weights as pageable blobs: the GPU holds a bounded expert pool
plus a page table over the full model, hits execute on the GPU, misses execute on the CPU —
and the whole model never has to fit in VRAM.

This repository is the complete research program: measurement infrastructure, traces,
working-set analysis, pre-registered experiments, a working **exact hybrid executor** inside
llama.cpp, and the raw artifacts behind every published number.

**Status:** E0–E7b complete · E8 (clean head-to-head) pending · next: v2 consolidated/async miss pass.

---

## The problem

A modern MoE model puts almost all of its parameters in routed experts. On the pilot hardware
(RTX 4080 SUPER, 16 GB), the expert set of Qwen3-30B-A3B-Q4_K_M alone is
**17,553,162,240 B (16.35 GiB)** — more than the card's usable 15,942 MiB. Full-GPU execution
is physically impossible regardless of software, and the incumbent (`--cpu-moe`: GPU attention +
CPU-resident experts) pays ~26.5 ms of CPU expert work per decoded token.

Prior work (Apple's *LLM in a flash*, Fiddler, MoE-Infinity, Pre-gated MoE) and two live
llama.cpp discussions (#23324 — disk-paging PoC, #24528 — open RFC for a VRAM expert cache) show
the idea works but keep dying on the same mistake: **cache misses on the critical path as
synchronous PCIe copies**. Five prior PR attempts failed this way per RFC #24528.

## Our bet

1. **Locality is domain-shaped, but budget share is the decisive number.** On our seller-desk
   traffic the feasible-budget hit rate (~77.6% at 96/128 slots) decomposes into ~75% slot
   share plus ~2.6 points of genuine domain skew. Prediction is the seasoning; overlap is the
   meal. This reproduces the #24528 replay-study conclusion on our data.
2. **Misses never wait.** The win comes from the execution model — what runs where, and what
   never crosses PCIe — not from predicting routing.
3. **Wall-clock is the only metric.** Hit rate is diagnostic, not the goal.

## Results so far (E0–E7b)

Full detail with provenance for every number: [`blobpager/article/paper.md`](blobpager/article/paper.md)
(raw artifacts in `blobpager/data/` and `blobpager/logs/`; session-by-session notebook in
[`blobpager/article/lab-log.md`](blobpager/article/lab-log.md)).

### Measurement (E0–E5)

| Result | Number |
| --- | --- |
| E0 — expert set vs usable VRAM | 17,553,162,240 B (16.35 GiB) vs 15,942 MiB → full-GPU impossible |
| E1 — routing traces (Qwen train) | 480 req, 27,387 tokens, 1,287,189 records |
| E2 — GGUF expert layout | already expert-major: 3 contiguous byte runs per expert (no #23324-style scatter) |
| E3 — pins train→heldout hit rate at 96/128 slots | **77.9%** (LRU at same budget: 25.1%) |
| E3 — GLM REAP cross-check (24/48 slots) | pins 52.1% vs LRU 8.3% — same pattern |
| E4 — pool staging (4,512 blobs, 11.99 GiB) | one-time **1.61 s** (pread 14.1 GB/s + H2D 18.5 GB/s) |
| E5 — live hit rate on the serving path | prefill **77.68%**, decode **82.76%** |
| E5 — decode miss load | mean **1.38 of 8 draws** per layer (17.25%); VRAM peak 14,391 / 15,942 MiB |

Two structural findings: (a) routing is near-uniform and LRU is pathological on these streams —
frequency pins transfer perfectly to unseen traffic (0 unseen keys); (b) decode concentrates
*more* than prefill (82.76% vs 77.68%), raising the decode-side ceiling for hybrid execution.

### The design-space sweep (E6)

`llama-bench`, `-t 20 -r 2`, canonical build e613ef2:

| CPU-expert layers | pp512 tok/s | tg128 tok/s |
| --- | --- | --- |
| 48 (baseline) | 615.22 ± 13.57 | 25.23 ± 3.47 |
| 20 | 1191.13 | 27.14 |
| 16 | 1410.29 | 35.67 |
| 12 | 1672.44 | 50.31 |
| 8 | — | fails: context-creation OOM |

Prefill scales cleanly (+172% at ncmoe=12), but decode is **non-monotonic**: a mid-zone dead
valley at 32–40 CPU layers (15.73–17.30 tok/s) sits *below* the all-CPU-expert baseline —
hybrid execution done naively (per-layer GPU↔CPU ping-pong) is worse than no hybrid at all.
This is the failure mode that killed the five prior PR attempts, now quantified.

### Hybrid execution v1 (E7a/E7b) — built, validated, timed

`src/llama-blobpager.{h,cpp}` + three surgical hooks in llama.cpp; active only under
`BLOBPAGER_PINS=<pins file>`, byte-identical to unpatched llama.cpp otherwise. Per serving
layer at decode: a 97-slot GPU pool (96 frequency pins + zero dummy), `get_rows(SLOTS, ids)`
remap (miss → dummy), gate/up/down `mul_mat_id` against the pools produce hit terms (miss lanes
exactly 0), and an eval-callback stop at each router computes **only the missed experts** on the
CPU (mean 1.38/layer) and writes a correction the graph adds after the weighted sum. Every
top-8 draw is computed exactly once; no fallback weights; pool (13.0 GiB) staged at load in 1.329 s.

**Exactness validation** (thresholds fixed in code before the runs):

| Check | Result |
| --- | --- |
| 0-pin hybrid (all-miss) vs baseline | **bit-exact** — max diff 0.000000, 0 token mismatches |
| Pool fill read-back (47 layers × 97 slots × 3 tensors = 13,536 slots) | **0 mismatches** |
| GPU-vs-CPU kernel self-test (layer-0 gate pool, 8 lanes) | max diff **0.000455** (rounding scale) |
| 96-pin hybrid vs baseline, 64-token decode | sampled tokens identical **65/65** |
| Control: upstream `-ncmoe 20` vs `-ncmoe 48` (no blobpager) | tokens identical 17/17 |

**Timing** (both arms through the identical observation harness, 2 prefill + 32-token decode,
canon `LLAMA_GRAPH_REUSE_DISABLE=1`): baseline-with-callback 121.4 s vs hybrid-with-callback
**74.8 s — 1.622× faster end-to-end**. Host miss pass measured at **185.8 µs/layer** (521 miss
draws over 376 invocations = 1.386/layer, matching E5's 1.38). Per token, CPU expert work drops
from ~26.5 ms to ~9.3 ms. The v2 projection (consolidated/async miss pass, one sync per ubatch
instead of 47) is ~13.1 ms attention + ~9.3 ms hybrid MoE ≈ 22.4 ms/token ≈ **45 tok/s**, mid-band
of the pre-registered 35–65.

### A numerics finding worth its own note

The 96-pin hybrid's logits moved by up to **4.866536** vs baseline — far beyond rounding. Before
accepting a bug we ran a control **with no BlobPager involvement**: upstream llama.cpp with
`-ncmoe 20` vs `-ncmoe 48` shifts the same logits by up to **1.94398**. Conclusion:
GPU-vs-CPU expert-matmul rounding, compounded through 47 residual layers, shifts final MoE
logits by O(1) in *stock* llama.cpp — without changing greedy outputs. Any exact-inference
claim in hybrid MoE serving must therefore be stated as *architectural exactness* (proven
bit-exact in the all-CPU limit) plus *token identity* (holds), not logit equality.
Pre-registered threshold analysis and the full argument: paper §E7b, §7.

## Repository layout

```
README.md                      ← this file
LICENSE                        (MIT)
blobpager/
  article/paper.md             the article (results enter only as artifacts)
  article/lab-log.md           session-by-session notebook
  plan/                        pre-registrations (hypothesis → method → metrics → refutation bands)
  tools/                       gen_corpus.py, gguf_meta.py, gguf_layout.py, blobpack_manifest.py,
                               emit_pins.py, workset.py
  data/                        corpus, traces (GLM + Qwen), manifest, pins, pager run outputs
  logs/                        raw benchmark logs, E7a microbench JSON, E7b logits dumps
llama.cpp/                     snapshot of llama.cpp (base commit e613ef2) + BlobPager:
  src/llama-blobpager.{h,cpp}  the hybrid executor (pool, SLOTS remap, host miss pass)
  src/llama-graph.cpp          build_moe_ffn hybrid branch (pool arms + correction add)
  src/llama-model.cpp          model-load pool construction hook
  src/llama-context.cpp        eval-callback wrapper (router-stop observation)
  examples/blobpager-trace/    zero-source-patch routing-trace harness
  examples/blobpager-pager/    live hit/miss accounting + --dump-logits validation harness
  examples/blobpager-bench/    E7a 97-slot pool microbenchmark
```

Excluded from the repo (recreate locally): `blobpager/models/` (the 18.6 GB pilot GGUFs),
`llama.cpp/build*`, `llama.cpp/models` (upstream test models). Model download:
`Qwen3-30B-A3B-Q4_K_M` (18,556,685,824 B) and `GLM-4.7-Flash-REAP-23B-A3B-Q4_K_S` from the
usual GGUF release pages.

## Reproduce

Build (CUDA 13.4.92 user-space pip wheels, no sudo, arch 89):

```bash
cd llama.cpp
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda --target llama-cli llama-bench blobpager-pager blobpager-bench -j
```

Baseline canon (the row every experiment is measured against):

```bash
cd build-cuda/bin
LLAMA_GRAPH_REUSE_DISABLE=1 ./llama-bench -m <model>.gguf -ngl 99 -ncmoe 48 -t 20 -p 512 -n 128 -r 2
```

Hybrid decode (96 frequency pins + host miss pass):

```bash
cd build-cuda/bin
BLOBPAGER_PINS=../../../blobpager/data/pins-qwen.txt \
LLAMA_GRAPH_REUSE_DISABLE=1 ./blobpager-pager -m <model>.gguf \
  -ngl 99 -ncmoe 48 -t 20 -c 1024 -b 512 -ub 512 --n-gen 32 --pool-fill 0
```

Full protocol (corpus regeneration, tracing, pin emission, preregistrations, per-experiment
commands): `blobpager/article/paper.md` Appendix A.

## Protocol

Every number in the article must exist as an artifact before it enters prose, copied
digit-for-digit; experiments are pre-registered in `blobpager/plan/` with refutation bands
before they run; discrepancies between two measurements are flagged, never silently resolved.
`blobpager/article/lab-log.md` is the append-only session notebook.

## Links

- llama.cpp Discussion #24528 — RFC: MoE expert cache · #23324 — disk-paging PoC
- Fiddler arXiv 2402.07033 · MoE-Infinity arXiv 2401.14361 · Pre-gated MoE arXiv 2308.12066
- Apple LLM-in-a-flash arXiv 2312.11514 · dvmazur/mixtral-offloading

## License

MIT (same as upstream llama.cpp). BlobPager-specific code lives in
`src/llama-blobpager.*`, `examples/blobpager-*`, and `blobpager/`.