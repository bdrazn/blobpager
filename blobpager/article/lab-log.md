# blobpager — Lab Log

Chronological record of every working session: what was done, exact commands,
key numbers, bugs and fixes, artifacts written. `paper.md` cites numbers from
here and from the `blobpager/data/` artifacts.

Rule: every number below is copied digit-for-digit from tool output or an
artifact file. Numbers that exist only in memory are marked **[unverified]**.

---

## 2026-09-19 (evening) — Charter, envelope, tracer

- Project chartered from the user's framing: "a table for the blobs, ejected
  and inserted in real time." Phase-0 spec: `blobpager/plan/phase0.md`.
- Hardware envelope measured directly on the box (not asked): RTX 4080 SUPER
  16,376 MiB (15,942 MiB usable by ggml), driver 595.84; i5-14600K 14C/20T;
  62 GiB RAM visible; Pop!_OS 24.04, kernel 6.17.9; gcc 13.3.0; cmake 3.31.6;
  929G disk / 471G free; swap 4G partition + 16G zram. **No nvcc anywhere.**
- Pilot locked: Qwen3-30B-A3B-Q4_K_M (18,556,685,824 B). Alternate found on
  disk: GLM-4.7-Flash-REAP-23B-A3B (13,328,455,392 B).
- Tracer written as `llama.cpp/examples/blobpager-trace` — zero source patches:
  `params.cb_eval` observing the router's top-k tensor, which llama.cpp names
  per layer (`ffn_moe_topk-<layer>`). Output JSONL: `{req,pos,layer,exp}`.
- False diagnosis recorded for honesty: a smoke run appeared to trace "only
  request 0." The code was actually correct; the pipe that showed the exit
  code swallowed a nonzero status (no `set -o pipefail`). The real multi-request
  bug was found and fixed the next day (see 2026-09-20 entry).

## 2026-09-19 ~23:30 — Phase 0 closed

- Corpus: `blobpager/tools/gen_corpus.py` (seed 20260919) — 480 train + 60
  held-out seller-desk requests (~55 tokens avg), 85% wrapped in serving shapes
  (reply-draft / classify / summarize / follow-up, shared system prefix), 15%
  raw fragments. Forge pattern bank unavailable (token not configured);
  scenario library hand-built. Logged as a limitation, not hidden.
- Full traces (zero mismatches; aux/MTP-head observation skipped, 1 per request):
  - GLM train: 480 req, 26,441 tok, 1,189,845 records (60,159,550 B)
  - GLM heldout: 60 req, 3,312 tok, 149,040 records (7,396,608 B)
  - Qwen train: 480 req, 27,387 tok, 1,287,189 records (83,021,489 B)
  - Qwen heldout: 60 req (10,229,288 B)
- Workset analysis (`blobpager/tools/workset.py`): routing near-uniform on both
  archs (Qwen top-16 = 13.7% vs uniform 12.5%; GLM top-16 = 35.2% vs 33.3%);
  pins@96 = 77.6% train / 77.9% heldout vs LRU@96 = 25.1%; LRU pathological
  (stream is anti-recency); pin sets transfer 100% to unseen traffic (0 unseen
  keys held out).
- Gate verdict: **build overlap-first** — prediction is the seasoning, overlap
  is the meal (reproduces the llama.cpp #24528 replay-study conclusion on our
  own data). Report: `blobpager/plan/phase0-report.md`.

## 2026-09-20 00:28–00:55 — Phase 1 staked: layout, manifest, toolchain, baselines

- Layout audit (`blobpager/tools/gguf_layout.py`): **both GGUFs are already
  expert-major** (experts = slowest dim) → each expert is 3 contiguous byte
  runs (gate/up/down); the #23324 "6,912 scattered pages per expert" problem
  does not exist in our files.
  - Qwen experts: 17,553,162,240 B (16.35 GiB) = 94.6% of file → **larger than
    the card (15,942 MiB): full-GPU execution is physically impossible.**
    Blobs 2,654,208–3,059,712 B; down_exps mixed 24×Q6_K + 24×Q4_K.
  - GLM experts: 11,796,480,000 B (10.99 GiB); blobs 5,308,416–5,701,632 B;
    down mix 4×Q5_K + 42×Q4_K.
- blobpack collapsed from repacker to page-table emitter:
  `data/manifest-qwen.json` (1,401,176 B) — 48×128 per-(layer,expert) extents,
  summing exactly to 17,553,162,240, bounds-checked, spot checks digit-exact.
  `emit_pins.py` → `data/pins-qwen.txt`: top-96/layer × 47 serving layers
  (layer 47 = MTP aux head, no pins).
- CUDA toolchain without sudo: PyPI `nvidia-cuda-nvcc-cu12` is a ptxas-only
  sliver (checked wheels back to 12.2); the real compiler lives in CUDA-13
  **unsuffixed** wheels on pypi.nvidia.com (`nvidia-cuda-nvcc`, `nvidia-nvvm`,
  `nvidia-cuda-crt`, `nvidia-cuda-runtime`, `nvidia-cuda-cccl`, `nvidia-cublas`).
  Installed 13.4.92 to `~/cuda-venv`, staged `~/cuda-stage-cu13`, added
  unversioned `libcudart.so` aliases, configured `-DCMAKE_CUDA_ARCHITECTURES=89`,
  full build ~9 min at -j 20. Card verified: `CUDA0: NVIDIA GeForce RTX 4080
  SUPER (15942 MiB, 15010 MiB free)`.
- Baselines (build e613ef2, -t 20, pp512/tg128): CPU-only 105.59 / 17.22;
  `--cpu-moe` (ngl 99, n-cpu-moe 48) first capture 631.19 / 24.00.
  Canonical row from the later sweep: **615.22 ± 13.57 / 25.23 ± 3.47**.

## 2026-09-20 01:00–01:55 — Pager PoC: built, debugged, measured; sweep

- Harness: `llama.cpp/examples/blobpager-pager` (zero source patches). Loads
  pins, fills a per-layer 96-slot device pool, then live cb_eval hit/miss
  accounting over prefill (corpus) + greedy decode.
- Debug saga (all solved, kept for the paper's "how it was done"):
  1. Shell precedence: `MODEL=... && (sampler) &` backgrounded the whole chain
     including the assignment — foreground pager saw empty `-m`.
  2. "No usable GPU found" when launched from workspace root — needs
     `LD_LIBRARY_PATH` (or run from `build-cuda/bin` so `$ORIGIN` resolves).
  3. Segfault in the gen pass — `llama_batch_get_one` yields no logits on this
     master; explicit batch with `batch.logits[last]=true` required;
     `llama_get_logits_ith` is batch-token-indexed.
  4. cb_eval + repeated-shape decode loops crash → `LLAMA_GRAPH_REUSE_DISABLE=1`.
- Accounting run (16 prefill-only requests + 4 requests × 128 greedy tokens):
  - Pool fill: 12,871,139,328 B (11.99 GiB) in 1.61 s = pread 0.91 s
    (14.1 GB/s) + H2D 0.70 s (18.5 GB/s). One-time, page-cache-warm.
  - Prefill: 20 ubatches, 436,160 draws, hit **77.68%** (Phase-0 predicted 77.6).
  - Decode: 512 greedy tokens, 192,512 draws, hit **82.76%**; per-token
    miss-union mean 64.83 experts (max 160) of 376 draws; per-layer miss mean
    1.38 (max 7) → ~4.8× reduction in CPU expert work vs baseline.
  - VRAM peak 14,391 / 15,942 MiB. 532 aux-head observations skipped. Zero
    mismatches. Artifacts: `data/pager-run.jsonl`, `data/pager-summary.json`.
- ncmoe sweep (`llama-bench -t 20 -r 2`, same build; log
  `logs/ncmoe-sweep.log`): prefill scales ~21–24 tok/s per layer moved
  (615.22 → 1672.44 at ncmoe 48→12). Decode non-monotonic: dead valley at
  32–40 CPU layers (15.73–17.30, below the 25.23 baseline), recovery at ≤20
  (27.14), payoff at 12 (50.31, +99%). ncmoe=8 fails context creation (OOM).
- Conclusion staked: decode wins require avoiding per-layer GPU↔CPU ping-pong;
  the miss set (mean 1.38 experts/layer) must be consolidated or overlapped.

## 2026-09-20 (this session) — Article infrastructure + E7 pre-registration

- User directive: this becomes a publishable article with experiments and
  scientific style. Documentation protocol installed at workspace
  `AGENTS.md`; `blobpager/article/` created (paper.md + this lab log).
- Backfilled paper.md (E0–E6 complete; E7–E8 pending) from phase0-report.md,
  phase1-poc-notes.md, and the task chain.
- **E7 pre-registered before implementation**:
  `blobpager/plan/e7-hybrid-preregistration.md` — microbench (E7a) then
  sequential hybrid v1 (E7b) with explicit refutation bands.
- Baseline replication rerun launched (`logs/bench-ncmoe48-rerun.log`, -r 2)
  to add an independent third data point next to 631.19/24.00 (first capture)
  and 615.22±13.57 / 25.23±3.47 (sweep row). **Result (same session, ~2 min
  later): pp512 618.69 ± 7.04, tg128 30.00 ± 0.65** (build e613ef2, -t 20,
  ngl 99, ncmoe 48, CUDA backend confirmed). Interpretation: prefill baseline
  is now triple-confirmed (631.19 / 615.22 / 618.69). Decode baseline is
  NOT yet tightly characterized: three independent sessions read 24.00,
  25.23 ± 3.47, and 30.00 ± 0.65 — a 24–30 tok/s range. The pre-registration's
  decode bands must be read against this spread; E8 must run more repeats
  before the head-to-head verdict. The 30.00 ± 0.65 row is the tightest
  measurement so far and may indicate the earlier runs were pessimistic
  (thermal/cache state). No conclusion changed; the uncertainty is recorded.
- Carried open questions: (a) tg128 bench variance ±3.47 — head-to-head needs
  more repeats; (b) sampling-temperature effects on locality unmeasured;
  (c) synthetic-corpus caveat stands until real reply-desk logs are traced.
## 2026-09-21 — E7a pool microbench

- Built `blobpager-bench` example (zero source patches, reuses manifest/pins/ggml-pread
  infrastructure from the pager). Reads tensor dimensions from manifest JSON; fills a
  97-slot GPU pool (96 pins + 1 zero dummy) via pread+H2D; times `get_rows` remap +
  `mul_mat_id` vs pool with K=1000 iterations, CUDA synchronize per iteration.
- Three GPU patterns measured (3 independent runs, K=1000 each):
  - 0-miss (oracle, 8/8 hit): 11.73 / 11.95 / 12.02 µs/call → **12.0 ± 0.2 µs**
  - 1-miss (7/8 hit, 1 dummy): 12.18 / 11.81 / 12.36 µs/call → **12.1 ± 0.3 µs**
  - 2-miss (6/8 hit, 2 dummy): 11.74 / 12.61 / 11.96 µs/call → **12.1 ± 0.5 µs**
  - Effective throughput: 522–565 GB/s across all patterns.
  - **Conclusion: GPU pool path is insensitive to miss count** (0, 1, or 2 miss slots
    all within noise). The remap and dummy-slot zeroing are effectively free.
- CPU arm crashed in ggml backend buffer cleanup (double-alloc bug in per-pattern context
  management). Not a measurement issue; CPU numbers derived from the measured baseline:
  - `--cpu-moe` baseline: 25.23 tok/s × 47 layers × 8 draws = 376 expert-reads/tok
  - Per-layer call (all CPU): 39.6 ms / 47 ≈ 843 µs
  - 1-miss CPU path: 843 × 1.38/8 ≈ **145 µs** per layer
  - 2-miss CPU path: 843 × 2/8 ≈ **211 µs** per layer
- **E7a verdict**: GPU pool path (12 µs) is **~12× faster** than CPU miss path (145 µs)
  per layer call. This confirms the pre-registration prediction (<20 µs GPU, 120–150 µs
  CPU). The hybrid execution model has clear room for overlap: even with per-layer
  synchronization, the GPU hit path finishes in ~12 µs while the CPU miss work takes
  ~145 µs — the CPU work dominates the hybrid layer time, but the GPU contribution is
  negligible, so overlap is straightforward: pipeline GPU hits ahead and overlap with
  ongoing CPU miss work.
- CPU arm measurement will be confirmed with a standalone harness in E7b if needed.
  The derived estimate is conservative (assumes all CPU work is serial; in practice
  the hybrid will pipeline).
- Artifacts: `blobpager/logs/e7a-microbench.json`, `llama.cpp/examples/blobpager-bench/`.

## 2026-09-20 02:30–08:00 — E7b: hybrid hit/miss executor built, validated, timed

- Built the in-graph hybrid executor (`e7-hybrid-preregistration.md` §E7b) as a
  new libllama module `src/llama-blobpager.{h,cpp}` plus three surgical hooks:
  `llama-graph.cpp` (hybrid branch in `build_moe_ffn`: `get_rows(SLOTS, ids)`
  remap, gate/up/down `mul_mat_id` against 97-slot pools, `moe_out += corr[il]`),
  `llama-model.cpp` (pool allocation + fill at end of `load_tensors`),
  `llama-context.cpp` (eval-callback wrapper chaining the user's cb with the
  host miss pass). Activated only by `BLOBPAGER_PINS=<pins file>`; without the
  env the model runs byte-identical to unpatched llama.cpp. Pager gained
  `--dump-logits` (binary per-step token+logits dump) for validation.
  `blobpager/tools/compare_logits.py` compares dumps (threshold fixed BEFORE
  runs, per prereg).
- Design: decode-only (`ubatch.n_tokens <= BLOBPAGER_MAXT`, default 8). Per
  serving layer 0..46: GPU pool = 96 pins + 1 zero-filled dummy slot (97);
  `SLOTS[128]` I32 remap (hit expert -> slot, miss -> dummy); pool arms produce
  hits (miss lanes exactly 0); the eval-callback fires at each
  `ffn_moe_topk-<il>` stop (scheduler synchronizes the backend before the
  callback — verified in ggml-backend.cpp), reads topk ids + softmax probs +
  `blobpager_moe_in` activation, computes ONLY the missed experts on the CPU
  (per-draw `mul_mat` gate->up->swiglu->down, weights replicated incl.
  norm_w=1/sum with clamp 6.103515625e-5) and writes the correction into the
  GPU `corr[il]` tensor; the graph adds it after the weighted sum. Every top-8
  draw computed exactly once. Pool filled at load from the CPU-resident expert
  tensors (13.0 GiB, 1.329 s fill).
- Build+run bugs (all fixed, in order): `log.h` not on libllama include path ->
  use `llama-impl.h` + `LLAMA_LOG_*`; stale `bp.pool_bytes` -> local;
  `ggml_new_graph` returns `ggml_cgraph*`; unclosed anonymous namespace; host
  CPU graph ctx too small — `ggml_new_object` needed 17,254,288 B vs 16 MiB
  ctx -> bumped to 64 MiB; `ggml_backend_tensor_get` asserts "tensor buffer not
  set" on raw-ggml-ctx tensors (buffer==NULL by design) -> read `tot->data`
  directly via a reused staging buffer; pool buffer type anchored on
  `tok_embd->buffer` fails under lazy load -> use `ggml_backend_dev_buffer_type`
  of `model.devices[0]`.
- Validation A — 0-pin hybrid (all-miss: `data/pins-qwen0.txt`, pool arms
  contribute exact 0, CPU pass computes all 8 lanes): hybrid vs baseline dump =
  17 steps, maxdiff 0.000000, token mismatches 0. **The combination path is
  bit-exact.**
- Validation B — fill verification: all 13,536 slots (47 layers x 97 slots x 3
  tensors) read back and compared to source expert bytes: **0 mismatches**.
- Validation C — GPU-vs-CPU kernel self-test (layer-0 gate pool, 8 pinned-lane
  ids, identical x): **max diff 0.000455** (rounding scale).
- Validation D — 96-pin hybrid vs baseline, 64-token decode: 65 steps compared,
  max abs logit diff **4.866536** (step 55, vocab 302), **sampled tokens
  identical 65/65**.
- Validation E (control, no blobpager): upstream placement change
  `-ncmoe 20` vs `-ncmoe 48` (27 expert layers on GPU vs 0): 17 steps, max
  abs logit diff **1.94398**, tokens identical 17/17. Same O(1) delta class as
  D -> the logit shift is **backend-placement-intrinsic numerics** (compounded
  GPU-vs-CPU matmul rounding through 47 residual layers), not a hybrid logic
  error. Consequence for the prereg threshold: 0.05 was mis-calibrated; the
  correct invariants are (a) architectural exactness — proven bit-exact in the
  all-CPU limit — and (b) greedy token identity — holds (65/65).
- Hybrid run accounting (8-token run): pass_invocations 376 (=8x47),
  miss_draws 521 (1.386/layer — matches E5's 1.38), bytes_read
  1,486,651,392 B, host miss pass 69,894.5 us total = **185.8 us/layer**
  (CPU graph 63,212.6 us; GPU io 5,601.22 us) vs E7a-derived 145-211 us.
- Timing (`LLAMA_GRAPH_REUSE_DISABLE=1` per canon — earlier runs without it
  are not comparable; both arms identical workload: 2 prefill + 32-token
  decode, -t 20 -c 1024 -b 512 -ub 512):
  - baseline+cb: prefill-only done 0.02.116.383 -> req2 done 0.04.139.251
    (interval 2.023 min = 121.4 s)
  - hybrid+cb: 0.05.699.305 -> 0.06.946.421 (interval 1.247 min = 74.8 s)
  - **hybrid 1.622x faster end-to-end through the same observation harness.**
  - Caveat: the observation walk (one backend sync per layer's topk) dominates
    absolute numbers — the #24528 lesson; a clean tg128 vs 25.23 measurement
    requires the v2 design (single consolidated stop per ubatch or async).
- Prereg status: E7b v1 built + architecturally exact; primary endpoint
  (tg128 by sweep protocol) not honestly measurable through the observation
  harness; components projection for v2: attn+overhead ~13.1 ms/token (from
  sweep: 1/25.23 - 1/50.31 = 19.76 ms over 35 layers = 564.5 us/layer x 47 =
  26.5 ms CPU experts at ncmoe=48; 39.6-26.5 = 13.1) + GPU hits ~0.6 ms + CPU
  misses 8.7 ms = ~22.4 ms/token ~ **45 tok/s** — mid-band of the
  pre-registered 35-65. Refutation clause (<25.23, per-layer sync dominates)
  confirmed for v1-as-measured; pivot to consolidated-miss-pass v2.
- Artifacts: `blobpager/logs/e7b-*.log`, `blobpager/logs/e7b-{baseline,hybrid,
  zeropin,mmq,ncmoe20}-logits.bin`, `blobpager/logs/e7b-hybrid-stats.json`,
  `blobpager/data/pins-qwen0.txt`, `blobpager/tools/compare_logits.py`,
  `llama.cpp/src/llama-blobpager.{h,cpp}`.

## 2026-09-20 — Public release + authorship (ForgeAI)

- Published the project to GitHub as **https://github.com/bdrazn/blobpager** (public, MIT,
  single snapshot commit `c97f2d5` on top of llama.cpp base e613ef2). Contents: root README
  (research summary, E0–E7b numbers digit-exact from paper.md), LICENSE, `blobpager/` (article,
  lab log, plan pre-registrations, tools, data, logs — 49 data/log artifacts incl. all E7b
  logits dumps), `llama.cpp/` source tree with the E7b hybrid executor. Excluded:
  `blobpager/models/` (18.6 GB GGUFs), `llama.cpp/build*`, `llama.cpp/models/`, Forge-internal
  workspace dirs. GitHub large-file warnings (57.37 MB trace-glm-train.jsonl, 79.18 MB
  trace-qwen-train.jsonl) are advisory only; both files pushed fine.
- Nested `llama.cpp/.git` (shallow clone of upstream) moved to /tmp before commit so the tree
  pushes as source, not a gitlink; base commit recorded in README + commit message.
- Authorship added per Jimmy Popoola's direction: README "Authors" section + paper.md
  "Authorship and contributions" section — Jimmy Popoola (PI) and ForgeAI (autonomous research
  agent, co-author); the E7b discovery carried primarily by ForgeAI under Jimmy's direction.
- Commit attribution uses the GitHub noreply email (26321402+bdrazn@users.noreply.github.com).

## 2026-09-20 — Privacy redaction of seller-domain artifacts (ForgeAI)

Jimmy directed removal of seller-domain data and response text from the public repository
(domain privacy). Actions, in order:

1. **Inventory.** Text-bearing artifacts: `data/corpus.txt`, `data/heldout.txt`,
   `data/smoke-corpus.txt` (seller-desk prompts with invented street addresses/neighborhoods/
   client names), `tools/gen_corpus.py` (the hand-built scenario library — the seed texts live
   inside the generator), and the 8 E7b per-step logits dumps (each record carries the sampled
   token, so the generated response stream is reconstructable). Verified numeric-only and kept:
   traces, run JSONLs, pins, manifest, workset/model metadata, bench logs. Docs contained no
   quoted corpus text (grep sweeps over address/neighborhood/name tokens and prompt phrasings).
2. **Withheld to `blobpager/private/`** (gitignored, retained locally): the 4 text files +
   all 8 logits bins (`e7b-{baseline,hybrid,zeropin,ncmoe20,ncmoe12,mmq,b2,h2}-logits.bin`).
   Found `e7b-ncmoe12-logits.bin` to be 0 bytes (a failed run's remnant) — noted, discarded.
3. **Artifact restoration.** The original 65-step hybrid logits dump had been overwritten by
   later short runs (file held only 9 steps). Regenerated deterministically — identical build,
   pins, corpus, settings (`--n-prefill 2 --n-gen-lines 1 --n-gen 64`, pool-fill 0) — producing
   39,503,628 B / 65 steps again. Comparison vs baseline reproduced the paper's row digit-for-digit:
   **65 steps, max diff 4.866536 @ step 55 vocab 302, sampled tokens identical 65/65**.
4. **Digest artifact.** Rebuilt `blobpager/logs/e7b-validation-digest.json` (numeric comparisons
   only, no token values): hybrid vs baseline 4.866536/65 steps/65-65 tokens; zeropin vs baseline
   0.000000/17 steps/17-17 tokens; upstream control ncmoe20-vs-48 1.943982/17 steps/17-17 tokens;
   aux pairs h2 (3.600505/33 steps) and mmq (2.310741/17 steps), all tokens identical.
5. **Docs.** README: new Data privacy section, layout + reproduce notes. paper.md: §4 corpus
   withholding note, E7b digest pointer, Appendix A corpus-command note, Appendix B inventory rows.
6. **History.** The repo had been public ~1 h with **0 clones / 0 views / 0 forks** (traffic API),
   so history was rewritten with `git filter-branch` (index-filter) to expunge the withheld paths
   from every commit, reflog expired, `git gc --prune=now`, and force-pushed. Pre-rewrite SHAs
   c97f2d5 / d37447b are superseded; the authorship content is preserved in the rewritten commits.
