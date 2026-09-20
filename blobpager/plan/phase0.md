# Phase 0 spec — trace & measure

## The question

Does our real-estate seller-desk traffic concentrate on a small expert working set — enough to
justify static pins plus a slot-pool pager?

## Pilot model

- Default candidate: **Qwen3-30B-A3B** (128 experts/layer, top-8, ~3.3B active) — the exact model
  the #23324 disk-paging PoC validated. Q4_K_M is ~18 GB on disk; the PoC used Q6_K (~25 GB).
- Alternate if VRAM/RAM is tight: **gpt-oss-20b** (32 experts/layer, top-4, native MXFP4,
  16 GB-class) — much smaller expert count, so locality structure differs; useful contrast.
- Final pick waits on hardware specs.

## Trace format

JSONL, one record per (token, layer) expert selection:

```json
{"req":"r0007","pos":123,"layer":14,"expert":87}
```

Streaming append, buffered writes, no locking on the hot path. Volume estimate: ~48 layers ×
top-8 × ~500 tokens × 300 requests ≈ 57M records worst case — a few GB. Fine on NVMe.

## Patch point

llama.cpp graph builder where router logits are reduced to expert ids (build_moe_ffn /
llm_build_moe_ffn in `llama-graph.cpp` on current master — confirm exact names against our
checkout before patching). Gate by env var `BLOBPAGER_TRACE=<path>`; unset = zero overhead.

Fallback: the #23324 PoC branch already intercepts expert ids — reuse its interceptor if our
patch fights upstream churn.

## Corpus

~200–500 seller-desk requests × up to ~500 generated tokens. Sources: Forge real-estate pattern
bank (buyer/seller response patterns) plus real reply-desk logs if provided. Keep a held-out
slice to test whether pins transfer to unseen traffic.

## Analysis (workset.py)

- distinct (layer, expert) keys; per-token and per-conversation working sets
- hit-rate vs cache-size curves: LRU, LFU, LFRU, calibrated pins (10-token calibration pass), oracle
- prefetch lookahead accuracy: expert overlap between layer l router output and layer l+d needs
- baseline to beat: #24528 replay — 62,400 accesses, 5,083 distinct keys, LRU 5.8% worse than
  shipped LFRU, perfect predictor ≤ +12.8% I/O

## Decision gate

- Pin hit rate ≥ ~40% at feasible cache budget → Phase 1 static pinning first.
- Locality weak but temporally stable → Phase 2 dynamic-first with aggressive prefetch.
- Neither → domain isn't special; overlap-first engineering is the whole game (still valuable,
  different emphasis).