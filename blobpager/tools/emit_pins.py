#!/usr/bin/env python3
"""Emit blobpager/data/pins-qwen.txt from manifest-qwen.json.

C++-friendly whitespace format (token stream, no JSON needed):

  LAYERS <n_layers_with_pins>
  LAYER <l> <npins> <pool_bytes>       # pool_bytes = full copy size for this layer
  TENSOR <name> <base_offset> <expert_bytes>   # x3 per layer (gate/up/down)
  PINS <id0> <id1> ...                 # npins ids, order = pool slot order

Only layers carrying pins_top96 are emitted (Qwen serving layers 0..46;
the MTP aux block 47 has no pins and never routes in serving).
"""
import json
import sys

mpath = sys.argv[1] if len(sys.argv) > 1 else 'blobpager/data/manifest-qwen.json'
opath = sys.argv[2] if len(sys.argv) > 2 else 'blobpager/data/pins-qwen.txt'

m = json.load(open(mpath))
L = m['layers']

entries = []
for lk in sorted(L.keys(), key=int):
    lv = L[lk]
    pins = lv.get('pins_top96')
    if not pins:
        continue
    assert len(pins) == m['pin_len'], f"layer {lk}: {len(pins)} pins != {m['pin_len']}"
    entries.append((int(lk), lv, pins))

out = [f"LAYERS {len(entries)}"]
total = 0
for l, lv, pins in entries:
    tb = lv['tensors']
    nbytes = sum(tv['expert_bytes'] for tv in tb.values()) * len(pins)
    total += nbytes
    out.append(f"LAYER {l} {len(pins)} {nbytes}")
    for tname, tv in tb.items():
        out.append(f"TENSOR {tname} {tv['offset']} {tv['expert_bytes']}")
    out.append("PINS " + " ".join(str(p) for p in pins))

open(opath, 'w').write("\n".join(out) + "\n")
print(f"wrote {opath}: layers={len(entries)} total_pool_bytes={total} ({total / 2**30:.3f} GiB)")