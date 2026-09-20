#!/usr/bin/env python3
"""blobpack_manifest.py — emit the blob page table for a MoE GGUF.

Reads the GGUF tensor table and writes manifest.json with, for every
(layer, expert): the byte extents of its gate/up/down runs inside the
source GGUF, plus per-layer frequency pin lists from a Phase-0 trace
if provided. This is the pager's index — the GGUFs we target are
already expert-major (gguf_layout.py verdict), so no repack is needed
for the pilots; a repack path remains for interleaved files later.

Usage:
  python3 blobpack_manifest.py --model <gguf> --out <manifest.json> \
      [--trace blobpager/data/trace-qwen-train.jsonl --pins 96]

Manifest shape:
{
  "model": {...},
  "layers": {
    "0": {
      "tensors": {"ffn_gate_exps": {"offset": N, "expert_bytes": N, "type": "Q4_K"}, ...},
      "expert_count": 128,
      "experts": [
        {"e": 0, "bytes": N, "runs": [{"off": N, "len": N}, ...]}, ...
      ],
      "pins_top96": [ids...]       # if --trace given
    }, ...
  }
}
"""
import argparse
import json
import struct
import sys
from collections import Counter

sys.path.insert(0, __file__.rsplit('/', 1)[0])
import gguf_layout as G


def parse_gguf(path):
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            raise ValueError('not GGUF')
        ver, n_ten, n_kv = struct.unpack('<IQQ', f.read(20))
        kv = {}
        for _ in range(n_kv):
            k = G.rd_str(f)
            (t,) = struct.unpack('<I', f.read(4))
            kv[k] = G.rd_val(f, t)
        tens = []
        for _ in range(n_ten):
            name = G.rd_str(f)
            (nd,) = struct.unpack('<I', f.read(4))
            ne = struct.unpack('<' + 'Q' * nd, f.read(8 * nd))
            (tt,) = struct.unpack('<I', f.read(4))
            (off,) = struct.unpack('<Q', f.read(8))
            tens.append((name, ne, tt, off))
    return kv, tens, ver


def trace_pins(trace_path, pins):
    """Per-layer expert ids ranked by draw frequency (train trace)."""
    streams = {}
    for line in open(trace_path):
        r = json.loads(line)
        if r.get('type') == 'meta':
            continue
        streams.setdefault(r['layer'], []).extend(r['exp'])
    pins_per_layer = {}
    for layer, draws in streams.items():
        c = Counter(draws)
        ranked = [e for e, _ in c.most_common(pins)]
        pins_per_layer[str(layer)] = ranked
    return pins_per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--trace', default=None, help='train trace for pin lists')
    ap.add_argument('--pins', type=int, default=96, help='pin list length')
    a = ap.parse_args()

    kv, tens, ver = parse_gguf(a.model)
    arch = kv.get('general.architecture', '?')
    E = kv.get(f'{arch}.expert_count')
    U = kv.get(f'{arch}.expert_used_count')
    NB = kv.get(f'{arch}.block_count')
    LD = kv.get(f'{arch}.leading_dense_block_count', 0)

    pins = trace_pins(a.trace, a.pins) if a.trace else {}

    layers = {}
    for name, ne, tt, off in tens:
        if 'ffn_gate_exps' not in name and 'ffn_up_exps' not in name and 'ffn_down_exps' not in name:
            continue
        blk = name.split('.')[1]
        role = name.split('.')[-2]           # ffn_gate_exps / ffn_up_exps / ffn_down_exps
        rb = G.row_bytes(tt, ne[0])
        if rb is None:
            raise ValueError(f'unsupported quant type {tt} in {name}')
        expert_bytes = rb * ne[1]            # contiguous per expert (expert-major verified)
        L = layers.setdefault(blk, {'tensors': {}, 'expert_count': ne[2], 'experts': None})
        L['tensors'][role] = {'offset': off, 'expert_bytes': expert_bytes,
                              'type': G.TNAME.get(tt, f't{tt}'), 'ne': list(ne)}

    expert_total = 0
    for blk, L in layers.items():
        E_l = L['expert_count']
        exps = []
        for e in range(E_l):
            runs = []
            tot = 0
            for role, T in sorted(L['tensors'].items()):
                runs.append({'role': role, 'off': T['offset'] + e * T['expert_bytes'],
                             'len': T['expert_bytes']})
                tot += T['expert_bytes']
            exps.append({'e': e, 'bytes': tot, 'runs': runs})
            expert_total += tot
        L['experts'] = exps
        if str(blk) in pins:
            L[f'pins_top{a.pins}'] = pins[str(blk)]

    manifest = {
        'model': {'path': a.model, 'arch': arch, 'gguf_version': ver,
                  'block_count': NB, 'leading_dense': LD, 'expert_count': E,
                  'expert_used': U, 'file_bytes': __import__('os').path.getsize(a.model)},
        'expert_bytes_total': expert_total,
        'pin_len': a.pins if a.trace else None,
        'layers': layers,
    }
    with open(a.out, 'w') as f:
        json.dump(manifest, f)

    n_l = len(layers)
    print(f'manifest -> {a.out}')
    print(f'  arch={arch} layers_with_experts={n_l} expert_count={E} used={U}')
    print(f'  expert bytes total: {expert_total:,} ({expert_total / 2**30:.2f} GiB)')
    if a.trace:
        print(f'  pin lists: top-{a.pins} per layer from {a.trace}')
    # sanity: spot-check layer 0, expert 0 extents
    b0 = layers['0'] if '0' in layers else layers[min(layers)]
    r0 = b0['experts'][0]['runs']
    print('  spot check first expert runs:', [(r['role'], r['off'], r['len']) for r in r0])


if __name__ == '__main__':
    main()