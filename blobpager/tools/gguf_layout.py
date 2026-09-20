#!/usr/bin/env python3
"""gguf_layout.py — inspect MoE expert-tensor layout in GGUF files.

blobpack design input: are routed-expert tensors expert-major contiguous
(each expert = one contiguous byte run per tensor), or interleaved
(expert dim innermost / scattered across the whole tensor)?

Usage: python3 gguf_layout.py <model.gguf> [more.gguf ...]
"""
import struct
import sys

# ggml type id -> (bytes, vals) per quant block; identity types carry 1 val per "block"
BLOCK = {
    0: (4, 1), 1: (2, 1), 30: (2, 1),  # F32, F16, BF16
    2: (18, 32), 3: (22, 32), 6: (22, 32), 7: (26, 32), 8: (34, 32),
    10: (84, 256), 11: (110, 256), 12: (144, 256), 13: (176, 256),
    14: (210, 256), 15: (260, 256), 20: (132, 256),
}
TNAME = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0',
         10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K',
         16: 'IQ2_XXS', 17: 'IQ2_XS', 20: 'IQ4_NL', 30: 'BF16'}


def rd_str(f):
    (n,) = struct.unpack('<Q', f.read(8))
    return f.read(n).decode('utf-8', 'replace')


def rd_val(f, t):
    if t == 0:
        return struct.unpack('<B', f.read(1))[0]
    if t == 1:
        return struct.unpack('<b', f.read(1))[0]
    if t == 2:
        return struct.unpack('<H', f.read(2))[0]
    if t == 3:
        return struct.unpack('<h', f.read(2))[0]
    if t == 4:
        return struct.unpack('<I', f.read(4))[0]
    if t == 5:
        return struct.unpack('<i', f.read(4))[0]
    if t == 6:
        return struct.unpack('<f', f.read(4))[0]
    if t == 7:
        return bool(f.read(1)[0])
    if t == 8:
        return rd_str(f)
    if t == 10:
        return struct.unpack('<Q', f.read(8))[0]
    if t == 11:
        return struct.unpack('<q', f.read(8))[0]
    if t == 12:
        return struct.unpack('<d', f.read(8))[0]
    if t == 9:
        (at,) = struct.unpack('<I', f.read(4))
        (n,) = struct.unpack('<Q', f.read(8))
        return [rd_val(f, at) for _ in range(n)]
    raise ValueError(f'unsupported gguf kv type {t}')


def row_bytes(tt, ne0):
    b, n = BLOCK.get(tt, (None, None))
    return None if b is None else -(-ne0 // n) * b


def inspect(path):
    print(f'\n=== {path}')
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            print('  not a GGUF file')
            return
        ver, n_ten, n_kv = struct.unpack('<IQQ', f.read(20))
        kv = {}
        for _ in range(n_kv):
            k = rd_str(f)
            (t,) = struct.unpack('<I', f.read(4))
            kv[k] = rd_val(f, t)
        arch = kv.get('general.architecture', '?')
        E = kv.get(f'{arch}.expert_count')
        U = kv.get(f'{arch}.expert_used_count')
        NB = kv.get(f'{arch}.block_count')
        LD = kv.get(f'{arch}.leading_dense_block_count', 0)
        print(f'gguf v{ver}: arch={arch} blocks={NB} experts={E} used={U} leading_dense={LD}')
        tens = []
        for _ in range(n_ten):
            name = rd_str(f)
            (nd,) = struct.unpack('<I', f.read(4))
            ne = struct.unpack('<' + 'Q' * nd, f.read(8 * nd))
            (tt,) = struct.unpack('<I', f.read(4))
            (off,) = struct.unpack('<Q', f.read(8))
            tens.append((name, ne, tt, off))
        pos = f.tell()
        align = kv.get('general.alignment', 32)
        print(f'tensor infos end at {pos:,}; aligned data_start={-(-pos // align) * align:,}')
        exp_t = [t for t in tens
                 if 'ffn_gate_exps' in t[0] or 'ffn_up_exps' in t[0] or 'ffn_down_exps' in t[0]]
        rout = [t for t in tens if 'ffn_gate_inp' in t[0]]
        print(f'expert tensors: {len(exp_t)} (x3 sets = {len(exp_t) / 3:g} blocks), router tensors: {len(rout)}')
        counts = {'EXPERT-MAJOR': 0, 'INTERLEAVED': 0, '?': 0}
        for name, ne, tt, off in exp_t[:6]:
            tn = TNAME.get(tt, f'type{tt}')
            rb = row_bytes(tt, ne[0]) if ne else None
            verdict, eb = '?', None
            if len(ne) == 3:
                if E is not None and ne[2] == E:
                    verdict = 'EXPERT-MAJOR'
                    eb = rb * ne[1] if rb else None
                elif E is not None and ne[0] == E:
                    verdict = 'INTERLEAVED'
            elif len(ne) == 2 and E is not None and ne[1] % E == 0:
                verdict = 'EXPERT-MAJOR'
                eb = rb * (ne[1] // E) if rb else None
            counts[verdict if verdict in counts else '?'] += 1
            print(f'  {name}: ne={list(ne)} {tn} off={off:,} expert_bytes={eb} -> {verdict}')
        g = [t for t in exp_t if 'ffn_gate_exps' in t[0]]
        if g and len(g[0][1]) == 3 and E is not None and g[0][1][2] == E:
            ne, tt = g[0][1], g[0][2]
            rb = row_bytes(tt, ne[0])
            if rb:
                eb = rb * ne[1]
                blob = 3 * eb
                est = eb * E * len(g)
                print(f'  per-expert/layer: {eb:,} B ({eb / 1e6:.2f} MB) | '
                      f'full blob g+u+d: {blob:,} B = {blob // 4096:,} x 4KB pages | '
                      f'est expert total: {est / 1e9:.2f} GB over {len(g)} gate layers')
        if rout:
            name, ne, tt, off = rout[0]
            print(f'  router {name}: ne={list(ne)} {TNAME.get(tt, tt)}')


if __name__ == '__main__':
    for p in sys.argv[1:]:
        inspect(p)