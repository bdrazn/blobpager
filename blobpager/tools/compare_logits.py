#!/usr/bin/env python3
# compare_logits.py — E7b hybrid-vs-baseline logits validation (blobpager).
#
# Reads two --dump-logits binaries (magic "BLG1": uint32 magic, int32 n_vocab,
# then per step: int32 sampled token + n_vocab float32 logits).
#
# Agreement thresholds FIXED BEFORE THE RUN (E7b pre-registration, 2026-09-20):
#   1. sampled tokens must agree at every compared step (hard requirement);
#   2. max |Δ logit| is reported and must be <= 0.05 (soft requirement —
#      GPU-vs-CPU matmul rounding differs; the architecture claim is that
#      every top-8 draw is computed exactly once, not bit-identical kernels).
# Any token mismatch invalidates the run.
import array
import struct
import sys

MAGIC = 0x424C4731
THRESH = 0.05

def load(path):
    d = open(path, 'rb').read()
    magic, n_vocab = struct.unpack_from('<Ii', d, 0)
    if magic != MAGIC:
        raise SystemExit(f'{path}: bad magic {magic:#x}')
    rec = 4 + 4 * n_vocab
    n, rem = divmod(len(d) - 8, rec)
    if rem != 0:
        raise SystemExit(f'{path}: truncated dump (len {len(d)}, rec {rec})')
    toks, lgs = [], []
    off = 8
    for _ in range(n):
        (t,) = struct.unpack_from('<i', d, off)
        off += 4
        a = array.array('f')
        a.frombytes(d[off:off + 4 * n_vocab])
        off += 4 * n_vocab
        toks.append(t)
        lgs.append(a)
    return n_vocab, toks, lgs

def main():
    if len(sys.argv) != 3:
        raise SystemExit('usage: compare_logits.py <baseline.bin> <hybrid.bin>')
    nv_a, tok_a, lg_a = load(sys.argv[1])
    nv_b, tok_b, lg_b = load(sys.argv[2])
    if nv_a != nv_b:
        raise SystemExit(f'vocab mismatch: {nv_a} vs {nv_b}')
    n = min(len(tok_a), len(tok_b))
    if len(tok_a) != len(tok_b):
        print(f'WARN: step count differs: {len(tok_a)} vs {len(tok_b)}; comparing {n}')

    tok_match = all(tok_a[i] == tok_b[i] for i in range(n))
    max_d = 0.0
    max_i, max_j = -1, -1
    argmax_match = 0
    for i in range(n):
        x, y = lg_a[i], lg_b[i]
        if max(x) == max(y) and x.index(max(x)) == y.index(max(y)):
            argmax_match += 1
        for j in range(nv_a):
            d = x[j] - y[j]
            if d < 0:
                d = -d
            if d > max_d:
                max_d, max_i, max_j = d, i, j

    ok = tok_match and max_d <= THRESH
    print(f'steps_compared      : {n}')
    print(f'tokens_match        : {tok_match}')
    print(f'argmax_match        : {argmax_match}/{n}')
    print(f'max_abs_logit_diff  : {max_d:.6f} (at step {max_i}, vocab {max_j})')
    print(f'threshold           : {THRESH}')
    print(f'VERDICT             : {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1

if __name__ == '__main__':
    sys.exit(main())