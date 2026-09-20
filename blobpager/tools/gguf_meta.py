#!/usr/bin/env python3
"""Dump GGUF metadata (no external deps) + compute MoE expert-pool accounting.

Usage: python3 gguf_meta.py <model.gguf> [--json out.json]
Parses the GGUF header directly: all scalar/string KVs printed or saved;
arrays are skipped efficiently (only their type+count is read).
"""
import json
import struct
import sys


def _r(fmt, f):
    sz = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(sz))[0]


def _r_string(f):
    n = _r("<Q", f)
    return f.read(n).decode("utf-8", errors="replace")


def _skip_value(f, vtype):
    if vtype == 8:  # string
        _r_string(f)
    elif vtype == 9:  # array
        et = _r("<I", f)
        n = _r("<Q", f)
        for _ in range(n):
            _skip_value(f, et)
    elif vtype in (0, 7):
        f.read(1)
    elif vtype in (1,):
        f.read(1)
    elif vtype in (2,):
        f.read(2)
    elif vtype in (3,):
        f.read(2)
    elif vtype in (4, 6):
        f.read(4)
    elif vtype in (5,):
        f.read(4)
    elif vtype == 10:
        f.read(8)
    elif vtype == 11:
        f.read(8)
    elif vtype == 12:
        f.read(8)
    else:
        raise ValueError(f"unknown gguf vtype {vtype}")


def _read_value(f, vtype):
    if vtype == 8:
        return _r_string(f)
    if vtype == 9:
        et = _r("<I", f)
        n = _r("<Q", f)
        if et in (8, 9):  # array of strings/arrays: skip, return shape
            for _ in range(n):
                _skip_value(f, et)
            return f"<array[{n}]>"
        vals = []
        fmtmap = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
        fmt = fmtmap.get(et)
        if fmt is None:
            raise ValueError(f"bad array elem type {et}")
        for _ in range(n):
            vals.append(_r(fmt, f))
        return vals[0] if len(vals) == 1 else vals
    fmtmap = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    if vtype in fmtmap:
        return _r(fmtmap[vtype], f)
    raise ValueError(f"unknown gguf vtype {vtype}")


def main():
    path = sys.argv[1]
    out_json = None
    if "--json" in sys.argv:
        out_json = sys.argv[sys.argv.index("--json") + 1]

    meta = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", "not a GGUF file"
        version = _r("<I", f)
        n_tensors = _r("<Q", f)
        n_kv = _r("<Q", f)
        for _ in range(n_kv):
            key = _r_string(f)
            vtype = _r("<I", f)
            try:
                meta[key] = _read_value(f, vtype)
            except (ValueError, MemoryError):
                _skip_value(f, vtype)
                meta[key] = "<unparsed>"

        # tensor infos (names + dims only — enough for expert-pool accounting)
        tensors = []
        for _ in range(n_tensors):
            name = _r_string(f)
            nd = _r("<I", f)
            dims = [_r("<Q", f) for _ in range(nd)]
            ttype = _r("<I", f)
            _r("<Q", f)  # offset
            tensors.append((name, dims, ttype))

    arch = str(meta.get("general.architecture", "?"))
    sel = {}
    for k, v in meta.items():
        if any(s in k for s in (
                "arch", "block_count", "expert_count", "expert_used", "leading_dense",
                "embedding_length", "moe_intermediate", "expert_feed_forward",
                "attention.head_count", "key_length", "value_length", "context_length",
                "general.name", "general.file_type", "mtp", "nextn", "quantize")):
            if v != "<unparsed>":
                sel[k] = v

    # expert tensor accounting: count distinct expert-weight tensor groups
    n_exp_tensor_bytes_est = None
    expert_tensor_count = sum(1 for n, _, _ in tensors if "ffn" in n and ("exps" in n or "expert" in n))
    total_tensors = n_tensors

    print(f"file: {path}")
    print(f"gguf v{version}, {n_tensors} tensors, {n_kv} kv pairs")
    for k in sorted(sel):
        print(f"  {k} = {sel[k]}")
    print(f"expert-ish tensors: {expert_tensor_count} of {total_tensors}")

    if out_json:
        with open(out_json, "w") as f:
            json.dump({"meta": sel, "arch": arch, "n_tensors": n_tensors,
                       "expert_tensors": expert_tensor_count}, f, indent=2, default=str)
        print(f"saved -> {out_json}")


if __name__ == "__main__":
    main()