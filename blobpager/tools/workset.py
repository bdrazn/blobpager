#!/usr/bin/env python3
"""blobpager workset analysis: expert-locality hit rates from routing traces.

Reads JSONL traces (one record per (req,pos,layer): {"req","pos","layer","exp":[..]}).
Each MoE layer is an independent expert-access stream; the cache budget is
expressed in slots per layer (S of E experts per layer).

Policies simulated per layer:
  lru        - classic LRU cache, budget S
  static     - top-S experts by frequency over the FULL train corpus (in-sample ceiling)
  cal        - pins from first CAL_FRAC of train requests, evaluated on the rest
  cross      - pins from ALL of train, evaluated on the held-out corpus (pin transfer)
  oracle     - upper bound if S covers every expert the eval slice touches

Writes a JSON summary and prints markdown tables.
"""
import argparse
import json
import math
from collections import Counter, OrderedDict

CAL_FRAC = 0.30


def parse_trace(path):
    """returns (layers dict: layer -> [expert ids in stream order], n_req, n_tokens)"""
    streams = {}
    n_req = 0
    n_tokens = 0
    expert_used = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") == "meta":
                if "expert_used" in r:
                    expert_used = r["expert_used"]
                continue
            layer = r["layer"]
            streams.setdefault(layer, []).extend(r["exp"])
            if layer == min_key(streams):
                n_req = max(n_req, r["req"] + 1)
    return streams, n_req, expert_used


def min_key(d):
    try:
        return min(d)
    except ValueError:
        return -1


def lru_hit_rate(stream, S):
    if S <= 0:
        return 0.0
    distinct = set(stream)
    if S >= len(distinct):
        return 1.0
    cache = OrderedDict()
    hits = 0
    for e in stream:
        if e in cache:
            hits += 1
            cache.move_to_end(e)
        else:
            if len(cache) >= S:
                cache.popitem(last=False)
            cache[e] = True
    return hits / len(stream)


def static_pin_hit_rate(stream, pins):
    if not stream:
        return 0.0
    hits = sum(1 for e in stream if e in pins)
    return hits / len(stream)


def top_pins(counter, P):
    return frozenset(e for e, _ in counter.most_common(P))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="blobpager/data/trace-glm-train.jsonl")
    ap.add_argument("--heldout", default="blobpager/data/trace-glm-heldout.jsonl")
    ap.add_argument("--budgets", default="2,4,6,8,12,16,24,32,48")
    ap.add_argument("--out", default="blobpager/data/workset-summary.json")
    args = ap.parse_args()

    train, n_req_tr, used_tr = parse_trace(args.train)
    hold, n_req_ho, used_ho = parse_trace(args.heldout)
    layers = sorted(train)
    n_layers = len(layers)
    E = len(train[layers[0]] and set())  # placeholder, computed below
    # expert pool size per layer: max expert id + 1 per layer (assumes dense-ish ids)
    pool = max(max(s) + 1 for s in train.values())

    tr_tokens = sum(len(s) for s in train.values())
    ho_tokens = sum(len(s) for s in hold.values())
    tr_keys = sum(len(set(s)) for s in train.values())
    ho_keys = sum(len(set(s)) for s in hold.values())
    tr_new_keys_ho = sum(len(set(s) - set(train[l])) for l, s in hold.items())

    # train split: calibration (first 30% of requests) vs eval (rest)
    # request id ranges: cal = req < cal_cut, eval = req >= cal_cut
    # streams are concatenated per layer; split by token order: find boundary via
    # first occurrence count: total draws from reqs < cal_cut
    cal_cut_req = int(n_req_tr * CAL_FRAC)
    # reconstruct per-layer request boundaries: draws are in req order; we know
    # tokens-per-req only via records, so recount from the trace again:
    per_req_tokens = Counter()
    with open(args.train) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "meta":
                continue
            if r["layer"] == layers[0]:
                per_req_tokens[r["req"]] += len(r["exp"])
    # draws per layer up to cal boundary
    cal_draws = {l: 0 for l in layers}
    tok = 0
    for req in range(cal_cut_req):
        tok += per_req_tokens[req]
    for l in layers:
        cal_draws[l] = tok  # same token count for every layer

    budgets = [int(x) for x in args.budgets.split(",")]

    out = {
        "model_trace": {"train": args.train, "heldout": args.heldout},
        "train": {"requests": n_req_tr, "tokens": tr_tokens // n_layers,
                  "distinct_keys": tr_keys, "pool_per_layer": pool,
                  "layers": n_layers, "expert_used": used_tr},
        "heldout": {"requests": n_req_ho, "tokens": ho_tokens_placeholder() if False else sum(len(s) for s in hold.values()) // n_layers,
                    "distinct_keys": ho_keys, "new_keys_vs_train": tr_new_keys_ho,
                    "expert_used": used_ho},
        "lru": {}, "static": {}, "cal": {}, "cross": {}, "concentration": {},
        "per_layer_distinct": {},
    }

    # concentration curve: top-j experts cover what fraction of all draws (per layer, averaged)
    for j in (4, 8, 12, 16, 24, 32, 48, 64, 96, 128):
        covs = []
        for l in layers:
            c = Counter(train[l])
            covs.append(sum(c for _, c in c.most_common(j)) / len(train[l]))
        out["concentration"][f"top{j}"] = sum(covs) / n_layers

    for l in layers:
        out["per_layer_distinct"][l] = len(set(train[l]))

    pd = [out["per_layer_distinct"][l] for l in layers]
    out["per_layer_distinct_stats"] = {
        "min": min(pd), "median": sorted(pd)[len(pd) // 2], "max": max(pd),
    }

    for S in budgets:
        # in-sample static pins (train -> train)
        hits = 0.0
        for l in layers:
            c = Counter(train[l])
            pins = top_pins(c, S)
            hits += sum(1 for e in train[l] if e in pins) / len(train[l])
        out["static"][S] = round(hits / n_layers, 4)

        # calibrated pins (train-cal -> train-eval)
        hits = 0.0
        for l in layers:
            s = train[l]
            cal_counter = Counter(s[: cal_draws[l]])
            pins = top_pins(cal_counter, S)
            ev = s[cal_draws[l]:]
            hits += sum(1 for e in ev if e in pins) / len(ev)
        out["cal"][S] = round(hits / n_layers, 4)

        # cross-corpus pins (train -> heldout)
        hits = 0.0
        for l in layers:
            c = Counter(train[l])
            pins = top_pins(c, S)
            hits += sum(1 for e in hold[l] if e in pins) / len(hold[l])
        out["cross"][S] = round(hits / n_layers, 4)

        # LRU (train)
        hits = sum(lru_hit_rate(train[l], S) for l in layers)
        out["lru"][S] = round(hits / n_layers, 4)

    # LRU on heldout (cold cache) for the largest sensible budgets
    out["lru_heldout"] = {}
    for S in budgets:
        hits = sum(lru_hit_rate(hold[l], S) for l in layers)
        out["lru_heldout"][S] = round(hits / n_layers, 4)

    # oracle per budget on heldout: fraction of layers where distinct <= S
    out["oracle_heldout"] = {}
    for S in budgets:
        ok = sum(1 for l in layers if len(set(hold[l])) <= S)
        out["oracle_heldout"][S] = round(ok / n_layers, 4)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # ---- report ----
    print(f"== structural ==")
    print(f"pool/layer={pool} layers={n_layers} expert_used={used_tr}")
    print(f"train: {n_req_tr} reqs, {tr_tokens // n_layers} tokens, {tr_keys} distinct keys")
    print(f"heldout: {n_req_ho} reqs, {sum(len(s) for s in hold.values()) // n_layers} tokens, "
          f"{ho_keys} distinct keys, {tr_new_keys_ho} keys not in train "
          f"({100.0 * tr_new_keys_ho / max(1, ho_keys):.1f}% of heldout keys)")
    print(f"per-layer distinct (train): min {out['per_layer_distinct_stats']['min']} "
          f"median {out['per_layer_distinct_stats']['median']} max {out['per_layer_distinct_stats']['max']} of pool {pool}")
    print()
    print("== concentration (share of draws covered by top-j experts per layer, train) ==")
    for j in (4, 8, 12, 16, 24, 32, 48, 64, 96, 128):
        k = f"top{j}"
        if k in out["concentration"]:
            print(f"  {k:>6}: {100 * out['concentration'][k]:5.1f}%")
    print()
    print("== hit rates (share of expert draws served from the VRAM cache) ==")
    hdr = "S/layer |  " + " | ".join(f"{b:>4}" for b in budgets)
    print(f"policy  |  budget (slots per layer)  ->  hit %")
    print(hdr)
    for name in ("lru", "cal", "cross", "static"):
        row = " | ".join(f"{100 * out[name][b]:4.1f}" for b in budgets)
        print(f"{name:>7} |  {row}")
    print()
    print("== heldout (pin transfer + cold LRU) ==")
    for name in ("lru_heldout", "cross", "oracle_heldout"):
        row = " | ".join(f"{100 * out[name][b]:4.1f}" for b in budgets)
        print(f"{name:>14} |  {row}")
    print()
    print(f"saved -> {args.out}")


# tiny helper used before definition order settles
def ho_tokens_placeholder():
    return 0


if __name__ == "__main__":
    main()