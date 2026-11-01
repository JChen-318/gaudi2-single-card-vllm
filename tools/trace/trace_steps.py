"""Step sequence and per-recipe device totals: python3 trace_steps.py <trace>"""
import collections, gzip, json, re, sys
ev = json.load(gzip.open(sys.argv[1], "rt"))["traceEvents"]
X = [e for e in ev if e.get("ph") == "X" and "dur" in e]
K = sorted((e for e in X if e.get("cat") == "kernel"), key=lambda e: e["ts"])
print("events", len(ev), "kernel", len(K))
t0 = K[0]["ts"]
hl = sorted((e for e in X if e.get("cat", "?") == "?" and re.match(r"(prefill|model_forward|prepare_input_tensors|compute_logits|dflash|sampler|decode|execute|step)", e.get("name", ""))), key=lambda e: e["ts"])
for e in hl:
    print(f"  {(e['ts']-t0)/1e3:9.1f} {e['dur']/1e3:8.1f} ms  {e['name'][:60]}")
def union(iv):
    tot = 0; ce = None
    for s, t in sorted(iv):
        if ce is None or s > ce: tot += t - s; ce = t
        elif t > ce: tot += t - ce; ce = t
    return tot
print(f"span {(K[-1]['ts']+K[-1]['dur']-t0)/1e3:.1f} ms busy {union([(e['ts'],e['ts']+e['dur']) for e in K])/1e3:.1f} ms")
by = collections.defaultdict(list)
for e in K: by[str(e["args"].get("recipeHandle"))].append(e)
ag = sorted(by.items(), key=lambda kv: -len(kv[1]))
rows = []
for h, es in ag:
    es.sort(key=lambda e: e["ts"]); runs = 1
    for a, b in zip(es, es[1:]):
        if b["ts"] - a["ts"] > 2000: runs += 1
    ops = collections.Counter()
    for e in es: ops[re.sub(r"0x[0-9A-Fa-f]+(_\d+)?", "*", e["name"])[:18]] += e["dur"]
    rows.append((union([(e["ts"], e["ts"] + e["dur"]) for e in es]), h, runs, ops))
for b, h, runs, ops in sorted(rows, reverse=True)[:18]:
    print(f"  {h[-8:]:>8} busy {b/1e3:8.1f} ms runs~{runs:4d} " + " ".join(f"{n}:{v/1e3:.0f}" for n, v in ops.most_common(5)))
