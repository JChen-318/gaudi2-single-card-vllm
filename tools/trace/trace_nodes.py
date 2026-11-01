"""Per graph node of each recipe: python trace_nodes.py <trace> <recipe suffix,...> [rows]"""
import collections, gzip, json, re, sys
ev = json.load(gzip.open(sys.argv[1], "rt"))["traceEvents"]
want = sys.argv[2].split(",")
rows = int(sys.argv[3]) if len(sys.argv) > 3 else 28
K = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"]
def union(iv):
    tot = 0; ce = None
    for s, t in sorted(iv):
        if ce is None or s > ce: tot += t - s; ce = t
        elif t > ce: tot += t - ce; ce = t
    return tot
for w in want:
    es = [e for e in K if str((e.get("args") or {}).get("recipeHandle", "")).endswith(w)]
    runs = 0; last = None
    by = collections.defaultdict(list)
    for e in es:
        n = (e["args"].get("EventName") or e["name"])
        by[(re.sub(r"(?<=[/_])\d+(?=[/_]|$)", "N", n), e["name"].split(" ")[0][:24], e["args"].get("dataType"))].append(e)
    r = union([(e["ts"], e["ts"] + e["dur"]) for e in es])
    # number of executions: from the most frequent first-seen event per run is hard; use busy per run printed by trace_recipes
    print(f"\n=== recipe ...{w}: events {len(es)}, busy {r/1000:.2f} ms")
    tot = collections.Counter()
    for k, v in by.items(): tot[k] = union([(e["ts"], e["ts"] + e["dur"]) for e in v])
    for k, u in tot.most_common(rows):
        v = by[k]
        print(f"  wall {u/1000:7.2f}  eng {sum(e['dur'] for e in v)/1000:8.2f}  n {len(v):6d}  {k[1]:24} {str(k[2]):8} {k[0][:88]}")
