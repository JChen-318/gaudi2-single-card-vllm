# SPDX-License-Identifier: MIT
"""Device-kernel attribution for an h3prof Chrome trace.

1. Prints one sample 'kernel' event per engine type (MME / TPC / DMA) so the args are visible.
2. Groups kernel time by every string arg that looks like a node/graph/recipe identifier.
3. Engine-level union busy time per engine family and per name class.
"""
import json, sys, re
from collections import defaultdict

d = json.load(open(sys.argv[1]))
tname = {}
for e in d["traceEvents"]:
    if e.get("ph") == "M" and e.get("name") == "thread_name":
        tname[(e["pid"], e["tid"])] = e["args"].get("name", "")
ks = [e for e in d["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"]
print("kernel events", len(ks))
seen = set()
for e in ks:
    fam = re.sub(r"[ 0-9]+$", "", tname.get((e["pid"], e["tid"]), "?"))
    if (fam, e["name"][:12]) in seen or len(seen) > 12: continue
    seen.add((fam, e["name"][:12]))
    print("SAMPLE", fam, json.dumps(e)[:1200])
keys = defaultdict(int)
for e in ks[:20000]:
    for k, v in (e.get("args") or {}).items():
        if isinstance(v, str): keys[k] += 1
print("string arg keys:", dict(keys))
cand = [k for k in keys if re.search(r"node|graph|recipe|op|name", k, re.I)]
for k in cand:
    agg = defaultdict(lambda: [0.0, 0])
    for e in ks:
        fam = "MME" if "MME" in tname.get((e["pid"], e["tid"]), "") else ("TPC" if "TPC" in tname.get((e["pid"], e["tid"]), "") else "OTHER")
        v = (e.get("args") or {}).get(k)
        if v is None: continue
        a = agg[(fam, str(v)[:90])]; a[0] += e["dur"]; a[1] += 1
    print(f"\n### grouped by arg '{k}' ({len(agg)} groups), top 40 by summed us")
    for (fam, v), (t, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:40]:
        print(f"{fam:5} {t/1e3:10.1f} ms x{c:<7} {v}")
