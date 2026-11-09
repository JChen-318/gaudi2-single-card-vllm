# SPDX-License-Identifier: MIT
"""Summarise a torch.profiler Chrome trace (CPU+HPU) from h3prof.

Per (process, thread) lane: event count, summed duration, union busy time, span, and the
top names by summed duration. Device lanes (non cpu_op categories) are what matter for
'DiT other compute'. Output: text to stdout and optional JSON.
"""
import argparse, json
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--top", type=int, default=25)
ap.add_argument("--json")
a = ap.parse_args()
with open(a.trace) as f:
    d = json.load(f)
ev = d["traceEvents"]
pname, tname = {}, {}
for e in ev:
    if e.get("ph") == "M":
        if e.get("name") == "process_name": pname[e["pid"]] = e["args"].get("name")
        if e.get("name") == "thread_name": tname[(e["pid"], e["tid"])] = e["args"].get("name")
lanes = defaultdict(list)
cats = defaultdict(int)
for e in ev:
    if e.get("ph") == "X" and "dur" in e:
        lanes[(e["pid"], e["tid"], e.get("cat", ""))].append(e)
        cats[e.get("cat", "")] += 1
print("categories:", dict(cats))


def union(iv):
    iv.sort(); tot = 0; cs = ce = None
    for s, t in iv:
        if cs is None or s > ce:
            if cs is not None: tot += ce - cs
            cs, ce = s, t
        else: ce = max(ce, t)
    if cs is not None: tot += ce - cs
    return tot


out = []
for (pid, tid, cat), es in sorted(lanes.items(), key=lambda kv: -sum(x["dur"] for x in kv[1])):
    by = defaultdict(lambda: [0.0, 0])
    for e in es:
        by[e["name"]][0] += e["dur"]; by[e["name"]][1] += 1
    s0 = min(e["ts"] for e in es); s1 = max(e["ts"] + e["dur"] for e in es)
    busy = union([(e["ts"], e["ts"] + e["dur"]) for e in es])
    lane = {"pid": pid, "tid": tid, "cat": cat, "process": pname.get(pid), "thread": tname.get((pid, tid)),
            "events": len(es), "sum_ms": sum(e["dur"] for e in es) / 1e3, "busy_ms": busy / 1e3, "span_ms": (s1 - s0) / 1e3,
            "top": [(n, round(v[0] / 1e3, 3), v[1]) for n, v in sorted(by.items(), key=lambda kv: -kv[1][0])[:a.top]]}
    out.append(lane)
    print(f"\n== {lane['process']} / {lane['thread']} cat={cat} events={lane['events']} sum={lane['sum_ms']:.1f}ms "
          f"busy={lane['busy_ms']:.1f}ms span={lane['span_ms']:.1f}ms")
    for n, ms, c in lane["top"]:
        print(f"   {ms:10.3f} ms  x{c:<6} {n[:110]}")
if a.json:
    with open(a.json, "w") as f: json.dump(out, f, indent=1)
