# SPDX-License-Identifier: MIT
"""Wall-clock attribution of an h3prof device trace by compiled graph (recipeHandle).

For every recipe: union of its kernel intervals over all engines (wall ms), MME and TPC
union busy, top MME module paths (EventName prefix) and top TPC kernels. Also the global
device span, union busy and idle. Each recipe gets a coarse class from its contents.
"""
import json, sys
from collections import defaultdict

d = json.load(open(sys.argv[1]))
tname = {}
for e in d["traceEvents"]:
    if e.get("ph") == "M" and e.get("name") == "thread_name":
        tname[(e["pid"], e["tid"])] = e["args"].get("name", "")
rec = defaultdict(lambda: {"all": [], "mme": [], "tpc": [], "mname": defaultdict(float), "tname": defaultdict(float)})
allv = []
for e in d["traceEvents"]:
    if e.get("ph") != "X" or e.get("cat") != "kernel": continue
    lane = tname.get((e["pid"], e["tid"]), "")
    a = e.get("args") or {}
    r = rec[a.get("recipeHandle", "?")]
    iv = (e["ts"], e["ts"] + e["dur"])
    r["all"].append(iv); allv.append(iv)
    if "MME" in lane:
        r["mme"].append(iv)
        p = "/".join(str(a.get("EventName", "")).split("/")[:3])
        r["mname"][p] += e["dur"]
    elif "TPC" in lane:
        r["tpc"].append(iv)
        r["tname"][e["name"]] += e["dur"]


def union(iv):
    iv = sorted(iv); tot = 0.0; cs = ce = None
    for s, t in iv:
        if cs is None or s > ce:
            if cs is not None: tot += ce - cs
            cs, ce = s, t
        elif t > ce: ce = t
    if cs is not None: tot += ce - cs
    return tot


span = (max(t for _, t in allv) - min(s for s, _ in allv)) / 1e3
busy = union(allv) / 1e3
print(f"device span {span:.1f} ms, any-engine busy {busy:.1f} ms, idle {span - busy:.1f} ms")
rows = []
for h, r in rec.items():
    w = union(r["all"]) / 1e3
    tn = sorted(r["tname"].items(), key=lambda kv: -kv[1])[:3]
    mn = sorted(r["mname"].items(), key=lambda kv: -kv[1])[:3]
    text = " ".join(n for n, _ in tn) + " " + " ".join(n for n, _ in mn)
    cls = ("attention" if ("sdpa" in text or "softmax" in text or "0x64A62CFE" in text or "BatchGemm" in text
                           or "attn" in text.lower() and "proj" not in text) else
           "gemm" if mn and r["mname"] and sum(r["mname"].values()) > 0.3 * sum(r["tname"].values() or [0]) else "tpc-other")
    rows.append((w, h, cls, union(r["mme"]) / 1e3, union(r["tpc"]) / 1e3, tn, mn))
rows.sort(key=lambda x: -x[0])
tot = defaultdict(float)
for w, h, cls, *_ in rows: tot[cls] += w
print("wall by class (ms):", {k: round(v, 1) for k, v in tot.items()}, "sum", round(sum(tot.values()), 1))
for w, h, cls, mb, tb, tn, mn in rows[:int(sys.argv[2]) if len(sys.argv) > 2 else 40]:
    print(f"{w:9.1f} ms {cls:9} mme {mb:8.1f} tpc {tb:8.1f} | TPC " +
          ", ".join(f"{n[:34]} {v/1e3:.0f}" for n, v in tn) + " | MME " + ", ".join(f"{n[:60]} {v/1e3:.0f}" for n, v in mn))
