# SPDX-License-Identifier: MIT
"""Characterise small compiled graphs in an h3prof DiT trace.

Per recipeHandle: occurrences (launch windows), wall union, kernel name mix. Graphs whose
wall per occurrence is below --max-ms are grouped by their kernel-name signature, and the
CPU-side aten ops overlapping their launches are counted to name them. Also counts runtime
'Launch' events and the host gap between consecutive device graphs.
"""
import argparse, json
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--max-ms", type=float, default=10.0)
a = ap.parse_args()
d = json.load(open(a.trace))
ev = d["traceEvents"]
tname = {(e["pid"], e["tid"]): e["args"].get("name", "") for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
rec = defaultdict(list)
for e in ev:
    if e.get("ph") == "X" and e.get("cat") == "kernel":
        rec[(e.get("args") or {}).get("recipeHandle", "?")].append(e)


def windows(es, gap=50.0):
    es = sorted(es, key=lambda e: e["ts"]); out = []
    for e in es:
        s, t = e["ts"], e["ts"] + e["dur"]
        if out and s <= out[-1][1] + gap: out[-1][1] = max(out[-1][1], t)
        else: out.append([s, t])
    return out


groups = defaultdict(lambda: {"recipes": 0, "occ": 0, "wall": 0.0, "kern": defaultdict(float)})
allw = []
for h, es in rec.items():
    w = windows(es); allw += w
    wall = sum(t - s for s, t in w)
    per = wall / len(w) / 1e3
    if per >= a.max_ms: continue
    kn = defaultdict(float)
    for e in es:
        lane = tname.get((e["pid"], e["tid"]), "")
        kn[("MME:" if "MME" in lane else "") + e["name"]] += e["dur"]
    sig = " + ".join(sorted(k for k, v in sorted(kn.items(), key=lambda kv: -kv[1])[:3]))
    g = groups[sig]; g["recipes"] += 1; g["occ"] += len(w); g["wall"] += wall
    for k, v in kn.items(): g["kern"][k] += v
tot = sum(g["wall"] for g in groups.values()) / 1e3
print(f"small graphs (< {a.max_ms} ms per launch): {sum(g['occ'] for g in groups.values())} launches, "
      f"{sum(g['recipes'] for g in groups.values())} recipes, wall {tot:.1f} ms")
for sig, g in sorted(groups.items(), key=lambda kv: -kv[1]["wall"])[:30]:
    print(f"{g['wall']/1e3:8.1f} ms  launches {g['occ']:5}  recipes {g['recipes']:4}  {g['wall']/g['occ']:.0f} us/launch  {sig[:120]}")
allw.sort()
gaps = [allw[i + 1][0] - allw[i][1] for i in range(len(allw) - 1) if allw[i + 1][0] > allw[i][1]]
print(f"device graph windows {len(allw)}, inter-graph idle {sum(gaps)/1e3:.1f} ms, gaps>20us {sum(1 for g in gaps if g > 20)}")
launch = [e for e in ev if e.get("ph") == "X" and e.get("name") == "Launch"]
print(f"runtime Launch events {len(launch)}, host time in Launch {sum(e['dur'] for e in launch)/1e3:.1f} ms")
cpu = defaultdict(lambda: [0, 0.0])
for e in ev:
    if e.get("ph") == "X" and e.get("cat") == "cpu_op":
        cpu[e["name"]][0] += 1; cpu[e["name"]][1] += e["dur"]
print("top CPU ops by count:")
for n, (c, t) in sorted(cpu.items(), key=lambda kv: -kv[1][0])[:25]:
    print(f"  {c:6} x  {t/1e3:8.1f} ms  {n[:90]}")
