"""Per compiled-region host submit vs device idle: shows where host stops being ahead of device.
Usage: python trace_regions.py <trace>"""
import gzip, json, sys, bisect
ev = json.load(gzip.open(sys.argv[1], "rt"))["traceEvents"]
X = [e for e in ev if e.get("ph") == "X" and "dur" in e]
tn = {(e["pid"], e["tid"]): e["args"].get("name") for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
ENG = next(p for (p, t), n in tn.items() if n and "VLLM::EngineCor" in n)
th = lambda e: tn.get((e["pid"], e["tid"]), "") or ""
K = sorted((e for e in X if e.get("cat") == "kernel"), key=lambda e: e["ts"])
merged = []
for e in K:
    s, t = e["ts"], e["ts"] + e["dur"]
    if merged and s <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], t)
    else: merged.append([s, t])
t0 = merged[0][0]
def idle(a, b):
    busy = sum(max(0, min(b, t) - max(a, s)) for s, t in merged if t > a and s < b)
    return (b - a) - busy
main = [e for e in X if e["pid"] == ENG and "EngineCo" in th(e)]
hl = [e for e in X if e["pid"] == ENG and e.get("cat", "?") == "?"]
regions = sorted((e for e in main if e["name"].startswith("Torch-Compiled Region")), key=lambda e: e["ts"])
look = sorted((e for e in main if e["name"].startswith("TorchDynamo Cache Lookup")), key=lambda e: e["ts"])
def phase(ts):
    act = [e for e in hl if e["ts"] <= ts <= e["ts"] + e["dur"]]
    return min(act, key=lambda e: e["dur"])["name"][:28] if act else "-"
# top-level regions only
top = []
for r in regions:
    if top and r["ts"] < top[-1]["ts"] + top[-1]["dur"]: continue
    top.append(r)
print(f"{'#':>3} {'host ms':>8} {'host dur':>8} {'lookup':>7} {'dev idle since prev':>19} {'dev backlog ms':>14}  phase / region")
prev = t0
for i, r in enumerate(top):
    lk = sum(l["dur"] for l in look if prev <= l["ts"] < r["ts"] + r["dur"] and l["ts"] >= (top[i-1]["ts"] + top[i-1]["dur"] if i else t0))
    # backlog: when host finishes submitting this region, how far ahead is device's last busy point
    end = r["ts"] + r["dur"]
    j = bisect.bisect_right([m[0] for m in merged], end)
    # time after `end` until first idle gap >=50us
    k = j - 1 if j else 0
    t = merged[k][1] if k >= 0 else end
    while k + 1 < len(merged) and merged[k + 1][0] - t < 50: k += 1; t = max(t, merged[k][1])
    print(f"{i:3d} {(r['ts']-t0)/1e3:8.2f} {r['dur']/1e3:8.2f} {lk/1e3:7.2f} {idle(prev, end)/1e3:19.2f} {max(0,t-end)/1e3:14.2f}  {phase(r['ts'])} | {r['name'][22:60]}")
    prev = end
