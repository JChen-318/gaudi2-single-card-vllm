"""Side-by-side device kernel breakdown of several one-round HW traces.

Usage: python trace_hw_cmp.py <label=trace.json.gz> ...
Per trace: kernel span, device busy, MME/TPC busy; then per kernel name the wall time (union over engines)
and summed engine time, sorted by growth between the first and last trace.
"""
import collections, gzip, json, re, sys

def union(iv):
    iv = sorted(iv); tot = 0; cs = ce = None
    for s, t in iv:
        if cs is None or s > ce:
            if cs is not None: tot += ce - cs
            cs, ce = s, t
        else: ce = max(ce, t)
    if cs is not None: tot += ce - cs
    return tot

res = []
for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    ev = json.load(gzip.open(path, "rt"))["traceEvents"]
    tname = {(e["pid"], e["tid"]): e["args"].get("name") for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
    K = [e for e in ev if e.get("ph") == "X" and "dur" in e and e.get("cat") == "kernel"]
    hl = [e for e in ev if e.get("ph") == "X" and "dur" in e and e.get("cat", "?") == "?"]
    mf = [e["name"] for e in hl if e["name"].startswith(("model_forward", "compute_logits"))]
    dec = [round(e["dur"] / 1000, 1) for e in hl if e["name"] == "decode"]
    def cls(tid):
        n = tname.get((0, tid), "") or ""
        return "TPC" if "TPC" in n else "MME" if "MME" in n else "DMA" if "DMA" in n else "other"
    k0, k1 = min(e["ts"] for e in K), max(e["ts"] + e["dur"] for e in K)
    bycls = collections.defaultdict(list); byk = collections.defaultdict(list)
    for e in K:
        iv = (e["ts"], e["ts"] + e["dur"])
        bycls[cls(e["tid"])].append(iv)
        byk[re.sub(r"0x[0-9A-Fa-f]+(_\d+)?", "0x*", e["name"])].append(iv)
    res.append(dict(label=label, span=(k1 - k0) / 1000, busy=union([(e["ts"], e["ts"] + e["dur"]) for e in K]) / 1000,
                    cls={c: union(v) / 1000 for c, v in bycls.items()}, n=len(K), mf=mf, dec=dec,
                    k={n: (union(v) / 1000, sum(t - s for s, t in v) / 1000, len(v)) for n, v in byk.items()}))

W = 26
print(f"{'':34}" + "".join(f"{r['label']:>{W}}" for r in res))
print(f"{'phases':34}" + "".join(f"{(r['mf'][0] if r['mf'] else '?')[:W-1]:>{W}}" for r in res))
print(f"{'decode step (host, ms)':34}" + "".join(f"{str(r['dec']):>{W}}" for r in res))
for key, f in (("kernel span ms", lambda r: r["span"]), ("device busy ms", lambda r: r["busy"]),
               ("  MME busy", lambda r: r["cls"].get("MME", 0)), ("  TPC busy", lambda r: r["cls"].get("TPC", 0)),
               ("  DMA busy", lambda r: r["cls"].get("DMA", 0)), ("kernel events", lambda r: r["n"])):
    print(f"{key:34}" + "".join(f"{f(r):>{W}.2f}" for r in res))
names = set().union(*(r["k"] for r in res))
g = lambda r, n: r["k"].get(n, (0, 0, 0))
order = sorted(names, key=lambda n: -(g(res[-1], n)[0] - g(res[0], n)[0]))
print(f"\n{'kernel  (wall ms / eng-ms / count)':34}" + "".join(f"{r['label']:>{W}}" for r in res))
for n in order[:int(28)]:
    print(f"{n[:33]:34}" + "".join(f"{g(r,n)[0]:>9.3f}/{g(r,n)[1]:>8.2f}/{g(r,n)[2]:>6d} " for r in res))
