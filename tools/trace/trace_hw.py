"""Device-side breakdown of a vLLM-Gaudi E2E trace captured with HW trace on.

Usage: python trace_hw.py <trace.json.gz>
Maps every device kernel event to the host phase that enqueued its recipe, then reports per decode
round: device busy (union over all engines), idle gaps, per-phase device time and top kernels.
"""
import bisect, collections, gzip, json, re, sys

ev = json.load(gzip.open(sys.argv[1], "rt"))["traceEvents"]
X = [e for e in ev if e.get("ph") == "X" and "dur" in e]
pname = {e["pid"]: e["args"].get("name") for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
tname = {(e["pid"], e["tid"]): e["args"].get("name") for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
ENG = next(p for (p, t), n in tname.items() if n and "VLLM::EngineCor" in n)
K = [e for e in X if e.get("cat") == "kernel"]
print(f"events={len(ev)} kernel={len(K)} span={(max(e['ts']+e['dur'] for e in X)-min(e['ts'] for e in X))/1000:.1f} ms")

def union(iv):
    iv = sorted(iv); tot = 0; cs = ce = None; merged = []
    for s, t in iv:
        if cs is None or s > ce:
            if cs is not None: merged.append((cs, ce))
            cs, ce = s, t
        else: ce = max(ce, t)
    if cs is not None: merged.append((cs, ce))
    return sum(b - a for a, b in merged), merged

hl = sorted([e for e in X if e["pid"] == ENG and e.get("cat", "?") == "?"], key=lambda e: e["ts"])
print("high-level events:", collections.Counter(re.sub(r"\d+", "N", e["name"]) for e in hl).most_common(12))
PH = ("model_forward", "compute_logits", "sampler", "dflash2_draft_model", "dflash2_candidate_selector",
      "dflash2_proposer", "prepare_input_tensors", "prefill", "decode")
def phase_at(ts):
    act = [e for e in hl if e["ts"] <= ts <= e["ts"] + e["dur"] and e["name"].startswith(PH)]
    if not act: return "(between phases)"
    n = min(act, key=lambda e: e["dur"])["name"]
    return re.sub(r"_bs\d+.*", "", n)

# recipe handle -> phase at host enqueue
h2p = {}
for e in X:
    m = re.match(r"enqueueWithExternalEvents recipeHandle (0x[0-9a-f]+)", e.get("name", ""))
    if m:
        h2p.setdefault(int(m.group(1), 16), collections.Counter())[phase_at(e["ts"])] += 1
def kphase(e):
    h = (e.get("args") or {}).get("recipeHandle")
    try: c = h2p.get(int(h))
    except (TypeError, ValueError): c = None
    return c.most_common(1)[0][0] if c else "(unmapped)"
for e in K: e["_ph"] = kphase(e)
print("kernel events by enqueue phase:", collections.Counter(e["_ph"] for e in K).most_common())

dec = [e for e in hl if e["name"] == "decode"]
print(f"decode rounds in window: {len(dec)}; durations ms: {[round(d['dur']/1000,1) for d in dec]}")
k0, k1 = min(e["ts"] for e in K), max(e["ts"] + e["dur"] for e in K)
busy, merged = union([(e["ts"], e["ts"] + e["dur"]) for e in K])
print(f"\nkernel activity span {(k1-k0)/1000:.2f} ms, device busy (any engine) {busy/1000:.2f} ms = {100*busy/(k1-k0):.0f}%")

def cls(tid):
    n = tname.get((0, tid), "") or ""
    return "TPC" if "TPC" in n else "MME" if "MME" in n else "DMA" if "DMA" in n else (n or str(tid))
bycls = collections.defaultdict(list)
for e in K: bycls[cls(e["tid"])].append((e["ts"], e["ts"] + e["dur"]))
print("busy by engine class (union within class):", {c: round(union(v)[0] / 1000, 2) for c, v in bycls.items()})

# full rounds: decode start -> next decode start
starts = [d["ts"] for d in dec]
rounds = list(zip(starts[:-1], starts[1:])) or [(k0, k1)]
print(f"\ncomplete rounds: {len(rounds) if len(starts) > 1 else 0} (else whole window)")
for a, b in rounds:
    kk = [e for e in K if e["ts"] >= a and e["ts"] < b]
    bz, mg = union([(max(e["ts"], a), min(e["ts"] + e["dur"], b)) for e in kk])
    print(f"\n=== round {((b-a)/1000):.2f} ms: device busy {bz/1000:.2f} ms ({100*bz/(b-a):.0f}%), kernel events {len(kk)}")
    # idle gaps > 0.3 ms with host phase
    gaps = []
    cur = a
    for s, t in mg:
        if s - cur > 300: gaps.append((cur, s))
        cur = max(cur, t)
    if b - cur > 300: gaps.append((cur, b))
    gp = collections.defaultdict(float)
    for s, t in gaps: gp[phase_at((s + t) / 2)] += t - s
    print("  device idle gaps >0.3ms by host phase (ms):", {k: round(v / 1000, 2) for k, v in sorted(gp.items(), key=lambda kv: -kv[1])})
    print("  per enqueue phase: device busy union ms | summed engine-time ms | events")
    byp = collections.defaultdict(list)
    for e in kk: byp[e["_ph"]].append(e)
    for p, es in sorted(byp.items(), key=lambda kv: -union([(x['ts'], x['ts']+x['dur']) for x in kv[1]])[0]):
        u = union([(x["ts"], x["ts"] + x["dur"]) for x in es])[0]
        print(f"    {p:30} {u/1000:7.2f} | {sum(x['dur'] for x in es)/1000:8.2f} | {len(es)}")
        byk = collections.defaultdict(list)
        for x in es: byk[re.sub(r"0x[0-9A-Fa-f]+_\d+", "0x*", x["name"])].append((x["ts"], x["ts"] + x["dur"]))
        for n, iv in sorted(byk.items(), key=lambda kv: -union(kv[1])[0])[:9]:
            print(f"        {union(iv)[0]/1000:7.3f} ms wall  {sum(t-s for s,t in iv)/1000:8.3f} eng-ms  {len(iv):6d}  {n[:60]}")
