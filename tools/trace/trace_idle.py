"""Where the device sits idle inside one decode round: gaps between kernel activity, by host phase and gap size.
Usage: python trace_idle.py <trace.json.gz>"""
import collections, gzip, json, re, sys
ev = json.load(gzip.open(sys.argv[1], "rt"))["traceEvents"]
X = [e for e in ev if e.get("ph") == "X" and "dur" in e]
K = sorted(((e["ts"], e["ts"] + e["dur"]) for e in X if e.get("cat") == "kernel"))
hl = [e for e in X if e.get("cat", "?") == "?" and e["pid"] != 0]
names = collections.Counter(re.sub(r"\d+", "N", e["name"]) for e in hl)
PH = ("model_forward", "compute_logits", "sampler", "dflash2_draft_model", "dflash2_candidate_selector",
      "dflash2_proposer", "prepare_input_tensors", "decode", "prefill")
def phase_at(ts):
    act = [e for e in hl if e["ts"] <= ts <= e["ts"] + e["dur"] and e["name"].startswith(PH)]
    if not act: return "(between phases)"
    return re.sub(r"_bs\d+.*|\d+_seq.*", "", min(act, key=lambda e: e["dur"])["name"])
merged = []
for s, t in K:
    if merged and s <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], t)
    else: merged.append([s, t])
gaps = [(a[1], b[0]) for a, b in zip(merged, merged[1:]) if b[0] > a[1]]
span = merged[-1][1] - merged[0][0]; busy = sum(t - s for s, t in merged)
print(f"kernel span {span/1000:.2f} ms, busy {busy/1000:.2f} ms, idle {(span-busy)/1000:.2f} ms in {len(gaps)} gaps")
by = collections.defaultdict(lambda: [0, 0.0, 0, 0.0, 0, 0.0])
for s, t in gaps:
    d = t - s; p = phase_at((s + t) / 2); i = 0 if d < 100 else 2 if d < 1000 else 4
    by[p][i] += 1; by[p][i + 1] += d
print(f"{'host phase':30} {'<0.1ms gaps':>16} {'0.1-1ms gaps':>16} {'>1ms gaps':>16} {'total ms':>9}")
for p, v in sorted(by.items(), key=lambda kv: -(kv[1][1] + kv[1][3] + kv[1][5])):
    print(f"{p:30} {v[0]:6d} {v[1]/1000:7.2f}ms {v[2]:6d} {v[3]/1000:7.2f}ms {v[4]:6d} {v[5]/1000:7.2f}ms {(v[1]+v[3]+v[5])/1000:9.2f}")
print("host phase durations (ms):", {re.sub(r'_bs\d+.*|\d+_seq.*', '', e['name']): round(e['dur'] / 1000, 2) for e in sorted(hl, key=lambda e: e['ts']) if e['name'].startswith(PH)})
