"""Round time from the v6 step timer: python steps_stats.py <steps.log> [since_perf_counter]

Uses consecutive execute_model entries ("E") whose previous sample ("S") was a
single decode with no prefill; reports median/percentiles and tokens per round.
"""
import statistics
import sys

rows = [line.split() for line in open(sys.argv[1]) if line.strip()]
since = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
events = [(float(t), k, int(d), int(p), int(n)) for t, k, d, p, n in rows if float(t) >= since]
periods, tokens = [], []
last_e = None
last_s = None
for t, kind, dec, pre, n in events:
    if kind == "E":
        if last_e is not None and last_s is not None and last_s[2] >= 1 and last_s[3] == 0:
            gap = (t - last_e) * 1e3
            if gap < 500:  # skip idle gaps between requests
                periods.append((gap, last_s[2], last_s[4]))
        last_e = t
    else:
        last_s = (t, kind, dec, pre, n)
for batch in (1, 2):
    sel = [g for g, d, n in periods if d == batch]
    tok = [n for g, d, n in periods if d == batch]
    if len(sel) < 5:
        continue
    q = statistics.quantiles(sel, n=20)
    print(f"B={batch}: rounds={len(sel)} median={statistics.median(sel):.2f} ms p10={q[1]:.2f} p90={q[17]:.2f} "
          f"mean={statistics.mean(sel):.2f}  tokens/round={statistics.mean(tok) / batch:.2f}  "
          f"tok/s={sum(tok) / (sum(sel) / 1e3):.1f}")
