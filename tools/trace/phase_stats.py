"""Host phase durations per steady decode round from the step timer marks.
python phase_stats.py <steps.log> [first_line]

Marks: E execute_model entry, R readback parsed, P next round prepared, D drafter queued,
F next forward queued, Q presample queued, S sample_tokens returns. Segments are taken between
consecutive marks in time order, so either ordering of P and D is reported correctly.
"""
import collections
import statistics
import sys

lines = [l.split() for l in open(sys.argv[1]) if l.strip()]
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
rounds, cur = [], None
for t, kind, *rest in lines[start:]:
    t = float(t)
    if kind == "E":
        if cur and "S" in cur and cur.get("dec") in (1, 2) and cur.get("pre") == 0:
            cur["E2"] = t
            rounds.append(cur)
        cur = {"E": t, "order": ["E"]}
    elif cur is not None:
        cur[kind] = t
        cur["order"].append(kind)
        if kind == "S":
            cur["dec"], cur["pre"] = int(rest[0]), int(rest[1])
for batch in (1, 2):
    sel = [r for r in rounds if r["dec"] == batch and r["E2"] - r["E"] < 0.5]
    if len(sel) < 5:
        continue
    segments = collections.defaultdict(list)
    for r in sel:
        order = r["order"] + ["E2"]
        for a, b in zip(order, order[1:]):
            segments[f"{a}->{b}"].append((r[b] - r[a]) * 1e3)
    print(f"B={batch} rounds={len(sel)} (median ms; E execute, R readback, P next prepared, D drafter queued, "
          f"F forward queued, Q presample queued, S return)")
    for name, values in sorted(segments.items(), key=lambda kv: -len(kv[1])):
        if len(values) >= len(sel) // 2:
            print(f"  {name:8s} {statistics.median(values):7.2f}")
    print(f"  ROUND    {statistics.median((r['E2'] - r['E']) * 1e3 for r in sel):7.2f}")
