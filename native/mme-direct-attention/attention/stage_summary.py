# SPDX-License-Identifier: MIT
"""Summarize stage-ablation logs: median device wait per mode and derived stage costs.

Usage: python stage_summary.py evidence <tag>   (reads evidence/stage-<mode>-<tag>.log)
Derived numbers are differences of medians; each includes that stage's own dispatch/fence cost.
"""
import json
import statistics
import sys
from pathlib import Path

MODES = ("floor", "lbw", "qk", "qk-exp", "qk-exp-pv", "full", "exp", "pv", "norm")
root, tag = Path(sys.argv[1]), sys.argv[2]
med = {}
print(f"{'mode':<10} {'n':>2} {'total_us':>9} {'wait_us':>9} {'min':>8} {'max':>8} {'bytes':>6}")
for mode in MODES:
    path = root / f"stage-{mode}-{tag}.log"
    if not path.exists():
        continue
    rows = [json.loads(line.split(" ", 1)[1]) for line in path.read_text(errors="replace").splitlines()
            if line.startswith("PIPELINE_REPLAY ")]
    if not rows:
        print(f"{mode:<10} no replays")
        continue
    total = [r["ms"] * 1000 for r in rows]
    wait = [r.get("wait_us", 0) for r in rows]
    med[mode] = statistics.median(total)
    print(f"{mode:<10} {len(rows):>2} {med[mode]:>9.1f} {statistics.median(wait):>9.1f} "
          f"{min(total):>8.1f} {max(total):>8.1f} {rows[0].get('command_bytes', 0):>6}")

def diff(a, b):
    return med[a] - med[b] if a in med and b in med else float("nan")

print()
print("derived (difference of medians, us):")
for label, value in (
    ("LBW monitor/SOB writes", diff("lbw", "floor")),
    ("QK incl. dispatch", diff("qk", "floor")),
    ("exp/sum after QK (+fence)", diff("qk-exp", "qk")),
    ("PV after exp (+fence)", diff("qk-exp-pv", "qk-exp")),
    ("normalize after PV (+fence)", diff("full", "qk-exp-pv")),
    ("exp alone", diff("exp", "floor")),
    ("PV alone", diff("pv", "floor")),
    ("normalize alone", diff("norm", "floor")),
):
    print(f"  {label:<30} {value:8.1f}")
