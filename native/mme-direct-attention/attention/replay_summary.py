# SPDX-License-Identifier: MIT
"""Median replay time / RMS for runtime_attention_probe.py logs.

Usage: python replay_summary.py evidence/opt-*.log
"""
import json
import statistics
import sys
from pathlib import Path

print(f"{'log':<28} {'n':>2} {'median_us':>9} {'min':>7} {'max':>7} {'rms':>11} {'ok':>3}")
for name in sys.argv[1:]:
    path = Path(name)
    lines = path.read_text(errors="replace").splitlines()
    rows = [json.loads(l.split(" ", 1)[1]) for l in lines if l.startswith("PIPELINE_REPLAY ")]
    ok = any(l.startswith("RESULT_RUNTIME_ATTENTION ") for l in lines)
    if not rows:
        print(f"{path.stem:<28} no replays ok={ok}")
        continue
    total = [r["ms"] * 1000 for r in rows]
    rms = rows[-1].get("rms")
    print(f"{path.stem:<28} {len(rows):>2} {statistics.median(total):>9.1f} {min(total):>7.1f} "
          f"{max(total):>7.1f} {'-' if rms is None else format(rms, '.9f'):>11} {'yes' if ok else 'NO':>3}")
