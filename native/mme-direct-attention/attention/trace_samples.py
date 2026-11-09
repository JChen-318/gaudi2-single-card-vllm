# SPDX-License-Identifier: MIT
"""Print one sample event (with args) per (category, name) of interest from a Chrome trace."""
import json, sys
d = json.load(open(sys.argv[1]))
want = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
seen = set()
for e in d["traceEvents"]:
    if e.get("ph") != "X": continue
    key = (e.get("cat"), e.get("name"))
    if key in seen: continue
    if want and e.get("name") not in want and e.get("cat") not in want: continue
    seen.add(key)
    print(json.dumps(e)[:900])
    if len(seen) > 40: break
