"""Summarise TORCH_LOGS=graph_breaks output in a vLLM service log.
Usage: python3 gb_parse.py <log> [max_examples]
Groups breaks by (user code location, reason first line); prints counts and one full example each."""
import collections, re, sys

path = sys.argv[1]
nex = int(sys.argv[2]) if len(sys.argv) > 2 else 12
txt = open(path, errors="ignore").read().replace("\r", "\n")
txt = re.sub(r"^\((?:EngineCore|APIServer)[^)]*\) ?", "", txt, flags=re.M)
# entries start at a line mentioning 'Graph break' and run until the next log record header
parts = re.split(r"(?m)^(?=.*\[__graph_breaks\])", txt)
if len(parts) < 2:
    parts = re.split(r"(?m)^(?=.*Graph break)", txt)
groups = collections.OrderedDict()
for p in parts[1:]:
    body = p[:6000]
    end = re.search(r"(?m)^(?:INFO|WARNING|ERROR|DEBUG) \d\d-\d\d \d\d:\d\d:\d\d|^\S+ warmup processing", body[200:])
    if end:
        body = body[:200 + end.start()]
    loc = re.search(r'Graph break in user code at (\S+?):(\d+)', body) or re.search(r'File "([^"]+)", line (\d+), in (\w+)', body)
    reason = re.search(r"(?:Graph Break Reason|Reason|Explanation): *(.+)", body)
    hint = re.search(r"(?m)^\s*(Unsupported[^\n]*|.*Dynamic control flow[^\n]*|.*data-dependent[^\n]*|.*Developer debug context:[^\n]*)", body)
    key = ((loc.group(1).split("/")[-1] + ":" + loc.group(2)) if loc else "?",
           (reason.group(1) if reason else (hint.group(1).strip() if hint else body.strip().split("\n")[0]))[:150])
    g = groups.setdefault(key, [0, body])
    g[0] += 1
print(f"graph-break records: {sum(g[0] for g in groups.values())}, distinct: {len(groups)}")
for (loc, reason), (n, body) in sorted(groups.items(), key=lambda kv: -kv[1][0]):
    print(f"{n:6d}  {loc:38} {reason}")
print("\n================ examples")
for i, ((loc, reason), (n, body)) in enumerate(sorted(groups.items(), key=lambda kv: -kv[1][0])):
    if i >= nex:
        break
    print(f"\n----- [{n}x] {loc}\n" + "\n".join(l[:230] for l in body.strip().split("\n")[:34]))
