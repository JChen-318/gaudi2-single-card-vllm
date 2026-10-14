import json, sys
# Compare two greedy2.py outputs: python3 gcmp.py <reference.json> <candidate.json>
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
same = 0
for k in a:
    if k not in b:
        print(f"{k:12} missing in candidate"); continue
    x, y = a[k]["text"], b[k]["text"]
    n = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
    ok = x == y
    same += ok
    print(f"{k:12} {'IDENTICAL' if ok else 'differs'}  tokens {a[k]['tokens']}/{b[k]['tokens']}  common prefix {n}/{len(x)} chars"
          + ("" if ok else f"  ref[{x[max(0,n-25):n+35]!r}] new[{y[max(0,n-25):n+35]!r}]"))
print(f"{same}/{len(a)} identical")
