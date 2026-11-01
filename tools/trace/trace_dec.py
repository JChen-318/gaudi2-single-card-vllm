"""Decode round breakdown: python3 trace_dec.py <trace>"""
import collections, gzip, json, re, sys
ev = json.load(gzip.open(sys.argv[1], "rt"))["traceEvents"]
X = [e for e in ev if e.get("ph") == "X" and "dur" in e]
K = sorted((e for e in X if e.get("cat") == "kernel"), key=lambda e: e["ts"])
t0 = K[0]["ts"]; t1 = max(e["ts"] + e["dur"] for e in K)
def merge(iv):
    out = []
    for s, t in sorted(iv):
        if out and s <= out[-1][1]: out[-1][1] = max(out[-1][1], t)
        else: out.append([s, t])
    return out
def union(iv): return sum(b - a for a, b in merge(iv))
print(f"span {(t1-t0)/1e3:.2f} ms  busy {union([(e['ts'],e['ts']+e['dur']) for e in K])/1e3:.2f} ms  kernels {len(K)}")
def cls(e):
    n = (e["args"].get("EventName") or "") + " " + e["name"]
    for pat, name in ((r"gate_up_proj", "ffn gate_up"), (r"down_proj", "ffn down"), (r"in_proj_qkvz", "gdn in_proj"), (r"in_proj_ba", "gdn in_proj_ba"),
                      (r"out_proj", "gdn out_proj"), (r"qkv_proj", "attn qkv"), (r"o_proj", "attn o_proj"), (r"lm_head|score|embed_tokens", "lm_head/embed"),
                      (r"flashinfer_gaudi_gdn", "gdn fused kernel"), (r"sdpa|softmax", "attn softmax/sdpa"), (r"rotary|rope", "rope"), (r"norm", "norm")):
        if re.search(pat, n): return name
    if "BatchGemm" in e["name"] or "batch_gemm" in n: return "BatchGemm (attn QK/AV)"
    if e["name"].startswith("GEMM"): return "other GEMM"
    if "gather" in n or "index_" in n: return "gather/index"
    if "fused_kernel" in e["name"]: return "fused tpc (unnamed)"
    return "other tpc"
by = collections.defaultdict(list)
for e in K: by[cls(e)].append(e)
print(f"{'class':28}{'wall ms':>9}{'eng ms':>9}{'events':>8}")
for c, es in sorted(by.items(), key=lambda kv: -union([(e['ts'],e['ts']+e['dur']) for e in kv[1]])):
    print(f"{c:28}{union([(e['ts'],e['ts']+e['dur']) for e in es])/1e3:9.2f}{sum(e['dur'] for e in es)/1e3:9.2f}{len(es):8d}")
runs = []
for e in K:
    h = str(e["args"].get("recipeHandle"))
    if runs and runs[-1][0] == h: runs[-1][1].append(e)
    else: runs.append((h, [e]))
print(f"\nrecipe runs {len(runs)}; timeline (ms): start wall busy gap_before  recipe  top")
prev = t0; gaps = []; rows = []
names = {}
for h, es in runs:
    s = min(e["ts"] for e in es); t = max(e["ts"] + e["dur"] for e in es)
    if s - prev > 30: gaps.append(((prev - t0) / 1e3, (s - prev) / 1e3))
    ops = collections.Counter()
    for e in es: ops[cls(e)] += e["dur"]
    rows.append(((s - t0) / 1e3, (t - s) / 1e3, union([(e["ts"], e["ts"] + e["dur"]) for e in es]) / 1e3, (s - prev) / 1e3, h[-6:], " ".join(f"{k}:{v/1e3:.1f}" for k, v in ops.most_common(3))))
    prev = max(prev, t)
for r in rows:
    if r[2] > 0.08 or r[3] > 0.1: print(f"{r[0]:7.2f} {r[1]:6.2f} {r[2]:6.2f} {r[3]:6.2f}  {r[4]} {r[5]}")
print(f"\nidle gaps >30us: total {sum(g for _, g in gaps):.2f} ms in {len(gaps)}; biggest: {sorted(((round(g,2), round(a,1)) for a, g in gaps), reverse=True)[:14]}")
