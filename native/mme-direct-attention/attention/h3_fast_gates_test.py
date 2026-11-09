# SPDX-License-Identifier: MIT
"""H3DIRECT_FAST_GATES=1 rejection-path test on a real layer (isolated, card idle).

1. Normal input: output SHA must equal --expect-sha (validated standard path).
2. NaN / +Inf / -Inf injected into one head of Q, K or V (one case at a time):
   exactly that head must fall back (dummy fallback writes a marker value), all
   other heads must be bit-identical to the normal output."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
import habana_frameworks.torch.core  # noqa: F401
import h3_direct_attention as hd

p = argparse.ArgumentParser()
p.add_argument("dump", type=Path)
p.add_argument("--repeats", type=int, default=1)
p.add_argument("--expect-sha", default=os.environ.get("H3_EXPECT_SHA", "0b66cf40ed2bbcadd4c19ab94754fdadbd3da34b3ebab445e6cf291061d48138"))
a = p.parse_args()
assert hd.POW2_Q and os.environ.get("H3DIRECT_FAST_GATES") == "1", "run with H3DIRECT_POW2Q=1 H3DIRECT_FAST_GATES=1"
d = torch.load(a.dump, map_location="cpu", weights_only=True, mmap=True)
scale = float(d["scale"])
q, k, v = (d[n].permute(0, 2, 1, 3).contiguous().to("hpu") for n in ("q", "k", "v"))
torch.hpu.synchronize()
MARK = 3.0
calls = []


def fallback(qq, kk, vv, s):
    calls.append(qq.shape[1])
    return torch.full_like(qq, MARK)


def run(qq, kk, vv):
    out = hd.direct_max_free_attention(qq, kk, vv, scale, fallback)
    torch.hpu.synchronize()
    return out


sha = lambda t: hashlib.sha256(t.detach().cpu().contiguous().view(torch.int16).numpy().tobytes()).hexdigest()  # noqa: E731
normal = run(q, k, v)
normal_sha = sha(normal)
results = dict(normal_sha=normal_sha, normal_matches=normal_sha == a.expect_sha, fallback_calls_normal=list(calls))
assert normal_sha == a.expect_sha, results
assert not calls, "normal input must not fall back"
normal_cpu = normal[0].cpu()
cases = []
for name, tensor, head, value in (("q", q, 3, float("nan")), ("k", k, 7, float("nan")), ("v", v, 11, float("inf")),
                                   ("k", k, 20, float("-inf")), ("q", q, 41, float("inf")), ("v", v, 55, float("nan"))):
    bad = tensor.clone()
    bad[0, head, 1234, 5] = value
    args = {"q": q, "k": k, "v": v}
    args[name] = bad
    calls.clear()
    before = hd.STATS["fallback_heads"]
    out = run(args["q"], args["k"], args["v"])[0].cpu()
    fell = hd.STATS["fallback_heads"] - before
    others = [h for h in range(out.shape[0]) if h != head]
    same = bool(torch.equal(out[others], normal_cpu[others]))
    marked = bool(torch.all(out[head] == MARK))
    case = dict(tensor=name, head=head, value=str(value), fallback_heads=fell, other_heads_identical=same,
                bad_head_fell_back=marked)
    cases.append(case)
    print("FAST_GATE_CASE " + json.dumps(case), flush=True)
    assert fell == 1 and same and marked, case
again = run(q, k, v)
results.update(cases=cases, recovered_sha=sha(again), stats=hd.STATS)
assert sha(again) == normal_sha
print("FAST_GATES_OK " + json.dumps(results), flush=True)
