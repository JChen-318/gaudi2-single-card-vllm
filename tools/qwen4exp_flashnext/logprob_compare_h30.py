#!/usr/bin/env python
"""Compare two serving paths token by token through their temp-0 logprobs.

    logprob_compare_h30.py --save ref.json            # on the reference server (eager)
    logprob_compare_h30.py --compare ref.json --out cmp.json   # on the candidate server

Temp-0 outputs of two numerically different paths match until the first position where the top
two candidates are nearly tied, then diverge. Up to that point both runs see the same prefix, so
the chosen-token logprobs are directly comparable. The report gives, per prompt, the first
divergent position, the eager top-1/top-2 margin there (a tie flip has a margin of a few hundredths),
and the logprob differences over the shared prefix.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import bench_util_h30 as B  # noqa: E402

PROMPTS = (
    ("essay", "Write a long detailed essay about the history of the Roman Empire.", 500),
    ("code", "Write a Python implementation of a red-black tree with insertion and deletion, with comments.", 500),
    ("math", "A train leaves at 9:40 and travels 237 km at 79 km/h, then 150 km at 100 km/h. When does it "
             "arrive? Solve step by step.", 400),
    ("korean", "조선 왕조의 역사를 시대별로 자세히 설명해줘.", 400),
    ("cross2048", B.PASSAGE * 60 + "Explain in detail what the passage above says and why it repeats.", 500),
    ("long8k", B.PASSAGE * 256 + "Summarize the above, then list three questions it raises.", 200),
)


def run(prompt: str, max_tokens: int) -> list[dict]:
    body = json.dumps({
        "model": B.MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 5,
    }).encode()
    req = urllib.request.Request(B.BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        data = json.load(resp)
    out = []
    for item in data["choices"][0]["logprobs"]["content"]:
        out.append({"token": item["token"], "logprob": item["logprob"],
                    "top": [(t["token"], t["logprob"]) for t in item["top_logprobs"]]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    ap.add_argument("--compare")
    ap.add_argument("--out")
    args = ap.parse_args()
    runs = {name: run(prompt, n) for name, prompt, n in PROMPTS}
    if args.save:
        with open(args.save, "w") as f:
            json.dump(runs, f, ensure_ascii=False)
        print("saved", {k: len(v) for k, v in runs.items()})
        return 0
    with open(args.compare) as f:
        ref = json.load(f)
    report = {}
    for name, got in runs.items():
        want = ref[name]
        n = min(len(want), len(got))
        first = next((i for i in range(n) if want[i]["token"] != got[i]["token"]), None)
        shared = n if first is None else first
        deltas = [abs(want[i]["logprob"] - got[i]["logprob"]) for i in range(shared)]
        row = {"tokens": n, "first_divergence": first, "shared": shared,
               "mean_abs_dlogprob": round(sum(deltas) / max(1, len(deltas)), 5),
               "max_abs_dlogprob": round(max(deltas, default=0.0), 5)}
        if first is not None:
            top = want[first]["top"]
            row["ref_margin_top1_top2"] = round(top[0][1] - top[1][1], 4) if len(top) > 1 else None
            row["ref_token"], row["got_token"] = want[first]["token"], got[first]["token"]
            row["got_token_rank_in_ref_top5"] = next((r for r, (t, _) in enumerate(top) if t == got[first]["token"]),
                                                     None)
        report[name] = row
        print(name, json.dumps(row, ensure_ascii=False), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
