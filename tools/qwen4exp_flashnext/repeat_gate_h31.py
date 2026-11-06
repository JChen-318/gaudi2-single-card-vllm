#!/usr/bin/env python
"""Does a request come out the same whatever ran before it? (H31 gate)

H31 found the decode graph handing the GDN state pools to its compiled step twice (a detached
alias to read, the pool to write). The next request's prefill then depended on what the previous
request had decoded: the same prompt sent twice in a row gave a different first-token logprob than
the same prompt sent after another one. A temp-0 golden does not catch that, because it only
compares one run order. This gate sends A, B, B, C, B and requires every B to match exactly
(tokens and logprobs of the first ``--tokens`` completion tokens).

    repeat_gate_h31.py            # exit code = number of mismatching B runs
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import bench_util_h30 as B  # noqa: E402
import real_long_h31 as R  # noqa: E402


def run(prompt: str, n: int) -> list[tuple[str, float]]:
    body = json.dumps({
        "model": B.MODEL,
        "messages": [{"role": "user", "content": prompt + "\n\nSummarize the material above."}],
        "max_tokens": n, "temperature": 0.0, "ignore_eos": True, "logprobs": True, "top_logprobs": 1,
    }).encode()
    req = urllib.request.Request(B.BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        data = json.load(resp)
    return [(t["token"], t["logprob"]) for t in data["choices"][0]["logprobs"]["content"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=80)
    ap.add_argument("--size", type=int, default=1500, help="prompt B length in tokens")
    args = ap.parse_args()
    text = R.corpus()
    prompt_a = R.prompt_of(text[200000:], 6000)
    prompt_b = R.prompt_of(text[300000:], args.size)
    prompt_c = R.prompt_of(text[700000:], 300)
    order = [("A", prompt_a), ("B", prompt_b), ("B", prompt_b), ("C", prompt_c), ("B", prompt_b)]
    outs = [(name, run(prompt, args.tokens if name == "B" else 4)) for name, prompt in order]
    runs_b = [(i, out) for i, (name, out) in enumerate(outs) if name == "B"]
    ref_index, ref = runs_b[0]
    bad = 0
    for index, out in runs_b[1:]:
        same = out == ref
        bad += not same
        first = next((j for j, (x, y) in enumerate(zip(ref, out)) if x != y), None)
        print(f"B#{index} after {outs[index - 1][0]}: {'same' if same else f'differs at token {first}'}"
              f" (first-token logprob {out[0][1]:.6f} vs {ref[0][1]:.6f})")
    print("repeat gate:", "PASS" if not bad else f"FAIL ({bad})")
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
