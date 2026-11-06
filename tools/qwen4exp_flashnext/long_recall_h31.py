#!/usr/bin/env python
"""Needle recall and speed at long context (H31).

Builds a prompt of about ``--tokens`` tokens from the real-text corpus of ``real_long_h31.py``, hides
one sentence with a five-digit code at ``--depth`` (fraction of the prompt), asks for the code, and
reports whether the answer contains it, TTFT, decode TPOT and the server's prefill peak lines.

    long_recall_h31.py --tokens 100000,200000 --depth 0.1,0.5,0.9
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import bench_util_h30 as B  # noqa: E402
import real_long_h31 as R  # noqa: E402


def ask(prompt: str, max_tokens: int) -> tuple[str, float, list[float]]:
    body = json.dumps({
        "model": B.MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(B.BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    stamps: list[float] = []
    text: list[str] = []
    with urllib.request.urlopen(req, timeout=7200) as resp:
        for line in resp:
            if line.startswith(b"data:") and b"[DONE]" not in line and b'"content"' in line:
                stamps.append(time.time())
                try:
                    text.append(json.loads(line[5:])["choices"][0]["delta"].get("content") or "")
                except (ValueError, KeyError, IndexError):
                    pass
    return "".join(text), (stamps[0] - t0) * 1000 if stamps else float("nan"), stamps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="100000")
    ap.add_argument("--depth", default="0.5")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()
    corpus = R.corpus()
    while len(corpus) < 2_000_000:
        corpus = corpus + corpus  # long targets wrap around the corpus
    rng = random.Random(args.seed)
    ok = 0
    total = 0
    for size in (int(s) for s in args.tokens.split(",")):
        body = R.prompt_of(corpus, size)
        for depth in (float(d) for d in args.depth.split(",")):
            code = rng.randrange(10000, 99999)
            cut = int(len(body) * depth)
            cut = body.rfind("\n", 0, cut) + 1 or cut
            needle = f"\nThe secret code for the archive is {code}. Remember this number.\n"
            prompt = body[:cut] + needle + body[cut:] + "\n\nWhat is the secret code for the archive? Reply with the number only."
            text, ttft, stamps = ask(prompt, args.max_tokens)
            gaps = [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]
            hit = str(code) in text
            ok += hit
            total += 1
            print(json.dumps({"tokens": size, "depth": depth, "code": code, "hit": hit, "answer": text[:80],
                              "ttft_ms": round(ttft), "prefill_tok_s": round(size / ttft * 1000),
                              "tpot_median_ms": round(statistics.median(gaps), 2) if gaps else None}), flush=True)
    print(f"recall {ok}/{total}")
    return total - ok


if __name__ == "__main__":
    raise SystemExit(main())
