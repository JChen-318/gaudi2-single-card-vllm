#!/usr/bin/env python
"""Answer-quality gate for changes that are not bit-identical to the eager path.

Rounding can legitimately move a temp-0 output, so byte equality cannot judge those changes. This
checks what must survive: factual answers, and recall of a fact placed at the start of a long
context, which exercises the QSA key/value cache and the GDN state across thousands of tokens.

    qa_gate_h30.py --out result.json     # exit status is the number of failed cases
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import bench_util_h30 as B  # noqa: E402

FILLER = B.PASSAGE
SHORT = (
    ("What is 17 * 23? Give the number.", r"\b391\b"),
    ("What is the capital of Australia?", r"Canberra"),
    ("Who wrote the novel Pride and Prejudice?", r"Austen"),
    ("What is the chemical symbol for gold?", r"\bAu\b"),
    ("In what year did World War II end?", r"\b1945\b"),
    ("What is the largest planet in our solar system?", r"Jupiter"),
    ("Write a Python function named add that returns the sum of two numbers.", r"def add"),
    ("Translate 'good morning' into Spanish.", r"[Bb]uenos d"),
    ("What is the square root of 144?", r"\b12\b"),
    ("Who wrote The Old Man and the Sea?", r"Hemingway"),
    ("What is the boiling point of water at sea level in Celsius?", r"\b100\b"),
    ("Name the author of the play Hamlet.", r"Shakespeare"),
)
RECALL = tuple(
    (f"Remember this: the secret code is {code}.\n\n" + FILLER * units + "\nWhat is the secret code? Give the number.",
     rf"\b{code}\b")
    for code, units in ((7342, 16), (5819, 64), (2967, 256), (4183, 700))
)


def corrupted(text: str) -> str | None:
    if re.search(r"!{4,}", text):
        return "run of !"
    if re.search(r"(.)\1{24,}", text):
        return "repeated character"
    if len(text) < 8:
        return "almost empty"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--short-only", action="store_true")
    args = ap.parse_args()
    rows = []
    failed = 0
    for prompt, pattern in SHORT + (() if args.short_only else RECALL):
        t0 = time.time()
        text = B.stream(prompt, args.max_tokens)["text"]
        bad = corrupted(text)
        hit = re.search(pattern, text) is not None
        ok = hit and bad is None
        failed += 0 if ok else 1
        rows.append({"prompt": prompt[:80], "tokens_prompt_chars": len(prompt), "pattern": pattern, "ok": ok,
                     "corrupt": bad, "seconds": round(time.time() - t0, 1), "text": text})
        print(("PASS" if ok else "FAIL"), pattern, f"{rows[-1]['seconds']}s", bad or "",
              "" if ok else repr(text[-160:]), flush=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"passed {len(rows) - failed}/{len(rows)}", flush=True)
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
