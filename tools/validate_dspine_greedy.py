#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare speculative tokens with independent one-token target prefill calls."""

import argparse
import json
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8016")
    parser.add_argument("--model", default="qwen3.8-27b-dspine")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    def generate(prompt, length):
        body = {"model": args.model, "prompt": prompt, "max_tokens": length,
                "temperature": 0, "return_token_ids": True, "ignore_eos": True,
                "add_special_tokens": False}
        req = urllib.request.Request(args.base + "/v1/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as response:
            return json.load(response)

    full = generate("def factorial(n: int) -> int:\n    ", args.tokens)
    choice = full["choices"][0]
    prefix = full.get("prompt_token_ids") or choice.get("prompt_token_ids")
    speculative = choice["token_ids"]
    assert prefix and len(speculative) == args.tokens
    matches = []
    for index, expected in enumerate(speculative):
        actual = generate(prefix + speculative[:index], 1)["choices"][0]["token_ids"][0]
        matches.append({"position": index, "speculative": expected, "target_prefill": actual,
                        "match": expected == actual})
        print(json.dumps(matches[-1]), flush=True)
    Path(args.output).write_text(json.dumps({"generation": full, "comparisons": matches}, indent=2),
                                 encoding="utf-8")
    assert all(item["match"] for item in matches), "Speculative vs target-only prefill mismatch"


if __name__ == "__main__":
    main()
