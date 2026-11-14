#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure warm, single-request decode throughput from streamed token IDs."""

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path

WORKLOADS = {
    "code": "Write a Python merge_sort function with type hints, then explain its time complexity.",
    "chinese": "用中文详细解释投机解码的草稿生成、目标验证和回退过程，并举一个例子。",
    "english": "Explain how a hash table works, including collisions, resizing, and time complexity with examples.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8016")
    parser.add_argument("--model", default="qwen3.8-27b-dspine")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--temperatures", default="0,1")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = {"label": args.label, "arguments": vars(args), "runs": [], "summary": {}}

    def save():
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    def request(workload, temperature, warmup):
        body = {
            "model": args.model, "messages": [{"role": "user", "content": WORKLOADS[workload]}],
            "max_tokens": args.max_tokens, "ignore_eos": True, "temperature": temperature,
            "top_k": 1 if temperature == 0 else 20, "top_p": 1 if temperature == 0 else 0.95,
            "seed": 42, "chat_template_kwargs": {"enable_thinking": False}, "stream": True,
            "stream_options": {"include_usage": True}, "return_token_ids": True,
        }
        req = urllib.request.Request(args.base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        start = time.perf_counter()
        first = last = None
        first_count = 0
        token_ids = []
        parts = []
        usage = None
        with urllib.request.urlopen(req, timeout=1800) as response:
            for line in response:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                data = json.loads(line[6:])
                if "error" in data:
                    raise RuntimeError(data["error"])
                if data.get("usage"):
                    usage = data["usage"]
                for choice in data.get("choices", []):
                    ids = choice.get("token_ids") or []
                    if ids:
                        now = time.perf_counter()
                        if first is None:
                            first, first_count = now, len(ids)
                        last = now
                        token_ids.extend(ids)
                    parts.append(choice.get("delta", {}).get("content") or "")
        end = time.perf_counter()
        if not usage or usage["completion_tokens"] != len(token_ids) or len(token_ids) < 2 or last == first:
            raise RuntimeError(f"Missing/inconsistent streaming token IDs: {len(token_ids)} tokens, usage={usage}")
        row = {
            "workload": workload, "temperature": temperature, "warmup": warmup,
            "usage": usage, "ttft_seconds": first - start, "e2e_seconds": end - start,
            "decode_seconds": last - first, "first_chunk_tokens": first_count,
            "decode_tokens_per_second": (len(token_ids) - first_count) / (last - first),
            "e2e_tokens_per_second": len(token_ids) / (end - start),
            "token_ids_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
            "text": "".join(parts),
        }
        results["runs"].append(row)
        save()
        print(json.dumps({k: v for k, v in row.items() if k not in ("text", "token_ids_sha256")}), flush=True)

    for temperature in map(float, args.temperatures.split(",")):
        for workload in WORKLOADS:
            for _ in range(args.warmup):
                request(workload, temperature, True)
            for _ in range(args.repeats):
                request(workload, temperature, False)
            rows = [r for r in results["runs"] if not r["warmup"] and r["workload"] == workload
                    and r["temperature"] == temperature]
            key = f"{workload}/temperature={temperature:g}"
            results["summary"][key] = {
                metric: statistics.median(r[metric] for r in rows)
                for metric in ("decode_tokens_per_second", "e2e_tokens_per_second", "ttft_seconds")
            }
            save()
    results["metric_definition"] = (
        "Decode throughput = (streamed completion token IDs minus first token-bearing chunk IDs) / "
        "(last token-bearing chunk arrival minus first token-bearing chunk arrival). "
        "Single request, fixed generated token budget, localhost; warmup runs excluded from medians. "
        "Output equality is recorded by hash but is not a pass/fail criterion."
    )
    output.with_suffix(".metrics.txt").write_bytes(urllib.request.urlopen(args.base + "/metrics", timeout=30).read())
    save()
    print(json.dumps(results["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
