#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise deterministic/sampled DSpine serving, streaming and concurrent requests."""

import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8016")
    parser.add_argument("--model", default="qwen3.8-27b-dspine")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=0, help="Untimed requests before the repeated greedy pair")
    parser.add_argument("--allow-repeat-variation", action="store_true",
                        help="Record greedy variation without failing the functional smoke check")
    args = parser.parse_args()

    def request(prompt, temperature=0, stream=False):
        body = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens, "temperature": temperature, "stream": stream,
                "seed": 42, "chat_template_kwargs": {"enable_thinking": False}}
        if temperature:
            body.update(top_p=0.95, top_k=20)
        if stream:
            body["stream_options"] = {"include_usage": True}
        req = urllib.request.Request(args.base + "/v1/chat/completions",
                                     json.dumps(body).encode(), {"Content-Type": "application/json"})
        start = time.monotonic()
        first = None
        usage = None
        text = ""
        with urllib.request.urlopen(req, timeout=1800) as response:
            if stream:
                for line in response:
                    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                        continue
                    data = json.loads(line[6:])
                    if data.get("usage"):
                        usage = data["usage"]
                    for choice in data.get("choices", []):
                        content = choice.get("delta", {}).get("content") or ""
                        if content:
                            first = first or time.monotonic()
                            text += content
            else:
                data = json.load(response)
                usage = data["usage"]
                text = data["choices"][0]["message"]["content"] or ""
        end = time.monotonic()
        assert text and usage and usage["completion_tokens"] > 0, "Empty or incomplete generation"
        result = {"prompt": prompt, "temperature": temperature, "stream": stream, "text": text,
                  "usage": usage, "seconds": end - start,
                  "ttft_seconds": None if first is None else first - start,
                  "e2e_tokens_per_second": usage["completion_tokens"] / (end - start)}
        if first:
            result["stream_decode_tokens_per_second"] = (usage["completion_tokens"] - 1) / (end - first)
        print(json.dumps({k: v for k, v in result.items() if k not in ("text", "prompt")}), flush=True)
        return result

    prompt = "Write a Python merge_sort function with type hints, then explain its time complexity."
    for _ in range(args.warmup):
        request(prompt, stream=True)
    results = [request(prompt, stream=True), request(prompt, stream=True)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    repeat_equal = results[0]["text"] == results[1]["text"]
    results.append(request("用中文简要解释投机解码为什么能保持目标模型的输出分布。", temperature=1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(request, p, 0, True) for p in (
            "Explain how a hash table handles collisions.", "Write a short poem about autumn.")]
        results.extend(job.result() for job in jobs)
    metrics = urllib.request.urlopen(args.base + "/metrics", timeout=30).read().decode()
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".metrics.txt").write_text(metrics, encoding="utf-8")
    output.with_suffix(".summary.json").write_text(json.dumps({
        "functional_requests_completed": len(results), "repeated_greedy_equal": repeat_equal,
        "repeat_variation_allowed": args.allow_repeat_variation,
    }, indent=2), encoding="utf-8")
    if not args.allow_repeat_variation:
        assert repeat_equal, "Repeated greedy requests changed output; inspect saved generations"


if __name__ == "__main__":
    main()
