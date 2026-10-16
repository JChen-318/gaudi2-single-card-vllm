"""Post-start priming: once /health answers, send the request mixes that first exercise rarely warmed paths
(single and simultaneous requests, random sampling and greedy, a request joining another's decode), so that any
shape the warm-up missed is compiled now instead of stalling the first real user. Started by
scripts/serve.sh --daemon (PRIME_AFTER_START=0 turns it off); one JSON line per step, a step over 30 s is flagged.

    python3 tools/prime.py [--base http://127.0.0.1:8000] [--model NAME] [--wait 10800]
"""
import argparse
import json
import os
import threading
import time
import urllib.request


def post(base, model, prompt, max_tokens, greedy, timeout=900):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    if greedy:
        body["temperature"] = 0
    headers = {"Content-Type": "application/json"}
    if os.environ.get("API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["API_KEY"]
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())["usage"]["completion_tokens"]


def filler(words):
    return " ".join(f"record {i}: value {i * 37 % 911}, status {('ok', 'retry', 'slow')[i % 3]}." for i in range(words))


PROMPTS = {
    "code": "Write a Python function that merges overlapping intervals, with a docstring and two tests.",
    "essay": "写一篇关于秋天的短文，三段。",
    "list": "List ten facts about the Pacific Ocean, one per line.",
    "log": filler(60) + "\nSummarize the records above in three sentences.",
}


def step(name, base, model, jobs, stagger=0.0):
    """jobs: [(prompt, max_tokens, greedy)] sent together (``stagger`` seconds apart)."""
    t0 = time.perf_counter()
    results, errors = [], []

    def run(job, delay):
        time.sleep(delay)
        try:
            results.append(post(base, model, *job))
        except Exception as exc:  # report and continue: priming must never block the service
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=run, args=(job, i * stagger)) for i, job in enumerate(jobs)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    seconds = round(time.perf_counter() - t0, 2)
    print(json.dumps({"step": name, "seconds": seconds, "tokens": results, "errors": errors,
                      "slow": seconds > 30}), flush=True)
    return seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=os.environ.get("MODEL", "qwen3.8-27b-fp8-mtp"))
    parser.add_argument("--wait", type=int, default=10800, help="seconds to wait for /health")
    args = parser.parse_args()
    deadline = time.time() + args.wait
    while True:
        try:
            if urllib.request.urlopen(args.base + "/health", timeout=5).status == 200:
                break
        except Exception:
            pass
        if time.time() > deadline:
            print(json.dumps({"error": "service not healthy in time"}), flush=True)
            return
        time.sleep(10)
    print(json.dumps({"healthy": time.strftime("%H:%M:%S")}), flush=True)
    p = PROMPTS
    total = 0.0
    total += step("single sampled", args.base, args.model, [(p["code"], 128, False)])
    total += step("single greedy", args.base, args.model, [(p["list"], 128, True)])
    total += step("pair sampled", args.base, args.model, [(p["code"], 256, False), (p["essay"], 256, False)])
    total += step("pair greedy", args.base, args.model, [(p["list"], 256, True), (p["log"], 256, True)])
    total += step("pair mixed", args.base, args.model, [(p["essay"], 256, False), (p["log"], 256, True)])
    total += step("join during decode", args.base, args.model, [(p["log"], 400, False), (p["list"], 128, False)],
                  stagger=2.0)
    total += step("pair sampled again", args.base, args.model, [(p["list"], 256, False), (p["code"], 256, False)])
    print(json.dumps({"done": time.strftime("%H:%M:%S"), "seconds": round(total, 1)}), flush=True)


if __name__ == "__main__":
    main()
