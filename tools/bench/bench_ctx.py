import json, sys, time, urllib.request

# Decode speed vs context length. Usage: python3 bench_ctx.py <tag> <approx_prompt_tokens> [max_tokens] [n_parallel=1]
# Prints prompt_tokens, TTFT, decode tok/s, and ms per speculative round (from /metrics deltas, single client only).
tag, approx = sys.argv[1], int(sys.argv[2])
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 400
BASE = "http://127.0.0.1:8000"


def metric(name):
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    return sum(float(l.split()[-1]) for l in txt.splitlines() if l.startswith(name))


lines, i = [], 0
while sum(len(x) for x in lines) < approx * 3.6:      # ~3.6 chars/token for this log-like English text
    i += 1
    lines.append(f"[2026-10-{1 + i % 28:02d} {i % 24:02d}:{(i * 7) % 60:02d}:{(i * 13) % 60:02d}] worker-{i % 17} "
                 f"job {1000 + i} stage {('fetch', 'parse', 'index', 'merge', 'flush')[i % 5]} took {(i * 37) % 911} ms, "
                 f"queue depth {(i * 5) % 43}, status {('ok', 'ok', 'retry', 'ok', 'slow')[i % 5]}")
prompt = ("Below is a service log.\n\n" + "\n".join(lines) +
          "\n\nWrite a detailed incident-style report in English about what this log shows: workload pattern, slow "
          "stages, retries, and recommendations. Use several paragraphs.")
body = {"model": "qwen3.8-27b-fp8-mtp", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
        "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
d0, a0 = metric("vllm:spec_decode_num_drafts_total"), metric("vllm:spec_decode_num_accepted_tokens_total")
req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
t0 = time.perf_counter(); first = last = None; usage = None
with urllib.request.urlopen(req, timeout=1800) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        usage = d.get("usage") or usage
        for ch in d.get("choices", []):
            if ch.get("delta", {}).get("content"):
                now = time.perf_counter() - t0
                first = first if first is not None else now
                last = now
rounds = metric("vllm:spec_decode_num_drafts_total") - d0
acc = metric("vllm:spec_decode_num_accepted_tokens_total") - a0
n = usage["completion_tokens"]
print(json.dumps({"tag": tag, "prompt_tokens": usage["prompt_tokens"], "completion_tokens": n, "ttft_s": round(first, 2),
                  "decode_tok_s": round((n - 1) / (last - first), 1), "rounds": int(rounds),
                  "ms_per_round": round((last - first) / max(rounds, 1) * 1000, 1),
                  "tok_per_round": round(n / max(rounds, 1), 2)}))
