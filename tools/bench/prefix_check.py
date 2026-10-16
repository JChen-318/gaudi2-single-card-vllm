"""Prefix-cache check: TTFT of follow-up turns and greedy output with and without a cache hit.

    python3 prefix_check.py <approx_prompt_tokens> [turns=3] [max_tokens=96]

Turn 1 sends a long log plus a question; each next turn re-sends the conversation with the previous answer and a
new question (what a chat client does). Every turn is sent three times at temperature 0: once normally (may reuse the
cached prefix) and twice with a fresh cache_salt (no reuse, full prefill). The two full prefills show how far greedy
decoding is reproducible at all on this stack; the cached answer should match about as well. The TTFT difference is
the saving. Env: BASE_URL, MODEL, API_KEY."""
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("MODEL", "qwen3.8-27b-fp8-mtp")
approx = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
turns = int(sys.argv[2]) if len(sys.argv) > 2 else 3
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 96
HEADERS = {"Content-Type": "application/json"}
if os.environ.get("API_KEY"):
    HEADERS["Authorization"] = "Bearer " + os.environ["API_KEY"]

lines, i = [], 0
while sum(len(x) for x in lines) < approx * 3.6:
    i += 1
    lines.append(f"[2026-10-{1 + i % 28:02d} {i % 24:02d}:{(i * 7) % 60:02d}:{(i * 13) % 60:02d}] worker-{i % 17} "
                 f"job {1000 + i} stage {('fetch', 'parse', 'index', 'merge', 'flush')[i % 5]} took {(i * 37) % 911} ms, "
                 f"queue depth {(i * 5) % 43}, status {('ok', 'ok', 'retry', 'ok', 'slow')[i % 5]}")
log = "Below is a service log.\n\n" + "\n".join(lines)
questions = ["Which worker appears most often with status slow? Answer briefly.",
             "Summarise the retry pattern in two sentences.",
             "Which stage has the longest single duration, and roughly how long?",
             "List three recommendations, one line each.",
             "What is the largest queue depth you can find?"]


def chat(messages, salt=None):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    if salt:
        body["cache_salt"] = salt
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), HEADERS)
    t0 = time.perf_counter()
    first, text, usage = None, [], None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            usage = d.get("usage") or usage
            for ch in d.get("choices", []):
                piece = ch.get("delta", {}).get("content")
                if piece:
                    first = first if first is not None else time.perf_counter() - t0
                    text.append(piece)
    return "".join(text), first, (usage or {}).get("prompt_tokens")


messages = [{"role": "user", "content": log + "\n\n" + questions[0]}]
def common_prefix(a, b):
    return next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


ok = True
for turn in range(turns):
    cached, ttft, prompt_tokens = chat(messages)
    fresh, ttft_fresh, _ = chat(messages, salt=f"nocache-{time.time()}")
    fresh2, _, _ = chat(messages, salt=f"nocache2-{time.time()}")
    print(json.dumps({"turn": turn + 1, "prompt_tokens": prompt_tokens, "ttft_s": round(ttft, 2),
                      "ttft_no_cache_s": round(ttft_fresh, 2), "cached_eq_fresh": cached == fresh,
                      "fresh_eq_fresh": fresh == fresh2, "common_chars_cached_fresh": common_prefix(cached, fresh),
                      "common_chars_fresh_fresh": common_prefix(fresh, fresh2), "length": len(fresh),
                      "answer": cached[:60]}), flush=True)
    ok &= cached == fresh or fresh != fresh2
    messages += [{"role": "assistant", "content": cached},
                 {"role": "user", "content": questions[(turn + 1) % len(questions)]}]
print("prefix check:", "consistent" if ok else "cached answer differs where full prefills agree")
