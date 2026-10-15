"""Long-context retrieval check: hide a code in a long log-like text and ask for it (temperature 0).
python3 needle.py <approx_tokens> [positions...]   positions are fractions of the text, default 0.05 0.5 0.95"""
import json
import os
import random
import sys
import time
import urllib.request

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("MODEL", "qwen3.8-27b-fp8-mtp")
approx = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
positions = [float(x) for x in sys.argv[2:]] or [0.05, 0.5, 0.95]
rng = random.Random(7)
services = ["auth", "billing", "search", "gateway", "storage", "scheduler", "mailer", "cache"]
levels = ["INFO", "DEBUG", "WARN"]


def line(i):
    return (f"2026-10-14 0{rng.randint(0, 9)}:{rng.randint(10, 59)}:{rng.randint(10, 59)} {rng.choice(levels)} "
            f"[{rng.choice(services)}] request {rng.randint(100000, 999999)} handled in {rng.randint(1, 900)} ms, "
            f"queue depth {rng.randint(0, 64)}, worker {rng.randint(1, 32)}\n")


lines = [line(i) for i in range(approx // 40)]  # ~40 tokens per line
ok = 0
for pos in positions:
    code = str(rng.randint(100000, 999999))
    doc = list(lines)
    doc.insert(int(len(doc) * pos), f"NOTE: the maintenance access code for tonight is {code}. Keep it.\n")
    prompt = ("Below is a service log.\n\n" + "".join(doc) +
              "\nQuestion: what is the maintenance access code mentioned in the log? Answer with the number only.")
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 20, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    headers = {"Content-Type": "application/json"}
    if os.environ.get("API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["API_KEY"]
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), headers)
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    ans = r["choices"][0]["message"]["content"].strip()
    hit = code in ans
    ok += hit
    print(json.dumps({"position": pos, "prompt_tokens": r["usage"]["prompt_tokens"], "expected": code,
                      "answer": ans[:40], "correct": hit, "seconds": round(time.time() - t0, 1)}), flush=True)
print(f"needle: {ok}/{len(positions)} correct")
