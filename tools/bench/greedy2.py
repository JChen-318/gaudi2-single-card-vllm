import json, sys, threading, time, urllib.request

# Greedy (temperature=0) outputs over short, long-context and 2-way concurrent requests, to compare numerics
# before/after a change. Usage: python3 greedy2.py <out.json> [max_tokens] [ctx sizes, comma separated]
BASE = "http://127.0.0.1:8000"
SHORT = [
    "用 Python 实现一个线程安全的 LRU 缓存类，支持 get/put/过期时间，并写出完整的 pytest 单元测试。",
    "解释 TCP 三次握手和四次挥手的过程，说明 TIME_WAIT 状态存在的原因，并给出 Linux 下相关内核参数的调优建议。",
    "写一个 Go 语言的 HTTP 服务，提供 /users 的增删改查 REST 接口，使用 sync.Map 存储，并包含优雅关闭。",
]


def ctx_prompt(approx):
    lines, i = [], 0
    while sum(len(x) for x in lines) < approx * 3.6:
        i += 1
        lines.append(f"[2026-10-{1 + i % 28:02d} {i % 24:02d}:{(i * 7) % 60:02d}:{(i * 13) % 60:02d}] worker-{i % 17} "
                     f"job {1000 + i} stage {('fetch', 'parse', 'index', 'merge', 'flush')[i % 5]} took {(i * 37) % 911} ms, "
                     f"queue depth {(i * 5) % 43}, status {('ok', 'ok', 'retry', 'ok', 'slow')[i % 5]}")
    return ("Below is a service log.\n\n" + "\n".join(lines) +
            "\n\nWrite a detailed incident-style report in English about what this log shows: workload pattern, slow "
            "stages, retries, and recommendations. Use several paragraphs.")


def ask(prompt, max_tokens):
    body = {"model": "qwen3.8-27b-fp8-mtp", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    return {"text": d["choices"][0]["message"]["content"], "tokens": d["usage"]["completion_tokens"],
            "prompt_tokens": d["usage"]["prompt_tokens"], "finish": d["choices"][0]["finish_reason"],
            "seconds": round(time.perf_counter() - t0, 1)}


out, max_tokens = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 200
sizes = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [300, 4000, 16000]
res = {}
for i, p in enumerate(SHORT):
    res[f"short{i}"] = ask(p, max_tokens)
for n in sizes:
    res[f"ctx{n}"] = ask(ctx_prompt(n), max_tokens)


def pair(tag, prompts):
    got = [None] * len(prompts)
    def run(i):
        got[i] = ask(prompts[i], max_tokens)
    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(prompts))]
    [t.start() for t in threads]; [t.join() for t in threads]
    for i, g in enumerate(got):
        res[f"{tag}{i}"] = g


pair("pair_short", SHORT[:2])
pair("pair_ctx", [ctx_prompt(sizes[min(1, len(sizes) - 1)]), ctx_prompt(sizes[0])])
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
for k, r in res.items():
    print(f"{k:12} prompt {r['prompt_tokens']:6d} out {r['tokens']:4d} {r['finish']:7} {r['seconds']:6.1f}s  {r['text'][:60]!r}")
