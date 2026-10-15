import json, os, sys, time, urllib.request

# Streaming chat request; prints TTFT, decode tok/s (from usage), total time.
# Usage: python3 bench.py <tag> [max_tokens] [prompt_id]
PROMPTS = [
    "用 Python 实现一个线程安全的 LRU 缓存类，支持 get/put/过期时间，并写出完整的 pytest 单元测试。",
    "解释 TCP 三次握手和四次挥手的过程，说明 TIME_WAIT 状态存在的原因，并给出 Linux 下相关内核参数的调优建议。",
    "写一个 Go 语言的 HTTP 服务，提供 /users 的增删改查 REST 接口，使用 sync.Map 存储，并包含优雅关闭。",
]
tag = sys.argv[1]
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 512
pid = int(sys.argv[3]) if len(sys.argv) > 3 else 0
body = {
    "model": os.environ.get("MODEL", "qwen3.8-27b-fp8-mtp"),
    "messages": [{"role": "user", "content": PROMPTS[pid]}],
    "max_tokens": max_tokens,
    "stream": True,
    "stream_options": {"include_usage": True},
    "chat_template_kwargs": {"enable_thinking": False},
}
req = urllib.request.Request(os.environ.get("BASE_URL", "http://127.0.0.1:8000") + "/v1/chat/completions", json.dumps(body).encode(),
                             dict({"Content-Type": "application/json"}, **({"Authorization": "Bearer " + os.environ["API_KEY"]} if os.environ.get("API_KEY") else {})))
t0 = time.perf_counter()
first = last = None
usage = None
with urllib.request.urlopen(req, timeout=600) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        if d.get("usage"):
            usage = d["usage"]
        for ch in d.get("choices", []):
            if ch.get("delta", {}).get("content"):
                now = time.perf_counter() - t0
                first = first if first is not None else now
                last = now
n = usage["completion_tokens"] if usage else 0
dec = (n - 1) / (last - first) if first is not None and last > first else 0
print(json.dumps({"tag": tag, "prompt": pid, "completion_tokens": n, "ttft_s": round(first or 0, 3),
                  "total_s": round(time.perf_counter() - t0, 3), "decode_tok_s": round(dec, 2)}))
