"""Speed over diverse prompts: python3 bench_multi.py [max_tokens=512]
Single client per prompt; rounds and accepted tokens from /metrics deltas."""
import json
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "qwen3.8-27b-fp8-mtp"
MAX_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 512

ARTICLE = ("人工智能芯片的竞争正在从单纯的算力比拼转向系统级的效率竞争。过去几年，训练大模型主要依赖通用图形处理器，"
           "但随着模型规模扩大，内存带宽、互联带宽和能耗逐渐成为瓶颈。许多厂商开始设计专用加速器，"
           "在矩阵乘法单元之外加入大容量高带宽内存，并通过片间高速互联把成百上千张卡组织成一个训练集群。"
           "与此同时，推理场景对延迟和成本更加敏感：一次对话请求往往只需要很少的计算，却要反复读取全部模型权重，"
           "因此推理更像是一个带宽问题而不是算力问题。为此业界发展出了量化、投机解码、连续批处理、分页注意力等技术。"
           "量化把权重从十六位压缩到八位甚至四位，直接减少了每次读取的数据量；投机解码用一个小模型先猜出若干个词，"
           "再由大模型一次性验证，从而把多次串行的权重读取合并为一次；连续批处理让不同请求共享同一次权重读取；"
           "分页注意力则让长上下文的键值缓存可以像操作系统的内存页一样灵活分配。") * 3

CODE_SNIPPET = '''def find_duplicates(items):
    result = []
    for i in range(len(items)):
        for j in range(len(items)):
            if i != j and items[i] == items[j] and items[i] not in result:
                result.append(items[i])
    return result
'''

PROMPTS = [
    ("zh_explain", "用通俗的语言解释一下什么是通货膨胀，以及它对普通人生活的影响，举几个例子。", False),
    ("zh_essay", "写一篇六百字左右的散文，主题是秋天的傍晚，要有画面感。", False),
    ("en_essay", "Write a persuasive essay arguing that cities should invest more in public transportation.", False),
    ("en_science", "Explain step by step how mRNA vaccines train the immune system, for a high-school audience.", False),
    ("py_code", "用 Python 实现一个线程安全的 LRU 缓存类，支持 get/put/过期时间，并写出完整的 pytest 单元测试。", False),
    ("go_code", "写一个 Go 语言的 HTTP 服务，提供 /users 的增删改查 REST 接口，使用 sync.Map 存储，并包含优雅关闭。", False),
    ("ts_react", "Write a React component in TypeScript for a sortable, filterable data table with pagination. "
                 "Include the type definitions and a short usage example.", False),
    ("code_review", "解释下面这段代码的作用，指出性能问题，并给出优化后的版本和复杂度分析：\n\n" + CODE_SNIPPET, False),
    ("math", "一个水池有甲、乙两个进水管和一个丙出水管。单开甲 6 小时注满，单开乙 8 小时注满，单开丙 12 小时放空。"
             "三管同时打开，几小时注满？请分步写出推理过程，再推广到一般情形。", False),
    ("translate", "把下面这段话翻译成地道的英文：\n\n" + ARTICLE[:400], False),
    ("summarize", "请用要点形式总结下面这篇文章，并给出三个延伸问题：\n\n" + ARTICLE, False),
    ("json_extract", "从下面的简历中抽取信息，输出 JSON，字段包括 name, email, years_of_experience, skills(数组), "
                     "education(数组，每项含 school, degree, year)：\n\n张伟，邮箱 zhangwei@example.com，"
                     "2015 年毕业于浙江大学计算机系获学士学位，2018 年获清华大学硕士学位。"
                     "先后在两家公司从事后端开发八年，熟悉 Go、Python、Kubernetes、PostgreSQL 和 Kafka。", False),
    ("md_table", "列出 15 种常见的数据结构，用 Markdown 表格给出每种的插入、删除、查找的平均时间复杂度和典型用途。", False),
    ("think_logic", "有三个盒子，分别标着'苹果'、'橘子'、'苹果和橘子'，但所有标签都贴错了。"
                    "你只能从一个盒子里拿出一个水果看，如何确定每个盒子里装的是什么？", True),
    ("repeat", "把下面这段文字原样抄写三遍，每遍前面加上编号：\n\n" + ARTICLE[:300], False),
]


def metric(name):
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    return sum(float(l.split()[-1]) for l in txt.splitlines() if l.startswith(name + "{"))


def run(tag, prompt, thinking, results):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": thinking}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            usage = d.get("usage") or usage
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    now = time.perf_counter() - t0
                    first = first if first is not None else now
                    last = now
    n = usage["completion_tokens"]
    results[tag] = {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": n, "ttft_s": round(first, 3),
                    "decode_tok_s": round((n - 1) / max(last - first, 1e-6), 1), "decode_s": last - first}


def single(tag, prompt, thinking):
    d0 = metric("vllm:spec_decode_num_drafts_total")
    a0 = metric("vllm:spec_decode_num_accepted_tokens_total")
    results = {}
    run(tag, prompt, thinking, results)
    rounds = metric("vllm:spec_decode_num_drafts_total") - d0
    accepted = metric("vllm:spec_decode_num_accepted_tokens_total") - a0
    res = results[tag]
    res["rounds"] = int(rounds)
    res["accepted_per_round"] = round(accepted / max(rounds, 1), 2)
    res["ms_per_round"] = round(res.pop("decode_s") / max(rounds, 1) * 1000, 1)
    print(json.dumps({"tag": tag, **res}, ensure_ascii=False), flush=True)
    return res


def pair(tag_a, tag_b):
    prompts = {t: (p, th) for t, p, th in PROMPTS}
    results = {}
    d0 = metric("vllm:spec_decode_num_drafts_total")
    threads = [threading.Thread(target=run, args=(t, prompts[t][0], prompts[t][1], results)) for t in (tag_a, tag_b)]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0
    rounds = metric("vllm:spec_decode_num_drafts_total") - d0
    total = sum(r["completion_tokens"] for r in results.values())
    print(json.dumps({"pair": f"{tag_a}+{tag_b}", "wall_s": round(wall, 2), "total_tokens": total,
                      "total_tok_s": round(total / wall, 1),
                      "per_request_tok_s": [results[t]["decode_tok_s"] for t in (tag_a, tag_b)],
                      "ms_per_round_est": round(wall / max(rounds / 2, 1) * 1000, 1)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    single("warmup", PROMPTS[0][1], False)
    rows = [(tag, single(tag, prompt, thinking)) for tag, prompt, thinking in PROMPTS]
    total_tokens = sum(r["completion_tokens"] for _, r in rows)
    total_rounds = sum(r["rounds"] for _, r in rows)
    print(json.dumps({"summary_single": {
        "prompts": len(rows), "median_decode_tok_s": sorted(r["decode_tok_s"] for _, r in rows)[len(rows) // 2],
        "mean_accepted_per_round": round(sum(r["accepted_per_round"] for _, r in rows) / len(rows), 2),
        "median_ms_per_round": sorted(r["ms_per_round"] for _, r in rows)[len(rows) // 2],
        "tokens_per_round_overall": round(total_tokens / max(total_rounds, 1), 2)}}, ensure_ascii=False), flush=True)
    for a, b in (("py_code", "zh_essay"), ("en_essay", "summarize"), ("ts_react", "math")):
        pair(a, b)
