#!/usr/bin/env python
"""Long-context speed on non-repetitive text (H31).

bench_util_h30.py repeats one sentence, so its prefill routes every chunk to a few experts and its
recall is trivial. Real prompts spread over all 512 experts. This benchmark builds prompts of a
target token length from repository documents and source files (fixed order, so runs compare), then
streams a fixed number of completion tokens (ignore_eos) and reports:

    ttft_ms, prefill tok/s, TPOT median, and every inter-token gap above --stall-ms with its position

    real_long_h31.py --sizes 2000,8000,16000,30000 --decode 256 --out result.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import bench_util_h30 as B  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "docs/OPTIMIZATION.md", "docs/CONFIG.md", "docs/INSTALL.md", "docs/SNAPSHOT.md", "docs/DOCKER.md",
    "vllm-gaudi/vllm_gaudi/models/qwen4exp/qsa.py", "vllm-gaudi/vllm_gaudi/models/qwen4exp/packed_moe.py",
    "vllm-gaudi/vllm_gaudi/models/qwen4exp/model.py", "vllm-gaudi/vllm_gaudi/models/qwen3_5.py",
    "vllm-gaudi/vllm_gaudi/extension/ops.py", "vllm-gaudi/vllm_gaudi/v1/worker/hpu_model_runner.py",
)


def corpus() -> str:
    # The files as committed at HEAD, so editing them in the working tree does not change the prompts.
    import subprocess

    parts = []
    for rel in SOURCES:
        try:
            text = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, check=True).stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Not a git checkout (an exported tree): read the file as it is on disk.
            path = ROOT / rel
            if not path.is_file():
                continue
            text = path.read_bytes()
        parts.append(f"\n\n===== {rel} =====\n" + text.decode(errors="replace"))
    if not parts:
        # The callers grow the corpus by doubling it until it is long enough: an empty one never gets there.
        raise SystemExit(f"no corpus: none of {len(SOURCES)} source files found under {ROOT}")
    return "".join(parts)


def tokens(text: str) -> int:
    body = json.dumps({"model": B.MODEL, "prompt": text}).encode()
    req = urllib.request.Request(B.BASE + "/tokenize", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return int(json.load(resp)["count"])


def prompt_of(text: str, target: int) -> str:
    if target <= 64:
        return "Write a long detailed essay about the history of the Roman Empire."
    # Characters per token is roughly constant over the corpus; two corrections land within 1%.
    n = target * 3
    for _ in range(3):
        got = tokens(text[:n])
        n = int(n * target / max(got, 1))
    return text[:n]


def run(prompt: str, decode: int, sampled: bool = False) -> list[float]:
    request = {
        "model": B.MODEL,
        "messages": [{"role": "user", "content": prompt + "\n\nSummarize the material above."}],
        "max_tokens": decode, "ignore_eos": True, "stream": True,
    }
    if not sampled:
        # --sampled leaves temperature out: the server then uses generation_config.json
        # (temperature 1.0, top_k 20, top_p 0.95), which is what chat clients get.
        request["temperature"] = 0.0
    body = json.dumps(request).encode()
    req = urllib.request.Request(B.BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    stamps = [time.time()]
    text = []
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for line in resp:
            if line.startswith(b"data:") and b"[DONE]" not in line and b'"content"' in line:
                stamps.append(time.time())
                try:
                    delta = json.loads(line[5:])["choices"][0]["delta"].get("content") or ""
                except (ValueError, KeyError, IndexError):
                    delta = ""
                text.append(delta)
    LAST_TEXT[:] = ["".join(text)]
    return stamps


LAST_TEXT: list[str] = [""]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="2000,8000,16000,30000")
    ap.add_argument("--decode", type=int, default=256)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--stall-ms", type=float, default=150.0)
    ap.add_argument("--skip", type=int, default=0, help="start this many characters into the corpus (fresh text)")
    ap.add_argument("--sampled", action="store_true", help="server-default sampling instead of greedy")
    ap.add_argument("--out")
    ap.add_argument("--write-golden", help="store each greedy completion (first rep) here")
    ap.add_argument("--golden", help="compare each greedy completion with this file")
    args = ap.parse_args()
    text = corpus()[args.skip:]
    rows = []
    golden_out: dict[str, str] = {}
    golden_in = json.loads(Path(args.golden).read_text()) if args.golden else {}
    for size in (int(s) for s in args.sizes.split(",")):
        prompt = prompt_of(text, size)
        n_prompt = tokens(prompt)
        for rep in range(args.reps):
            stamps = run(prompt, args.decode, args.sampled)
            gaps = [(b - a) * 1000 for a, b in zip(stamps[1:], stamps[2:])]
            ttft = (stamps[1] - stamps[0]) * 1000
            row = {"size": size, "prompt_tokens": n_prompt, "rep": rep, "ttft_ms": round(ttft),
                   "prefill_tok_s": round(n_prompt / ttft * 1000), "decoded": len(stamps) - 1,
                   "tpot_median_ms": round(statistics.median(gaps), 2) if gaps else None,
                   "tpot_mean_ms": round(statistics.fmean(gaps), 2) if gaps else None,
                   "stalls": [(i + 1, round(g)) for i, g in enumerate(gaps) if g > args.stall_ms]}
            name = f"{size}@{args.skip}"
            if rep == 0:
                golden_out[name] = LAST_TEXT[0]
            if name in golden_in:
                want, got = golden_in[name], LAST_TEXT[0]
                row["golden"] = "same" if want == got else next(
                    (f"differs at char {i}" for i, (a, b) in enumerate(zip(want, got)) if a != b),
                    f"length {len(want)} vs {len(got)}")
            rows.append(row)
            print(json.dumps(row), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
    if args.write_golden:
        Path(args.write_golden).write_text(json.dumps(golden_out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
