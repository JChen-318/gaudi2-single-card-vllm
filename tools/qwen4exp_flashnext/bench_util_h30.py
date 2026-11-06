#!/usr/bin/env python
"""H30 live-server benchmark: decode/prefill speed with card utilization and an output gate.

Every run measures the same fixed prompts and attributes hl-smi samples to the phase they fell in.
``--golden FILE`` compares each temp-0 completion with a stored reference and reports the first
differing character; ``--write-golden`` stores the current output as that reference.

    .venv/bin/python tools/qwen4exp_flashnext/bench_util_h30.py --tag baseline \
        --out tools/qwen4exp_flashnext/bench_h30_baseline.json --write-golden /path/golden.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = os.environ.get("QWEN4EXP_MODEL", os.path.expanduser("~/models/qwen4exp-flashnext-plain"))

PASSAGE = (
    "The quick survey of Mediterranean trade routes, grain supply, taxation and provincial "
    "administration shows how logistics shaped imperial politics over many centuries of expansion "
    "and decline. "
)
#: name -> (prompt, completion tokens). The repeat counts give about 7 / 500 / 2,000 / 8,000 tokens.
CASES = {
    "short": ("Write a long detailed essay about the history of the Roman Empire.", 128),
    "p500": (PASSAGE * 16 + "Summarize the above in one sentence.", 64),
    "p2000": (PASSAGE * 64 + "Summarize the above in one sentence.", 64),
    "p8000": (PASSAGE * 256 + "Summarize the above in one sentence.", 32),
}


class Sampler(threading.Thread):
    """Polls hl-smi; ``window`` returns the mean and peak of the samples inside a time range."""

    def __init__(self, period: float = 0.1) -> None:
        super().__init__(daemon=True)
        self.period = period
        self.rows: list[tuple[float, float, float]] = []
        self.done = threading.Event()

    def run(self) -> None:
        cmd = ["hl-smi", "-Q", "power.draw,utilization.aip", "-f", "csv,noheader,nounits"]
        while not self.done.is_set():
            t = time.time()
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
                watts, aip = (float(x) for x in out.split("\n")[0].split(",")[:2])
                self.rows.append((t, watts, aip))
            except (ValueError, subprocess.SubprocessError):
                pass
            time.sleep(self.period)

    def window(self, t0: float, t1: float) -> dict[str, float | int | None]:
        rows = [r for r in self.rows if t0 <= r[0] <= t1]
        if not rows:
            return {"n": 0, "watts_mean": None, "watts_peak": None, "aip_mean": None, "aip_peak": None}
        return {
            "n": len(rows),
            "watts_mean": round(statistics.mean(r[1] for r in rows), 1),
            "watts_peak": max(r[1] for r in rows),
            "aip_mean": round(statistics.mean(r[2] for r in rows), 1),
            "aip_peak": max(r[2] for r in rows),
        }


def stream(prompt: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    stamps: list[float] = []
    text: list[str] = []
    usage: dict = {}
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            if not raw.startswith(b"data:") or b"[DONE]" in raw:
                continue
            chunk = json.loads(raw[5:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                if piece:
                    stamps.append(time.time())
                    text.append(piece)
    return {"t0": t0, "stamps": stamps, "text": "".join(text), "usage": usage}


def measure(name: str, prompt: str, n: int, sampler: Sampler) -> dict:
    r = stream(prompt, n)
    stamps = r["stamps"]
    gaps = [(b - a) * 1e3 for a, b in zip(stamps, stamps[1:])][8:]
    prompt_tokens = r["usage"].get("prompt_tokens")
    ttft = stamps[0] - r["t0"]
    return {
        "case": name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": r["usage"].get("completion_tokens"),
        "ttft_ms": round(ttft * 1e3, 1),
        "prefill_tok_s": round(prompt_tokens / ttft, 1) if prompt_tokens else None,
        "tpot_p50_ms": round(statistics.median(gaps), 2) if gaps else None,
        "decode_tok_s": round(1e3 / statistics.median(gaps), 2) if gaps else None,
        "prefill_card": sampler.window(r["t0"], stamps[0]),
        "decode_card": sampler.window(stamps[0], stamps[-1]),
        "text": r["text"],
    }


def first_diff(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default="short,p500,p2000,p8000")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--golden")
    ap.add_argument("--write-golden")
    args = ap.parse_args()

    sampler = Sampler()
    sampler.start()
    time.sleep(0.5)
    rows: list[dict] = []
    for rep in range(args.reps):
        for name in args.cases.split(","):
            prompt, n = CASES[name]
            row = measure(name, prompt, n, sampler)
            row["rep"] = rep
            rows.append(row)
            shown = {k: v for k, v in row.items() if k != "text"}
            print(json.dumps(shown), flush=True)
    sampler.done.set()

    rc = 0
    if args.write_golden:
        golden = {r["case"]: r["text"] for r in rows if r["rep"] == 0}
        with open(args.write_golden, "w") as f:
            json.dump(golden, f, ensure_ascii=False, indent=1)
    if args.golden:
        with open(args.golden) as f:
            golden = json.load(f)
        for r in rows:
            ref = golden.get(r["case"])
            if ref is None:
                continue
            d = first_diff(ref, r["text"])
            r["golden_first_diff"] = d
            if d >= 0:
                rc = 1
                print(f"GOLDEN MISMATCH {r['case']} rep={r['rep']} at char {d}: "
                      f"ref={ref[max(0, d - 20):d + 20]!r} got={r['text'][max(0, d - 20):d + 20]!r}")
        print("golden:", "PASS" if rc == 0 else "FAIL")

    summary: dict[str, dict] = {}
    for name in args.cases.split(","):
        sel = [r for r in rows if r["case"] == name]
        tp = [r["tpot_p50_ms"] for r in sel if r["tpot_p50_ms"]]
        summary[name] = {
            "prompt_tokens": sel[0]["prompt_tokens"],
            "ttft_ms": round(statistics.median(r["ttft_ms"] for r in sel), 1),
            "prefill_tok_s": sel[0]["prefill_tok_s"],
            "tpot_p50_ms": round(statistics.median(tp), 2) if tp else None,
            "tpot_all": tp,
            "prefill_card": sel[-1]["prefill_card"],
            "decode_card": sel[-1]["decode_card"],
        }
    with open(args.out, "w") as f:
        json.dump({"tag": args.tag, "summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=1))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
