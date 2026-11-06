#!/usr/bin/env python
"""Restart the Qwen4Exp server per variant and measure it, with the temp-0 output gate.

    sweep_h30.py --out DIR --golden golden.json --variants variants.json

variants.json is a list of {"name": str, "env": {...}, "preload": bool, "args": [...]}. Each variant:
stop the running server, start it with the production command plus the variant's env, warm it,
pin the threads to one CCD, measure, and append one JSON line to DIR/sweep.jsonl.
A variant that fails to start is recorded and the sweep moves on. The last variant in the file
is left running.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_util_h30 as B  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
TCMALLOC = "/lib/x86_64-linux-gnu/libtcmalloc.so.4"
BASE_ENV = {
    "PATH": "/usr/bin/habanatools:" + os.environ["PATH"],
    "GC_KERNEL_PATH": "/usr/lib/habanalabs/libtpc_kernels.so",
    "HABANA_LOGS": "/var/log/habana_logs/",
    "HABANA_SCAL_BIN_PATH": "/opt/habanalabs/engines_fw",
    "HABANA_PLUGINS_LIB_PATH": "/usr/lib/habanatools/habana_plugins",
    "PT_HPU_LAZY_MODE": "0",
    "VLLM_HPU_PREFILL_TILE_REPLAY": "0",
}
ARGS = [
    ".venv/bin/vllm", "serve", B.MODEL,
    "--load-format", "qwen4exp_plain", "--language-model-only", "--dtype", "bfloat16",
    "--max-model-len", "32768", "--max-num-batched-tokens", "2048", "--max-num-seqs", "1",
    "--num-gpu-blocks", "768", "--no-enable-prefix-caching", "--enforce-eager",
    "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml", "--port", "8000",
]


def server_pids() -> list[int]:
    """API server processes, matched on argv so this script's own command line never matches."""
    found = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            argv = open(f"/proc/{name}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if len(argv) > 2 and argv[1].endswith(b"bin/vllm") and argv[2] == b"serve":
            found.append(int(name))
    return found


def engine_alive() -> bool:
    return subprocess.run(["pgrep", "-x", "VLLM::EngineCor"], capture_output=True).returncode == 0 or any(
        b"VLLM::EngineCore" in open(f"/proc/{n}/cmdline", "rb").read()
        for n in os.listdir("/proc") if n.isdigit() and os.path.exists(f"/proc/{n}/cmdline")
    )


def stop_server() -> None:
    for pid in server_pids():
        os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 90
    while time.time() < deadline and (server_pids() or engine_alive()):
        time.sleep(1)
    if server_pids() or engine_alive():
        raise RuntimeError("server did not stop within 90 s")
    time.sleep(3)


def healthy() -> bool:
    try:
        with urllib.request.urlopen(B.BASE + "/health", timeout=2) as resp:
            return resp.status == 200
    except OSError:
        return False


def start_server(variant: dict, log_path: str) -> None:
    env = {**os.environ, **BASE_ENV, **variant.get("env", {})}
    if variant.get("preload"):
        env["LD_PRELOAD"] = TCMALLOC
    cmd = [".venv/bin/python", *ARGS, *variant.get("args", [])]
    if variant.get("drop_eager"):
        cmd.remove("--enforce-eager")
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    deadline = time.time() + variant.get("start_timeout", 900)
    while time.time() < deadline:
        if healthy():
            return
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        time.sleep(2)
    raise RuntimeError("server did not become healthy")


def measure(variant: dict, golden: dict | None) -> dict:
    sampler = B.Sampler()
    sampler.start()
    time.sleep(0.3)
    if not variant.get("cases", True):
        return {}  # start only: the caller measures first-use behaviour itself
    # The first requests pay shape setup and give the pipeline threads CPU time to identify them.
    for name in ("short", "p2000"):
        B.stream(B.CASES[name][0], 12)
    subprocess.run([sys.executable, os.path.join(HERE, "pin_engine.py"), variant.get("layout", "pin")], check=True,
                   stdout=subprocess.DEVNULL)
    B.stream(B.CASES["short"][0], 12)  # first request after re-pinning is slow once
    out: dict = {}
    for name in variant.get("cases", ["short", "p2000", "p8000"]):
        rows = [B.measure(name, *B.CASES[name], sampler) for _ in range(variant.get("reps", 3))]
        tp = sorted(r["tpot_p50_ms"] for r in rows)
        tt = sorted(r["ttft_ms"] for r in rows)
        diffs = [B.first_diff(golden[name], r["text"]) for r in rows] if golden and name in golden else []
        out[name] = {
            "tpot_ms": tp[len(tp) // 2], "tpot_all": tp, "ttft_ms": tt[len(tt) // 2],
            "prefill_tok_s": round(rows[0]["prompt_tokens"] / (tt[len(tt) // 2] / 1e3), 1),
            "golden_diff": diffs, "decode_card": rows[-1]["decode_card"], "prefill_card": rows[-1]["prefill_card"],
        }
    sampler.done.set()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--golden")
    ap.add_argument("--variants", required=True)
    args = ap.parse_args()
    variants = json.load(open(args.variants))
    golden = json.load(open(args.golden)) if args.golden else None
    os.makedirs(args.out, exist_ok=True)
    for variant in variants:
        name = variant["name"]
        row: dict = {"name": name, "env": variant.get("env", {}), "preload": bool(variant.get("preload"))}
        t0 = time.time()
        try:
            stop_server()
            start_server(variant, os.path.join(args.out, f"serve_{name}.log"))
            row["start_s"] = round(time.time() - t0, 1)
            row["result"] = measure(variant, golden)
        except Exception as exc:  # noqa: BLE001 - one bad variant must not end the sweep
            row["error"] = repr(exc)[:400]
        with open(os.path.join(args.out, "sweep.jsonl"), "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "env"})[:700], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
