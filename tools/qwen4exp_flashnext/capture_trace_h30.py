#!/usr/bin/env python
"""Capture one short device trace from a live server started with --profiler-config.

The hardware trace is expensive (tools/README.md), so each capture stays small:

    capture_trace_h30.py decode   # start and stop around a few decode steps of a running request
    capture_trace_h30.py prefill  # start just before a ~2K prompt, stop at its first token

The request runs in a thread and the profiler window is opened from the main thread.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import bench_util_h30 as B  # noqa: E402


def post(path: str, timeout: float = 1800.0) -> None:
    req = urllib.request.Request(B.BASE + path, data=b"", method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        print(path, resp.status, f"{time.time() - t0:.1f}s", flush=True)


def stream_into(prompt: str, n: int, stamps: list[float]) -> None:
    body = json.dumps({
        "model": B.MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        B.BASE + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for line in resp:
            if line.startswith(b"data:") and b"[DONE]" not in line:
                stamps.append(time.time())


def main() -> int:
    mode = sys.argv[1]
    stamps: list[float] = []
    # A warm request first: the profiler's first capture carries startup cost on the host side.
    warm: list[float] = []
    stream_into(B.CASES["short"][0], 16, warm)
    if mode == "decode":
        worker = threading.Thread(target=stream_into, args=(B.CASES["short"][0], 400, stamps))
        worker.start()
        while len(stamps) < 40:
            time.sleep(0.005)
        post("/start_profile")
        began = len(stamps)
        while len(stamps) < began + 6:
            time.sleep(0.002)
        post("/stop_profile")
        worker.join()
    elif mode in ("prefill", "prefill4k"):
        prompt = B.CASES["p2000"][0] if mode == "prefill" else B.PASSAGE * 128 + "Summarize the above."
        post("/start_profile")
        worker = threading.Thread(target=stream_into, args=(prompt, 2, stamps))
        worker.start()
        while not stamps:
            time.sleep(0.005)
        post("/stop_profile")
        worker.join()
    else:
        raise SystemExit(__doc__)
    print("tokens", len(stamps), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
