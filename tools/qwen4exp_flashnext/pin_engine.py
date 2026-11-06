#!/usr/bin/env python
"""Pin a running vLLM engine's Habana pipeline threads to one CCD.

    pin_engine.py pin      # main/lowering/execute get their own physical cores on CCD1
    pin_engine.py ccd1     # the whole process on CCD1, scheduler free inside it
    pin_engine.py free     # back to 0-31

Thread roles come from /proc comm names, and the busiest thread of each role is the one that gets
a dedicated core. Run it after the first request so the pipeline threads have accumulated CPU time.
"""

from __future__ import annotations

import os
import subprocess
import sys

CCD0 = "0-7,16-23"
CCD1 = "8-15,24-31"


def engine_pid() -> int:
    out = subprocess.check_output(["pgrep", "-f", "VLLM::EngineCore"], text=True).split()
    return int(out[0])


def taskset(tid: str | int, cpus: str) -> None:
    subprocess.run(["taskset", "-cp", cpus, str(tid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    mode = sys.argv[1]
    pid = engine_pid()
    tasks = os.listdir(f"/proc/{pid}/task")
    comm = {}
    for t in tasks:
        try:
            comm[t] = open(f"/proc/{pid}/task/{t}/comm").read().strip()
        except OSError:
            pass

    def busiest(prefix: str) -> str:
        best, most = "", -1
        for t, name in comm.items():
            if name.startswith(prefix):
                try:
                    used = int(open(f"/proc/{pid}/task/{t}/schedstat").read().split()[0])
                except OSError:
                    continue
                if used > most:
                    best, most = t, used
        return best

    if mode == "free":
        for t in comm:
            taskset(t, "0-31")
    elif mode == "ccd1":
        for t in comm:
            taskset(t, CCD1)
    elif mode == "pin":
        spin = busiest("AccThread")
        low, exe = busiest("Pipeline Lower"), busiest("Pipeline Exec")
        for t in comm:
            taskset(t, "11-15,27-31")
        taskset(spin, CCD0)
        taskset(pid, "8")
        taskset(low, "9")
        taskset(exe, "10")
        print(f"pid={pid} main->8 lowering {low}->9 execute {exe}->10 spinner {spin}->CCD0 rest->11-15,27-31")
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
