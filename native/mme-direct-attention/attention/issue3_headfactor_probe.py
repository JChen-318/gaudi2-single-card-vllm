# SPDX-License-Identifier: MIT
"""Does sdpa_recomp_fwd (FusedSDPA with stats) compile for head counts not divisible by PT_HPU_SDPA_HEAD_FACTOR?
Each case in its own subprocess so one compile failure does not end the probe."""
import os
import subprocess
import sys

if len(sys.argv) > 1:
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    h, n = int(sys.argv[1]), int(sys.argv[2])
    q = torch.randn(1, h, n, 128, dtype=torch.bfloat16).to("hpu")
    out = torch.ops.hpu.sdpa_recomp_fwd(q, q, q, None, 0.0, 128 ** -.5, False, True, "fast", None, "right")[0]
    print("OK", h, n, float(out.float().abs().mean().cpu()))
    sys.exit(0)
for factor in ("2", "1"):
    for h in (1, 2, 3):
        for n in (37632, 118):
            env = dict(os.environ, PT_HPU_SDPA_HEAD_FACTOR=factor)
            r = subprocess.run([sys.executable, __file__, str(h), str(n)], env=env, capture_output=True, text=True, timeout=600)
            ok = "OK" in r.stdout
            print(f"HEADFACTOR factor={factor} heads={h} keys={n} -> {'ok' if ok else 'FAIL ' + (r.stderr.strip().splitlines() or ['?'])[-1][:120]}", flush=True)
