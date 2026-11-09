# SPDX-License-Identifier: MIT
"""Fresh-process sanity check: HPU special-function ops vs CPU (absolute, not before/after).

Prints max relative error per op; exits non-zero if any op is far off BF16/FP32 accuracy.
Optional --dump PATH saves the HPU outputs for bitwise comparison across processes.
"""
import argparse
import json
import sys

import torch
import habana_frameworks.torch.core  # noqa: F401

p = argparse.ArgumentParser()
p.add_argument("--dump")
p.add_argument("--compare")
args = p.parse_args()
torch.manual_seed(5)
xs = {"bf16": (torch.randn(64, 1024) * 3).bfloat16(), "f32": torch.rand(64, 1024) * 4 + 0.25}
ops = {"exp": torch.exp, "tanh": torch.tanh, "sigmoid": torch.sigmoid, "erf": torch.erf,
       "softmax": lambda t: torch.softmax(t, dim=-1), "rsqrt": torch.rsqrt, "log": torch.log,
       "gelu": torch.nn.functional.gelu}
result, outputs, bad = {}, {}, False
for kind, x in xs.items():
    xh = x.to("hpu")
    for name, f in ops.items():
        if kind == "bf16" and name in ("rsqrt", "log"):
            continue
        ref = f(x.double()).float()
        out = f(xh).cpu()
        outputs[f"{kind}-{name}"] = out
        err = ((out.float() - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()
        limit = 0.02 if kind == "bf16" else 1e-3
        result[f"{kind}-{name}"] = err
        bad |= not (err < limit)
print("SPECIAL_OPS " + json.dumps(result))
if args.dump:
    torch.save(outputs, args.dump)
if args.compare:
    old = torch.load(args.compare)
    same = {k: torch.allclose(old[k], v, rtol=0, atol=0, equal_nan=True) for k, v in outputs.items()}
    print("SPECIAL_OPS_BITWISE_SAME " + json.dumps(same))
    bad |= not all(same.values())
sys.exit(1 if bad else 0)
