# SPDX-License-Identifier: MIT
"""Export one H3 head/query tile for the independent SRAM correctness runner.

This script uses CPU tensors only. Its FP32 reference applies scale after QK,
as in standard attention; the device path receives BF16 pre-scaled queries.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("dump", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--head", type=int, default=0)
p.add_argument("--query-start", type=int, default=0)
args = p.parse_args()
torch.set_num_threads(8)
d = torch.load(args.dump, map_location="cpu", weights_only=True, mmap=True)
q = d["q"][0, args.query_start:args.query_start + 128, args.head].contiguous()
k, v = (d[name][0, :, args.head].contiguous() for name in ("k", "v"))
if q.shape != (128, 128) or k.shape != v.shape or k.shape[1] != 128:
    raise ValueError("Expected a full 128-row tile and head dimension 128")
if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
    raise ValueError("Expected BF16 Q/K/V")
scale = float(d["scale"]) if d.get("scale") is not None else 128 ** -0.5
reference = torch.softmax((q.float() @ k.float().T) * scale, dim=-1) @ v.float()
valid = k.shape[0]
pad = -valid % 128
kp = torch.nn.functional.pad(k, (0, 0, 0, pad))
vp = torch.nn.functional.pad(v, (0, 0, 0, pad))
qs = q * scale
bounds = qs.float().norm(dim=-1) * k.float().norm(dim=-1).max()
with args.output.open("wb") as f:
    f.write(b"G2ATTN1\0")
    f.write(struct.pack("<4I", valid + pad, valid, 128, 128))
    for x in (qs, kp, vp):
        f.write(x.contiguous().view(torch.uint16).numpy().tobytes())
    f.write(reference.contiguous().numpy().tobytes())
metadata = {"source": str(args.dump), "head": args.head, "query_start": args.query_start,
            "keys": valid + pad, "valid_keys": valid, "scale": scale,
            "bound_min": bounds.min().item(), "bound_max": bounds.max().item(),
            "bytes": args.output.stat().st_size,
            "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata), flush=True)
