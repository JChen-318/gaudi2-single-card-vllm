# SPDX-License-Identifier: MIT
"""CPU-only export of distinct consecutive query tiles, shared K/V and FP32 reference."""
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
p.add_argument("--blocks", type=int, default=16)
p.add_argument("--full-head", action="store_true", help="All queries, padded to a multiple of --blocks tiles")
a = p.parse_args()
torch.set_num_threads(8)
d = torch.load(a.dump, map_location="cpu", weights_only=True, mmap=True)
q = d["q"][0, a.query_start:(None if a.full_head else a.query_start + a.blocks * 128), a.head].contiguous()
valid_queries = len(q)
group_blocks = a.blocks
if a.full_head:
    if a.query_start or not 2 <= group_blocks <= 24:
        raise ValueError("Full head needs query-start=0 and group size 2..24")
    padded_rows = (len(q) + group_blocks * 128 - 1) // (group_blocks * 128) * (group_blocks * 128)
    q = torch.nn.functional.pad(q, (0, 0, 0, padded_rows - len(q)))
    a.blocks = padded_rows // 128
k, v = (d[n][0, :, a.head].contiguous() for n in ("k", "v"))
if not 2 <= a.blocks <= 320 or q.shape != (a.blocks * 128, 128) or k.shape != v.shape:
    raise ValueError("Expected full 128-row tiles, shared K/V and head dimension 128")
assert q.dtype == k.dtype == v.dtype == torch.bfloat16 and k.shape[1] == 128
scale = float(d["scale"]) if d.get("scale") is not None else 128 ** -0.5
qs = q * scale
bound = qs.float().norm(dim=-1) * k.float().norm(dim=-1).max()
if not torch.isfinite(bound).all() or bound.max() > 80:
    raise ValueError(f"Unsupported bound: {bound.max().item()}")
refs = []
for b in range(a.blocks):
    scores = (q[b * 128:(b + 1) * 128].float() @ k.float().T) * scale
    refs.append(torch.softmax(scores, -1) @ v.float())
reference = torch.cat(refs)
valid = len(k)
pad = -valid % 128
kp, vp = (torch.nn.functional.pad(t, (0, 0, 0, pad)) for t in (k, v))
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open("wb") as f:
    f.write(struct.pack("<8s6I", b"G2BATCH1", valid + pad, valid, 128, a.blocks, len(q), valid_queries if a.full_head else 0))
    for t in (qs, kp, vp):
        f.write(t.view(torch.uint16).numpy().tobytes())
    f.write(reference.numpy().tobytes())
meta = dict(source=str(a.dump), head=a.head, query_start=a.query_start, blocks=a.blocks,
            full_head=a.full_head, valid_queries=valid_queries, group_blocks=group_blocks,
            keys=valid + pad, valid_keys=valid, scale=scale, bound_max=bound.max().item(),
            bytes=a.output.stat().st_size, sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())
a.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta), flush=True)
