# SPDX-License-Identifier: Apache-2.0
"""One-time repack of the routed experts into a single flat file.

Booting from the safetensors checkpoint walks 73,728 expert tensors
(512 experts x 3 projections x 48 layers) through Python, which dominates
the ~14-minute load. The pack stores the same packed bytes in a fixed
layout so a boot reads 144 contiguous slabs and issues one H2D copy per
slab.

File layout (little-endian, fixed order):

  header: 512-byte JSON block (NUL padded), starts with ``STRATA-Q4E-PACK``
  data:   for layer in 0..47, for kind in (gate_proj, up_proj, down_proj):
            packed  [E, N, K/2] U8   (checkpoint bytes verbatim)
            scales  [E, N, K/16] U8  (checkpoint bytes verbatim)
            scale2  [E] F32          (per-expert scalar from weight_scale_2)

Run:
  python tools/qwen4exp_flashnext/pack_experts.py \
      --checkpoint ~/models/qwen4exp-flashnext-plain
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vllm-gaudi"))

MAGIC = "STRATA-Q4E-PACK"
HEADER_BYTES = 512
KINDS = ("gate_proj", "up_proj", "down_proj")


def shard_headers(checkpoint: Path):
    for shard in sorted((checkpoint / "tensors").glob("*.safetensors")):
        with open(shard, "rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            yield shard, json.loads(handle.read(size))


def index_tensors(checkpoint: Path) -> dict[str, Path]:
    found = {}
    for shard, header in shard_headers(checkpoint):
        for name in header:
            if name != "__metadata__" and ".mlp.experts." in name:
                found[name] = shard
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default=None, help="default: <checkpoint>/expert-pack.bin")
    ap.add_argument("--verify", type=int, default=8, help="experts to bit-verify per kind")
    args = ap.parse_args()
    checkpoint = Path(args.checkpoint)
    out = Path(args.output) if args.output else checkpoint / "expert-pack.bin"
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing pack: {out}")

    where = index_tensors(checkpoint)
    layers = sorted({int(n.split(".layers.")[1].split(".")[0]) for n in where})
    experts = sorted({int(n.split(".experts.")[1].split(".")[0]) for n in where if ".experts." in n})
    assert layers == list(range(len(layers))), layers
    num_layers, num_experts = len(layers), len(experts)
    sample = where["model.language_model.layers.0.mlp.experts.0.gate_proj.weight"]

    from safetensors import safe_open

    handles: dict[Path, object] = {}

    def tensor(name: str) -> torch.Tensor:
        shard = where[name]
        if shard not in handles:
            handles[shard] = safe_open(shard, framework="pt", device="cpu")
        return handles[shard].get_tensor(name)

    # Geometry from layer 0, verified against every later layer while writing.
    geometry = {}
    for kind in KINDS:
        base = f"model.language_model.layers.0.mlp.experts.0.{kind}.weight"
        packed = tensor(base)
        scales = tensor(base + "_scale")
        geometry[kind] = (tuple(packed.shape), tuple(scales.shape))
    header = {
        "magic": MAGIC,
        "version": 1,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "kinds": list(KINDS),
        "geometry": {k: {"packed": list(p), "scales": list(s)} for k, (p, s) in geometry.items()},
        "sample_shard": sample.name,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    verify_at = sorted(set([0, 1, num_experts // 2, num_experts - 1] + list(range(args.verify))))
    with open(out, "wb") as handle:
        blob = json.dumps(header).encode()
        assert len(blob) <= HEADER_BYTES, "header overflow"
        handle.write(blob + b"\x00" * (HEADER_BYTES - len(blob)))
        for layer in range(num_layers):
            for kind in KINDS:
                packed_shape, scales_shape = geometry[kind]
                pack_buf = torch.empty((num_experts, *packed_shape), dtype=torch.uint8)
                scale_buf = torch.empty((num_experts, *scales_shape), dtype=torch.uint8)
                scale2 = torch.zeros(num_experts, dtype=torch.float32)
                for expert in range(num_experts):
                    base = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{kind}.weight"
                    packed = tensor(base)
                    if tuple(packed.shape) != packed_shape:
                        raise SystemExit(f"{base}: shape {tuple(packed.shape)} != {packed_shape}")
                    plane = tensor(base + "_scale").view(torch.uint8)
                    if tuple(plane.shape) != scales_shape:
                        raise SystemExit(f"{base}_scale: {tuple(plane.shape)} != {scales_shape}")
                    pack_buf[expert] = packed
                    scale_buf[expert] = plane
                    s2 = tensor(base + "_scale_2")
                    scale2[expert] = float(s2.reshape(-1)[0])
                for buf in (pack_buf, scale_buf, scale2):
                    handle.write(buf.contiguous().numpy().tobytes())
                    written += buf.numel() * buf.element_size()
            print(f"  layer {layer + 1}/{num_layers} packed", flush=True)

    # Bit-verify sampled experts against the checkpoint.
    stride = {kind: None for kind in KINDS}

    def slab_offset(layer: int, kind: str) -> int:
        # Fixed order: layer-major, then kind; each slab is pack+scale+scale2.
        sizes = []
        for k in KINDS:
            p, s = geometry[k]
            sizes.append(num_experts * int(torch.Size(p).numel())
                         + num_experts * int(torch.Size(s).numel())
                         + num_experts * 4)
        head = HEADER_BYTES + layer * sum(sizes) + sum(sizes[:KINDS.index(kind)])
        return head

    with open(out, "rb") as handle:
        for kind in KINDS:
            p, s = geometry[kind]
            p_bytes = num_experts * int(torch.Size(p).numel())
            s_bytes = num_experts * int(torch.Size(s).numel())
            for expert in verify_at:
                layer = expert % num_layers
                base = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{kind}.weight"
                want = tensor(base).numpy().tobytes()
                handle.seek(slab_offset(layer, kind) + expert * int(torch.Size(p).numel()))
                got = handle.read(len(want))
                assert got == want, f"packed mismatch at {base}"
                want_scale = tensor(base + "_scale").view(torch.uint8).numpy().tobytes()
                handle.seek(slab_offset(layer, kind) + p_bytes + expert * int(torch.Size(s).numel()))
                got_scale = handle.read(len(want_scale))
                assert got_scale == want_scale, f"scale mismatch at {base}"
    print(f"pack written: {out} ({written / 2**30:.1f} GiB data), verified "
          f"{len(verify_at)} experts/kind bit-exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
