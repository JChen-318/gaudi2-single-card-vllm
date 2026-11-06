# SPDX-License-Identifier: Apache-2.0
"""Rebuild model.safetensors.index.json from the actual tensors in a
converted checkpoint directory.

The CSF->plain-scale conversion renames tensors (``*.weight_scale.
nvfp4_csf_fixed``/``.nvfp4_csf_exceptions`` collapse into a single
``*.weight_scale`` plane), so the source index is stale. This tool scans
every output shard header, rebuilds ``weight_map`` and ``total_size``,
and cross-checks the tensor count against the source index when given.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path


def shard_tensors(path: Path) -> dict[str, tuple[str, int]]:
    """name -> (dtype string, nbytes) from a safetensors file header."""
    with path.open("rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    out: dict[str, tuple[str, int]] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        nelem = 1
        for d in meta["shape"]:
            nelem *= d
        nbytes = nelem * {  # safetensors dtype char widths
            "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
            "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
            "I32": 4, "U32": 4, "F32": 4,
            "I64": 8, "U64": 8, "F64": 8,
        }[meta["dtype"]]
        out[name] = (meta["dtype"], nbytes)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="checkpoint dir with tensors/*.safetensors")
    ap.add_argument("--src-index", help="source index for cross-checking counts")
    args = ap.parse_args()

    root = Path(args.dir).expanduser()
    shards = sorted((root / "tensors").glob("*.safetensors"))
    if not shards:
        print(f"no shards under {root/'tensors'}", file=sys.stderr)
        return 1

    weight_map: dict[str, str] = {}
    total = 0
    dtypes: dict[str, set[str]] = {}
    for shard in shards:
        for name, (dtype, nbytes) in shard_tensors(shard).items():
            if name in weight_map:
                print(f"DUPLICATE tensor {name} in {shard.name}", file=sys.stderr)
                return 1
            weight_map[name] = shard.name
            total += nbytes
            dtypes.setdefault(dtype, set()).add(name)

    index = {
        "metadata": {"total_size": total, "total_parameters": len(weight_map)},
        "weight_map": dict(sorted(weight_map.items())),
    }
    out_path = root / "model.safetensors.index.json"
    out_path.chmod(0o644) if out_path.exists() else None
    out_path.write_text(json.dumps(index, indent=2))
    print(f"wrote {out_path}: {len(weight_map)} tensors, {total/2**30:.2f} GiB")
    for dtype, names in sorted(dtypes.items()):
        print(f"  {dtype}: {len(names)}")

    if args.src_index:
        src = json.loads(Path(args.src_index).read_text())
        src_map = src["weight_map"]
        gone = sorted(
            n for n in src_map
            if n not in weight_map and ("nvfp4_csf" in n or n in weight_map)
        )
        csf_src = [n for n in src_map if "nvfp4_csf" in n]
        missing = sorted(n for n in src_map if n not in weight_map and "nvfp4_csf" not in n)
        print(f"source: {len(src_map)} tensors; csf tensors dropped: {len(csf_src)}")
        print(f"non-csf tensors missing from output: {len(missing)}")
        for n in missing[:10]:
            print(f"  MISSING {n}")
        new_names = [n for n in weight_map if n not in src_map]
        print(f"new plain weight_scale planes: {len(new_names)}")
        if missing:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
