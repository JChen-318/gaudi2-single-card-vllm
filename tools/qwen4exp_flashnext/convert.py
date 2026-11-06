# SPDX-License-Identifier: Apache-2.0
"""Expand CSF-compressed NVFP4 scales of Qwen3.8-Flash-Next-FP4-Local.

Reads the Hugging Face snapshot (or any directory with ``tensors/*.safetensors``),
decodes every ``nvfp4_csf_fixed``/``nvfp4_csf_exceptions`` pair into a plain
``weight_scale`` F8_E4M3 plane (the layout ModelOpt already uses for the MTP
experts), verifies the NVFP4 block-saturation invariant per tensor, and writes
a converted checkpoint with every other tensor passed through byte-identically.

Examples:
  # QA pass only, no output written
  python convert.py --src ~/.cache/huggingface/hub/models--nvidia--*/snapshots/*/ \
      --shards model-00001 --verify-only

  # Full conversion of every shard
  python convert.py --src <snapshot> --out ~/models/qwen4exp-flashnext-plain
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import safe_open, save_file

sys.path.insert(0, str(Path(__file__).parent))
from csf import (  # noqa: E402
    decode_csf_plane,
    exception_window_violations,
    fixed_plane_shape,
    nan_scale_count,
    saturation_ratio,
    unpack_e2m1_codes,
)

CSF_FIXED_SUFFIX = ".weight_scale.nvfp4_csf_fixed"
CSF_EXC_SUFFIX = ".weight_scale.nvfp4_csf_exceptions"


def _to_uint8(t: torch.Tensor) -> np.ndarray:
    if t.dtype == torch.float8_e4m3fn:
        t = t.view(torch.uint8)
    return t.contiguous().numpy()


def find_shards(src: Path, pattern: str | None) -> list[Path]:
    roots = sorted(src.glob("*.safetensors")) + sorted(src.glob("tensors/*.safetensors"))
    if pattern:
        roots = [p for p in roots if pattern in p.name]
    seen: set[Path] = set()
    unique = [p for p in roots if not (p in seen or seen.add(p))]
    return unique


def process_shard(
    shard: Path,
    out_path: Path | None,
    min_saturation: float,
) -> dict:
    stats = {
        "shard": shard.name,
        "csf_tensors": 0,
        "passthrough_tensors": 0,
        "sat_min": 1.0,
        "sat_mean": 0.0,
        "window_violations": 0,
        "nan_scales": 0,
        "errors": [],
        "bytes_out": 0,
        "seconds": 0.0,
    }
    t0 = time.time()
    output: dict[str, torch.Tensor] = {}
    sat_values: list[float] = []

    with safe_open(str(shard), framework="pt", device="cpu") as f:
        names = list(f.keys())
        csf_bases = sorted(n[: -len(CSF_FIXED_SUFFIX)] for n in names if n.endswith(CSF_FIXED_SUFFIX))
        drop = set()
        for base in csf_bases:
            drop.add(base + CSF_FIXED_SUFFIX)
            drop.add(base + CSF_EXC_SUFFIX)

        for base in csf_bases:
            try:
                weight = f.get_tensor(base + ".weight")
                fixed = _to_uint8(f.get_tensor(base + CSF_FIXED_SUFFIX))
                exc = _to_uint8(f.get_tensor(base + CSF_EXC_SUFFIX)).view("<u4")
                scale2 = f.get_tensor(base + ".weight_scale_2")

                rows = weight.shape[0]
                k = weight.shape[1] * 2
                cols = k // 16
                expect = fixed_plane_shape(rows, cols)
                if tuple(fixed.shape) != expect:
                    raise ValueError(f"fixed shape {fixed.shape} != {expect}")

                plane = decode_csf_plane(fixed, exc, rows, cols)
                stats["window_violations"] += exception_window_violations(fixed, exc, rows, cols)
                stats["nan_scales"] += nan_scale_count(plane)

                codes = unpack_e2m1_codes(_to_uint8(weight))
                sat = saturation_ratio(codes)
                sat_values.append(sat)
                stats["sat_min"] = min(stats["sat_min"], sat)
                if sat < min_saturation:
                    stats["errors"].append(f"{base}: saturation {sat:.4f} < {min_saturation}")

                stats["csf_tensors"] += 1
                if out_path is not None:
                    output[base + ".weight"] = weight.contiguous()
                    output[base + ".weight_scale"] = (
                        torch.from_numpy(np.ascontiguousarray(plane)).view(torch.float8_e4m3fn)
                    )
                    output[base + ".weight_scale_2"] = scale2
            except Exception as exc:  # noqa: BLE001
                stats["errors"].append(f"{base}: {type(exc).__name__}: {exc}")

        if out_path is not None:
            for name in names:
                if name in drop or name in output:
                    continue
                output[name] = f.get_tensor(name).contiguous()
                stats["passthrough_tensors"] += 1

    if sat_values:
        stats["sat_mean"] = float(np.mean(sat_values))
    if out_path is not None and not stats["errors"]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(output, str(out_path), metadata={"format": "pt"})
        stats["bytes_out"] = out_path.stat().st_size
        del output
    stats["seconds"] = round(time.time() - t0, 1)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="snapshot dir (contains tensors/*.safetensors)")
    ap.add_argument("--out", help="output dir (omit with --verify-only)")
    ap.add_argument("--shards", help="substring filter on shard file names")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--min-saturation", type=float, default=0.98)
    ap.add_argument("--report", help="where to write the JSON report (default: <out>/csf_convert_report.json)")
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    shards = find_shards(src, args.shards)
    if not shards:
        print(f"no shards found under {src} (filter={args.shards!r})", file=sys.stderr)
        return 1
    if not args.verify_only and not args.out:
        print("--out is required unless --verify-only", file=sys.stderr)
        return 1

    out = Path(args.out).expanduser() if args.out else None
    results = []
    for shard in shards:
        rel = shard.relative_to(src)
        out_path = (out / rel) if out else None
        stats = process_shard(shard, out_path, args.min_saturation)
        results.append(stats)
        flag = "FAIL" if stats["errors"] else "ok"
        print(
            f"[{flag}] {shard.name}: csf={stats['csf_tensors']} pass={stats['passthrough_tensors']} "
            f"sat_min={stats['sat_min']:.4f} sat_mean={stats['sat_mean']:.4f} "
            f"viol={stats['window_violations']} nan={stats['nan_scales']} {stats['seconds']}s"
        )
        for err in stats["errors"][:5]:
            print(f"       error: {err}")

    if out is not None:
        for aux in src.iterdir():
            if aux.is_file():
                shutil.copy2(aux, out / aux.name)
    report_path = Path(args.report) if args.report else (out or Path("/tmp")) / "csf_convert_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=1))
    print(f"report: {report_path}")
    return 1 if any(r["errors"] for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
