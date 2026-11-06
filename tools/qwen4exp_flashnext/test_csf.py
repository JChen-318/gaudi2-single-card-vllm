# SPDX-License-Identifier: Apache-2.0
"""Self-test for csf.py: synthetic round-trips plus, when available, a
regression against real bytes pulled from shard 1 of the checkpoint.

Run:  python tools/qwen4exp_flashnext/test_csf.py [--real-dir /tmp]

The real-byte regression expects the files fetched during format analysis:
  e0_fixed.bin        (51840 B, gate_proj nvfp4_csf_fixed, [40, 1296])
  e0_exceptions.bin   (4180 B, 1045 U32 entries)
  e0_weight64k.bin    (~125 KB prefix of gate_proj packed weights)
  mtp_scale.bin       (102400 B, plain E4M3 plane [640, 160])
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from csf import (  # noqa: E402
    E4M3_LUT,
    decode_csf_plane,
    dequant_nvfp4,
    encode_csf_plane,
    e2m1_values,
    exception_window_violations,
    saturation_ratio,
    unpack_e2m1_codes,
)

RNG = np.random.default_rng(0)


def random_plane(rows: int, cols: int, lo: int = 0x58, hi: int = 0x78) -> np.ndarray:
    """A plausible scale plane: bell-shaped over an E4M3 byte window."""
    centers = RNG.integers(lo, hi - 12, size=(rows, 1))
    spread = RNG.normal(6, 3, size=(rows, cols))
    return np.clip(centers + spread, lo, hi).astype(np.uint8)


def test_roundtrip(rows: int, cols: int) -> None:
    plane = random_plane(rows, cols)
    fixed, exc = encode_csf_plane(plane)
    got = decode_csf_plane(fixed, exc, rows, cols)
    assert np.array_equal(got, plane), f"round-trip mismatch for {rows}x{cols}"
    print(f"  round-trip {rows}x{cols}: OK ({len(exc)} exceptions)")


def test_e2m1_symmetry() -> None:
    codes = np.arange(16, dtype=np.uint8)
    vals = e2m1_values(codes)
    assert np.allclose(vals[8:], -vals[:8])
    assert vals[7] == 6.0 and vals[15] == -6.0
    print("  E2M1 sign/magnitude symmetry: OK")


def test_dequant_smoke() -> None:
    packed = RNG.integers(0, 256, size=(4, 16), dtype=np.uint8)  # K=32
    plane = np.full((4, 2), 0x68, dtype=np.uint8)  # 2 blocks
    w = dequant_nvfp4(packed, plane, 6.64847e-05)
    assert w.shape == (4, 32)
    scale_val = float(E4M3_LUT[0x68]) * 6.64847e-05
    codes = unpack_e2m1_codes(packed)
    assert np.allclose(np.abs(w).max(), 6.0 * scale_val, rtol=1e-5)
    assert saturation_ratio(codes[:, :16]) in (0.0, 1.0)  # deterministic per block
    print(f"  dequant smoke: max|w|={np.abs(w).max():.5f} matches scale*6: OK")


def test_real_bytes(base: Path) -> None:
    fixed = np.fromfile(base / "e0_fixed.bin", dtype=np.uint8)
    exc = np.fromfile(base / "e0_exceptions.bin", dtype="<u4")
    plane = decode_csf_plane(fixed, exc, 640, 160)

    violations = exception_window_violations(fixed, exc, 640, 160)
    assert violations == 0, f"{violations} exceptions reconstruct their own window value"
    print(f"  real bytes: window violations 0/{len(exc)}: OK")

    wpath = base / "e0_weight64k.bin"
    if wpath.exists():
        wbytes = np.fromfile(wpath, dtype=np.uint8)
        rows = wbytes.size // 1280
        codes = unpack_e2m1_codes(wbytes[: rows * 1280].reshape(rows, 1280))
        sat = saturation_ratio(codes)
        assert sat == 1.0, f"saturation {sat:.4f} != 1.0"
        print(f"  real bytes: block saturation 1.0 over {rows} rows: OK")

    mtp = np.fromfile(base / "mtp_scale.bin", dtype=np.uint8)
    assert mtp.size == 102400
    assert not np.isnan(E4M3_LUT[mtp]).any(), "NaN codes in plain MTP plane"
    print("  MTP plain plane: no NaN codes: OK")


def _snapshot_dir() -> Path | None:
    """The raw NVIDIA CSF export in the local HF cache, if present."""
    hub = Path.home() / ".cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots"
    if not hub.is_dir():
        return None
    for candidate in sorted(hub.iterdir()):
        if (candidate / "model.safetensors.index.json").is_file():
            return candidate
    return None


def test_real_bytes_snapshot(snap: Path) -> None:
    """Real CSF tensors streamed from the raw HF snapshot (a few nvfp4_csf_*
    tensors): window violations 0, full block saturation, no NaN scale codes,
    and a plain-E4M3 plane (MTP) with no NaN codes."""
    import json

    from safetensors import safe_open

    index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]

    def get(name: str) -> np.ndarray:
        import torch

        entry = index[name]
        path = snap / entry
        if not path.is_file():
            path = snap / "tensors" / entry
        with safe_open(path, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(name)
        if tensor.dtype == torch.float8_e4m3fn:
            tensor = tensor.view(torch.uint8)  # E4M3 code bytes for the LUT
        return tensor.numpy()

    expert = "model.language_model.layers.0.mlp.experts.0.gate_proj"
    fixed = get(expert + ".weight_scale.nvfp4_csf_fixed")
    exc = get(expert + ".weight_scale.nvfp4_csf_exceptions").astype(np.uint32)
    assert fixed.shape == (40, 1296) and fixed.dtype == np.uint8, fixed.shape
    plane = decode_csf_plane(fixed, exc, 640, 160)
    violations = exception_window_violations(fixed, exc, 640, 160)
    assert violations == 0, f"{violations} exceptions reconstruct their own window value"
    assert not np.isnan(E4M3_LUT[plane]).any(), "NaN scale codes after CSF decode"
    print(f"  snapshot CSF decode: window violations 0/{len(exc)}, no NaN codes: OK")

    weight = get(expert + ".weight")
    rows = min(weight.shape[0], 256)
    codes = unpack_e2m1_codes(weight[:rows])
    sat = saturation_ratio(codes)
    assert sat == 1.0, f"saturation {sat:.4f} != 1.0"
    print(f"  snapshot expert weights: block saturation 1.0 over {rows} rows: OK")

    mtp = get("mtp.layers.0.mlp.experts.0.gate_proj.weight_scale").reshape(-1)
    assert mtp.size == 640 * 160
    assert not np.isnan(E4M3_LUT[mtp]).any(), "NaN codes in plain MTP plane"
    print("  snapshot MTP plain plane: no NaN codes: OK")


def main() -> int:
    base = Path(sys.argv[sys.argv.index("--real-dir") + 1]) if "--real-dir" in sys.argv else Path("/tmp")
    print("test_csf:")
    test_roundtrip(32, 40)
    test_roundtrip(64, 160)
    test_roundtrip(2560, 40)
    test_e2m1_symmetry()
    test_dequant_smoke()
    if (base / "e0_fixed.bin").exists():
        test_real_bytes(base)
    elif (snap := _snapshot_dir()) is not None:
        # No stashed fixtures: stream the real bytes from the HF snapshot.
        test_real_bytes_snapshot(snap)
    else:
        print("  (no real-byte fixtures and no HF snapshot; skipping regression)")
    print("all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
