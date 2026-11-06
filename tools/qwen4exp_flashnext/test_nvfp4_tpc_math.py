# SPDX-License-Identifier: Apache-2.0
"""Golden-model test for the NVFP4 dequant TPC kernel's bit math.

Mirrors vllm-gaudi/csrc/qwen4exp/include/qwen4exp_nvfp4_indexed_dequant_bf16.h
exactly (integer fixed-point construction of e2m1 * e4m3 into BF16 bits) and
compares it against a float32 reference over all 256 nibble x 256 scale-byte
combinations, plus real checkpoint bytes when /tmp/e0_*.bin exist.

Run: python tools/qwen4exp_flashnext/test_nvfp4_tpc_math.py [--real-dir /tmp]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from csf import E2M1_VALUES, E4M3_LUT, decode_csf_plane  # noqa: E402


def kernel_bits_scalar(nib: int, s: int) -> int:
    mag, odd = nib & 7, nib & 1
    e, m = (s >> 3) & 15, s & 7
    base = (m << 2) if e == 0 else ((m + 8) << 2)
    extra = (base >> 1) if (odd == 1 and mag != 1) else 0
    fixed0 = base + extra
    efield = e + (mag >> 1) + 119 + (1 if e == 0 else 0)
    if fixed0 >= 64:
        fixed0 >>= 1
        efield += 1
    for _ in range(3):
        if fixed0 < 32:
            fixed0 <<= 1
            efield -= 1
    bits = ((fixed0 - 32) << 2) | (efield << 7)
    if mag == 0:
        bits = 0
    if e == 0 and m == 0:
        bits = 0
    return bits ^ (((nib & 8) << 12) ^ ((s & 0x80) << 8))


def ref_bf16_bits(nib: int, s: int) -> int:
    val = E2M1_VALUES[nib & 7] * (-1.0 if nib & 8 else 1.0) * E4M3_LUT[s]
    return int((np.array([val], dtype=np.float32).view(np.uint32) >> 16)[0])


def main() -> int:
    bad = nan = 0
    for n in range(256):
        for s in range(256):
            kb, rb = kernel_bits_scalar(n, s), ref_bf16_bits(n, s)
            if kb != rb and (s & 0x7F) != 0x7F:
                bad += 1
            elif kb != rb:
                nan += 1
    print(f"test_nvfp4_tpc_math: exhaustive mismatches={bad}, NaN-scale codes={nan} (never occur)")
    assert bad == 0
    base = Path(sys.argv[sys.argv.index("--real-dir") + 1]) if "--real-dir" in sys.argv else Path("/tmp")
    if (base / "e0_fixed.bin").exists():
        fixed = np.fromfile(base / "e0_fixed.bin", dtype=np.uint8)
        exc = np.fromfile(base / "e0_exceptions.bin", dtype="<u4")
        plane = decode_csf_plane(fixed, exc, 640, 160)
        w = np.fromfile(base / "e0_weight64k.bin", dtype=np.uint8)
        rows = w.size // 1280
        wb = w[: rows * 1280].reshape(rows, 1280)
        codes = np.empty((rows, 2560), dtype=np.uint16)
        codes[:, 0::2] = wb & 0x0F
        codes[:, 1::2] = wb >> 4
        scale_exp = np.repeat(plane[:rows], 16, axis=1)
        kbits = np.vectorize(kernel_bits_scalar)(codes, scale_exp).astype(np.uint16)
        ref_bits = (np.float32(
            E2M1_VALUES[codes & 7] * np.where(codes & 8, -1, 1) * E4M3_LUT[scale_exp]
        ).view(np.uint32) >> 16).astype(np.uint16)
        assert np.array_equal(kbits, ref_bits), "real-bytes bit mismatch"
        print(f"  real expert-0 gate_proj rows={rows}: bit-exact OK")
    print("all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
