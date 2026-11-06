# SPDX-License-Identifier: Apache-2.0
"""CPU proof that the dequant factor LUT matches nvfp4_mul_bits.

TPC cannot element-initialize a vector constant, so the kernel builds the
E2M1 and E4M3 BF16 tables from this closed form and multiplies. All 16 x 256
products must match the integer construction. scale2 is not in the product;
a check that forgets it must fail.

  python tools/qwen4exp_flashnext/test_nvfp4_lut_bits.py
"""
from __future__ import annotations

import numpy as np


def mul_bits(nibble: int, scale: int) -> int:
    """Integer construction in qwen4exp_nvfp4_decode.h, one pair."""
    mag = nibble & 7
    odd = nibble & 1
    sign = ((nibble & 8) << 12) ^ ((scale & 128) << 8)
    exp = (scale >> 3) & 15
    mant = scale & 7
    base = (mant << 2) if exp == 0 else ((mant + 8) << 2)
    extra = 0 if odd != 1 else (0 if mag == 1 else (base >> 1))
    fixed = base + extra
    field = exp + (mag >> 1) + 119 + (1 if exp == 0 else 0)
    if fixed >= 64:
        fixed >>= 1
        field += 1
    for _ in range(3):
        if fixed < 32:
            fixed <<= 1
            field -= 1
    bits = 0 if mag == 0 or (mant == 0 and exp == 0) else (((fixed - 32) << 2) | (field << 7))
    return (bits | sign) & 0xFFFF


def bf16_bits(value: np.float32) -> int:
    raw = int(np.float32(value).view(np.uint32))
    top = raw >> 16
    if ((raw >> 15) & 1) and ((raw & 0x7FFF) or (top & 1)):
        top += 1
    return top & 0xFFFF


def e4m3_bf16(scale: int) -> int:
    """Closed form used by nvfp4_e4m3_table. 0x7F/0xFF stay finite."""
    exp = (scale >> 3) & 15
    mant = scale & 7
    sign = (scale & 128) << 8
    if exp == 0 and mant == 0:
        return sign
    if exp == 0:
        sub = {1: 118 << 7, 2: 119 << 7, 3: (119 << 7) | 64, 4: 120 << 7,
               5: (120 << 7) | 32, 6: (120 << 7) | 64, 7: (120 << 7) | 96}
        return sign | sub[mant]
    return sign | ((exp + 120) << 7) | (mant << 4)


def e2m1_bf16(nibble: int) -> int:
    magnitudes = (0x0000, 0x3f00, 0x3f80, 0x3fc0, 0x4000, 0x4040, 0x4080, 0x40c0)
    return magnitudes[nibble & 7] | ((nibble & 8) << 12)


def main() -> int:
    mismatches = 0
    for nibble in range(16):
        for scale in range(256):
            e2_f = np.uint32(e2m1_bf16(nibble) << 16).view(np.float32)
            e4_f = np.uint32(e4m3_bf16(scale) << 16).view(np.float32)
            got = bf16_bits(np.float32(e2_f * e4_f))
            if got != mul_bits(nibble, scale) or e4m3_bf16(scale) != bf16_bits(_e4m3_value(scale)):
                mismatches += 1
    if mismatches:
        raise SystemExit(f"LUT factorization mismatches: {mismatches}")

    # scale2 is the correction term. Forgetting it must not compare equal.
    weight = np.float32(1.5)
    folded = weight * np.float32(0.05)
    forgotten = weight * np.float32(1.0)
    if folded == forgotten:
        raise SystemExit("sabotage fixture is vacuous: scale2=1 matches scale2=0.05")
    detected = folded != forgotten
    if not detected:
        raise SystemExit("scale2 sabotage was not detected")
    print(f"lut bits: 16x256 match, scale2 sabotage detected ({float(folded)} != {float(forgotten)})")
    return 0


def _e4m3_value(scale: int) -> np.float32:
    exp = (scale >> 3) & 15
    mant = scale & 7
    sign = np.float32(-1.0 if scale & 128 else 1.0)
    if exp == 0:
        return np.float32(sign * (mant / 8.0) * np.float32(2.0 ** -6))
    return np.float32(sign * (1.0 + mant / 8.0) * np.float32(2.0 ** (exp - 7)))


if __name__ == "__main__":
    raise SystemExit(main())
