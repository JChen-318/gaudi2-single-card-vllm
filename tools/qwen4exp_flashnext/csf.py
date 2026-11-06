# SPDX-License-Identifier: Apache-2.0
"""CSF (Compressed Scale Factors) codec for ModelOpt NVFP4 checkpoints.

Layout reverse-engineered from ``nvidia/Qwen3.8-Flash-Next-FP4-Local`` (ModelOpt
MIXED_PRECISION, quantized with modelopt v0.47.0). For every NVFP4 linear whose
weight is ``[rows, K/2]`` packed E2M1 (group size 16 along K), the E4M3
block-scale plane ``[rows, K/16]`` is stored as three tensors:

* ``<base>.weight_scale.nvfp4_csf_fixed``  U8 ``[rows/16, 16 + 8*cols]`` where
  ``cols = K/16``. Per group of 16 output rows: 16 header bytes (one per row),
  then ``cols`` nibbles per row packed two per byte, low nibble first.
* ``<base>.weight_scale.nvfp4_csf_exceptions``  U32 array, each entry packed as
  ``(true_e4m3_byte << 24) | index`` with ``index = row*cols + col``.
* ``<base>.weight_scale_2``  F32 scalar, the per-tensor global scale.

Reconstruction is lossless: ``scale_byte = header + nibble`` for values inside
the 16-wide window; exceptions override. Validated on shard 1 (expert 0,
gate_proj): 0 of 1045 exceptions fall inside their window, and the decoded
plane reproduces the 100% block-saturation invariant of NVFP4 quantization.

The encoder here is a *reference* encoder for round-trip tests; NVIDIA's
original encoder picks headers the same way (window start per row) but is not
needed for decoding.
"""

from __future__ import annotations

import numpy as np

E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def _e4m3_lut() -> np.ndarray:
    codes = np.arange(256, dtype=np.int32)
    sign = np.where(codes & 0x80, -1.0, 1.0)
    exp = (codes >> 3) & 0x0F
    man = codes & 0x07
    normal = sign * (1.0 + man / 8.0) * np.power(2.0, exp.astype(np.float64) - 7.0)
    subnormal = sign * (man / 8.0) * np.power(2.0, -6.0)
    val = np.where(exp == 0, subnormal, normal)
    val = np.where((codes & 0x7F) == 0x7F, np.nan, val)  # E4M3FN NaN codes
    return val.astype(np.float32)


E4M3_LUT = _e4m3_lut()


def fixed_plane_shape(rows: int, cols: int) -> tuple[int, int]:
    """Expected ``nvfp4_csf_fixed`` shape for a ``[rows, cols]`` scale plane."""
    if rows % 16 != 0:
        raise ValueError(f"rows={rows} not divisible by 16")
    if cols % 2 != 0:
        raise ValueError(f"cols={cols} not even")
    return rows // 16, 16 + 8 * cols


def _split_fixed(fixed: np.ndarray, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    expect = fixed_plane_shape(rows, cols)
    fixed = np.asarray(fixed, dtype=np.uint8).reshape(expect)
    hdr = fixed[:, :16]
    payload = fixed[:, 16:].reshape(rows // 16, 16, cols // 2)
    return hdr, payload


def _nibbles(payload: np.ndarray, rows: int, cols: int) -> np.ndarray:
    groups = rows // 16
    nib = np.empty((groups, 16, cols), dtype=np.uint8)
    nib[..., 0::2] = payload & 0x0F
    nib[..., 1::2] = payload >> 4
    return nib


def decode_csf_plane(fixed: np.ndarray, exceptions: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Decode a CSF-compressed E4M3 scale plane into ``[rows, cols]`` bytes."""
    hdr, payload = _split_fixed(fixed, rows, cols)
    nib = _nibbles(payload, rows, cols)
    plane = hdr[:, :, None].astype(np.int32) + nib.astype(np.int32)
    if plane.max() > 255:
        raise ValueError("header + nibble exceeds 255: layout mismatch")
    plane = plane.astype(np.uint8).reshape(rows, cols)
    exceptions = np.asarray(exceptions, dtype=np.uint64).ravel()
    if exceptions.size:
        val = (exceptions >> 24).astype(np.uint8)
        idx = (((exceptions >> 16) & 1) << 16) | (exceptions & 0xFFFF)
        if int(idx.max()) >= rows * cols:
            raise ValueError(f"exception index {int(idx.max())} outside plane of {rows * cols}")
        flat = plane.reshape(-1)
        flat[idx] = val
    return plane


def exception_window_violations(
    fixed: np.ndarray, exceptions: np.ndarray, rows: int, cols: int
) -> int:
    """Count exceptions whose value equals ``header + nibble`` at its index.

    Every exception exists because the window value is wrong; an exception that
    matches its window reconstruction indicates a layout misunderstanding.
    """
    hdr, payload = _split_fixed(fixed, rows, cols)
    nib = _nibbles(payload, rows, cols).reshape(rows, cols)
    hdr_arr = hdr.reshape(-1)  # group-major: row index = g*16 + r
    exceptions = np.asarray(exceptions, dtype=np.uint64).ravel()
    if exceptions.size == 0:
        return 0
    val = (exceptions >> 24).astype(np.int32)
    idx = (((exceptions >> 16) & 1) << 16) | (exceptions & 0xFFFF)
    row = (idx // cols).astype(np.int64)
    col = (idx % cols).astype(np.int64)
    recon = hdr_arr[row].astype(np.int32) + nib[row, col].astype(np.int32)
    return int((recon == val).sum())


def unpack_e2m1_codes(packed: np.ndarray) -> np.ndarray:
    """``[r, K/2]`` U8 packed nibbles -> ``[r, K]`` codes (0..15, sign in bit 3)."""
    packed = np.asarray(packed, dtype=np.uint8)
    out = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.uint8)
    out[..., 0::2] = packed & 0x0F
    out[..., 1::2] = packed >> 4
    return out


def e2m1_values(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.uint8)
    return np.where(codes & 0x08, -1.0, 1.0).astype(np.float32) * E2M1_VALUES[codes & 0x07]


def saturation_ratio(codes: np.ndarray) -> float:
    """Fraction of 16-value blocks whose max |E2M1| code value is exactly 6."""
    vals = np.abs(e2m1_values(codes))
    k = vals.shape[-1]
    if k % 16 != 0:
        raise ValueError(f"K={k} not divisible by 16")
    blocks = vals.reshape(*vals.shape[:-1], k // 16, 16).max(axis=-1)
    return float((blocks == 6.0).mean())


def nan_scale_count(plane: np.ndarray) -> int:
    plane = np.asarray(plane, dtype=np.uint8)
    return int(((plane & 0x7F) == 0x7F).sum())


def dequant_nvfp4(packed: np.ndarray, plane: np.ndarray, scale2: float) -> np.ndarray:
    codes = unpack_e2m1_codes(packed)
    scale_vals = E4M3_LUT[np.asarray(plane, dtype=np.uint8)]
    if scale_vals.shape != codes.shape:
        scale_vals = np.repeat(scale_vals, 16, axis=-1)  # one scale per 16 K values
    return e2m1_values(codes) * scale_vals * np.float32(scale2)


def dequant_mxfp8(weight: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """``weight`` [r, K] E4M3 bytes, ``scale`` [r, K/32] UE8M0 bytes."""
    scale = np.asarray(scale, dtype=np.uint8)
    factor = np.power(2.0, scale.astype(np.int32) - 127).astype(np.float32)
    return E4M3_LUT[np.asarray(weight, dtype=np.uint8)] * np.repeat(factor, 32, axis=-1)


def encode_csf_plane(plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference encoder (window start = row min) used by the self-test."""
    rows, cols = plane.shape
    fixed_plane_shape(rows, cols)
    plane = np.asarray(plane, dtype=np.int32)
    groups = rows // 16
    hdr = np.zeros((groups, 16), dtype=np.uint8)
    payload = np.zeros((groups, 16, cols), dtype=np.uint8)
    exc_idx: list[int] = []
    exc_val: list[int] = []
    for g in range(groups):
        for r in range(16):
            row = plane[g * 16 + r]
            h = min(int(row.min()), 255 - 15)
            hdr[g, r] = h
            delta = row - h
            inside = (delta >= 0) & (delta <= 15)
            payload[g, r] = np.where(inside, delta, np.clip(delta, 0, 15)).astype(np.uint8)
            for c in np.nonzero(~inside)[0]:
                exc_idx.append((g * 16 + r) * cols + int(c))
                exc_val.append(int(plane[g * 16 + r, c]))
    order = np.argsort(exc_idx)
    idx_arr = np.array(exc_idx, dtype=np.uint64)[order] if exc_idx else np.array([], dtype=np.uint64)
    val_arr = np.array(exc_val, dtype=np.uint64)[order] if exc_val else np.array([], dtype=np.uint64)
    exc = (val_arr << 24) | idx_arr
    packed = payload[..., 0::2] | (payload[..., 1::2] << 4)  # low nibble first
    flat = np.concatenate([hdr, packed.reshape(groups, 8 * cols)], axis=1)
    return flat, exc.astype(np.uint32)
