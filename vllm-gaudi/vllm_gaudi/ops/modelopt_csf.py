# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ModelOpt CSF scale decoding and MIXED_PRECISION format dispatch for Gaudi2.

This module is the host-side front end of the Qwen4Exp ``MIXED_PRECISION``
ModelOpt checkpoint.  It has no vLLM or Habana imports at module level, so it
can be exercised on a plain CPU host (``validation/qwen4exp/`` loads the file
directly by path).  Nothing here allocates an HPU tensor.

Three on-disk encodings appear in the checkpoint and are handled here:

``NVFP4`` / ``W4A16_NVFP4`` (group 16)
    ``weight`` U8 ``[N, K/2]`` -- two E2M1 nibbles per byte, low nibble first.
    ``weight_scale`` is *either* a plain E4M3 plane ``[N, K/16]`` *or* the
    CSF-compressed form ``weight_scale.nvfp4_csf_fixed`` U8
    ``[N/16, 16 + 8*(K/16)]`` plus ``weight_scale.nvfp4_csf_exceptions`` U32.
    ``weight_scale_2`` F32 scalar is the per-tensor global scale
    (``amax / (6*448)``, a multiplier -- see
    ``vllm-core/.../quantization/modelopt.py`` ModelOptNvFp4W4A16LinearMethod).
    Dequantization is ``w = e2m1(code) * e4m3(scale_byte) * scale2``; the
    weights are never expanded here.

``MXFP8`` (group 32)
    ``weight`` E4M3 ``[N, K]``, ``weight_scale`` U8 ``[N, K/32]`` holding the
    UE8M0 exponent byte (value ``2**(byte - 127)``).  Gaudi2's MME wants one
    inverse scale per output channel, so the block scales are folded into a
    per-row power-of-two rescale (:func:`mxfp8_to_fp8_per_channel`).

CSF layout
    Taken from the NVIDIA export (the original 96 GiB Hugging Face snapshot,
    whose expert scales really are CSF; the converted export in
    ``~/models/qwen4exp-flashnext-plain`` stores plain planes).  The
    reconstruction is verified two ways in
    ``validation/qwen4exp/test_csf_loader.py``: byte-for-byte against the
    converted export's plain planes for five real expert tensors (different
    layers, expert ids and projections), and structurally on the CSF tensors
    themselves -- 0 of the exceptions reproduce their window value and the
    decoded planes keep NVFP4's 100% block saturation.  The layout is::

        plane[g*16 + r, c] = fixed[g, r] + nibble(payload[g, r, c])
        payload[g, r, c]   = fixed[g, 16 + r*(cols/2) + c/2], low nibble for
                             even c, high nibble for odd c
        exception record   = (value_byte << 24) | ((index >> 16) & 1) << 16
                             | (index & 0xFFFF), index = row*cols + col

    ``16 + 8*cols`` bytes per 16-row group: 16 base bytes plus ``cols``
    nibbles per row (confirmed on disk: a 640x160 plane is stored
    ``[40, 1296]``).  Exceptions override the window reconstruction for scale
    bytes outside ``[base, base + 15]``.  Values are E4M3 *bytes*, not floats;
    the decoded plane is stored back as ``float8_e4m3fn`` so the packed
    weights keep their 4-bit residency.

Resident layout and the TPC preshuffle
    The decode is a *load-time* step: CSF is expanded on the host into the flat
    ``float8_e4m3fn`` plane ``[E, N, K/16]`` the GEMV reads, in memory, and no
    file is ever written back into the checkpoint directory.  The layout of
    those resident planes (byte order, TPC vector geometry, the preshuffle and
    its ``VLLM_QWEN4EXP_TPC_LAYOUT`` switch) is defined once, for Track 3 and
    for this module, in :mod:`vllm_gaudi.ops.nvfp4_tpc_layout`; every loader
    here ends with :func:`tpc_layout.tpc_planes`, so a plane that reaches the
    device is always in the declared layout (``TPC_GEMV_LAYOUT``: byte-identical
    to the checkpoint, K/2 on the FCD axis) or in the opt-in padded variant.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

import torch

__all__ = [
    "CSF_EXCEPTIONS_SUFFIX",
    "CSF_FIXED_SUFFIX",
    "E4M3_FNUZ_MAX",
    "E4M3_MAX",
    "KIND_FP8",
    "KIND_MXFP8",
    "KIND_NVFP4_PACKED",
    "KIND_UNQUANTIZED",
    "KIND_W4A16_BF16",
    "MXFP8_GROUP_SIZE",
    "NVFP4_GROUP_SIZE",
    "LayerFormat",
    "NvFp4PackedWeights",
    "NvFp4ScaleSpec",
    "classify_layer",
    "count_csf_window_violations",
    "csf_fixed_plane_shape",
    "dequant_mxfp8",
    "dequant_nvfp4",
    "dequant_w4a16_nvfp4",
    "detect_nvfp4_scale",
    "dispatch_kind",
    "e2m1_lut",
    "e4m3_lut",
    "is_layer_ignored",
    "load_expert_w13",
    "load_experts",
    "load_mxfp8_as_fp8",
    "load_packed_nvfp4",
    "load_w4a16_as_bf16",
    "mxfp8_scale_to_float",
    "mxfp8_to_fp8_per_channel",
    "nvfp4_block_saturation_ratio",
    "resolve_quant_algo",
    "unpack_e2m1_codes",
]

NVFP4_GROUP_SIZE = 16
MXFP8_GROUP_SIZE = 32

#: Largest finite value of ``torch.float8_e4m3fn`` (OCP E4M3).
E4M3_MAX = 448.0
#: Largest finite value Gaudi2's FP8 MME represents (E4M3 FNUZ).
E4M3_FNUZ_MAX = 240.0

CSF_FIXED_SUFFIX = ".weight_scale.nvfp4_csf_fixed"
CSF_EXCEPTIONS_SUFFIX = ".weight_scale.nvfp4_csf_exceptions"
PLAIN_SCALE_SUFFIX = ".weight_scale"
GLOBAL_SCALE_SUFFIX = ".weight_scale_2"

CSF_ENCODING = "csf"
PLAIN_ENCODING = "plain"
MXFP8_ENCODING = "mxfp8_ue8m0"

#: Dispatch kinds returned by :func:`dispatch_kind`.
KIND_NVFP4_PACKED = "nvfp4_packed"  # packed 4-bit resident, plain E4M3 scales
KIND_MXFP8 = "mxfp8_to_fp8"  # requantized per-channel FP8 (1 byte/weight)
KIND_W4A16_BF16 = "w4a16_to_bf16"  # small matrices dequantized at load
KIND_FP8 = "fp8"  # plain ModelOpt FP8 checkpoint layers
KIND_UNQUANTIZED = "unquantized"

_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E4M3_NAN_BYTES = (0x7F, 0xFF)

#: Cached ``nvfp4_tpc_layout`` module (see :func:`tpc_layout_module`).
_TPC_LAYOUT_MODULE: Any = None


def tpc_layout_module():
    """The TPC layout contract module (:mod:`vllm_gaudi.ops.nvfp4_tpc_layout`).

    Imported lazily and by path when this file is loaded outside the package
    (``validation/qwen4exp/modelopt_support.py`` does exactly that), so the
    host-only suite never has to import ``vllm_gaudi`` -- and therefore never
    touches the Habana stack.
    """
    global _TPC_LAYOUT_MODULE
    if _TPC_LAYOUT_MODULE is not None:
        return _TPC_LAYOUT_MODULE
    if __package__:
        from vllm_gaudi.ops import nvfp4_tpc_layout as module
    else:  # loaded by file path: same directory, no package import
        import importlib.util
        import sys as _sys
        from pathlib import Path as _Path

        name = "qwen4exp_nvfp4_tpc_layout"
        module = _sys.modules.get(name)
        if module is None:
            path = _Path(__file__).resolve().with_name("nvfp4_tpc_layout.py")
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load the TPC layout module from {path}")
            module = importlib.util.module_from_spec(spec)
            _sys.modules[name] = module
            spec.loader.exec_module(module)
    _TPC_LAYOUT_MODULE = module
    return module


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------
def e2m1_lut(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``[16]`` table: E2M1 nibble (sign in bit 3) -> value."""
    magnitudes = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32)
    table = torch.cat([magnitudes, -magnitudes])
    return table.to(dtype)


def e4m3_lut(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``[256]`` table: E4M3 byte -> value, NaN for the two NaN codes.

    Built from torch's own ``float8_e4m3fn`` decoder so the table cannot drift
    from the dtype the FP8 path stores.
    """
    table = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
    table = table.to(dtype)
    return table


_LUT_CACHE: dict[tuple, torch.Tensor] = {}


def _lut_index(builder, index: torch.Tensor) -> torch.Tensor:
    """Look up ``builder()`` on ``index``'s device (cached per device).

    ``process_weights_after_loading`` runs with the module already on the
    accelerator, so a host-built LUT cannot index those bytes.
    """
    key = (getattr(builder, "__name__", repr(builder)), index.device.type, index.device.index)
    table = _LUT_CACHE.get(key)
    if table is None:
        table = builder().to(index.device)
        _LUT_CACHE[key] = table
    return table[index]


def _as_uint8(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        return tensor
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        return tensor.view(torch.uint8)
    raise TypeError(f"expected a uint8 or fp8 byte tensor, got {tensor.dtype}")


# ---------------------------------------------------------------------------
# CSF decoding
# ---------------------------------------------------------------------------
def csf_fixed_plane_shape(rows: int, cols: int) -> tuple[int, int]:
    """Shape of ``weight_scale.nvfp4_csf_fixed`` for a ``[rows, cols]`` plane."""
    if rows % 16 != 0:
        raise ValueError(f"CSF rows must be a multiple of 16, got {rows}")
    if cols % 2 != 0:
        raise ValueError(f"CSF scale columns must be even, got {cols}")
    return rows // 16, 16 + 8 * cols


def _csf_plane_bytes(rows: int, cols: int) -> int:
    return rows * cols


def count_csf_window_violations(
    fixed: torch.Tensor,
    exceptions: torch.Tensor,
    rows: int,
    cols: int,
) -> int:
    """Exceptions whose value equals ``base + nibble`` at their own index.

    Every exception exists because the window reconstruction is wrong there,
    so a non-zero count means the layout is misunderstood.
    """
    plane = _decode_csf_window(fixed, rows, cols)
    index = csf_exception_index(exceptions)
    value = csf_exception_value(exceptions)
    if index.numel() == 0:
        return 0
    flat = plane.reshape(-1)
    return int((flat.index_select(0, index) == value).sum().item())


def _decode_csf_window(fixed: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Window reconstruction ``base + nibble`` without the exception pass."""
    fixed = _as_uint8(fixed)
    expect = csf_fixed_plane_shape(rows, cols)
    if tuple(fixed.shape) != expect:
        raise ValueError(f"nvfp4_csf_fixed shape {tuple(fixed.shape)} != {expect}")
    groups = rows // 16
    base = fixed[:, :16].to(torch.int16)  # [groups, 16]
    payload = fixed[:, 16:].reshape(groups, 16, cols // 2)
    nibbles = torch.empty(groups, 16, cols, dtype=torch.int16, device=fixed.device)
    nibbles[..., 0::2] = (payload & 0x0F).to(torch.int16)
    nibbles[..., 1::2] = (payload >> 4).to(torch.int16)
    window = base.unsqueeze(-1) + nibbles
    if int(window.max().item()) > 255:
        raise ValueError("base + nibble overflowed a byte: CSF layout mismatch")
    return window.to(torch.uint8).reshape(rows, cols)


def csf_exception_index(exceptions: torch.Tensor) -> torch.Tensor:
    """Exception record -> flat index into the ``[rows, cols]`` plane.

    The index is 17 bits: bit 16 lives in bit 16 of the record, the low 16
    bits in the low half.  Real expert planes have ``rows*cols > 65536``
    (e.g. ``2560*40 = 102400``), so the 17th bit is load-bearing.
    """
    words = exceptions.to(torch.int64).reshape(-1)
    return (((words >> 16) & 1) << 16) | (words & 0xFFFF)


def csf_exception_value(exceptions: torch.Tensor) -> torch.Tensor:
    """Exception record -> replacement E4M3 byte."""
    words = exceptions.to(torch.int64).reshape(-1)
    return ((words >> 24) & 0xFF).to(torch.uint8)


def decode_csf_scale_plane(
    fixed: torch.Tensor,
    exceptions: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Decode a CSF-compressed E4M3 scale plane into ``[rows, cols]`` bytes."""
    plane = _decode_csf_window(fixed, rows, cols).reshape(-1)
    index = csf_exception_index(exceptions)
    value = csf_exception_value(exceptions)
    if index.numel():
        if int(index.max().item()) >= _csf_plane_bytes(rows, cols):
            raise ValueError(f"CSF exception index {int(index.max().item())} outside plane of "
                             f"{_csf_plane_bytes(rows, cols)}")
        plane = plane.index_put((index,), value)
    return plane.reshape(rows, cols)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------
class NvFp4ScaleSpec(NamedTuple):
    """Where a layer's NVFP4 block scales live and in which encoding."""

    encoding: str  # CSF_ENCODING or PLAIN_ENCODING
    rows: int
    cols: int
    scale_name: str | None
    fixed_name: str | None
    exceptions_name: str | None


def detect_nvfp4_scale(
    available: Iterable[str] | Mapping[str, Any],
    base: str,
    weight_shape: Sequence[int],
    group_size: int = NVFP4_GROUP_SIZE,
) -> NvFp4ScaleSpec:
    """Detect the on-disk NVFP4 scale encoding for one ``base`` module.

    ``available`` is either the set of tensor names in the checkpoint or a
    mapping of name -> anything.  The config's declared
    ``weight_scale_encoding`` is *not* trusted: this export declares ``csf``
    for layers whose scales are already plain E4M3 (the decoder experts in the
    converted checkpoint) and for the PLE n-gram shards in the original one.
    """
    names = set(available)
    if len(weight_shape) != 2:
        raise ValueError(f"NVFP4 weight must be 2D, got {tuple(weight_shape)}")
    rows, packed_cols = int(weight_shape[0]), int(weight_shape[1])
    cols = packed_cols * 2 // group_size

    fixed_name = f"{base}{CSF_FIXED_SUFFIX}"
    exceptions_name = f"{base}{CSF_EXCEPTIONS_SUFFIX}"
    plain_name = f"{base}{PLAIN_SCALE_SUFFIX}"

    if fixed_name in names:
        if exceptions_name not in names:
            raise ValueError(f"{base}: CSF fixed plane without an exceptions tensor")
        return NvFp4ScaleSpec(CSF_ENCODING, rows, cols, None, fixed_name, exceptions_name)
    if plain_name in names:
        return NvFp4ScaleSpec(PLAIN_ENCODING, rows, cols, plain_name, None, None)
    raise ValueError(f"{base}: neither {PLAIN_SCALE_SUFFIX} nor the CSF tensors are present")


def classify_layer(
    base: str,
    header: Mapping[str, Any],
    declared_algo: str | None = None,
) -> "LayerFormat":
    """Classify one module from a safetensors-style header.

    ``header`` maps tensor name -> ``{"dtype": str, "shape": [int, ...]}``
    (the JSON header of a shard is enough).  ``declared_algo`` is the
    ``quant_algo`` from ``config.json`` and is recorded, not used: NVFP4 and
    W4A16_NVFP4 are *byte-identical* on disk (packed U8 + group-16 E4M3 scales
    + F32 global scale) and differ only in whether the activations are 4-bit,
    so a per-matrix check cannot tell them apart.  Which loader to use is
    decided per module prefix by :func:`dispatch_kind`.
    """
    weight_name = f"{base}.weight"
    if weight_name not in header:
        raise KeyError(f"{weight_name} not in header")
    weight = header[weight_name]
    weight_dtype = str(weight["dtype"]).upper()
    weight_shape = tuple(int(dim) for dim in weight["shape"])
    scale_name = f"{base}{PLAIN_SCALE_SUFFIX}"
    declared = declared_algo.upper() if declared_algo else None

    if weight_dtype.startswith("F8"):
        scale = header.get(scale_name)
        if scale is None:
            return LayerFormat(base, KIND_UNQUANTIZED, weight_dtype, weight_shape, "none", None, None,
                               declared)
        scale_dtype = str(scale["dtype"]).upper()
        scale_shape = tuple(int(dim) for dim in scale["shape"])
        if scale_dtype != "U8":
            raise ValueError(f"{base}: E4M3 weight with a {scale_dtype} scale is not a "
                             "supported encoding")
        cols = weight_shape[-1] // MXFP8_GROUP_SIZE
        if scale_shape != (weight_shape[0], cols):
            raise ValueError(f"{base}: MXFP8 scale {scale_shape} != "
                             f"({weight_shape[0]}, {cols})")
        return LayerFormat(base, KIND_MXFP8, weight_dtype, weight_shape, MXFP8_ENCODING, scale_shape,
                           MXFP8_GROUP_SIZE, declared)

    if weight_dtype == "U8":
        if len(weight_shape) != 2 or weight_shape[1] * 2 % NVFP4_GROUP_SIZE != 0:
            raise ValueError(f"{base}: unsupported packed weight shape {weight_shape}")
        cols = weight_shape[1] * 2 // NVFP4_GROUP_SIZE
        fixed_name = f"{base}{CSF_FIXED_SUFFIX}"
        if fixed_name in header:
            encoding = CSF_ENCODING
            scale_shape = csf_fixed_plane_shape(weight_shape[0], cols)
        elif scale_name in header:
            encoding = PLAIN_ENCODING
            scale = header[scale_name]
            scale_shape = tuple(int(dim) for dim in scale["shape"])
            if scale_shape != (weight_shape[0], cols):
                raise ValueError(f"{base}: plain scale {scale_shape} != "
                                 f"({weight_shape[0]}, {cols})")
        else:
            return LayerFormat(base, KIND_UNQUANTIZED, weight_dtype, weight_shape, "none", None, None,
                               declared)
        return LayerFormat(base, KIND_NVFP4_PACKED, weight_dtype, weight_shape, encoding, scale_shape,
                           NVFP4_GROUP_SIZE, declared)

    return LayerFormat(base, KIND_UNQUANTIZED, weight_dtype, weight_shape, "none", None, None, declared)


class LayerFormat(NamedTuple):
    """Result of :func:`classify_layer`."""

    base: str
    kind: str
    weight_dtype: str
    weight_shape: tuple[int, ...]
    scale_encoding: str
    scale_shape: tuple[int, ...] | None
    group_size: int | None
    declared_algo: str | None = None


# ---------------------------------------------------------------------------
# NVFP4 math (reference; experts stay packed)
# ---------------------------------------------------------------------------
def unpack_e2m1_codes(packed: torch.Tensor) -> torch.Tensor:
    """``[..., K/2]`` U8 -> ``[..., K]`` E2M1 codes, low nibble first."""
    packed = _as_uint8(packed)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    codes = torch.stack([low, high], dim=-1).flatten(-2)
    return codes


def nvfp4_block_saturation_ratio(codes: torch.Tensor, group_size: int = NVFP4_GROUP_SIZE) -> float:
    """Fraction of groups whose largest ``|E2M1|`` is exactly 6.

    NVFP4 quantization is per-block ``amax``-normalized, so a correctly decoded
    plane saturates every block of an amax-quantized tensor.
    """
    values = e2m1_lut()[codes.to(torch.int64)].abs()
    k = values.shape[-1]
    if k % group_size:
        raise ValueError(f"K={k} not divisible by {group_size}")
    blocks = values.reshape(*values.shape[:-1], k // group_size, group_size).amax(dim=-1)
    return float((blocks == 6.0).float().mean().item())


def dequant_nvfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    scale2: torch.Tensor | float = 1.0,
    group_size: int = NVFP4_GROUP_SIZE,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``[N, K/2]`` packed + ``[N, K/16]`` E4M3 bytes -> ``[N, K]`` dequantized.

    Reference math for the expert planes -- and the *device* row dequant of the
    PLE n-gram gather.  Both lookup tables go through :func:`_lut_index`, which
    caches one table per device, so the same function dequantizes a host row
    set (the CPU twin) and a device-resident row set after an async H2D: the
    tensor's device decides, and no host-built table is ever indexed by device
    bytes.  On CPU the values are bit-identical to the previous host-only LUT
    (same tables, same multiply order).
    """
    packed = _as_uint8(packed)
    scale_bytes = _as_uint8(scales)
    if packed.dim() != 2:
        raise ValueError(f"packed NVFP4 weight must be 2D [N, K/2], got shape {tuple(packed.shape)}")
    rows, half = packed.shape
    k = half * 2
    if k % group_size:
        raise ValueError(f"K={k} not divisible by group_size={group_size}")
    if scale_bytes.shape != (rows, k // group_size):
        raise ValueError(f"scale shape {tuple(scale_bytes.shape)} != "
                         f"({rows}, {k // group_size})")
    codes = unpack_e2m1_codes(packed)
    values = _lut_index(e2m1_lut, codes.to(torch.int64))
    flat_scale = scale_bytes.reshape(-1).to(torch.int64)
    blocks = _lut_index(e4m3_lut, flat_scale).reshape(rows, k // group_size)
    out = values * blocks.repeat_interleave(group_size, dim=-1)
    factor = torch.as_tensor(scale2, dtype=torch.float32, device=out.device).reshape(-1)[:1]
    return (out * factor).to(dtype)


# ---------------------------------------------------------------------------
# MXFP8 -> per-channel FP8
# ---------------------------------------------------------------------------
def mxfp8_scale_to_float(scale: torch.Tensor, *, nan_value: float = 0.0) -> torch.Tensor:
    """UE8M0 byte -> ``2**(byte - 127)`` float32 (0xFF is a NaN code)."""
    bytes_ = _as_uint8(scale).to(torch.int64)
    exponent = bytes_.float() - 127.0
    factor = torch.exp2(exponent)
    nan_mask = (bytes_ & 0xFF) == 0xFF
    if bool(nan_mask.any()):
        factor = torch.where(nan_mask, torch.full_like(factor, nan_value), factor)
    return factor


def dequant_mxfp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = MXFP8_GROUP_SIZE,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``[N, K]`` E4M3 weights + ``[N, K/32]`` UE8M0 bytes -> ``[N, K]`` floats."""
    values = _as_uint8(weight).to(torch.int64)
    values = _lut_index(e4m3_lut, values)
    factor = mxfp8_scale_to_float(scale)
    if values.shape[-1] % group_size:
        raise ValueError(f"K={values.shape[-1]} not divisible by {group_size}")
    if factor.shape != (*values.shape[:-1], values.shape[-1] // group_size):
        raise ValueError(f"scale shape {tuple(factor.shape)} does not match weights "
                         f"{tuple(values.shape)} for group_size={group_size}")
    expanded = factor.repeat_interleave(group_size, dim=-1)
    return (values * expanded).to(dtype)


def mxfp8_to_fp8_per_channel(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    group_size: int = MXFP8_GROUP_SIZE,
    mode: str = "pow2",
    fp8_max: float = E4M3_MAX,
    gaudi2_correction: bool = False,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold MXFP8 block scales into one FP8 inverse scale per output channel.

    Gaudi2's FP8 MME applies ``B_scale_inv`` per output channel, so an MXFP8
    layer can only run there after its 32-wide UE8M0 scales are folded into one
    scale per row.  Two modes:

    ``pow2`` (default)
        ``row_scale = max_k 2**(e_k - 127)`` and ``w' = w * 2**(e_k - e_max)``.
        The rescale is a power of two, so every value that stays in the normal
        E4M3 range keeps its mantissa exactly.  Only values that land in the
        subnormal range (``|w'| < 2**-6``) are rounded to the ``2**-9``
        quantum, and only those below ``2**-9`` are lost entirely.
        ``|w'| <= E4M3_MAX`` holds by construction, so clamping never fires.
    ``amax``
        ``row_scale = amax(W) / 448`` (the ModelOpt weight_scale convention),
        which re-rounds every value to E4M3 but keeps the full dynamic range.

    Returns ``(weight_e4m3 [N, K], weight_scale [N] f32)`` where
    ``weight * weight_scale`` reconstructs the dequantized weight.

    ``gaudi2_correction`` mirrors ``vllm_gaudi.extension.ops.gaudi_weight_wrapper``
    (halve the FP8 value, double the inverse scale) so the result matches what
    the plugin's loader does for a plain ModelOpt FP8 checkpoint.  Apply it
    exactly once -- the MXFP8 linear method must not also wrap its loader.
    """
    values = _as_uint8(weight).to(torch.int64)
    values = _lut_index(e4m3_lut, values)
    factor = mxfp8_scale_to_float(scale)
    k = values.shape[-1]
    if k % group_size:
        raise ValueError(f"K={k} not divisible by group_size={group_size}")
    if factor.shape != (*values.shape[:-1], k // group_size):
        raise ValueError(f"scale shape {tuple(factor.shape)} does not match weights "
                         f"{tuple(values.shape)} for group_size={group_size}")

    nan_blocks = int(((_as_uint8(scale) & 0xFF) == 0xFF).sum().item())
    rows = values.shape[0]
    expanded = factor.repeat_interleave(group_size, dim=-1)
    dequantized = values * expanded

    if mode == "pow2":
        row_scale = factor.amax(dim=-1)
    elif mode == "amax":
        amax = dequantized.abs().amax(dim=-1)
        row_scale = amax / fp8_max
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'pow2' or 'amax'")
    positive = row_scale > 0
    row_scale = torch.where(positive, row_scale, torch.ones_like(row_scale))
    scaled = dequantized / row_scale.unsqueeze(-1)

    clamped = int((scaled.abs() > fp8_max).sum().item())
    scaled = scaled.clamp(-fp8_max, fp8_max)
    weight_fp8 = scaled.to(torch.float8_e4m3fn)

    if gaudi2_correction:
        # Same arithmetic as hpu_ops.gaudi_weight_wrapper: the value halves and
        # the inverse scale doubles, so the product is unchanged.
        weight_fp8 = (weight_fp8.to(torch.float32) * 0.5).to(torch.float8_e4m3fn)
        row_scale = row_scale * 2.0

    if diagnostics is not None:
        reconstructed = weight_fp8.to(torch.float32) * row_scale.unsqueeze(-1)
        error = reconstructed - dequantized
        energy = dequantized.pow(2).sum()
        diagnostics.update(
            mode=mode,
            rows=rows,
            columns=k,
            nan_scale_blocks=nan_blocks,
            clamped_values=clamped,
            zero_row_scales=int((~positive).sum().item()),
            flushed_values=int(((dequantized != 0) & (reconstructed == 0)).sum().item()),
            max_abs_error=float(error.abs().max().item()),
            relative_l2_error=float((error.pow(2).sum() / energy.clamp_min(1e-30)).sqrt().item()),
            gaudi2_correction=bool(gaudi2_correction),
        )
    return weight_fp8, row_scale.to(torch.float32)


# ---------------------------------------------------------------------------
# W4A16_NVFP4 -> BF16 (reuses vLLM's emulation math)
# ---------------------------------------------------------------------------
def dequant_w4a16_nvfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    scale2: torch.Tensor | float,
    dtype: torch.dtype = torch.bfloat16,
    group_size: int = NVFP4_GROUP_SIZE,
    swizzle: bool = False,
) -> torch.Tensor:
    """Dequantize a W4A16 NVFP4 matrix with vLLM's reference implementation.

    ``dequantize_to_dtype`` from
    ``vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py``
    is imported rather than reimplemented: same nibble decode, same
    ``scale * global_scale`` folding, same ``swizzle`` switch.  ModelOpt stores
    the plane linearly (``swizzle=False``); ``scale2`` is the multiplier, not a
    reciprocal.
    """
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        dequantize_to_dtype,
    )

    packed = _as_uint8(packed).contiguous()
    scale_bytes = _as_uint8(scales).contiguous()
    global_scale = torch.as_tensor(scale2, dtype=torch.float32).reshape(-1)[:1].contiguous()
    return dequantize_to_dtype(
        packed,
        scale_bytes,
        global_scale,
        dtype,
        group_size,
        swizzle=swizzle,
    )


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
class NvFp4PackedWeights(NamedTuple):
    """Resident NVFP4 planes for one expert group: 4-bit weights, E4M3 scales.

    ``packed`` ``[E, N, K/2]`` U8, ``scales`` ``[E, N, K/16]`` E4M3 bytes,
    ``scale2`` ``[E]`` F32.  ``packed[e] * scales[e] * scale2[e]`` is the
    dequantized expert; nothing here expands it.  ``layout`` names the resident
    layout (see :mod:`vllm_gaudi.ops.nvfp4_tpc_layout`); the default is the
    Track 3 GEMV's declared layout, which is byte-identical to the checkpoint.
    """

    packed: torch.Tensor
    scales: torch.Tensor
    scale2: torch.Tensor
    layout: str = "tpc_e_n_k2_v1"

    @property
    def k(self) -> int:
        """Input width the planes describe (``packed`` holds two nibbles/byte)."""
        return int(self.packed.shape[2]) * 2

    def geometry(self, *, require_kernel_caps: bool = True) -> dict[str, Any]:
        """The TPC geometry report for these planes (never raises on caps)."""
        return tpc_layout_module().geometry_report(
            int(self.packed.shape[0]),
            int(self.packed.shape[1]),
            self.k,
            layout=self.layout,
            require_kernel_caps=require_kernel_caps,
        )

    def to(self, device: torch.device | str | None = None, non_blocking: bool = False) -> "NvFp4PackedWeights":
        def move(tensor: torch.Tensor) -> torch.Tensor:
            if device is None:
                return tensor
            return tensor.to(device=device, non_blocking=non_blocking)

        return NvFp4PackedWeights(move(self.packed), move(self.scales), move(self.scale2), self.layout)

    def kernel_layout(self) -> "NvFp4PackedWeights":
        """The layout the Track 3 GEMV declares, validated and contiguous.

        Both in-tree GEMV kernels address a 256-byte packed vector as
        ``v_u8_ld_tnsr_b((int5){tile*256, row, expert}, packed)``, i.e. Synapse
        dimension 0 -- the fastest -- is the *packed K axis*.  Synapse reverses
        torch axes, so the plane must stay ``[E, N, K/2]``: the checkpoint order.

        This used to permute to ``[K/2, N, E]``, which puts the *expert* axis on
        the FCD and makes one 256-byte vector span 256 experts; no kernel in
        the tree reads that, so the declared layout is the identity permutation
        and this returns a contiguous copy after re-checking the geometry.
        """
        report = self.geometry(require_kernel_caps=False)
        if not report["ok"]:
            raise ValueError("resident planes are not in a TPC-loadable layout: "
                             + "; ".join(report["violations"]))
        def owned(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.detach().clone(memory_format=torch.contiguous_format)

        return NvFp4PackedWeights(owned(self.packed), owned(self.scales), owned(self.scale2),
                                  self.layout)


def _source_names(source: Mapping[str, Any] | Any) -> set[str]:
    if isinstance(source, Mapping):
        return set(source.keys())
    if hasattr(source, "keys"):
        return set(source.keys())
    raise TypeError("weight source must be a mapping or expose keys()")


def _source_tensor(source: Mapping[str, Any] | Any, name: str) -> torch.Tensor:
    if isinstance(source, Mapping):
        return source[name]
    if hasattr(source, "get_tensor"):
        return source.get_tensor(name)
    return source[name]  # type: ignore[index]


def _resolve_layout(layout: str | None) -> str:
    """Explicit layout, else ``VLLM_QWEN4EXP_TPC_LAYOUT`` (default declared)."""
    module = tpc_layout_module()
    return module.layout_from_env() if layout is None else layout


def _resident(packed: torch.Tensor,
              scales: torch.Tensor,
              layout: str,
              *,
              copy: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the TPC preshuffle once: ``(packed, scales)``, both resident.

    ``copy=False`` is for planes this function's caller just allocated (the
    ``torch.stack`` of per-expert planes), so the load does not duplicate a
    projection-sized buffer while the model is being loaded.
    """
    module = tpc_layout_module()
    k = int(packed.shape[-1]) * 2
    return module.tpc_planes(packed, scales, k, layout=layout, copy=copy)


def load_packed_nvfp4(
    source: Mapping[str, Any] | Any,
    base: str,
    *,
    group_size: int = NVFP4_GROUP_SIZE,
    scale2_name: str | None = None,
    layout: str | None = None,
) -> NvFp4PackedWeights:
    """Load one NVFP4 matrix, decoding CSF scales to a plain E4M3 plane.

    Returns a :class:`NvFp4PackedWeights` whose leading axis is ``1`` so a
    single matrix and an expert group share the same shape convention.
    ``scale2_name`` overrides the global-scale tensor.  When neither the
    override nor ``{base}.weight_scale_2`` exists, an
    ``...ngram_embedding.shard_N`` prefix falls back to the table's shared
    ``...ngram_embedding.weight_scale_2`` (the PLE shards have no per-shard
    scalar); a matrix with no global scale anywhere gets ``1.0``.
    """
    names = _source_names(source)
    weight = _source_tensor(source, f"{base}.weight")
    spec = detect_nvfp4_scale(names, base, tuple(weight.shape), group_size)

    if spec.encoding == CSF_ENCODING:
        # Decision A: decode on the host, in memory, straight into the flat
        # [rows, cols] E4M3 plane. Nothing is written anywhere.
        fixed = _source_tensor(source, spec.fixed_name)
        exceptions = _source_tensor(source, spec.exceptions_name)
        plane = decode_csf_scale_plane(fixed, exceptions, spec.rows, spec.cols)
    else:
        plane = _as_uint8(_source_tensor(source, spec.scale_name)).contiguous()
    layout = _resolve_layout(layout)
    scales = plane.view(torch.float8_e4m3fn).unsqueeze(0)

    if scale2_name is None:
        scale2_name = f"{base}{GLOBAL_SCALE_SUFFIX}"
    if scale2_name not in names and ".ngram_embedding.shard_" in base:
        # The PLE n-gram table stores one global scale for all 128 shards.
        scale2_name = base.split(".ngram_embedding.shard_", 1)[0] + ".ngram_embedding"
        scale2_name += GLOBAL_SCALE_SUFFIX
    if scale2_name in names:
        scale2 = torch.as_tensor(_source_tensor(source, scale2_name), dtype=torch.float32).reshape(-1)[:1]
    else:
        scale2 = torch.ones(1, dtype=torch.float32)

    packed, scales = _resident(_as_uint8(weight).contiguous().unsqueeze(0), scales, layout)
    return NvFp4PackedWeights(packed, scales, scale2, layout)


def load_experts(
    source: Mapping[str, Any] | Any,
    prefix: str,
    num_experts: int,
    *,
    projection: str = "gate_proj",
    group_size: int = NVFP4_GROUP_SIZE,
    experts_attr: str = "experts",
    dtype: torch.dtype = torch.float32,
    layout: str | None = None,
) -> NvFp4PackedWeights:
    """Assemble one projection of a layer's routed experts.

    ``prefix`` is the MoE module (e.g. ``model.language_model.layers.0.mlp``)
    and the checkpoint names are ``{prefix}.{experts_attr}.{e}.{projection}``.
    Every expert's scales are decoded on the host (CSF -> plain E4M3) while the
    weights stay packed, so the result is the resident layout the GEMV kernel
    consumes: ``packed [E, N, K/2]`` U8, ``scales [E, N, K/16]`` E4M3,
    ``scale2 [E]`` F32.

    ``dtype`` upcasts ``scale2`` only -- packed and scale planes keep their
    storage class.  ``layout`` selects the resident TPC layout (default:
    ``VLLM_QWEN4EXP_TPC_LAYOUT``, i.e. the Track 3 GEMV's declared layout);
    the CSF decode and the preshuffle both happen here, once, in memory.
    """
    layout = _resolve_layout(layout)
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    names = _source_names(source)
    packed_rows: list[torch.Tensor] = []
    scale_rows: list[torch.Tensor] = []
    scale2_rows: list[torch.Tensor] = []
    spec: NvFp4ScaleSpec | None = None
    for expert in range(num_experts):
        base = f"{prefix}.{experts_attr}.{expert}.{projection}"
        weight = _source_tensor(source, f"{base}.weight")
        if spec is None:
            spec = detect_nvfp4_scale(names, base, tuple(weight.shape), group_size)
        if spec.encoding == CSF_ENCODING:
            plane = decode_csf_scale_plane(
                _source_tensor(source, f"{base}{CSF_FIXED_SUFFIX}"),
                _source_tensor(source, f"{base}{CSF_EXCEPTIONS_SUFFIX}"),
                spec.rows,
                spec.cols,
            )
        else:
            plane = _as_uint8(_source_tensor(source, f"{base}{PLAIN_SCALE_SUFFIX}")).contiguous()
        packed_rows.append(_as_uint8(weight).contiguous())
        scale_rows.append(plane.view(torch.float8_e4m3fn))
        scale2 = _source_tensor(source, f"{base}{GLOBAL_SCALE_SUFFIX}")
        scale2_rows.append(torch.as_tensor(scale2, dtype=torch.float32).reshape(1))

    # ``torch.stack`` allocated both planes, so they are already owned: the
    # preshuffle must not copy them again.
    return NvFp4PackedWeights(
        *_resident(torch.stack(packed_rows, dim=0),
                   torch.stack(scale_rows, dim=0),
                   layout,
                   copy=False),
        torch.cat(scale2_rows).to(dtype),
        layout,
    )


def load_expert_w13(
    source: Mapping[str, Any] | Any,
    prefix: str,
    num_experts: int,
    *,
    group_size: int = NVFP4_GROUP_SIZE,
    dtype: torch.dtype = torch.float32,
    layout: str | None = None,
) -> NvFp4PackedWeights:
    """``gate_proj`` and ``up_proj`` stacked along N (``[E, 2N, K/2]``).

    ModelOpt quantized the fused ``w13`` pair against one global scale, so both
    projections must agree on ``weight_scale_2``; a mismatch would silently
    dequantize half of the fused matrix by the wrong factor.
    """
    layout = _resolve_layout(layout)
    gate = load_experts(source, prefix, num_experts, projection="gate_proj",
                        group_size=group_size, dtype=dtype, layout=layout)
    up = load_experts(source, prefix, num_experts, projection="up_proj",
                      group_size=group_size, dtype=dtype, layout=layout)
    if not torch.allclose(gate.scale2.float(), up.scale2.float(), rtol=1e-6, atol=0.0):
        raise ValueError(f"{prefix}: gate_proj and up_proj disagree on weight_scale_2 "
                         f"({gate.scale2.tolist()[:4]} vs {up.scale2.tolist()[:4]}); they cannot "
                         "share one fused NVFP4 global scale")
    return NvFp4PackedWeights(
        torch.cat([gate.packed, up.packed], dim=1).contiguous(),
        torch.cat([gate.scales, up.scales], dim=1).contiguous(),
        gate.scale2,
        layout,
    )


def load_w4a16_as_bf16(
    source: Mapping[str, Any] | Any,
    base: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    group_size: int = NVFP4_GROUP_SIZE,
) -> torch.Tensor:
    """Dequantize one W4A16_NVFP4 matrix to ``[N, K]`` (vision MLP / merger)."""
    names = _source_names(source)
    weight = _source_tensor(source, f"{base}.weight")
    spec = detect_nvfp4_scale(names, base, tuple(weight.shape), group_size)
    if spec.encoding != PLAIN_ENCODING:
        raise ValueError(f"{base}: W4A16 load expects a plain E4M3 scale plane, got CSF")
    scale2 = _source_tensor(source, f"{base}{GLOBAL_SCALE_SUFFIX}")
    return dequant_w4a16_nvfp4(
        weight,
        _source_tensor(source, spec.scale_name),
        scale2,
        dtype=dtype,
        group_size=group_size,
    )


def load_mxfp8_as_fp8(
    source: Mapping[str, Any] | Any,
    base: str,
    *,
    group_size: int = MXFP8_GROUP_SIZE,
    gaudi2_correction: bool = False,
    mode: str = "pow2",
    diagnostics: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one MXFP8 matrix as ``(weight_e4m3 [N, K], weight_scale [N] f32)``."""
    weight = _source_tensor(source, f"{base}.weight")
    scale = _source_tensor(source, f"{base}{PLAIN_SCALE_SUFFIX}")
    return mxfp8_to_fp8_per_channel(
        weight,
        scale,
        group_size=group_size,
        mode=mode,
        gaudi2_correction=gaudi2_correction,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# MIXED_PRECISION dispatch (pure; mirrors vLLM's ModelOptMixedPrecisionConfig)
# ---------------------------------------------------------------------------
def is_layer_ignored(prefix: str, ignore: Sequence[str]) -> bool:
    """ModelOpt ``ignore`` matching: exact, substring, then wildcard."""
    if not ignore:
        return False
    if prefix in ignore:
        return True
    for pattern in ignore:
        if fnmatch.fnmatchcase(prefix, pattern):
            return True
        if pattern in prefix:
            return True
    return False


def _prefix_candidates(prefix: str) -> tuple[str, ...]:
    """The same alias expansion vLLM applies for this checkpoint's naming."""
    candidates = [prefix]
    if prefix.endswith(".lm_head"):
        candidates.append("lm_head")
    if prefix.startswith("language_model.model."):
        candidates.append("model.language_model." + prefix[len("language_model.model."):])
    elif prefix.startswith("model.language_model."):
        candidates.append("language_model.model." + prefix[len("model.language_model."):])
    return tuple(dict.fromkeys(candidates))


_FUSED_PROJECTION_SHARDS = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}


def resolve_quant_algo(
    prefix: str,
    quantized_layers: Mapping[str, Mapping[str, Any]],
    packed_modules_mapping: Mapping[str, Sequence[str]] | None = None,
) -> str | None:
    """Resolve the ``quant_algo`` a vLLM layer prefix is declared with.

    Mirrors ``ModelOptMixedPrecisionConfig._resolve_quant_algo`` (direct,
    packed/fused, child-prefix, then parent-fallback for fused projections) so
    the host-only tests can check the real ``quantized_layers`` mapping without
    importing vLLM.  Returns the upper-cased algo or ``None``.
    """
    def lookup(candidate: str) -> str | None:
        entry = quantized_layers.get(candidate)
        if entry is None:
            return None
        return str(entry["quant_algo"]).upper()

    for candidate in _prefix_candidates(prefix):
        algo = lookup(candidate)
        if algo is not None:
            return algo

    proj_name = prefix.rsplit(".", 1)[-1]
    mapping = packed_modules_mapping or {}
    if proj_name in mapping:
        base = prefix.rsplit(".", 1)[0]
        algos = set()
        for base_candidate in _prefix_candidates(base):
            for shard_name in mapping[proj_name]:
                algo = lookup(f"{base_candidate}.{shard_name}")
                if algo is not None:
                    algos.add(algo)
        if len(algos) == 1:
            return algos.pop()
        if len(algos) > 1:
            raise ValueError(f"mixed quant_algo within fused layer {prefix}: {algos}")

    for candidate in _prefix_candidates(prefix):
        prefix_dot = candidate + "."
        for key, info in quantized_layers.items():
            if key.startswith(prefix_dot):
                # First match in config order, exactly like vLLM's strategy 3.
                # A coarse prefix (e.g. "...layers.0.mlp") therefore resolves to
                # the first child declared for it -- here the NVFP4 expert
                # group, which is the algorithm a MoE block wants to see.
                return str(info["quant_algo"]).upper()

    if prefix.endswith(".experts"):
        parent_dot = prefix.rsplit(".experts", 1)[0] + "."
        for key, info in quantized_layers.items():
            if key.startswith(parent_dot):
                return str(info["quant_algo"]).upper()

    shard_names = _FUSED_PROJECTION_SHARDS.get(proj_name)
    if shard_names is not None:
        for candidate in _prefix_candidates(prefix):
            parent_dot = candidate.rsplit(".", 1)[0] + "."
            algos = set()
            for shard_name in shard_names:
                algo = lookup(f"{parent_dot}{shard_name}")
                if algo is not None:
                    algos.add(algo)
            if len(algos) == 1:
                return algos.pop()
            if len(algos) > 1:
                raise ValueError(f"mixed quant_algo within fused layer {prefix}: {algos}")
    return None


def dispatch_kind(
    prefix: str,
    quantized_layers: Mapping[str, Mapping[str, Any]],
    ignore: Sequence[str] = (),
    packed_modules_mapping: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """Which loader/quant method a prefix needs on Gaudi2.

    ``NVFP4`` and ``W4A16_NVFP4`` both keep their weights packed when the
    prefix is an expert group (the MTP experts are declared W4A16_NVFP4 but are
    still 4-bit routed experts); W4A16 outside an expert group is a small
    vision matrix and is dequantized to BF16 at load.
    """
    if is_layer_ignored(prefix, ignore):
        return KIND_UNQUANTIZED
    algo = resolve_quant_algo(prefix, quantized_layers, packed_modules_mapping)
    if algo is None and ".experts" in prefix:
        # Convenience beyond vLLM's config (which keys quantized_layers at the
        # group level): a per-expert prefix resolves through its ".experts"
        # ancestor so callers can query any module of the group.
        ancestor = prefix.split(".experts", 1)[0] + ".experts"
        algo = resolve_quant_algo(ancestor, quantized_layers, packed_modules_mapping)
    if algo is None and ".ngram_embedding.shard_" in prefix:
        # Same convenience for the PLE n-gram table, whose 128 shards are
        # declared once at "...ngram_embedding" and loaded one shard at a time.
        # Without this every shard would look unquantized, and a U8 packed
        # table read as BF16 would be silently wrong.
        ancestor = prefix.split(".ngram_embedding.shard_", 1)[0] + ".ngram_embedding"
        algo = resolve_quant_algo(ancestor, quantized_layers, packed_modules_mapping)
    if algo is None:
        return KIND_UNQUANTIZED
    if algo == "MXFP8":
        return KIND_MXFP8
    if algo == "NVFP4":
        return KIND_NVFP4_PACKED
    if algo == "W4A16_NVFP4":
        if _is_expert_prefix(prefix):
            return KIND_NVFP4_PACKED
        return KIND_W4A16_BF16
    if algo in ("FP8", "FP8_PER_CHANNEL_PER_TOKEN", "FP8_PB_WO"):
        return KIND_FP8
    raise ValueError(f"unsupported quant_algo {algo!r} for {prefix}")


def _is_expert_prefix(prefix: str) -> bool:
    return prefix.endswith(".experts") or ".experts." in prefix or prefix.endswith(".ngram_embedding")
