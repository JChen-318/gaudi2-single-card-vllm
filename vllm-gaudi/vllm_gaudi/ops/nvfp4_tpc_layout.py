# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The NVFP4 packed-weight layout the Track 3 TPC GEMV declares, and its preshuffle.

Why this module exists
----------------------
Two pieces of code have to agree on the *byte order* of a 4-bit expert plane:

* the **loader** (``vllm_gaudi/ops/modelopt_csf.py``, driven by
  ``NvFp4ExpertStore.load_from_source``), which turns checkpoint tensors into
  resident HBM planes once per model load;
* the **decode GEMV** (``native/qwen4exp-nvfp4-moe``), which loads those planes
  with 256-byte TPC vector loads and never has a Python side.

Neither can import the other, so the contract lives here as a versioned
constant plus pure transform functions.  ``modelopt_csf`` imports this module
(by package or by file path, so it also works from the host-only validation
suite) and applies :func:`preshuffle_packed` / :func:`preshuffle_scales` once at
load time; the kernel's host GlueCode enforces the same geometry in
``qwen4exp_nvfp4_moe_gemv_bf16_gaudi2.cpp``.

The declared layout (``TPC_GEMV_LAYOUT``)
-----------------------------------------
For one projection of one layer the resident planes are

==============  =========================  ==================================
tensor          torch shape                meaning
==============  =========================  ==================================
``packed``      ``[E, N, K/2]`` uint8      two E2M1 nibbles per byte
``scales``      ``[E, N, K/16]`` e4m3       one E4M3 block scale per 16 inputs
``scale2``      ``[E]`` float32            one global scale per expert
==============  =========================  ==================================

with the *last* torch axis contiguous.  Synapse reverses torch axes, so the TPC
index space of ``packed`` is ``(K/2, N, E)``: dimension 0 — the fastest, the one
a ``v_u8_ld_tnsr_b`` walks — is the packed K axis.  That is exactly what both
in-tree GEMV kernels address::

    v_u8_ld_tnsr_b((int5){tile * 256, row, expert, 0, 0}, packed)   # 512 k
    v_u8_ld_tnsr_b((int5){tile *  32, row, expert, 0, 0}, scales)   # 8 scales

Worked byte example (K = 32, N = 1, one expert)
----------------------------------------------
``packed[0, 0, :]`` holds K/2 = 16 bytes.  Byte ``j`` carries two nibbles: the
**low** nibble is ``k = 2j`` and the **high** nibble is ``k = 2j + 1``::

    byte   0     1     2     ...   15
    k     0,1   2,3   4,5   ...  30,31
    low   k=0   k=2   k=4         k=30
    high  k=1   k=3   k=5         k=31

Nibble values are E2M1 magnitudes ``(0, .5, 1, 1.5, 2, 3, 4, 6)`` with bit 3 as
the sign (so ``0x7`` = +6 and ``0xF`` = -6).  Scale byte ``c`` covers
``k in [16c, 16c + 16)``, i.e. ``scales[0, 0, c]`` multiplies
``packed[0, 0, 8c .. 8c+7]``.

The TPC vector geometry on top of that: 256 bytes = 512 ``k`` = four 64-byte
dual groups of 128 ``k`` each (``TPC_K_PER_DUAL_GROUP``), and the scale plane
needs 512/16 = 32 bytes per vector (``TPC_SCALE_BYTES_PER_VECTOR``).

Alignment the GEMV requires (enforced by its GlueCode, restated here)
---------------------------------------------------------------------
* ``K % 128 == 0`` -- ``K/2`` is a whole number of 64-byte dual groups, so one
  packed dual group is 64 packed bytes and never straddles a K tile;
* ``scales.shape[0] * 8 == K/2`` and ``scales.shape[1] == N`` (one scale per 16
  inputs);
* ``E % 64 == 0`` and ``N % 4 == 0`` (the kernel's block/row granularity).

:func:`check_geometry` re-derives exactly those conditions on the host, before a
single byte is copied to the device.

The preshuffle
--------------
``TPC_GEMV_LAYOUT`` keeps the checkpoint's byte order: the layout the kernel
declares is the layout the checkpoint stores, so its transform is the identity
(**verified against the GlueCode's access patterns** by
``validation/qwen4exp/test_csf_memory_path.py::
test_track3_gemv_declares_the_layout_this_loader_produces``).  What the loader must still do
is *own* the bytes: one contiguous ``[E, N, K/2]`` plane per projection, built
once, never a view of the checkpoint's mmap.  That is what
:func:`preshuffle_packed` returns for the default layout.

``TPC_GEMV_LAYOUT_PADDED`` is available for a strict "every vector load stays
inside its own row" variant: the GEMV issues 256-byte packed loads (and 32-byte
stepped 256-byte scale loads), so a row whose byte count is not a multiple of
256 over-reads into the next row (harmless inside one allocation, but it is an
over-read).  Padding the row stride to 256 bytes removes it and costs HBM:
``down_proj`` has ``K/2 = 320`` bytes, so its plane grows by 60 % -- measured on
the checkpoint's 48 MoE layers x 512 experts that is ``+11.3 GiB`` for
``down_proj`` alone (``+1.4 GiB`` more for its scale rows).  It is therefore
**opt-in**; the default stays the declared, unpadded layout.

Selecting the layout
--------------------
``VLLM_QWEN4EXP_TPC_LAYOUT`` selects the value used by the loaders:

===================  ==========================================================
value                meaning
===================  ==========================================================
``auto`` (default)   ``TPC_GEMV_LAYOUT`` -- the declared layout, no padding
``identity``         alias of ``auto`` (kept for the "no preshuffle" switch)
``tpc_e_n_k2_v1``    explicit ``TPC_GEMV_LAYOUT``
``tpc_e_n_k2_pad256_v1``  padded variant (costs the HBM described above)
===================  ==========================================================

``VLLM_QWEN4EXP_TPC_PRESHUFFLE=0`` is accepted as an alias for ``identity`` so
a run can be pinned to "never touch the bytes" without knowing layout names.
"""

from __future__ import annotations

import os
from typing import Final

import torch

__all__ = [
    "DEFAULT_LAYOUT",
    "LAYOUT_ENV",
    "NVFP4_GROUP_SIZE",
    "PRESHUFFLE_SWITCH",
    "TPC_DUAL_GROUP_BYTES",
    "TPC_GEMV_LAYOUT",
    "TPC_GEMV_LAYOUT_PADDED",
    "TPC_K_PER_DUAL_GROUP",
    "TPC_K_PER_VECTOR",
    "TPC_LAYOUT_IDENTITY",
    "TPC_LAYOUT_VERSION",
    "TPC_SCALE_BYTES_PER_VECTOR",
    "TPC_VECTOR_BYTES",
    "check_geometry",
    "choose_row_bytes",
    "geometry_report",
    "describe",
    "layout_from_env",
    "packed_row_bytes",
    "preshuffle_packed",
    "preshuffle_scales",
    "scale_row_bytes",
    "scale_row_bytes_for_layout",
    "tpc_planes",
    "unpreshuffle_packed",
    "unpreshuffle_scales",
]

#: Bumped whenever the meaning of a byte position changes; the loaders log it
#: and the kernels' checks are matched against it.
TPC_LAYOUT_VERSION: Final[int] = 1

#: The layout the Track 3 decode GEMV declares (bytes unchanged from the
#: checkpoint: torch ``[E, N, K/2]`` / ``[E, N, K/16]``, last axis contiguous).
TPC_GEMV_LAYOUT: Final[str] = "tpc_e_n_k2_v1"

#: Same byte order, but every row is padded to a whole number of TPC vectors.
TPC_GEMV_LAYOUT_PADDED: Final[str] = "tpc_e_n_k2_pad256_v1"

#: "Leave the checkpoint bytes exactly as they are" (no padding, no reorder).
TPC_LAYOUT_IDENTITY: Final[str] = "identity"

#: What the loaders use when nothing is selected.
DEFAULT_LAYOUT: Final[str] = TPC_GEMV_LAYOUT

LAYOUT_ENV: Final[str] = "VLLM_QWEN4EXP_TPC_LAYOUT"
PRESHUFFLE_SWITCH: Final[str] = "VLLM_QWEN4EXP_TPC_PRESHUFFLE"

# -- TPC vector geometry -----------------------------------------------------
#: One ``uchar256`` / ``bfloat256`` register.
TPC_VECTOR_BYTES: Final[int] = 256
#: One of the four byte groups inside a TPC vector (``mov_dual_group`` granularity).
TPC_DUAL_GROUP_BYTES: Final[int] = 64
#: E2M1 nibbles covered by one 256-byte packed load.
TPC_K_PER_VECTOR: Final[int] = 2 * TPC_VECTOR_BYTES  # 512
#: K elements covered by one 64-byte dual group (``K_PER_PART`` in the kernel).
TPC_K_PER_DUAL_GROUP: Final[int] = 2 * TPC_DUAL_GROUP_BYTES  # 128
#: E4M3 scale bytes consumed by one packed vector.
TPC_SCALE_BYTES_PER_VECTOR: Final[int] = TPC_K_PER_VECTOR // 16  # 32
#: One E4M3 block scale per 16 inputs (NVFP4).
NVFP4_GROUP_SIZE: Final[int] = 16

_LAYOUTS = (TPC_GEMV_LAYOUT, TPC_GEMV_LAYOUT_PADDED, TPC_LAYOUT_IDENTITY)
_ALIASES = {
    "auto": TPC_GEMV_LAYOUT,
    "default": TPC_GEMV_LAYOUT,
    "identity": TPC_LAYOUT_IDENTITY,
    "natural": TPC_LAYOUT_IDENTITY,
    "none": TPC_LAYOUT_IDENTITY,
    TPC_GEMV_LAYOUT: TPC_GEMV_LAYOUT,
    TPC_GEMV_LAYOUT_PADDED: TPC_GEMV_LAYOUT_PADDED,
}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def packed_row_bytes(k: int) -> int:
    """Packed bytes per output row: two E2M1 nibbles per byte."""
    if k <= 0 or k % 2:
        raise ValueError(f"K must be a positive even number, got {k}")
    return k // 2


def scale_row_bytes(k: int) -> int:
    """E4M3 scale bytes per output row: one per 16 inputs."""
    if k <= 0 or k % NVFP4_GROUP_SIZE:
        raise ValueError(f"K must be a positive multiple of {NVFP4_GROUP_SIZE}, got {k}")
    return k // NVFP4_GROUP_SIZE


def choose_row_bytes(k: int, layout: str = DEFAULT_LAYOUT) -> int:
    """Resident bytes per row for ``packed`` (and, scaled, for ``scales``).

    ``TPC_GEMV_LAYOUT`` keeps ``K/2``.  ``TPC_GEMV_LAYOUT_PADDED`` rounds the
    row up to :data:`TPC_VECTOR_BYTES` so a 256-byte vector load cannot leave
    its own row; ``scales`` is rounded to :data:`TPC_DUAL_GROUP_BYTES` there.
    """
    layout = _canonical(layout)
    if layout == TPC_GEMV_LAYOUT_PADDED:
        row = packed_row_bytes(k)
        return (row + TPC_VECTOR_BYTES - 1) // TPC_VECTOR_BYTES * TPC_VECTOR_BYTES
    return packed_row_bytes(k)


def scale_row_bytes_for_layout(k: int, layout: str = DEFAULT_LAYOUT) -> int:
    """Resident scale bytes per row for a layout (see :func:`choose_row_bytes`)."""
    layout = _canonical(layout)
    row = scale_row_bytes(k)
    if layout == TPC_GEMV_LAYOUT_PADDED:
        return (row + TPC_DUAL_GROUP_BYTES - 1) // TPC_DUAL_GROUP_BYTES * TPC_DUAL_GROUP_BYTES
    return row


def geometry_report(
    experts: int,
    rows: int,
    k: int,
    *,
    layout: str = DEFAULT_LAYOUT,
    require_kernel_caps: bool = True,
) -> dict[str, object]:
    """Derived geometry plus every Track 3 GEMV constraint this plane breaks.

    Never raises (beyond a non-positive shape): the loaders call it to log and
    to let the model fall back to the dequant path, while
    :func:`check_geometry` turns the same report into a hard failure for tests
    and for the on-card path.
    """
    layout = _canonical(layout)
    if experts <= 0 or rows <= 0 or k <= 0:
        raise ValueError(f"planes must be positive, got E={experts}, N={rows}, K={k}")
    packed_row = choose_row_bytes(k, layout)
    violations: list[str] = []
    if layout in (TPC_GEMV_LAYOUT, TPC_LAYOUT_IDENTITY) and packed_row % TPC_DUAL_GROUP_BYTES:
        violations.append(f"K/2 = {packed_row} B is not a whole number of "
                          f"{TPC_DUAL_GROUP_BYTES}-byte TPC dual groups")
    if require_kernel_caps:
        if k % TPC_K_PER_DUAL_GROUP:
            violations.append(f"K={k} is not a multiple of {TPC_K_PER_DUAL_GROUP}")
        if rows % 4:
            violations.append(f"N={rows} is not a multiple of 4 (rows per member)")
        if experts % 64:
            violations.append(f"E={experts} is not a multiple of 64 (block slots)")
    return {
        "layout": layout,
        "layout_version": TPC_LAYOUT_VERSION,
        "packed_row": packed_row,
        "scale_row": scale_row_bytes_for_layout(k, layout),
        "k_per_vector": TPC_K_PER_VECTOR,
        "packed_tiles": (packed_row + TPC_VECTOR_BYTES - 1) // TPC_VECTOR_BYTES,
        "scale_bytes_per_vector": TPC_SCALE_BYTES_PER_VECTOR,
        "ok": not violations,
        "violations": tuple(violations),
    }


def check_geometry(
    experts: int,
    rows: int,
    k: int,
    *,
    layout: str = DEFAULT_LAYOUT,
    require_kernel_caps: bool = True,
) -> dict[str, object]:
    """Validate a plane trio against the Track 3 GEMV contract.

    Returns the geometry report and raises ``ValueError`` naming every broken
    constraint; ``require_kernel_caps=False`` keeps the dual-group alignment
    check but skips the ``E % 64`` / ``N % 4`` / ``K % 128`` device caps, which
    matter for a real layer but not for a small matrix in a unit test.
    """
    report = geometry_report(experts, rows, k, layout=layout, require_kernel_caps=require_kernel_caps)
    if report["violations"]:
        raise ValueError(f"{layout} cannot feed the Track 3 GEMV: "
                         + "; ".join(report["violations"]))
    return report


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------
def preshuffle_packed(
    packed: torch.Tensor,
    layout: str = DEFAULT_LAYOUT,
    *,
    copy: bool = True,
) -> torch.Tensor:
    """``[E, N, K/2]`` checkpoint planes -> the resident TPC packed plane.

    ``TPC_GEMV_LAYOUT`` (the declared layout, default): the checkpoint byte
    order already *is* the declared one, so the transform is a contiguous,
    self-owned copy of the plane -- the GEMV's job is a layout obligation
    (``[E, N, K/2]``, K/2 on the FCD axis), not a permutation, and the identity
    permutation is what its host access pattern declares.

    ``TPC_GEMV_LAYOUT_PADDED`` returns ``[E, N, ceil(K/2, 256)]`` with each
    row's tail zeroed.  E2M1 code 0 is the value 0, so the pad cannot change a
    dot product; it only keeps every 256-byte vector load inside its own row.

    ``TPC_LAYOUT_IDENTITY`` returns ``packed`` untouched (no copy, no pad).

    ``copy=False`` skips the ownership copy for a plane the caller just built
    (``torch.stack`` of the per-expert planes already allocated it), which is
    how the expert loader avoids a second full-size copy per projection.
    """
    layout = _canonical(layout)
    if layout == TPC_LAYOUT_IDENTITY:
        return packed
    if packed.dim() != 3:
        raise ValueError(f"packed plane must be [E, N, K/2], got {tuple(packed.shape)}")
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed plane must be uint8, got {packed.dtype}")
    row_bytes = int(packed.shape[2])
    k = row_bytes * 2
    want = choose_row_bytes(k, layout)
    if want == row_bytes:
        # Declared layout: one contiguous buffer that the loader owns.
        contiguous = packed.contiguous()
        return contiguous.clone() if copy else contiguous
    out = torch.zeros((int(packed.shape[0]), int(packed.shape[1]), want), dtype=torch.uint8)
    out[:, :, :row_bytes] = packed
    return out


def unpreshuffle_packed(plane: torch.Tensor, k: int, layout: str = DEFAULT_LAYOUT) -> torch.Tensor:
    """Inverse of :func:`preshuffle_packed`: resident plane -> ``[E, N, K/2]``."""
    layout = _canonical(layout)
    if layout == TPC_LAYOUT_IDENTITY:
        return plane
    row_bytes = packed_row_bytes(k)
    if plane.dim() != 3 or int(plane.shape[2]) < row_bytes:
        raise ValueError(f"resident plane {tuple(plane.shape)} cannot hold K/2 = {row_bytes} bytes")
    if int(plane.shape[2]) == row_bytes:
        return plane.contiguous()
    return plane[:, :, :row_bytes].contiguous()


def _scale_bytes(scales: torch.Tensor) -> torch.Tensor:
    """Byte view of an E4M3 / uint8 scale plane."""
    if scales.dtype == torch.uint8:
        return scales
    if scales.dtype == torch.float8_e4m3fn:
        return scales.view(torch.uint8)
    raise TypeError(f"scale plane must be uint8 or float8_e4m3fn, got {scales.dtype}")


def preshuffle_scales(
    scales: torch.Tensor,
    layout: str = DEFAULT_LAYOUT,
    *,
    copy: bool = True,
) -> torch.Tensor:
    """``[E, N, K/16]`` E4M3 scale planes -> the resident TPC scale plane.

    Same rule as :func:`preshuffle_packed`: identical bytes for the declared
    layout (one contiguous copy, dtype preserved), row-tail zero padding for the
    padded layout (byte 0x00 is E4M3 ``0.0``, so padding is inert).
    ``copy=False`` behaves as in :func:`preshuffle_packed`.
    """
    layout = _canonical(layout)
    if layout == TPC_LAYOUT_IDENTITY:
        return scales
    if scales.dim() != 3:
        raise ValueError(f"scale plane must be [E, N, K/16], got {tuple(scales.shape)}")
    byte_view = _scale_bytes(scales)
    row_bytes = int(byte_view.shape[2])
    want = scale_row_bytes_for_layout(row_bytes * NVFP4_GROUP_SIZE, layout)
    if want == row_bytes:
        contiguous = byte_view.contiguous()
        return (contiguous.clone() if copy else contiguous).view(scales.dtype)
    out = torch.zeros((int(byte_view.shape[0]), int(byte_view.shape[1]), want), dtype=torch.uint8)
    out[:, :, :row_bytes] = byte_view
    return out.view(scales.dtype)


def unpreshuffle_scales(scales: torch.Tensor, k: int, layout: str = DEFAULT_LAYOUT) -> torch.Tensor:
    """Inverse of :func:`preshuffle_scales`: resident plane -> ``[E, N, K/16]``."""
    layout = _canonical(layout)
    if layout == TPC_LAYOUT_IDENTITY:
        return scales
    row_bytes = scale_row_bytes(k)
    if int(scales.shape[2]) < row_bytes:
        raise ValueError(f"resident scale plane {tuple(scales.shape)} cannot hold K/16 = {row_bytes}")
    if int(scales.shape[2]) == row_bytes:
        return scales.contiguous()
    return scales[:, :, :row_bytes].contiguous()


def tpc_planes(
    packed: torch.Tensor,
    scales: torch.Tensor,
    k: int,
    *,
    layout: str = DEFAULT_LAYOUT,
    copy: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply both halves of the preshuffle once: ``(packed, scales)``, both resident.

    The row widths are cross-checked against ``k`` first, so a plane whose
    scale row does not describe the same K as its packed row fails here rather
    than inside a kernel.
    """
    if packed_row_bytes(k) != int(packed.shape[-1]):
        raise ValueError(f"packed row {int(packed.shape[-1])} != K/2 = {packed_row_bytes(k)} for K={k}")
    if scale_row_bytes(k) != int(scales.shape[-1]):
        raise ValueError(f"scale row {int(scales.shape[-1])} != K/16 = {scale_row_bytes(k)} for K={k}")
    return preshuffle_packed(packed, layout, copy=copy), preshuffle_scales(scales, layout, copy=copy)


# ---------------------------------------------------------------------------
# Selection / logging
# ---------------------------------------------------------------------------
def _canonical(layout: str | None) -> str:
    if layout is None:
        return DEFAULT_LAYOUT
    key = str(layout).strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(f"unknown TPC layout {layout!r}; expected one of {sorted(_LAYOUTS)}")


def layout_from_env(env: dict[str, str] | None = None) -> str:
    """The layout the loaders should apply, from the process environment."""
    environ = os.environ if env is None else env
    preshuffle = str(environ.get(PRESHUFFLE_SWITCH, "1")).strip().lower()
    if preshuffle in ("0", "false", "off", "no"):
        return TPC_LAYOUT_IDENTITY
    return _canonical(environ.get(LAYOUT_ENV, DEFAULT_LAYOUT))


def describe(layout: str = DEFAULT_LAYOUT, k: int | None = None) -> str:
    """One-line, log-friendly summary of a layout (optionally for one ``K``)."""
    layout = _canonical(layout)
    if layout == TPC_LAYOUT_IDENTITY:
        text = (f"{layout} (v{TPC_LAYOUT_VERSION}): checkpoint bytes untouched -- "
                f"packed [E,N,K/2] u8 low-nibble=even-k, scales [E,N,K/16] e4m3")
    else:
        text = (f"{layout} (v{TPC_LAYOUT_VERSION}): packed [E,N,K/2] u8 low-nibble=even-k, "
                f"scales [E,N,K/16] e4m3, TPC dims (K/2,N,E), "
                f"{TPC_K_PER_VECTOR} k per {TPC_VECTOR_BYTES}-byte vector")
    if k is not None:
        text += (f"; K={k} -> packed row {choose_row_bytes(k, layout)} B, "
                 f"scale row {scale_row_bytes_for_layout(k, layout)} B")
    return text
