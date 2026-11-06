# SPDX-License-Identifier: Apache-2.0
"""Python-side contract for the Track C NVFP4 MoE decode GEMV.

The kernel lives in ``native/qwen4exp-nvfp4-moe`` and is registered as
``torch.ops.custom_op.custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2`` by
``hpu_qwen4exp_nvfp4_moe_pt2*.so`` (built by ``scripts/build_native.sh``
into ``build/native/``). This module is what a decode path (Track 4) imports to
decide whether that operator can serve a call, and to make the call:

    import sys
    sys.path.insert(0, "native/qwen4exp-nvfp4-moe")   # the directory has a dash
    from qwen4exp_nvfp4_moe_ops import available, gemv, OP_GEMV, LAYOUT_PACKED

    spec = GemvSpec(slots=10, experts=512, rows=1280, k=2560, device="hpu")
    supported, why = available(spec)
    if supported:
        y = gemv(ids, packed, scales, x, scale2)

``available()`` is pure Python: it never imports ``torch.hpu`` or
``habana_frameworks``, so it is safe on a host without the device (and is what
``validation/qwen4exp/test_nvfp4_moe_ops.py`` exercises on CPU). Only
``load_library()`` touches the HPU extension, and only when the device path asks
for it.

Operator (argument order is the host GlueCode's tensor order; no scalars):

    custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2(ids, packed, scales, x, scale2) -> Tensor
      ids    [S]        I32   selected expert per routed slot; < 0 or >= E -> zero row
      packed [E,N,K/2]  U8    E2M1 nibbles, low nibble = even k
      scales [E,N,K/16] U8    per-16 block scales, the bytes of a float8_e4m3fn plane
      x      [S,K]      BF16  routed activations
      scale2 [E]        F32   per-expert global scale, applied once after the F32 dot
    -> [S,N] BF16

Layouts (the names are the contract; the kernel reads exactly these):

* ``LAYOUT_PACKED`` = ``nvfp4_e_n_khalf``: ``[E, N, K/2]`` uint8, expert-major,
  one row-major K/2 plane per output row, low nibble of a byte = even k. This is
  the checkpoint's native layout (no TPC preshuffle); the kernel walks K with
  256-byte vector loads, i.e. 512 nibbles per load.
* ``LAYOUT_SCALES`` = ``e4m3_block16_e_n_k16``: ``[E, N, K/16]`` uint8 holding
  float8_e4m3fn bytes, one byte per 16 consecutive K elements, same row-major
  order as the packed plane. Passing the scale plane as a ``float8_e4m3fn``
  tensor is accepted and reinterpreted to uint8 without a copy (the kernel reads
  bytes); ``packed`` must already be uint8.

Geometry caps match the host glue exactly (``host/qwen4exp_nvfp4_moe_gemv_bf16_gaudi2.cpp``):
``1 <= S <= 64``, ``K % 128 == 0``, ``N % 4 == 0``, ``64 <= E <= 4096`` with
``E % 64 == 0``. The 4 is ``ROWS_PER_MEMBER``: one TPC member owns four output
rows so their weight loads issue together (see the kernel header).

Environment:

=============== ==============================================================
``VLLM_QWEN4EXP_TRACK_C_MOE``  ``0``/``false``/``off``/``no`` disables this
                               operator for selection (default on).
``VLLM_QWEN4EXP_MOE_LIB``      explicit path to the built torch-op library;
                               ``load_library()`` otherwise globs
                               ``build/native/hpu_qwen4exp_nvfp4_moe_pt2*.so``.
=============== ==============================================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

#: Operator names as registered in ``torch.ops.custom_op``.
OP_GEMV = "custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2"
OP_COUNT = "custom_qwen4exp_nvfp4_moe_count_i32_gaudi2"
OP_PREFIX = "custom_qwen4exp_nvfp4_moe_prefix_i32_gaudi2"
OP_SCATTER = "custom_qwen4exp_nvfp4_moe_scatter_i32_gaudi2"

LIB_ENV = "VLLM_QWEN4EXP_MOE_LIB"
SWITCH_ENV = "VLLM_QWEN4EXP_TRACK_C_MOE"

#: Layout names a caller can pass in a spec; the kernel reads these only.
LAYOUT_PACKED = "nvfp4_e_n_khalf"
LAYOUT_SCALES = "e4m3_block16_e_n_k16"

MAX_SLOTS = 64
MIN_EXPERTS = 64
MAX_EXPERTS = 4096
ROWS_PER_MEMBER = 4
K_ALIGN = 128
K_PER_VECTOR = 512  # one 256-byte packed load covers this many K elements

_OFF_VALUES = ("0", "false", "off", "no")
_ACCEPTED_SCALE_DTYPES = ("uint8", "float8_e4m3fn")


@dataclass(frozen=True)
class GemvSpec:
    """Everything ``available()`` needs to accept or reject one decode call."""

    slots: int
    experts: int
    rows: int
    k: int
    device: str = "hpu"
    packed_layout: str = LAYOUT_PACKED
    scales_layout: str = LAYOUT_SCALES
    packed_dtype: str = "uint8"
    scales_dtype: str = "uint8"
    x_dtype: str = "bfloat16"
    scale2_dtype: str = "float32"
    ids_dtype: str = "int32"

    @classmethod
    def from_tensors(
        cls,
        ids: Any,
        packed: Any,
        scales: Any,
        x: Any,
        scale2: Any,
        *,
        device: str | None = None,
    ) -> "GemvSpec":
        """Derive the spec from torch tensors (meta/CPU tensors work).

        Only ``shape``/``dtype``/``device`` are read: no data is touched and no
        device memory is allocated.
        """
        if packed.dim() != 3:
            raise ValueError(f"packed must be [E, N, K/2], got {tuple(packed.shape)}")
        if ids.dim() != 1:
            raise ValueError(f"ids must be [S], got {tuple(ids.shape)}")
        if x.dim() != 2:
            raise ValueError(f"x must be [S, K], got {tuple(x.shape)}")
        experts, rows, half = packed.shape
        return cls(
            slots=int(ids.numel()),
            experts=int(experts),
            rows=int(rows),
            k=int(half) * 2,
            device=device if device is not None else str(x.device.type),
            packed_dtype=_dtype_name(packed),
            scales_dtype=_dtype_name(scales),
            x_dtype=_dtype_name(x),
            scale2_dtype=_dtype_name(scale2),
            ids_dtype=_dtype_name(ids),
        )


def _dtype_name(tensor: Any) -> str:
    return str(tensor.dtype).replace("torch.", "")


def _switch_enabled(environ: Mapping[str, str]) -> bool:
    return environ.get(SWITCH_ENV, "1").strip().lower() not in _OFF_VALUES


def _op_registered(name: str = OP_GEMV) -> bool:
    """Whether the HPU custom op is registered, without importing the HPU runtime."""
    try:
        import torch
    except Exception:  # pragma: no cover - torch is a hard dependency of the port
        return False
    namespace = getattr(torch.ops, "custom_op", None)
    if namespace is None:
        return False
    return getattr(namespace, name, None) is not None


def available(
    spec: GemvSpec,
    *,
    op_registered: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[bool, str]:
    """``(True, "")`` when the Track C GEMV can serve ``spec``, else ``(False, why)``.

    The checks are the host GlueCode's caps plus the Python-side shape contract,
    so a caller can fall back (grouped bmm / dequant) instead of aborting a
    serve. ``op_registered`` and ``environ`` are injectable for tests; by
    default the switch and the operator registration are read live.
    """
    environ = os.environ if environ is None else environ
    if not _switch_enabled(environ):
        return False, f"{SWITCH_ENV} disables the Track C GEMV"
    if op_registered is None:
        op_registered = _op_registered()
    if not op_registered:
        return False, "operator not registered (build_native.sh / VLLM_QWEN4EXP_MOE_LIB)"
    if spec.packed_layout != LAYOUT_PACKED:
        return False, f"weights layout {spec.packed_layout!r} is not {LAYOUT_PACKED!r}"
    if spec.scales_layout != LAYOUT_SCALES:
        return False, f"scale layout {spec.scales_layout!r} is not {LAYOUT_SCALES!r}"
    if spec.packed_dtype != "uint8":
        return False, f"packed weights must be uint8, got {spec.packed_dtype}"
    if spec.scales_dtype not in _ACCEPTED_SCALE_DTYPES:
        return False, (f"scales must be uint8 or float8_e4m3fn bytes, "
                       f"got {spec.scales_dtype}")
    if spec.x_dtype != "bfloat16":
        return False, f"activations must be bfloat16, got {spec.x_dtype}"
    if spec.ids_dtype != "int32":
        return False, f"ids must be int32, got {spec.ids_dtype}"
    if spec.scale2_dtype != "float32":
        return False, f"scale2 must be float32, got {spec.scale2_dtype}"
    if not 1 <= spec.slots <= MAX_SLOTS:
        return False, f"S={spec.slots} outside 1..{MAX_SLOTS}"
    if spec.k <= 0 or spec.k % K_ALIGN:
        return False, f"K={spec.k} is not a positive multiple of {K_ALIGN}"
    if spec.rows <= 0 or spec.rows % ROWS_PER_MEMBER:
        return False, f"N={spec.rows} is not a positive multiple of {ROWS_PER_MEMBER}"
    if not MIN_EXPERTS <= spec.experts <= MAX_EXPERTS or spec.experts % 64:
        return False, (f"E={spec.experts} outside the 64-multiple range "
                       f"{MIN_EXPERTS}..{MAX_EXPERTS}")
    if spec.device != "hpu":
        # The operator registers HPU and Meta kernels only; CPU/meta tensors must
        # take the portable path (this is also the offline-test answer).
        return False, f"no Track C HPU kernel for device {spec.device!r}"
    return True, ""


def available_for_tensors(
    ids: Any,
    packed: Any,
    scales: Any,
    x: Any,
    scale2: Any,
    *,
    device: str | None = None,
    op_registered: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[bool, str]:
    """``available()`` plus the exact tensor-shape checks of the operator.

    ``device`` overrides the tensors' device (e.g. preflight an HPU call from
    CPU placeholders); it defaults to ``x.device.type``.
    """
    try:
        spec = GemvSpec.from_tensors(ids, packed, scales, x, scale2, device=device)
    except ValueError as error:
        return False, str(error)
    supported, why = available(spec, op_registered=op_registered, environ=environ)
    if not supported:
        return supported, why
    expected_scales = (spec.experts, spec.rows, spec.k // 16)
    if tuple(scales.shape) != expected_scales:
        return False, f"scale plane {tuple(scales.shape)} != {expected_scales}"
    if tuple(x.shape) != (spec.slots, spec.k):
        return False, f"activations {tuple(x.shape)} != {(spec.slots, spec.k)}"
    if tuple(scale2.shape) != (spec.experts,):
        return False, f"scale2 {tuple(scale2.shape)} != {(spec.experts,)}"
    return True, ""


def load_library(path: str | Path | None = None) -> bool:
    """Load the Track C torch-op library. Returns True when the op is registered.

    Imports ``habana_frameworks.torch`` first because the ``HPU`` dispatch key
    only exists after the HPU backend is registered; call this from the device
    path only (never from a host-only test). ``available()`` does not need it
    when ``op_registered`` is supplied.
    """
    if _op_registered():
        return True
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    override = os.environ.get(LIB_ENV)
    if override:
        candidates.append(Path(override))
    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend(sorted(repo_root.glob("build/native/hpu_qwen4exp_nvfp4_moe_pt2*.so")))
    import torch

    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            import habana_frameworks.torch  # noqa: F401  (registers the HPU key)
        except Exception:
            return False
        try:
            torch.ops.load_library(str(candidate))
        except Exception:
            continue
        if _op_registered():
            return True
    return False


def gemv(ids: Any, packed: Any, scales: Any, x: Any, scale2: Any) -> Any:
    """Run the fused decode GEMV; ``[S,N]`` BF16 for ``[S]`` ids.

    Raises ``RuntimeError`` with the ``available()`` reason when the operator
    cannot serve the call, so a caller can catch it and fall back. ``scales`` is
    reinterpreted from ``float8_e4m3fn`` to its byte view without a copy.
    """
    supported, why = available_for_tensors(ids, packed, scales, x, scale2)
    if not supported:
        raise RuntimeError(f"{OP_GEMV} cannot serve this call: {why}")
    import torch

    op = getattr(torch.ops.custom_op, OP_GEMV)
    scale_bytes = scales.contiguous()
    if scale_bytes.dtype.is_floating_point:  # float8_e4m3fn plane -> its bytes
        scale_bytes = scale_bytes.view(torch.uint8)
    return op(
        ids.to(dtype=torch.int32).contiguous(),
        packed.contiguous(),
        scale_bytes,
        x.contiguous(),
        scale2.contiguous(),
    )
