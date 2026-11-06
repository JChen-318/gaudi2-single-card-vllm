# SPDX-License-Identifier: Apache-2.0
"""Host-only tests for the Track C Python contract (native/qwen4exp-nvfp4-moe).

Covers ``qwen4exp_nvfp4_moe_ops.available(spec)`` - the capability check Track 4
uses to select the fused NVFP4 decode GEMV - and the tensor wrapper around the
operator. Everything runs on CPU: the module must never import ``torch.hpu`` or
``habana_frameworks`` to answer ``available()``, and the operator is only
reachable on HPU, so the positive path is checked with ``op_registered=True``
injected and the negative paths with real CPU/meta tensors.

Run: ``.venv/bin/python -m pytest -q validation/qwen4exp/test_nvfp4_moe_ops.py``
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "native" / "qwen4exp-nvfp4-moe"
sys.path.insert(0, str(NATIVE))

import qwen4exp_nvfp4_moe_ops as ops  # noqa: E402


def spec(**overrides):
    base = dict(slots=10, experts=512, rows=1280, k=2560, device="hpu")
    base.update(overrides)
    return ops.GemvSpec(**base)


def test_operator_names_and_layouts_are_the_contract():
    assert ops.OP_GEMV == "custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2"
    assert ops.LAYOUT_PACKED == "nvfp4_e_n_khalf"
    assert ops.LAYOUT_SCALES == "e4m3_block16_e_n_k16"
    assert (ops.MAX_SLOTS, ops.MIN_EXPERTS, ops.MAX_EXPERTS) == (64, 64, 4096)
    assert ops.ROWS_PER_MEMBER == 4 and ops.K_ALIGN == 128
    assert ops.LIB_ENV == "VLLM_QWEN4EXP_MOE_LIB"
    assert ops.SWITCH_ENV == "VLLM_QWEN4EXP_TRACK_C_MOE"


def test_available_accepts_the_decode_layer_shapes():
    for shape in ((10, 512, 1280, 2560), (10, 512, 2560, 640), (1, 64, 4, 128), (64, 4096, 8, 512)):
        slots, experts, rows, k = shape
        supported, why = ops.available(
            spec(slots=slots, experts=experts, rows=rows, k=k), op_registered=True, environ={})
        assert supported, (shape, why)


@pytest.mark.parametrize(
    "overrides, needle",
    [
        (dict(op_registered=False), "not registered"),
        (dict(device="cpu"), "device 'cpu'"),
        (dict(slots=0), "outside 1..64"),
        (dict(slots=65), "outside 1..64"),
        (dict(k=2624), "positive multiple of 128"),
        (dict(k=0), "positive multiple of 128"),
        (dict(rows=1282), "positive multiple of 4"),
        (dict(experts=32), "64-multiple range"),
        (dict(experts=100), "64-multiple range"),
        (dict(experts=4160), "64-multiple range"),
        (dict(packed_layout="nvfp4_preshuffled_e_n_khalf"), "weights layout"),
        (dict(scales_layout="e4m3_per_tensor"), "scale layout"),
        (dict(packed_dtype="int8"), "packed weights must be uint8"),
        (dict(scales_dtype="int8"), "scales must be uint8 or float8_e4m3fn"),
        (dict(x_dtype="float16"), "bfloat16"),
        (dict(ids_dtype="int64"), "ids must be int32"),
        (dict(scale2_dtype="bfloat16"), "scale2 must be float32"),
    ],
)
def test_available_rejects(overrides, needle):
    options = dict(overrides)
    op_registered = options.pop("op_registered", True)
    supported, why = ops.available(spec(**options), op_registered=op_registered, environ={})
    assert not supported and needle in why, why


def test_switch_environment_variable_disables_selection():
    for value in ("0", "false", "OFF", "no"):
        supported, why = ops.available(spec(), op_registered=True,
                                       environ={ops.SWITCH_ENV: value})
        assert not supported and "disables the Track C GEMV" in why
    supported, _ = ops.available(spec(), op_registered=True,
                                 environ={ops.SWITCH_ENV: "1"})
    assert supported


def test_available_never_loads_the_operator_library(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("available() must not load the HPU extension")

    monkeypatch.setattr(ops, "load_library", forbidden)
    assert ops.available(spec(), op_registered=True, environ={}) == (True, "")
    assert ops.available(spec(), op_registered=False, environ={})[0] is False


def _tensors(experts=64, rows=8, k=256, slots=4, scale_dtype=torch.uint8):
    packed = torch.zeros((experts, rows, k // 2), dtype=torch.uint8)
    scales = torch.zeros((experts, rows, k // 16), dtype=scale_dtype)
    x = torch.zeros((slots, k), dtype=torch.bfloat16)
    scale2 = torch.ones(experts, dtype=torch.float32)
    ids = torch.zeros(slots, dtype=torch.int32)
    return ids, packed, scales, x, scale2


def test_from_tensors_reads_shapes_dtypes_and_device():
    ids, packed, scales, x, scale2 = _tensors()
    derived = ops.GemvSpec.from_tensors(ids, packed, scales, x, scale2)
    assert (derived.slots, derived.experts, derived.rows, derived.k) == (4, 64, 8, 256)
    assert derived.device == "cpu"
    assert (derived.packed_dtype, derived.x_dtype) == ("uint8", "bfloat16")
    assert derived.scales_dtype == "uint8"

    supported, why = ops.available_for_tensors(ids, packed, scales, x, scale2,
                                               op_registered=True, environ={})
    assert not supported and "device 'cpu'" in why


def test_float8_scale_plane_is_accepted_and_normalized():
    ids, packed, scales, x, scale2 = _tensors(scale_dtype=torch.float8_e4m3fn)
    derived = ops.GemvSpec.from_tensors(ids, packed, scales, x, scale2, device="hpu")
    assert derived.scales_dtype == "float8_e4m3fn"
    supported, why = ops.available(derived, op_registered=True, environ={})
    assert supported, why


@pytest.mark.parametrize(
    "shape_override, needle",
    [
        (dict(scales_shape=(64, 8, 8)), "scale plane"),
        (dict(x_shape=(4, 128)), "activations"),
        (dict(scale2_shape=(32,)), "scale2"),
        (dict(ids_size=3), "activations"),
    ],
)
def test_shape_contract_rejects_mismatches(shape_override, needle):
    ids, packed, scales, x, scale2 = _tensors()
    if "scales_shape" in shape_override:
        scales = torch.zeros(shape_override["scales_shape"], dtype=torch.uint8)
    if "x_shape" in shape_override:
        x = torch.zeros(shape_override["x_shape"], dtype=torch.bfloat16)
    if "scale2_shape" in shape_override:
        scale2 = torch.ones(shape_override["scale2_shape"], dtype=torch.float32)
    if "ids_size" in shape_override:
        ids = torch.zeros(shape_override["ids_size"], dtype=torch.int32)
    supported, why = ops.available_for_tensors(ids, packed, scales, x, scale2, device="hpu",
                                               op_registered=True, environ={})
    assert not supported and needle in why, why


@pytest.mark.parametrize(
    "arg, value, needle",
    [("packed", torch.zeros((64, 8), dtype=torch.uint8), "packed must be"),
     ("ids", torch.zeros((2, 2), dtype=torch.int32), "ids must be"),
     ("x", torch.zeros((4,), dtype=torch.bfloat16), "x must be")],
)
def test_from_tensors_rejects_bad_rank(arg, value, needle):
    ids, packed, scales, x, scale2 = _tensors()
    tensors = dict(ids=ids, packed=packed, scales=scales, x=x, scale2=scale2)
    tensors[arg] = value
    with pytest.raises(ValueError, match=needle):
        ops.GemvSpec.from_tensors(**tensors)


def test_gemv_raises_a_fallback_friendly_error_on_cpu():
    ids, packed, scales, x, scale2 = _tensors()
    with pytest.raises(RuntimeError, match="cannot serve this call"):
        ops.gemv(ids, packed, scales, x, scale2)


def test_spec_is_immutable():
    with pytest.raises(Exception):
        spec().slots = 1
