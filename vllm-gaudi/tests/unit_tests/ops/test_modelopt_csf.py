# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ModelOpt CSF decoder and MIXED_PRECISION dispatch.

Everything here is CPU-only and synthetic: no Gaudi device, no checkpoint and
no Habana runtime is touched.  The three on-disk encodings of a ModelOpt
``MIXED_PRECISION`` checkpoint are exercised on hand-built tensors:

* NVFP4 with a plain E4M3 scale plane (what the converted export stores),
* NVFP4 with CSF-compressed scales (what the original NVIDIA export stores),
* MXFP8 with UE8M0 block scales,

plus the Gaudi2 conversions that ``hpu_modelopt.py`` applies at load time and
the per-prefix factory that picks the quant method.

The CSF encoder in this file is a *reference* encoder written from the
documented layout; it is only used to round-trip the decoder.
"""

from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm_gaudi.ops import modelopt_csf as csf
from vllm_gaudi.ops.hpu_modelopt import (
    HPUModelOptMixedPrecisionConfig,
    HPUModelOptMxFp8LinearMethod,
    HPUModelOptW4A16NvFp4LinearMethod,
    RoutedExperts,
    _register_mixed_precision_platform_support,
    hpu_modelopt_method_for,
    modelopt,
)

# ---------------------------------------------------------------------------
# synthetic checkpoint description (prefix -> declared algorithm)
# ---------------------------------------------------------------------------
QUANTIZED_LAYERS = {
    "model.language_model.layers.0.mlp.experts": {
        "quant_algo": "NVFP4",
        "group_size": 16,
        "weight_scale_encoding": "csf",
    },
    "model.language_model.layers.0.mlp.shared_expert.gate_proj": {"quant_algo": "MXFP8", "group_size": 32},
    "model.language_model.layers.0.self_attn.q_proj": {"quant_algo": "MXFP8", "group_size": 32},
    "model.language_model.layers.0.self_attn.k_proj": {"quant_algo": "MXFP8", "group_size": 32},
    "model.language_model.layers.0.self_attn.v_proj": {"quant_algo": "MXFP8", "group_size": 32},
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding": {"quant_algo": "NVFP4", "group_size": 16},
    "model.visual.blocks.0.mlp.linear_fc1": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
    "mtp.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
}
IGNORE = ["lm_head", "mtp.fc_embedding", "mtp.pre_fc_norm_*", "mtp.layers.0.mlp.shared_expert*"]

QUANT_CONFIG = {
    "quant_method": "modelopt",
    "quant_algo": "MIXED_PRECISION",
    "ignore": IGNORE,
    "quantized_layers": QUANTIZED_LAYERS,
}


# ---------------------------------------------------------------------------
# synthetic tensor builders
# ---------------------------------------------------------------------------
def _plane(rows: int, cols: int, seed: int, low: int = 0x60, high: int = 0x78) -> torch.Tensor:
    """A realistic E4M3 scale plane: bytes clustered in a 16-wide window."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(low, high, (rows, cols), generator=generator, dtype=torch.int64).to(torch.uint8)


def encode_csf(plane: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference encoder for the documented CSF layout.

    ``base`` is the row minimum (clamped so ``base + 15`` stays a byte), the
    payload holds one 4-bit delta per column (low nibble first), and every
    value outside ``[base, base + 15]`` becomes an exception record.
    """
    plane_i = plane.to(torch.int32)
    rows, cols = plane_i.shape
    assert rows % 16 == 0 and cols % 2 == 0
    groups = rows // 16
    view = plane_i.reshape(groups, 16, cols)
    base = view.amin(dim=-1).clamp(max=255 - 15)
    delta = view - base.unsqueeze(-1)
    inside = (delta >= 0) & (delta <= 15)
    nibbles = torch.where(inside, delta, torch.zeros_like(delta)).to(torch.uint8)
    payload = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
    fixed = torch.cat([base.to(torch.uint8), payload.reshape(groups, 8 * cols)], dim=1)

    flat_inside = inside.reshape(-1)
    index = torch.nonzero(~flat_inside, as_tuple=False).flatten().to(torch.int64)
    value = plane_i.reshape(-1).index_select(0, index)
    records = (value << 24) | (((index >> 16) & 1) << 16) | (index & 0xFFFF)
    return fixed, records.to(torch.uint32)


def _packed_nvfp4(rows: int, k: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (rows, k // 2), generator=generator, dtype=torch.int64).to(torch.uint8)


# ---------------------------------------------------------------------------
# lookup tables and nibble order
# ---------------------------------------------------------------------------
def test_e2m1_lut_matches_the_encoding():
    table = csf.e2m1_lut()
    assert table.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0,
                              -4.0, -6.0]


def test_e4m3_lut_decodes_known_bytes():
    table = csf.e4m3_lut()
    assert table[0x00].item() == 0.0
    assert table[0x38].item() == 1.0  # 1.0 in OCP E4M3
    assert table[0x7E].item() == csf.E4M3_MAX  # 448
    assert torch.isnan(table[0x7F]) and torch.isnan(table[0xFF])


def test_unpack_e2m1_codes_is_low_nibble_first():
    packed = torch.tensor([[0x21, 0xF0]], dtype=torch.uint8)
    assert csf.unpack_e2m1_codes(packed).tolist() == [[1, 2, 0, 15]]


def test_nvfp4_block_saturation_ratio():
    # block_saturation_ratio counts one block per 16 columns of every row.
    codes = torch.zeros(1, 32, dtype=torch.uint8)
    codes[0, 0] = 7  # magnitude 6 in the first block
    codes[0, 16] = 7  # magnitude 6 in the second block
    codes[0, 17] = 6  # magnitude 4, still saturated because of the 7
    assert csf.nvfp4_block_saturation_ratio(codes) == 1.0
    codes[0, 16] = 3  # second block peaks at 1.5
    assert csf.nvfp4_block_saturation_ratio(codes) == 0.5


# ---------------------------------------------------------------------------
# CSF decode: documented layout, synthetic round-trip, failure modes
# ---------------------------------------------------------------------------
def test_csf_fixed_plane_shape_matches_the_export():
    # The original NVIDIA export stores U8 [rows/16, 16 + 8*cols] for a
    # [rows, cols] plane (640x160 -> [40, 1296] on a real gate_proj).
    assert csf.csf_fixed_plane_shape(640, 160) == (40, 1296)
    assert csf.csf_fixed_plane_shape(2560, 40) == (160, 336)
    with pytest.raises(ValueError, match="multiple of 16"):
        csf.csf_fixed_plane_shape(600, 160)


@pytest.mark.parametrize("rows,cols", [(64, 8), (256, 40), (2560, 40)])
def test_csf_decode_roundtrips_the_reference_encoder(rows, cols):
    plane = _plane(rows, cols, seed=rows + cols)
    fixed, exceptions = encode_csf(plane)
    assert tuple(fixed.shape) == csf.csf_fixed_plane_shape(rows, cols)
    assert exceptions.numel() > 0
    decoded = csf.decode_csf_scale_plane(fixed, exceptions, rows, cols)
    assert decoded.dtype == torch.uint8
    assert torch.equal(decoded, plane)


def test_csf_exceptions_are_load_bearing():
    plane = _plane(64, 8, seed=5, low=0x40, high=0x80)  # wider than one window
    fixed, exceptions = encode_csf(plane)
    without = csf.decode_csf_scale_plane(fixed, torch.zeros(0, dtype=torch.uint32), 64, 8)
    assert not torch.equal(without, plane)
    assert csf.count_csf_window_violations(fixed, exceptions, 64, 8) == 0


def test_csf_wrong_nibble_order_is_detected():
    plane = _plane(32, 4, seed=9)
    fixed, exceptions = encode_csf(plane)
    groups = 32 // 16
    base = fixed[:, :16].to(torch.int16)
    payload = fixed[:, 16:].reshape(groups, 16, 4 // 2)
    swapped = torch.empty(groups, 16, 4, dtype=torch.int16)
    swapped[..., 0::2] = (payload >> 4).to(torch.int16)
    swapped[..., 1::2] = (payload & 0x0F).to(torch.int16)
    wrong = (base.unsqueeze(-1) + swapped).to(torch.uint8).reshape(32, 4)
    with_exceptions = csf.decode_csf_scale_plane(fixed, exceptions, 32, 4)
    assert not torch.equal(wrong, with_exceptions)


def test_csf_exception_index_is_17_bits():
    plane = _plane(2560, 40, seed=13, low=0x38, high=0x90)
    fixed, exceptions = encode_csf(plane)
    index = csf.csf_exception_index(exceptions)
    assert int(index.max()) >= 65536
    assert torch.equal(csf.decode_csf_scale_plane(fixed, exceptions, 2560, 40), plane)
    # Truncating the record to 16 bits breaks the plane: bit 16 is used.
    truncated = (exceptions.to(torch.int64) & 0xFFFF).to(torch.uint32)
    assert not torch.equal(csf.decode_csf_scale_plane(fixed, truncated, 2560, 40), plane)


def test_csf_validation_rejects_bad_inputs():
    plane = _plane(16, 4, seed=21)
    fixed, exceptions = encode_csf(plane)
    with pytest.raises(ValueError, match="!= "):
        csf.decode_csf_scale_plane(fixed[:, :-8], exceptions, 16, 4)
    out_of_range = torch.tensor([0x7F << 24 | 9999], dtype=torch.uint32)
    with pytest.raises(ValueError, match="outside plane"):
        csf.decode_csf_scale_plane(fixed, out_of_range, 16, 4)
    overflow = torch.full((1, 16 + 8 * 4), 250, dtype=torch.uint8)
    with pytest.raises(ValueError, match="overflowed"):
        csf.decode_csf_scale_plane(overflow, torch.zeros(0, dtype=torch.uint32), 16, 4)


# ---------------------------------------------------------------------------
# format detection (plain fast path vs CSF) and header classification
# ---------------------------------------------------------------------------
EXPERT = "model.language_model.layers.0.mlp.experts.0.gate_proj"


def test_detect_nvfp4_scale_prefers_csf_then_plain():
    weight_shape = (640, 1280)  # K = 2560, cols = 160
    csf_names = {f"{EXPERT}{csf.CSF_FIXED_SUFFIX}", f"{EXPERT}{csf.CSF_EXCEPTIONS_SUFFIX}"}
    spec = csf.detect_nvfp4_scale(csf_names, EXPERT, weight_shape)
    assert (spec.encoding, spec.rows, spec.cols) == (csf.CSF_ENCODING, 640, 160)
    assert spec.fixed_name.endswith(csf.CSF_FIXED_SUFFIX)

    plain_names = {f"{EXPERT}{csf.PLAIN_SCALE_SUFFIX}"}
    spec = csf.detect_nvfp4_scale(plain_names, EXPERT, weight_shape)
    assert spec.encoding == csf.PLAIN_ENCODING
    assert spec.scale_name == f"{EXPERT}.weight_scale"

    with pytest.raises(ValueError, match="neither"):
        csf.detect_nvfp4_scale(set(), EXPERT, weight_shape)
    with pytest.raises(ValueError, match="exceptions"):
        csf.detect_nvfp4_scale({f"{EXPERT}{csf.CSF_FIXED_SUFFIX}"}, EXPERT, weight_shape)


def test_classify_layer_distinguishes_the_three_kinds():
    plain_header = {
        f"{EXPERT}.weight": {"dtype": "U8", "shape": [640, 1280]},
        f"{EXPERT}.weight_scale": {"dtype": "F8_E4M3", "shape": [640, 160]},
        f"{EXPERT}.weight_scale_2": {"dtype": "F32", "shape": []},
    }
    fmt = csf.classify_layer(EXPERT, plain_header, "NVFP4")
    assert fmt.kind == csf.KIND_NVFP4_PACKED
    assert fmt.scale_encoding == csf.PLAIN_ENCODING
    assert fmt.group_size == 16 and fmt.declared_algo == "NVFP4"

    csf_header = {
        f"{EXPERT}.weight": {"dtype": "U8", "shape": [640, 1280]},
        f"{EXPERT}{csf.CSF_FIXED_SUFFIX}": {"dtype": "U8", "shape": [40, 1296]},
        f"{EXPERT}{csf.CSF_EXCEPTIONS_SUFFIX}": {"dtype": "U32", "shape": [1045]},
    }
    fmt = csf.classify_layer(EXPERT, csf_header)
    assert fmt.kind == csf.KIND_NVFP4_PACKED
    assert fmt.scale_encoding == csf.CSF_ENCODING
    assert fmt.scale_shape == (40, 1296)

    mxfp8 = "model.language_model.layers.0.mlp.shared_expert.gate_proj"
    fmt = csf.classify_layer(mxfp8, {
        f"{mxfp8}.weight": {"dtype": "F8_E4M3", "shape": [640, 2560]},
        f"{mxfp8}.weight_scale": {"dtype": "U8", "shape": [640, 80]},
    })
    assert fmt.kind == csf.KIND_MXFP8
    assert fmt.scale_encoding == csf.MXFP8_ENCODING
    assert fmt.group_size == 32 and fmt.scale_shape == (640, 80)

    fmt = csf.classify_layer("lm_head", {"lm_head.weight": {"dtype": "BF16", "shape": [152064, 2560]}})
    assert fmt.kind == csf.KIND_UNQUANTIZED
    assert fmt.scale_encoding == "none"


def test_classify_layer_rejects_inconsistent_headers():
    bad_mxfp8 = {
        "m.weight": {"dtype": "F8_E4M3", "shape": [128, 64]},
        "m.weight_scale": {"dtype": "U8", "shape": [128, 1]},
    }
    with pytest.raises(ValueError, match="MXFP8 scale"):
        csf.classify_layer("m", bad_mxfp8)
    bad_plain = {
        "m.weight": {"dtype": "U8", "shape": [128, 32]},
        "m.weight_scale": {"dtype": "F8_E4M3", "shape": [128, 3]},
    }
    with pytest.raises(ValueError, match="plain scale"):
        csf.classify_layer("m", bad_plain)


# ---------------------------------------------------------------------------
# NVFP4 reference dequantization
# ---------------------------------------------------------------------------
def test_dequant_nvfp4_matches_a_manual_reference():
    rows, k = 8, 64
    packed = _packed_nvfp4(rows, k, seed=31)
    plane = _plane(rows, k // 16, seed=32, low=0x70, high=0x78)
    scale2 = torch.tensor([0.125], dtype=torch.float32)

    out = csf.dequant_nvfp4(packed, plane, scale2, dtype=torch.float32)

    codes = csf.unpack_e2m1_codes(packed)
    values = csf.e2m1_lut()[codes.to(torch.int64)]
    blocks = csf.e4m3_lut()[plane.reshape(-1).to(torch.int64)].reshape(rows, k // 16)
    expected = values * blocks.repeat_interleave(16, dim=-1) * scale2
    assert torch.allclose(out, expected, rtol=0, atol=0)
    assert out.shape == (rows, k)


def test_dequant_nvfp4_validates_shapes():
    packed = _packed_nvfp4(4, 32, seed=1)
    with pytest.raises(ValueError, match="scale shape"):
        csf.dequant_nvfp4(packed, _plane(4, 1, seed=2), 1.0)
    with pytest.raises(ValueError, match="2D"):
        csf.dequant_nvfp4(packed[0], _plane(4, 2, seed=2), 1.0)


def test_dequant_nvfp4_matches_vllm_emulation_reference():
    emulation = pytest.importorskip(
        "vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils")
    rows, k = 16, 128
    packed = _packed_nvfp4(rows, k, seed=41)
    plane = _plane(rows, k // 16, seed=42)
    scale2 = torch.tensor([0.0625], dtype=torch.float32)

    ours = csf.dequant_nvfp4(packed, plane, scale2, dtype=torch.float32)
    reference = emulation.dequantize_to_dtype(packed, plane, scale2, torch.float32, 16, swizzle=False)
    # The two multiply in a different order, so allow one F32 rounding.
    assert torch.allclose(ours, reference, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# MXFP8 -> per-channel FP8
# ---------------------------------------------------------------------------
def test_mxfp8_scale_is_a_power_of_two():
    scale = torch.tensor([0x7F, 0x80, 0x7E, 0x00], dtype=torch.uint8)
    factors = csf.mxfp8_scale_to_float(scale)
    assert factors.tolist() == [1.0, 2.0, 0.5, 2.0**-127]
    nan = csf.mxfp8_scale_to_float(torch.tensor([0xFF], dtype=torch.uint8))
    assert nan.item() == 0.0


def test_dequant_mxfp8_multiplies_block_exponents():
    weight = torch.tensor([[0x38, 0x38, 0x38, 0x38]], dtype=torch.uint8).view(torch.float8_e4m3fn)
    scale = torch.tensor([[0x7F, 0x81]], dtype=torch.uint8)
    out = csf.dequant_mxfp8(weight, scale, group_size=2)
    assert out.tolist() == [[1.0, 1.0, 4.0, 4.0]]


def test_mxfp8_to_fp8_per_channel_stays_in_range():
    rows, k = 12, 256
    generator = torch.Generator().manual_seed(51)
    weight = (torch.randn(rows, k, generator=generator) * 8).to(torch.float8_e4m3fn)
    scale = torch.randint(0x74, 0x84, (rows, k // 32), generator=generator, dtype=torch.int64).to(torch.uint8)
    diagnostics: dict = {}

    weight_fp8, inverse_scale = csf.mxfp8_to_fp8_per_channel(weight, scale, diagnostics=diagnostics)

    assert weight_fp8.dtype == torch.float8_e4m3fn and weight_fp8.shape == (rows, k)
    assert inverse_scale.dtype == torch.float32 and inverse_scale.shape == (rows, )
    values = weight_fp8.to(torch.float32)
    assert torch.isfinite(values).all()
    assert float(values.abs().max()) <= csf.E4M3_MAX
    assert int(((weight_fp8.view(torch.uint8) & 0x7F) == 0x7F).sum()) == 0
    assert diagnostics["clamped_values"] == 0
    assert diagnostics["nan_scale_blocks"] == 0
    # The pow2 fold is exact except where a value underflows the E4M3 subnormal
    # floor; these are synthetic wide-range weights, so allow a looser bound
    # than the ~1e-6 measured on real MXFP8 planes.
    assert diagnostics["relative_l2_error"] < 1e-3

    # Documented tolerance: the only loss is values that underflow the E4M3
    # subnormal floor after the power-of-two rescale, and their absolute error
    # is bounded by half a subnormal quantum times the row scale.
    reference = csf.dequant_mxfp8(weight, scale)
    reconstructed = values * inverse_scale.unsqueeze(-1)
    difference = (reconstructed - reference).abs()
    assert float(difference.max()) == pytest.approx(diagnostics["max_abs_error"])
    assert float(difference.max()) <= 2.0**-9 * float(inverse_scale.max()) * 1.001
    # Only values that land in the E4M3 subnormal range after the rescale are
    # rounded (quantum 2**-9); everything in the normal range keeps its
    # mantissa exactly because the rescale is a power of two.
    lost = difference > 0
    normal_floor = (2.0**-6 * inverse_scale).unsqueeze(-1).expand_as(reference)
    assert bool((reference[lost].abs() < normal_floor[lost]).all())


def test_mxfp8_fold_matches_vllm_core_dequant_reference():
    """The per-channel fold reconstructs what vLLM core dequantizes.

    ``dequant_mxfp8_to_bf16`` (vllm-core) is an independent implementation of
    the same UE8M0 block math, so agreeing with it after folding the block
    exponents into one per-row scale is evidence about the conversion, not about
    a shared helper.
    """
    emulation = pytest.importorskip(
        "vllm.model_executor.layers.quantization.utils.mxfp8_utils",
        reason="vLLM core is required for the MXFP8 dequant reference",
    )
    rows, k = 16, 512
    generator = torch.Generator().manual_seed(53)
    weight = (torch.randn(rows, k, generator=generator) * 6).to(torch.float8_e4m3fn)
    scale = torch.randint(0x72, 0x86, (rows, k // 32), generator=generator,
                          dtype=torch.int64).to(torch.uint8)

    reference = emulation.dequant_mxfp8_to_bf16(weight, scale)
    assert reference.shape == weight.shape

    weight_fp8, inverse_scale = csf.mxfp8_to_fp8_per_channel(weight, scale)
    reconstructed = weight_fp8.to(torch.float32) * inverse_scale.unsqueeze(-1)
    assert torch.isfinite(reconstructed).all()
    assert float(reconstructed.abs().max()) <= csf.E4M3_MAX * float(inverse_scale.max())

    difference = (reconstructed - reference.float()).abs()
    relative_l2 = float((difference.pow(2).sum() / reference.float().pow(2).sum().clamp_min(1e-30)).sqrt())
    # BF16 reference rounding (2**-8) plus the E4M3 subnormal floor after the
    # power-of-two rescale; both are bounded, neither is a layout error.
    assert relative_l2 < 1e-2
    assert float(difference.max()) <= 2.0**-9 * float(inverse_scale.max()) * 1.001 \
        + float(reference.float().abs().max()) * 2.0**-8


def test_mxfp8_gaudi2_correction_halves_values_and_doubles_the_scale():
    rows, k = 8, 128
    generator = torch.Generator().manual_seed(61)
    # Keep every magnitude >= 0.5 so halving stays inside the normal E4M3 range
    # and is therefore exact.  Subnormal values are covered by the tolerance
    # assertion in test_mxfp8_to_fp8_per_channel_stays_in_range.
    magnitudes = torch.rand(rows, k, generator=generator) * 1.5 + 0.5
    signs = torch.where(torch.rand(rows, k, generator=generator) > 0.5, 1.0, -1.0)
    weight = (magnitudes * signs).to(torch.float8_e4m3fn)
    scale = torch.full((rows, k // 32), 0x7F, dtype=torch.uint8)

    plain, plain_scale = csf.mxfp8_to_fp8_per_channel(weight, scale)
    corrected, corrected_scale = csf.mxfp8_to_fp8_per_channel(weight, scale, gaudi2_correction=True)

    assert torch.equal(corrected_scale, plain_scale * 2.0)
    assert float(corrected.to(torch.float32).abs().max()) <= csf.E4M3_FNUZ_MAX
    assert torch.equal(corrected.to(torch.float32) * 2.0, plain.to(torch.float32))


def test_mxfp8_amax_mode_reproduces_the_matrix():
    rows, k = 4, 64
    generator = torch.Generator().manual_seed(71)
    weight = (torch.randn(rows, k, generator=generator)).to(torch.float8_e4m3fn)
    scale = torch.full((rows, k // 32), 0x7F, dtype=torch.uint8)
    weight_fp8, inverse_scale = csf.mxfp8_to_fp8_per_channel(weight, scale, mode="amax")
    reference = csf.dequant_mxfp8(weight, scale)
    reconstructed = weight_fp8.to(torch.float32) * inverse_scale.unsqueeze(-1)
    # amax mode re-rounds every value to E4M3 (3 mantissa bits -> up to 2**-4
    # relative per element), so it is close but not bit exact.
    relative_l2 = float(((reconstructed - reference).pow(2).sum()
                         / reference.pow(2).sum().clamp_min(1e-30)).sqrt())
    assert relative_l2 < 0.08
    assert float(inverse_scale.max()) > 0.0
    with pytest.raises(ValueError, match="unknown mode"):
        csf.mxfp8_to_fp8_per_channel(weight, scale, mode="nope")


# ---------------------------------------------------------------------------
# W4A16_NVFP4 -> BF16
# ---------------------------------------------------------------------------
def test_dequant_w4a16_matches_the_emulator_and_the_lut():
    emulation = pytest.importorskip(
        "vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils")
    rows, k = 16, 64
    packed = _packed_nvfp4(rows, k, seed=81)
    plane = _plane(rows, k // 16, seed=82)
    scale2 = torch.tensor([0.375], dtype=torch.float32)

    out = csf.dequant_w4a16_nvfp4(packed, plane, scale2, dtype=torch.bfloat16)
    reference = emulation.dequantize_to_dtype(packed, plane, scale2, torch.bfloat16, 16, swizzle=False)
    assert torch.equal(out, reference)
    # Same math as the packed-plane LUT used for the experts.
    assert torch.equal(out, csf.dequant_nvfp4(packed, plane, scale2, dtype=torch.bfloat16))


# ---------------------------------------------------------------------------
# loaders (resident contract, no expert expansion)
# ---------------------------------------------------------------------------
def _expert_source(num_experts: int, rows: int, k: int, *, csf_encoded: bool) -> dict[str, torch.Tensor]:
    source: dict[str, torch.Tensor] = {}
    prefix = "model.language_model.layers.0.mlp"
    for expert in range(num_experts):
        base = f"{prefix}.experts.{expert}.gate_proj"
        source[f"{base}.weight"] = _packed_nvfp4(rows, k, seed=100 + expert)
        plane = _plane(rows, k // 16, seed=200 + expert)
        if csf_encoded:
            fixed, exceptions = encode_csf(plane)
            source[f"{base}{csf.CSF_FIXED_SUFFIX}"] = fixed
            source[f"{base}{csf.CSF_EXCEPTIONS_SUFFIX}"] = exceptions
        else:
            source[f"{base}.weight_scale"] = plane.view(torch.float8_e4m3fn)
        source[f"{base}.weight_scale_2"] = torch.tensor([0.01 * (expert + 1)], dtype=torch.float32)
    return source


@pytest.mark.parametrize("csf_encoded", [False, True])
def test_load_experts_returns_the_resident_contract(csf_encoded):
    num_experts, rows, k = 3, 16, 32
    source = _expert_source(num_experts, rows, k, csf_encoded=csf_encoded)

    weights = csf.load_experts(source, "model.language_model.layers.0.mlp", num_experts, dtype=torch.float32)

    assert weights.packed.shape == (num_experts, rows, k // 2)
    assert weights.packed.dtype == torch.uint8
    assert weights.scales.shape == (num_experts, rows, k // 16)
    assert weights.scales.dtype == torch.float8_e4m3fn
    assert weights.scale2.shape == (num_experts, )
    assert weights.scale2.dtype == torch.float32
    assert weights.scale2.tolist() == pytest.approx([0.01, 0.02, 0.03])

    # Nothing was expanded: the dequantized expert is rebuilt from the planes.
    dequantized = csf.dequant_nvfp4(weights.packed[1], weights.scales[1], weights.scale2[1])
    assert dequantized.shape == (rows, k)
    assert dequantized.abs().max() > 0


def test_load_experts_rejects_bad_inputs():
    source = _expert_source(2, 16, 32, csf_encoded=False)
    with pytest.raises(ValueError, match="positive"):
        csf.load_experts(source, "model.language_model.layers.0.mlp", 0)
    with pytest.raises(KeyError):
        csf.load_experts(source, "model.language_model.layers.0.mlp", 5)


def test_load_packed_nvfp4_single_matrix_and_scale2_override():
    source = _expert_source(1, 16, 32, csf_encoded=True)
    base = "model.language_model.layers.0.mlp.experts.0.gate_proj"
    weights = csf.load_packed_nvfp4(source, base)
    assert weights.packed.shape == (1, 16, 16)
    assert weights.scales.shape == (1, 16, 2)
    assert weights.scale2.tolist() == pytest.approx([0.01])

    # The PLE n-gram shards have no per-shard global scale: it is passed in.
    source[f"{base}.shared_scale"] = torch.tensor([0.5], dtype=torch.float32)
    overridden = csf.load_packed_nvfp4(source, base, scale2_name=f"{base}.shared_scale")
    assert overridden.scale2.tolist() == pytest.approx([0.5])

    source.pop(f"{base}.weight_scale_2")
    source.pop(f"{base}.shared_scale")
    default = csf.load_packed_nvfp4(source, base)
    assert default.scale2.tolist() == [1.0]

    # PLE shards have no per-shard scalar: the table's shared scale is used.
    table = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    source[f"{table}.weight_scale_2"] = torch.tensor([0.25], dtype=torch.float32)
    source[f"{table}.shard_0.weight"] = _packed_nvfp4(16, 32, seed=1234)
    source[f"{table}.shard_0.weight_scale"] = _plane(16, 2, seed=1235).view(torch.float8_e4m3fn)
    shard = csf.load_packed_nvfp4(source, f"{table}.shard_0")
    assert shard.scale2.tolist() == pytest.approx([0.25])


def test_load_expert_w13_stacks_on_n():
    source = _expert_source(2, 16, 32, csf_encoded=False)
    prefix = "model.language_model.layers.0.mlp"
    for expert in range(2):
        base = f"{prefix}.experts.{expert}.up_proj"
        source[f"{base}.weight"] = _packed_nvfp4(16, 32, seed=300 + expert)
        source[f"{base}.weight_scale"] = _plane(16, 2, seed=400 + expert).view(torch.float8_e4m3fn)
        source[f"{base}.weight_scale_2"] = torch.tensor([0.01 * (expert + 1)], dtype=torch.float32)

    gate = csf.load_experts(source, prefix, 2, projection="gate_proj")
    w13 = csf.load_expert_w13(source, prefix, 2)
    assert w13.packed.shape == (2, 32, 16)
    assert torch.equal(w13.packed[:, :16], gate.packed)
    assert torch.equal(w13.scale2, gate.scale2)

    source["model.language_model.layers.0.mlp.experts.1.up_proj.weight_scale_2"] = torch.tensor(
        [0.99], dtype=torch.float32)
    with pytest.raises(ValueError, match="disagree on weight_scale_2"):
        csf.load_expert_w13(source, prefix, 2)


def test_load_mxfp8_and_w4a16_helpers():
    source = {
        "m.weight": (torch.randn(8, 128) * 2).to(torch.float8_e4m3fn),
        "m.weight_scale": torch.full((8, 4), 0x7F, dtype=torch.uint8),
        "v.weight": _packed_nvfp4(16, 64, seed=91),
        "v.weight_scale": _plane(16, 4, seed=92).view(torch.float8_e4m3fn),
        "v.weight_scale_2": torch.tensor([0.25], dtype=torch.float32),
    }
    weight_fp8, scale = csf.load_mxfp8_as_fp8(source, "m")
    assert weight_fp8.dtype == torch.float8_e4m3fn and scale.shape == (8, )

    dequantized = csf.load_w4a16_as_bf16(source, "v")
    assert dequantized.dtype == torch.bfloat16 and dequantized.shape == (16, 64)

    source.pop("v.weight_scale")
    fixed, exceptions = encode_csf(_plane(16, 4, seed=93))
    source[f"v{csf.CSF_FIXED_SUFFIX}"] = fixed
    source[f"v{csf.CSF_EXCEPTIONS_SUFFIX}"] = exceptions
    with pytest.raises(ValueError, match="plain E4M3 scale plane"):
        csf.load_w4a16_as_bf16(source, "v")


def test_packed_weights_move_and_kernel_layout():
    # K/2 = 64 bytes: one whole TPC dual group, i.e. loadable by the GEMV.
    weights = csf.NvFp4PackedWeights(
        torch.zeros(2, 4, 64, dtype=torch.uint8),
        torch.zeros(2, 4, 8, dtype=torch.float8_e4m3fn),
        torch.ones(2, dtype=torch.float32),
    )
    moved = weights.to("cpu")
    assert moved.packed.shape == (2, 4, 64)
    assert moved.layout == csf.tpc_layout_module().TPC_GEMV_LAYOUT
    assert moved.k == 128

    layout = weights.kernel_layout()
    # The declared Track 3 layout keeps torch [E, N, K/2]: Synapse reverses the
    # axes, so the fastest TPC dimension is the packed K axis and a 256-byte
    # vector load is `packed[tile*256, row, expert]`. Permuting to [K/2, N, E]
    # (which this helper used to do) puts the expert axis on the FCD instead and
    # no kernel in the tree reads that.
    assert layout.packed.shape == (2, 4, 64)
    assert layout.scales.shape == (2, 4, 8)
    assert layout.scale2.shape == (2, )
    assert torch.equal(layout.packed, weights.packed)
    assert layout.packed.data_ptr() != weights.packed.data_ptr(), "not an alias of the source"

    report = weights.geometry(require_kernel_caps=False)
    assert report["ok"] and report["packed_row"] == 64 and report["packed_tiles"] == 1

    # A plane that is not a whole number of dual groups cannot be TPC-loaded;
    # the loader logs that and the GEMV is not selected (never a silent read).
    narrow = csf.NvFp4PackedWeights(
        torch.zeros(2, 4, 8, dtype=torch.uint8),
        torch.zeros(2, 4, 1, dtype=torch.float8_e4m3fn),
        torch.ones(2, dtype=torch.float32),
    )
    assert not narrow.geometry(require_kernel_caps=False)["ok"]
    with pytest.raises(ValueError, match="not in a TPC-loadable layout"):
        narrow.kernel_layout()


# ---------------------------------------------------------------------------
# TPC layout module: the declared layout and the preshuffle hook
# ---------------------------------------------------------------------------
def _tpc():
    return csf.tpc_layout_module()


def test_tpc_layout_declares_the_gemv_geometry():
    tpc = _tpc()
    assert tpc.TPC_GEMV_LAYOUT == "tpc_e_n_k2_v1"
    assert tpc.TPC_LAYOUT_VERSION >= 1
    assert tpc.TPC_VECTOR_BYTES == 256 and tpc.TPC_DUAL_GROUP_BYTES == 64
    assert tpc.TPC_K_PER_VECTOR == 512 and tpc.TPC_K_PER_DUAL_GROUP == 128
    assert tpc.TPC_SCALE_BYTES_PER_VECTOR == 32
    assert tpc.NVFP4_GROUP_SIZE == csf.NVFP4_GROUP_SIZE
    # The worked example in the module docstring: K = 32 -> 16 bytes, 2 scales.
    assert tpc.packed_row_bytes(32) == 16
    assert tpc.scale_row_bytes(32) == 2
    assert tpc.choose_row_bytes(32) == 16
    assert tpc.choose_row_bytes(32, tpc.TPC_GEMV_LAYOUT_PADDED) == 256
    assert tpc.scale_row_bytes_for_layout(32, tpc.TPC_GEMV_LAYOUT_PADDED) == 64


def test_tpc_layout_env_switch(monkeypatch):
    tpc = _tpc()
    assert tpc.layout_from_env({}) == tpc.TPC_GEMV_LAYOUT
    assert tpc.layout_from_env({tpc.PRESHUFFLE_SWITCH: "1"}) == tpc.TPC_GEMV_LAYOUT
    assert tpc.layout_from_env({tpc.PRESHUFFLE_SWITCH: "0"}) == tpc.TPC_LAYOUT_IDENTITY
    assert tpc.layout_from_env({tpc.LAYOUT_ENV: "identity"}) == tpc.TPC_LAYOUT_IDENTITY
    assert tpc.layout_from_env({tpc.LAYOUT_ENV: "tpc_e_n_k2_pad256_v1"}) == tpc.TPC_GEMV_LAYOUT_PADDED
    with pytest.raises(ValueError, match="unknown TPC layout"):
        tpc.layout_from_env({tpc.LAYOUT_ENV: "nonsense"})

    # The switch also reaches the loaders.
    monkeypatch.setenv(tpc.PRESHUFFLE_SWITCH, "0")
    source = _expert_source(2, 16, 128, csf_encoded=True)
    weights = csf.load_experts(source, "model.language_model.layers.0.mlp", 2)
    assert weights.layout == tpc.TPC_LAYOUT_IDENTITY


def test_tpc_preshuffle_declared_layout_is_byte_identical():
    tpc = _tpc()
    packed = torch.randint(0, 256, (2, 4, 64), generator=torch.Generator().manual_seed(5),
                           dtype=torch.int64).to(torch.uint8)
    scales = _plane(2 * 4, 8, seed=6).reshape(2, 4, 8).view(torch.float8_e4m3fn)

    shuffled_packed, shuffled_scales = tpc.tpc_planes(packed, scales, 128)
    assert shuffled_packed.shape == packed.shape and shuffled_packed.dtype == torch.uint8
    assert torch.equal(shuffled_packed, packed)
    assert shuffled_packed.data_ptr() != packed.data_ptr(), "the loader must own the bytes"
    assert shuffled_scales.dtype == torch.float8_e4m3fn
    assert torch.equal(shuffled_scales.view(torch.uint8), scales.view(torch.uint8))
    assert shuffled_scales.data_ptr() != scales.data_ptr()

    # ... and the identity switch really is a no-op.
    same = tpc.preshuffle_packed(packed, tpc.TPC_LAYOUT_IDENTITY)
    assert same.data_ptr() == packed.data_ptr()


def test_tpc_preshuffle_padded_rows_are_inert():
    tpc = _tpc()
    # K = 192 -> K/2 = 96 bytes, so the padded layout grows the row to 256.
    packed = torch.randint(0, 256, (2, 4, 96), generator=torch.Generator().manual_seed(7),
                           dtype=torch.int64).to(torch.uint8)
    scales = _plane(2 * 4, 12, seed=8).reshape(2, 4, 12).view(torch.float8_e4m3fn)
    padded, padded_scales = tpc.tpc_planes(packed, scales, 192, layout=tpc.TPC_GEMV_LAYOUT_PADDED)
    assert padded.shape == (2, 4, 256)
    assert torch.equal(padded[:, :, :96], packed)
    assert int(padded[:, :, 96:].abs().sum()) == 0, "E2M1 code 0 is the value 0"
    assert padded_scales.shape == (2, 4, 64)
    assert int(padded_scales.view(torch.uint8)[:, :, 12:].abs().sum()) == 0

    assert torch.equal(tpc.unpreshuffle_packed(padded, 192), packed)
    assert torch.equal(tpc.unpreshuffle_scales(padded_scales, 192).view(torch.uint8),
                       scales.view(torch.uint8))


def test_tpc_check_geometry_matches_the_real_layer_shapes():
    tpc = _tpc()
    # The real MoE: 48 layers x 512 experts, gate/up N = 640, K = 2560 and
    # down N = 2560, K = 640.
    report = tpc.check_geometry(512, 640, 2560)
    assert report["ok"] and report["packed_row"] == 1280 and report["packed_tiles"] == 5
    assert report["scale_row"] == 160
    report = tpc.check_geometry(512, 2560, 640)
    assert report["ok"] and report["packed_row"] == 320 and report["packed_tiles"] == 2
    with pytest.raises(ValueError, match="TPC dual groups"):
        tpc.check_geometry(512, 640, 64)
    with pytest.raises(ValueError, match="multiple of 64"):
        tpc.check_geometry(8, 640, 2560)


def _resident_dot(packed_rows_u8, scale_rows_u8, x):
    """numpy reference over the *resident* planes.

    ``y[n] = sum_k E2M1[code(n, k)] * E4M3[scale(n, k/16)] * x[k]`` with the
    row walked exactly as the kernel does: byte ``j`` low nibble = ``k = 2j``,
    high nibble = ``k = 2j + 1``; scale byte ``c`` covers ``k in [16c, 16c+16)``.
    """
    import numpy as np

    packed = np.asarray(packed_rows_u8)
    rows, row_bytes = packed.shape
    k = row_bytes * 2
    scales = np.asarray(scale_rows_u8)[:, :k // 16]
    assert scales.shape == (rows, k // 16), (scales.shape, (rows, k // 16))
    codes = np.stack([packed & 0x0F, (packed >> 4) & 0x0F], axis=-1).reshape(rows, k)
    e2m1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
    magnitudes = e2m1[codes & 0x07]
    values = np.where((codes & 0x08) != 0, -magnitudes, magnitudes)
    weight = values * np.repeat(scales.astype(np.float64), 16, axis=1)
    return weight @ np.asarray(x, dtype=np.float64)


def test_tpc_preshuffled_dot_product_equals_unshuffled():
    """The preshuffle must not change a single product.

    Decoding the *resident* (preshuffled) plane reproduces the dot product of
    the untouched plane bit for bit: the declared layout is byte-identical, and
    the padded layout only appends zeros that multiply a zero activation.
    """
    import numpy as np

    tpc = _tpc()
    generator = torch.Generator().manual_seed(11)
    k, rows = 256, 4  # K/2 = 128 bytes = two TPC dual groups
    packed = torch.randint(0, 256, (rows, k // 2), generator=generator, dtype=torch.int64).to(torch.uint8)
    scales = _plane(rows, k // 16, seed=12)
    x = np.random.default_rng(13).standard_normal(k)

    reference = _resident_dot(packed.numpy(), scales.numpy(), x)

    for layout in (tpc.TPC_GEMV_LAYOUT, tpc.TPC_GEMV_LAYOUT_PADDED):
        resident_packed = tpc.preshuffle_packed(packed.unsqueeze(0), layout)[0]
        resident_scales = tpc.preshuffle_scales(scales.unsqueeze(0).view(torch.float8_e4m3fn),
                                                layout).view(torch.uint8)[0]
        wide = np.zeros(resident_packed.shape[1] * 2, dtype=np.float64)
        wide[:k] = x  # the resident row's padded tail sees a zero activation
        got = _resident_dot(resident_packed.numpy(), resident_scales.numpy(), wide)
        np.testing.assert_array_equal(got, reference)
        # ... and the inverse brings the exact checkpoint bytes back.
        assert torch.equal(tpc.unpreshuffle_packed(resident_packed.unsqueeze(0), k), packed.unsqueeze(0))
        assert torch.equal(tpc.unpreshuffle_scales(resident_scales.unsqueeze(0).view(torch.float8_e4m3fn),
                                                   k).view(torch.uint8), scales.unsqueeze(0))


def test_tpc_load_experts_applies_the_preshuffle():
    tpc = _tpc()
    source = _expert_source(3, 16, 128, csf_encoded=True)  # K/2 = 64 bytes per row
    weights = csf.load_experts(source, "model.language_model.layers.0.mlp", 3)
    assert weights.layout == tpc.TPC_GEMV_LAYOUT
    assert weights.packed.shape == (3, 16, 64)
    assert weights.scales.shape == (3, 16, 8)
    assert weights.packed.dtype == torch.uint8
    assert weights.scales.dtype == torch.float8_e4m3fn
    assert weights.geometry(require_kernel_caps=False)["ok"]
    # The declared layout keeps the checkpoint bytes, and the loader owns them.
    first = "model.language_model.layers.0.mlp.experts.0.gate_proj"
    assert torch.equal(weights.packed[0], source[f"{first}.weight"])
    assert weights.packed.data_ptr() != source[f"{first}.weight"].data_ptr()

    plain = {name: tensor for name, tensor in source.items()
             if not name.endswith(csf.CSF_FIXED_SUFFIX) and not name.endswith(csf.CSF_EXCEPTIONS_SUFFIX)}
    for expert in range(3):
        base = f"model.language_model.layers.0.mlp.experts.{expert}.gate_proj"
        plane = source[f"{base}{csf.CSF_FIXED_SUFFIX}"]
        exceptions = source[f"{base}{csf.CSF_EXCEPTIONS_SUFFIX}"]
        decoded = csf.decode_csf_scale_plane(plane, exceptions, 16, 8)
        assert torch.equal(weights.scales[expert].view(torch.uint8), decoded)
        plain[f"{base}.weight_scale"] = decoded.view(torch.float8_e4m3fn)
    # Same bytes as a plain-plane load: the CSF decode is the only difference.
    from_plain = csf.load_experts(plain, "model.language_model.layers.0.mlp", 3)
    assert torch.equal(from_plain.packed, weights.packed)
    assert torch.equal(from_plain.scales.view(torch.uint8), weights.scales.view(torch.uint8))

    # The padded layout rounds every row up to a whole 256-byte vector.
    padded = csf.load_experts(source, "model.language_model.layers.0.mlp", 3,
                              layout=tpc.TPC_GEMV_LAYOUT_PADDED)
    assert padded.layout == tpc.TPC_GEMV_LAYOUT_PADDED
    assert padded.packed.shape == (3, 16, 256)
    assert padded.scales.shape == (3, 16, 64)
    assert int(padded.packed[:, :, 64:].abs().sum()) == 0
    assert torch.equal(tpc.unpreshuffle_packed(padded.packed, 128), weights.packed)
    assert torch.equal(tpc.unpreshuffle_scales(padded.scales, 128).view(torch.uint8),
                       weights.scales.view(torch.uint8))


# ---------------------------------------------------------------------------
# MIXED_PRECISION dispatch (mirrors ModelOptMixedPrecisionConfig)
# ---------------------------------------------------------------------------
def test_is_layer_ignored_forms():
    assert csf.is_layer_ignored("lm_head", IGNORE)
    assert csf.is_layer_ignored("mtp.pre_fc_norm_hidden", IGNORE)  # wildcard
    assert csf.is_layer_ignored("mtp.fc_embedding", IGNORE)  # exact
    assert not csf.is_layer_ignored("model.language_model.layers.0.mlp.experts", IGNORE)
    assert not csf.is_layer_ignored("anything", [])


@pytest.mark.parametrize("prefix,algo", [
    ("model.language_model.layers.0.mlp.experts", "NVFP4"),
    ("model.language_model.layers.0.mlp.shared_expert.gate_proj", "MXFP8"),
    ("model.visual.blocks.0.mlp.linear_fc1", "W4A16_NVFP4"),
    ("model.visual.merger.linear_fc2", None),
    ("model.language_model.layers.0.mlp.gate", None),
])
def test_resolve_quant_algo(prefix, algo):
    assert csf.resolve_quant_algo(prefix, QUANTIZED_LAYERS) == algo


def test_resolve_quant_algo_fused_and_prefixed_forms():
    algorithm = "model.language_model.layers.0.self_attn"
    assert csf.resolve_quant_algo(f"{algorithm}.qkv_proj", QUANTIZED_LAYERS,
                                  {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}) == "MXFP8"
    # vLLM's alias direction: language_model.model.* <-> model.language_model.*
    assert csf.resolve_quant_algo(
        "language_model.model.layers.0.self_attn.q_proj", QUANTIZED_LAYERS) == "MXFP8"
    # A per-expert prefix reaches its group; the group itself resolves directly.
    assert csf.resolve_quant_algo("model.language_model.layers.0.mlp.experts.0.gate_proj",
                                  QUANTIZED_LAYERS) is None
    assert csf.dispatch_kind("model.language_model.layers.0.mlp.experts.0.gate_proj",
                             QUANTIZED_LAYERS) == csf.KIND_NVFP4_PACKED


def test_resolve_quant_algo_mixed_fused_layer_raises():
    layers = {
        "m.q_proj": {"quant_algo": "MXFP8"},
        "m.k_proj": {"quant_algo": "NVFP4"},
    }
    with pytest.raises(ValueError, match="mixed quant_algo"):
        csf.resolve_quant_algo("m.qkv_proj", layers, {"qkv_proj": ["q_proj", "k_proj", "v_proj"]})


@pytest.mark.parametrize("prefix,kind", [
    ("model.language_model.layers.0.mlp.experts", csf.KIND_NVFP4_PACKED),
    ("model.language_model.layers.1.ple.ple_embedding.ngram_embedding", csf.KIND_NVFP4_PACKED),
    # Per-shard PLE prefixes resolve through the n-gram embedding ancestor.
    ("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_7", csf.KIND_NVFP4_PACKED),
    ("model.language_model.layers.0.mlp.shared_expert.gate_proj", csf.KIND_MXFP8),
    ("model.visual.blocks.0.attn.qkv", csf.KIND_UNQUANTIZED),  # not in this synthetic map
    ("model.visual.blocks.0.mlp.linear_fc1", csf.KIND_W4A16_BF16),
    ("mtp.layers.0.mlp.experts", csf.KIND_NVFP4_PACKED),  # 4-bit routed experts
    ("lm_head", csf.KIND_UNQUANTIZED),
])
def test_dispatch_kind(prefix, kind):
    assert csf.dispatch_kind(prefix, QUANTIZED_LAYERS, IGNORE) == kind


def test_dispatch_kind_fp8_algorithms():
    layers = {
        "a.weight": {"quant_algo": "FP8"},
        "b.weight": {"quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"},
    }
    assert csf.dispatch_kind("a", layers) == csf.KIND_FP8
    assert csf.dispatch_kind("b", layers) == csf.KIND_FP8
    with pytest.raises(ValueError, match="unsupported quant_algo"):
        csf.dispatch_kind("c", {"c": {"quant_algo": "INT4"}})


# ---------------------------------------------------------------------------
# Gaudi2 quant-method factory and per-prefix dispatch
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mixed_config():
    return HPUModelOptMixedPrecisionConfig.from_config(dict(QUANT_CONFIG))


def test_factory_returns_the_gaudi_method_per_kind(mixed_config):
    assert isinstance(hpu_modelopt_method_for(mixed_config, csf.KIND_MXFP8),
                      HPUModelOptMxFp8LinearMethod)
    assert isinstance(hpu_modelopt_method_for(mixed_config, csf.KIND_W4A16_BF16),
                      HPUModelOptW4A16NvFp4LinearMethod)
    with pytest.raises(ValueError, match="no Gaudi2 ModelOpt quant method"):
        hpu_modelopt_method_for(mixed_config, csf.KIND_NVFP4_PACKED)


def test_mixed_config_dispatches_per_prefix(mixed_config):
    linear = Mock(spec=LinearBase)

    method = mixed_config.get_quant_method(linear, "model.language_model.layers.0.mlp.shared_expert.gate_proj")
    assert isinstance(method, HPUModelOptMxFp8LinearMethod)

    method = mixed_config.get_quant_method(linear, "model.visual.blocks.0.mlp.linear_fc1")
    assert isinstance(method, HPUModelOptW4A16NvFp4LinearMethod)

    assert isinstance(mixed_config.get_quant_method(linear, "model.language_model.layers.0.mlp.gate"),
                      UnquantizedLinearMethod)
    assert isinstance(mixed_config.get_quant_method(Mock(spec=ParallelLMHead), "lm_head"),
                      UnquantizedLinearMethod)
    # Attention without KV-cache quantization is left to the model.
    assert mixed_config.get_quant_method(Mock(spec=LinearBase), "model.language_model.layers.0.self_attn.q_proj") \
        is not None


def test_mixed_config_rejects_dense_nvfp4_and_routed_experts(mixed_config):
    with pytest.raises(ValueError, match="packed planes in HBM"):
        mixed_config.get_quant_method(Mock(spec=LinearBase), "model.language_model.layers.0.mlp.experts")
    with pytest.raises(ValueError, match="load_experts"):
        mixed_config.get_quant_method(Mock(spec=RoutedExperts), "model.language_model.layers.0.mlp.experts")


def test_mixed_config_exposes_the_track_d_loaders(mixed_config):
    source = _expert_source(2, 16, 32, csf_encoded=True)
    weights = mixed_config.load_experts(source, "model.language_model.layers.0.mlp", 2)
    assert weights.packed.shape == (2, 16, 16)
    assert weights.scales.dtype == torch.float8_e4m3fn

    source.update({
        "v.weight": _packed_nvfp4(16, 64, seed=95),
        "v.weight_scale": _plane(16, 4, seed=96).view(torch.float8_e4m3fn),
        "v.weight_scale_2": torch.tensor([0.25], dtype=torch.float32),
        "m.weight": torch.zeros(8, 128, dtype=torch.float8_e4m3fn),
        "m.weight_scale": torch.full((8, 4), 0x7F, dtype=torch.uint8),
    })
    assert mixed_config.load_vision_matrix_as_bf16(source, "v").dtype == torch.bfloat16
    weight_fp8, scale = mixed_config.load_mxfp8_linear(source, "m")
    assert weight_fp8.dtype == torch.float8_e4m3fn and scale.shape == (8, )


# ---------------------------------------------------------------------------
# MXFP8 linear method: post-load conversion keeps the FP8 contract
# ---------------------------------------------------------------------------
def test_mxfp8_linear_method_converts_blocks_to_per_channel():
    method = HPUModelOptMxFp8LinearMethod(Mock())
    # Pure CPU test of the fold: pin the Gaudi2 range correction off. It is
    # device-dependent (_hpu_gaudi2_scale_adjustment reads the live platform),
    # so leaving it to the host would make this pass without a card and fail
    # with one. The corrected contract has its own test below.
    method.scale_adjustment = False
    rows, k = 16, 128
    generator = torch.Generator().manual_seed(101)
    weight = (torch.randn(rows, k, generator=generator) * 2).to(torch.float8_e4m3fn)
    scale = torch.full((rows, k // 32), 0x7F, dtype=torch.uint8)

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)

    method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight.shape == (k, rows)  # transposed for fp8_gemm_v2
    assert layer.weight_scale.dtype == torch.float32
    assert layer.weight_scale.shape == (rows, )
    reference = csf.dequant_mxfp8(weight, scale)
    reconstructed = layer.weight.t().to(torch.float32) * layer.weight_scale.unsqueeze(-1)
    assert torch.allclose(reconstructed, reference, rtol=1e-4, atol=1e-6)


def test_mxfp8_conversion_applies_gaudi2_correction_once():
    """The correction halves values and doubles the scale (product preserved).

    Asserted on the pure conversion function: the linear method's behaviour
    depends on the live platform (``is_hpu_gaudi2``), which makes a class-level
    test order- and device-dependent.
    """
    rows, k = 8, 64
    generator = torch.Generator().manual_seed(7)
    weight = (torch.randn(rows, k, generator=generator) * 2).to(torch.float8_e4m3fn)
    scale = torch.full((rows, k // 32), 0x7F, dtype=torch.uint8)
    weight_fp8, row_scale = csf.mxfp8_to_fp8_per_channel(
        weight, scale, gaudi2_correction=True)
    reference = csf.dequant_mxfp8(weight, scale)
    reconstructed = weight_fp8.to(torch.float32) * row_scale.unsqueeze(-1)
    delta = (reconstructed - reference).abs()
    # Halving values and doubling the inverse scale cancels exactly for E4M3
    # normal values; only subnormals may round (documented trade-off).
    assert float(delta.max()) <= 1e-2, float(delta.max())
    assert torch.allclose(row_scale, torch.full((rows, ), 2.0, dtype=torch.float32))
    plain, plain_scale = csf.mxfp8_to_fp8_per_channel(weight, scale)
    assert torch.allclose(plain_scale, torch.ones(rows, dtype=torch.float32))
    # The corrected values are the plain ones halved (before E4M3 rounding).
    assert torch.allclose(weight_fp8.to(torch.float32),
                          (plain.to(torch.float32) * 0.5).to(torch.float8_e4m3fn).to(torch.float32),
                          atol=1e-2)


def test_mxfp8_linear_method_rejects_unconverted_checkpoint():
    method = HPUModelOptMxFp8LinearMethod(Mock())
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(torch.zeros(4, dtype=torch.uint8), requires_grad=False)
    with pytest.raises(TypeError, match="E4M3"):
        method.process_weights_after_loading(layer)


# ---------------------------------------------------------------------------
# W4A16 linear method: dequantize at load, then run as BF16
# ---------------------------------------------------------------------------
def test_w4a16_linear_method_dequantizes_at_load():
    method = HPUModelOptW4A16NvFp4LinearMethod(Mock())
    rows, k = 16, 64
    packed = _packed_nvfp4(rows, k, seed=111)
    plane = _plane(rows, k // 16, seed=112).view(torch.float8_e4m3fn)
    scale2 = torch.tensor([0.375], dtype=torch.float32)

    layer = torch.nn.Module()
    layer._hpu_dequant_dtype = torch.bfloat16
    layer.weight = torch.nn.Parameter(packed, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(plane, requires_grad=False)
    layer.weight_scale_2 = torch.nn.Parameter(scale2, requires_grad=False)

    method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.bfloat16
    assert layer.weight.shape == (rows, k)
    assert not hasattr(layer, "weight_scale")
    assert not hasattr(layer, "weight_scale_2")
    expected = csf.dequant_nvfp4(packed, plane.view(torch.uint8), scale2, dtype=torch.bfloat16)
    assert torch.equal(layer.weight, expected)

    x = torch.randn(2, 3, k, dtype=torch.bfloat16)
    bias = torch.randn(rows, dtype=torch.bfloat16)
    out = method.apply(layer, x, bias)
    assert out.shape == (2, 3, rows)
    assert torch.equal(out, torch.nn.functional.linear(x, layer.weight, bias))


def test_w4a16_linear_method_rejects_unpacked_checkpoint():
    method = HPUModelOptW4A16NvFp4LinearMethod(Mock())
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16), requires_grad=False)
    with pytest.raises(TypeError, match="packed U8"):
        method.process_weights_after_loading(layer)


# ---------------------------------------------------------------------------
# registration hook used by vLLM for quant_method='modelopt'
# ---------------------------------------------------------------------------
def test_modelopt_mixed_precision_is_registered_for_vllm():
    from vllm.model_executor.layers.quantization import get_quantization_config

    registered = get_quantization_config("modelopt_mixed")
    assert registered is HPUModelOptMixedPrecisionConfig
    assert modelopt.ModelOptMixedPrecisionConfig is HPUModelOptMixedPrecisionConfig
    assert HPUModelOptMixedPrecisionConfig.override_quantization_method(
        QUANT_CONFIG, None) == "modelopt_mixed"

    config = get_quantization_config("modelopt_mixed").from_config(dict(QUANT_CONFIG))
    assert isinstance(config, HPUModelOptMixedPrecisionConfig)
    assert config.kv_cache_quant_method is None
    assert config.nvfp4_config.group_size == 16


def test_modelopt_mixed_precision_is_registered_on_the_hpu_platform():
    """vLLM rejects a quant name missing from platform.supported_quantization."""
    from vllm.platforms.interface import Platform, PlatformEnum

    class FakeHpuPlatform(Platform):
        _enum = PlatformEnum.OOT
        device_name = "hpu"
        device_type = "hpu"
        supported_quantization = ["modelopt"]

    # The real vLLM gate: verify_quantization rejects the unregistered name.
    with pytest.raises(ValueError, match="not supported"):
        FakeHpuPlatform.verify_quantization("modelopt_mixed")

    _register_mixed_precision_platform_support(FakeHpuPlatform)
    assert FakeHpuPlatform.supported_quantization == ["modelopt", "modelopt_mixed"]
    FakeHpuPlatform.verify_quantization("modelopt_mixed")  # no longer raises

    # Idempotent: importing the plugin twice must not duplicate the entry.
    _register_mixed_precision_platform_support(FakeHpuPlatform)
    assert FakeHpuPlatform.supported_quantization == ["modelopt", "modelopt_mixed"]

    # Non-HPU platforms are left untouched (CPU test hosts, CUDA users).
    class FakeCpuPlatform(Platform):
        _enum = PlatformEnum.CPU
        device_name = "cpu"
        device_type = "cpu"
        supported_quantization = ["modelopt", "fp8"]

    _register_mixed_precision_platform_support(FakeCpuPlatform)
    assert FakeCpuPlatform.supported_quantization == ["modelopt", "fp8"]
