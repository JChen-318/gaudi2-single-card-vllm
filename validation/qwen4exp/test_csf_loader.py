# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ModelOpt CSF scale decoding against the real Qwen4Exp checkpoint.

What this file establishes, host-only:

* the three storage kinds really present in the export are recognised from a
  shard header alone (packed NVFP4 with plain E4M3 scales, packed NVFP4 with
  CSF scales, MXFP8 with UE8M0 scales);
* the CSF decoder reproduces the converted checkpoint's plain scale planes
  byte-for-byte, from the original export's ``nvfp4_csf_fixed`` /
  ``nvfp4_csf_exceptions`` tensors, including the 17-bit exception index;
* the documented layout is the only one that round-trips: a synthetic encoder
  written from the documented layout decodes back exactly, while dropping the
  exceptions or swapping the nibble order does not;
* the packed NVFP4 planes + scales agree with vLLM's own emulation reference
  within one F32 ulp.

Run::

    .venv/bin/python -m pytest -q validation/qwen4exp/test_csf_loader.py
"""

from __future__ import annotations

import pytest
import torch

import modelopt_support as support
from modelopt_support import csf, csf_source, plain_source  # noqa: F401  (pytest fixtures)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

EXPERT = "model.language_model.layers.0.mlp.experts.0.gate_proj"
EXPERT_DOWN = "model.language_model.layers.0.mlp.experts.0.down_proj"
EXPERT_DEEP = "model.language_model.layers.47.mlp.experts.127.up_proj"
EXPERT_MID = "model.language_model.layers.23.mlp.experts.200.gate_proj"
EXPERT_LAST = "model.language_model.layers.31.mlp.experts.511.down_proj"
#: Five real expert tensors, spread over layers / expert ids / projections, that
#: the decoder is checked against byte-for-byte (see the module docstring).
CSF_SAMPLES = [EXPERT, EXPERT_DOWN, EXPERT_DEEP, EXPERT_MID, EXPERT_LAST]

#: Three more real CSF tensors pulled from the raw HF snapshot (different
#: layers, expert ids and projections than CSF_SAMPLES), decoded straight from
#: the ``nvfp4_csf_*`` bytes and compared byte-for-byte with the converted
#: export's plain plane.
CSF_EXTRA_SAMPLES = [
    "model.language_model.layers.5.mlp.experts.1.down_proj",
    "model.language_model.layers.40.mlp.experts.511.up_proj",
    "model.language_model.layers.47.mlp.experts.300.gate_proj",
]

SHARED_GATE = "model.language_model.layers.0.mlp.shared_expert.gate_proj"
VISION_FC1 = "model.visual.blocks.0.mlp.linear_fc1"
VISION_MERGER_FC2 = "model.visual.merger.linear_fc2"


# ---------------------------------------------------------------------------
# (i) header classification of the three storage kinds
# ---------------------------------------------------------------------------
def test_classify_storage_kinds_plain_export(csf):
    root = support.require_checkpoint(support.PLAIN_CHECKPOINT, "converted plain")
    index = support.weight_map(root)

    def classify(base, declared=None):
        return csf.classify_layer(base, support.layer_header(root, base, index), declared)

    expert = classify(EXPERT, "NVFP4")
    assert expert.kind == csf.KIND_NVFP4_PACKED
    assert expert.scale_encoding == csf.PLAIN_ENCODING
    assert expert.weight_dtype == "U8"
    assert expert.weight_shape == (640, 1280)  # K = 2560, two nibbles per byte
    assert expert.scale_shape == (640, 160)  # one E4M3 block scale per 16 K
    assert expert.group_size == 16

    shared = classify("model.language_model.layers.0.mlp.shared_expert.gate_proj", "MXFP8")
    assert shared.kind == csf.KIND_MXFP8
    assert shared.scale_encoding == csf.MXFP8_ENCODING
    assert shared.weight_dtype == "F8_E4M3"
    assert shared.weight_shape == (640, 2560)
    assert shared.scale_shape == (640, 80)  # one UE8M0 byte per 32 K
    assert shared.group_size == 32

    vision_attn = classify("model.visual.blocks.0.attn.qkv", "MXFP8")
    assert vision_attn.kind == csf.KIND_MXFP8
    assert vision_attn.scale_shape == (3456, 36)

    vision_mlp = classify("model.visual.blocks.0.mlp.linear_fc1", "W4A16_NVFP4")
    assert vision_mlp.kind == csf.KIND_NVFP4_PACKED
    assert vision_mlp.declared_algo == "W4A16_NVFP4"
    assert vision_mlp.scale_encoding == csf.PLAIN_ENCODING
    assert vision_mlp.weight_shape == (4304, 576)  # K = 1152
    assert vision_mlp.scale_shape == (4304, 72)

    merger = classify("model.visual.merger.linear_fc1", "W4A16_NVFP4")
    assert merger.weight_shape == (4608, 2304)  # K = 4608
    assert merger.scale_shape == (4608, 288)

    lm_head = classify("lm_head")
    assert lm_head.kind == csf.KIND_UNQUANTIZED
    assert lm_head.scale_encoding == "none"
    assert lm_head.declared_algo is None


def test_classify_storage_kinds_csf_export(csf):
    root = support.require_checkpoint(support.CSF_CHECKPOINT, "original CSF")
    index = support.weight_map(root)
    header = support.layer_header(root, EXPERT, index)

    # The original export carries the compressed scale tensors instead of a
    # plain plane, and the fixed plane geometry is 16 + 8*cols bytes per group.
    assert f"{EXPERT}{csf.CSF_FIXED_SUFFIX}" in header
    assert f"{EXPERT}{csf.CSF_EXCEPTIONS_SUFFIX}" in header
    assert f"{EXPERT}.weight_scale" not in header
    fixed = header[f"{EXPERT}{csf.CSF_FIXED_SUFFIX}"]
    assert fixed["dtype"] == "U8"
    assert tuple(fixed["shape"]) == csf.csf_fixed_plane_shape(640, 160) == (40, 1296)
    exceptions = header[f"{EXPERT}{csf.CSF_EXCEPTIONS_SUFFIX}"]
    assert exceptions["dtype"] == "U32"

    classified = csf.classify_layer(EXPERT, header, "NVFP4")
    assert classified.kind == csf.KIND_NVFP4_PACKED
    assert classified.scale_encoding == csf.CSF_ENCODING
    assert classified.scale_shape == (40, 1296)

    # The PLE n-gram shards and the MTP experts are declared NVFP4/W4A16 but
    # ship plain E4M3 planes even in the original export: the declared
    # encoding is not the on-disk encoding, so detection has to look at bytes.
    ple = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0"
    ple_header = support.layer_header(root, ple, index)
    ple_classified = csf.classify_layer(ple, ple_header, "NVFP4")
    assert ple_classified.scale_encoding == csf.PLAIN_ENCODING
    assert ple_classified.weight_shape == (2500012, 80)
    assert ple_classified.scale_shape == (2500012, 10)
    mtp = "mtp.layers.0.mlp.experts.0.gate_proj"
    mtp_classified = csf.classify_layer(mtp, support.layer_header(root, mtp, index), "W4A16_NVFP4")
    assert mtp_classified.scale_encoding == csf.PLAIN_ENCODING


def test_detect_nvfp4_scale_prefers_csf(csf):
    weight_shape = (640, 1280)
    csf_names = {f"{EXPERT}{csf.CSF_FIXED_SUFFIX}", f"{EXPERT}{csf.CSF_EXCEPTIONS_SUFFIX}"}
    spec = csf.detect_nvfp4_scale(csf_names, EXPERT, weight_shape)
    assert spec.encoding == csf.CSF_ENCODING and spec.cols == 160

    plain_names = {f"{EXPERT}.weight_scale"}
    spec = csf.detect_nvfp4_scale(plain_names, EXPERT, weight_shape)
    assert spec.encoding == csf.PLAIN_ENCODING and spec.scale_name.endswith(".weight_scale")

    with pytest.raises(ValueError, match="neither"):
        csf.detect_nvfp4_scale(set(), EXPERT, weight_shape)
    with pytest.raises(ValueError, match="exceptions"):
        csf.detect_nvfp4_scale({f"{EXPERT}{csf.CSF_FIXED_SUFFIX}"}, EXPERT, weight_shape)


# ---------------------------------------------------------------------------
# CSF decoder vs the converted checkpoint (real bytes)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("base", CSF_SAMPLES)
def test_csf_decode_matches_converted_plane(csf, csf_source, plain_source, base):
    packed = csf.load_packed_nvfp4(csf_source, base)
    converted = csf.load_packed_nvfp4(plain_source, base)
    assert torch.equal(packed.packed, converted.packed)
    assert torch.equal(packed.scales.view(torch.uint8), converted.scales.view(torch.uint8))
    assert torch.equal(packed.scale2, converted.scale2)

    plane = csf.decode_csf_scale_plane(
        csf_source.get_tensor(f"{base}{csf.CSF_FIXED_SUFFIX}"),
        csf_source.get_tensor(f"{base}{csf.CSF_EXCEPTIONS_SUFFIX}"),
        packed.packed.shape[1],
        packed.scales.shape[2],
    )
    assert torch.equal(plane.view(torch.uint8), converted.scales.view(torch.uint8).squeeze(0))


@pytest.mark.parametrize("base", [EXPERT, EXPERT_DOWN])
def test_csf_real_metadata_invariants(csf, csf_source, base):
    weight = csf_source.get_tensor(f"{base}.weight")
    fixed = csf_source.get_tensor(f"{base}{csf.CSF_FIXED_SUFFIX}")
    exceptions = csf_source.get_tensor(f"{base}{csf.CSF_EXCEPTIONS_SUFFIX}")
    rows = weight.shape[0]
    cols = weight.shape[1] * 2 // csf.NVFP4_GROUP_SIZE

    # Every exception exists because the window reconstruction is wrong there.
    assert csf.count_csf_window_violations(fixed, exceptions, rows, cols) == 0
    assert exceptions.numel() > 0

    # NVFP4 amax quantization saturates every 16-wide block, so a correct
    # decode of the planes leaves that invariant intact.
    codes = csf.unpack_e2m1_codes(weight.view(torch.uint8))
    assert csf.nvfp4_block_saturation_ratio(codes) == 1.0

    index = csf.csf_exception_index(exceptions)
    assert rows * cols > 65536, "this tensor is meant to exercise the 17-bit index"
    assert int(index.max()) >= 65536
    assert int(index.max()) < rows * cols


@pytest.mark.parametrize("base", CSF_EXTRA_SAMPLES)
def test_csf_decode_extra_real_tensors(csf, csf_source, plain_source, base):
    """More real CSF bytes from the raw HF snapshot: decode -> plain plane,
    byte-for-byte, plus the metadata invariants of the CSF_SAMPLES tests."""
    weight = csf_source.get_tensor(f"{base}.weight")
    fixed = csf_source.get_tensor(f"{base}{csf.CSF_FIXED_SUFFIX}")
    exceptions = csf_source.get_tensor(f"{base}{csf.CSF_EXCEPTIONS_SUFFIX}")
    rows = weight.shape[0]
    cols = weight.shape[1] * 2 // csf.NVFP4_GROUP_SIZE

    plane = csf.decode_csf_scale_plane(fixed, exceptions, rows, cols)
    reference = plain_source.get_tensor(f"{base}.weight_scale").view(torch.uint8)
    assert plane.dtype == torch.uint8 and tuple(plane.shape) == (rows, cols)
    assert torch.equal(plane, reference), "CSF decode must reproduce the plain plane"

    # Structural invariants on the CSF tensors themselves.
    assert csf.count_csf_window_violations(fixed, exceptions, rows, cols) == 0
    assert exceptions.numel() > 0
    index = csf.csf_exception_index(exceptions)
    assert int(index.max()) < rows * cols
    # ... and on the planes: no NaN E4M3 code, saturated E2M1 blocks, a
    # single positive F32 global scale.
    assert not bool(((plane & 0x7F) == 0x7F).any()), "NaN scale byte decoded"
    codes = csf.unpack_e2m1_codes(weight.view(torch.uint8))
    assert csf.nvfp4_block_saturation_ratio(codes) == 1.0
    scale2 = plain_source.get_tensor(f"{base}.weight_scale_2")
    assert scale2.numel() == 1 and float(scale2.reshape(-1)[0]) > 0.0


def test_csf_dropping_bit_16_corrupts_the_plane(csf, csf_source):
    """The exception index is 17 bits; ignoring bit 16 changes real planes."""
    base = EXPERT_DOWN  # 2560 * 40 = 102400 entries, well past 2**16
    weight = csf_source.get_tensor(f"{base}.weight")
    fixed = csf_source.get_tensor(f"{base}{csf.CSF_FIXED_SUFFIX}")
    exceptions = csf_source.get_tensor(f"{base}{csf.CSF_EXCEPTIONS_SUFFIX}")
    rows, cols = weight.shape[0], weight.shape[1] * 2 // csf.NVFP4_GROUP_SIZE
    correct = csf.decode_csf_scale_plane(fixed, exceptions, rows, cols)

    truncated = (exceptions.to(torch.int64) & 0xFFFF).to(torch.uint32)
    wrong = csf.decode_csf_scale_plane(fixed, truncated, rows, cols)
    assert not torch.equal(correct, wrong)


# ---------------------------------------------------------------------------
# Documented layout: synthetic round-trip and the failure modes it excludes
# ---------------------------------------------------------------------------
def test_synthetic_roundtrip_documented_layout(csf):
    generator = torch.Generator().manual_seed(7)
    rows, cols = 256, 40
    centres = torch.randint(0x58, 0x70, (rows, 1), generator=generator)
    spread = torch.randn(rows, cols, generator=generator) * 4
    plane = (centres + spread).clamp(0x40, 0x7A).to(torch.uint8)

    fixed, exceptions = support.reference_encoder(plane)
    assert tuple(fixed.shape) == csf.csf_fixed_plane_shape(rows, cols)
    assert exceptions.numel() > 0
    assert torch.equal(csf.decode_csf_scale_plane(fixed, exceptions, rows, cols), plane)

    # Exceptions carry the values outside the 16-wide window: without them the
    # reconstruction is wrong, which is why they are not optional.
    assert csf.count_csf_window_violations(fixed, exceptions, rows, cols) == 0
    window_only = csf.decode_csf_scale_plane(fixed, torch.zeros(0, dtype=torch.uint32), rows, cols)
    assert not torch.equal(window_only, plane)


def test_synthetic_wrong_nibble_order_is_detected(csf):
    """High-nibble-first would not reproduce the plane: documents the payload order."""
    generator = torch.Generator().manual_seed(11)
    rows, cols = 64, 8
    plane = torch.randint(0x60, 0x70, (rows, cols), generator=generator).to(torch.uint8)
    fixed, exceptions = support.reference_encoder(plane)
    assert torch.equal(csf.decode_csf_scale_plane(fixed, exceptions, rows, cols), plane)

    groups = rows // 16
    base = fixed[:, :16].to(torch.int16)
    payload = fixed[:, 16:].reshape(groups, 16, cols // 2)
    swapped = torch.empty(groups, 16, cols, dtype=torch.int16)
    swapped[..., 0::2] = (payload >> 4).to(torch.int16)
    swapped[..., 1::2] = (payload & 0x0F).to(torch.int16)
    wrong = (base.unsqueeze(-1) + swapped).to(torch.uint8).reshape(rows, cols)
    assert not torch.equal(wrong, plane)


def test_synthetic_17bit_index_roundtrip(csf):
    """A plane larger than 2**16 entries needs the 17th index bit."""
    generator = torch.Generator().manual_seed(3)
    rows, cols = 2560, 40  # 102400 entries, the shape of a real down_proj plane
    plane = torch.randint(0x38, 0x90, (rows, cols), generator=generator).to(torch.uint8)
    fixed, exceptions = support.reference_encoder(plane)
    index = csf.csf_exception_index(exceptions)
    assert int(index.max()) >= 65536
    assert torch.equal(csf.decode_csf_scale_plane(fixed, exceptions, rows, cols), plane)


# ---------------------------------------------------------------------------
# NVFP4 packed planes + scales vs vLLM's emulation reference
# ---------------------------------------------------------------------------
def test_nvfp4_matches_emulation_reference(csf, plain_source):
    base = EXPERT
    packed = plain_source.get_tensor(f"{base}.weight")[:128].clone()
    scales = plain_source.get_tensor(f"{base}.weight_scale")[:128].clone()
    scale2 = plain_source.get_tensor(f"{base}.weight_scale_2").reshape(1)

    ours = csf.dequant_nvfp4(packed, scales, scale2, dtype=torch.float32)
    emulation = support.emulator()
    reference = emulation.dequantize_to_dtype(
        packed.contiguous(),
        scales.view(torch.uint8).contiguous(),
        scale2.float(),
        torch.float32,
        csf.NVFP4_GROUP_SIZE,
        swizzle=False,
    )
    assert ours.shape == reference.shape
    # Our LUT multiplies (value * block) * scale2, the emulator (block * scale2)
    # * value: one F32 rounding apart, never more.
    assert torch.allclose(ours, reference, rtol=1e-6, atol=0.0)
    max_rel = ((ours - reference).abs() / reference.abs().clamp_min(1e-30)).max()
    assert float(max_rel) < 1e-6


def test_nvfp4_requantization_roundtrip_tolerance(csf, plain_source):
    """Documented NVFP4 tolerance: the real plane is already on the E2M1 grid."""
    base = EXPERT
    packed = plain_source.get_tensor(f"{base}.weight")[:64].clone()
    scales = plain_source.get_tensor(f"{base}.weight_scale")[:64].clone()
    scale2 = plain_source.get_tensor(f"{base}.weight_scale_2").reshape(1)
    weight = csf.dequant_nvfp4(packed, scales, scale2, dtype=torch.bfloat16)

    emulation = support.emulator()
    # `ref_nvfp4_quant*` takes the *quantization* scale, i.e. the reciprocal of
    # weight_scale_2 (vLLM passes layer.input_global_scale_inv there).
    quant_scale = (1.0 / scale2.float()).reshape(1)
    round_tripped = emulation.ref_nvfp4_quant_dequant(weight, quant_scale, csf.NVFP4_GROUP_SIZE)
    error = round_tripped.float() - weight.float()
    rel_l2 = float((error.pow(2).sum() / weight.float().pow(2).sum().clamp_min(1e-30)).sqrt())
    # Bit-exact: the checkpoint's plane is saturated per block, so requantizing
    # the dequantized values lands on the same E2M1 codes and E4M3 scales.
    assert rel_l2 == 0.0
    assert float(error.abs().max()) == 0.0

    # Feeding the dequantization multiplier instead degenerates (E4M3 scale
    # underflows to zero), which is why the direction is asserted here.
    wrong = emulation.ref_nvfp4_quant_dequant(weight, scale2.float().reshape(1), csf.NVFP4_GROUP_SIZE)
    assert not torch.allclose(wrong.float(), weight.float())


# ---------------------------------------------------------------------------
# MXFP8 inside the E4M3 range after the per-channel fold
# ---------------------------------------------------------------------------
def test_mxfp8_fold_stays_inside_the_e4m3_range(csf, plain_source):
    weight = plain_source.get_tensor(f"{SHARED_GATE}.weight")
    scale = plain_source.get_tensor(f"{SHARED_GATE}.weight_scale")
    diagnostics: dict = {}

    weight_fp8, inverse_scale = csf.mxfp8_to_fp8_per_channel(weight, scale, diagnostics=diagnostics)

    assert weight_fp8.dtype == torch.float8_e4m3fn
    assert weight_fp8.shape == weight.shape
    assert inverse_scale.dtype == torch.float32 and inverse_scale.shape == (weight.shape[0], )
    values = weight_fp8.to(torch.float32)
    assert torch.isfinite(values).all()
    assert float(values.abs().max()) <= csf.E4M3_MAX
    assert int(((weight_fp8.view(torch.uint8) & 0x7F) == 0x7F).sum()) == 0, "NaN codes written"
    assert diagnostics["clamped_values"] == 0
    assert diagnostics["nan_scale_blocks"] == 0
    assert diagnostics["relative_l2_error"] < 1e-5

    # The same matrix after the Gaudi2 OCP->FNUZ correction stays inside the
    # range the MME represents, with the inverse scale doubled (exact product).
    corrected, corrected_scale = csf.mxfp8_to_fp8_per_channel(weight, scale, gaudi2_correction=True)
    assert torch.equal(corrected_scale, inverse_scale * 2.0)
    assert float(corrected.to(torch.float32).abs().max()) <= csf.E4M3_FNUZ_MAX
    assert int(((corrected.view(torch.uint8) & 0x7F) == 0x7F).sum()) == 0


# ---------------------------------------------------------------------------
# W4A16_NVFP4 -> BF16 dequant equality
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("base", [VISION_FC1, VISION_MERGER_FC2])
def test_w4a16_dequant_equals_the_emulator(csf, plain_source, base):
    emulation = support.emulator()
    packed = plain_source.get_tensor(f"{base}.weight")
    scale = plain_source.get_tensor(f"{base}.weight_scale")
    scale2 = plain_source.get_tensor(f"{base}.weight_scale_2").reshape(1)

    loaded = csf.load_w4a16_as_bf16(plain_source, base)
    reference = emulation.dequantize_to_dtype(
        packed.contiguous(),
        scale.view(torch.uint8).contiguous(),
        scale2.float(),
        torch.bfloat16,
        csf.NVFP4_GROUP_SIZE,
        swizzle=False,
    )
    assert loaded.dtype == torch.bfloat16
    assert tuple(loaded.shape) == (packed.shape[0], packed.shape[1] * 2)
    assert torch.equal(loaded, reference), "W4A16 dequant must reuse vLLM's emulation math"
    # ... and it is the same LUT math the packed expert path uses.
    assert torch.equal(loaded, csf.dequant_nvfp4(packed, scale, scale2, dtype=torch.bfloat16))


# ---------------------------------------------------------------------------
# PLE n-gram table: 128 NVFP4 shards with one shared global scale
# ---------------------------------------------------------------------------
def test_ple_ngram_table_contract(csf):
    root = support.require_checkpoint(support.PLAIN_CHECKPOINT, "converted plain")
    index = support.weight_map(root)
    table = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    assert f"{table}.weight_scale_2" in index, "the table carries one shared global scale"
    shard_names = [name for name in index if name.startswith(table + ".shard_")]
    assert len(shard_names) == 256, "128 shards x (packed weight, E4M3 plane)"
    assert not any(name.endswith(".weight_scale_2") for name in shard_names), \
        "shards have no per-shard global scale"

    header = support.layer_header(root, f"{table}.shard_0", index)
    classified = csf.classify_layer(f"{table}.shard_0", header, "NVFP4")
    assert classified.kind == csf.KIND_NVFP4_PACKED
    assert classified.scale_encoding == csf.PLAIN_ENCODING
    assert classified.weight_shape == (2500012, 80)  # K = 160
    assert classified.scale_shape == (2500012, 10)  # K / 16
    # Per-shard prefixes resolve to the packed kind, not "unquantized".
    config = support.quant_config(root)
    assert csf.dispatch_kind(f"{table}.shard_0", config["quantized_layers"],
                             config["ignore"]) == csf.KIND_NVFP4_PACKED
