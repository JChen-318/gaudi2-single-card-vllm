# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MIXED_PRECISION per-prefix dispatch and the three Gaudi2 conversions.

What this file establishes, host-only:

* the real ``quantization_config`` of the checkpoint declares exactly three
  algorithms (MXFP8 438, W4A16_NVFP4 58, NVFP4 49) and ``weight_scale_encoding:
  csf`` only on the NVFP4 entries, while the converted export contains no CSF
  tensor at all -- so the declaration cannot be trusted, bytes are inspected;
* :func:`modelopt_csf.dispatch_kind` reproduces vLLM's own
  ``ModelOptMixedPrecisionConfig._resolve_quant_algo`` on every declared prefix
  and on the fused/alias forms the model actually builds;
* MXFP8 -> per-channel FP8 stays inside the E4M3 range with no clamp and no
  NaN, and the Gaudi2 range correction only doubles the inverse scale;
* W4A16_NVFP4 -> BF16 equals vLLM's ``dequantize_to_dtype`` bit for bit;
* the expert loader returns the resident contract
  ``packed [E,N,K/2] U8 / scales [E,N,K/16] E4M3 / scale2 [E] F32`` and its
  dequantization agrees with the emulation reference;
* ``hpu_modelopt.py`` really wires those pieces (AST check, so the HPU plugin
  is never imported in this process).

Run::

    .venv/bin/python -m pytest -q validation/qwen4exp/test_mixed_precision_dispatch.py
"""

from __future__ import annotations

import ast
from collections import Counter

import pytest
import torch

import modelopt_support as support
from modelopt_support import csf, plain_source  # noqa: F401  (pytest fixtures)

MXFP8_EXPECTED = 438
W4A16_EXPECTED = 58
NVFP4_EXPECTED = 49

EXPERT_GROUP = "model.language_model.layers.0.mlp.experts"
SHARED_GATE = "model.language_model.layers.0.mlp.shared_expert.gate_proj"


# ---------------------------------------------------------------------------
# Declared configuration
# ---------------------------------------------------------------------------
def test_declared_config_and_csf_presence():
    root = support.require_checkpoint(support.PLAIN_CHECKPOINT, "converted plain")
    config = support.quant_config(root)
    layers = config["quantized_layers"]

    assert config["quant_method"] == "modelopt"
    assert config["quant_algo"] == "MIXED_PRECISION"
    assert config["kv_cache_quant_algo"] is None
    counts = Counter(entry["quant_algo"] for entry in layers.values())
    assert counts == {
        "MXFP8": MXFP8_EXPECTED,
        "W4A16_NVFP4": W4A16_EXPECTED,
        "NVFP4": NVFP4_EXPECTED,
    }, counts

    # `weight_scale_encoding: csf` is declared on the 48 routed-expert groups;
    # the PLE n-gram table is NVFP4/group-16 too but declares no encoding, and
    # its on-disk scales are plain E4M3 in both exports.
    encoding_entries = {name for name, entry in layers.items() if "weight_scale_encoding" in entry}
    assert len(encoding_entries) == 48
    assert all(layers[name]["weight_scale_encoding"] == "csf" for name in encoding_entries)
    assert all(layers[name]["group_size"] == 16 for name in encoding_entries)
    assert all(name.endswith(".mlp.experts") for name in encoding_entries)
    ple = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    assert layers[ple] == {"quant_algo": "NVFP4", "group_size": 16}

    # The converted export holds plain E4M3 planes only: the declaration above
    # describes the *original* export, not this one.
    index = support.weight_map(root)
    assert not [name for name in index if "nvfp4_csf" in name]


def test_original_export_holds_csf_tensors():
    root = support.require_checkpoint(support.CSF_CHECKPOINT, "original CSF")
    index = support.weight_map(root)
    fixed = [name for name in index if name.endswith(".weight_scale.nvfp4_csf_fixed")]
    exceptions = [name for name in index if name.endswith(".weight_scale.nvfp4_csf_exceptions")]
    # 48 decoder layers x 512 experts x 3 projections, and nothing else: the
    # PLE n-gram table and the MTP experts are plain E4M3 in both exports.
    assert len(fixed) == len(exceptions) == 48 * 512 * 3
    assert all(".mlp.experts." in name for name in fixed)


# ---------------------------------------------------------------------------
# Per-prefix dispatch
# ---------------------------------------------------------------------------
def _packed_mapping():
    return {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }


@pytest.mark.parametrize(
    "prefix, expected",
    [
        # routed experts stay packed, 4-bit, in HBM
        (EXPERT_GROUP, "nvfp4_packed"),
        (f"{EXPERT_GROUP}.0.gate_proj", "nvfp4_packed"),
        # MTP experts are declared W4A16_NVFP4 but are still routed experts
        ("mtp.layers.0.mlp.experts", "nvfp4_packed"),
        # PLE n-gram table: NVFP4 planes that live in host RAM
        ("model.language_model.layers.1.ple.ple_embedding.ngram_embedding", "nvfp4_packed"),
        # MXFP8 -> per-channel FP8
        (SHARED_GATE, "mxfp8_to_fp8"),
        ("model.language_model.layers.0.linear_attn.in_proj_qkv", "mxfp8_to_fp8"),
        ("model.language_model.layers.3.self_attn.q_proj", "mxfp8_to_fp8"),
        ("model.language_model.layers.3.self_attn.indexer.index_qk_proj", "mxfp8_to_fp8"),
        ("model.visual.blocks.0.attn.qkv", "mxfp8_to_fp8"),
        # W4A16 vision matrices -> BF16 at load
        ("model.visual.blocks.0.mlp.linear_fc1", "w4a16_to_bf16"),
        ("model.visual.blocks.0.mlp.linear_fc2", "w4a16_to_bf16"),
        ("model.visual.merger.linear_fc1", "w4a16_to_bf16"),
        # ignored / unquantized
        ("lm_head", "unquantized"),
        ("mtp.fc_embedding", "unquantized"),
        ("mtp.pre_fc_norm_embedding", "unquantized"),
        ("mtp.hyper_connection_mixer.hc_norm", "unquantized"),
        ("mtp.layers.0.self_attn.q_proj", "unquantized"),
        ("mtp.layers.0.mlp.shared_expert.gate_proj", "unquantized"),
        ("model.language_model.layers.0.mlp.gate", "unquantized"),
        ("model.visual.blocks.0.norm1", "unquantized"),
    ],
)
def test_dispatch_kind(csf, prefix, expected):
    root = support.require_checkpoint(support.PLAIN_CHECKPOINT, "converted plain")
    config = support.quant_config(root)
    kind = csf.dispatch_kind(
        prefix,
        config["quantized_layers"],
        config["ignore"],
        _packed_mapping(),
    )
    assert kind == expected


def test_fused_projection_names_resolve(csf):
    root = support.require_checkpoint(support.PLAIN_CHECKPOINT, "converted plain")
    config = support.quant_config(root)
    layers = config["quantized_layers"]
    # vLLM builds one qkv_proj for the three attention shards and one
    # gate_up_proj for the shared expert pair.
    assert csf.resolve_quant_algo("model.language_model.layers.3.self_attn.qkv_proj", layers,
                                  _packed_mapping()) == "MXFP8"
    assert csf.resolve_quant_algo("model.language_model.layers.0.mlp.shared_expert.gate_up_proj",
                                  layers, _packed_mapping()) == "MXFP8"
    # The alias direction vLLM also tries.
    assert csf.resolve_quant_algo("language_model.model.layers.3.self_attn.q_proj", layers) == "MXFP8"


def test_dispatch_matches_vllm(csf):
    """Every declared prefix resolves exactly like vLLM's own config."""
    root = support.require_checkpoint(support.PLAIN_CHECKPOINT, "converted plain")
    config = support.quant_config(root)
    layers = config["quantized_layers"]
    vllm_modelopt = pytest.importorskip(
        "vllm.model_executor.layers.quantization.modelopt",
        reason="vLLM core is required for the cross-check",
    )
    vllm_config = vllm_modelopt.ModelOptMixedPrecisionConfig.from_config(config)
    vllm_config.packed_modules_mapping = _packed_mapping()
    mapping = _packed_mapping()

    prefixes = list(layers)
    prefixes += [f"{name}.gate_proj" for name in layers if name.endswith(".mlp.experts")][:8]
    prefixes += [
        "model.language_model.layers.3.self_attn.qkv_proj",
        "model.language_model.layers.0.mlp.shared_expert.gate_up_proj",
        "lm_head",
        "model.visual.blocks.0.norm1",
    ]
    # Coarse parent prefixes take vLLM's "first child in config order" branch.
    # They are the ones that diverge if the child-prefix strategy is
    # reimplemented as a set/consistency check instead of a first match, so
    # they are part of the differential check.
    for name in layers:
        parts = name.split(".")
        for depth in range(2, len(parts)):
            prefixes.append(".".join(parts[:depth]))
    prefixes = sorted(set(prefixes))
    mismatches = []
    for prefix in prefixes:
        theirs = vllm_config._resolve_quant_algo(prefix)
        ours = csf.resolve_quant_algo(prefix, layers, mapping)
        if theirs != ours:
            mismatches.append((prefix, theirs, ours))
    assert not mismatches, mismatches[:5]

    # The ignore list agrees too (exact, substring and wildcard forms).
    for prefix in ("mtp.fc_embedding", "mtp.pre_fc_norm_hidden", "mtp.hyper_connection_mixer.hc_norm",
                   "mtp.layers.0.mlp.gate", "lm_head", "model.visual.blocks.0.attn.qkv"):
        assert vllm_config.is_layer_excluded(prefix) == csf.is_layer_ignored(prefix, config["ignore"]), prefix


# ---------------------------------------------------------------------------
# (iii) MXFP8 -> per-channel FP8 stays inside the E4M3 range
# ---------------------------------------------------------------------------
def test_mxfp8_conversion_stays_in_fp8_range(csf, plain_source):
    weight = plain_source.get_tensor(f"{SHARED_GATE}.weight")
    scale = plain_source.get_tensor(f"{SHARED_GATE}.weight_scale")
    diagnostics: dict = {}
    weight_fp8, weight_scale = csf.mxfp8_to_fp8_per_channel(weight, scale, diagnostics=diagnostics)

    assert weight_fp8.dtype == torch.float8_e4m3fn
    assert weight_fp8.shape == weight.shape
    assert weight_scale.dtype == torch.float32 and weight_scale.shape == (weight.shape[0], )

    values = weight_fp8.to(torch.float32)
    assert torch.isfinite(values).all()
    assert float(values.abs().max()) <= csf.E4M3_MAX
    assert int(((weight_fp8.view(torch.uint8) & 0x7F) == 0x7F).sum()) == 0, "NaN codes written"

    assert diagnostics["clamped_values"] == 0
    assert diagnostics["nan_scale_blocks"] == 0
    assert diagnostics["zero_row_scales"] == 0
    # The power-of-two rescale is exact except for values below the E4M3
    # subnormal floor: measured 38 of 1.6M here, relative L2 under 1e-6.
    assert diagnostics["relative_l2_error"] < 1e-5
    assert diagnostics["flushed_values"] < weight.numel() // 10000

    reference = csf.dequant_mxfp8(weight, scale, dtype=torch.float32)
    reconstructed = values * weight_scale.unsqueeze(-1)
    # The 38 values that flush below the E4M3 subnormal floor show up as
    # absolute error below 2e-6 (see diagnostics['max_abs_error']).
    assert torch.allclose(reconstructed, reference, rtol=1e-4, atol=1e-5)


def test_mxfp8_gaudi2_correction_only_doubles_the_scale(csf, plain_source):
    weight = plain_source.get_tensor(f"{SHARED_GATE}.weight")
    scale = plain_source.get_tensor(f"{SHARED_GATE}.weight_scale")
    plain_diag: dict = {}
    corrected_diag: dict = {}
    weight_a, scale_a = csf.mxfp8_to_fp8_per_channel(weight, scale, diagnostics=plain_diag)
    weight_b, scale_b = csf.mxfp8_to_fp8_per_channel(weight, scale, gaudi2_correction=True,
                                                     diagnostics=corrected_diag)

    # Same arithmetic as hpu_ops.gaudi_weight_wrapper: half the FP8 value,
    # double the inverse scale.
    assert torch.equal(scale_b, scale_a * 2.0)
    assert float(weight_b.to(torch.float32).abs().max()) <= csf.E4M3_FNUZ_MAX
    assert int(((weight_b.view(torch.uint8) & 0x7F) == 0x7F).sum()) == 0

    product_a = weight_a.to(torch.float32) * scale_a.unsqueeze(-1)
    product_b = weight_b.to(torch.float32) * scale_b.unsqueeze(-1)
    difference = (product_a - product_b).abs()
    relative = float((difference.pow(2).sum() / product_a.pow(2).sum()).sqrt())
    assert relative < 1e-5
    # The halving only touches values whose E4M3 mantissa is subnormal: every
    # differing element stays within one subnormal quantum (2**-9) and the
    # population is ~3e-4 of the matrix (551 of 1.6M here).
    assert float(difference.max()) <= 2.0**-9
    assert int((product_a != product_b).sum()) < weight.numel() // 1000


# ---------------------------------------------------------------------------
# (iii-b) bit-level verification of the fold, and the pinned default
# ---------------------------------------------------------------------------
def test_mxfp8_fold_is_bit_exact_in_higher_precision(csf, plain_source):
    """The pow2 fold of ``E4M3 * 2**(UE8M0-127)`` is bit-exact wherever the
    rescaled value stays representable: the golden is recomputed in float64
    from the raw bytes, ``dequant_mxfp8`` (the fp32 golden) equals it exactly,
    and every element of the folded product that differs is confined to the
    E4M3 subnormal rounding zone with an error of at most half its 2**-9
    quantum."""
    weight = plain_source.get_tensor(f"{SHARED_GATE}.weight")
    scale = plain_source.get_tensor(f"{SHARED_GATE}.weight_scale")
    diagnostics: dict = {}
    weight_fp8, row_scale = csf.mxfp8_to_fp8_per_channel(weight, scale, diagnostics=diagnostics)

    w_bytes = weight.view(torch.uint8)
    s_bytes = scale.view(torch.uint8)
    assert not bool(((w_bytes & 0x7F) == 0x7F).any()), "NaN weight codes in the plane"
    assert not bool(((s_bytes & 0xFF) == 0xFF).any()), "NaN scale codes in the plane"

    # Golden in float64: every operand is an exact power/mantissa, so the
    # products are exact and any later disagreement is the fold's doing.
    e4m3_64 = csf.e4m3_lut(torch.float64)
    values = e4m3_64[w_bytes.to(torch.int64)]
    factor = torch.exp2(s_bytes.to(torch.int64).to(torch.float64) - 127.0)
    golden = values * factor.repeat_interleave(csf.MXFP8_GROUP_SIZE, dim=-1)
    # The fp32 golden is the same product: no fp32 rounding occurs on the way.
    fp32_golden = csf.dequant_mxfp8(weight, scale, dtype=torch.float32).to(torch.float64)
    assert torch.equal(fp32_golden, golden)

    # row_scale is exactly the per-row max block scale (a power of two).
    row_max = factor.amax(dim=-1)
    assert torch.equal(row_scale.to(torch.float64), row_max)

    scaled = golden / row_max.unsqueeze(-1)  # exact power-of-two rescale
    folded = weight_fp8.to(torch.float64) * row_scale.to(torch.float64).unsqueeze(-1)
    exact = folded == golden
    inexact = ~exact

    # Every non-exact element lives in the subnormal rounding zone ...
    assert bool((scaled.abs()[inexact] < 2.0**-6).all()), \
        "a value outside the subnormal zone was re-rounded by the fold"
    # ... and its error is at most half the 2**-9 subnormal quantum (the
    # flushed population is counted by the diagnostics).
    assert int(inexact.sum()) >= int(diagnostics["flushed_values"])
    assert float((folded - golden).abs()[inexact].max()) <= 2.0**-10
    assert int(inexact.sum()) < weight.numel() // 1000
    # ... while every representable element reconstructs the golden exactly.
    assert torch.equal(folded[exact], golden[exact])


def test_scale_adjustment_is_pinned_false(csf):
    """The Gaudi2 E4M3FN->FNUZ halving stays OFF by default.

    The pure fold's ``gaudi2_correction`` default is False, and the plugin's
    MXFP8 linear method only ever forwards its runtime-config flag (False on a
    non-Gaudi2 host, and unless the runtime config enables it) into exactly
    that argument -- never a hardcoded True.
    """
    import inspect

    default = inspect.signature(csf.mxfp8_to_fp8_per_channel).parameters["gaudi2_correction"].default
    assert default is False
    source = support.HPU_MODELOPT_PATH.read_text()
    assert "gaudi2_correction=self.scale_adjustment" in source
    assert "self.scale_adjustment = _hpu_gaudi2_scale_adjustment()" in source
    assert "gaudi2_correction=True" not in source


# ---------------------------------------------------------------------------
# (iv) W4A16_NVFP4 -> BF16 equals the emulation reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("base", [
    "model.visual.blocks.0.mlp.linear_fc1",
    "model.visual.merger.linear_fc2",
])
def test_w4a16_dequant_equals_emulator(csf, plain_source, base):
    emulation = support.emulator()
    packed = plain_source.get_tensor(f"{base}.weight")
    scale = plain_source.get_tensor(f"{base}.weight_scale")
    scale2 = plain_source.get_tensor(f"{base}.weight_scale_2").reshape(1)

    ours = csf.load_w4a16_as_bf16(plain_source, base)
    reference = emulation.dequantize_to_dtype(
        packed.contiguous(),
        scale.view(torch.uint8).contiguous(),
        scale2.float(),
        torch.bfloat16,
        csf.NVFP4_GROUP_SIZE,
        swizzle=False,
    )
    assert torch.equal(ours, reference)
    # ... and it is the same math as the packed-plane LUT used for experts.
    assert torch.equal(ours, csf.dequant_nvfp4(packed, scale, scale2, dtype=torch.bfloat16))


# ---------------------------------------------------------------------------
# (a) expert loader: resident contract + emulation agreement
# ---------------------------------------------------------------------------
def test_expert_loader_contract(csf, plain_source):
    num_experts = 4
    prefix = "model.language_model.layers.0.mlp"
    experts = csf.load_experts(plain_source, prefix, num_experts, projection="gate_proj")
    assert experts.packed.shape == (num_experts, 640, 1280)
    assert experts.packed.dtype == torch.uint8
    assert experts.scales.shape == (num_experts, 640, 160)
    assert experts.scales.dtype == torch.float8_e4m3fn
    assert experts.scale2.shape == (num_experts, )
    assert experts.scale2.dtype == torch.float32
    assert torch.isfinite(experts.scale2).all() and (experts.scale2 > 0).all()

    # The declared Track 3 layout is the checkpoint order: torch [E, N, K/2],
    # so Synapse dimension 0 (the fastest) is the packed K axis and a 256-byte
    # vector load is packed[tile*256, row, expert]. The old [K/2, N, E]
    # permutation would put the expert axis on the FCD, which no kernel reads.
    kernel = experts.kernel_layout()
    assert kernel.packed.shape == (num_experts, 640, 1280)  # [E, N, K/2]
    assert kernel.scales.shape == (num_experts, 640, 160)  # [E, N, K/16]
    assert torch.equal(kernel.scale2, experts.scale2)
    assert experts.layout == csf.tpc_layout_module().TPC_GEMV_LAYOUT
    # Only 4 experts are loaded here, so the E % 64 device cap does not apply;
    # the row geometry does, and it is the real layer's.
    report = experts.geometry(require_kernel_caps=False)
    assert report["ok"], report["violations"]
    assert report["packed_tiles"] == 5 and report["scale_row"] == 160

    emulation = support.emulator()
    dequantized = csf.dequant_nvfp4(experts.packed[0], experts.scales[0], experts.scale2[0],
                                    dtype=torch.float32)
    reference = emulation.dequantize_to_dtype(
        experts.packed[0].contiguous(),
        experts.scales[0].view(torch.uint8).contiguous(),
        experts.scale2[0].reshape(1),
        torch.float32,
        csf.NVFP4_GROUP_SIZE,
        swizzle=False,
    )
    assert torch.allclose(dequantized, reference, rtol=1e-6, atol=0.0)
    # 4-bit residency: 8 nibbles per 4 packed bytes, i.e. 4 bits per weight.
    assert (experts.packed.numel() // num_experts) * 8 == 4 * dequantized.numel()


def test_expert_w13_shares_one_global_scale(csf, plain_source):
    prefix = "model.language_model.layers.0.mlp"
    w13 = csf.load_expert_w13(plain_source, prefix, 2)
    assert w13.packed.shape == (2, 1280, 1280)  # gate 640 + up 640 rows of N
    assert w13.scales.shape == (2, 1280, 160)
    gate = csf.load_experts(plain_source, prefix, 2, projection="gate_proj")
    up = csf.load_experts(plain_source, prefix, 2, projection="up_proj")
    assert torch.equal(gate.scale2, up.scale2)
    assert torch.equal(w13.scale2, gate.scale2)


# ---------------------------------------------------------------------------
# Plugin wiring (AST only: the HPU plugin is never imported here)
# ---------------------------------------------------------------------------
def _class_nodes(tree):
    return {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def _class_source(tree, name):
    node = _class_nodes(tree)[name]
    return ast.unparse(node)


def test_hpu_modelopt_wiring():
    if not support.HPU_MODELOPT_PATH.is_file():
        pytest.skip(f"{support.HPU_MODELOPT_PATH} is missing")
    tree = ast.parse(support.HPU_MODELOPT_PATH.read_text())
    classes = _class_nodes(tree)

    for name in ("HPUModelOptMixedPrecisionConfig", "HPUModelOptMxFp8LinearMethod",
                 "HPUModelOptW4A16NvFp4LinearMethod"):
        assert name in classes, f"{name} missing from hpu_modelopt.py"
    assert [ast.unparse(base) for base in classes["HPUModelOptMixedPrecisionConfig"].bases] == [
        "ModelOptMixedPrecisionConfig"
    ]
    assert [ast.unparse(base) for base in classes["HPUModelOptMxFp8LinearMethod"].bases] == [
        "HPUModelOptFp8PcPtLinearMethod"
    ]

    mixed = _class_source(tree, "HPUModelOptMixedPrecisionConfig")
    for method in ("get_quant_method", "load_experts", "load_expert_w13",
                   "load_vision_matrix_as_bf16", "load_mxfp8_linear"):
        assert f"def {method}" in mixed, f"{method} missing from the mixed config"
    assert "dispatch_kind" in mixed
    factory = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "hpu_modelopt_method_for")
    factory_source = ast.unparse(factory)
    assert "modelopt_csf.KIND_MXFP8" in factory_source
    assert "modelopt_csf.KIND_W4A16_BF16" in factory_source
    assert "HPUModelOptMxFp8LinearMethod" in factory_source
    assert "HPUModelOptW4A16NvFp4LinearMethod" in factory_source

    mxfp8 = _class_source(tree, "HPUModelOptMxFp8LinearMethod")
    assert "mxfp8_to_fp8_per_channel" in mxfp8
    assert "gaudi2_correction=self.scale_adjustment" in mxfp8

    w4a16 = _class_source(tree, "HPUModelOptW4A16NvFp4LinearMethod")
    assert "dequant_w4a16_nvfp4" in w4a16

    # The registry hook: vLLM builds its method -> config map inside
    # get_quantization_config() from the module attribute, so replacing the
    # attribute before that call selects this class (same mechanism the FP8
    # patch above uses).
    patches = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "ModelOptMixedPrecisionConfig"
                and isinstance(target.value, ast.Name) and target.value.id == "modelopt"
                for target in node.targets)
    ]
    assert patches, "modelopt.ModelOptMixedPrecisionConfig patch is missing"
    assert ast.unparse(patches[-1].value) == "HPUModelOptMixedPrecisionConfig"

    imported = {ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.Import)}
    assert "import vllm_gaudi.ops.modelopt_csf as modelopt_csf" in imported


# ---------------------------------------------------------------------------
# (v) W4A16 vision coverage: exactly the 56 small matrices
# ---------------------------------------------------------------------------
def test_w4a16_vision_name_set_is_exactly_56(csf):
    """``load_w4a16_as_bf16`` must cover exactly the 56 small vision matrices
    -- ``visual.blocks.0..26.mlp`` x2 plus the two merger pieces -- enumerated
    from the real index of the raw HF snapshot (headers streamed shard by
    shard; the tensors are never loaded all at once).

    The two remaining W4A16 declarations are the MTP expert groups, which are
    routed experts and keep their packed residency instead.
    """
    root = support.require_checkpoint(support.CSF_CHECKPOINT, "original CSF")
    index = support.weight_map(root)
    config = support.quant_config(root)
    layers = config["quantized_layers"]

    expected = {f"model.visual.blocks.{block}.mlp.linear_fc{part}"
                for block in range(27) for part in (1, 2)}
    expected |= {"model.visual.merger.linear_fc1", "model.visual.merger.linear_fc2"}
    assert len(expected) == 56

    declared = {name for name, entry in layers.items()
                if entry["quant_algo"] == "W4A16_NVFP4"}
    vision = {name for name in declared if name.startswith("model.visual.")}
    assert vision == expected, "the declared vision W4A16 set is not the expected 56"
    assert declared - expected == {"mtp.layers.0.mlp.experts", "mtp.layers.48.mlp.experts"}

    # Every one of the 56 is on disk as U8 packed nibbles + a plain F8_E4M3
    # scale plane + an F32 weight_scale_2, and dispatches to the BF16 loader.
    # Only shard headers are read (one JSON per distinct shard file).
    headers: dict[str, dict] = {}
    for name in sorted(expected):
        entry = index[f"{name}.weight"]
        if entry not in headers:
            headers[entry] = support.shard_header(support.shard_path(root, entry))
        header = headers[entry]
        weight = header[f"{name}.weight"]
        scale = header[f"{name}.weight_scale"]
        assert weight["dtype"] == "U8" and weight["shape"][1] * 2 % csf.NVFP4_GROUP_SIZE == 0
        assert scale["dtype"] == "F8_E4M3"
        assert scale["shape"] == [weight["shape"][0], weight["shape"][1] * 2 // csf.NVFP4_GROUP_SIZE]
        assert f"{name}.weight_scale_2" in index
        assert csf.dispatch_kind(name, layers, config["ignore"]) == csf.KIND_W4A16_BF16

    # Nothing else under model.visual. takes the W4A16 path: the rest of the
    # tower (attention, qkv, norms) is MXFP8 or unquantized.
    others = [name for name in layers
              if name.startswith("model.visual.")
              and layers[name]["quant_algo"] not in ("MXFP8", "W4A16_NVFP4")]
    assert not others
    assert len(headers) <= 21, "the header stream must touch only the checkpoint's shards"
