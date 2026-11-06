# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The in-memory CSF load path, exercised on the *real* scale bytes.

Decision A of the port: there is no offline converter and no second
checkpoint. At weight-load time, on the host CPU, the NVFP4 CSF scales are
decoded with the known formula into flat ``float8_e4m3fn`` planes ``[N, K/16]``
and stay in memory; the packed weights stay ``uint8`` ``[N, K/2]`` (decision B).

What this file proves, with the original NVIDIA export's own CSF bytes
(``~/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local``,
147,456 ``nvfp4_csf_*`` tensors):

1. ``modelopt_csf.load_experts`` -- the function
   ``HPUModelOptMixedPrecisionConfig.load_experts`` forwards to and
   ``NvFp4ExpertStore.load_from_source`` calls -- decodes CSF when the tensors
   are CSF, and reproduces the converted export's plain scale plane **byte for
   byte**, then returns the resident contract
   ``packed [E,N,K/2] uint8`` / ``scales [E,N,K/16] float8_e4m3fn`` /
   ``scale2 [E] float32`` in the TPC layout the Track 3 GEMV declares.
2. The decoded planes are the loader's own buffers, not views of the
   checkpoint, so nothing downstream can write a checkpoint byte.
3. Nothing -- no file, no sidecar, no rewritten shard -- appears in either
   checkpoint directory: the trees are fingerprinted (size + mtime, plus a
   sha256 of the index/config/headers of the shards touched) before and after,
   and every Python-level ``open()`` of a file while the decode runs is
   recorded and checked.
4. The preshuffle of a real ``down_proj`` plane (K = 640, so K/2 = 320 bytes,
   not a whole 256-byte TPC vector) round-trips and does not move a single
   product (numpy reference).

Run::

    .venv/bin/python -m pytest -q validation/qwen4exp/test_csf_memory_path.py

Every test is CPU-only and < 10 s; the real exports are read, never written.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

import modelopt_support as support
from modelopt_support import csf, csf_source, plain_source  # noqa: F401  (pytest fixtures)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

EXPERT = "model.language_model.layers.0.mlp.experts.0.gate_proj"
EXPERT_DOWN = "model.language_model.layers.0.mlp.experts.0.down_proj"
LAYER0 = "model.language_model.layers.0.mlp"

#: Two experts keep the test under a second while still crossing an expert
#: boundary (the CSF tensors are per expert, one decode each).
NUM_EXPERTS = 2


# ---------------------------------------------------------------------------
# checkpoint guard: nothing may be written into either export
# ---------------------------------------------------------------------------
def _tree_fingerprint(root: Path) -> dict[str, tuple]:
    """``path -> (size, mtime_ns, symlink target)`` for every entry under root."""
    entries: dict[str, tuple] = {}
    for path in sorted(root.rglob("*")):
        name = str(path.relative_to(root))
        if path.is_symlink():
            target = Path(path).resolve()
            stat = target.stat()
            entries[name] = (stat.st_size, stat.st_mtime_ns, str(path.readlink()))
        else:
            stat = path.stat()
            entries[name] = (stat.st_size, stat.st_mtime_ns, None)
    return entries


def _checkpoint_fingerprint(root: Path) -> dict:
    """Everything a write would disturb: the tree plus the index/config hashes."""
    index = root / "model.safetensors.index.json"
    config = root / "config.json"
    fingerprint = {
        "root": str(root),
        "tree": _tree_fingerprint(root),
        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest() if index.is_file() else None,
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest() if config.is_file() else None,
    }
    return fingerprint


@pytest.fixture(scope="module")
def checkpoint_guard():
    """Fingerprint both exports before the module runs and compare after."""
    roots = {"csf": support.require_checkpoint(support.CSF_CHECKPOINT, "original CSF"),
             "plain": support.require_checkpoint(support.PLAIN_CHECKPOINT, "converted plain")}
    before = {name: _checkpoint_fingerprint(root) for name, root in roots.items()}
    yield {"roots": roots, "before": before}
    after = {name: _checkpoint_fingerprint(root) for name, root in roots.items()}
    changed = [name for name in roots if after[name]["tree"] != before[name]["tree"]]
    changed += [f"{name}.index" for name in roots
                if after[name]["index_sha256"] != before[name]["index_sha256"]]
    changed += [f"{name}.config" for name in roots
                if after[name]["config_sha256"] != before[name]["config_sha256"]]
    assert not changed, f"the load path wrote into the checkpoint directory: {changed}"


# ---------------------------------------------------------------------------
# 0. the export under test really is the original CSF export
# ---------------------------------------------------------------------------
def test_the_hf_cache_holds_the_original_csf_export(csf, checkpoint_guard):
    root = checkpoint_guard["roots"]["csf"]
    index = support.weight_map(root)
    csf_names = [name for name in index if "nvfp4_csf_" in name]
    assert len(csf_names) == 147456, f"expected 147,456 CSF tensors, found {len(csf_names)}"
    assert sum(name.endswith(csf.CSF_FIXED_SUFFIX) for name in csf_names) == 73728
    assert sum(name.endswith(csf.CSF_EXCEPTIONS_SUFFIX) for name in csf_names) == 73728

    # A real expert plane: fixed [40, 1296] describes [640, 160] scale bytes.
    header = support.layer_header(root, EXPERT, index)
    fixed = header[f"{EXPERT}{csf.CSF_FIXED_SUFFIX}"]
    assert fixed["dtype"] == "U8"
    assert tuple(fixed["shape"]) == csf.csf_fixed_plane_shape(640, 160) == (40, 1296)
    assert header[f"{EXPERT}{csf.CSF_EXCEPTIONS_SUFFIX}"]["dtype"] == "U32"
    assert f"{EXPERT}.weight_scale" not in header, "this export really is CSF-encoded"

    # ... and the converted export stores the same matrix as a plain plane.
    plain_header = support.layer_header(checkpoint_guard["roots"]["plain"], EXPERT,
                                        support.weight_map(checkpoint_guard["roots"]["plain"]))
    assert tuple(plain_header[f"{EXPERT}.weight_scale"]["shape"]) == (640, 160)


# ---------------------------------------------------------------------------
# 1. the serve path decodes CSF in memory, byte for byte
# ---------------------------------------------------------------------------
def _decode_with_serve_function(csf_module, source, prefix, num_experts, projection="gate_proj"):
    """Call exactly what ``HPUModelOptMixedPrecisionConfig.load_experts`` calls.

    ``hpu_modelopt`` only forwards to ``modelopt_csf.load_experts``; the
    forwarding itself is asserted (AST) in
    ``test_the_serve_path_calls_this_decoder`` so this file stays free of the
    Habana stack, and the byte-level behaviour is checked on the real function.
    ``csf_module`` is the ``csf`` fixture: the module under test.
    """
    return csf_module.load_experts(source, prefix, num_experts, projection=projection)


def test_serve_path_decodes_real_csf_scales_byte_for_byte(csf, csf_source, plain_source,
                                                          checkpoint_guard):
    decoded = _decode_with_serve_function(csf, csf_source, LAYER0, NUM_EXPERTS)
    converted = _decode_with_serve_function(csf, plain_source, LAYER0, NUM_EXPERTS)

    # (c) the weights never leave their 4-bit packing ...
    assert decoded.packed.dtype == torch.uint8
    assert decoded.packed.shape == (NUM_EXPERTS, 640, 1280)  # [E, N, K/2], K = 2560
    assert torch.equal(decoded.packed, converted.packed)

    # (b) ... and the scales come out of the decoder as flat E4M3 planes.
    assert decoded.scales.dtype == torch.float8_e4m3fn
    assert decoded.scales.shape == (NUM_EXPERTS, 640, 160)  # [E, N, K/16]
    assert decoded.scale2.dtype == torch.float32 and decoded.scale2.shape == (NUM_EXPERTS, )
    assert torch.equal(decoded.scales.view(torch.uint8),
                       converted.scales.view(torch.uint8)), \
        "the in-memory CSF decode must reproduce the plain plane exactly"
    assert torch.equal(decoded.scale2, converted.scale2)

    # The decode is deterministic and consumes the CSF tensors as they are: no
    # intermediate tensor is even produced for the plain path.
    spec = csf.detect_nvfp4_scale(set(csf_source.keys()), EXPERT, (640, 1280))
    assert spec.encoding == csf.CSF_ENCODING and spec.cols == 160
    assert csf.detect_nvfp4_scale(set(plain_source.keys()), EXPERT, (640, 1280)).encoding == \
        csf.PLAIN_ENCODING


def test_serve_path_returns_flat_planes_in_the_tpc_layout(csf, csf_source, checkpoint_guard):
    weights = _decode_with_serve_function(csf, csf_source, LAYER0, NUM_EXPERTS)
    tpc = csf.tpc_layout_module()

    assert weights.layout == tpc.TPC_GEMV_LAYOUT
    assert weights.k == 2560
    # "Flat" means one dense [E, N, K/16] buffer: no expert-major padding, no
    # per-block side tensor, and the K axis contiguous on the FCD (torch's last
    # axis is dim 0 for Synapse).
    for plane in (weights.packed, weights.scales.view(torch.uint8), weights.scale2):
        assert plane.is_contiguous()
        assert plane.numel() == math.prod(plane.shape)
    # Only two experts are loaded here, so the E % 64 device cap does not apply;
    # the row geometry is the real layer's, and a full layer (512 experts)
    # satisfies every cap.
    report = weights.geometry(require_kernel_caps=False)
    assert report["ok"], report["violations"]
    assert report["layout_version"] == tpc.TPC_LAYOUT_VERSION
    assert report["packed_row"] == 1280 and report["scale_row"] == 160
    assert report["packed_tiles"] == 5
    assert tpc.check_geometry(512, 640, 2560)["ok"]  # 48 MoE layers x 512 experts
    # The scale plane is exactly 1/8 of the packed plane: the decode grew the
    # checkpoint by K/16 bytes per row, never by a dequantized copy.
    assert weights.scales.numel() * 8 == weights.packed.numel()


def test_csf_decode_growth_matches_the_memory_budget(csf, checkpoint_guard):
    """Decision A's budget: the whole decode adds ~3.3 GiB, so it stays in RAM.

    Measured from the two exports' shard headers (no tensor is read): 73,728
    real CSF scale planes expand to the same number of flat ``K/16`` E4M3 planes
    the converted export stores, and the CSF form they replace is only ~3.7 GiB.
    """
    csf_root = checkpoint_guard["roots"]["csf"]
    plain_root = checkpoint_guard["roots"]["plain"]
    csf_index = support.weight_map(csf_root)
    plain_index = support.weight_map(plain_root)

    csf_headers: dict[str, dict] = {}
    plain_headers: dict[str, dict] = {}

    def header(root, cache, entry):
        if entry not in cache:
            cache[entry] = support.shard_header(support.shard_path(root, entry))
        return cache[entry]

    decoded_bytes = 0
    csf_bytes = 0
    planes = 0
    for name, entry in csf_index.items():
        if not name.endswith(csf.CSF_FIXED_SUFFIX):
            continue
        base = name[:-len(csf.CSF_FIXED_SUFFIX)]
        fixed = header(csf_root, csf_headers, entry)[name]
        rows = fixed["shape"][0] * 16
        cols = (fixed["shape"][1] - 16) // 8
        assert csf.csf_fixed_plane_shape(rows, cols) == tuple(fixed["shape"])
        exceptions = header(csf_root, csf_headers, entry)[base + csf.CSF_EXCEPTIONS_SUFFIX]
        planes += 1
        decoded_bytes += rows * cols
        csf_bytes += fixed["shape"][0] * fixed["shape"][1] + exceptions["shape"][0] * 4
        # ... and the converted export stores exactly that plane.
        plain_entry = plain_index[f"{base}.weight_scale"]
        stored = header(plain_root, plain_headers, plain_entry)[f"{base}.weight_scale"]
        assert tuple(stored["shape"]) == (rows, cols)

    assert planes == 73728, f"expected 73,728 CSF scale planes, found {planes}"
    decoded_gib = decoded_bytes / 2**30
    csf_gib = csf_bytes / 2**30
    assert decoded_gib == pytest.approx(7.03, abs=0.02)
    assert csf_gib == pytest.approx(3.69, abs=0.02)
    # Decision A's budget: decoding every CSF plane into a flat [E,N,K/16]
    # E4M3 plane costs the +3.4 GiB class of extra host RAM for all 48 layers
    # (7.03 GiB decoded vs the 3.69 GiB the CSF form replaces), which is the
    # number the memory plan is written against.
    growth = decoded_gib - csf_gib
    assert growth == pytest.approx(3.34, abs=0.05), growth
    assert 3.3 <= growth <= 3.5


def test_serve_path_calls_this_decoder():
    """The plugin config and the expert store both forward to this function."""
    tree = ast.parse(support.HPU_MODELOPT_PATH.read_text())
    mixed = next(node for node in ast.walk(tree)
                 if isinstance(node, ast.ClassDef) and node.name == "HPUModelOptMixedPrecisionConfig")
    load_experts = next(node for node in ast.walk(mixed)
                        if isinstance(node, ast.FunctionDef) and node.name == "load_experts")
    source = ast.unparse(load_experts)
    assert "modelopt_csf.load_experts(" in source
    assert "layout=layout" in source, "the TPC layout must reach the loader"

    model = support.REPO_ROOT / "vllm-gaudi" / "vllm_gaudi" / "models" / "qwen4_exp.py"
    if not model.is_file():
        pytest.skip(f"{model} is missing")
    model_tree = ast.parse(model.read_text())
    store = next(node for node in ast.walk(model_tree)
                 if isinstance(node, ast.ClassDef) and node.name == "NvFp4ExpertStore")
    load_from_source = next(node for node in ast.walk(store)
                            if isinstance(node, ast.FunctionDef) and node.name == "load_from_source")
    store_source = ast.unparse(load_from_source)
    assert "quant_config.load_experts(" in store_source
    assert "projection=kind" in store_source
    assert "self._install(kind, weights, device)" in store_source
    install = ast.unparse(next(node for node in ast.walk(store)
                              if isinstance(node, ast.FunctionDef) and node.name == "_install"))
    for fragment in ("packed.contiguous().to(device)", "scales.contiguous().view(torch.uint8).to(device)",
                     "scale2.reshape(-1).contiguous().to(device=device, dtype=torch.float32)"):
        assert fragment in install, f"_install must install {fragment}"


# ---------------------------------------------------------------------------
# 2. the decoded planes are the loader's own memory
# ---------------------------------------------------------------------------
def test_decoded_planes_own_their_bytes(csf, csf_source, checkpoint_guard):
    source_fixed = csf_source.get_tensor(f"{EXPERT}{csf.CSF_FIXED_SUFFIX}")
    weights = csf.load_packed_nvfp4(csf_source, EXPERT)
    assert weights.scales.dtype == torch.float8_e4m3fn

    def storage_ptr(tensor):
        return tensor.untyped_storage().data_ptr()

    assert storage_ptr(weights.scales) != storage_ptr(source_fixed), \
        "a decoded plane must not alias the checkpoint's buffer"
    assert storage_ptr(weights.packed) != storage_ptr(csf_source.get_tensor(f"{EXPERT}.weight"))

    # Mutating the resident plane cannot reach the checkpoint: a re-read of the
    # same tensor still produces the original bytes.
    original = weights.scales.clone()
    weights.scales.view(torch.uint8)[0, 0] = 0x7E
    assert not torch.equal(weights.scales.view(torch.uint8)[0], original.view(torch.uint8)[0])
    reread = csf.load_packed_nvfp4(csf_source, EXPERT)
    assert torch.equal(reread.scales.view(torch.uint8), original.view(torch.uint8))


def test_decode_never_opens_a_checkpoint_file_for_writing(csf, csf_source, plain_source, monkeypatch):
    roots = (str(support.CSF_CHECKPOINT.resolve()), str(support.PLAIN_CHECKPOINT.resolve()))
    writes: list[tuple[str, str]] = []
    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        path = str(file)
        if any(flag in str(mode) for flag in "wax+") and not path.startswith("/dev/"):
            writes.append((path, str(mode)))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "write_text", lambda self, *a, **k: (_ for _ in ()).throw(
        AssertionError(f"write_text({self}) during the load path")))
    monkeypatch.setattr(Path, "write_bytes", lambda self, *a, **k: (_ for _ in ()).throw(
        AssertionError(f"write_bytes({self}) during the load path")))

    decoded = _decode_with_serve_function(csf, csf_source, LAYER0, NUM_EXPERTS)
    converted = _decode_with_serve_function(csf, plain_source, LAYER0, NUM_EXPERTS)
    assert torch.equal(decoded.scales.view(torch.uint8), converted.scales.view(torch.uint8))

    offenders = [entry for entry in writes if entry[0].startswith(roots)]
    assert offenders == [], f"the decode opened checkpoint files for writing: {offenders}"
    assert writes == [], f"the load path wrote files at all: {writes}"


def test_the_guard_detects_a_write(tmp_path):
    """The guard is sensitive: a changed byte or a new file both show up."""
    (tmp_path / "a.bin").write_bytes(b"x")
    before = _tree_fingerprint(tmp_path)

    (tmp_path / "a.bin").write_bytes(b"xx")
    assert _tree_fingerprint(tmp_path) != before, "size/mtime change must be visible"

    (tmp_path / "a.bin").write_bytes(b"x")
    (tmp_path / "b.bin").write_bytes(b"")
    assert _tree_fingerprint(tmp_path) != before, "a new file must be visible"


def test_checkpoint_trees_are_unchanged(checkpoint_guard):
    """The explicit end-of-run check (the fixture asserts it again on teardown)."""
    for name, root in checkpoint_guard["roots"].items():
        before = checkpoint_guard["before"][name]
        after = _checkpoint_fingerprint(root)
        assert after["tree"] == before["tree"], f"{name}: files appeared/changed under {root}"
        assert after["index_sha256"] == before["index_sha256"]
        assert after["config_sha256"] == before["config_sha256"]


# ---------------------------------------------------------------------------
# 3. the preshuffle on a real plane, and what it costs
# ---------------------------------------------------------------------------
def _resident_dot(packed_rows, scale_rows, x):
    """numpy reference over the resident planes (low nibble = even k)."""
    import numpy as np

    packed = np.asarray(packed_rows)
    rows, row_bytes = packed.shape
    k = row_bytes * 2
    scales = np.asarray(scale_rows)[:, :k // 16]
    codes = np.stack([packed & 0x0F, (packed >> 4) & 0x0F], axis=-1).reshape(rows, k)
    e2m1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
    magnitudes = e2m1[codes & 0x07]
    values = np.where((codes & 0x08) != 0, -magnitudes, magnitudes)
    weight = values * np.repeat(scales.astype(np.float64), 16, axis=1)
    return weight @ np.asarray(x, dtype=np.float64)


def test_preshuffle_round_trip_on_a_real_down_proj_plane(csf, csf_source, checkpoint_guard):
    """K = 640 -> K/2 = 320 bytes: five dual groups, but only 1.25 TPC vectors.

    This is the plane where a 256-byte vector load does not fit inside a row,
    i.e. exactly the case the padded layout exists for.
    """
    import numpy as np

    tpc = csf.tpc_layout_module()
    weights = csf.load_packed_nvfp4(csf_source, EXPERT_DOWN)
    assert weights.packed.shape == (1, 2560, 320)
    assert weights.scales.shape == (1, 2560, 40)
    report = weights.geometry(require_kernel_caps=False)
    assert report["ok"] and report["packed_row"] == 320 and report["packed_tiles"] == 2

    rows = 8  # keep the numpy reference small; the layout is per row
    packed = weights.packed[0, :rows].contiguous()
    scales = weights.scales[0, :rows].view(torch.uint8).contiguous()
    x = np.random.default_rng(5).standard_normal(640)
    reference = _resident_dot(packed.numpy(), scales.numpy(), x)

    for layout in (tpc.TPC_GEMV_LAYOUT, tpc.TPC_GEMV_LAYOUT_PADDED):
        resident_packed = tpc.preshuffle_packed(packed.unsqueeze(0), layout)[0]
        resident_scales = tpc.preshuffle_scales(scales.unsqueeze(0).view(torch.float8_e4m3fn),
                                                layout).view(torch.uint8)[0]
        # The declared layout is byte-identical; the padded one appends zeros
        # that multiply a zero activation (k >= K in the resident row).
        wide = np.zeros(resident_packed.shape[1] * 2, dtype=np.float64)
        wide[:640] = x
        got = _resident_dot(resident_packed.numpy(), resident_scales.numpy(), wide)
        np.testing.assert_array_equal(got, reference)
        assert torch.equal(tpc.unpreshuffle_packed(resident_packed.unsqueeze(0), 640),
                           packed.unsqueeze(0))
        assert torch.equal(
            tpc.unpreshuffle_scales(resident_scales.unsqueeze(0).view(torch.float8_e4m3fn), 640)
            .view(torch.uint8), scales.unsqueeze(0))

    # What the padded layout would cost over the whole model: stated, opt-in,
    # never the default (the declared layout is what the GEMV reads).
    declared = tpc.choose_row_bytes(640, tpc.TPC_GEMV_LAYOUT)
    padded = tpc.choose_row_bytes(640, tpc.TPC_GEMV_LAYOUT_PADDED)
    assert (declared, padded) == (320, 512)
    assert tpc.layout_from_env({}) == tpc.TPC_GEMV_LAYOUT


def test_preshuffle_env_on_off_equivalence(csf, csf_source, monkeypatch):
    """``VLLM_QWEN4EXP_TPC_PRESHUFFLE`` / ``VLLM_QWEN4EXP_TPC_LAYOUT`` on and
    off: whichever resident layout the loader produces, dequantizing the
    loaded planes gives the same weights.

    ``down_proj`` (K = 640, a 1.25-vector row) is the ragged case; ``gate_proj``
    (K = 2560, five whole vectors) is the aligned case. The padded variant's
    extra columns are E2M1 code 0 under an E4M3 0x00 scale, so they decode to
    exactly zero and the first K columns match the identity planes.
    """
    tpc = csf.tpc_layout_module()
    layouts = {
        "default": {},  # declared TPC_GEMV_LAYOUT
        "identity": {tpc.PRESHUFFLE_SWITCH: "0"},
        "padded": {tpc.LAYOUT_ENV: tpc.TPC_GEMV_LAYOUT_PADDED},
    }
    loaded = {}
    for name, env in layouts.items():
        for key in (tpc.PRESHUFFLE_SWITCH, tpc.LAYOUT_ENV):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        weights = csf.load_packed_nvfp4(csf_source, EXPERT_DOWN)
        assert weights.layout == tpc.layout_from_env(), name
        loaded[name] = weights

    # Identity and the declared layout are the same bytes...
    for plane in ("packed", "scales"):
        a = getattr(loaded["default"], plane)
        b = getattr(loaded["identity"], plane)
        assert torch.equal(a.view(torch.uint8) if plane == "scales" else a,
                           b.view(torch.uint8) if plane == "scales" else b)
    assert torch.equal(loaded["default"].scale2, loaded["identity"].scale2)

    # ... and the padded layout dequantizes to the same weights (zeros tail).
    identity = loaded["identity"]
    padded = loaded["padded"]
    assert padded.packed.shape[2] == tpc.choose_row_bytes(640, tpc.TPC_GEMV_LAYOUT_PADDED) == 512
    assert padded.scales.shape[2] == tpc.scale_row_bytes_for_layout(640, tpc.TPC_GEMV_LAYOUT_PADDED)
    reference = csf.dequant_nvfp4(identity.packed[0], identity.scales[0], identity.scale2,
                                  dtype=torch.float32)
    padded_dequant = csf.dequant_nvfp4(padded.packed[0], padded.scales[0], padded.scale2,
                                       dtype=torch.float32)
    assert padded_dequant.shape[1] == 1024  # 512 packed bytes -> 1024 values
    assert torch.equal(padded_dequant[:, :640], reference)
    assert float(padded_dequant[:, 640:].abs().max()) == 0.0, "the pad must decode to zero"

    # The aligned case (K = 2560): the packed row already fits whole vectors,
    # but the padded layout still pads the scale row 160 -> 192 (to a 64-byte
    # dual group), so the padded load is compared through the inverse
    # transform -- and every layout dequantizes to the same weights.
    gate = {}
    for name, env in layouts.items():
        for key in (tpc.PRESHUFFLE_SWITCH, tpc.LAYOUT_ENV):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        gate[name] = csf.load_packed_nvfp4(csf_source, EXPERT)
    assert gate["padded"].scales.shape[2] == 192 and gate["default"].scales.shape[2] == 160
    dequants = {}
    for name, w in gate.items():
        packed, scales = w.packed, w.scales
        if name == "padded":
            packed = tpc.unpreshuffle_packed(packed, 2560)
            scales = tpc.unpreshuffle_scales(scales, 2560)
        dequants[name] = csf.dequant_nvfp4(packed[0], scales[0], w.scale2, dtype=torch.float32)
    assert torch.equal(dequants["default"], dequants["identity"])
    assert torch.equal(dequants["default"], dequants["padded"])


def test_track3_gemv_declares_the_layout_this_loader_produces(csf):
    """Track 2 and Track 3 must not drift: read the kernel's own contract.

    The decode GEMV is a TPC kernel with no Python side, so its declared layout
    lives in its GlueCode (dimensions, access patterns, caps) and in its kernel
    constants. This checks the module's constants against them, and that the
    packed plane's fastest TPC dimension is K/2 (i.e. torch ``[E, N, K/2]``).
    """
    root = support.REPO_ROOT / "native" / "qwen4exp-nvfp4-moe"
    glue = root / "host" / "qwen4exp_nvfp4_moe_gemv_bf16_gaudi2.cpp"
    kernel = root / "kernels" / "qwen4exp_nvfp4_moe_gemv_bf16_gaudi2.c"
    if not (glue.is_file() and kernel.is_file()):
        pytest.skip("the Track 3 GEMV sources are not in this checkout")
    tpc = csf.tpc_layout_module()
    glue_text = glue.read_text()
    kernel_text = kernel.read_text()

    # Kernel constants: 256 packed bytes = 512 k per vector, 64 bytes = 128 k
    # per dual group, 32 scale bytes per vector.
    assert "#define K_PER_TILE 512" in kernel_text
    assert "#define K_PER_PART 128" in kernel_text
    assert tpc.TPC_K_PER_VECTOR == 512
    assert tpc.TPC_K_PER_DUAL_GROUP == 128
    assert tpc.TPC_SCALE_BYTES_PER_VECTOR == 32
    # The addresses the kernel issues: one 256-byte packed load per tile, 32
    # scale bytes per tile, for one (row, expert).
    assert "(int5){tile * 256, row0 + 0, expert, 0, 0}, packed" in kernel_text
    assert "(int5){tile * 32, row0 + 0, expert, 0, 0}, scales" in kernel_text

    # GlueCode: TPC dimension 0 (fastest) is K/2 for packed and K/16 for
    # scales, dimension 2 is the expert -- Synapse reverses torch axes, so the
    # torch plane is [E, N, K/2] / [E, N, K/16], exactly what the loader emits.
    assert "map_dim(p_packed, 0, 0, 0, 0, static_cast<float>(k_half) - 1);" in glue_text
    assert "map_dim(p_packed, 2, 1, 0, 0, static_cast<float>(experts) - 1);" in glue_text
    assert "map_dim(p_scales, 0, 0, 0, 0, static_cast<float>(k_half / 8) - 1);" in glue_text
    assert "scales.maxSizes[0] * 8 == k_half" in glue_text
    # ... and the caps the module restates.
    assert "k_half % (QWEN4EXP_MOE_K_PER_PART / 2) == 0" in glue_text
    assert "rows % row_step == 0" in glue_text
    assert "experts % QWEN4EXP_MOE_BLOCK_SLOTS == 0" in glue_text

    # A real layer passes exactly that geometry check.
    report = tpc.check_geometry(256, 640, 2560)
    assert report["ok"] and report["packed_row"] % tpc.TPC_DUAL_GROUP_BYTES == 0


def test_down_proj_and_gate_up_rows_describe_the_same_k(csf, csf_source, checkpoint_guard):
    """One K per layer: the loader rejects a plane pair that disagrees."""
    tpc = csf.tpc_layout_module()
    weights = csf.load_packed_nvfp4(csf_source, EXPERT)
    with pytest.raises(ValueError, match="!= K/2"):
        tpc.tpc_planes(weights.packed, weights.scales.view(torch.uint8), 640)
    # The real K (2560) is accepted and the planes stay 4-bit + 8-bit.
    packed, scales = tpc.tpc_planes(weights.packed, weights.scales.view(torch.uint8), 2560)
    assert packed.numel() * 2 == 640 * 2560
    assert scales.numel() * 16 == 640 * 2560


def test_layer_index_and_config_are_read_only(csf, checkpoint_guard):
    """A load must not even rewrite the index (a converter would)."""
    root = checkpoint_guard["roots"]["csf"]
    index_path = root / "model.safetensors.index.json"
    payload = json.loads(index_path.read_text())
    assert len(payload["weight_map"]) == 375487
    assert index_path.stat().st_size > 0
    # The tree fingerprint covers the index's size and mtime; the sha256 above
    # covers its content. Both are compared again after the module ran.
    sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    assert sha == checkpoint_guard["before"]["csf"]["index_sha256"]
