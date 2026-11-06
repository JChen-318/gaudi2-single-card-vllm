"""Host-only checks of the Qwen4Exp NVFP4 MoE TPC library (Track C).

Four independent checks, no device and no Habana runtime:

1. ``test_glue_geometry`` runs the library's own host geometry test binary
   (``native/qwen4exp-nvfp4-moe/tests/glue_geometry.cpp``) on the built
   ``libqwen4exp_nvfp4_moe_kernels.so``: GUID registration, GlueCode index
   spaces for the shapes this checkpoint serves (hidden 2560, moe intermediate
   640, 512 experts, 10 routed slots) and the documented caps.
2. ``test_shim_reports_new_guids`` loads the library behind
   ``native/tpc-kernel-shim`` and asserts the four new GUIDs are part of the
   merged single-file kernel list (the list ``GC_KERNEL_PATH`` points at).
3. ``test_simulator_runs_the_kernels`` runs the real TPC ELFs in the Habana TPC
   simulator (``libtpcsim_shared``, host only) and checks the harness verdict:
   one-hot decode bit-exact, dense accumulation against a double reference, and
   the bucketing integer arithmetic exact.
4. The remaining tests check the numerics: a numpy port of the kernel's
   ``nvfp4_mul_bits`` bit construction and of the E2M1-LUT x E4M3-scale decode
   are compared against vllm-core's ``dequantize_to_dtype`` (the NVFP4
   reference), and the F32 accumulate + per-expert ``scale2`` epilogue is
   compared against the same reference on synthetic data.

The reference is read out of ``vllm-core`` by AST and executed with a stub
platform object: importing vllm would activate the HPU plugin (and with it
``habana_frameworks``), which this host-only test must not do.

Build the binaries first (``scripts/build_native.sh`` does exactly this):

    cmake -S native/qwen4exp-nvfp4-moe -B build/native/qwen4exp-nvfp4-moe \\
        -DCMAKE_BUILD_TYPE=Release
    cmake --build build/native/qwen4exp-nvfp4-moe --parallel
    cp build/native/qwen4exp-nvfp4-moe/libqwen4exp_nvfp4_moe_kernels.so build/native/
    cp build/native/qwen4exp-nvfp4-moe/qwen4exp_nvfp4_moe_geometry_test build/native/

Run: ``.venv/bin/python -m pytest -q validation/qwen4exp/test_nvfp4_moe_geometry.py``

This checkout's ``pytest_compat`` plugin imports vllm (and with it
``habana_frameworks``) for every pytest run; this module itself touches neither.
``-p no:vllm_gaudi_compat`` runs it with no HPU import at all.
"""
import ast
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "native" / "qwen4exp-nvfp4-moe"
BUILD = ROOT / "build" / "native"
LIBRARY = BUILD / "libqwen4exp_nvfp4_moe_kernels.so"
GEOMETRY_TEST = BUILD / "qwen4exp_nvfp4_moe_geometry_test"
SIMULATOR = BUILD / "qwen4exp_nvfp4_moe_sim"
SHIM = BUILD / "libkiro_tpc_shim.so"
SHIM_CHECK = BUILD / "shim_check"
STOCK = Path("/usr/lib/habanalabs/libtpc_kernels.so")

GUID_PREFIX = "custom_qwen4exp_nvfp4_moe_"
GUIDS = (
    "custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2",
    "custom_qwen4exp_nvfp4_moe_count_i32_gaudi2",
    "custom_qwen4exp_nvfp4_moe_prefix_i32_gaudi2",
    "custom_qwen4exp_nvfp4_moe_scatter_i32_gaudi2",
)

REFERENCE_SOURCE = (ROOT / "vllm-core" / "vllm" / "model_executor" / "layers" /
                    "quantization" / "utils" / "nvfp4_emulation_utils.py")

# --------------------------------------------------------------------------
# 1. the library's own glue geometry test
# --------------------------------------------------------------------------


def _run(argv, **kwargs):
    return subprocess.run(argv, capture_output=True, text=True, timeout=60,
                          check=False, **kwargs)


def test_glue_geometry():
    if not (LIBRARY.exists() and GEOMETRY_TEST.exists()):
        pytest.skip("build the library first (scripts/build_native.sh)")
    result = _run([str(GEOMETRY_TEST), str(LIBRARY)])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "glue validation: all tests passed" in result.stdout
    for guid in GUIDS:
        assert guid in result.stdout, guid


def test_simulator_runs_the_kernels():
    """The TPC simulator executes the real ELFs; the harness compares against a
    double-precision reference (one-hot decode bit-exact, dense accumulation and
    the bucketing integer arithmetic)."""
    if not SIMULATOR.exists():
        pytest.skip("TPC simulator harness not built (needs libtpcsim_shared)")
    result = _run([str(SIMULATOR)])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "simulator validation: all tests passed" in result.stdout
    assert "outputs bit-exact vs decoded weights" in result.stdout
    assert "exact mismatches=0" in result.stdout
    assert "counts exact" in result.stdout


# --------------------------------------------------------------------------
# 2. the shim's merged GUID list contains the new kernels
# --------------------------------------------------------------------------


def test_shim_reports_new_guids():
    if not (SHIM.exists() and SHIM_CHECK.exists() and LIBRARY.exists()):
        pytest.skip("build the shim and the library first (scripts/build_native.sh)")
    # The same order scripts/build_native.sh uses, with whichever of the other
    # custom libraries exist in this checkout.
    components = [
        BUILD / "libmtp_sampling_kernels_v2.so",
        BUILD / "libflashinfer_gaudi_kernels.so",
        LIBRARY,
        STOCK,
    ]
    components = [path for path in components if path.exists()]
    assert LIBRARY in components and STOCK in components
    env = {"KIRO_TPC_LIBS": ":".join(str(path) for path in components)}
    result = _run([str(SHIM_CHECK), str(SHIM), *map(str, components)], env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "missing_from_merged=0" in result.stdout
    custom = [line for line in result.stdout.splitlines() if line.startswith("custom kernels:")]
    assert len(custom) == 1
    merged = set(custom[0][len("custom kernels:"):].split())
    for guid in GUIDS:
        assert guid in merged, f"{guid} missing from the shim's merged list"
    # every GUID of every component survives the merge
    assert "sum=" in result.stdout and "merged_unique=" in result.stdout


# --------------------------------------------------------------------------
# 3. numerics against the vllm-core NVFP4 reference
# --------------------------------------------------------------------------


class _StubPlatform:
    """dequantize_to_dtype only asks whether the Triton path applies."""

    @staticmethod
    def is_cuda_alike():
        return False


def _load_reference():
    """Execute vllm-core's NVFP4 dequant reference without importing vllm."""
    wanted = {"break_fp4_bytes", "convert_swizzled_to_linear", "dequantize_to_dtype"}
    tree = ast.parse(REFERENCE_SOURCE.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
                getattr(target, "id", "") == "kE2M1ToFloat_handle" for target in node.targets):
            nodes.append(node)
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = {
        "torch": torch,
        "SimpleNamespace": __import__("types").SimpleNamespace,
        "current_platform": _StubPlatform(),
    }
    exec(compile(module, str(REFERENCE_SOURCE), "exec"), namespace)
    return namespace


REFERENCE = _load_reference()

E2M1_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def e4m3_to_float(bits):
    """Exact E4M3 (fn) decode: the scale values NVFP4 stores per 16 elements."""
    b = np.asarray(bits, dtype=np.uint8).astype(np.int32)
    sign = np.where((b >> 7) & 1, -1.0, 1.0).astype(np.float32)
    exponent = (b >> 3) & 0xF
    mantissa = b & 0x7
    assert not np.any((exponent == 0xF) & (mantissa == 0x7)), "NaN E4M3 scale"
    # e == 0: subnormal, value = m/8 * 2**-6   (E4M3 bias 7)
    # e > 0: normal,    value = (1 + m/8) * 2**(e-7)
    scale = np.where(exponent == 0, 2.0 ** -6, np.exp2(exponent.astype(np.float64) - 7.0))
    significand = np.where(exponent == 0, mantissa / 8.0, 1.0 + mantissa / 8.0)
    return (sign * scale * significand).astype(np.float32)


def nvfp4_decode_lut(packed, scales, block=16):
    """The mathematical NVFP4 decode: E2M1 LUT x per-16 E4M3 scale.

    packed [N, K/2] U8, low nibble first (even k), scales [N, K/16] U8.
    Returns [N, K] float32: the exact product, which is representable in BF16.
    """
    packed = np.asarray(packed, dtype=np.uint8)
    nibbles = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    nibbles[:, 0::2] = packed & 0x0F
    nibbles[:, 1::2] = (packed >> 4) & 0x0F
    values = E2M1_LUT[nibbles & 0x07] * np.where(nibbles & 0x08, -1.0, 1.0).astype(np.float32)
    per_block = e4m3_to_float(scales)
    return values * np.repeat(per_block, block, axis=1)


def nvfp4_mul_bits(nibble, scale):
    """numpy transcription of nvfp4_mul_bits() (include/*_decode.h), BF16 bits."""
    nibble = np.asarray(nibble, dtype=np.uint16)
    scale = np.asarray(scale, dtype=np.uint16)
    mask = np.uint16(0xFFFF)
    mag = nibble & 7
    odd = nibble & 1
    sign = ((nibble & 8) << 12) ^ ((scale & 128) << 8)
    e = (scale >> 3) & 15
    m = scale & 7
    base_n = ((m + 8) << 2) & mask
    base_s = (m << 2) & mask
    base = np.where(e == 0, base_s, base_n).astype(np.uint16)
    extra = np.where(odd == 1, np.where(mag == 1, 0, base >> 1), 0).astype(np.uint16)
    fixed0 = (base + extra) & mask
    efield = (e + (mag >> 1) + 119 + np.where(e == 0, 1, 0)).astype(np.uint16)
    over = np.where(fixed0 >= 64, 1, 0).astype(np.uint16)
    fixed0 = np.where(fixed0 >= 64, fixed0 >> 1, fixed0).astype(np.uint16)
    efield = (efield + over) & mask
    for _ in range(3):
        under = np.where(fixed0 < 32, 1, 0).astype(np.uint16)
        fixed0 = np.where(fixed0 < 32, (fixed0 << 1) & mask, fixed0).astype(np.uint16)
        efield = (efield - under) & mask
    bits = (((fixed0 - 32) << 2) | (efield << 7)) & mask
    bits = np.where(mag == 0, 0, bits).astype(np.uint16)
    bits = np.where((m == 0) & (e == 0), 0, bits).astype(np.uint16)
    return (bits | sign).astype(np.uint16)


def _synthetic(num_rows, k, seed):
    """Random packed nibbles and E4M3 scales; no NaN scales (0x7F/0xFF)."""
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 256, size=(num_rows, k // 2), dtype=np.uint8)
    scales = rng.integers(0, 256, size=(num_rows, k // 16), dtype=np.uint8)
    scales[scales == 0x7F] = 0x3F
    scales[scales == 0xFF] = 0xBF
    return packed, scales


def test_decode_is_bit_exact_against_vllm_reference():
    """E2M1 LUT x E4M3 scale == vllm-core dequantize_to_dtype, bit for bit."""
    rows, k = 6, 512
    packed, scales = _synthetic(rows, k, seed=0)
    ours = nvfp4_decode_lut(packed, scales)
    reference = REFERENCE["dequantize_to_dtype"](
        torch.from_numpy(packed), torch.from_numpy(scales),
        global_scale=torch.ones(1, dtype=torch.float32), dtype=torch.float32,
        block_size=16, swizzle=False)
    reference = reference.numpy().reshape(rows, k)
    # The product has at most 7 significand bits, so both routes are exact F32.
    assert np.array_equal(ours.view(np.uint32), reference.view(np.uint32)), (
        f"max abs diff {np.abs(ours - reference).max()}")


def test_bit_construction_matches_lut():
    """nvfp4_mul_bits (the kernel's integer decode) == LUT x E4M3 scale, bit exact."""
    nibbles = np.arange(16, dtype=np.uint8)
    scale_bytes = np.array([b for b in range(256) if b not in (0x7F, 0xFF)], dtype=np.uint8)
    grid_nibble, grid_scale = np.meshgrid(nibbles, scale_bytes, indexing="ij")
    bits = nvfp4_mul_bits(grid_nibble, grid_scale)
    sign = np.where(grid_nibble & 0x08, -1.0, 1.0).astype(np.float32)
    exact = (sign * E2M1_LUT[grid_nibble & 0x07] * e4m3_to_float(grid_scale)).astype(np.float32)
    expected = (exact.view(np.uint32) >> 16).astype(np.uint16)
    mismatch = int(np.count_nonzero(bits != expected))
    assert mismatch == 0, f"{mismatch} of {bits.size} (nibble, scale) pairs differ"


def test_f32_accumulate_and_scale2_match_reference():
    """Kernel arithmetic: F32 block sum, then scale2 once at the end.

    The kernel accumulates bf16(e2m1*e4m3) * x in F32 and multiplies by the
    per-expert scale2 after the sum; the reference dequantizes to F32 and sums in
    float64 with scale2 folded per element. The decode itself is bit-exact (test
    above), so the remaining difference is F32 summation order plus where scale2
    is applied: bound it instead of expecting bit equality.
    """
    rows, k, slots = 4, 1280, 3      # 1280 = K of the w13 expert projection
    packed, scales = _synthetic(rows, k, seed=7)
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((slots, k)) * 0.1).astype(np.float32)
    scale2 = np.array([0.5, 1.0, 0.75, 1.25], dtype=np.float32)
    weights = nvfp4_decode_lut(packed, scales)
    ours = np.einsum("nk,sk->ns", weights, x, dtype=np.float32) * scale2[:, None]
    reference = REFERENCE["dequantize_to_dtype"](
        torch.from_numpy(packed), torch.from_numpy(scales),
        global_scale=torch.ones(1, dtype=torch.float32), dtype=torch.float32,
        block_size=16, swizzle=False).numpy().reshape(rows, k)
    expected = np.einsum("nk,sk->ns", reference.astype(np.float64),
                         x.astype(np.float64)) * scale2[:, None]
    tolerance = np.maximum(np.abs(expected), 1e-3)
    assert np.max(np.abs(ours - expected) / tolerance) < 2e-3
    # Applying scale2 after the sum must be equivalent to folding it in per element
    # up to the same F32 noise (the kernel's epilogue is a mathematical factor).
    folded = np.einsum("nk,sk->ns", weights * scale2[:, None], x, dtype=np.float32)
    assert np.max(np.abs(folded - ours) / tolerance) < 2e-3
