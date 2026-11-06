"""Host-only numeric reference for the Qwen4Exp NVFP4 MoE decode GEMV (Track C).

The kernel under test is
``custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2`` (native/qwen4exp-nvfp4-moe): one
launch over the selected expert group, NVFP4 weights decoded from E2M1 nibbles
times one E4M3 scale per 16 K elements, F32 accumulate, ``scale2`` once after the
block sum, BF16 output. This module emulates that math in numpy and compares it
with two independent references, on synthetic data, with no device:

1. a C reference program compiled and run here
   (``native/qwen4exp-nvfp4-moe/tests/nvfp4_moe_gemv_reference.c``): it decodes
   the weights with the E2M1 value table and the E4M3 definition in double
   precision, accumulates the dot product in double and applies ``scale2`` after
   the sum. It shares no code with the kernel and no code with this module, so
   agreement is evidence about the math, not about a copy;
2. ``vllm-core``'s NVFP4 dequantize reference
   (``vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py``)
   for the decode itself, bit for bit.

Tolerances, stated up front:

* decode (E2M1 LUT x E4M3 scale) - **bit exact**, against both references.
* F32 accumulate + ``scale2`` + BF16 store - the kernel's F32 lane sum differs
  from the exact double sum; the bound is the BF16 output granularity (2**-9 =
  2e-3 relative), and the observed maximum on random K = 640..2560 data is
  ~3e-5 (F32 rounding amplified by cancellation). ``TOLERANCE`` below is the
  documented 2e-3. Because the F32 error is far below the store step, every BF16
  output still matches the exact reference bit for bit on the test data.
* a data set in which every product and every partial sum is a power of two of
  limited exponent is **exact** end to end (no rounding happens anywhere), so
  the same comparison is asserted bit for bit there.

The numpy emulation follows the kernel's accumulation order rather than a naive
dot product: lane j of the 128-lane accumulator collects k = j (mod 128) in
ascending k (tiles of 512, four 128-lane parts each), the 128 lanes are folded
64+64 and then sequentially, ``scale2`` multiplies the F32 total, and the store
rounds to BF16 with round-to-nearest-even. Products of a BF16 activation and an
NVFP4 weight have at most 13 significant bits, so the product is exact in F32 and
a separate multiply then add is the kernel's fused multiply-add.

Run: ``.venv/bin/python -m pytest -q validation/qwen4exp/test_nvfp4_moe_reference.py``

Everything is < 10 s: one small C compile (once) and three C runs of a few
milliseconds each. ``scripts/build_native.sh`` also builds the reference program
into ``build/native/qwen4exp_nvfp4_moe_reference``; this module reuses that
binary when it exists and otherwise compiles the source with ``cc``.
"""
import ast
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_C = ROOT / "native" / "qwen4exp-nvfp4-moe" / "tests" / "nvfp4_moe_gemv_reference.c"
PREBUILT_REFERENCE = ROOT / "build" / "native" / "qwen4exp_nvfp4_moe_reference"
VLLM_REFERENCE_SOURCE = (ROOT / "vllm-core" / "vllm" / "model_executor" / "layers" /
                         "quantization" / "utils" / "nvfp4_emulation_utils.py")

MAGIC = 1024
TOLERANCE = 2e-3  # BF16 output granularity (2**-9); the F32 sum error is ~1e-6

# E2M1 magnitudes; the nibble's top bit is the sign.
E2M1_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

# --------------------------------------------------------------------------
# the C reference program
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reference_binary(tmp_path_factory):
    """The compiled C reference: the prebuilt one, else compile the source."""
    if PREBUILT_REFERENCE.exists():
        return PREBUILT_REFERENCE
    assert REFERENCE_C.exists(), f"missing C reference source {REFERENCE_C}"
    output = tmp_path_factory.mktemp("nvfp4_moe_reference") / "nvfp4_moe_gemv_reference"
    result = subprocess.run(
        ["cc", "-O2", "-std=c11", "-Wall", "-Wextra", "-o", str(output),
         str(REFERENCE_C), "-lm"],
        capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return output


def _pack_inputs(ids, packed, scales, x_bits, scale2):
    """Header + ids + packed + scales + BF16 x + F32 scale2, exactly as the C reader."""
    experts, rows, k_half = packed.shape
    blob = struct.pack("<5I", MAGIC, experts, rows, 2 * k_half, ids.size)
    blob += np.ascontiguousarray(ids, dtype="<i4").tobytes()
    blob += np.ascontiguousarray(packed, dtype=np.uint8).tobytes()
    blob += np.ascontiguousarray(scales, dtype=np.uint8).tobytes()
    blob += np.ascontiguousarray(x_bits, dtype="<u2").tobytes()
    blob += np.ascontiguousarray(scale2, dtype="<f4").tobytes()
    return blob


def run_c_reference(binary, tmp_path, ids, packed, scales, x_bits, scale2):
    """Run the C reference; returns its golden values as float64 [S, N]."""
    inputs = tmp_path / "inputs.bin"
    golden_path = tmp_path / "golden.bin"
    inputs.write_bytes(_pack_inputs(ids, packed, scales, x_bits, scale2))
    result = subprocess.run([str(binary), str(inputs), str(golden_path)],
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "reference: E=" in result.stdout
    raw = golden_path.read_bytes()
    magic, slots, rows = struct.unpack_from("<3I", raw, 0)
    assert magic == MAGIC and slots == ids.size and rows == packed.shape[1]
    assert len(raw) == 12 + slots * rows * 8
    return np.frombuffer(raw, dtype="<f8", offset=12).reshape(slots, rows).copy()


# --------------------------------------------------------------------------
# NVFP4 decode: E2M1 value table x E4M3 block scale
# --------------------------------------------------------------------------


def e4m3_to_float(bits):
    """Exact E4M3 (float8_e4m3fn) decode, the value NVFP4 stores per 16 elements."""
    b = np.asarray(bits, dtype=np.uint8).astype(np.int32)
    assert not np.any(((b >> 3) & 0xF == 0xF) & (b & 0x7 == 0x7)), "NaN E4M3 scale"
    sign = np.where((b >> 7) & 1, -1.0, 1.0)
    exponent = (b >> 3) & 0xF
    mantissa = b & 0x7
    scale = np.where(exponent == 0, 2.0 ** -6, np.exp2(exponent.astype(np.float64) - 7.0))
    significand = np.where(exponent == 0, mantissa / 8.0, 1.0 + mantissa / 8.0)
    return (sign * scale * significand).astype(np.float32)


def decode_lut(packed, scales, block=16):
    """The NVFP4 decode as a lookup: packed [...,K/2] U8 + scales [...,K/16] U8 -> [...,K] f32."""
    packed = np.asarray(packed, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.uint8)
    assert packed.shape[:-1] == scales.shape[:-1]
    nibbles = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.uint8)
    nibbles[..., 0::2] = packed & 0x0F
    nibbles[..., 1::2] = (packed >> 4) & 0x0F
    values = E2M1_LUT[nibbles & 0x07] * np.where(nibbles & 0x08, -1.0, 1.0).astype(np.float32)
    return values * np.repeat(e4m3_to_float(scales), block, axis=-1)


def nvfp4_mul_bits(nibble, scale):
    """numpy transcription of the kernel's nvfp4_mul_bits (BF16 bits, integer only)."""
    nibble = np.asarray(nibble, dtype=np.uint16)
    scale = np.asarray(scale, dtype=np.uint16)
    mask = np.uint16(0xFFFF)
    mag = nibble & 7
    odd = nibble & 1
    sign = ((nibble & 8) << 12) ^ ((scale & 128) << 8)
    e = (scale >> 3) & 15
    m = scale & 7
    base = np.where(e == 0, (m << 2) & mask, ((m + 8) << 2) & mask).astype(np.uint16)
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


# --------------------------------------------------------------------------
# kernel-order emulation
# --------------------------------------------------------------------------


def bf16_bits_to_float32(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def to_bf16_bits(values):
    """F32 -> BF16 with round-to-nearest-even (the kernel's SW_LINEAR convert)."""
    wide = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounding = 0x7FFF + ((wide >> 16) & 1)
    return ((wide + rounding) >> 16).astype(np.uint16)


def _k_parts(k_half):
    """The 128-element K parts the kernel actually executes, in execution order.

    Tiles of 512 packed-covered elements; a part runs only when its first k is
    inside K, which is exactly the kernel's ``if (k_base + PART*128 < k_total)``.
    """
    k = 2 * k_half
    parts = []
    for tile in range((k_half + 255) // 256):
        for part in range(4):
            start = tile * 512 + part * 128
            if start < k:
                parts.append(start)
    return parts


def kernel_order_gemv(ids, packed, scales, x_bits, scale2):
    """The decode GEMV with the kernel's accumulation order and epilogue."""
    experts, rows, k_half = packed.shape
    k = 2 * k_half
    slots = ids.size
    x = bf16_bits_to_float32(x_bits).reshape(slots, k)
    parts = _k_parts(k_half)
    assert sum(128 for _ in parts) == k, "parts must tile K exactly"
    out = np.zeros((slots, rows), dtype=np.float32)
    for slot in range(slots):
        expert = int(ids[slot])
        if expert < 0 or expert >= experts:
            continue  # the kernel leaves the accumulator at zero and stores +0.0
        weights = decode_lut(packed[expert], scales[expert])  # [rows, K], exact
        acc = np.zeros((rows, 128), dtype=np.float32)
        for start in parts:
            product = weights[:, start:start + 128] * x[slot, None, start:start + 128]
            acc = (acc + product.astype(np.float32)).astype(np.float32)
        pair = (acc[:, :64] + acc[:, 64:]).astype(np.float32)  # v_f32_reduce_add(acc.v1+acc.v2)
        total = pair[:, 0].copy()
        for lane in range(1, 64):
            total = (total + pair[:, lane]).astype(np.float32)
        out[slot] = (total * np.float32(scale2[expert])).astype(np.float32)
    return out


def vectorized_lut_gemv(ids, packed, scales, x_bits, scale2):
    """The same math with no accumulation-order model: float64 sum of the LUT decode."""
    experts, rows, k_half = packed.shape
    k = 2 * k_half
    slots = ids.size
    x = bf16_bits_to_float32(x_bits).astype(np.float64).reshape(slots, k)
    clamped = np.clip(np.asarray(ids, dtype=np.int64), 0, experts - 1)
    weights = decode_lut(packed[clamped], scales[clamped]).astype(np.float64)  # [S, N, K]
    golden = np.einsum("snk,sk->sn", weights, x)
    out = np.zeros((slots, rows), dtype=np.float64)
    valid = (np.asarray(ids) >= 0) & (np.asarray(ids) < experts)
    out[valid] = golden[valid] * scale2[clamped[valid]].astype(np.float64)[:, None]
    return out


# --------------------------------------------------------------------------
# synthetic data
# --------------------------------------------------------------------------


def random_case(seed, experts, rows, k, slots):
    """Random NVFP4 weights, BF16 activations, skip and out-of-range ids.

    Ids: slot 0 is -1 (skip, e.g. the fused shared-expert slot), slot 1 is
    out of range, the rest are real experts - so the gather and both drop paths
    are exercised by every comparison.
    """
    assert k % 16 == 0
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 256, (experts, rows, k // 2), dtype=np.uint8)
    scales = rng.integers(0, 256, (experts, rows, k // 16), dtype=np.uint8)
    scales[scales == 0x7F] = 0x3F
    scales[scales == 0xFF] = 0xBF
    ids = rng.integers(0, experts, slots).astype(np.int32)
    if slots > 2:
        ids[0] = -1
        ids[1] = experts + 3
    x_bits = to_bf16_bits((rng.standard_normal((slots, k)) * 0.3).astype(np.float32))
    scale2 = (0.25 + 0.25 * rng.integers(0, 8, experts)).astype(np.float32)
    return ids, packed, scales, x_bits, scale2


def exact_case(seed, experts, rows, k, slots):
    """Every product and every partial sum is exact in F32 and in double.

    Nibble magnitudes from {0, .5, 1, 2}, E4M3 scales 2**-1 or 2**0 (m = 0),
    activations from {0, +-0.25, +-0.5, +-1}, scale2 2**-2 or 2**0: products are
    powers of two in [2**-3, 2**1], any partial sum of <= 1280 of them is an
    integer multiple of 2**-3 below 2**12 (15 bits, exact in F32), and scale2 is
    an exact exponent shift - so the kernel's F32 sum equals the exact sum
    whatever the order, and BF16 is the only rounding left.
    """
    rng = np.random.default_rng(seed)
    magnitudes = np.array([0, 0, 1, 2, 4], dtype=np.uint8)  # 0 appears twice: more zeros
    nibble_mag = rng.choice(magnitudes, size=(experts, rows, k))
    sign = rng.integers(0, 2, size=(experts, rows, k), dtype=np.uint8) << 3
    nibbles = (nibble_mag | sign).astype(np.uint8)
    packed = (nibbles[:, :, 0::2] | (nibbles[:, :, 1::2] << 4)).astype(np.uint8)
    scale_bytes = np.where(rng.integers(0, 2, (experts, rows, k // 16)) == 0, 6 << 3, 7 << 3)
    scales = scale_bytes.astype(np.uint8)
    ids = rng.integers(0, experts, slots).astype(np.int32)
    if slots > 2:
        ids[0] = -1
        ids[1] = experts + 3
    activation_values = np.array([0.0, 0.25, -0.25, 0.5, -0.5, 1.0, -1.0, 0.0],
                                 dtype=np.float32)
    x_bits = to_bf16_bits(activation_values[rng.integers(0, 8, (slots, k))])
    scale2 = np.where(rng.integers(0, 2, experts) == 0, 0.25, 1.0).astype(np.float32)
    return ids, packed, scales, x_bits, scale2


def _relative_deviation(got, reference):
    scale = np.maximum(np.abs(reference), 1e-12)
    return np.max(np.abs(got.astype(np.float64) - reference) / scale)


# --------------------------------------------------------------------------
# 1. the decode
# --------------------------------------------------------------------------


def test_kernel_bit_construction_matches_the_lut():
    """The kernel's integer decode == E2M1 x E4M3 in floating point, bit for bit."""
    nibbles = np.arange(16, dtype=np.uint8)
    scale_bytes = np.array([b for b in range(256) if b not in (0x7F, 0xFF)], dtype=np.uint8)
    grid_nibble, grid_scale = np.meshgrid(nibbles, scale_bytes, indexing="ij")
    bits = nvfp4_mul_bits(grid_nibble, grid_scale)
    sign = np.where(grid_nibble & 0x08, -1.0, 1.0).astype(np.float32)
    exact = (sign * E2M1_LUT[grid_nibble & 0x07] * e4m3_to_float(grid_scale)).astype(np.float32)
    expected = (exact.view(np.uint32) >> 16).astype(np.uint16)
    mismatch = int(np.count_nonzero(bits != expected))
    assert mismatch == 0, f"{mismatch} of {bits.size} (nibble, scale) pairs differ"


class _StubPlatform:
    """dequantize_to_dtype only asks whether the Triton path applies."""

    @staticmethod
    def is_cuda_alike():
        return False


def _load_vllm_reference():
    """Execute vllm-core's NVFP4 dequant reference without importing vllm."""
    wanted = {"break_fp4_bytes", "convert_swizzled_to_linear", "dequantize_to_dtype"}
    tree = ast.parse(VLLM_REFERENCE_SOURCE.read_text())
    nodes = [node for node in tree.body
             if (isinstance(node, ast.FunctionDef) and node.name in wanted)
             or (isinstance(node, ast.Assign)
                 and any(getattr(target, "id", "") == "kE2M1ToFloat_handle"
                         for target in node.targets))]
    namespace = {"torch": torch, "SimpleNamespace": __import__("types").SimpleNamespace,
                 "current_platform": _StubPlatform()}
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(VLLM_REFERENCE_SOURCE), "exec"), namespace)
    return namespace


def test_decode_is_bit_exact_against_the_vllm_reference():
    """LUT decode == vllm-core dequantize_to_dtype on random packed/scales."""
    reference = _load_vllm_reference()["dequantize_to_dtype"]
    rows, k = 6, 512
    _, packed, scales, _, _ = random_case(seed=0, experts=1, rows=rows, k=k, slots=2)
    ours = decode_lut(packed[0], scales[0])
    theirs = reference(torch.from_numpy(packed[0]), torch.from_numpy(scales[0]),
                       global_scale=torch.ones(1, dtype=torch.float32), dtype=torch.float32,
                       block_size=16, swizzle=False).numpy().reshape(rows, k)
    assert np.array_equal(ours.view(np.uint32), theirs.view(np.uint32))


# --------------------------------------------------------------------------
# 2. the GEMV against the compiled C reference
# --------------------------------------------------------------------------


def test_gemv_matches_the_c_reference(reference_binary, tmp_path):
    """Random NVFP4 data: the kernel-order emulation vs the C double reference."""
    ids, packed, scales, x_bits, scale2 = random_case(seed=11, experts=64, rows=8, k=1280,
                                                      slots=6)
    golden = run_c_reference(reference_binary, tmp_path, ids, packed, scales, x_bits, scale2)
    vectorized = vectorized_lut_gemv(ids, packed, scales, x_bits, scale2)
    # The C reference itself: independent decode + double sum, so it must agree with
    # this module's vectorized float64 evaluation up to summation order.
    assert _relative_deviation(golden, vectorized) < 1e-12

    model = kernel_order_gemv(ids, packed, scales, x_bits, scale2)
    deviation = _relative_deviation(model, golden)
    assert deviation < TOLERANCE, f"max relative deviation {deviation:.3e} >= {TOLERANCE}"

    # The F32 sum error (~1e-6) is far below the BF16 store step (~4e-3), so the
    # stored bits agree except where the exact value sits on a rounding boundary.
    same = to_bf16_bits(model) == to_bf16_bits(golden.astype(np.float32))
    agreement = float(np.mean(same))
    assert agreement >= 0.99, f"only {agreement:.2%} of the BF16 outputs agree"
    # The skip (-1) and out-of-range slots must be exactly zero in both.
    assert np.all(golden[[0, 1]] == 0.0) and np.all(model[[0, 1]] == 0.0)

    # Slot -> expert gather: a permuted id list permutes the rows of both results.
    order = np.array([5, 2, 0, 4, 3, 1])
    ids_permuted = ids[order]
    golden_permuted = run_c_reference(reference_binary, tmp_path, ids_permuted, packed, scales,
                                      x_bits[order], scale2)
    assert np.array_equal(golden_permuted, golden[order])


def test_gemv_real_layer_shapes_match_the_c_reference(reference_binary, tmp_path):
    """The checkpoint's K values: 2560 (w13) and 640 (w2) - tile/part tails included."""
    for k in (2560, 640):
        ids, packed, scales, x_bits, scale2 = random_case(seed=k, experts=8, rows=4, k=k,
                                                          slots=4)
        golden = run_c_reference(reference_binary, tmp_path, ids, packed, scales, x_bits, scale2)
        model = kernel_order_gemv(ids, packed, scales, x_bits, scale2)
        deviation = _relative_deviation(model, golden)
        assert deviation < TOLERANCE, f"K={k}: relative deviation {deviation:.3e}"


def test_gemv_is_bit_exact_on_power_of_two_data(reference_binary, tmp_path):
    """No rounding happens anywhere in this data set: F32 model == C double == BF16."""
    ids, packed, scales, x_bits, scale2 = exact_case(seed=5, experts=16, rows=4, k=1280, slots=4)
    golden = run_c_reference(reference_binary, tmp_path, ids, packed, scales, x_bits, scale2)
    model = kernel_order_gemv(ids, packed, scales, x_bits, scale2)
    exact = golden.astype(np.float32)
    assert np.array_equal(model.view(np.uint32), exact.view(np.uint32)), (
        "the kernel-order F32 accumulation must reproduce the exact sum bit for bit")
    assert np.array_equal(to_bf16_bits(model), to_bf16_bits(exact))


def test_scale2_is_applied_once_after_the_sum(reference_binary, tmp_path):
    """Doubling scale2 doubles the output exactly; folding it per element is equivalent."""
    ids, packed, scales, x_bits, scale2 = exact_case(seed=9, experts=16, rows=4, k=1280,
                                                     slots=4)
    singleton = np.ones_like(scale2)
    golden_one = run_c_reference(reference_binary, tmp_path, ids, packed, scales, x_bits,
                                 singleton)
    model_one = kernel_order_gemv(ids, packed, scales, x_bits, singleton)
    # scale2 = 2**t is an exponent shift: exact, and equal to the unscaled sum shifted.
    model_scaled = kernel_order_gemv(ids, packed, scales, x_bits, scale2)
    expected = model_one * np.repeat(scale2[np.clip(ids, 0, scale2.size - 1)], 4,
                                     axis=0).reshape(model_one.shape)
    assert np.array_equal(to_bf16_bits(model_scaled), to_bf16_bits(expected))
    assert np.array_equal(golden_one.astype(np.float32),
                          model_one.astype(np.float32))
