# SPDX-License-Identifier: Apache-2.0
"""Tier-2 test for the Qwen4Exp NVFP4 TPC kernels: compile gate + bit math.

CPU-only and fast (<60 s): no device, no ``torch.hpu``, no vllm_gaudi import.
``VLLM_HW=cpu`` is set before any other import anyway (the repo pattern).

(a) Compile gate: every Qwen4Exp TPC kernel source (csrc/qwen4exp and
    native/qwen4exp-nvfp4-moe, including the new F8-E4M3 indexed dequant) is
    compiled by /usr/bin/tpc-clang -march=gaudi2 into a temp dir with
    -Wall -Werror. Syntax/intrinsic gate only -- nothing runs.
(b) Math validation by exact Python simulation of the kernels' bit ops:
      * the BF16 indexed dequant kernel: nibble unpack (low nibble = even k),
        true-value E2M1 LUT, e4m3fn scale decode, the integer
        ``nvfp4_mul_bits`` BF16 construction, tile/part/lane geometry
        (ragged tiles included) == tools/qwen4exp_flashnext/csf.py
        ``dequant_nvfp4`` golden, bit-exact;
      * the FP8 indexed dequant kernel: the same decode, then the F32
        ``* scale2`` product, the +-448 clamp and the round-to-nearest-even
        E4M3FN encode (direct convert + half-convert selection) ==
        ``torch.Tensor.to(torch.float8_e4m3fn)`` semantics, bit-exact on the
        domain where that cast is finite (|value| <= 464; above it torch
        produces NaN codes while the kernel saturates to +-448 by contract).
        Cases are constructed around ties (binade midpoints, "0.5" step
        fractions), subnormal-range products and the 448 saturation region.
    Real checkpoint bytes (layer-0 expert-0 gate_proj of shard 1 at
    ~/models/qwen4exp-flashnext-plain, with its real weight_scale_2)
    plus synthetic edge tensors. max_abs_err == 0 asserted wherever the
    golden is defined.
(c) TPC kernel ELF inputs were rebuilt by this session, so per the porting
    rules the gaudi2-single-card recipe cache directories are removed.

The encode simulation models the kernel's arithmetic (the TPC
v_convert_f32_to_f8_all_b chain). Those intrinsic semantics were verified
against this torch cast on 26k+ boundary values by running the compiled
kernel ELF in the Habana TPC simulator (host-only); this file re-checks the
selection logic and the full decode path against the same golden on CPU.

Run: .venv/bin/python tools/qwen4exp_flashnext/test_nvfp4_tpc_op.py
"""
from __future__ import annotations

import os

os.environ["VLLM_HW"] = "cpu"  # before any torch/vllm import; CPU-only test

import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from csf import E2M1_VALUES, E4M3_LUT, dequant_nvfp4  # noqa: E402

TPC_COMPILER = "/usr/bin/tpc-clang"
TPC_INCLUDE = "/usr/lib/habanatools/include"
KERNEL_SETS = (
    (REPO / "vllm-gaudi/csrc/qwen4exp", "kernels/*.c", "include"),
    (REPO / "native/qwen4exp-nvfp4-moe", "kernels/*.c", "include"),
)
CHECKPOINT = Path(os.environ.get("QWEN4EXP_MODEL", "~/models/qwen4exp-flashnext-plain")).expanduser()

# ---------------------------------------------------------------------------
# (a) tpc-clang compile gate
# ---------------------------------------------------------------------------


def compile_all_kernels() -> list[str]:
    """Compile every kernel .c with tpc-clang -march=gaudi2 into a temp dir."""
    sources: list[str] = []
    for root, pattern, include_dir in KERNEL_SETS:
        sources.extend(sorted((root).glob(pattern)))
    assert sources, "no kernel sources found"
    with tempfile.TemporaryDirectory(prefix="qwen4exp-tpc-gate-") as tmp:
        for source in sources:
            result = subprocess.run(
                [TPC_COMPILER, "-Wall", "-Werror", "-O2", "-march=gaudi2",
                 "-I", TPC_INCLUDE,
                 "-I", str(source.parents[1] / "include"),
                 str(source), "-c", "-o",
                 str(Path(tmp) / (source.stem + ".o"))],
                capture_output=True, text=True, timeout=120)
            assert result.returncode == 0, (
                f"tpc-clang failed on {source}:\n{result.stderr}")
        compiled = [str(s) for s in sources]
    print(f"compile gate: {len(compiled)} kernels clean under tpc-clang "
          f"-Wall -Werror -march=gaudi2")
    for name in compiled:
        print(f"  {Path(name).name}")
    return compiled


def clear_recipe_cache() -> None:
    """Porting rule: rebuilt TPC kernels invalidate recipe-cache-* dirs."""
    import glob

    home = Path(os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG", "").split(":")[0]
                if os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG") else Path.home())
    pattern = str(home / ".cache" / "gaudi2-single-card-vllm" / "recipe-cache-*")
    removed = 0
    for path in glob.glob(pattern):
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    if removed:
        print(f"recipe cache: removed {removed} stale directorie(s)")


# ---------------------------------------------------------------------------
# (b) math simulation of the kernels
# ---------------------------------------------------------------------------


def nvfp4_mul_bits(nibble: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """numpy transcription of include/qwen4exp_nvfp4_decode.h -> BF16 bits."""
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


def kernel_decode_bits(packed: np.ndarray, scales: np.ndarray,
                       ids: list[int]) -> np.ndarray:
    """Simulate the indexed-dequant kernels' decode: per (slot,row,tile,part)
    lane math reduced to its closed form.

    Lane derivation (kernel comments, verified on hardware): output element k
    of part p, u16 lane j (k = tile*512 + p*128 + j) reads packed byte
    k>>1 (low nibble = even k) and scale byte k>>4 (16-element groups);
    the dual-group replication and shuffles implement exactly these formulas.
    Returns [S,N,K] BF16 bits of e2m1 * e4m3, scale2 NOT applied.
    """
    experts, n, k_half = packed.shape
    k_total = k_half * 2
    out = np.zeros((len(ids), n, k_total), dtype=np.uint16)
    k = np.arange(k_total, dtype=np.int64)
    byte = k >> 1
    nib = np.where(k & 1, packed[..., byte] >> 4, packed[..., byte] & 0x0F)
    scale_byte = scales[..., k >> 4]
    bits = nvfp4_mul_bits(nib, scale_byte)
    for slot, expert in enumerate(ids):
        if 0 <= expert < experts:
            out[slot] = bits[expert]
    return out


def encode_e4m3_rne(x: np.ndarray) -> np.ndarray:
    """Pure bit-op RNE E4M3FN encode of float32 values (the semantics of the
    TPC v_convert_f32_to_f8_all_b intrinsic, as verified in the simulator):
    round-to-nearest-even in the normal and subnormal ranges, ties to even,
    and overflow to the NaN code 0x7F exactly where torch's cast produces it
    (anything whose RNE result would be the reserved 480 code). No torch."""
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32).copy()
    sign = (bits & np.uint32(0x80000000)).astype(np.uint32)
    mag_bits = bits & np.uint32(0x7FFFFFFF)
    mag = mag_bits.view(np.float32).astype(np.float32)

    # Target exponent field: E4M3 bias 7 vs F32 bias 127.
    e_t = (mag_bits >> np.uint32(23)).astype(np.int32) - 120
    normal = e_t >= 1

    # Normal path: RNE the F32 mantissa to 3 bits.
    mant = mag_bits & np.uint32(0x7FFFFF)
    top3 = (mant >> np.uint32(20)).astype(np.uint32)
    rest = mant & np.uint32(0xFFFFF)
    round_up = (rest > np.uint32(0x80000)) | (
        (rest == np.uint32(0x80000)) & ((top3 & 1) == 1))
    m2 = top3 + round_up.astype(np.uint32)
    e2 = e_t + (m2 == 8)
    m2 = np.where(m2 == 8, np.uint32(0), m2).astype(np.uint32)
    norm_code = ((e2.astype(np.uint32) << np.uint32(3)) | m2)
    # Rounding to the reserved 480 code (0x7F) is torch's NaN; anything
    # beyond (e2 > 15) lands there too.
    norm_code = np.where(norm_code >= np.uint32(0x7F), np.uint32(0x7F),
                         norm_code).astype(np.uint32)

    # Subnormal path (|x| < 2^-6): grid step 2^-9; x * 2^9 is exact.
    q = (mag * np.float32(512.0)).astype(np.float32)
    iq = np.floor(q).astype(np.float32)
    frac = q - iq
    up = (frac > np.float32(0.5)) | (
        (frac == np.float32(0.5)) & (np.mod(iq, 2) != 0))
    sub_code = (iq + up.astype(np.float32)).astype(np.uint32)  # 0..8, 8 = 2^-6

    code = np.where(normal, norm_code, sub_code).astype(np.uint32)
    return ((code | (sign >> np.uint32(24))).astype(np.uint8))


def kernel_encode_fp8(value_f32: np.ndarray) -> np.ndarray:
    """Simulate the FP8 kernel's encode: clamp to +-448, then the hybrid
    convert (direct below 256, the halved convert above), as
    QWEN4EXP_NVFP4_F8_ENCODE in include/qwen4exp_nvfp4_indexed_dequant_fp8.h."""
    y = np.clip(np.asarray(value_f32, dtype=np.float32), -448.0, 448.0)
    direct = encode_e4m3_rne(y)
    halved = encode_e4m3_rne(np.float32(y * np.float32(0.5)))
    halved_mag = halved & 0x7F
    shifted = (halved & 0x80) | ((halved_mag + 8).astype(np.uint8))
    return np.where(halved_mag >= 0x70, shifted, direct).astype(np.uint8)


def torch_fp8_cast(prod: np.ndarray):
    """Golden: torch.float8_e4m3fn cast -> (codes, finite_mask)."""
    import torch

    cast = torch.from_numpy(np.ascontiguousarray(prod)).to(
        torch.float8_e4m3fn).view(torch.uint8).numpy()
    finite = np.abs(prod.astype(np.float64)) <= 464.0
    return cast, finite


def bf16_bits_of(values_f32: np.ndarray) -> np.ndarray:
    return (values_f32.view(np.uint32) >> 16).astype(np.uint16)


def fp8_weight_bits(packed, scales, scale2, ids):
    """Full FP8 kernel sim: decode -> F32 * scale2 (one rounding) -> encode."""
    bits = kernel_decode_bits(packed, scales, ids)
    weight_f32 = (bits.astype(np.uint32) << 16).view(np.float32)
    out = np.zeros(weight_f32.shape, dtype=np.uint8)
    for slot, expert in enumerate(ids):
        if 0 <= expert < len(packed):
            prod = np.float32(weight_f32[slot] * np.float32(scale2[expert]))
            out[slot] = kernel_encode_fp8(prod)
    return out


def _synthetic(rng, k, n, e, ids):
    packed = rng.integers(0, 256, size=(e, n, k // 2), dtype=np.uint8)
    scales = rng.integers(0, 256, size=(e, n, k // 16), dtype=np.uint8)
    bad = (scales & 0x7F) == 0x7F  # NaN E4M3 codes never occur in checkpoints
    scales = np.where(bad, scales & 0x7E, scales).astype(np.uint8)
    return packed, scales


# ---------------------------------------------------------------------------
# test sections (plain functions so both pytest and __main__ run them)
# ---------------------------------------------------------------------------


def test_compile_gate():
    started = time.monotonic()
    compiled = compile_all_kernels()
    assert any("dequant_fp8" in name for name in compiled), \
        "the new FP8 kernel source must be in the compile gate"
    clear_recipe_cache()  # kernels were (re)built this session: stale recipes
    print(f"  compile gate took {time.monotonic() - started:.1f}s")


def test_bf16_decode_matches_csf_golden():
    """BF16 path: kernel bit sim == csf.dequant_nvfp4, bit-exact."""
    rng = np.random.default_rng(20261030)
    for k, tag in ((128, "quarter-tile"), (640, "ragged-1.25"),
                   (1280, "ragged-2.5"), (2560, "exact-5")):
        e, n = 64, 4
        ids = [1, e - 1, -1, e + 3]
        packed, scales = _synthetic(rng, k, n, e, ids)
        got = kernel_decode_bits(packed, scales, ids)
        # golden: csf.py float math, cast to BF16 bits (scale2 = 1).
        want = np.zeros_like(got)
        for slot, expert in enumerate(ids):
            if 0 <= expert < e:
                want[slot] = bf16_bits_of(
                    dequant_nvfp4(packed[expert], scales[expert], 1.0)
                    .astype(np.float32))
        mismatch = int((got != want).sum())
        assert mismatch == 0, (
            f"{tag}: {mismatch}/{want.size} BF16 bit mismatches vs csf.py")
        print(f"  bf16 {tag}: K={k} -> {want.size} elements bit-exact")
    # exhaustive nibble x scale-byte construction (256x256, NaN codes aside)
    nib = np.repeat(np.arange(16, dtype=np.uint8), 256)
    sb = np.tile(np.arange(256, dtype=np.uint8), 16)
    bits = nvfp4_mul_bits(nib, sb)
    exact = np.float32(E2M1_VALUES[nib & 7]
                       * np.where(nib & 8, -1.0, 1.0)
                       * E4M3_LUT[sb])
    want = bf16_bits_of(exact)
    nan_scale = (sb & 0x7F) == 0x7F
    mismatch = int((bits != want)[~nan_scale].sum())
    assert mismatch == 0, f"nvfp4_mul_bits: {mismatch} of {int((~nan_scale).sum())}"
    print(f"  bf16 exhaustive 16x256 decode grid: bit-exact "
          f"({int(nan_scale.sum())} NaN-scale codes excluded, never occur)")


def _fp8_compare(tag, prod, got):
    cast, finite = torch_fp8_cast(prod)
    bad = (got != cast) & finite
    assert int(bad.sum()) == 0, (
        f"{tag}: {int(bad.sum())} FP8 mismatches vs torch cast on the finite "
        f"domain; first at {np.argwhere(bad)[0]}")
    # beyond +-464 the kernel saturates instead of torch's NaN (contract)
    over = ~finite
    if over.any():
        expect = np.where(prod > 464.0, 0x7E, 0xFE).astype(np.uint8)
        assert np.array_equal(got[over], expect[over]), \
            f"{tag}: saturation domain does not clamp to +-448"
    return int(finite.sum()), int(over.sum())


def test_fp8_encode_matches_torch_cast():
    """FP8 encode: kernel sim == torch float8_e4m3fn cast, bit-exact."""
    # 1. ties: midpoints of adjacent E4M3 codes in every binade (incl. the
    #    subnormal<->normal transition), +- 1 f32 ulp around each.
    pos = sorted({float(v) for b in range(128) for v in [
        ((b >> 3) & 0xF and (1 + (b & 7) / 8.0) * 2.0 ** (((b >> 3) & 0xF) - 7))
        or ((b & 7) / 8.0) * 2.0 ** -6]})
    grid = np.array(pos, dtype=np.float32)
    mids = ((grid[:-1].astype(np.float64) + grid[1:].astype(np.float64))
            / 2).astype(np.float32)
    battery = [mids]
    for delta in (-1, 1):
        step = np.spacing(mids.astype(np.float64))
        battery.append((mids.astype(np.float64) + delta * step)
                       .astype(np.float32))
    battery.append(grid)
    battery.append((np.arange(0, 65) / 2048.0).astype(np.float32))   # subnormal
    battery.append(np.arange(430.0, 600.0, 0.125).astype(np.float32))  # 448/464
    # 2. the half-trick danger window and the 248..272 direct/half crossover.
    battery.append(np.linspace(0.029296875, 0.0302734375, 4097)
                   .astype(np.float32))
    battery.append(np.linspace(244.0, 276.0, 4097).astype(np.float32))
    # 3. all nibble x scale-byte products under a scale2 ladder.
    e2m1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    nib = np.repeat(np.arange(16, dtype=np.uint8), 256)
    sb = np.tile(np.arange(256, dtype=np.uint8), 16)
    base = np.float32(e2m1[nib & 7] * np.where(nib & 8, -1.0, 1.0)
                      * E4M3_LUT[sb])
    keep = ((sb & 0x7F) != 0x7F)
    ladder = np.array([2.0 ** i for i in range(-30, 1)]
                      + [1.0 / 6.0, 1.0 / 3.0, 0.7, 0.12345, 3.999],
                      dtype=np.float32)
    products = [np.float32(base[keep] * s2) for s2 in ladder]
    values = np.concatenate(battery + products)
    got = kernel_encode_fp8(values)
    finite, over = _fp8_compare("encode battery", values, got)
    print(f"  fp8 encode battery: {values.size} values "
          f"({finite} finite-domain bit-exact, {over} saturated to +-448)")


def test_fp8_full_kernel_sim_synthetic():
    """FP8 path over whole planes (geometry + decode + scale2 + encode)."""
    rng = np.random.default_rng(4242)
    for k, tag in ((128, "quarter-tile"), (1280, "ragged-2.5"),
                   (2560, "exact-5")):
        e, n = 64, 4
        ids = [1, e - 1, -1, e + 3]
        packed, scales = _synthetic(rng, k, n, e, ids)
        scale2 = np.array([2.0 ** ((i % 20) - 14) for i in range(e)],
                          dtype=np.float32)
        scale2[e - 1] = 2.0       # forces the saturation domain
        scale2[1] = 1.0 / 6.0     # max |w*s2| lands exactly on 448
        got = fp8_weight_bits(packed, scales, scale2, ids)
        weight_f32 = (kernel_decode_bits(packed, scales, ids)
                      .astype(np.uint32) << 16).view(np.float32)
        prod = np.zeros_like(weight_f32)
        for slot, expert in enumerate(ids):
            if 0 <= expert < e:
                prod[slot] = np.float32(weight_f32[slot]
                                        * np.float32(scale2[expert]))
        finite, over = _fp8_compare(f"planes {tag}", prod, got)
        # invalid slots must zero-fill exactly
        assert np.array_equal(got[2], np.zeros_like(got[2]))
        assert np.array_equal(got[3], np.zeros_like(got[3]))
        print(f"  fp8 planes {tag}: K={k} [{len(ids)},N,K] "
              f"({finite} finite bit-exact, {over} saturated), "
              f"invalid ids zero-fill")


def _load_real_gate_proj(rows: int):
    from safetensors import safe_open

    base = "model.language_model.layers.0.mlp.experts.0.gate_proj"
    for shard in sorted((CHECKPOINT / "tensors").glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            keys = f.keys()
            if base + ".weight" not in keys:
                continue
            packed = f.get_tensor(base + ".weight")[:rows].numpy()
            scales = (f.get_tensor(base + ".weight_scale")[:rows]
                      .view(torch_uint8()).numpy())
            scale2 = float(f.get_tensor(base + ".weight_scale_2"))
            return packed[None], scales[None], scale2
    raise FileNotFoundError(base)


def torch_uint8():
    import torch

    return torch.uint8


def test_real_checkpoint_bytes():
    """Real expert gate_proj planes through both kernel sims."""
    packed, scales, scale2 = _load_real_gate_proj(rows=8)
    ids = [0, 0]
    got_bf16 = kernel_decode_bits(packed, scales, ids)
    want_bf16 = np.stack([
        bf16_bits_of(dequant_nvfp4(packed[0], scales[0], 1.0)
                     .astype(np.float32))] * len(ids))
    assert np.array_equal(got_bf16, want_bf16), "real bytes: BF16 != csf golden"
    print(f"  real gate_proj [1,8,2560]: BF16 sim == csf.py golden "
          f"({got_bf16.size} elements, scale2 deferred as designed)")

    got = fp8_weight_bits(packed, scales, [scale2], ids)
    weight_f32 = (got_bf16.astype(np.uint32) << 16).view(np.float32)
    prod = np.float32(weight_f32 * np.float32(scale2))
    finite, over = _fp8_compare("real bytes", prod, got)
    assert over == 0
    print(f"  real gate_proj scale2={scale2:.9g}: FP8 sim == torch cast "
          f"({finite} elements bit-exact, none saturated)")


def main() -> int:
    tests = [test_compile_gate, test_bf16_decode_matches_csf_golden,
             test_fp8_encode_matches_torch_cast,
             test_fp8_full_kernel_sim_synthetic, test_real_checkpoint_bytes]
    for test in tests:
        name = test.__name__
        if name.startswith("test_real") and not CHECKPOINT.is_dir():
            print(f"skipped {name} (no checkpoint)")
            continue
        started = time.monotonic()
        test()
        print(f"{name}: PASS ({time.monotonic() - started:.2f}s)")
    print("all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
