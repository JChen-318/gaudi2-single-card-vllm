# SPDX-License-Identifier: Apache-2.0
"""On-HPU verification of the NVFP4 indexed-dequant TPC kernel (bring-up check).

Runs the compiled custom op against a numpy fp32 golden built from the same
E2M1/E4M3 LUTs used by the conversion pipeline. Bit-exact comparison of the
BF16 output; NaN scale codes (0x7F / 0xFF) are excluded from synthetic inputs
(they never occur in the checkpoint and the kernel treats them as NaN too).

Covered risks:
  * scale shuffle indices 16..31 (mov_dual_group replication) — full-random
    per-group scales so any mis-replication shows up as a value mismatch
    (K=2560 exercises all five exact tiles of a gate_proj matrix),
  * ragged trailing tiles — K=1280 (2.5 tiles) and K=640 (1.25 tiles),
  * overread-dominated small tile — K=128 (0.25 tiles),
  * -1 / out-of-range expert ids must zero-fill,
  * real checkpoint bytes (layer 0 gate_proj and down_proj experts).

Run with the kernel database and eager mode:
  GC_KERNEL_PATH=vllm-gaudi/csrc/qwen4exp/build/libqwen4exp_gaudi2_kernels.so \
  PT_HPU_LAZY_MODE=0 python tools/qwen4exp_flashnext/test_nvfp4_tpc_hpu.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO / "vllm-gaudi"))

from csf import E2M1_VALUES, E4M3_LUT  # noqa: E402

KERNEL = REPO / "vllm-gaudi/csrc/qwen4exp/build/libqwen4exp_gaudi2_kernels.so"
EXTENSION = (REPO / "vllm-gaudi/csrc/qwen4exp/pytorch").glob("hpu_qwen4exp_nvfp4_dequant_pt2*.so")
CHECKPOINT = Path(os.environ.get("QWEN4EXP_MODEL", "~/models/qwen4exp-flashnext-plain")).expanduser()


def golden_bits(packed: np.ndarray, scales: np.ndarray, ids: list[int]) -> np.ndarray:
    """fp32 reference: e2m1 * e4m3 per 16-group, cast to BF16 top bits.

    Torch layout matches the op: packed [E,N,K/2], scales [E,N,K/16] ->
    output [S,N,K].
    """
    e, n, k_half = packed.shape
    k = k_half * 2
    nib = np.empty((e, n, k_half, 2), dtype=np.uint8)
    nib[..., 0] = packed & 0x0F
    nib[..., 1] = packed >> 4
    codes = nib.reshape(e, n, k)
    groups = np.repeat(scales, 16, axis=2)  # [E,N,K/16] -> [E,N,K]
    vals = (E2M1_VALUES[codes & 7]
            * np.where(codes & 8, -1.0, 1.0)
            * E4M3_LUT[groups])
    out = np.zeros((len(ids), n, k), dtype=np.uint16)
    for slot, expert in enumerate(ids):
        if 0 <= expert < e:
            out[slot] = (vals[expert].astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    return out


def synth_case(rng, k, n, e, ids, tag):
    import torch

    packed = rng.integers(0, 256, size=(e, n, k // 2), dtype=np.uint8)
    scales = rng.integers(0, 256, size=(e, n, k // 16), dtype=np.uint8)
    bad = (scales & 0x7F) == 0x7F  # NaN E4M3 codes never occur in the checkpoint
    scales = np.where(bad, scales & 0x7E, scales).astype(np.uint8)
    return tag, packed, scales, ids


def fp8_golden(packed: np.ndarray, scales: np.ndarray, scale2: np.ndarray, ids: list[int]) -> np.ndarray:
    """BF16 decode, one F32 scale2 multiply, then torch's e4m3fn cast.

    Above |448| the kernel saturates to ±448 (0x7E / 0xFE) instead of emitting
    torch's NaN. Out-of-range expert ids stay zero, matching the BF16 kernel.
    """
    import torch

    bits = golden_bits(packed, scales, ids)
    weight = (bits.astype(np.uint32) << 16).view(np.float32)
    out = np.zeros(weight.shape, dtype=np.uint8)
    for slot, expert in enumerate(ids):
        if not 0 <= expert < packed.shape[0]:
            continue
        prod = np.float32(weight[slot] * np.float32(scale2[expert]))
        cast = torch.from_numpy(np.ascontiguousarray(prod)).to(
            torch.float8_e4m3fn).view(torch.uint8).numpy()
        finite = np.abs(prod.astype(np.float64)) <= 464.0
        saturated = np.where(prod > 464.0, np.uint8(0x7E), np.uint8(0xFE))
        out[slot] = np.where(finite, cast, saturated).astype(np.uint8)
    return out


def run_fp8_case(op, torch, tag, packed, scales, scale2, ids):
    ids_t = torch.tensor(ids, dtype=torch.int32, device="hpu")
    packed_t = torch.from_numpy(np.ascontiguousarray(packed)).to("hpu")
    scales_t = torch.from_numpy(np.ascontiguousarray(scales)).to("hpu")
    scale2_t = torch.from_numpy(np.ascontiguousarray(scale2)).to("hpu")
    out = op(ids_t, packed_t, scales_t, scale2_t)
    torch.hpu.synchronize()
    got = out.cpu().view(torch.uint8).numpy()
    want = fp8_golden(packed, scales, scale2, ids)
    mismatch = got != want
    count = int(mismatch.sum())
    assert count == 0, (f"{tag}: {count}/{want.size} FP8 mismatches; first at "
                        f"{np.argwhere(mismatch)[0]}: got 0x{got[mismatch][0]:02x} "
                        f"want 0x{want[mismatch][0]:02x}")
    print(f"  {tag}: K={packed.shape[2]*2} N={packed.shape[1]} E={packed.shape[0]} "
          f"S={list(ids)} -> {want.size} FP8 elements bit-exact OK")


def run_case(op, torch, tag, packed, scales, ids):
    ids_t = torch.tensor(ids, dtype=torch.int32, device="hpu")
    packed_t = torch.from_numpy(np.ascontiguousarray(packed)).to("hpu")
    scales_t = torch.from_numpy(np.ascontiguousarray(scales)).to("hpu")
    # scale2=1 is exact in BF16, so the folded kernel matches the unscaled golden.
    scale2_t = torch.ones(packed.shape[0], dtype=torch.float32, device="hpu")
    out = op(ids_t, packed_t, scales_t, scale2_t)
    torch.hpu.synchronize()
    got = out.cpu().view(torch.uint16).numpy()
    # The kernel folds scale2=1 with the same BF16 mul as Python, which
    # flushes signed zero. Compare against that fold, not the raw decode.
    want_t = torch.from_numpy(golden_bits(packed, scales, ids).copy()).view(torch.bfloat16)
    want_t = want_t * torch.ones(want_t.shape[0], 1, 1, dtype=torch.bfloat16)
    want = want_t.view(torch.uint16).numpy()
    # CPU BF16 mul keeps -0. The production fold runs on Gaudi and flushes it.
    want = np.where(want == np.uint16(0x8000), np.uint16(0), want)
    mismatch = (got != want)
    count = int(mismatch.sum())
    total = want.size
    assert count == 0, (f"{tag}: {count}/{total} mismatched elements; first at "
                        f"{np.argwhere(mismatch)[0]}: got bits 0x{got[mismatch][0]:04x} "
                        f"want 0x{want[mismatch][0]:04x}")
    print(f"  {tag}: K={packed.shape[2]*2} N={packed.shape[1]} E={packed.shape[0]} "
          f"S={list(ids)} -> {total} elements bit-exact OK")


def real_expert_planes(layer: int, kind: str, count: int):
    """Load [E,N,K/2] packed + [E,N,K/16] scale planes from the checkpoint.

    The checkpoint already stores each expert as [N, K/2] / [N, K/16] rows,
    so stacking experts on axis 0 gives the op's torch layout directly.
    """
    import torch
    from safetensors import safe_open

    packed_list, scale_list = [], []
    for expert in range(count):
        base = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{kind}.weight"
        shard = next((s for s in sorted((CHECKPOINT / "tensors").glob("*.safetensors"))
                      if _shard_has(s, base)), None)
        if shard is None:
            raise FileNotFoundError(base)
        with safe_open(shard, framework="pt", device="cpu") as f:
            weight = f.get_tensor(base)             # [N, K/2] U8
            plane = f.get_tensor(base + "_scale")   # [N, K/16] float8_e4m3fn
        packed_list.append(weight.numpy())                        # [N, K/2]
        scale_list.append(plane.view(torch.uint8).numpy())        # [N, K/16]
    packed = np.ascontiguousarray(np.stack(packed_list, axis=0))   # [E, N, K/2]
    scales = np.ascontiguousarray(np.stack(scale_list, axis=0))    # [E, N, K/16]
    return packed, scales


def _shard_has(shard: Path, name: str) -> bool:
    from safetensors import safe_open

    with safe_open(shard, framework="pt", device="cpu") as f:
        return name in f.keys()


def real_expert_planes_many(layer: int, kind: str, count: int):
    """Load every expert plane with one sweep over the shards.

    ``real_expert_planes`` opens each shard for every expert, which is 512
    shard scans for the H9 shape. Here the shard set and its key set are
    visited once per file.
    """
    import torch
    from safetensors import safe_open

    prefix = f"model.language_model.layers.{layer}.mlp.experts."
    weight = {}
    scale = {}
    for shard in sorted((CHECKPOINT / "tensors").glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for expert in range(count):
                name = f"{prefix}{expert}.{kind}.weight"
                if name not in keys:
                    continue
                weight[expert] = f.get_tensor(name).numpy()              # [N, K/2]
                scale[expert] = f.get_tensor(name + "_scale").view(torch.uint8).numpy()
        if len(weight) == count and len(scale) == count:
            break
    missing = sorted(set(range(count)) - set(weight))
    if missing:
        raise FileNotFoundError(f"{kind}: {len(missing)} experts missing, first {missing[:8]}")
    packed = np.ascontiguousarray(np.stack([weight[e] for e in range(count)], axis=0))
    scales = np.ascontiguousarray(np.stack([scale[e] for e in range(count)], axis=0))
    return packed, scales


def run_wide_case(op, torch, tag, packed, scales, ids):
    """H9(a): one 512-id launch must equal the old 8x64 launches + torch.cat.

    ``ids`` is descending, so an implementation that sorts or reorders the ids
    differs from the chunked path. A descending id list also crosses every
    64-id chunk boundary with a different expert, which catches a kernel that
    reuses the previous chunk's expert index.
    """
    experts = packed.shape[0]
    assert len(ids) == experts == 512, f"wide case expects 512 ids, got {len(ids)}"
    packed_t = torch.from_numpy(packed).to("hpu")
    scales_t = torch.from_numpy(scales).to("hpu")
    scale2_t = torch.ones(experts, dtype=torch.float32, device="hpu")
    one = op(torch.tensor(ids, dtype=torch.int32, device="hpu"), packed_t, scales_t, scale2_t)
    pieces = [
        # Fresh buffers, not slices: a view is what the bridge rejects in lazy
        # mode (H8 trap 1), and this mirrors the retired chunk loop exactly.
        op(torch.tensor(ids[start:start + 64], dtype=torch.int32, device="hpu"),
           packed_t, scales_t, scale2_t)
        for start in range(0, experts, 64)
    ]
    old = torch.cat(pieces, dim=0)
    torch.hpu.synchronize()
    assert torch.equal(one, old), f"{tag}: single launch != 8x64 + cat"

    # Independent CPU golden on slots that straddle chunk boundaries.
    slots = [0, 63, 64, 127, 128, 255, 256, 383, 448, 511]
    got = one.cpu().view(torch.uint16).numpy()
    for slot in slots:
        expert = ids[slot]
        bits = golden_bits(packed[expert:expert + 1], scales[expert:expert + 1], [0])
        want_t = torch.from_numpy(bits.copy()).view(torch.bfloat16)
        want_t = want_t * torch.ones(want_t.shape[0], 1, 1, dtype=torch.bfloat16)
        want = want_t.view(torch.uint16).numpy()
        want = np.where(want == np.uint16(0x8000), np.uint16(0), want)
        if not np.array_equal(got[slot], want[0]):
            bad = int((got[slot] != want[0]).sum())
            raise AssertionError(f"{tag}: slot {slot} (expert {expert}) {bad} bits differ from golden")
    print(f"  {tag}: S=512 one launch == 8x64+cat bit-exact; "
          f"{len(slots)} boundary slots match CPU golden")

    # Negative control: scale2 is per expert and must be read at the wide S
    # too. Doubling the last chunk's experts has to move only their rows.
    tampered = scale2_t.clone()
    tampered[torch.tensor(ids[448:], dtype=torch.int64, device="hpu")] = 2.0
    moved = op(torch.tensor(ids, dtype=torch.int32, device="hpu"), packed_t, scales_t, tampered)
    torch.hpu.synchronize()
    last = moved[448:]
    first = moved[:448]
    base_last = one[448:]
    base_first = one[:448]
    if torch.equal(last, base_last):
        raise AssertionError(f"{tag}: last-64 scale2 sabotage changed nothing")
    if not torch.equal(first, base_first):
        raise AssertionError(f"{tag}: scale2 sabotage leaked into the first 448 slots")
    print(f"  {tag}: last-64 scale2 sabotage detected, first 448 rows unchanged")


def main() -> int:
    # The shell may carry a stale GC_KERNEL_PATH pointing at the stock
    # libtpc_kernels.so (then custom GUIDs are unknown and every compile fails
    # with synStatus 26). Always force the qwen4exp kernel database.
    os.environ["GC_KERNEL_PATH"] = str(KERNEL)
    assert Path(os.environ["GC_KERNEL_PATH"]).is_file(), "kernel database missing"

    import torch
    import habana_frameworks.torch.core  # noqa: F401  (HPU init)

    extension = next(EXTENSION, None)
    assert extension is not None, "build the pytorch extension first (setup.py build_ext --inplace)"
    torch.ops.load_library(str(extension))
    op = torch.ops.custom_op.custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2
    op_fp8 = torch.ops.custom_op.custom_qwen4exp_nvfp4_indexed_dequant_fp8_gaudi2

    rng = np.random.default_rng(20261025)
    print(f"test_nvfp4_tpc_hpu: kernel={os.environ['GC_KERNEL_PATH']}")

    # Synthetic: shuffle 16..31 (K=2560 = 5 exact tiles), ragged 2.5/1.25 tiles,
    # overread 0.25 tile, zero-fill sentinels.
    run_case(op, torch, *synth_case(rng, 2560, 192, 3, [0, 2, 1, -1], "shuffle-exact-tiles"))
    run_case(op, torch, *synth_case(rng, 1280, 160, 2, [1, 0], "ragged-2.5-tiles"))
    run_case(op, torch, *synth_case(rng, 640, 128, 2, [0, 3], "ragged-1.25-tiles"))
    run_case(op, torch, *synth_case(rng, 128, 96, 2, [1, 99], "overread-0.25-tile"))
    # H27: the dequant kernel owns four output rows per member, so a row count
    # that is not a multiple of four exercises the clamped tail group (its loads
    # repeat the last row and its duplicate stores must be identical bits).
    run_case(op, torch, *synth_case(rng, 2560, 643, 2, [1, 0], "ragged-rows-643"))
    run_case(op, torch, *synth_case(rng, 1280, 641, 2, [0, 1], "ragged-rows-641"))
    run_case(op, torch, *synth_case(rng, 640, 2, 2, [1], "ragged-rows-2"))
    run_case(op, torch, *synth_case(rng, 256, 1, 1, [0], "single-row"))

    # Real checkpoint bytes: gate_proj (K=2560 exact) and down_proj (K=640 ragged).
    packed, scales = real_expert_planes(0, "gate_proj", 4)
    run_case(op, torch, "real-gate-proj", packed, scales, [0, 3, 2, 1, -1])
    packed, scales = real_expert_planes(0, "down_proj", 3)
    run_case(op, torch, "real-down-proj", packed, scales, [2, 0, 1])

    # H9(a): the 64-id call contract is gone. All E=512 experts in one launch
    # per projection, bit-identical to the retired 8x64 chunks + torch.cat.
    wide_ids = list(range(511, -1, -1))
    packed, scales = real_expert_planes_many(0, "gate_proj", 512)
    run_wide_case(op, torch, "real-gate-proj-512", packed, scales, wide_ids)
    del packed, scales
    packed, scales = real_expert_planes_many(0, "down_proj", 512)
    run_wide_case(op, torch, "real-down-proj-512", packed, scales, wide_ids)
    del packed, scales

    # FP8 path folds scale2 before the e4m3fn encode. A small scale2 stays in
    # the finite domain; 8.0 forces the ±448 saturation contract on large
    # E2M1×E4M3 products. K=128 satisfies K%128==0.
    fp8_packed, fp8_scales, _ = synth_case(rng, 128, 32, 2, [0, 1, -1], "fp8")[1:]
    run_fp8_case(op_fp8, torch, "fp8-finite", fp8_packed, fp8_scales,
                 np.array([0.05, 0.05], dtype=np.float32), [0, 1, -1])
    run_fp8_case(op_fp8, torch, "fp8-saturate", fp8_packed, fp8_scales,
                 np.array([8.0, 8.0], dtype=np.float32), [1, 0])

    print("all HPU kernel tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
