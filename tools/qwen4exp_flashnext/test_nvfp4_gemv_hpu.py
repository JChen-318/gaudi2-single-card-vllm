# SPDX-License-Identifier: Apache-2.0
"""HPU check: fused NVFP4 GEMV against dequant-then-dot.

The kernel decodes E2M1 x E4M3 to the same BF16 bits as the indexed dequant
kernel, accumulates the dot in F32, then multiplies the per-expert scale2.
It must not requantize the weights. A large gate-shaped call is timed against
materializing BF16 and running a GEMM, which is the decode path it replaces.

Run:
  PT_HPU_LAZY_MODE=0 python tools/qwen4exp_flashnext/test_nvfp4_gemv_hpu.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KERNEL = REPO / "vllm-gaudi/csrc/qwen4exp/build/libqwen4exp_gaudi2_kernels.so"
EXT = next((REPO / "vllm-gaudi/csrc/qwen4exp/pytorch").glob(
    "hpu_qwen4exp_nvfp4_dequant_pt2*.so"))
# Set before the HPU runtime is imported; main() re-asserts it unconditionally.
os.environ["GC_KERNEL_PATH"] = str(KERNEL)
os.environ.setdefault("PT_HPU_LAZY_MODE", "0")

sys.path.insert(0, str(REPO / "vllm-gaudi"))

import torch  # noqa: E402

import habana_frameworks.torch.core  # noqa: F401, E402


def _reference(packed, scales, scale2, ids, activation):
    """F32 dot of the exact BF16 decode, then scale2. ``activation`` is [S, K]."""
    from vllm_gaudi.models.qwen4exp.weights import dequant_nvfp4

    rows = []
    for slot, expert in enumerate(ids.tolist()):
        weight = dequant_nvfp4(packed[int(expert)].cpu(), scales[int(expert)].cpu(), 1.0).float()
        rows.append(weight @ activation[slot].float().cpu())
    out = torch.stack(rows) * scale2[ids.long()].float().cpu().unsqueeze(-1)
    return out.to(torch.bfloat16)


def _check(name, packed, scales, scale2, ids, activation, op):
    got = op(ids, packed, scales, activation, scale2).cpu()
    want = _reference(packed.cpu(), scales.cpu(), scale2.cpu(), ids.cpu(), activation.cpu())
    gap = (got.float() - want.float()).abs()
    scale = want.float().abs().max().clamp_min(1e-6)
    rel = (gap.max() / scale).item()
    cos = torch.nn.functional.cosine_similarity(
        got.float().reshape(1, -1), want.float().reshape(1, -1)
    ).item()
    print(f"  {name}: rel {rel:.3e} cosine {cos:.8f} out {tuple(got.shape)}")
    if not (rel < 2e-2 and cos > 0.999):
        raise SystemExit(f"{name} diverged")
    return got


def _check_h10_direct(name, packed, scales, scale2, ids, x_token, top_k, op):
    """H10: gemv(x=[T, K]) must be 100% bit-identical to gemv(x_rep=[S, K]).

    ``x_rep = x.repeat_interleave(top_k, dim=0)`` is the pre-H10 gate/up
    input: slot ``s`` reads replicated row ``s``, which is token row
    ``s // top_k``. The direct [T, K] call must produce exactly the same
    BF16 bits (viewed as uint16, so -0/+0 and NaN payloads also count) --
    this is an index change only, so any differing element is a failure,
    not a tolerance case. Using distinct token rows also catches a kernel
    that always reads x row 0: slots of tokens 1..T-1 would then match
    the replicated call only if the replicated call were wrong too.
    """
    x_rep = x_token.repeat_interleave(top_k, dim=0)
    slots = ids.shape[0]
    if x_rep.shape[0] != slots:
        raise SystemExit(f"{name}: x_rep rows {x_rep.shape[0]} != S={slots}")

    def fresh(tensor):
        # Lazy mode: repeat_interleave lowers to expand+clone+view, and a
        # view as a TPC-op input dies at graph compile with
        # "Empty tensor optional ... Strided Params" (H8 trap 1, same as
        # ids in packed_moe.materialized_ids). Copy into a fresh buffer.
        out = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        out.copy_(tensor)
        return out

    # Drain between the two calls: in lazy mode two independent GEMV roots
    # with different x shapes in one graph do not keep both outputs intact
    # (direct-first is corrupted; rep-first and chained/consumed forms are
    # exact -- probed separately). Production grouped_gemv consumes gate/up
    # at once (silu*mul -> down -> sum), i.e. the chained form. Like every
    # other check in this file, each call here is drained before the next.
    #
    # H29: the order below is ``rep`` first *because* of that artifact. It
    # used to run ``direct`` first, and the T=1 case (x_rows=1, the glue's
    # special a=0 / range [0,0] spelling) then came back corrupted in lazy
    # mode -- all but 2 of 6400 elements differed, i.e. the whole-output
    # alias described above, not an index error (eager is 0/6400 and
    # 0/25600, and every other check in this file is exact). Both call
    # shapes are still compared bit-for-bit; only the emission order moved.
    replicated = op(ids, packed, scales, fresh(x_rep), scale2).cpu()
    direct = op(ids, packed, scales, fresh(x_token), scale2).cpu()
    # Compare BF16 bit patterns, not values: torch.equal on BF16 already
    # does this, but the uint16 view makes the differing count printable.
    direct_bits = direct.view(torch.uint16)
    replicated_bits = replicated.view(torch.uint16)
    differing = int((direct_bits != replicated_bits).sum().item())
    total = direct_bits.numel()
    print(f"  {name}: T={x_token.shape[0]} S={slots} top_k={top_k} "
          f"out {tuple(direct.shape)} differing {differing}/{total} "
          f"({'100% bit-identical' if differing == 0 else 'MISMATCH'})")
    if differing != 0 or not torch.equal(direct, replicated):
        raise SystemExit(
            f"{name}: direct [T, K] != repeat_interleave [S, K] "
            f"({differing}/{total} elements differ)")


def _sabotage(op, device):
    """scale2 must be read per expert, not dropped, not indexed by slot.

    Both experts carry identical weight bytes and both slots get the identical
    activation row, so the only degree of freedom left is scale2. Swapping the
    two scale2 entries must swap the two output rows. A kernel that folded a
    constant, used slot index instead of expert id, or read scale2 from the
    wrong expert cannot satisfy that.
    """
    gen = torch.Generator().manual_seed(29)
    rows, k, s = 64, 512, 1e-4
    packed = torch.randint(0, 256, (2, rows, k // 2), dtype=torch.uint8, generator=gen)
    scales = torch.randint(1, 120, (2, rows, k // 16), dtype=torch.uint8, generator=gen)
    packed[1] = packed[0]
    scales[1] = scales[0]
    ids = torch.tensor([0, 1], dtype=torch.int32, device=device)
    row = (torch.randn(1, k, generator=gen) * 0.02).bfloat16().to(device)
    act = row.expand(2, k).contiguous()

    def run(scale2):
        return op(ids, packed.to(device), scales.to(device), act,
                  torch.tensor(scale2, dtype=torch.float32, device=device)).cpu().float()

    normal = run([s, 3.0 * s])   # expert 1 is 3x expert 0
    swapped = run([3.0 * s, s])  # sabotage: the two entries exchanged
    zeroed = run([0.0, s])       # sabotage: expert 0 muted

    r_normal = (normal[1].mean() / normal[0].mean()).item()
    r_swapped = (swapped[1].mean() / swapped[0].mean()).item()
    muted = zeroed[0].abs().max().item()
    print(f"  scale2 sabotage: ratio normal {r_normal:.4f} (expect 3), "
          f"swapped {r_swapped:.4f} (expect 0.3333), muted row |max| {muted:.1e}")
    if not (2.9 < r_normal < 3.1):
        raise SystemExit(f"scale2 sabotage: unsabotaged ratio {r_normal:.4f} != 3")
    if not (0.32 < r_swapped < 0.34):
        raise SystemExit(
            f"scale2 sabotage not detected: swapped ratio {r_swapped:.4f} "
            "(scale2 is ignored, constant, or indexed by slot rather than expert)")
    if muted != 0.0:
        raise SystemExit(f"scale2=0 did not mute the row (|max| {muted:.3e})")


def main() -> int:
    # The shell may carry a stale GC_KERNEL_PATH pointing at the stock
    # libtpc_kernels.so (then the custom GUIDs are unknown and every compile
    # fails with synStatus 26). Always force the qwen4exp kernel database.
    os.environ["GC_KERNEL_PATH"] = str(KERNEL)
    assert Path(os.environ["GC_KERNEL_PATH"]).is_file(), "kernel database missing"

    torch.ops.load_library(str(EXT))
    op = torch.ops.custom_op.custom_qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2
    device = torch.device("hpu")
    gen = torch.Generator().manual_seed(11)

    def planes(experts, rows, k):
        packed = torch.randint(0, 256, (experts, rows, k // 2), dtype=torch.uint8, generator=gen)
        scales = torch.randint(1, 120, (experts, rows, k // 16), dtype=torch.uint8, generator=gen)
        scale2 = torch.rand(experts, generator=gen) * 1e-4 + 1e-5
        return packed, scales, scale2

    # Small exact-tile case, then the real gate shape (N=640, K=2560, S=10).
    for name, experts, rows, k, slots in (
        ("small", 4, 32, 128, 3),
        ("gate", 8, 640, 2560, 10),
        ("down", 8, 2560, 640, 10),
    ):
        packed, scales, scale2 = planes(experts, rows, k)
        ids = torch.randint(0, experts, (slots,), dtype=torch.int32, generator=gen)
        activation = (torch.randn(slots, k, generator=gen) * 0.02).bfloat16()
        _check(
            name,
            packed.to(device),
            scales.to(device),
            scale2.to(device),
            ids.to(device),
            activation.to(device),
            op,
        )

    # Safeguard: a kernel that silently drops scale2 would still pass the
    # comparisons above whenever scale2 is uniform, so sabotage it explicitly.
    _sabotage(op, device)

    # H10 explicit: direct per-token x=[T, K] (gate/up) must be 100%
    # bit-identical to the pre-H10 per-slot replicate
    # x_rep = x.repeat_interleave(10, dim=0) ([S, K]), at the real gate
    # shape N=640, K=2560. T=1 -> S=10 and T=4 -> S=40 (top_k=10).
    packed_h10, scales_h10, scale2_h10 = planes(8, 640, 2560)
    packed_h10_d = packed_h10.to(device)
    scales_h10_d = scales_h10.to(device)
    scale2_h10_d = scale2_h10.to(device)
    for tokens, slots in ((1, 10), (4, 40)):
        ids_h10 = torch.randint(0, 8, (slots,), dtype=torch.int32, generator=gen)
        x_token = (torch.randn(tokens, 2560, generator=gen) * 0.02).bfloat16()
        _check_h10_direct(
            f"h10-direct-T{tokens}",
            packed_h10_d,
            scales_h10_d,
            scale2_h10_d,
            ids_h10.to(device),
            x_token.to(device),
            10,
            op,
        )
    del packed_h10_d, scales_h10_d, scale2_h10_d

    # Decode-shaped gate: fused GEMV versus materialize-then-GEMM.
    packed, scales, scale2 = planes(512, 640, 2560)
    ids = torch.randint(0, 512, (10,), dtype=torch.int32, generator=gen)
    activation = (torch.randn(10, 2560, generator=gen) * 0.02).bfloat16()
    packed_d, scales_d = packed.to(device), scales.to(device)
    scale2_d, ids_d, act_d = scale2.to(device), ids.to(device), activation.to(device)
    dequant = torch.ops.custom_op.custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2
    for _ in range(2):
        op(ids_d, packed_d, scales_d, act_d, scale2_d)
        weight = dequant(ids_d, packed_d, scales_d, scale2_d)
        torch.nn.functional.linear(act_d, weight.reshape(10 * 640, 2560))
    torch.hpu.synchronize()
    iters = 8
    # Lazy mode CSEs identical calls and drops unused outputs (H8 traps 2/3),
    # so every call gets a distinct id list and the chain ends in a host read.
    # The slope between a 8- and 16-call chain cancels the per-step graph cost
    # instead of attributing it to one kernel.
    bench_ids = [
        torch.randint(0, 512, (10,), dtype=torch.int32, generator=gen).to(device)
        for _ in range(2 * iters)
    ]

    def gemv_call(i):
        return op(bench_ids[i], packed_d, scales_d, act_d, scale2_d)

    def dense_call(i):
        weight = dequant(bench_ids[i], packed_d, scales_d, scale2_d)
        return torch.nn.functional.linear(act_d, weight.reshape(10 * 640, 2560))

    def chain(fn, count):
        acc = None
        for i in range(count):
            out = fn(i)
            acc = out if acc is None else acc + out
        return acc.float().sum().item()  # drain: the step cannot be eliminated

    def slope(fn):
        chain(fn, iters)  # compile both chain shapes first
        chain(fn, 2 * iters)

        def measure(count):
            samples = []
            for _ in range(3):
                t0 = time.perf_counter()
                chain(fn, count)
                samples.append(time.perf_counter() - t0)
            return sorted(samples)[1]

        return (measure(2 * iters) - measure(iters)) / iters * 1e3

    gemv_ms = slope(gemv_call)
    dense_ms = slope(dense_call)
    print(f"  gate S=10: gemv {gemv_ms:.2f} ms, dequant+gemm {dense_ms:.2f} ms")

    # H9(a): 80 slots is the decode ceiling (B=8, top_k=10). The host glue used
    # to reject S>64, so the widened DECODE_MAX_SLOTS could not run; this is
    # the correctness check for one 80-slot launch.
    ids80 = torch.randint(0, 512, (80,), dtype=torch.int32, generator=gen)
    act80 = (torch.randn(80, 2560, generator=gen) * 0.02).bfloat16()
    _check("gate-s80", packed_d, scales_d, scale2_d, ids80.to(device), act80.to(device), op)
    print("nvfp4 gemv matches the exact BF16 decode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
