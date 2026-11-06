# SPDX-License-Identifier: Apache-2.0
"""Decode MoE GEMV benchmark for the H8 indexed NVFP4 kernel on Gaudi2.

Times the fused GEMV (``custom_qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2``) at
the real Decode MoE shapes of the Qwen4Exp port, plus the full one-layer decode
step (gate + up + SwiGLU + down + ``index_add_``). The materialize path
(indexed dequant to BF16, then GEMM) is timed on the same inputs as the
baseline the GEMV replaces, and the full-layer output is compared against it so
the benchmark cannot silently time a wrong graph.

Per projection it reports ``kernel_ms`` (device time per call, the number to
compare across shapes), ``e2e_ms`` (one call plus a host drain - what one decode
step pays) and the ``dequant+bmm`` baseline the fused kernel replaces.
``GB/s`` divides the bytes the kernel must read per call (packed + scales +
activation + output) by ``kernel_ms``, so the small shapes can be judged against
Gaudi2's HBM roofline instead of in isolation.

Measurement caveat worth keeping: lazy mode drops any op whose result is never
read, so every loop here feeds a live accumulator and every sample is drained.
A discarded-result loop reported a flat 4.7 us for every shape, which is how the
first version of this file managed 12 TB/s on a 59 MB read.

Run both modes - eager measures device time per kernel, lazy measures the
graph-replay cost the engine actually pays:

  GC_KERNEL_PATH=$PWD/vllm-gaudi/csrc/qwen4exp/build/libqwen4exp_gaudi2_kernels.so \
  PT_HPU_LAZY_MODE=1 .venv/bin/python tools/qwen4exp_flashnext/bench_decode_moe.py \
      --json tools/qwen4exp_flashnext/bench_lazy.json
  GC_KERNEL_PATH=$PWD/vllm-gaudi/csrc/qwen4exp/build/libqwen4exp_gaudi2_kernels.so \
  PT_HPU_LAZY_MODE=0 .venv/bin/python tools/qwen4exp_flashnext/bench_decode_moe.py \
      --json tools/qwen4exp_flashnext/bench_eager.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KERNEL = REPO / "vllm-gaudi/csrc/qwen4exp/build/libqwen4exp_gaudi2_kernels.so"
EXT = next((REPO / "vllm-gaudi/csrc/qwen4exp/pytorch").glob(
    "hpu_qwen4exp_nvfp4_dequant_pt2*.so"), None)
# Must be set before the HPU runtime is imported.
os.environ["GC_KERNEL_PATH"] = str(KERNEL)
os.environ.setdefault("PT_HPU_LAZY_MODE", "1")

sys.path.insert(0, str(REPO / "vllm-gaudi"))

import numpy as np  # noqa: E402

# Decode shapes of the port. E is the checkpoint's routed-expert count; the
# kernels only ever touch the S selected slots.
EXPERTS = 512
GATE_UP = dict(n=640, k=2560)    # gate_proj / up_proj
DOWN = dict(n=2560, k=640)       # down_proj
SLOTS = (10, 40, 80)             # B=1/4/8 at top_k=10 (H10 decode ceiling)
TOKENS = ((1, 10), (4, 10), (8, 10))  # (B, top_k) for the full-layer step


def make_planes(rng: np.random.Generator, experts: int, n: int, k: int):
    """Packed E2M1 nibbles + E4M3 block scales, as the checkpoint stores them."""
    packed = rng.integers(0, 256, size=(experts, n, k // 2), dtype=np.uint8)
    scales = rng.integers(0, 256, size=(experts, n, k // 16), dtype=np.uint8)
    bad = (scales & 0x7F) == 0x7F  # NaN E4M3 codes never occur in the checkpoint
    scales = np.where(bad, scales & 0x7E, scales).astype(np.uint8)
    # Real per-expert weight_scale_2 magnitudes.
    scale2 = (rng.random(experts, dtype=np.float32) * 1e-4 + 1e-5).astype(np.float32)
    return packed, scales, scale2


def _materialized_ids(ids, device):
    """Production id-list normalization (see packed_moe.materialized_ids)."""
    from vllm_gaudi.models.qwen4exp.packed_moe import materialized_ids

    return materialized_ids(ids, device)


def _lazy() -> bool:
    return os.environ.get("PT_HPU_LAZY_MODE", "1") != "0"


def _consume(tensor):
    """Force the graph that produced ``tensor`` to execute, and drain the queue.

    ``.float().sum().item()`` is the cheapest way to make a value observable from
    the host. Lazy mode is free to drop an op whose output nothing reads, so a
    timing loop that throws its results away measures an empty graph. That is
    not hypothetical: an earlier version of this script reported the same 4.7 us
    for S=10 and S=64, i.e. 12 TB/s for a shape that must read 59 MB.
    """
    return tensor.float().sum().item()


def _chained(fn, count: int):
    """``count`` calls of ``fn`` whose results all feed one accumulator.

    Chaining keeps every call live (nothing can be eliminated) and costs one add
    of the ``[S, N]`` output per call, 0.5% of the weight bytes at the widest
    gate shape.

    ``fn`` receives the iteration index. Callers must make each call distinct
    (a different id list), otherwise the lazy IR common-subexpression-eliminates
    identical nodes: a 40-deep chain of the *same* op on the *same* ids then
    measures one kernel plus 40 adds, which is how a first version of this file
    reported a flat 8.5 us for S=10 and S=64 alike.
    """
    acc = fn(0)
    for i in range(1, count):
        acc = acc + fn(i)
    return acc


def _slope(fn, iters: int, samples: int) -> float:
    """Lazy: ms per call from the slope between ``iters`` and ``2 * iters``.

    Two chain lengths cancel the fixed per-sample cost (host drain, graph
    launch) exactly instead of dividing it by ``iters``, which is what made a
    first version of this benchmark attribute 150 us of drain to a 5 us kernel.
    """
    import time

    _consume(_chained(fn, iters))       # compile both chain shapes first
    _consume(_chained(fn, 2 * iters))

    def measure(count):
        times = []
        for _ in range(samples):
            t0 = time.perf_counter()
            _consume(_chained(fn, count))
            times.append(time.perf_counter() - t0)
        return statistics.median(times)

    return (measure(2 * iters) - measure(iters)) / iters * 1e3


def _batched(fn, iters: int, samples: int) -> float:
    """Eager: ``iters`` calls back to back, one synchronize, divide.

    Eager mode executes every call as it arrives, so nothing can be eliminated
    and no graph is built; this is the cleanest device-time measurement and the
    one the cross-shape table uses.
    """
    import time

    import torch

    _consume(fn(0))
    times = []
    for _ in range(samples):
        t0 = time.perf_counter()
        for i in range(iters):
            fn(i)
        torch.hpu.synchronize()
        times.append((time.perf_counter() - t0) / iters * 1e3)
    return statistics.median(times)


def _bench(fn, iters: int, samples: int) -> float:
    """ms per call of ``fn``, measured the way the current mode allows."""
    return _slope(fn, iters, samples) if _lazy() else _batched(fn, iters, samples)


def _single(fn, samples: int) -> float:
    """ms for one call plus a host drain - what one decode step costs."""
    import time

    _consume(fn(0))
    times = []
    for _ in range(samples):
        t0 = time.perf_counter()
        _consume(fn(0))
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3


def _step(fn, samples: int) -> float:
    """One decode step of a multi-op chain (gate + up + SwiGLU + down + add).

    Lazy builds the whole step as one graph and a chain of ten steps trips a
    graph limit ("Empty tensor optional ... Strided Params"), so lazy reports the
    single-step latency while eager reports the amortized step. Both include the
    Python dispatch of the step, which is what the engine pays per step.
    """
    return _single(fn, samples) if _lazy() else _batched(fn, 5, samples)


def bench_projection(name: str, op, packed, scales, scale2, n: int, k: int, rows_out: dict):
    import torch

    print(f"\n== {name}: E={packed.shape[0]} N={n} K={k} ==")
    print(f"{'S':>4} {'kernel_ms':>10} {'e2e_ms':>8} {'GB/s':>8} {'dequant+bmm_ms':>15} {'speedup':>8}")
    for slots in SLOTS:
        # 2 * iters + 2 distinct routing results so no two chained calls are the
        # same node (see _chained); they are tiny and share the same shape.
        pool = 2 * 40 + 2
        ids_pool = [torch.randint(0, packed.shape[0], (slots,), dtype=torch.int32, device="hpu")
                    for _ in range(pool)]
        # torch.rand has no BF16 kernel on this HPU build; build in F32 and cast.
        x = ((torch.rand(slots, k, device="hpu") - 0.5) * 0.04).bfloat16()
        fuse = lambda i: op(ids_pool[i], packed, scales, x, scale2)  # noqa: E731
        kernel_ms = _bench(fuse, iters=40, samples=7)
        e2e_ms = _single(fuse, samples=5)
        # Baseline: materialize [S,N,K] BF16 for the selected experts, then a
        # per-slot matrix-vector product (torch bmm on the decode batch).
        dequant = torch.ops.custom_op.custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2

        def materialize(i):
            weight = dequant(ids_pool[i], packed, scales, scale2)
            return torch.bmm(weight, x.unsqueeze(-1)).squeeze(-1)

        base_ms = _bench(materialize, iters=5, samples=3)
        # Bytes the fused kernel must read (weights are never re-materialized).
        nbytes = slots * (n * k / 2 + n * k / 16) + slots * k * 2 + slots * n * 2
        gbs = nbytes / (kernel_ms * 1e6)
        rows_out.append(dict(kind=name, n=n, k=k, s=slots, kernel_ms=kernel_ms,
                             e2e_ms=e2e_ms, gbps=gbs, materialize_ms=base_ms,
                             speedup=base_ms / kernel_ms))
        print(f"{slots:>4} {kernel_ms:>10.4f} {e2e_ms:>8.4f} {gbs:>8.1f} "
              f"{base_ms:>15.4f} {base_ms / kernel_ms:>7.2f}x")


def bench_full_layer(op, planes, rows_out: dict):
    """One decode MoE layer: gate + up + SwiGLU + down + index_add_."""
    import torch
    import torch.nn.functional as F

    dequant = torch.ops.custom_op.custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2
    gate_p, gate_s, gate_s2 = planes["gate"]
    up_p, up_s, up_s2 = planes["up"]
    down_p, down_s, down_s2 = planes["down"]
    hidden_size = GATE_UP["k"]
    intermediate = GATE_UP["n"]

    print("\n== full decode MoE layer (gate + up + SwiGLU + down + index_add_) ==")
    print(f"{'B':>2} {'T':>2} {'S':>4} {'step_ms':>9} {'floor_ms':>8} {'net_ms':>8} "
          f"{'materialize_ms':>15} {'speedup':>8} {'rel':>9}")
    for batch, top_k in TOKENS:
        slots = batch * top_k
        hidden = ((torch.rand(batch, hidden_size, device="hpu") - 0.5) * 0.04).bfloat16()
        topk_ids = torch.randint(0, EXPERTS, (batch, top_k), dtype=torch.int32, device="hpu")
        topk_w = torch.rand(batch, top_k, device="hpu").bfloat16()
        # The pt2 bridge cannot take a view of the routing result as the op's
        # id input in lazy mode; materialized_ids builds a real 1-D tensor
        # (see packed_moe.materialized_ids).
        flat_ids = _materialized_ids(topk_ids, torch.device("hpu"))
        token_of_slot = torch.arange(batch, device="hpu").repeat_interleave(top_k)

        def fused(_i=0):
            shared = hidden.index_select(0, token_of_slot)
            gate = op(flat_ids, gate_p, gate_s, shared, gate_s2)
            up = op(flat_ids, up_p, up_s, shared, up_s2)
            act = F.silu(gate) * up
            down = op(flat_ids, down_p, down_s, act.contiguous(), down_s2)
            acc = torch.zeros(batch, hidden_size, device="hpu", dtype=torch.bfloat16)
            acc.index_add_(0, token_of_slot, down * topk_w.reshape(-1, 1))
            return acc

        def materialize(_i=0):
            shared = hidden.index_select(0, token_of_slot)
            gate = torch.bmm(dequant(flat_ids, gate_p, gate_s, gate_s2),
                             shared.unsqueeze(-1)).squeeze(-1)
            up = torch.bmm(dequant(flat_ids, up_p, up_s, up_s2),
                           shared.unsqueeze(-1)).squeeze(-1)
            act = F.silu(gate) * up
            down = torch.bmm(dequant(flat_ids, down_p, down_s, down_s2),
                             act.bfloat16().unsqueeze(-1)).squeeze(-1)
            acc = torch.zeros(batch, hidden_size, device="hpu", dtype=torch.bfloat16)
            acc.index_add_(0, token_of_slot, down * topk_w.reshape(-1, 1))
            return acc

        got = fused(0)
        want = materialize(0)
        gap = (got.float() - want.float()).abs().max().item()
        scale = want.float().abs().max().clamp_min(1e-6).item()
        rel = gap / scale
        if not rel < 2e-2:
            raise SystemExit(f"B={batch}: fused layer diverged from the materialize path (rel {rel:.3e})")

        # A chain of ten layer steps trips a lazy-graph limit ("Empty tensor
        # optional ... Strided Params"), and a decode step drains the queue
        # anyway, so the layer is measured one step at a time.
        step_ms = _step(fused, samples=7)
        base_ms = _step(materialize, samples=5)
        # Floor: the same single-call measurement of a trivial op, i.e. what the
        # Python dispatch plus drain costs before any kernel runs. Subtract it to
        # read the layer's device time out of step_ms.
        floor_ms = _step(lambda _i=0: hidden.sum(), samples=7)
        rows_out.append(dict(kind="layer", batch=batch, top_k=top_k, s=slots,
                             step_ms=step_ms, floor_ms=floor_ms, net_ms=step_ms - floor_ms,
                             materialize_ms=base_ms,
                             speedup=base_ms / step_ms, rel=rel))
        print(f"{batch:>2} {batch:>2} {slots:>4} {step_ms:>9.4f} {floor_ms:>8.4f} "
              f"{step_ms - floor_ms:>8.4f} {base_ms:>15.4f} "
              f"{base_ms / step_ms:>7.2f}x {rel:>9.2e}")


def bench_grouped_ab(op, planes, rows_out: list, batches=None):
    """H10 A/B: old grouped_gemv vs new H10 grouped_gemv, plus 48 layers.

    Old (pre-H10): ``arange + repeat_interleave`` builds ``token_of_slot``,
    ``hidden.index_select`` replicates the token rows to ``[S, H]`` for
    gate/up, and the mix is ``zeros_like + index_add_`` over slots.

    New (H10, mirrors ``packed_moe.grouped_gemv``): gate/up receive
    ``hidden`` ``[T, H]`` directly (the GEMV kernel maps slot ``s`` to
    token row ``s // top_k``), down still receives the per-slot
    ``[S, I]`` activation, and the mix is a contiguous
    ``view(T, top_k, H).sum(1)`` (``sum(0, keepdim=True)`` for T=1).

    Both forms must be 100% bit-identical (checked as BF16 bit patterns,
    drained separately in lazy mode). ``flat_ids`` is materialized once
    and shared, as in production; only the helper-op difference above is
    timed. The 48-layer number is 48 sequential steps chained through a
    live accumulator and drained per step (a single 48-step lazy graph
    trips the graph limit noted in ``_step``); it is the wall clock a
    48-layer decode token pays for MoE under per-layer draining, not a
    single compiled 48-layer graph.
    """
    import torch
    import torch.nn.functional as F

    gate_p, gate_s, gate_s2 = planes["gate"]
    up_p, up_s, up_s2 = planes["up"]
    down_p, down_s, down_s2 = planes["down"]
    hidden_size = GATE_UP["k"]

    print("\n== grouped_gemv A/B: old (repeat_interleave/index_select + "
          "zeros_like/index_add_) vs new H10 (direct [T,K] + view/sum) ==")
    print(f"{'B':>2} {'S':>4} {'old_ms':>9} {'new_ms':>9} {'speedup':>8} "
          f"{'saved_%':>8} {'differing':>10} {'old48_ms':>10} {'new48_ms':>10} "
          f"{'saved48_ms':>11} {'saved48_%':>9}")
    for batch, top_k in TOKENS:
        if batches is not None and batch not in batches:
            continue
        slots = batch * top_k
        hidden = ((torch.rand(batch, hidden_size, device="hpu") - 0.5) * 0.04).bfloat16()
        topk_ids = torch.randint(0, EXPERTS, (batch, top_k), dtype=torch.int32, device="hpu")
        topk_w = torch.rand(batch, top_k, device="hpu").bfloat16()
        flat_ids = _materialized_ids(topk_ids, torch.device("hpu"))
        flat_w = topk_w.reshape(-1, 1)
        # Lazy: a step's intermediates must stay alive until its graph has
        # executed. Returning only the output drops gate/up/act before the
        # drain in the caller, and repeated steps then intermittently die
        # with "Empty tensor optional ... Strided Params" (probed: a
        # keep-alive + synchronize + mark_step loop survives 200 steps).
        # The list is cleared after each drain in time48 / per batch here.
        keep: list = []

        def fresh(tensor):
            # A produced tensor (index_select / silu*mul output) as a
            # custom-op activation input intermittently loses its strided
            # params across lazy subgraph boundaries; a fresh buffer is
            # the materialized form the bridge accepts (cf. ids).
            out = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
            out.copy_(tensor)
            return out

        def old_step(h, _batch=batch, _top_k=top_k):
            token_of_slot = torch.arange(_batch, device="hpu").repeat_interleave(_top_k)
            shared = fresh(h.index_select(0, token_of_slot))
            gate = op(flat_ids, gate_p, gate_s, shared, gate_s2)
            up = op(flat_ids, up_p, up_s, shared, up_s2)
            act = F.silu(gate) * up
            down = op(flat_ids, down_p, down_s, fresh(act), down_s2)
            acc = torch.zeros_like(h)
            acc.index_add_(0, token_of_slot, down * flat_w.to(dtype=down.dtype))
            keep.append((shared, gate, up, act, down, acc))
            return acc

        def new_step(h, _batch=batch, _top_k=top_k):
            gate = op(flat_ids, gate_p, gate_s, h, gate_s2)
            up = op(flat_ids, up_p, up_s, h, up_s2)
            act = F.silu(gate) * up
            down = op(flat_ids, down_p, down_s, fresh(act), down_s2)
            weighted = down * flat_w.to(dtype=down.dtype)
            out = (weighted.sum(dim=0, keepdim=True) if _batch == 1
                   else weighted.view(_batch, _top_k, hidden_size).sum(dim=1))
            keep.append((gate, up, act, down, weighted, out))
            return out

        # Bit-exactness first, drained separately (see _check_h10_direct
        # in test_nvfp4_gemv_hpu.py for the lazy independent-roots trap).
        old_cpu = old_step(hidden).cpu()
        new_cpu = new_step(hidden).cpu()
        differing = int((old_cpu.view(torch.uint16) != new_cpu.view(torch.uint16)).sum().item())
        if differing != 0 or not torch.equal(old_cpu, new_cpu):
            raise SystemExit(
                f"B={batch}: H10 grouped_gemv differs from old form "
                f"({differing}/{old_cpu.numel()} elements differ)")

        old_ms = _step(lambda _i=0: old_step(hidden), samples=7)
        new_ms = _step(lambda _i=0: new_step(hidden), samples=7)

        def time48(fn, samples=3):
            _consume(fn(hidden))  # compile / warm this step shape

            def one_run():
                # 48 sequential steps chained through a host scalar
                # (nothing can be DCE'd), drained per step. The GEMV input
                # stays the leaf ``hidden`` every layer: feeding the
                # produced sum/index_add_ output straight into the next
                # GEMV as activation -- or carrying a lazy accumulator
                # tensor across all 48 steps -- dies in lazy mode with
                # "Empty tensor optional ... Strided Params" (the tensor
                # accumulator after ~37 steps, probed in isolation), and
                # in the real model inter-layer hidden passes through
                # HC/norm ops anyway, never MoE-output -> GEMV directly.
                host_acc = 0.0
                t0 = time.perf_counter()
                for _layer in range(48):
                    host_acc += _consume(fn(hidden))  # per-layer drain
                    if _lazy():
                        torch.hpu.synchronize()
                        try:
                            import habana_frameworks.torch.core as _hcore

                            _hcore.mark_step()
                        except Exception:
                            pass
                    keep.clear()  # executed: intermediates may be freed
                if host_acc == float("inf"):  # keep the chain observable
                    print("  warning: 48-layer chain overflowed")
                return time.perf_counter() - t0

            return statistics.median(one_run() for _ in range(samples)) * 1e3

        old48_ms = time48(old_step)
        new48_ms = time48(new_step)
        keep.clear()
        saved48 = old48_ms - new48_ms
        rows_out.append(dict(
            kind="grouped-ab", batch=batch, top_k=top_k, s=slots,
            old_ms=old_ms, new_ms=new_ms, speedup=old_ms / new_ms,
            saved_pct=(old_ms - new_ms) / old_ms * 100.0, differing=differing,
            old48_ms=old48_ms, new48_ms=new48_ms, saved48_ms=saved48,
            saved48_pct=saved48 / old48_ms * 100.0 if old48_ms else 0.0))
        print(f"{batch:>2} {slots:>4} {old_ms:>9.4f} {new_ms:>9.4f} "
              f"{old_ms / new_ms:>7.2f}x {(old_ms - new_ms) / old_ms * 100:>7.1f}% "
              f"{differing:>10} {old48_ms:>10.2f} {new48_ms:>10.2f} "
              f"{saved48:>11.2f} {saved48 / old48_ms * 100 if old48_ms else 0:>8.1f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="write raw numbers here")
    parser.add_argument("--only", choices=("all", "proj", "layer", "ab"), default="all",
                        help="limit the run; separate processes keep the lazy graph state clean")
    parser.add_argument("--batches", type=str, default=None,
                        help="comma-separated B list for the grouped A/B (e.g. 1,4,8); "
                             "in lazy mode run one B per process")
    parser.add_argument("--kernel", type=Path, default=KERNEL,
                        help="libqwen4exp_gaudi2_kernels.so to load (A/B against another build)")
    args = parser.parse_args()

    if not args.kernel.is_file():
        raise SystemExit(f"kernel database missing: {args.kernel}")
    os.environ["GC_KERNEL_PATH"] = str(args.kernel.resolve())
    if EXT is None:
        raise SystemExit("build the pytorch extension first "
                         "(cd vllm-gaudi/csrc/qwen4exp/pytorch && python setup.py build_ext --inplace)")

    import torch
    import habana_frameworks.torch.core  # noqa: F401  (HPU init)

    torch.ops.load_library(str(EXT))
    op = torch.ops.custom_op.custom_qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2
    dequant = torch.ops.custom_op.custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2
    assert dequant is not None

    rng = np.random.default_rng(20261103)
    print(f"bench_decode_moe: kernel={os.environ['GC_KERNEL_PATH']}")
    print(f"  E={EXPERTS}  mode={'lazy' if _lazy() else 'eager'}")

    def to_hpu(packed, scales, scale2):
        return (torch.from_numpy(packed).to("hpu"),
                torch.from_numpy(scales).to("hpu"),
                torch.from_numpy(scale2).to("hpu"))

    planes = {}
    for kind, geom in (("gate", GATE_UP), ("up", GATE_UP), ("down", DOWN)):
        packed, scales, scale2 = make_planes(rng, EXPERTS, geom["n"], geom["k"])
        planes[kind] = to_hpu(packed, scales, scale2)
    print(f"  device planes: {sum(p.numel() * p.element_size() for p in
                               (t for plane in planes.values() for t in plane)) / 2**30:.2f} GiB")

    rows: list[dict] = []
    if args.only in ("all", "proj"):
        gate_p, gate_s, gate_s2 = planes["gate"]
        bench_projection("gate", op, gate_p, gate_s, gate_s2, GATE_UP["n"], GATE_UP["k"], rows)
        up_p, up_s, up_s2 = planes["up"]
        bench_projection("up", op, up_p, up_s, up_s2, GATE_UP["n"], GATE_UP["k"], rows)
        down_p, down_s, down_s2 = planes["down"]
        bench_projection("down", op, down_p, down_s, down_s2, DOWN["n"], DOWN["k"], rows)
    if args.only in ("all", "layer"):
        bench_full_layer(op, planes, rows)
    if args.only in ("all", "ab"):
        batches = (tuple(int(b) for b in args.batches.split(",") if b.strip())
                   if args.batches else None)
        bench_grouped_ab(op, planes, rows, batches=batches)

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
