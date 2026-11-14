# SPDX-License-Identifier: Apache-2.0
"""Compare CPU and real HPU bridge/refinement in eager and compiled modes."""

import runpy
from pathlib import Path

import habana_frameworks.torch.core  # noqa: F401
import torch


def main():
    reference = runpy.run_path(str(Path(__file__).resolve().parents[1] / "validation/test_dspine.py"))
    ops = reference["ops"]
    bridge = reference["make_bridge"]()
    device_bridge = reference["make_bridge"]().to("hpu")
    torch.manual_seed(5)
    lm = torch.randn(64, 32).bfloat16()
    norm = torch.randn(32).bfloat16()
    logits = torch.zeros(3, 248320).bfloat16()
    logits[1, [123456, 248000]] = 2
    logits[2, 91:107] = 1
    expected_ids = ops.stable_topk_ids(logits, 16)
    for compiled in (False, True):
        fn = (torch.compile(ops.stable_topk_ids, backend="hpu_backend", fullgraph=True, dynamic=False)
              if compiled else ops.stable_topk_ids)
        for _ in range(3):
            torch.testing.assert_close(fn(logits.to("hpu"), 16).cpu(), expected_ids, rtol=0, atol=0)
        print("PASS HPU topk boundary ties", "compiled", compiled, flush=True)
    for block in (8, 16):
        hidden = torch.randn(2 * block, 32).bfloat16()
        residual = torch.randn_like(hidden)
        ids = torch.zeros(2, block, dtype=torch.long)
        ids[:, 0] = torch.tensor([3, 17])
        expected = bridge(hidden, residual, ids, 4, block)
        for compiled in (False, True):
            fn = (torch.compile(device_bridge, backend="hpu_backend", fullgraph=True, dynamic=False)
                  if compiled else device_bridge)
            actual = fn(hidden.to("hpu"), residual.to("hpu"), ids.to("hpu"), 4, block)
            for result, target in zip(actual, expected):
                torch.testing.assert_close(result.cpu(), target, rtol=0.03, atol=0.02)
            print("PASS HPU bridge", block, "compiled", compiled, flush=True)
        steps = block - 1
        candidates = torch.rand(2, steps, 64).topk(4, dim=-1).indices.sort(-1).values
        raw = torch.randn(2, steps, 32).bfloat16()
        rms = torch.rand(2, steps, 1)
        query = torch.randn(2, steps, 8).bfloat16()
        anchors = torch.tensor([3, 17])
        args = (candidates, raw, rms, query, anchors, lm, norm)
        expected = ops.serial_refine(*args, bridge)
        device_args = tuple(t.to("hpu") for t in args)
        for compiled in (False, True):
            fn = (torch.compile(ops.serial_refine, backend="hpu_backend", fullgraph=True, dynamic=False)
                  if compiled else ops.serial_refine)
            actual = fn(*device_args, device_bridge).cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            print("PASS HPU serial", steps, "compiled", compiled, flush=True)


if __name__ == "__main__":
    main()
