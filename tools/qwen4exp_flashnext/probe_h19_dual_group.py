# SPDX-License-Identifier: Apache-2.0
"""H19 Step 3 diagnostic: which dual group does v_u8_shuffle_b read?

The production kernels always replicate the source with
``v_u8_mov_dual_group_all_b`` before shuffling, so every existing use feeds the
shuffle four *identical* 64-byte groups and cannot observe how the instruction
addresses a source whose four dual groups differ. The Question-A pre-swizzle
depends on exactly that, so this probe runs the indexed dequant kernel with the
mov_dg removed: the shuffle then sees the raw 256-byte tile (four different
groups) and the decoded BF16 values reveal, lane by lane, which source byte
each output byte came from.

Model M1 "same dual group": out byte b of group g reads source byte
    g*64 + ((b>>2) & 63)      -> group g reads its own 64-byte window
Model M2 "always group 0":   out byte b reads source byte ((b>>2) & 63)
    -> all four output groups repeat the first 64 source bytes
Model M3 "flat 128-byte":    out byte b reads source byte (b>>2) (0..63 used
    twice) -- same numbers as M2 for the low half, so M2/M3 are reported
    together.

Data: every packed byte = (j*5+3)&0xFF so the 256 byte values are distinct;
scales are all E4M3 1.0 (0x38) and scale2 is 1.0, so the stored BF16 is
e2m1(nibble) and inverts to the nibble (codes 0 and 8 both flush to +0).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
KERNEL = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/h19/b_dg/build/libqwen4exp_gaudi2_kernels.so")
EXT = next((REPO / "vllm-gaudi/csrc/qwen4exp/pytorch").glob(
    "hpu_qwen4exp_nvfp4_dequant_pt2*.so"))
os.environ["GC_KERNEL_PATH"] = str(KERNEL)
os.environ.setdefault("PT_HPU_LAZY_MODE", "0")

import torch  # noqa: E402
import habana_frameworks.torch.core  # noqa: F401, E402

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
K = 512
N = 4


def pattern(half: int) -> np.ndarray:
    j = np.arange(half)
    return ((j * 5 + 3) & 0xFF).astype(np.uint8)


def models_per_part(part: int):
    """expected source byte index per lane, for each model, for one part."""
    lane = np.arange(128)
    g = lane // 32
    b = 2 * lane                     # low byte of the lane
    m1 = g * 64 + ((b >> 2) & 63)
    m2 = (b >> 2) & 63
    return m1, m2


def lanes_from_bytes(src: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """what the kernel's shr/and produces: even lane -> low nibble, odd -> high."""
    out = src[idx]
    lane = np.arange(128)
    return np.where(lane % 2 == 0, out & 0xF, (out >> 4) & 0xF)


def main() -> int:
    print(f"probe_dual_group: kernel={KERNEL}")
    torch.ops.load_library(str(EXT))
    op = torch.ops.custom_op.custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2
    half = K // 2
    packed = np.tile(pattern(half), (1, N, 1)).reshape(1, N, half)
    scales = np.full((1, N, K // 16), 0x38, dtype=np.uint8)
    ids = torch.tensor([0], dtype=torch.int32, device="hpu")
    out = op(ids,
             torch.from_numpy(np.ascontiguousarray(packed)).to("hpu"),
             torch.from_numpy(np.ascontiguousarray(scales)).to("hpu"),
             torch.ones(1, dtype=torch.float32, device="hpu"))
    torch.hpu.synchronize()
    got = out.cpu().float().numpy()          # [S, N, K] -> torch view of the op output
    print(f"  out shape {got.shape}")
    # invert e2m1: map value -> candidate nibbles (0 and 8 both -> +0)
    cand = {}
    for code in range(16):
        v = E2M1[code & 7] * (-1.0 if code & 8 else 1.0)
        cand.setdefault(float(v) + 0.0, []).append(code)
    row = got.reshape(1, N, -1)[0, 0]
    nibbles = []
    for v in row:
        key = float(v) + 0.0
        opts = cand.get(key)
        if opts is None:
            nibbles.append((-1,))
        else:
            nibbles.append(tuple(opts))
    src = pattern(half)
    srt = np.sort(src)
    print(f"  first 16 lanes of part 0: "
          f"{[n[0] if len(n) == 1 else n for n in nibbles[:16]]}")
    verdict = []
    for name, fn in (("M1 same-group", lambda: models_per_part(0)[0]),
                     ("M2 group-0", lambda: models_per_part(0)[1])):
        ok = 0
        tot = 0
        for part in range(4):
            m1, m2 = models_per_part(part)
            idx = m1 if name.startswith("M1") else m2
            want = lanes_from_bytes(src, idx)
            for lane in range(128):
                pos = part * 128 + lane
                c = nibbles[pos]
                tot += 1
                ok += 1 if (want[lane] in c) else 0
        verdict.append((name, ok, tot))
    for name, ok, tot in verdict:
        print(f"  {name:14s} lanes matching: {ok}/{tot} "
              f"({100.0*ok/tot:5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
