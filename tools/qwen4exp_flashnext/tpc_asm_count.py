# SPDX-License-Identifier: Apache-2.0
"""Count TPC VLIW bundles and instructions in a ``tpc-clang -S`` dump.

TPC issue is one bundle per cycle (up to five instructions per bundle), so the
bundle count of a hot block is its cycle estimate and the instruction count is
its issue-slot pressure. The unit's Before/After tables are produced with this
script so the two sides are counted the same way:

  tpc-clang -O2 -march=gaudi2 -S kernels/<kernel>.c -o /tmp/<kernel>.S
  python tools/qwen4exp_flashnext/tpc_asm_count.py /tmp/<kernel>.S [--block 3]

Per basic block (``// %bb.N:`` markers) it prints instructions / bundles and
the ``ld_l_v`` / ``st_l_v`` local-memory traffic (the compiler's spill counter).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

BLOCK = re.compile(r"^// %bb\.(\d+):")
# A bundle opener is a line that is exactly "{" (the instructions follow on
# their own lines). Anything else that starts with a tab and a mnemonic is a
# single-instruction bundle.
MNEMONIC = re.compile(r"^\t([A-Za-z_][\w.]*)\b")


def parse(path: Path) -> tuple[dict[int, list[str]], list[str]]:
    """(per-block instruction lists, prologue instruction list)."""
    blocks: dict[int, list[str]] = {}
    prologue: list[str] = []
    current = prologue
    index = -1
    for raw in path.read_text().splitlines():
        match = BLOCK.match(raw)
        if match:
            index = int(match.group(1))
            current = blocks.setdefault(index, [])
            continue
        mnemonic = MNEMONIC.match(raw)
        if mnemonic:
            name = mnemonic.group(1)
            if name in ("NOP", "halt"):
                continue
            current.append(name)
    return blocks, prologue


def summarise(instructions: list[str]) -> dict:
    return {
        "instr": len(instructions),
        "bundles": len(instructions),  # placeholder, replaced below
        "spill_ld": sum(1 for name in instructions if name == "ld_l_v"),
        "spill_st": sum(1 for name in instructions if name == "st_l_v"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asm", type=Path)
    parser.add_argument("--block", type=int, default=None,
                        help="also print the instruction list of this %%bb.N")
    args = parser.parse_args()

    text = args.asm.read_text().splitlines()
    # Bundles: one per "{ ... }" group plus one per single instruction line.
    bundles = instrs = 0
    depth = 0
    per_block_bundles: dict[int, int] = {}
    per_block_instr: dict[int, int] = {}
    block = -1
    for raw in text:
        match = BLOCK.match(raw)
        if match:
            block = int(match.group(1))
            per_block_bundles.setdefault(block, 0)
            per_block_instr.setdefault(block, 0)
            continue
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped == "{":
            depth += 1
            bundles += 1
            per_block_bundles[block] = per_block_bundles.get(block, 0) + 1
            continue
        if stripped == "}":
            depth = max(0, depth - 1)
            continue
        mnemonic = MNEMONIC.match(raw)
        if not mnemonic:
            continue
        if mnemonic.group(1) in ("halt",):
            continue
        instrs += 1
        per_block_instr[block] = per_block_instr.get(block, 0) + 1
        if depth == 0:
            bundles += 1
            per_block_bundles[block] = per_block_bundles.get(block, 0) + 1

    spill_ld = sum(1 for line in text if MNEMONIC.match(line)
                   and MNEMONIC.match(line).group(1) == "ld_l_v")
    spill_st = sum(1 for line in text if MNEMONIC.match(line)
                   and MNEMONIC.match(line).group(1) == "st_l_v")
    print(f"{args.asm}")
    print(f"  total: instr {instrs}  bundles {bundles}  "
          f"ld_l_v {spill_ld}  st_l_v {spill_st}")
    for index in sorted(per_block_instr):
        print(f"  %bb.{index:<3d} instr {per_block_instr[index]:5d}  "
              f"bundles {per_block_bundles.get(index, 0):5d}")
    if args.block is not None:
        blocks, prologue = parse(args.asm)
        print(f"  %bb.{args.block} mnemonics: "
              f"{' '.join(blocks.get(args.block, []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
