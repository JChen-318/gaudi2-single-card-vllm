#!/usr/bin/env python3
"""Which cells the three QSA selection laws keep, enumerated without a device.

The three laws in play (see PORTING_MAP.md §1.4 D2/D3):

  * STRATA host   ``QsaState::select``      — a *cell* cut at ``selection_width = min(cells, top_k+block-1)``
  * STRATA device ``qsa_select_mask.c``     — the same cut, with the row's own block forced inside it
  * ours          ``qsa_math.py``           — a *block* cut at ``top_k // block`` plus the incomplete tail

They are three different functions of the scores, so this script only pins the *widths* — the part that can be
checked arithmetically. It asserts the claims PORTING_MAP.md makes and prints the table it quotes.

Exit code 0 on success, 1 on any failure. Runs in well under a second.
"""
from __future__ import annotations

import sys

BLOCK = 4          # indexer_compress_ratio
TOP_K = 2048       # indexer_budget
WIDTH_SEL = TOP_K + BLOCK - 1   # QsaConfig::selection_width


def strata_cells(size: int) -> int:
    """A cell cut: keep min(size, 2051)."""
    return min(size, WIDTH_SEL)


def ours_cells(size: int) -> int:
    """A block cut: up to 512 whole complete blocks, then the incomplete tail."""
    block_topk = TOP_K // BLOCK
    complete = min(size // BLOCK, block_topk)
    tail = size - (size // BLOCK) * BLOCK
    return complete * BLOCK + tail


def main() -> int:
    failures = 0

    def check(ok: bool, message: str) -> None:
        nonlocal failures
        print(("ok   " if ok else "FAIL ") + message)
        failures += 0 if ok else 1

    print(f"block={BLOCK} top_k={TOP_K} width_sel={WIDTH_SEL}\n")
    print(f"{'size':>7} {'r':>2} {'strata':>7} {'ours':>6} {'delta':>6}")
    for size in (2050, 2051, 2052, 2053, 4096, 65536, 65537, 65538, 65539):
        strata, ours = strata_cells(size), ours_cells(size)
        print(f"{size:7d} {size % BLOCK:2d} {strata:7d} {ours:6d} {strata - ours:6d}")
    print()

    check(all(ours_cells(s) <= strata_cells(s) for s in range(1, 400_000)),
          "ours is never wider than Strata's (sweep 1..399999)")
    check(max(strata_cells(s) - ours_cells(s) for s in range(1, 400_000)) == BLOCK - 1,
          f"the widest gap is block-1 = {BLOCK - 1} cells")
    check(all((ours_cells(s) == strata_cells(s)) == (s % BLOCK == BLOCK - 1)
              for s in range(WIDTH_SEL + 1, 400_000)),
          "for rows longer than width_sel they coincide exactly when size % block == block-1")
    check(ours_cells(TOP_K) == TOP_K, "a row of exactly top_k cells keeps all of them")
    check(ours_cells(WIDTH_SEL) == WIDTH_SEL, "and so does one of width_sel cells (identity selection)")

    print(f"\n{failures} failure(s)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
