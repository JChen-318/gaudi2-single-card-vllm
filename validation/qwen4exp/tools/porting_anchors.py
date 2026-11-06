# SPDX-License-Identifier: Apache-2.0
"""Verify every ``path:line`` anchor (and its quoted snippet) in PORTING_MAP.md.

Host-only: reads text files, spawns nothing, finishes in milliseconds.

Two roots: the STRATA reference checkout and this repository. A path's first
component decides the root, so the map stays readable while remaining auditable.

Recognised patterns inside one markdown line::

    ... `hpu/tpc/qsa_select_mask.c:50` ...                     anchor only
    ... `hpu/tpc/qsa_select_mask.c:50-63` — `code ...`          anchor + quote

``em dash`` (U+2014) is the only separator treated as "this quote must be at
that anchor", which is the convention PORTING_MAP.md uses throughout. The quote
must occur inside the anchor's line range with whitespace collapsed. An anchor
with no quote is still checked for existence and bounds. Every anchor is
reported, so an unverified claim is visible as "anchor, no quote".

Usage::

    .venv/bin/python validation/qwen4exp/tools/porting_anchors.py [-q] [--map PATH]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

STRATA_ROOT = Path(os.environ.get("STRATA_ROOT", "~/gaudi2-single-card-strata")).expanduser()
PORT_ROOT = Path(__file__).resolve().parents[3]
MAP = Path(__file__).resolve().parents[1] / "PORTING_MAP.md"

ROOTS = {
    "hpu": STRATA_ROOT,
    "src": STRATA_ROOT,
    "include": STRATA_ROOT,
    "docs": STRATA_ROOT,
    "vllm-gaudi": PORT_ROOT,
    "validation": PORT_ROOT,
}

PATH_RE = r"[A-Za-z0-9_./+-]+\.(?:c|cpp|hpp|h|py|json|md|inc|txt)"
# `<path>:<first>[-<last>]` optionally followed by ` — ` and a quoted snippet.
PAIR = re.compile(
    rf"`(?P<path>{PATH_RE}):(?P<first>\d+)(?:-(?P<last>\d+))?`"
    rf"(?:\s*\u2014\s*`(?P<quote>[^`]+)`)?"
)

# Shorthand used in the prose (a bare basename) -> the one file it can mean. An
# entry is only added when the basename is unique in both trees, so a new file
# with the same name forces this table to be revisited instead of silently
# resolving to the wrong one.
SHORTHAND = {
    "glue.cpp": "hpu/tpc/glue.cpp",
    "hc_ple.cpp": "hpu/src/hc_ple.cpp",
    "hc_ple.hpp": "hpu/include/strata/hpu/hc_ple.hpp",
    "hc_write.c": "hpu/tpc/hc_write.c",
    "hc_write_moe_routed.c": "hpu/tpc/hc_write_moe_routed.c",
    "iq4q8_gemv_packed.c": "hpu/tpc/iq4q8_gemv_packed.c",
    "model_abi.hpp": "hpu/include/strata/hpu/model_abi.hpp",
    "moe_mme_count.c": "hpu/tpc/moe_mme_count.c",
    "moe_mme_place.c": "hpu/tpc/moe_mme_place.c",
    "moe_mme_scatter.c": "hpu/tpc/moe_mme_scatter.c",
    "ngram.hpp": "include/strata/kernels/ngram.hpp",
    "packed_matmul.c": "hpu/tpc/packed_matmul.c",
    "ple_embedding.py": "vllm-gaudi/vllm_gaudi/models/qwen4exp/ple_embedding.py",
    "ple_runtime.py": "vllm-gaudi/vllm_gaudi/models/qwen4exp/ple_runtime.py",
    "qsa.py": "vllm-gaudi/vllm_gaudi/models/qwen4exp/qsa.py",
    "qsa_math.py": "vllm-gaudi/vllm_gaudi/models/qwen4exp/qsa_math.py",
    "qsa_select_mask.c": "hpu/tpc/qsa_select_mask.c",
    "qsa_state.cpp": "hpu/src/qsa_state.cpp",
    "qsa_state.hpp": "hpu/include/strata/hpu/qsa_state.hpp",
    "qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2.c": "vllm-gaudi/csrc/qwen4exp/kernels/qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2.c",
    "u4_moe_bucket.c": "hpu/tpc/u4_moe_bucket.c",
}


def collapse(text: str) -> str:
    return " ".join(text.split())


def resolve(path: str) -> Path | None:
    if "/" not in path:
        path = SHORTHAND.get(path, "")
        if not path:
            return None
    root = ROOTS.get(path.split("/", 1)[0])
    return None if root is None else root / path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default=str(MAP))
    parser.add_argument("-q", "--quiet", action="store_true", help="only the summary")
    args = parser.parse_args(argv)

    map_path = Path(args.map)
    lines = map_path.read_text(encoding="utf-8").splitlines()
    anchors = quotes = 0
    failures: list[str] = []

    for number, line in enumerate(lines, start=1):
        for match in PAIR.finditer(line):
            path = match.group("path")
            first = int(match.group("first"))
            last = int(match.group("last") or first)
            quote = match.group("quote")
            anchors += 1
            resolved = resolve(path)
            if resolved is None:
                failures.append(f"line {number}: no root for {path}")
                continue
            if not resolved.is_file():
                failures.append(f"line {number}: missing file {path}")
                continue
            source = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
            if first < 1 or last > len(source):
                failures.append(f"line {number}: {path}:{first}-{last} beyond {len(source)} lines")
                continue
            note = f"{path}:{first}-{last}"
            if quote is None:
                if not args.quiet:
                    print(f"ok  {note} (anchor)")
                continue
            quotes += 1
            window = collapse("\n".join(source[first - 1 : last]))
            if collapse(quote) not in window:
                failures.append(f"line {number}: {note} lacks quote {quote!r}")
                continue
            if not args.quiet:
                print(f"ok  {note} (quote)")

    print(f"\n{anchors} anchors, {quotes} quoted snippets, {len(failures)} failures")
    for failure in failures:
        print("FAIL " + failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
