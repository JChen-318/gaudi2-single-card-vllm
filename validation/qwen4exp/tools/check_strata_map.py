# SPDX-License-Identifier: Apache-2.0
"""Verify the anchors of ``validation/qwen4exp/STRATA_QWEN4EXP_MAP.md`` against the sources.

Two independent checks, both host-only (stdlib, no torch, no device):

1. **Every anchor resolves.** The map cites sources as `` `path:line` `` or
   `` `path:line-line` `` in backticks. Each path is resolved under the tree it
   belongs to (STRATA for ``hpu/ src/ include/ docs/ tools/ third_party/``, this
   repo for ``vllm-gaudi/ validation/ native/ scripts/``), and the line (or both
   range ends) must exist. A file that moved or shrank breaks the check.
2. **The load-bearing lines say what the map says they say.** ``CRITICAL`` below
   pins the ~40 lines a reader is asked to trust: the four-bit LUT multiply and
   its scale epilogue, the two weight-plane address formulas, the preshuffle, the
   `parts` defaults, the glue caps, the GEMV store, the NVFP4 reference decoder,
   the PLE table constants and reader, the selection law, the HC mix/write
   constants and the chain struct. Each is checked as a substring of that line.

Usage::

    .venv/bin/python validation/qwen4exp/tools/check_strata_map.py
    .venv/bin/python validation/qwen4exp/tools/check_strata_map.py --list

Exit code 0 on success, 1 on any failed check (details on stderr).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

STRATA = Path(os.environ.get("STRATA_ROOT", "~/gaudi2-single-card-strata")).expanduser()
PORT = Path(__file__).resolve().parents[3]
MAP = PORT / "validation/qwen4exp/STRATA_QWEN4EXP_MAP.md"

ANCHOR = re.compile(r"`((?:hpu|src|include|docs|tools|third_party|vllm-gaudi|validation|native|scripts)"
                    r"/[A-Za-z0-9_./+-]+):(\d+)(?:-(\d+))?`")

# (path, line, substring) - the text the map quotes at that anchor.
CRITICAL: list[tuple[str, int, str]] = [
    # --- the 4-bit MoE GEMV: layout, LUT multiply, scale epilogue, store -------
    ("hpu/tpc/iq4q8_gemv.c", 2, "One member = (slot, column part, group of 5 row tiles of 256 rows)"),
    ("hpu/tpc/iq4q8_gemv.c", 7, "codes  at (((g*blocks + b)*16 + c/2)*5 + t)*256"),
    ("hpu/tpc/iq4q8_gemv.c", 10, "scales at codes_bytes + ((g*blocks + b)*5 + t)*1024"),
    ("hpu/tpc/iq4q8_gemv.c", 13, "u8 shuffle reads mask bits 0-4"),
    ("hpu/tpc/iq4q8_gemv.c", 20, "const int tiles=rows/256,groups=tiles/5,blocks=columns/32,per_part=blocks/parts;"),
    ("hpu/tpc/iq4q8_gemv.c", 43, "v_u8_shuffle_b(lut,v|(unsigned char)0x80,0,lut)"),
    ("hpu/tpc/iq4q8_gemv.c", 45, "ACC=v_i8_mac_b(w1,x1,v_i8_mac_b(w0,x0,ACC,0,1,0),0,1,0)"),
    ("hpu/tpc/iq4q8_gemv.c", 53, "partial[T][0]=v_f32_mac_b(v_convert_i32_to_f32_b(ACC.v1,0)*s0,dxv,partial[T][0])"),
    ("hpu/tpc/iq4q8_gemv.c", 61, "int5 o={(group*5+t)*256+64*k,slot,part,0,0};"),
    ("hpu/tpc/iq4q8_gemv_packed.c", 27, "const int base=((group*blocks+b)*16*5)*256;"),
    ("hpu/tpc/iq4q8_gemv_packed.c", 33, "int5 at={base+(c/2)*5*256,erow,0,0,0};"),
    ("hpu/tpc/iq4q8_gemv_packed.c", 34, "i0=(unsigned char)(0x80|(c&15))"),
    ("hpu/tpc/iq4q8_gemv_packed.c", 79, "int5 sp={codes_bytes+((group*blocks+b)*5)*1024,erow,0,0,0};"),
    ("hpu/tpc/iq4q8_gemv_packed_pf.c", 2, "five code vectors are issued before this pair's MACs consume theirs"),
    ("hpu/tpc/iq4q8_gemv_packed_pf.c", 31, "n0=v_u8_ld_tnsr_b(at,W);at[0]+=256;"),
    ("hpu/src/resident_experts.cpp", 44, "const uint64_t tile=row/256,i=row%256,group=tile/5,t=tile%5,lane=4*(i%64)+i/64;"),
    ("hpu/src/resident_experts.cpp", 48, "out[((base+p)*5+t)*256+lane]=uint8_t(codes[32*b+2*p]|(codes[32*b+2*p+1]<<4));"),
    ("hpu/src/resident_experts.cpp", 49, "std::memcpy(scale_out+(((group*blocks+b)*5+t)*256+i)*4,&scales[b],4);"),
    ("hpu/src/resident_experts.cpp", 199, 'gemv_parts_env("STRATA_HPU_GEMV_PARTS_GU",2)'),
    ("hpu/src/resident_experts.cpp", 205, "gemv(x_codes,w_gu,s_gu,parts,h,2*ff,per_row,gu_count)"),
    # --- the glue caps and index spaces --------------------------------------
    ("hpu/tpc/glue.cpp", 938, "bytes=r/2*c+r*(c/32)*4"),
    ("hpu/tpc/glue.cpp", 939, "(kernel_id==35 || kernel_id==47 || kernel_id==54)?c/16:c"),
    ("hpu/tpc/glue.cpp", 943, "out.indexSpaceGeometry[0]=s*a.parts*(r/(kernel_id==47?256:1280));"),
    # --- the NVFP4 reference decoder already in STRATA -------------------------
    ("hpu/tpc/fp4_values.inc", 8, "0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12"),
    ("hpu/tpc/packed_matmul.c", 469, "static float packed_ue4m3(unsigned int x) {"),
    ("hpu/tpc/packed_matmul.c", 472, "const int e=exp==0?-10:exp-8;"),
    ("hpu/tpc/packed_matmul.c", 487, "dot+=packed_kvalues_fp4[pair&15]"),
    ("hpu/src/expert_scales.cpp", 13, 'NVFP4 .scale must be F32 with one value per expert'),
    # --- QSA, GDN, HC ---------------------------------------------------------
    ("hpu/src/qsa_state.cpp", 164, "uint32_t QsaConfig::selection_width"),
    ("hpu/src/qsa_state.cpp", 430, "score+=std::max(dot(query.data()+uint64_t(head)*c.index_width,pooled,c.index_width),0.0);"),
    ("hpu/tpc/qsa_select_mask.c", 7, "A block's score is the sum over the indexer heads"),
    ("hpu/tpc/qsa_select_mask.c", 50, "const int forced=size%block?size-size%block:size;"),
    ("hpu/tpc/gdn_step.c", 2, "q is already L2-normalized/scaled"),
    ("hpu/tpc/hc_mix.c", 8, "sum+=v_f32_ld_tnsr_b(p,normalized)*strata_sigmoid(v_f32_ld_tnsr_b(p,gate));"),
    ("hpu/tpc/hc_write.c", 7, "float64 weight=strata_sigmoid(scalar)*2.0f;"),
    # --- PLE ------------------------------------------------------------------
    ("include/strata/kernels/ngram.hpp", 35, "PLE_N_HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM"),
    ("include/strata/kernels/ngram.hpp", 46, "PLE_TABLE_ROWS = 320001536ull"),
    ("include/strata/kernels/ngram.hpp", 52, "PLE_ROW_BYTES_MAX = PLE_ROW_BYTES_BF16"),
    ("src/kernels/ngram.cpp", 217, "kMaxPrefetch = 16"),
    ("src/kernels/ngram.cpp", 322, "impl_->reader.open(gguf_path, table_offset, n_rows, io.max_inflight, io.cache_rows"),
    ("hpu/src/ple_input.cpp", 125, "table_.gather_batch_threads(rows.data(),tokens.size(),out,error,16)"),
    # --- prefill MME bucketing, chaining --------------------------------------
    ("hpu/src/moe_mme.cpp", 73, "g.gemm(tiles_row,tile_upper,end_row)"),
    ("hpu/src/moe_mme.cpp", 77, "strata_moe_mme_scatter_f32"),
    ("hpu/include/strata/hpu/decode_chain.hpp", 20, "struct DecodeChain {"),
    ("hpu/include/strata/hpu/decode_chain.hpp", 36, "void advance() {current=1-current;}"),
    ("hpu/src/layer.cpp", 34, 'const std::string kChainWorkspace="verify_chain";'),
    # --- our port -------------------------------------------------------------
    ("native/qwen4exp-nvfp4-moe/kernels/qwen4exp_nvfp4_moe_gemv_bf16_gaudi2.c", 46,
     "const ushort128 decoded = nvfp4_mul_bits(nibble, scale);"),
    ("native/qwen4exp-nvfp4-moe/kernels/qwen4exp_nvfp4_moe_gemv_bf16_gaudi2.c", 48,
     "ACC = v_bf16_mac_acc32_b(weight, XV, ACC);"),
    ("native/qwen4exp-nvfp4-moe/kernels/qwen4exp_nvfp4_moe_gemv_bf16_gaudi2.c", 64,
     "const float64 total = v_f32_reduce_add(ACC.v1 + ACC.v2) * s2;"),
    ("native/qwen4exp-nvfp4-moe/host/qwen4exp_nvfp4_moe_gemv_bf16_gaudi2.cpp", 73,
     "slots <= QWEN4EXP_MOE_MAX_DECODE_SLOTS"),
    ("native/qwen4exp-nvfp4-moe/host/qwen4exp_nvfp4_moe_gemv_bf16_gaudi2.cpp", 79,
     "scales.maxSizes[0] * 8 == k_half"),
    ("vllm-gaudi/vllm_gaudi/models/qwen4exp/packed_moe.py", 480,
     'os.environ.get("VLLM_QWEN4EXP_NVFP4_GEMV", "1") != "0"'),
    ("vllm-gaudi/vllm_gaudi/models/qwen4exp/packed_moe.py", 547,
     'nvfp4_gemv_enabled() and hidden.device.type == "hpu"'),
    ("vllm-gaudi/vllm_gaudi/models/qwen4exp/ple_embedding.py", 47, "class PLEShardTable:"),
    ("vllm-gaudi/vllm_gaudi/models/qwen4exp/ple_runtime.py", 424, "for s in torch.unique(shard).tolist():"),
]


def tree_for(path: str) -> Path:
    return STRATA if path.split("/", 1)[0] in {
        "hpu", "src", "include", "docs", "tools", "third_party"} else PORT


def read_line(path: str, line: int) -> tuple[bool, str]:
    resolved = tree_for(path) / path
    if not resolved.is_file():
        return False, f"no such file: {resolved}"
    try:
        lines = resolved.read_text(errors="replace").splitlines()
    except OSError as error:  # pragma: no cover - unreadable file
        return False, f"cannot read {resolved}: {error}"
    if not 1 <= line <= len(lines):
        return False, f"{path}:{line} is out of range (file has {len(lines)} lines)"
    return True, lines[line - 1]


def find_text(path: str, expect: str) -> int | None:
    """Line containing ``expect`` anywhere in the file (tolerates shifts)."""
    resolved = tree_for(path) / path
    if not resolved.is_file():
        return None
    for index, line in enumerate(resolved.read_text(errors="replace").splitlines(), 1):
        if expect in line:
            return index
    return None


def iter_anchors(text: str) -> list[tuple[str, int, int]]:
    out = []
    for match in ANCHOR.finditer(text):
        path, first, last = match.group(1), int(match.group(2)), match.group(3)
        out.append((path, first, int(last) if last else first))
    return out


def check_map() -> tuple[list[str], list[str]]:
    problems: list[str] = []
    notes: list[str] = []
    if not MAP.is_file():
        return [f"missing {MAP}"], notes
    text = MAP.read_text(errors="replace")
    anchors = iter_anchors(text)
    seen: set[tuple[str, int, int]] = set()
    for path, first, last in anchors:
        if (path, first, last) in seen:
            continue
        seen.add((path, first, last))
        if last < first:
            problems.append(f"{path}:{first}-{last} has a reversed range")
            continue
        for line in (first, last):
            ok, detail = read_line(path, line)
            if not ok:
                problems.append(detail)
                break
    notes.append(f"anchors: {len(seen)} distinct path:line citations in {MAP.name}")

    for path, line, expect in CRITICAL:
        ok, detail = read_line(path, line)
        if not ok:
            problems.append(f"critical: {detail}")
        elif expect not in detail:
            # Sibling tracks keep editing these files; accept a pin whose text
            # still exists, just at a shifted line, and say so.
            shifted = find_text(path, expect)
            if shifted is None:
                problems.append(
                    f"critical: {path}:{line} does not contain {expect!r} "
                    f"(and the text is gone from the file); it is {detail.strip()[:120]!r}")
            else:
                notes.append(f"critical: {path}:{line} shifted to :{shifted}")
    notes.append(f"critical: {len(CRITICAL)} pinned lines checked")
    return problems, notes


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="print the pinned lines instead of checking")
    args = parser.parse_args(argv)
    if args.list:
        for path, line, expect in CRITICAL:
            print(f"{path}:{line}\t{expect}")
        return 0
    if not STRATA.is_dir():
        print(f"missing STRATA tree at {STRATA}", file=sys.stderr)
        return 1
    problems, notes = check_map()
    for note in notes:
        print(note)
    if problems:
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        print(f"\n{len(problems)} problem(s)", file=sys.stderr)
        return 1
    print("all anchors resolve and every pinned line matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
