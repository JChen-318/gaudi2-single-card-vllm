# SPDX-License-Identifier: Apache-2.0
"""Deterministic inventory of the STRATA Qwen3.8-Flash-Next port and our port of it.

This is the extraction helper behind ``validation/qwen4exp/STRATA_QWEN4EXP_MAP.md``:
it prints the module/kernel tables that the map quotes, so every number in the
map can be regenerated instead of trusted.

What it reads (read-only, no device, no torch, stdlib only):

  * ``hpu/tpc/glue.cpp``   - the kernel registration table. The **registration
    order is the kernel id** the glue's shape checks switch on (``id==22`` is
    ``strata_iq4q8_gemv_f32``). Ids are append-only, so they are stable
    identifiers worth quoting in the map.
  * ``hpu/tpc/*.c``        - every TPC kernel: LOC and the first comment line.
  * a fixed area map for ``hpu/src/*.cpp`` (the host side) and the Python/C++
    artifacts of our port, with LOC.
  * our four GUIDs          - parsed out of ``native/qwen4exp-nvfp4-moe/host``
    rather than restated here.

Usage::

    .venv/bin/python validation/qwen4exp/tools/strata_module_index.py            # markdown
    .venv/bin/python validation/qwen4exp/tools/strata_module_index.py --json     # machine readable
    .venv/bin/python validation/qwen4exp/tools/strata_module_index.py --section kernels

Exit code 0 on success, 1 if a path the inventory depends on is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

STRATA = Path(os.environ.get("STRATA_ROOT", "~/gaudi2-single-card-strata")).expanduser()
PORT = Path(__file__).resolve().parents[3]
NATIVE = PORT / "native/qwen4exp-nvfp4-moe"
PY_QWEN4EXP = PORT / "vllm-gaudi/vllm_gaudi/models/qwen4exp"

# The host-side modules, grouped by the part of the port they implement. The
# areas are the ones the map's section 1 walks; the roles are one line each.
HOST_AREAS: dict[str, list[tuple[str, str]]] = {
    "entry / scheduling": [
        ("model.cpp", "Model: layer table, decode_chain / verify_chain driver, PLE gather, head"),
        ("layer.cpp", "Layer: fused per-layer graph assembly (QSA or GDN + HC + MoE)"),
        ("layer_flow.cpp", "execute_layer / execute_layer_window: the host-side layer order"),
        ("model_plan.cpp", "startup plan: geometry, PLE source, memory admission"),
        ("runtime.cpp", "device, streams, events"),
        ("generation.cpp", "token loop, sampling, checkpoint schedule"),
    ],
    "QSA (indexer + sparse attention)": [
        ("qsa.cpp", "attention/projection graphs, FP8 KV cache read/write, gather, dense window"),
        ("qsa_block.cpp", "QsaBlock: decode/prefill plumbing, device gather/select bindings"),
        ("qsa_state.cpp", "the indexer host law: pooling, block scores, selection, KV commit"),
        ("qsa_window.cpp", "window batching of the same state machine"),
        ("qsa_window_flow.cpp", "prefill window flow for QSA layers"),
    ],
    "GDN (gated delta net)": [
        ("gdn.cpp", "GDN body graphs: projections, conv, prepare, norm, step"),
        ("gdn_block.cpp", "GdnBlock: state ping-pong, checkpoint/restore, external step"),
        ("gdn_chunked.cpp", "chunked gated delta rule for prefill windows (MME, kGdnChunk=64)"),
        ("gdn_window.cpp", "window flow for GDN layers"),
    ],
    "MoE (decode GEMV + prefill MME)": [
        ("resident_experts.cpp", "resident IQ4 pack, table upload, define_resident_moe (decode)"),
        ("moe.cpp", "expert executors: legacy BF16, direct FP8, TPC, grouped FP8 decode"),
        ("moe_mme.cpp", "grouped MME prefill: bucketing, tile expand, two GEMMs, join"),
        ("u4_resident.cpp", "U4 resident store + MME A/B layouts"),
        ("packed_expert_flow.cpp", "packed expert call sequencing"),
        ("expert.cpp", "expert residency / cache"),
        ("expert_cache.cpp", "expert cache replacement"),
        ("expert_scales.cpp", "NVFP4 per-expert F32 .scale (the analogue of our scale2)"),
        ("packed_quant.cpp", "Q8_1 / S2 activation quantizers"),
        ("cpu_experts.cpp", "CPU expert pool (the NVFP4 CPU reference lives here)"),
    ],
    "hyper-connections (HC)": [
        ("hc_ple.cpp", "HC read/write graphs, PLE block and PLE window graphs"),
        ("hc_window.cpp", "HC window flow"),
    ],
    "PLE (n-gram table)": [
        ("ple_input.cpp", "PleInput: hash config, row ids, gather/gather_rows"),
        ("ple_source.cpp", "table source preflight + RAM/reader byte plans"),
        ("ple_window.cpp", "window PLE (row history)"),
        ("ple_window_flow.cpp", "window PLE flow"),
    ],
    "launch chaining / I/O": [
        ("linear_graph.cpp", "LM head, packed matmul graph"),
        ("control_graph.cpp", "control-vector graph"),
        ("host_recipe.cpp", "HostRecipe: one pinned input block, one output block, launch/wait"),
        ("packed_expert_flow.cpp", "expert call sequencing"),
    ],
}

TPC_KERNEL_GROUPS: dict[str, list[str]] = {
    "MoE decode GEMV (4-bit)": [
        "iq4q8_gemv", "iq4q8_gemv_pipe", "iq4q8_gemv_packed", "iq4q8_gemv_packed_pf",
        "iq4q8_gemv_tile", "iq4q8_gemv_group", "iq4q8_gemv_pair", "iq4q8_gemv_rows",
    ],
    "MoE activation / routing": [
        "q8_quant", "q8_quant_packed", "q8_quant_shuffle", "route_topk",
        "route_group_starts", "route_group_order", "route_group_blocks", "route_share",
    ],
    "MoE prefill bucketing + MME": [
        "moe_mme_count", "moe_mme_prefix", "moe_mme_tilemap", "moe_mme_scatter",
        "moe_mme_place", "moe_mme_gather", "moe_mme_gather_fp8", "moe_mme_act",
        "moe_mme_act_fp8", "moe_mme_act_fast", "moe_mme_join", "moe_mme_join_perm",
        "moe_mme_join_fp8", "moe_act_packed", "iq4_expand_bf16", "iq4_expand_tiles",
        "iq4_expand_tiles_fp8",
    ],
    "MoE resident-U4 epilogues": ["u4_moe_bucket", "u4_bucket_blocks", "u4_route_blocks",
                                  "u4_moe_gather", "u4_moe_gu_epi", "u4_moe_down_epi",
                                  "u4_moe_join"],
    "QSA": ["qsa_prep", "qsa_gather", "qsa_gather_fp8", "qsa_gather_rows_fp8",
            "qsa_gather_blocks_fp8", "qsa_gather_window_bf16", "qsa_index_prep",
            "qsa_index_cells", "qsa_select_mask", "qsa_select_bits", "qsa_select_expand",
            "qsa_decode_blocks", "qsa_window_blocks", "qsa_verify_softmax",
            "qsa_verify_partials", "qsa_virtual"],
    "GDN": ["gdn_conv", "gdn_conv_rows", "gdn_conv_rows8", "gdn_conv_final",
            "gdn_prepare", "gdn_prepare_rows", "gdn_norm", "gdn_norm_rows",
            "gdn_norm_rows8", "gdn_norm_fast", "gdn_step", "gdn_step_cached",
            "gdn_step_pipe", "gdn_step_bcast", "gdn_step_rows4", "gdn_step_window"],
    "HC / PLE": ["hc_mix", "hc_mix_dual", "hc_mix_window", "hc_mix_window_dual",
                 "hc_write", "hc_write_moe", "hc_write_moe_routed", "hc_down_bf16",
                 "hc_down_rows_bf16", "ple_gate", "ple_gate_rows", "ple_gate_simd",
                 "ple_conv", "ple_conv_rows", "row_norm", "row_norm_dual",
                 "row_norm_simd"],
}

GLUE_ENTRY = re.compile(r'\{"(strata_[a-z0-9_]+)"\s*,\s*strata_[a-z0-9_]+_elf_start\s*,'
                        r'\s*strata_[a-z0-9_]+_elf_end\s*,\s*(\d+)\s*,\s*(\d+)\s*\}')
GUID = re.compile(r'custom_qwen4exp_[a-z0-9_]+')


def loc(path: Path) -> int:
    try:
        return sum(1 for _ in path.open("rb"))
    except FileNotFoundError:
        return 0


def first_comment(path: Path) -> str:
    """The kernel's first comment line: its own statement of what it does."""
    try:
        for line in path.read_text(errors="replace").splitlines():
            stripped = line.strip()
            if stripped.startswith("//"):
                return stripped.lstrip("/ ").strip()
    except FileNotFoundError:
        return ""
    return ""


def glue_table() -> list[dict]:
    """Every registered kernel, with the id the glue's shape checks use."""
    path = STRATA / "hpu/tpc/glue.cpp"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    out = []
    for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        m = GLUE_ENTRY.search(line)
        if not m:
            continue
        guid, ins, outs = m.group(1), int(m.group(2)), int(m.group(3))
        out.append({"id": len(out), "guid": guid, "inputs": ins, "outputs": outs,
                    "glue_line": line_no})
    return out


def kernel_inventory() -> list[dict]:
    table = {entry["guid"][len("strata_"):-len("_f32")] if entry["guid"].endswith("_f32")
             else entry["guid"][len("strata_"):]: entry
             for entry in glue_table()}
    out = []
    for path in sorted((STRATA / "hpu/tpc").glob("*.c")):
        stem = path.stem
        entry = table.get(stem) or table.get(stem.replace("_bf16", "")) or {}
        out.append({"kernel": stem, "loc": loc(path), "id": entry.get("id"),
                    "guid": entry.get("guid"), "first_comment": first_comment(path)})
    return out


def host_inventory() -> list[dict]:
    out = []
    for area, files in HOST_AREAS.items():
        for name, role in files:
            path = STRATA / "hpu/src" / name
            out.append({"area": area, "file": f"hpu/src/{name}", "loc": loc(path),
                        "role": role, "exists": path.is_file()})
    return out


def our_inventory() -> dict:
    kernels = []
    for path in sorted((NATIVE / "kernels").glob("*.c")):
        kernels.append({"file": str(path.relative_to(PORT)), "loc": loc(path)})
    host = []
    for path in sorted((NATIVE / "host").glob("*.cpp")):
        host.append({"file": str(path.relative_to(PORT)), "loc": loc(path)})
    # A registration name always ends in a digit; the bare prefix
    # (`kCustomPrefix = "custom_qwen4exp_nvfp4_moe_"`) must not be counted as one.
    guids = sorted({m.group(0) for path in (NATIVE / "host").glob("*.cpp")
                    for m in GUID.finditer(path.read_text(errors="replace"))
                    if m.group(0)[-1].isdigit()})
    python = []
    for path in sorted(PY_QWEN4EXP.glob("*.py")):
        python.append({"file": str(path.relative_to(PORT)), "loc": loc(path)})
    shipped = []
    for path in sorted((PORT / "vllm-gaudi/csrc/qwen4exp").rglob("*")):
        if path.suffix in {".c", ".cpp", ".h", ".hpp"}:
            shipped.append({"file": str(path.relative_to(PORT)), "loc": loc(path)})
    return {"native_kernels": kernels, "native_host": host, "guids": guids,
            "python": python, "shipped_csrc": shipped}


def render(section: str | None) -> str:
    lines: list[str] = []
    want = section or "all"

    if want in ("all", "kernels"):
        kernels = kernel_inventory()
        ids = [k for k in kernels if k["id"] is not None]
        lines += [f"## STRATA TPC kernels ({len(kernels)} `.c` files, {sum(k['loc'] for k in kernels)} LOC; "
                  f"{len(ids)} registered in `hpu/tpc/glue.cpp`)", ""]
        lines += ["| id | guid | in/out | kernel file | LOC | what the file says it does |",
                  "|---:|---|---|---|---:|---|"]
        table = {e["guid"]: e for e in glue_table()}
        for entry in glue_table():
            stem = entry["guid"][len("strata_"):]
            stem = stem[:-len("_f32")] if stem.endswith("_f32") else stem
            path = STRATA / "hpu/tpc" / f"{stem}.c"
            comment = first_comment(path).replace("|", "/")[:96]
            lines.append(f"| {entry['id']} | `{entry['guid']}` | {entry['inputs']}/{entry['outputs']} | "
                         f"`hpu/tpc/{stem}.c` | {loc(path)} | {comment} |")
        unregistered = [k["kernel"] for k in kernels if k["id"] is None]
        lines += ["", f"Not in the table (helpers/includes): {', '.join(sorted(unregistered)) or 'none'}", ""]

    if want in ("all", "host"):
        host = host_inventory()
        lines += [f"## STRATA host modules ({len(host)} files, {sum(h['loc'] for h in host)} LOC)", ""]
        area = None
        for entry in host:
            if entry["area"] != area:
                area = entry["area"]
                lines += ["", f"**{area}**", "", "| file | LOC | role |", "|---|---:|---|"]
            mark = "" if entry["exists"] else " **(missing)**"
            lines.append(f"| `{entry['file']}` | {entry['loc']} | {entry['role']}{mark} |")
        lines.append("")

    if want in ("all", "ours"):
        ours = our_inventory()
        lines += ["## Our port's artifacts", "",
                  f"- native kernels: {len(ours['native_kernels'])} files, "
                  f"{sum(k['loc'] for k in ours['native_kernels'])} LOC",
                  f"- native host/glue: {len(ours['native_host'])} files, "
                  f"{sum(k['loc'] for k in ours['native_host'])} LOC",
                  f"- Python (`qwen4exp/`): {len(ours['python'])} files, "
                  f"{sum(k['loc'] for k in ours['python'])} LOC",
                  f"- shipped `csrc/qwen4exp`: {len(ours['shipped_csrc'])} files, "
                  f"{sum(k['loc'] for k in ours['shipped_csrc'])} LOC",
                  "", "Our registered GUIDs:", ""]
        for guid in ours["guids"]:
            lines.append(f"- `{guid}`")
        lines.append("")

    if want in ("all", "groups"):
        kernels = {k["kernel"]: k for k in kernel_inventory()}
        lines += ["## TPC kernels by area", "", "| area | kernels (id) |", "|---|---|"]
        for area, stems in TPC_KERNEL_GROUPS.items():
            cells = []
            for stem in stems:
                for candidate in (stem, stem + "_fp8", stem + "_bf16"):
                    if candidate in kernels:
                        ident = kernels[candidate]["id"]
                        cells.append(f"`{candidate}`" + (f" ({ident})" if ident is not None else ""))
                        break
            lines.append(f"| {area} | {' '.join(cells)} |")
        lines.append("")

    if want in ("all", "summary"):
        kernels = kernel_inventory()
        host = host_inventory()
        ours = our_inventory()
        lines += ["## Summary", "",
                  f"- STRATA: {len(kernels)} TPC kernel sources, {len(glue_table())} registered kernels, "
                  f"{len(host)} host modules ({sum(h['loc'] for h in host)} LOC named in this inventory)",
                  f"- ours: {len(ours['native_kernels'])} native kernels + {len(ours['native_host'])} glue files, "
                  f"{len(ours['guids'])} custom GUIDs, {len(ours['python'])} Python modules",
                  ""]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit the inventory as JSON")
    parser.add_argument("--section", choices=["kernels", "host", "ours", "groups", "summary", "all"],
                        default="all")
    args = parser.parse_args(argv)
    if not STRATA.is_dir():
        print(f"missing STRATA tree at {STRATA}", file=sys.stderr)
        return 1
    if not NATIVE.is_dir():
        print(f"missing native library sources at {NATIVE}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"glue": glue_table(), "kernels": kernel_inventory(),
                          "host": host_inventory(), "ours": our_inventory()}, indent=1))
        return 0
    print(render(None if args.section == "all" else args.section))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
