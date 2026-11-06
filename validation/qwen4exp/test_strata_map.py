# SPDX-License-Identifier: Apache-2.0
"""Host-only checks that ``STRATA_QWEN4EXP_MAP.md`` still describes the trees it cites.

Three properties, each of which fails loudly when the document rots:

1. every ``path:line`` citation resolves and the 55 pinned lines still carry the
   text the map quotes (``tools/check_strata_map.py``);
2. the module/kernel counts and the kernel ids in the map's appendix are what
   ``hpu/tpc/glue.cpp`` and the two trees actually contain today
   (``tools/strata_module_index.py``);
3. the five call contracts are present and each names at least one anchor.

No torch, no device, no network: the tools are stdlib only by construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent / "tools"
MAP = Path(__file__).resolve().parent / "STRATA_QWEN4EXP_MAP.md"

sys.path.insert(0, str(TOOLS))
try:
    import check_strata_map  # noqa: E402
    import strata_module_index  # noqa: E402
finally:
    # Do not leave the tools directory on sys.path for every other test module:
    # these two names are imported once and live in sys.modules from here on.
    sys.path.remove(str(TOOLS))


def test_anchor_checker_sees_both_trees():
    """The checker must not silently pass because a tree is missing."""
    if not check_strata_map.STRATA.is_dir():
        pytest.skip("STRATA tree is not present on this host")
    assert check_strata_map.PORT.is_dir()
    assert (check_strata_map.STRATA / "hpu/tpc/glue.cpp").is_file()


def test_map_anchors_resolve_and_pinned_lines_match():
    problems, notes = check_strata_map.check_map()
    assert not problems, "\n".join(problems)
    # A document that lost its anchors would also produce no problems.
    anchors = int(next(n for n in notes if n.startswith("anchors:")).split()[1])
    assert anchors >= 80, f"only {anchors} distinct anchors left in {MAP.name}"
    assert any(n.startswith("critical: 55 ") for n in notes)


def test_kernel_ids_and_counts_match_the_map():
    entries = strata_module_index.glue_table()
    assert len(entries) == 112, "the kernel table changed size; re-derive the ids in the map"
    by_name = {entry["guid"]: entry["id"] for entry in entries}
    # The ids the map quotes (sections 1 and 2 and appendix A). Ids are append-only
    # in the glue, so these are stable identifiers, not incidental ordering.
    expected = {
        "strata_gdn_step_f32": 0,
        "strata_gdn_conv_f32": 1,
        "strata_iq4q8_gemv_f32": 22,
        "strata_q8_quant_packed_f32": 34,
        "strata_iq4q8_gemv_packed_f32": 35,
        "strata_moe_act_packed_f32": 38,
        "strata_hc_write_moe_f32": 42,
        "strata_iq4q8_gemv_tile_f32": 47,
        "strata_iq4q8_gemv_group_f32": 53,
        "strata_iq4q8_gemv_packed_pf_f32": 54,
        "strata_iq4q8_gemv_pair_f32": 67,
        "strata_moe_mme_join_perm_f32": 71,
        "strata_moe_mme_count_f32": 72,
        "strata_moe_mme_prefix_f32": 73,
        "strata_moe_mme_scatter_f32": 74,
        "strata_iq4_expand_tiles_bf16": 77,
        "strata_qsa_select_mask_f32": 91,
        "strata_qsa_gather_fp8": 95,
        "strata_ple_gate_rows_f32": 103,
        "strata_ple_conv_rows_f32": 104,
    }
    assert {name: by_name.get(name) for name in expected} == expected
    assert len(strata_module_index.kernel_inventory()) == 108


def test_module_index_sees_our_four_guids():
    ours = strata_module_index.our_inventory()
    assert sorted(ours["guids"]) == [
        "custom_qwen4exp_nvfp4_moe_count_i32_gaudi2",
        "custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2",
        "custom_qwen4exp_nvfp4_moe_prefix_i32_gaudi2",
        "custom_qwen4exp_nvfp4_moe_scatter_i32_gaudi2",
    ]
    assert len(ours["native_kernels"]) == 4


def test_the_five_call_contracts_are_present():
    text = MAP.read_text()
    contracts = ["### C1 — Resident weight plane", "### C2 — Decode GEMV call",
                 "### C3 — Bucketing triple", "### C4 — PLE row contract", "### C5 — Chain and workspace call"]
    for heading in contracts:
        assert heading in text, f"missing contract: {heading}"
    # Each contract must cite at least one file:line anchor of its own.
    body = text.split("## 6. The five call contracts", 1)[1].split("## 7.", 1)[0]
    assert len(check_strata_map.iter_anchors(body)) >= 10
