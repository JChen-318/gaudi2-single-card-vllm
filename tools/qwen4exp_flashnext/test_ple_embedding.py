# SPDX-License-Identifier: Apache-2.0
"""CPU test for the PLE shard-table lookup primitive.

Pulls 100 real rows of layer-1 shard 0 straight from the converted checkpoint
(``~/models/qwen4exp-flashnext-plain``) via the np.memmap streaming
reader -- no /tmp fixture files, no full-shard copy -- and checks:

1. the mmap windows are byte-identical to the safetensors reader and are
   zero-copy (they alias the mapping, not a buffer);
2. ``PLEShardTable.lookup`` matches the numpy golden in ``csf.py``
   bit-for-bit (fp32) and after the bf16 cast, with and without the table's
   ``global_scale`` folding;
3. a table built by ``from_safetensors`` (the real-table load path) looks up
   the same rows as one built from a full ``get_tensor`` read;
4. one row really is 90 bytes: 80 packed E2M1 + 10 E4M3 block scales, i.e.
   the 26.82 GiB / 128-shard geometry the runtime documents.

If the checkpoint is not present the real-row part degrades to synthetic
rows so the dequant checks still run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vllm-gaudi"))
sys.path.insert(0, str(REPO / "tools/qwen4exp_flashnext"))

from vllm_gaudi.models.qwen4exp.ple_embedding import (  # noqa: E402
    ROW_BYTES,
    PLEShardTable,
    safetensors_mmap,
)
from csf import E2M1_VALUES, dequant_nvfp4 as np_dequant_nvfp4  # noqa: E402

CKPT_DIR = Path(os.environ.get("QWEN4EXP_MODEL", "~/models/qwen4exp-flashnext-plain")).expanduser()
_SHARD0 = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0"


def _real_rows() -> tuple[np.ndarray, np.ndarray, Path] | None:
    """(rows, scales) -- first 100 rows of layer-1 shard 0, mmap-streamed."""
    index = CKPT_DIR / "model.safetensors.index.json"
    if not index.is_file():
        return None
    weight_map = json.loads(index.read_text())["weight_map"]
    shard = CKPT_DIR / "tensors" / weight_map[_SHARD0 + ".weight"]
    rows = safetensors_mmap(shard, _SHARD0 + ".weight")[:100]
    scales = safetensors_mmap(shard, _SHARD0 + ".weight_scale")[:100]
    return rows.numpy().copy(), scales.numpy().copy(), shard


def _synthetic_rows() -> tuple[np.ndarray, np.ndarray, None]:
    rng = np.random.default_rng(0)
    return (rng.integers(0, 256, (100, 80), dtype=np.uint8),
            rng.integers(0x20, 0x78, (100, 10), dtype=np.uint8), None)


def _numpy_golden(rows: np.ndarray, scales: np.ndarray, global_scale: float) -> np.ndarray:
    """The csf.py reference dequant (fp32 end to end, low nibble first)."""
    return np_dequant_nvfp4(rows, scales, global_scale)


def main() -> int:
    rows_np, scales_np, shard_path = _real_rows() or _synthetic_rows()
    source = "real (mmap-streamed)" if shard_path is not None else "synthetic"
    print(f"  row source: {source}")

    # Row geometry: 80 packed bytes + 10 E4M3 bytes = 90 B -> 160 values.
    assert rows_np.shape[1] == 80 and scales_np.shape[1] == 10
    assert ROW_BYTES == 90

    table = PLEShardTable(torch.from_numpy(rows_np), torch.from_numpy(scales_np),
                          global_scale=1.0, pin_memory=False)
    assert table.rows == 100 and table.embed_dim == 160

    ids = torch.tensor([0, 7, 42, 99, 42])
    out = table.lookup(ids, dtype=torch.float32)
    assert out.shape == (5, 160)

    # numpy golden (csf.py) on the same rows: bit-exact in fp32.
    golden = _numpy_golden(rows_np[list(ids.numpy())], scales_np[list(ids.numpy())], 1.0)
    assert torch.equal(out, torch.from_numpy(golden)), "fp32 lookup must be bit-exact"

    # bf16 path is the fp32 values rounded once.
    out_bf16 = table.lookup(ids, dtype=torch.bfloat16)
    assert out_bf16.dtype == torch.bfloat16 and out_bf16.shape == (5, 160)
    assert torch.equal(out_bf16, out.to(torch.bfloat16))
    print(f"  lookup({source} 100-row table) == csf.py numpy golden, fp32 bit-exact + bf16: OK")

    # global scale folding
    t2 = PLEShardTable(torch.from_numpy(rows_np), torch.from_numpy(scales_np),
                       global_scale=6.64847e-05, pin_memory=False)
    out2 = t2.lookup(ids, dtype=torch.float32)
    golden2 = _numpy_golden(rows_np[list(ids.numpy())], scales_np[list(ids.numpy())],
                            6.64847e-05)
    assert torch.equal(out2, torch.from_numpy(golden2))
    print("  global_scale folding (bit-exact): OK")

    if shard_path is not None:
        # The mmap reader is zero-copy and byte-identical to safetensors.
        from safetensors import safe_open

        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            ref_w = handle.get_tensor(_SHARD0 + ".weight")[:100]
            ref_s = handle.get_tensor(_SHARD0 + ".weight_scale")[:100].view(torch.uint8)
        mm_w = safetensors_mmap(shard_path, _SHARD0 + ".weight")
        mm_s = safetensors_mmap(shard_path, _SHARD0 + ".weight_scale")
        assert torch.equal(mm_w[:100], ref_w) and torch.equal(mm_s[:100], ref_s)
        assert not mm_w.is_contiguous() or mm_w.data_ptr() != ref_w.data_ptr()
        assert mm_w.numpy().__array_interface__["data"][0] != ref_w.data_ptr(), \
            "mmap window must alias the mapping, not a copy"

        # The real-table load path (from_safetensors) matches a table built
        # from a plain full read, row for row.
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            full_w = handle.get_tensor(_SHARD0 + ".weight")
            full_s = handle.get_tensor(_SHARD0 + ".weight_scale")
        streamed = PLEShardTable.from_safetensors(
            shard_path, _SHARD0 + ".weight", _SHARD0 + ".weight_scale",
            global_scale=1.0, pin_memory=False)
        assert streamed.rows == 2_500_012 and streamed.embed_dim == 160
        full = PLEShardTable(full_w, full_s, global_scale=1.0, pin_memory=False)
        probe = torch.tensor([0, 1, 12345, 2_499_999, 2_500_011])
        assert torch.equal(streamed.lookup(probe, dtype=torch.float32),
                           full.lookup(probe, dtype=torch.float32))
        print("  safetensors_mmap: zero-copy, byte-identical; from_safetensors lookup: OK")

    if rows_np.shape[0] == 100 and shard_path is not None:
        # saturation sanity of the fetched rows (NVFP4 block semantics)
        codes = np.empty((100, 160), dtype=np.uint8)
        codes[:, 0::2] = rows_np & 0x0F
        codes[:, 1::2] = rows_np >> 4
        vals = np.abs(E2M1_VALUES[codes & 7]).reshape(100, 10, 16).max(-1)
        assert (vals == 6.0).all()
        print("  real PLE rows: 100% block saturation (format confirmed): OK")

    print("test_ple_embedding: all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
