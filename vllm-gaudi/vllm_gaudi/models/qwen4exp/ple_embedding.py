# SPDX-License-Identifier: Apache-2.0
"""Host-resident PLE n-gram embedding lookup for the Qwen4Exp HPU port.

Storage layout (from the checkpoint, validated): the per-layer n-gram table is
split into 128 row shards; each shard stores

  ``ngram_embedding.shard_<i>.weight``      U8 [rows, 80]  packed E2M1 (2/byte)
  ``ngram_embedding.shard_<i>.weight_scale`` F8_E4M3 [rows, 10]  per-16 scales

so one row is exactly **90 bytes** (80 packed weights + 10 E4M3 scale bytes)
and decodes to 160 NVFP4 values. The real table is 128 shards x 2,500,012 rows
= 320,001,536 rows = 28,800,138,240 B = **26.82 GiB**, and it lives in host
RAM as packed NVFP4 -- never as a device tensor, never decoded (the per-step
gather/H2D/dequant path is ``ple_runtime.PLENGramEmbedding.lookup``).

Real-table loading (:func:`safetensors_mmap`) is np.memmap-style streaming
straight from the checkpoint shards: the safetensors JSON header gives each
tensor's ``data_offsets``, a read-only ``np.memmap`` window is wrapped with
``torch.from_numpy``, and no full copy of a 225 MB shard -- let alone the
26.82 GiB table -- is ever made. Pages enter the page cache on first touch,
which is STRATA's ``mmap`` residency mode; pinning remains an explicit
opt-in per shard (see ``PLEShardTable``).

HPU strategy (vs upstream's CUDA UVA kernel): the *packed rows* move, not the
table. A lookup gathers 90 B rows on the CPU (``index_select``), stages them
in a pinned buffer, transfers once, and dequantizes through the same
cached-LUT path the expert planes use (``modelopt_csf.dequant_nvfp4``) on the
destination device; on a CPU host the identical function dequantizes the host
rows -- the CPU twin of the device op, chosen by the tensors' device. Gaudi's
slice node rejects the strided uint8 nibble expand once the device is busy,
and a decoded row is 320 B vs 90 B packed, so moving packed rows plus a
device-side LUT decode is both smaller and graph-safe.
"""

from __future__ import annotations

import json
import struct
import warnings
from pathlib import Path

import torch

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _e4m3_lut() -> torch.Tensor:
    codes = torch.arange(256, dtype=torch.int32)
    sign = torch.where((codes & 0x80) != 0, -1.0, 1.0)
    exp = (codes >> 3) & 0x0F
    man = codes & 0x07
    val = sign * (1.0 + man / 8.0) * torch.exp2(exp.float() - 7.0)
    val = torch.where(exp == 0, sign * (man / 8.0) * torch.exp2(torch.tensor(-6.0)), val)
    return torch.where((codes & 0x7F) == 0x7F, torch.tensor(float("nan")), val).float()


E2M1_LUT = torch.tensor(_E2M1_VALUES, dtype=torch.float32)
E4M3_LUT = _e4m3_lut()
BLOCK = 16  # NVFP4 group size
#: One shard row: 80 packed E2M1 bytes (160 values) + 10 E4M3 block scales.
ROW_BYTES = 80 + 160 // BLOCK


def _safetensors_header(path: Path) -> tuple[dict, int]:
    """``(tensor entries, data-region offset)`` of a safetensors shard."""
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header, 8 + length


def safetensors_mmap(path: str | Path, name: str) -> torch.Tensor:
    """Zero-copy, read-only ``np.memmap`` view of one safetensors tensor.

    The header's ``data_offsets`` locate the tensor's bytes inside the shard;
    a read-only memory-mapped window of exactly that byte range is wrapped
    with ``torch.from_numpy``, so a 225 MB PLE shard streams from the page
    cache without a full-copy read (a 128-shard load touches 26.82 GiB of
    file-backed pages, never 26.82 GiB of freshly allocated buffers). Only
    the byte-sized dtypes the PLE tables use are accepted; ``F8_E4M3`` planes
    come back as their ``uint8`` code view, which is what the LUT decode
    indexes.
    """
    import numpy as np

    header, base = _safetensors_header(Path(path))
    if name not in header:
        raise KeyError(f"{name} is not in {path}")
    entry = header[name]
    dtype = str(entry["dtype"]).upper()
    if dtype not in ("U8", "F8_E4M3"):
        raise ValueError(f"{name}: PLE mmap streaming supports U8/F8_E4M3, got {dtype}")
    begin, end = (int(v) for v in entry["data_offsets"])
    window = np.memmap(path, dtype=np.uint8, mode="r", offset=base + begin, shape=(end - begin, ))
    with warnings.catch_warnings():
        # The window is read-only by design: the table is never written.
        warnings.simplefilter("ignore", UserWarning)
        return torch.from_numpy(window.reshape(tuple(entry["shape"])))


class PLEShardTable:
    """One 128th of a PLE layer table, resident on the host as packed NVFP4.

    ``packed`` U8 ``[rows, 80]`` and ``scales`` U8 ``[rows, 10]`` (the byte
    view of the checkpoint's F8_E4M3 plane) keep the checkpoint residency:
    one row = 90 bytes = 160 values. Construction accepts either in-memory
    tensors or :func:`safetensors_mmap` windows; ``pin_memory=True`` makes
    exactly one pinned copy per shard, and leaving it off keeps the table
    purely file-backed (the production choice -- pinning all 26.82 GiB is a
    host-lock hazard, not a speed-up, since the per-step transfer stages
    through its own small pinned buffer).
    """

    def __init__(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        *,
        global_scale: float = 1.0,
        pin_memory: bool = True,
    ) -> None:
        if packed.ndim != 2 or scales.ndim != 2:
            raise ValueError("packed and scales must be 2-D [rows, cols]")
        rows, width = packed.shape
        if width * 2 != scales.shape[1] * BLOCK:
            raise ValueError(
                f"scale columns ({scales.shape[1]}) do not cover {width * 2} values"
            )
        if scales.shape[0] != rows:
            raise ValueError("packed and scales row counts differ")
        self.rows = rows
        self.embed_dim = width * 2
        if packed.device.type == "cpu" and pin_memory:
            packed = packed.pin_memory()
            scales = scales.pin_memory()
        # Real shards store the scale plane as float8 codes; view them as
        # bytes so lookups index the LUT by code. Converting the float8
        # VALUES to integers instead would decode garbage (e.g. code 0x5F
        # is 60.0, a completely different scale).
        if scales.dtype != torch.uint8:
            scales = scales.view(torch.uint8)
        self.packed = packed.contiguous()          # U8 [rows, width]
        self.scales = scales.contiguous()          # U8 [rows, width*2/16]
        self.global_scale = float(global_scale)

    @classmethod
    def from_safetensors(
        cls,
        shard_path: str | Path,
        weight_name: str,
        scale_name: str,
        *,
        global_scale: float = 1.0,
        pin_memory: bool = False,
    ) -> "PLEShardTable":
        """Build a shard table by mmap-streaming two tensors, no full copy."""
        return cls(
            safetensors_mmap(shard_path, weight_name),
            safetensors_mmap(shard_path, scale_name),
            global_scale=global_scale,
            pin_memory=pin_memory,
        )

    def lookup_packed(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather packed rows and scale rows for ``ids`` from host memory."""
        if ids.device.type != "cpu":
            ids = ids.cpu()
        packed_rows = self.packed.index_select(0, ids.long())
        scale_rows = self.scales.index_select(0, ids.long())
        return packed_rows, scale_rows

    @staticmethod
    def dequant(
        packed_rows: torch.Tensor,
        scale_rows: torch.Tensor,
        global_scale: float,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Decode gathered packed rows to ``[..., embed_dim]`` on any device.

        One code path for host and device: this is
        ``modelopt_csf.dequant_nvfp4``, whose E2M1 and E4M3 tables are cached
        per device, so the *tensors'* device decides where the decode runs.
        A CPU test drives the exact same arithmetic the post-H2D device
        decode executes (the CPU twin of the device op); nothing here ever
        builds a host table for device bytes.
        """
        from vllm_gaudi.ops import modelopt_csf

        lead = packed_rows.shape[:-1]
        packed2d = packed_rows.reshape(-1, packed_rows.shape[-1]).contiguous()
        scales2d = scale_rows.reshape(-1, scale_rows.shape[-1]).contiguous()
        out = modelopt_csf.dequant_nvfp4(
            packed2d,
            scales2d,
            global_scale,
            dtype=torch.float32,
        )
        out = out.reshape(*lead, -1).to(dtype)
        if device is not None:
            out = out.to(device)
        return out

    def lookup(
        self,
        ids: torch.Tensor,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Host gather -> dequant (host or device twin) -> optional transfer."""
        packed_rows, scale_rows = self.lookup_packed(ids)
        decoded = self.dequant(packed_rows, scale_rows, self.global_scale, dtype, device)
        return decoded


__all__ = ["PLEShardTable", "E2M1_LUT", "E4M3_LUT", "ROW_BYTES", "safetensors_mmap"]
