# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only helpers for the ModelOpt MIXED_PRECISION checks (Track B).

Everything here runs without a Gaudi device and without the Habana stack in
the import path: the code under test (``vllm_gaudi/ops/modelopt_csf.py``) is
loaded straight from its file, and the checkpoint is read through
``safetensors`` (headers are plain JSON; tensors are read on demand).

Environment overrides:

``QWEN4EXP_PLAIN_CHECKPOINT``
    the converted "plain" export (default
    ``~/models/qwen4exp-flashnext-plain``).
``QWEN4EXP_CSF_CHECKPOINT``
    the original NVIDIA export carrying the CSF scale tensors (default: the
    Hugging Face cache snapshot). Tests that need real CSF bytes skip when it
    is absent.
"""

from __future__ import annotations

import importlib.util
import json
import os
import struct
import sys
from pathlib import Path

import pytest

# Keep the HPU platform plugin out of the test process: the emulation helpers
# only need torch. Must be set before `import vllm`.
os.environ.setdefault("VLLM_PLUGINS", "none")

REPO_ROOT = Path(__file__).resolve().parents[2]
CSF_MODULE_PATH = REPO_ROOT / "vllm-gaudi" / "vllm_gaudi" / "ops" / "modelopt_csf.py"
HPU_MODELOPT_PATH = REPO_ROOT / "vllm-gaudi" / "vllm_gaudi" / "ops" / "hpu_modelopt.py"

PLAIN_CHECKPOINT = Path(
    os.environ.get("QWEN4EXP_PLAIN_CHECKPOINT", os.path.expanduser("~/models/qwen4exp-flashnext-plain")))


def _locate_csf_export() -> Path:
    """The original NVIDIA CSF export in the Hugging Face cache.

    ``QWEN4EXP_CSF_CHECKPOINT`` wins; otherwise the snapshot is discovered under
    ``models--nvidia--Qwen3.8-Flash-Next-FP4-Local`` (a re-download can change
    the commit hash, so the name is not hardcoded).  A path without an index is
    returned as-is and the tests skip on it.
    """
    override = os.environ.get("QWEN4EXP_CSF_CHECKPOINT")
    if override:
        return Path(override).expanduser()
    hub = Path(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser()
    model = hub / "models--nvidia--Qwen3.8-Flash-Next-FP4-Local"
    candidates = sorted((model / "snapshots").glob("*")) + sorted((model / "snapshots").glob("*/"))
    for candidate in candidates:
        if (candidate / "model.safetensors.index.json").is_file():
            return candidate
    return model / "snapshots" / "missing"


CSF_CHECKPOINT = _locate_csf_export()


def load_csf_module():
    """Load ``modelopt_csf.py`` by path (no ``vllm_gaudi`` package import)."""
    if not CSF_MODULE_PATH.is_file():
        pytest.skip(f"{CSF_MODULE_PATH} is missing")
    spec = importlib.util.spec_from_file_location("qwen4exp_modelopt_csf", CSF_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_checkpoint(root: Path, what: str) -> Path:
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        pytest.skip(f"{what} checkpoint not available at {root}")
    return root


def weight_map(root: Path) -> dict[str, str]:
    return json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]


def shard_path(root: Path, entry: str) -> Path:
    """Resolve an index entry; the two exports disagree on the ``tensors/`` prefix."""
    direct = root / entry
    if direct.is_file():
        return direct
    prefixed = root / "tensors" / entry
    if prefixed.is_file():
        return prefixed
    raise FileNotFoundError(f"cannot resolve shard {entry} under {root}")


def shard_header(path: Path) -> dict[str, dict]:
    """Read only the JSON header of a safetensors shard."""
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header


def layer_header(root: Path, base: str, index: dict[str, str] | None = None) -> dict[str, dict]:
    """Header of the shard holding ``base + '.weight'``."""
    index = index or weight_map(root)
    return shard_header(shard_path(root, index[f"{base}.weight"]))


def quant_config(root: Path) -> dict:
    return json.loads((root / "config.json").read_text())["quantization_config"]


class CheckpointSource:
    """Lazy multi-shard tensor source: ``keys()`` + ``get_tensor(name)``.

    This is the shape :func:`modelopt_csf.load_experts` accepts; it keeps the
    21 shard handles open instead of re-opening one per expert tensor.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self._index = weight_map(self.root)
        self._handles: dict[str, object] = {}

    def keys(self) -> list[str]:
        return list(self._index)

    def __contains__(self, name: object) -> bool:
        return name in self._index

    def get_tensor(self, name: str):
        entry = self._index[name]
        handle = self._handles.get(entry)
        if handle is None:
            from safetensors import safe_open

            handle = safe_open(str(shard_path(self.root, entry)), framework="pt", device="cpu")
            self._handles[entry] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        self._handles.clear()


@pytest.fixture(scope="module")
def csf():
    return load_csf_module()


@pytest.fixture(scope="module")
def plain_source():
    root = require_checkpoint(PLAIN_CHECKPOINT, "converted plain")
    source = CheckpointSource(root)
    yield source
    source.close()


@pytest.fixture(scope="module")
def csf_source():
    root = require_checkpoint(CSF_CHECKPOINT, "original CSF")
    source = CheckpointSource(root)
    yield source
    source.close()


def emulator():
    """The real ``nvfp4_emulation_utils`` module (vLLM core, CPU path)."""
    module = pytest.importorskip(
        "vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils",
        reason="vLLM core is required for the emulation reference",
    )
    return module


def reference_encoder(plane):
    """Encode a ``[rows, cols]`` E4M3-byte plane into the documented CSF layout.

    Reference encoder used only by the tests: ``base`` is the row minimum
    clamped so that ``base + 15`` stays inside a byte, the payload holds the
    4-bit delta per column (low nibble first), and every value outside the
    window becomes an exception record.
    """
    import torch

    plane = plane.to(torch.int32)
    rows, cols = plane.shape
    if rows % 16 or cols % 2:
        raise ValueError("rows must be a multiple of 16 and cols even")
    groups = rows // 16
    view = plane.reshape(groups, 16, cols)
    base = view.amin(dim=-1).clamp(max=255 - 15)
    delta = view - base.unsqueeze(-1)
    inside = (delta >= 0) & (delta <= 15)
    nibbles = torch.where(inside, delta, delta.clamp(0, 15)).to(torch.uint8)
    packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
    fixed = torch.cat([base.to(torch.uint8), packed.reshape(groups, 8 * cols)], dim=1)

    flat_inside = inside.reshape(-1)
    index = torch.nonzero(~flat_inside, as_tuple=False).flatten().to(torch.int64)
    value = plane.reshape(-1).index_select(0, index).to(torch.int64)
    records = (value << 24) | (((index >> 16) & 1) << 16) | (index & 0xFFFF)
    return fixed, records.to(torch.uint32)
