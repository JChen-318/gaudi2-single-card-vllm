# SPDX-License-Identifier: Apache-2.0
"""CPU checks for vllm_gaudi/omni/regional_compile.py with torch.compile's "eager" backend."""
import importlib
import sys
import types

import pytest
import torch
from torch import nn


@pytest.fixture
def rc(monkeypatch):
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda name: types.SimpleNamespace(info=lambda *a, **k: None,
                                                                warning=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_mod)
    monkeypatch.delitem(sys.modules, "vllm_gaudi.omni.regional_compile", raising=False)
    monkeypatch.setenv("VLLM_GAUDI_OMNI_HPU_COMPILE_BACKEND", "eager")
    return importlib.import_module("vllm_gaudi.omni.regional_compile")


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x):
        return torch.nn.functional.gelu(self.lin(x)) + x


class DiT(nn.Module):
    _repeated_blocks = ["Block"]

    def __init__(self, n=3):
        super().__init__()
        self.blocks = nn.ModuleList(Block() for _ in range(n))
        self.head = nn.Linear(8, 8)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def test_compiles_every_repeated_block_and_keeps_results(rc):
    torch.manual_seed(0)
    pipe = nn.Module()
    pipe.transformer = DiT()
    x = torch.randn(4, 8)
    ref = pipe.transformer(x)
    assert rc.compile_repeated_blocks(pipe) == 3
    assert all(getattr(b, "_vllm_gaudi_compiled", False) for b in pipe.transformer.blocks)
    assert not getattr(pipe.transformer.head, "_vllm_gaudi_compiled", False)
    assert torch.allclose(pipe.transformer(x), ref, atol=1e-6)
    assert rc.compile_repeated_blocks(pipe) == 0  # idempotent


def test_hook_owned_callable_is_the_one_compiled(rc):
    dit = DiT(n=1)
    block = dit.blocks[0]
    original = block.forward
    block._omni_original_forward = original
    dispatcher = lambda *a, **k: block._omni_original_forward(*a, **k)  # noqa: E731
    block.forward = dispatcher
    assert rc.compile_repeated_blocks(dit) == 1
    assert block.forward is dispatcher and block._omni_original_forward is not original


def test_non_module_pipeline_uses_transformer_attributes(rc):
    pipe = types.SimpleNamespace(transformer=DiT(n=2), _dit_modules=("transformer",))
    assert rc.compile_repeated_blocks(pipe) == 2


def test_no_repeated_blocks_runs_eagerly(rc):
    pipe = nn.Module()
    pipe.transformer = nn.Linear(4, 4)
    assert rc.compile_repeated_blocks(pipe) == 0


def test_disabled_by_default(rc, monkeypatch):
    monkeypatch.delenv("VLLM_GAUDI_OMNI_HPU_COMPILE", raising=False)
    assert not rc.compile_enabled()
    monkeypatch.setenv("VLLM_GAUDI_OMNI_HPU_COMPILE", "1")
    assert rc.compile_enabled()
