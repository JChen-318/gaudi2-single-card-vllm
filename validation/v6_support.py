"""Shared helpers for the v6-v10 offline tests (no vLLM install, no HPU).

Source files are executed from this repository (``VLLM_GAUDI_SRC`` overrides the package
directory). Only the few vLLM / plugin modules these files import are replaced by stubs.
"""
import ast
import logging
import os
import sys
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get("VLLM_GAUDI_SRC", HERE.parent / "vllm-gaudi" / "vllm_gaudi"))
REPO = HERE.parent


def _module(name):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
    return sys.modules[name]


def _stub():
    if getattr(sys.modules.get("vllm.logger"), "_v6_stub", False):
        return
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.models", "vllm_gaudi", "vllm_gaudi.v1",
                 "vllm_gaudi.v1.sample", "vllm_gaudi.v1.spec_decode", "vllm_gaudi.v1.worker"):
        _module(name)
    logger_mod = _module("vllm.logger")
    logger_mod.init_logger = lambda name: logging.getLogger(name)
    logger_mod._v6_stub = True
    utils = _module("vllm_gaudi.utils")
    utils.async_h2d_copy = lambda source, device=None, **_: source.clone()
    dflash = _module("vllm.model_executor.models.qwen3_dflash")

    class DFlashQwen3Model(torch.nn.Module):

        def embed_input_ids(self, input_ids):
            return self.embed_tokens(input_ids) * 2.0

        def forward(self, input_ids, positions, input_embeds=None):
            if input_embeds is None:
                input_embeds = self.embed_input_ids(input_ids)
            hidden_states, residual = input_embeds, None
            for layer in self.layers:
                hidden_states, residual = layer(positions=positions, hidden_states=hidden_states, residual=residual)
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states

    dflash.DFlashQwen3Model = DFlashQwen3Model


_stub()


def _load(path, name):
    """Execute ``vllm_gaudi/<path>`` as module ``name``."""
    source = (SRC / path).read_text(encoding="utf-8")
    module = types.ModuleType(name)
    module.__file__ = str(SRC / path)
    sys.modules[name] = module
    exec(compile(source, str(SRC / path), "exec"), module.__dict__)
    return module


def extract(path, names, namespace):
    """Execute only the named top-level functions/classes of a repository file into ``namespace``."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    missing = set(names) - {n.name for n in nodes}
    if missing:
        raise LookupError(f"{missing} not found in {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace
