"""Warmup compile-time breakdown (VLLM_HPU_COMPILE_TIMES) with stand-ins for the Habana backend modules."""
import logging
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v6_support  # noqa: E402

kiro = v6_support._load("v1/worker/kiro_v11.py", "kiro_v11_timers")


def _fake_backend(monkeypatch):
    compilers = types.ModuleType("compilers")
    calls = []
    for name in kiro.CompileTimers.PASSES:
        setattr(compilers, name, lambda *a, _n=name, **k: calls.append(_n))

    def hpu_compiler_inner(graph, inputs):
        for name in kiro.CompileTimers.PASSES:
            getattr(compilers, name)(graph)  # looked up on the module at call time, like the real backend
        return "compiled"

    compilers.hpu_compiler_inner = hpu_compiler_inner
    recipe = types.ModuleType("_recipe_compiler_C")
    recipe.graph_compile = lambda *a, **k: 7
    package = types.ModuleType("habana_frameworks.torch.dynamo.compile_backend")
    package.compilers, package._recipe_compiler_C = compilers, recipe
    for name in ("habana_frameworks", "habana_frameworks.torch", "habana_frameworks.torch.dynamo"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch.dynamo.compile_backend", package)
    return compilers, recipe, calls


def test_wraps_backend_passes_and_synapse_compile(monkeypatch, caplog):
    compilers, recipe, calls = _fake_backend(monkeypatch)
    timers = kiro.CompileTimers()
    timers.install()
    assert timers.installed
    assert compilers.hpu_compiler_inner("g", []) == "compiled" and recipe.graph_compile() == 7
    compilers.hpu_compiler_inner("g", [])
    assert calls == list(kiro.CompileTimers.PASSES) * 2
    assert timers.stats["habana backend passes (total)"][0] == 2
    assert timers.stats["  partition_module"][0] == 2
    assert timers.stats["synapse graph_compile (recipe cache)"][0] == 1
    with caplog.at_level(logging.INFO):
        timers.report(12.0)
    assert "Compile time breakdown over warmup (12 s)" in caplog.text


def test_missing_backend_leaves_timers_off(monkeypatch):
    monkeypatch.setitem(sys.modules, "habana_frameworks", None)
    timers = kiro.CompileTimers()
    timers.install()
    assert not timers.installed
