"""Per-bucket warmup memory measurement without gc.collect() (VLLM_HPU_WARMUP_PHASE_GC)."""
import gc
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v6_support  # noqa: E402

kiro = v6_support._load("v1/worker/kiro_v11.py", "kiro_v11_warmup_gc")


def _fake_profiler(monkeypatch, readings):
    class HabanaMemoryProfiler:
        @staticmethod
        def current_device_memory_usage():
            return readings.pop(0)

    module = types.ModuleType("vllm_gaudi.extension.profiler")
    module.HabanaMemoryProfiler = HabanaMemoryProfiler
    monkeypatch.setitem(sys.modules, "vllm_gaudi.extension.profiler", module)


def test_measures_device_memory_delta_without_collecting(monkeypatch):
    _fake_profiler(monkeypatch, [1000, 1600])
    calls = []
    monkeypatch.setattr(gc, "collect", lambda *a, **k: calls.append(1) or 0)
    with kiro.NoGcMemoryProfiler() as prof:
        pass
    assert calls == []
    assert prof.initial_device_memory == 1000
    assert prof.final_device_memory == 1600
    assert prof.consumed_device_memory == 600


def test_exit_runs_when_the_body_raises(monkeypatch):
    _fake_profiler(monkeypatch, [5, 3])
    prof = kiro.NoGcMemoryProfiler()
    try:
        with prof:
            raise RuntimeError("bucket failed")
    except RuntimeError:
        pass
    assert prof.consumed_device_memory == -2


def test_runner_switch_and_phase_collections_are_wired():
    src = (v6_support.SRC / "v1/worker/hpu_model_runner.py").read_text(encoding="utf-8")
    body = src[src.index("    def warmup_graphs("):src.index("    def _warmup_tail_tokens(")]
    assert "phase_gc = gaudi_envs.VLLM_HPU_WARMUP_PHASE_GC" in body
    assert "kiro_v11.NoGcMemoryProfiler() if phase_gc else HabanaMemoryProfiler()" in body
    assert body.count("if phase_gc:\n            gc.collect()") == 2  # once before and once after the bucket loop
    envs = (v6_support.SRC / "envs.py").read_text(encoding="utf-8")
    assert 'environment_variables["VLLM_HPU_WARMUP_PHASE_GC"]' in envs
    assert '"VLLM_HPU_WARMUP_PHASE_GC", "0"' in envs  # off by default
