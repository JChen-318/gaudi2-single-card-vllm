"""Steady-decode guard skipping (VLLM_HPU_GUARD_SKIP): round keys, the warm/recheck cycle and the
retry of a region that has to recompile under the skip stance. CPU, ``aot_eager`` backend."""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import v6_support  # noqa: E402

kiro = v6_support._load("v1/worker/kiro_v11.py", "kiro_v11_guard")
torch._dynamo.config.cache_size_limit = 64


class Inner(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)

    def forward(self, x):
        return torch.relu(self.lin(x))


class Outer(torch.nn.Module):

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.lin = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.inner(x) + self.lin(x)


def _compile(obj, skip):
    return kiro.protect(torch.compile(obj, backend="aot_eager", dynamic=False), skip)


def _graphs():
    from torch._dynamo.utils import counters
    return counters["stats"]["unique_graphs"]


@pytest.fixture(autouse=True)
def _reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def test_warm_skip_recheck_cycle_and_results():
    skip = kiro.GuardSkip(warm=2, recheck=5)
    outer = _compile(Outer(_compile(Inner(), skip)), skip)
    fn = _compile(lambda a: a * 3, skip)
    inputs = {"A": torch.randn(2, 8), "B": torch.randn(3, 8)}
    with torch.no_grad():
        ref = {name: (outer(x), fn(x)) for name, x in inputs.items()}
        graphs = _graphs()
        pattern = []
        for step in range(14):
            name = "A" if step % 2 == 0 else "B"
            with skip.round(("k", name)) as skipped:
                out, out_fn = outer(inputs[name]), fn(inputs[name])
            assert torch.equal(out, ref[name][0]) and torch.equal(out_fn, ref[name][1])
            pattern.append("S" if skipped else "g")
    # Two guarded rounds per key qualify it; after 4 skipped rounds one guarded recheck.
    assert "".join(pattern) == "ggggSSSSSSSSgg"
    assert _graphs() == graphs  # nested protected module stays inlined: no new graphs, no breaks
    assert skip.stats["keys"] == 2 and skip.stats["retried"] == 0


def test_new_graph_resets_the_key():
    skip = kiro.GuardSkip(warm=2)
    fn = _compile(lambda a: a + 1, skip)
    with torch.no_grad():
        with skip.round("k"):
            fn(torch.ones(2))
        with skip.round("k"):
            fn(torch.ones(4))  # compiles: the key is not clean
        assert skip.clean["k"] == 0
        with skip.round("k"):
            fn(torch.ones(4))
        with skip.round("k"):
            fn(torch.ones(4))
        with skip.round("k") as skipped:
            assert skipped


def test_unseen_input_under_skip_is_retried_with_full_guards():
    skip = kiro.GuardSkip(warm=1)
    outer = _compile(Outer(Inner()), skip)
    a, c = torch.randn(2, 8), torch.randn(5, 8)
    with torch.no_grad():
        outer(a)
        outer(torch.randn(3, 8))  # two entries: the shape guard is a differentiating guard
        with skip.round("k"):
            outer(a)
        with skip.round("k") as skipped:
            out = outer(c)
        expect = outer(c)
    assert skipped and torch.equal(out, expect)
    assert skip.stats["retried"] == 1 and skip.clean["k"] == 0


def test_other_runtime_errors_propagate():
    skip = kiro.GuardSkip()

    def boom(x):
        raise RuntimeError("unrelated failure")

    wrapped = kiro.protect(boom, skip)
    with pytest.raises(RuntimeError, match="unrelated"):
        wrapped(torch.ones(1))
    assert skip.stats["retried"] == 0


def test_disabled_or_missing_key_never_skips():
    skip = kiro.GuardSkip(warm=1)
    for _ in range(3):
        with skip.round(None) as skipped:
            assert not skipped
    skip.disabled = True
    for _ in range(3):
        with skip.round("k") as skipped:
            assert not skipped


@dataclass
class Meta:
    block_list: torch.Tensor
    seq_lens: list = field(default_factory=list)
    is_prompt: bool = False
    groups: tuple = ()


def test_shape_key_tracks_shapes_and_flags_not_values():
    a = Meta(torch.zeros(4, dtype=torch.int32), [5, 6], False, (Meta(torch.zeros(4)), ))
    b = Meta(torch.ones(4, dtype=torch.int32), [9, 10], False, (Meta(torch.ones(4)), ))
    assert kiro.shape_key(a) == kiro.shape_key(b)
    assert kiro.shape_key(a) != kiro.shape_key(Meta(torch.zeros(8, dtype=torch.int32), [5, 6]))
    assert kiro.shape_key(a) != kiro.shape_key(Meta(torch.zeros(4, dtype=torch.int64), [5, 6]))
    assert kiro.shape_key(a) != kiro.shape_key(Meta(torch.zeros(4, dtype=torch.int32), [5]))
    assert kiro.shape_key(a) != kiro.shape_key(Meta(torch.zeros(4, dtype=torch.int32), [5, 6], True))
    hash(kiro.shape_key(a))
