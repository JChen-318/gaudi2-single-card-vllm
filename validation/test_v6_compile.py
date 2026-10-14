"""v6 compile checks (no vLLM, no HPU): private code objects, group-0 embedding, draft stack."""
import ast
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from v6_support import SRC, _load  # noqa: E402

kiro_v6 = _load("v1/worker/kiro_v6.py", "kiro_v6_under_test")
draft_stack = _load("v1/spec_decode/dflash_draft_stack.py", "dflash_draft_stack_under_test")


def _group_class():
    tree = ast.parse((SRC / "models/qwen3_next.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HpuQwen3DecoderLayerGroup")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "qwen3_next.py", "exec"), namespace)
    return namespace["HpuQwen3DecoderLayerGroup"]


class _Layer(torch.nn.Module):

    def __init__(self, idx):
        super().__init__()
        self.lin = torch.nn.Linear(16, 16)
        self.idx = idx

    def forward(self, x):
        return super().__call__ is not None and self.lin(x) + self.idx


def test_private_code_module_and_methods():
    a, b = _Layer(1), _Layer(2)
    x = torch.randn(3, 16)
    ra, rb = a(x), b(x)
    kiro_v6.with_private_code(a)
    kiro_v6.with_private_code(b)
    assert type(a) is not type(b) and isinstance(a, _Layer) and type(a).__name__ == "_Layer"
    assert type(a).forward.__code__ is not type(b).forward.__code__
    assert torch.equal(a(x), ra) and torch.equal(b(x), rb)
    ca = torch.compile(a, backend="eager")
    cb = torch.compile(b, backend="eager")
    assert torch.equal(ca(x), ra) and torch.equal(cb(x), rb)
    m = kiro_v6.with_private_code(a.forward)
    assert isinstance(m, types.MethodType) and torch.equal(m(x), ra)
    assert torch.equal(torch.compile(m, backend="eager")(x), ra)
    c = _Layer(3)
    c.forward = types.MethodType(_Layer.forward, c)
    kiro_v6.with_private_code(c)
    assert c.__dict__["forward"].__func__.__code__ is not _Layer.forward.__code__
    assert torch.equal(c(x), _Layer.forward(c, x))
    f = kiro_v6.with_private_code(torch.nn.functional.relu)  # builtin-ish: returned unchanged or copied
    assert torch.equal(f(x), torch.relu(x))


def test_private_code_separates_dynamo_caches():
    from torch.profiler import ProfilerActivity, profile
    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.accumulated_cache_size_limit = 4096

    def run(groups, x):
        for _ in range(2):
            for g in groups:
                g(x)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            for _ in range(10):
                for g in groups:
                    g(x)
        times = sorted(e.cpu_time_total for e in prof.events() if e.name == "TorchDynamo Cache Lookup")
        return times[len(times) // 2]

    x = torch.randn(4, 16)
    shared = [torch.compile(_Layer(i), backend="eager") for i in range(19)]
    private = [torch.compile(kiro_v6.with_private_code(_Layer(i)), backend="eager") for i in range(19)]
    t_shared, t_private = run(shared, x), run(private, x)
    print(f"lookup median shared={t_shared:.1f}us private={t_private:.1f}us")
    assert t_private < t_shared


class _Dec(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)

    def forward(self, positions, hidden_states, residual):
        residual = hidden_states + residual
        return self.lin(residual) * 0.5, residual


def test_group_embeds_first():
    group_cls = _group_class()
    torch.manual_seed(0)
    embed = torch.nn.Embedding(50, 8)
    group = group_cls((_Dec(), _Dec()))
    ids = torch.randint(0, 50, (6, ))
    pos = torch.arange(6)
    h = embed(ids)
    ref = group(positions=pos, hidden_states=h, residual=torch.zeros_like(h))
    object.__setattr__(group, "_embed", embed)
    out = group(positions=pos, input_ids=ids)
    assert torch.equal(ref[0], out[0]) and torch.equal(ref[1], out[1])
    compiled = torch.compile(kiro_v6.with_private_code(group), backend="eager", fullgraph=True)
    out2 = compiled(positions=pos, input_ids=ids)
    assert torch.equal(ref[0], out2[0]) and torch.equal(ref[1], out2[1])
    out3 = compiled(positions=pos, hidden_states=h, residual=torch.zeros_like(h))
    assert torch.equal(ref[0], out3[0])


class _DraftLayer(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + hidden_states
        return torch.tanh(self.lin(residual)), residual


class _Norm(torch.nn.Module):

    def forward(self, x, residual):
        y = x + residual
        return y / y.norm(dim=-1, keepdim=True), y


def test_draft_stack_matches_upstream_forward():
    base = sys.modules["vllm.model_executor.models.qwen3_dflash"].DFlashQwen3Model

    class Fake(base):

        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(40, 8)
            self.layers = torch.nn.ModuleList([_DraftLayer() for _ in range(5)])
            self.norm = _Norm()

    torch.manual_seed(1)
    inner = Fake()
    root = torch.nn.Module()
    root.inner = inner
    ids = torch.randint(0, 40, (16, ))
    pos = torch.arange(16)
    ref = inner(ids, pos)
    draft_stack.install_draft_stack(root, lambda m: torch.compile(m, backend="eager", fullgraph=True))
    out = inner(ids, pos)
    assert torch.equal(ref, out)
    emb = inner.embed_input_ids(ids)
    assert torch.equal(inner(ids, pos, emb), ref)
    assert torch.equal(root(ids, pos) if False else inner(ids, pos), ref)
