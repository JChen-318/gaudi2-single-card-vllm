"""Does Dynamo cache lookup cost grow when N module instances share one forward code object?
Measures the profiler's 'TorchDynamo Cache Lookup' event directly (no compute noise)."""
import sys, types, statistics, torch
import torch._dynamo
from torch.profiler import profile, ProfilerActivity

torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 8192
N = int(sys.argv[1]) if len(sys.argv) > 1 else 19
SUB = int(sys.argv[2]) if len(sys.argv) > 2 else 16


class Layer(torch.nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.lin = torch.nn.Linear(64, 64)
        self.norm = torch.nn.LayerNorm(64)
        self.layer_idx = idx          # per-instance constant, as in real decoder layers
        self.flag = None

    def forward(self, x):
        if self.flag is None:
            x = x + self.layer_idx
        return self.norm(self.lin(x))


class Group(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self._layers = torch.nn.ModuleList(Layer(base + i) for i in range(SUB))

    def forward(self, x):
        for l in self._layers:
            x = l(x)
        return x


def unique_code_class(i):
    f = Group.forward
    code = f.__code__.replace(co_name=f"forward_g{i}")
    fn = types.FunctionType(code, f.__globals__, f"forward_g{i}", f.__defaults__, f.__closure__)
    return type(f"Group{i}", (Group,), {"forward": fn})


def run(groups, x, rounds=30):
    for _ in range(3):
        for g in groups: g(x)
    with profile(activities=[ProfilerActivity.CPU]) as p:
        for _ in range(rounds):
            for g in groups: g(x)
    d = [e.cpu_time_total for e in p.events() if e.name == "TorchDynamo Cache Lookup"]
    return statistics.median(d), len(d)


x = torch.randn(8, 64)
with torch.inference_mode():
    shared = [torch.compile(Group(i * SUB), backend="eager", dynamic=False) for i in range(N)]
    uniq = [torch.compile(unique_code_class(i)(i * SUB), backend="eager", dynamic=False) for i in range(N)]
    for name, gs in (("shared code object", shared), ("unique code objects", uniq)):
        med, n = run(gs, x)
        print(f"{name:22s} N={N}: median lookup {med:7.1f} us  ({n} lookups)")
