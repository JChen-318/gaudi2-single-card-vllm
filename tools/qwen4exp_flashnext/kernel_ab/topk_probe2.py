from pathlib import Path
import os, sys, time, torch
os.environ.setdefault("PT_HPU_LAZY_MODE", "0")
import habana_frameworks.torch.core  # noqa
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "vllm-gaudi"))
from vllm_gaudi.models.qwen4exp.qsa_math import stable_topk_desc
dev = torch.device("hpu")
g = torch.Generator().manual_seed(1)
for cols in (2048, 8192, 16384, 32768, 65536):
    base = torch.randn(512, cols, generator=g)
    s = torch.relu(base)
    cut = torch.randint(cols // 2, cols, (512, 1), generator=g)
    s = s.masked_fill(torch.arange(cols)[None] >= cut, float("-inf")).to(dev)
    k = 512
    got = stable_topk_desc(s, k).cpu()
    ref = torch.argsort(s.cpu(), dim=-1, stable=True, descending=True)[:, :k]
    hpu_ref = torch.argsort(s, dim=-1, stable=True, descending=True)[:, :k].cpu()
    for name, fn in (("argsort", lambda: torch.argsort(s, dim=-1, stable=True, descending=True)[:, :k]),
                     ("topk", lambda: stable_topk_desc(s, k))):
        fn(); torch.hpu.synchronize(); t = time.perf_counter()
        for _ in range(5): fn()
        torch.hpu.synchronize(); ms = (time.perf_counter() - t) / 5 * 1000
        print(f"cols {cols:6d} {name:8s} {ms:8.2f} ms")
    print(f"  equal cpu-argsort {torch.equal(got, ref)}  equal hpu-argsort {torch.equal(got, hpu_ref)}")
