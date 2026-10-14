"""Check that the DFlash2 candidate-selector helpers trace into a single Dynamo graph.

`flashinfer_gaudi.dflash2.top_k` / `select_path` consult the tactics manifest. When the manifest was read through
an `lru_cache`-wrapped function, Dynamo looked through the decorator, tried to trace `importlib.resources` and
broke the graph at every call site (ten fragment frames per decode round on the device). The manifest is now a
module-level dict loaded at import, which Dynamo traces as a constant.

Runs on CPU: the HPU check is forced on and the native ops are absent, so the reference math still executes.

Usage (from the repository root): python validation/dynamo/check_selector_graph.py
"""
import os, shutil, sys, tempfile, warnings

warnings.filterwarnings("ignore")
import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
SRC = os.environ.get("FLASHINFER_GAUDI_SRC", os.path.join(REPO, "vllm-gaudi", "flashinfer_gaudi"))

with tempfile.TemporaryDirectory() as root:
    shutil.copytree(SRC, os.path.join(root, "flashinfer_gaudi"))
    # Only the dflash2 submodule is needed; the package __init__ imports the HPU stack.
    open(os.path.join(root, "flashinfer_gaudi", "__init__.py"), "w").write("")
    sys.path.insert(0, root)
    import flashinfer_gaudi.dflash2 as dflash2
    import flashinfer_gaudi._native as native

    dflash2._is_hpu = lambda tensor: True   # take the gated branches as on the device
    native.load_native_extensions()         # the server loads these long before the first decode
    steps, k = dflash2._QWEN38_DFLASH2_STEPS, dflash2._QWEN38_DFLASH2_TOP_K

    def region(candidate_ids, scores, logits):
        path = dflash2.select_path(candidate_ids, scores)
        values, ids = dflash2.top_k(logits, k, sorted=True, deterministic=True)
        return path, values, ids

    args = (torch.randint(0, 1000, (2, steps, k), dtype=torch.int64), torch.randn(2, steps, k, k),
            torch.randn(2 * steps, 4096))
    torch._dynamo.reset()
    report = torch._dynamo.explain(region)(*args)

print(f"graphs: {report.graph_count}  graph breaks: {report.graph_break_count}")
for reason in report.break_reasons[:5]:
    print("  ", " | ".join(line.strip() for line in str(reason.reason).strip().splitlines()[:3])[:220])
sys.exit(0 if report.graph_break_count == 0 else 1)
