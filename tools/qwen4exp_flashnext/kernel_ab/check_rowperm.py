"""Row-permuted gate/up planes: decode GEMV chain bit-identical, prefill chunk body close to natural."""
import os, sys, zlib
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
os.environ["GC_KERNEL_PATH"] = str(REPO / "vllm-gaudi/csrc/qwen4exp/build/libqwen4exp_gaudi2_kernels.so") + ":/usr/lib/habanalabs/libtpc_kernels.so"
os.environ.setdefault("PT_HPU_LAZY_MODE", "0")
sys.path.insert(0, str(REPO / "vllm-gaudi"))
import numpy as np, torch
import torch.nn.functional as F
import habana_frameworks.torch.core  # noqa
torch.ops.load_library(str(next((REPO / "vllm-gaudi/csrc/qwen4exp/pytorch").glob("hpu_qwen4exp_nvfp4_dequant_pt2*.so"))))
from vllm_gaudi.models.qwen4exp.packed_moe import k4_row_order, _moe_chunk_body_k4, _moe_chunk_body_k4rp, _moe_chunk_body
from vllm_gaudi.models.qwen4exp.decode_graph import _k4_layout
ops = torch.ops.custom_op
dev = torch.device("hpu")
E, H, I = 512, 2560, 640
def planes(name, rows, k):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    return [torch.from_numpy(a).to(dev) for a in (
        rng.integers(0, 256, (E, rows, k // 2), dtype=np.uint8), rng.integers(40, 70, (E, rows, k // 16), dtype=np.uint8),
        (rng.random(E, dtype=np.float32) * 1e-2 + 1e-3).astype(np.float32))]
gate, up, down = planes("gate", I, H), planes("up", I, H), planes("down", H, I)
perm = k4_row_order(I, dev)
gate_p = [gate[0].index_select(1, perm).contiguous(), gate[1].index_select(1, perm).contiguous(), gate[2]]
up_p = [up[0].index_select(1, perm).contiguous(), up[1].index_select(1, perm).contiguous(), up[2]]
rng = np.random.default_rng(5)
bad = 0
for T, S in ((1, 10), (2, 20)):
    ids = torch.from_numpy(rng.choice(E, S, replace=False).astype(np.int32)).to(dev)
    x = torch.from_numpy((rng.standard_normal((T, H)) * .5).astype(np.float32)).bfloat16().to(dev)
    act = ops.custom_qwen4exp_nvfp4_indexed_swiglu_gemv_k4_bf16_gaudi2(ids, *gate, *up, _k4_layout(x))
    ref = ops.custom_qwen4exp_nvfp4_indexed_gemv_k4_bf16_gaudi2(ids, down[0], down[1], _k4_layout(act), down[2])
    act_p = ops.custom_qwen4exp_nvfp4_indexed_swiglu_gemv_k4_bf16_gaudi2(ids, *gate_p, *up_p, _k4_layout(x))
    same_act = torch.equal(act_p.cpu(), _k4_layout(act).cpu())
    got = ops.custom_qwen4exp_nvfp4_indexed_gemv_k4_bf16_gaudi2(ids, down[0], down[1], act_p, down[2])
    same = torch.equal(got.cpu().view(torch.int16), ref.cpu().view(torch.int16))
    bad += not (same and same_act)
    print(f"decode T={T} S={S}: swiglu out == k4_layout(natural): {same_act}; down bits identical: {same}")
for n_c, cap in ((64, 64), (64, 512), (16, 128)):
    ids = torch.from_numpy(rng.choice(E, n_c, replace=False).astype(np.int32)).to(dev)
    xb = (torch.randn(n_c, cap, H) * 0.5).bfloat16().to(dev)
    nat = _moe_chunk_body(xb, ids, *gate, *up, *down).float().cpu()
    k4 = _moe_chunk_body_k4(_k4_layout(xb.view(-1, H)).view(n_c, cap, H), ids, *gate, *up, *down).float().cpu()
    rp = _moe_chunk_body_k4rp(_k4_layout(xb.view(-1, H)).view(n_c, cap, H), ids, *gate_p, *up_p, *down).float().cpu()
    scale = nat.abs().max()
    d_k4 = float((k4 - nat).abs().max() / scale); d_rp = float((rp - nat).abs().max() / scale)
    f_k4 = float((k4 != nat).float().mean()) * 100; f_rp = float((rp != nat).float().mean()) * 100
    ok = d_rp < 2e-2
    bad += not ok
    print(f"chunk n={n_c} cap={cap}: k4 vs natural max rel {d_k4:.2e} ({f_k4:.2f}% differ); k4rp vs natural {d_rp:.2e} ({f_rp:.2f}% differ)")
print("rowperm check:", "PASS" if not bad else f"FAIL {bad}")
