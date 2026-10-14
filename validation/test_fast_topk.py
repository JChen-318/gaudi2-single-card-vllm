"""Two-stage top-k and sort-free top-k/top-p against the upstream implementations (CPU).

References are the repository's own sources: ``apply_top_k_top_p_pytorch`` from vllm-core and
``top_k_reference`` from flashinfer_gaudi, extracted with AST and run unchanged.
"""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import v6_support  # noqa: E402

ft = v6_support._load("v1/sample/fast_topk_topp.py", "vllm_gaudi.v1.sample.fast_topk_topp")
REF = v6_support.extract(v6_support.REPO / "vllm-core/vllm/v1/sample/ops/topk_topp_sampler.py",
                         ["apply_top_k_top_p_pytorch", "apply_top_k_only"], {"torch": torch})
apply_top_k_top_p_pytorch = REF["apply_top_k_top_p_pytorch"]
FIG = v6_support.extract(v6_support.SRC.parent / "flashinfer_gaudi/dflash2.py",
                         ["_is_hpu", "_require_integer", "top_k_reference"], {"torch": torch})
top_k_reference = FIG["top_k_reference"]
OPS = v6_support.extract(v6_support.SRC / "ops/hpu_dflash2.py", ["_chunked_top_k"], {"torch": torch})
_chunked_top_k = OPS["_chunked_top_k"]

VOCAB = 248320


def _logits(rows, seed, ties=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, VOCAB, generator=g) * 3
    return x.to(torch.bfloat16).to(torch.float32) if ties else x


def _kept(masked):
    return torch.isfinite(masked)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("k", [1, 20, 64])
def test_chunked_topk_values(seed, k):
    x = _logits(7, seed, ties=seed % 2 == 1)
    values, indices = ft.chunked_topk(x, k)
    assert torch.equal(values, x.topk(k, dim=-1).values)
    assert torch.equal(x.gather(1, indices.long()), values)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("k,p", [(20, 0.95), (20, None), (1, 0.5), (64, 0.9), (5, 1.0), (40, 0.99)])
def test_fast_matches_upstream_without_ties(seed, k, p):
    x = _logits(8, 100 + seed)
    kk = torch.full((8, ), k, dtype=torch.int32)
    pp = None if p is None else torch.full((8, ), p)
    ref = apply_top_k_top_p_pytorch(x.clone(), kk, pp)
    fast = ft.fast_top_k_top_p(x.clone(), kk, pp)
    assert torch.equal(_kept(ref), _kept(fast))
    assert torch.equal(torch.where(_kept(ref), ref, 0), torch.where(_kept(fast), fast, 0))


@pytest.mark.parametrize("seed", range(6))
def test_fast_with_ties_keeps_a_superset_of_boundary_ties(seed):
    x = _logits(8, 300 + seed, ties=True)
    kk = torch.full((8, ), 20, dtype=torch.int32)
    pp = torch.full((8, ), 0.95)
    ref = _kept(apply_top_k_top_p_pytorch(x.clone(), kk, pp))
    fast = _kept(ft.fast_top_k_top_p(x.clone(), kk, pp))
    assert not bool((ref & ~fast).any())
    extra = fast & ~ref
    for row in range(8):
        if extra[row].any():
            assert torch.all(x[row][extra[row]] == x[row][ref[row]].min())


@pytest.mark.parametrize("seed", range(4))
def test_candidate_topk_matches_reference(seed):
    sys.modules["vllm_gaudi.v1.sample.fast_topk_topp"] = ft
    exact = _logits(14, 900 + seed)  # float32: no ties
    rv, ri = top_k_reference(exact, 16, sorted=True, deterministic=True)
    cv, ci = _chunked_top_k(exact, 16)
    assert torch.equal(rv, cv) and torch.equal(ri.long(), ci.long())
