"""Pooled rejection-sampling RNG (VLLM_HPU_RNG_POOL): independence of the per-round draws, refills,
the per-request generator bypass and the hook in the HPU rejection sampler."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import v6_support  # noqa: E402

kiro = v6_support._load("v1/worker/kiro_v11.py", "vllm_gaudi.v1.worker.kiro_v11")
sampler_ns = v6_support.extract(
    v6_support.SRC / "v1/sample/hpu_rejection_sampler.py",
    ["_generate_rectangular_uniform_probs", "_generate_request_exponential_noise"],
    {"torch": torch, "profile_stage": lambda name: (lambda function: function)})


def _with_pool(rounds):
    kiro.RNG_POOL = kiro.RngPool(rounds) if rounds else None
    return kiro.RNG_POOL


def test_pool_off_inactive_or_seeded_falls_back():
    _with_pool(0)
    assert kiro.pooled_draw("uniform", (2, 8), torch.float32, "cpu", {}) is None
    pool = _with_pool(4)
    assert kiro.pooled_draw("uniform", (2, 8), torch.float32, "cpu", {0: torch.Generator()}) is None
    pool.active = False
    assert kiro.pooled_draw("uniform", (2, 8), torch.float32, "cpu", {}) is None
    _with_pool(0)


def test_draws_are_owned_distinct_and_refilled():
    pool = _with_pool(3)
    draws = [kiro.pooled_draw("uniform", (2, 8), torch.float32, "cpu", {}) for _ in range(7)]
    assert pool.stats == {"refills": 3, "draws": 7}
    for draw in draws:
        assert draw.shape == (2, 8) and draw.is_contiguous() and draw._base is None
    flat = torch.stack(draws).reshape(7, -1)
    assert len({tuple(row.tolist()) for row in flat}) == 7
    # Other shapes or kinds keep their own pools.
    kiro.pooled_draw("uniform", (1, 8), torch.float32, "cpu", {})
    kiro.pooled_draw("exponential", (2, 8), torch.float32, "cpu", {})
    assert pool.stats["refills"] == 5
    _with_pool(0)


def test_distributions():
    torch.manual_seed(0)
    _with_pool(16)
    uniform = torch.cat([kiro.pooled_draw("uniform", (4, 8), torch.float32, "cpu", {}).flatten() for _ in range(200)])
    exponential = torch.cat(
        [kiro.pooled_draw("exponential", (2, 4096), torch.float32, "cpu", {}).flatten() for _ in range(40)])
    assert 0.0 <= float(uniform.min()) and float(uniform.max()) < 1.0
    assert abs(float(uniform.mean()) - 0.5) < 0.01 and abs(float(uniform.var()) - 1 / 12) < 0.004
    assert float(exponential.min()) >= 0.0
    assert abs(float(exponential.mean()) - 1.0) < 0.01 and abs(float(exponential.var()) - 1.0) < 0.03
    _with_pool(0)


def test_rejection_sampler_hooks_use_the_pool_only_without_generators():
    pool = _with_pool(8)
    uniform = sampler_ns["_generate_rectangular_uniform_probs"](2, 8, [7, 7], {}, "cpu")
    noise = sampler_ns["_generate_request_exponential_noise"](2, 50, [7, 7], {}, "cpu", False)
    assert uniform.shape == (2, 8) and noise.shape == (2, 50) and noise.dtype == torch.float32
    assert pool.stats["draws"] == 2
    generator = torch.Generator().manual_seed(1)
    seeded = sampler_ns["_generate_rectangular_uniform_probs"](2, 8, [7, 7], {1: generator}, "cpu")
    again = sampler_ns["_generate_rectangular_uniform_probs"](2, 8, [7, 7], {1: torch.Generator().manual_seed(1)},
                                                             "cpu")
    assert pool.stats["draws"] == 2  # seeded requests keep the reference draw order
    assert torch.equal(seeded[1, :7], again[1, :7])
    fp64 = sampler_ns["_generate_request_exponential_noise"](1, 50, [7], {}, "cpu", True)
    assert fp64.dtype == torch.float64
    _with_pool(0)
