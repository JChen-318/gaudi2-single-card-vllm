"""Image input (VISION=1): the bucketed vision tower's padding and the request guard's image switch.

    PYTHONPATH=vllm-gaudi python -m pytest -q validation/core/test_vision_input.py
"""
import asyncio
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(os.environ.get("VLLM_GAUDI_SRC", ROOT / "vllm-gaudi" / "vllm_gaudi"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


buckets_mod = _load(SRC / "models" / "qwen3_vision_buckets.py", "qwen3_vision_buckets_under_test")


def test_bucket_choice(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_VISION_BUCKETS", "1024, 256,576")
    sizes = buckets_mod.buckets()
    assert sizes == [256, 576, 1024]
    assert buckets_mod.bucket_for(256, sizes) == 256
    assert buckets_mod.bucket_for(260, sizes) == 576
    assert buckets_mod.bucket_for(1024, sizes) == 1024
    assert buckets_mod.bucket_for(2000, sizes) == 2000  # larger than every bucket: unpadded


def test_padding_mask_is_block_diagonal():
    mask = buckets_mod.padding_mask(3, 5, "cpu")[0, 0]
    expected = torch.tensor([[1, 1, 1, 0, 0]] * 3 + [[0, 0, 0, 1, 1]] * 2, dtype=torch.bool)
    assert torch.equal(mask, expected)


@pytest.mark.parametrize("real,size", [(256, 576), (100, 256), (64, 64)])
def test_padded_masked_attention_equals_unpadded(real, size):
    """What the tower relies on: real rows of masked attention over [real | padding] equal plain attention over
    the real rows, whatever the padding holds (zeros after the identity rotation here)."""
    gen = torch.Generator().manual_seed(real)
    q, k, v = (torch.randn(1, 4, real, 32, generator=gen) for _ in range(3))
    reference = F.scaled_dot_product_attention(q, k, v)
    pad = (0, 0, 0, size - real)
    out = F.scaled_dot_product_attention(F.pad(q, pad), F.pad(k, pad), F.pad(v, pad),
                                         attn_mask=buckets_mod.padding_mask(real, size, "cpu"))
    assert torch.allclose(out[:, :, :real], reference, atol=1e-5)
    assert torch.isfinite(out).all()


def test_install_is_off_without_buckets(monkeypatch):
    monkeypatch.delenv("VLLM_HPU_VISION_BUCKETS", raising=False)
    assert buckets_mod.install() is False


# ---------------------------------------------------------------------------------------------- request guard
starlette = pytest.importorskip("starlette")
guard_mod = _load(SRC.parent / "dflash2_request_guard.py", "dflash2_request_guard_under_test")


def _status(payload, allow):
    os.environ["VLLM_HPU_ALLOW_MULTIMODAL"] = "1" if allow else "0"
    reached = []

    async def app(scope, receive, send):
        reached.append(True)

    guard = guard_mod.DFlash2RequestGuard(app)
    sent = []
    body = json.dumps(payload).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []}
    asyncio.run(guard(scope, receive, send))
    if reached:
        return 200
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


IMAGE = {"messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}, {"type": "text", "text": "what?"}]}]}
VIDEO = {"messages": [{"role": "user", "content": [
    {"type": "video_url", "video_url": {"url": "http://x/v.mp4"}}, {"type": "text", "text": "what?"}]}]}
TEXT = {"messages": [{"role": "user", "content": "hi"}]}


def test_guard_images_follow_the_switch():
    try:
        assert _status(IMAGE, allow=False) == 400
        assert _status(IMAGE, allow=True) == 200
        assert _status(VIDEO, allow=True) == 400
        assert _status(TEXT, allow=False) == 200
        assert _status({**IMAGE, "logprobs": True}, allow=True) == 400  # other checks unchanged
    finally:
        os.environ.pop("VLLM_HPU_ALLOW_MULTIMODAL", None)
