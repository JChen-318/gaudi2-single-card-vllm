# SPDX-License-Identifier: Apache-2.0
"""HPUModelRunner._execute_mm_encoder encodes only the items the scheduler scheduled (issue #2).

A follow-up turn of a conversation carries the earlier images as features whose tokens are already computed
(prefix-cache hit). They are not scheduled, their encoder-cache entries were freed, and they can arrive with
metadata only (image_grid_thw, no pixel_values). Encoding them made embed_multimodal return None and killed the
engine. The method is extracted from the source and run on CPU stand-ins."""
import ast
import os
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.environ.get("VLLM_GAUDI_SRC", os.path.join(HERE, "..", "..", "vllm-gaudi", "vllm_gaudi"))
SRC = open(os.path.join(PKG, "v1", "worker", "hpu_model_runner.py"), encoding="utf-8").read()


def _method():
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.FunctionDef) and node.name == "_execute_mm_encoder":
            code = "\n".join(line[4:] for line in ast.get_source_segment(SRC, node, padded=True).splitlines())
            break
    else:
        raise AssertionError("_execute_mm_encoder not found")

    def group_and_batch_mm_kwargs(mm_kwargs, device=None, pin_memory=False):
        for modality, data in mm_kwargs:
            yield modality, 1, data

    def sanity_check_mm_encoder_outputs(outputs, expected_num_items):
        assert isinstance(outputs, (list, tuple, torch.Tensor)), f"got {type(outputs)}"
        assert len(outputs) == expected_num_items

    ns = {"torch": torch, "MultiModalKwargsItem": object, "PlaceholderRange": object,
          "group_and_batch_mm_kwargs": group_and_batch_mm_kwargs,
          "sanity_check_mm_encoder_outputs": sanity_check_mm_encoder_outputs}
    exec(code, ns)
    return ns["_execute_mm_encoder"]


def _feature(i, pixels):
    data = {"image_grid_thw": torch.tensor([[1, 4, 4]])}
    if pixels:
        data["pixel_values"] = torch.full((16, 8), float(i))
    return types.SimpleNamespace(identifier=f"img{i}", modality="image", data=data,
                                 mm_position=types.SimpleNamespace(is_embed=None))


def _runner(features):
    def embed_multimodal(**kwargs):
        if "pixel_values" not in kwargs:  # what Qwen3-VL does for a group without pixels
            return None
        return [kwargs["pixel_values"][:4]]

    return types.SimpleNamespace(requests={"r": types.SimpleNamespace(mm_features=features)}, encoder_cache={},
                                 model=types.SimpleNamespace(embed_multimodal=embed_multimodal), device="cpu",
                                 pin_memory=False)


def test_only_scheduled_items_are_encoded():
    run = _method()
    runner = _runner([_feature(0, False), _feature(1, False), _feature(2, False), _feature(3, True)])
    sched = types.SimpleNamespace(scheduled_encoder_inputs={"r": [3]})
    run(runner, sched, ["r"])  # used to raise: embed_multimodal returned None for images 0-2
    assert list(runner.encoder_cache) == ["img3"]
    assert torch.equal(runner.encoder_cache["img3"], torch.full((4, 8), 3.0))


def test_nothing_scheduled_encodes_nothing():
    run = _method()
    runner = _runner([_feature(0, True)])
    run(runner, types.SimpleNamespace(scheduled_encoder_inputs={}), ["r"])
    assert runner.encoder_cache == {}


def test_cached_and_data_less_items_are_skipped():
    run = _method()
    feats = [_feature(0, True), _feature(1, True)]
    feats[1].data = None
    runner = _runner(feats)
    runner.encoder_cache["img0"] = torch.zeros(4, 8)
    run(runner, types.SimpleNamespace(scheduled_encoder_inputs={"r": [0, 1]}), ["r"])
    assert list(runner.encoder_cache) == ["img0"] and not runner.encoder_cache["img0"].any()


def test_other_batch_requests_are_left_alone():
    run = _method()
    runner = _runner([_feature(0, True)])
    runner.requests["other"] = types.SimpleNamespace(mm_features=[_feature(9, True)])
    run(runner, types.SimpleNamespace(scheduled_encoder_inputs={"r": [0], "other": [0]}), ["r"])
    assert list(runner.encoder_cache) == ["img0"]
