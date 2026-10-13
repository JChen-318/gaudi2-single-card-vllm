# SPDX-License-Identifier: Apache-2.0

import os
from contextlib import nullcontext
from typing import ContextManager

import torch


_PROFILE_STAGES = os.getenv("VLLM_DSV4_PROFILE_STAGES", "0").lower() in (
    "1",
    "true",
)


def dsv4_profile_stage(name: str) -> ContextManager:
    if not _PROFILE_STAGES:
        return nullcontext()
    return torch.profiler.record_function(f"dsv4::{name}")
