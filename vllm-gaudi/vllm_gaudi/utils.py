from functools import cache
import os
from vllm.config import ModelConfig
import vllm.utils.torch_utils as torch_utils
from vllm_gaudi.extension.runtime import get_config
import vllm.v1.core.sched.async_scheduler as _async_sched_module
from vllm_gaudi.v1.core.sched.hpu_async_scheduler import HPUAsyncScheduler
from typing import Any, Optional, TypeVar, Union
import torch
import habana_frameworks.torch as htorch
import numpy as np
import numpy.typing as npt
import math
import atexit
import json
import re
import time
import types
import vllm_gaudi.envs as gaudi_envs

T = TypeVar("T")
U = TypeVar("U")


@cache
def is_fake_hpu() -> bool:
    return os.environ.get("VLLM_USE_FAKE_HPU", "0") != "0"


@cache
def hpu_device_string():
    device_string = "hpu" if not is_fake_hpu() else "cpu"
    return device_string


@cache
def hpu_backend_string():
    backend_string = "hccl" if not is_fake_hpu() else "gloo"
    return backend_string


def has_quant_config(model_config: ModelConfig) -> bool:
    return model_config.quantization == "inc" or os.getenv("QUANT_CONFIG", None) is not None


def async_h2d_copy(source, dest_tensor=None, dtype=None, device="hpu"):
    """
    Asynchronously transfer data from host to device.

    Args:
        source: CPU tensor or raw data to transfer
        dest_tensor: Optional pre-allocated destination tensor
        dtype: Required if source is raw data
        device: Target device

    Returns:
        torch.Tensor on target device
    """
    if isinstance(source, torch.Tensor):
        if dest_tensor is not None:
            # Copy into pre-allocated destination tensor
            return dest_tensor.copy_(source, non_blocking=True)
        # Create new device tensor and copy
        assert source.device.type == "cpu", "Source tensor must be on CPU for asynchronous transfer"
        target = torch.empty_like(source, device=device)
        return target.copy_(source, non_blocking=True)
    # Create tensor from data and transfer to device
    if dtype is None:
        raise ValueError("dtype must be specified when source is not a tensor")
    cpu_tensor = torch.tensor(source, dtype=dtype, device="cpu")
    return cpu_tensor.to(device, non_blocking=True)


def async_h2d_update(source: torch.Tensor, dest: torch.Tensor, indices: list[int], device="hpu"):
    """
    Asynchronously update specific rows of a device tensor from a CPU tensor.

    Args:
        source: CPU tensor with data to copy
        dest: Device tensor to update
        indices: List of row indices in dest to update
        device: Target device
    """
    idx = torch.tensor(indices, dtype=torch.long, device=device)
    vals = source[indices].to(device, non_blocking=True)
    dest.index_copy_(0, idx, vals)


def getattr_nested(obj: Any, name: str, *default: Any) -> Any:
    """Like built-in getattr but supports dot-separated nested attributes.

    Examples:
        getattr_nested(obj, 'a.b.c')  is equivalent to  obj.a.b.c
        getattr_nested(obj, 'a.b', None)  returns None when any
            intermediate or final attribute is missing.

    Args:
        obj: Root object.
        name: Dot-separated attribute path.
        *default: Optional default returned when the attribute is missing.
            At most one default value may be provided (same contract as
            built-in ``getattr``).

    Raises:
        TypeError: If more than one default value is provided.
        AttributeError: If the attribute is missing and no default was given.
    """
    if len(default) > 1:
        raise TypeError(f"getattr_nested expected at most 3 arguments, got {2 + len(default)}")
    parts = name.split(".")
    try:
        for part in parts:
            obj = getattr(obj, part)
        return obj
    except AttributeError:
        if default:
            return default[0]
        raise


def setattr_nested(obj: Any, name: str, value: Any) -> None:
    """Like built-in setattr but supports dot-separated nested attributes.

    Examples:
        setattr_nested(obj, 'a.b.c', val)  is equivalent to  obj.a.b.c = val

    Args:
        obj: Root object.
        name: Dot-separated attribute path.  All parts except the last
            must already exist as attributes.
        value: Value to assign to the final attribute.

    Raises:
        AttributeError: If any intermediate attribute does not exist.
    """
    parts = name.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def make_ndarray_with_pad_align(
    x: list[list[T]],
    pad: T,
    dtype: npt.DTypeLike,
    *,
    max_len_align: int = 1024,
) -> npt.NDArray:
    """
    Make a padded array from 2D inputs.
    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    """
    # Unlike for most functions, map is faster than a genexpr over `len`
    max_len = max(map(len, x), default=0)
    max_len_aligned = math.ceil(max_len / max_len_align) * max_len_align
    padded_x = np.full((len(x), max_len_aligned), pad, dtype=dtype)

    for ind, blocktb in enumerate(x):
        assert len(blocktb) <= max_len_aligned
        padded_x[ind, :len(blocktb)] = blocktb

    return padded_x


def make_tensor_with_pad_align(
    x: list[list[T]],
    pad: T,
    dtype: torch.dtype,
    *,
    max_len_align: int = 1024,
    device: Optional[Union[str, torch.device]] = None,
    pin_memory: bool = False,
) -> torch.Tensor:
    """
    Make a padded tensor from 2D inputs.
    The padding is applied to the end of each inner list until it reaches
    max_len_aligned, max_len_aligned is max_len rounding to the nearest
    `max_len_align`.
    """
    np_dtype = torch_utils.TORCH_DTYPE_TO_NUMPY_DTYPE[dtype]
    padded_x = make_ndarray_with_pad_align(x, pad, np_dtype, max_len_align=max_len_align)

    tensor = torch.from_numpy(padded_x).to(device)
    if pin_memory:
        tensor = tensor.pin_memory()

    return tensor


if not htorch.utils.internal.is_lazy():

    def make_tensor_with_pad_hpu(
        x: list[list[T]],
        pad: T,
        dtype: torch.dtype,
        *,
        max_len: int | None = None,
        device: Optional[Union[str, torch.device]] = None,
        pin_memory: bool = False,
    ) -> torch.Tensor:
        """
        Make a padded tensor from 2D inputs.

        The padding is applied to the end of each inner list until it reaches
        `max_len`.

        HPU-compatible replacement for make_tensor_with_pad.
        Uses pure PyTorch (pad_sequence) instead of NumPy.
        """
        if not x:
            return torch.empty((0, 0), dtype=dtype, device=device)

        # 1. Convert python lists to CPU tensors first
        tensors = [torch.tensor(item, dtype=dtype, device="cpu") for item in x]

        # 2. Use pad_sequence (pure torch implementation)
        #    batch_first=True -> (Batch, Seq)
        tensor = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=pad)

        # 3. Handle max_len truncation or specific padding if requested
        if max_len is not None:
            batch_size, seq_len = tensor.shape
            if seq_len < max_len:
                padding_size = max_len - seq_len
                padding = torch.full((batch_size, padding_size), pad, dtype=dtype, device="cpu")
                tensor = torch.cat((tensor, padding), dim=1)
            elif seq_len > max_len:
                tensor = tensor[:, :max_len]

        # 4. Move to target device
        tensor = tensor.to(device)

        # 5. Pin memory if requested
        if pin_memory:
            tensor = tensor.pin_memory()

        return tensor

    torch_utils.make_tensor_with_pad = make_tensor_with_pad_hpu


def make_mrope_positions_tensor_with_pad(input_positions: list[list[int]], input_mrope_positions: list[list[list[int]]],
                                         max_prompt_len: int, pad: int) -> list[list[int]]:
    # If no mrope positions, returns a flatten (seq_len,)
    if all(mrope_position is None for mrope_position in input_mrope_positions):
        return torch_utils.make_tensor_with_pad(input_positions,
                                                max_len=max_prompt_len,
                                                pad=0,
                                                dtype=torch.long,
                                                device="cpu").flatten()
    # Otherwise, Qwen2.5-VL expects positions in a (3, seq_len)
    # we are going to pad each seq_data in the list
    # using either MRope values or regular position
    mrope_input_positions: list[list[int]] = [[] for _ in range(3)]
    for idx in range(3):
        for b_idx, input_mrope_position in enumerate(input_mrope_positions):
            positions = input_mrope_position[idx] if input_mrope_position is not None else input_positions[b_idx]
            padding_size = max_prompt_len - len(positions)
            assert padding_size >= 0
            padded_positions = positions + (max_prompt_len - len(positions)) * [pad]
            mrope_input_positions[idx].extend(padded_positions)
    return torch.tensor(mrope_input_positions, dtype=torch.long, device="cpu")


# Dynamo guard filters for a frozen serving model. Compiled regions are
# re-guarded on every call; these guard families cannot change after warm-up:
#  - TENSOR_MATCH on nn.Parameter: weights are loaded before compilation and
#    never resized (in-place value updates are still read at call time).
#  - EMPTY_NN_MODULE_HOOKS_DICT: no hooks are registered after compilation in
#    eager/compile mode (the lazy-mode mark_step hook is added before compile).
#  - DUPLICATE_INPUT on Python modules: one imported module reached through two
#    global names is the same object for the life of the process.
# Input, constant, global and module-attribute guards are always kept, so new
# bucket shapes and runtime state flags still select or recompile graphs.
_GUARD_CENSUS: dict[str, Any] = {"compiles": 0, "entries": 0, "dropped": 0, "types": {}, "names": {}}
_GUARD_CENSUS_LAST_DUMP = [0.0]
_GUARD_INDEX_RE = re.compile(r"\[\d+\]|\.\d+(?=\.|\]|'|$)|\b_\d+\b")


def _dump_guard_census(path: str) -> None:
    names = sorted(_GUARD_CENSUS["names"].items(), key=lambda kv: -kv[1])[:80]
    data = {k: _GUARD_CENSUS[k] for k in ("compiles", "entries", "dropped", "types")}
    data["top_names"] = names
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)
    except OSError:
        pass


def _is_static_guard(entry, skip_params: bool, skip_static: bool) -> bool:
    if entry.guard_type == "TENSOR_MATCH":
        return (skip_params or skip_static) and isinstance(entry.value, torch.nn.Parameter)
    if not skip_static:
        return False
    if entry.guard_type == "EMPTY_NN_MODULE_HOOKS_DICT":
        return True
    return entry.guard_type == "DUPLICATE_INPUT" and isinstance(entry.value, types.ModuleType)


def _hpu_static_guard_filter(entries) -> list[bool]:
    skip_params = gaudi_envs.VLLM_HPU_SKIP_PARAM_GUARDS
    skip_static = gaudi_envs.VLLM_HPU_SKIP_STATIC_GUARDS
    keep = [not _is_static_guard(entry, skip_params, skip_static) for entry in entries]
    census_path = gaudi_envs.VLLM_HPU_GUARD_CENSUS_FILE
    if census_path:
        census = _GUARD_CENSUS
        if census["compiles"] == 0:
            atexit.register(_dump_guard_census, census_path)
        census["compiles"] += 1
        census["entries"] += len(entries)
        census["dropped"] += keep.count(False)
        for entry, kept in zip(entries, keep):
            kind = "param" if isinstance(entry.value, torch.nn.Parameter) else type(entry.value).__name__
            key = f"{entry.guard_type}/{kind}/{'kept' if kept else 'dropped'}"
            census["types"][key] = census["types"].get(key, 0) + 1
            name = _GUARD_INDEX_RE.sub("[*]", entry.name)
            if name in census["names"] or len(census["names"]) < 20000:
                census["names"][name] = census["names"].get(name, 0) + 1
        now = time.monotonic()
        if now - _GUARD_CENSUS_LAST_DUMP[0] > 20.0:
            _GUARD_CENSUS_LAST_DUMP[0] = now
            _dump_guard_census(census_path)
    return keep


def _hpu_guard_filter():
    if (gaudi_envs.VLLM_HPU_SKIP_PARAM_GUARDS or gaudi_envs.VLLM_HPU_SKIP_STATIC_GUARDS
            or gaudi_envs.VLLM_HPU_GUARD_CENSUS_FILE):
        return _hpu_static_guard_filter
    return None


class HPUCompileConfig:
    """
    Configuration class, which holds arguments that will be
    passed to torch compile with HPU backend.
    """

    def __init__(self, fullgraph: Optional[bool] = None, dynamic: Optional[bool] = None):
        """
        Allow to override the environment variables for corner case scenarios
        when single functions are compiled with torch.compile decorator.
        Env variables should not be overwritten when it comes to compilation
        of the whole model.
        """
        self.fullgraph = fullgraph if fullgraph is not None else get_config().fullgraph_compilation
        self.dynamic = dynamic if dynamic is not None else get_config().dynamic_shapes_compilation
        self.regional_compilation = get_config().regional_compilation

    def get_compile_args(self) -> dict[str, Any]:
        """
        Returns a dictionary of compile arguments that can be used
        with torch.compile method or decorator
        """
        if self.dynamic:
            args = {"backend": "hpu_backend", "fullgraph": self.fullgraph, "options": {"force_static_compile": True}}
        else:
            args = {"backend": "hpu_backend", "fullgraph": self.fullgraph, "dynamic": False}
        guard_filter = _hpu_guard_filter()
        if guard_filter is not None:
            # torch.compile pops guard_filter_fn before options reach the backend.
            args.setdefault("options", {})["guard_filter_fn"] = guard_filter
        return args


_async_sched_module.AsyncScheduler = HPUAsyncScheduler  # type: ignore[misc]

# Guard Prometheus counters against negative prompt-token counts that can arise
# when KV-cache blocks are invalidated during OOM and token-count bookkeeping
# becomes temporarily inconsistent.  Prometheus counters require non-negative
# increments; clamping here prevents a crash in PrometheusStatLogger.record().
# noqa: E402 cannot be hoisted to the module top: this monkey-patch must run
# AFTER the `_async_sched_module.AsyncScheduler = HPUAsyncScheduler` rebind
# above so vllm.v1.metrics.stats picks up the HPU scheduler symbol.
import vllm.v1.metrics.stats as _stats_module  # noqa: E402

_stats_get_by_source_orig = _stats_module.PromptTokenStats.get_by_source


def _hpu_get_by_source(self, source: str) -> int:
    return max(0, _stats_get_by_source_orig(self, source))


_prompt_token_stats_cls: Any = _stats_module.PromptTokenStats
_prompt_token_stats_cls.get_by_source = _hpu_get_by_source


def patch_nixl_utils_for_hpu():
    """Patch vllm.distributed.nixl_utils to use nixl._api instead of rixl._api.

    Upstream vLLM gates NIXL imports on is_cuda(), falling back to rixl._api
    for all other platforms. HPU needs nixl._api (same as CUDA), so we
    monkey-patch the module-level symbols before anything else imports them.
    """
    import logging

    logger = logging.getLogger(__name__)
    try:
        from nixl._api import nixl_agent as _NixlWrapper
        from nixl._api import nixl_agent_config as _nixl_agent_config
    except ImportError:
        return
    try:
        from nixl._bindings import nixlXferTelemetry as _nixlXferTelemetry
    except ImportError:
        _nixlXferTelemetry = None  # type: ignore[assignment]

    import vllm.distributed.nixl_utils as _nixl_mod

    _nixl_mod.NixlWrapper = _NixlWrapper  # type: ignore[attr-defined]
    _nixl_mod.nixl_agent_config = _nixl_agent_config  # type: ignore[attr-defined]
    if _nixlXferTelemetry is not None:
        _nixl_mod.nixlXferTelemetry = _nixlXferTelemetry  # type: ignore[attr-defined]
    logger.info("Patched vllm.distributed.nixl_utils for HPU (nixl._api)")
