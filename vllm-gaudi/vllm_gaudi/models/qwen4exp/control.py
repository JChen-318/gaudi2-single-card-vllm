# SPDX-License-Identifier: Apache-2.0
"""One host copy of the step's control tensors.

Strata keeps token ids and positions on the host and only syncs the device
once per token. QSA and PLE were reading those same scalars back from HPU
inside every layer (``.item()`` / ``.tolist()``), which drains the pipeline.
The model forward copies them once, before the layer loop, and the blocks
read the CPU twin for control flow. Compute tensors stay on the device.

H23-2 adds :func:`cached_span_plan`: the QSA request-span / chunk-start
extraction is *the same answer for all twelve QSA layers of one step*, so it
runs at most once per step and every later layer reuses the plan. The cache
is keyed on the identity of the two control tensors (a strong reference is
kept, so a recycled ``id`` cannot alias a stale entry), their ``_version``
(PyTorch bumps it on every in-place write), the token count and the step
generation that :func:`bind_host_control` assigns.
"""
from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field

import torch


@dataclass
class HostControl:
    positions: torch.Tensor | None
    input_ids: torch.Tensor | None
    query_start_loc: torch.Tensor | None
    generation: int = 0
    #: The device tensor ``positions`` was copied from (H28-2). Kept so a
    #: layer can tell "this is the step's positions" from "some other tensor
    #: of the same shape" without comparing 3xN values on device.
    positions_source: torch.Tensor | None = None
    #: Are the three M-RoPE position rows identical? ``None`` for 1-D
    #: positions or when the runner handed something unexpected. Answered
    #: from the CPU twin, so it costs no device sync.
    positions_equal: bool | None = None


_CTRL: ContextVar[HostControl | None] = ContextVar("qwen4exp_host_ctrl", default=None)

# Bumped once per bound step. Only a counter: it makes "one plan per step"
# explicit and separates two steps that happen to reuse the same host tensors
# (the CPU runner path passes positions by reference).
_GENERATION = 0


def _cpu(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None or tensor.device.type == "cpu":
        return tensor
    return tensor.detach().to(device="cpu")


def bind_host_control(
    positions: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    query_start_loc: torch.Tensor | None,
) -> Token:
    """Copy the three control tensors to CPU and install them for this step."""
    global _GENERATION
    _GENERATION += 1
    twin = _cpu(positions)
    return _CTRL.set(
        HostControl(
            twin,
            _cpu(input_ids),
            _cpu(query_start_loc),
            _GENERATION,
            positions_source=positions,
            positions_equal=_rows_equal(twin),
        )
    )


def reset_host_control(token: Token) -> None:
    _CTRL.reset(token)


def host_control() -> HostControl | None:
    return _CTRL.get()


def cpu_twin(tensor: torch.Tensor | None, field: str) -> torch.Tensor | None:
    """CPU copy of ``tensor`` when the bound control tensor has the same shape."""
    ctrl = _CTRL.get()
    if ctrl is None or tensor is None:
        return None
    twin = getattr(ctrl, field, None)
    if twin is None or tuple(twin.shape) != tuple(tensor.shape):
        return None
    return twin


def _rows_equal(twin: torch.Tensor | None) -> bool | None:
    """``True``/``False`` for a ``[3, N]`` CPU positions tensor, else ``None``."""
    if twin is None or twin.ndim != 2 or twin.shape[0] != 3:
        return None
    try:
        return bool(torch.equal(twin[0], twin[1]) and torch.equal(twin[1], twin[2]))
    except RuntimeError:  # pragma: no cover - exotic dtype
        return None


def positions_rows_equal(positions: torch.Tensor | None) -> bool | None:
    """Are the three M-RoPE rows of ``positions`` identical (text-only step)?

    The single-request serve path builds M-RoPE positions as three copies of
    the same token positions (``Qwen4ExpModel.get_mrope_input_positions``),
    and the interleaved composition then only ever reads values that equal
    row 0. Answering this from the host twin that :func:`bind_host_control`
    already copied keeps it off the device: ``None`` means "unknown here",
    never "not equal", so a caller that has no host copy takes the general
    path instead of paying a device→host read for the answer.
    """
    if positions is None or positions.ndim != 2:
        return None
    ctrl = _CTRL.get()
    if ctrl is not None and positions is ctrl.positions_source:
        return ctrl.positions_equal
    if positions.device.type == "cpu":
        return _rows_equal(positions)
    return None


# ---------------------------------------------------------------------------
# H23-2: one request-span plan per step
# ---------------------------------------------------------------------------
@dataclass
class SpanPlan:
    """Resolved request spans and their first-row chunk starts for one step."""

    num_tokens: int
    positions: torch.Tensor | None
    query_start_loc: torch.Tensor | None
    versions: tuple[int, int]
    generation: int
    spans: list[tuple[int, int]] = field(default_factory=list)
    #: ``positions[0, start]`` per span (the chunk's first cache position).
    starts: list[int] = field(default_factory=list)

    def matches(
        self,
        num_tokens: int,
        positions: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        versions: tuple[int, int],
    ) -> bool:
        return (
            self.num_tokens == num_tokens
            and self.positions is positions
            and self.query_start_loc is query_start_loc
            and self.versions == versions
        )


_PLAN: SpanPlan | None = None


def _version(tensor: torch.Tensor | None) -> int | None:
    """``Tensor._version``, or ``None`` for an inference tensor.

    vLLM executes the model under ``torch.inference_mode()``, and inference
    tensors do not track a version counter (``int(tensor._version)`` raises
    ``RuntimeError: Inference tensors do not track version counter``). The
    caller decides what a missing counter means; this must not raise.
    """
    if tensor is None:
        return -1
    try:
        return int(tensor._version)
    except RuntimeError:
        return None


def cached_span_plan(
    num_tokens: int,
    query_start_loc: torch.Tensor | None,
    positions: torch.Tensor | None,
    compute,
) -> SpanPlan:
    """Return this step's :class:`SpanPlan`, computing it at most once.

    ``compute()`` must return ``(spans, starts)``. It runs only on a miss, so
    the twelve QSA layers of one decode step perform a single host extraction
    (and, on HPU, a single device→host transfer) instead of twelve.

    Live serve (H25 Phase 2): the runner's tensors are inference tensors, so
    their version counter is unavailable. A bound step generation still
    separates two forwards, and every layer of one forward shares it, so the
    plan is reused per step exactly as in the offline benches. Without a bound
    control (direct layer calls) a missing counter disables the cache instead:
    an in-place write would be invisible there.
    """
    global _PLAN
    ctrl = _CTRL.get()
    generation = 0 if ctrl is None else ctrl.generation
    versions = (_version(positions), _version(query_start_loc))
    if ctrl is None and None in versions:
        spans, starts = compute()
        return SpanPlan(num_tokens, positions, query_start_loc, (-1, -1), generation,
                        spans, starts)
    # A missing counter inside a bound step is safe: the generation already
    # scopes reuse to this forward.
    versions = tuple(-1 if v is None else v for v in versions)
    plan = _PLAN
    if (
        plan is not None
        and plan.generation == generation
        and plan.matches(num_tokens, positions, query_start_loc, versions)
    ):
        return plan
    spans, starts = compute()
    _PLAN = SpanPlan(num_tokens, positions, query_start_loc, versions, generation,
                     spans, starts)
    return _PLAN


def invalidate_span_plan() -> None:
    """Drop the cached plan (tests; a caller that mutates positions in place)."""
    global _PLAN
    _PLAN = None


# ---------------------------------------------------------------------------
# H28-2: one MRoPE cos||sin composition per step
# ---------------------------------------------------------------------------
@dataclass
class StepCosSin:
    """The step's composed RoPE rows and the identity they were built from."""

    positions: torch.Tensor
    version: int
    key: tuple
    generation: int
    value: torch.Tensor


_COS_SIN: list[StepCosSin] = []
#: Slots kept per step. Two are needed in practice: the step's own positions
#: (shared by twelve layers) and a block-start subset (``_pool_new_blocks``),
#: which must not evict the former or the twelve layers would ping-pong.
_COS_SIN_SLOTS = 4


def cached_mrope_cos_sin(
    positions: torch.Tensor,
    key: tuple,
    compute,
):
    """Return this step's ``cos||sin`` rows, composing them at most once.

    ``key`` identifies everything outside ``positions`` that the composition
    depends on (the cache tensor, the M-RoPE sections, the layout). Every QSA
    layer of a step asks for the same rows -- the main q/k RoPE and the
    indexer's partial RoPE both read them -- so a cache hit removes eleven of
    the twelve compositions per step and one of the two per layer.

    Same reuse rule as :func:`cached_span_plan`: identity of the positions
    tensor plus its ``_version`` (PyTorch bumps it on every in-place write),
    plus the step generation that :func:`bind_host_control` assigns. A live
    serve runs under ``torch.inference_mode()``, where the version counter is
    unavailable; the generation alone then scopes reuse to one forward.
    Without a bound control a missing counter disables the cache, because an
    in-place write would be invisible there.
    """
    ctrl = _CTRL.get()
    generation = 0 if ctrl is None else ctrl.generation
    version = _version(positions)
    if ctrl is None and version is None:
        return compute()
    version = -1 if version is None else version
    for entry in _COS_SIN:
        if (
            entry.generation == generation
            and entry.positions is positions
            and entry.version == version
            and entry.key == key
        ):
            return entry.value
    value = compute()
    if ctrl is not None:
        # A bound step owns the cache: drop the previous forward's rows so a
        # device tensor is never kept alive past its step.
        _COS_SIN[:] = [e for e in _COS_SIN if e.generation == generation]
    _COS_SIN.append(StepCosSin(positions, version, key, generation, value))
    if len(_COS_SIN) > _COS_SIN_SLOTS:
        del _COS_SIN[0]
    return value


def invalidate_mrope_cos_sin() -> None:
    """Drop the cached rows (tests; a caller that mutates positions in place)."""
    _COS_SIN.clear()
