# SPDX-License-Identifier: Apache-2.0
"""Packed NVFP4 routed experts for the Qwen4Exp HPU port.

Materializing every routed expert in BF16 is about 225 GiB, which does not
fit in 96 GiB HBM. The checkpoint planes stay packed ([E, N, K/2] U8 +
[E, N, K/16] E4M3). A forward dequantizes only the selected experts through
``custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2`` (at most 512 ids per
call, the kernel's S cap) and folds ``weight_scale_2`` in BF16, matching the
kernel contract.
"""
from __future__ import annotations

import os
from pathlib import Path

# Before torch / any HPU graph compile: importing torch+habana first must
# not leave GC_KERNEL_PATH pointing only at the stock TPC library (which
# lacks the custom NVFP4 GUIDs). Idempotent; see _gc_kernel_path.
from ._gc_kernel_path import ensure_gc_kernel_path

ensure_gc_kernel_path()

import torch  # noqa: E402
from torch import nn  # noqa: E402

# Lazy-mode fresh-storage copies for custom-op activation inputs (see
# _materialize); a no-op beyond .contiguous() in eager mode.
from ._materialize import lazy_mark_step as _lazy_mark_step  # noqa: E402
from ._materialize import is_hpu_lazy_mode as _is_hpu_lazy_mode  # noqa: E402
from ._materialize import materialized as _materialized  # noqa: E402

# Kernel S cap (H9(a)): one launch covers every distinct expert of an E=512
# layer, so U=512 prefill needs no chunk loop and no ``torch.cat``.
MAX_DEQUANT_IDS = 512
# Decode grouped-GEMV routing gate. Deliberately separate from the kernel cap
# above: B=8, top_k=10 gives 80 slots, the largest decode step that fits the
# decode path. Reusing MAX_DEQUANT_IDS here would route T=32 prefill
# (T*top_k=320) into the decode GEMV launch (H10 widens this to 512).
DECODE_MAX_SLOTS = 80

_DEQUANT_OP = "custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2"
_DEQUANT_LIB_ENV = "VLLM_QWEN4EXP_DEQUANT_LIB"


def _static_shape_mode(device: torch.device | None = None) -> bool:
    """True when a host read of a tensor value would see unexecuted data.

    Under ``PT_COMPILE_ONLY_MODE`` (engine warmup; ``hpu_model_runner``
    enters it through ``bridge_config.env_setting``) Synapse compiles
    recipes without executing tensors, so output storage is intentionally
    uninitialized and ``int(tensor)`` can return arbitrary garbage -- on
    HPU that made ``torch.arange(negative)`` fail in ``compact_expert_ids``.
    The same holds while an HPU graph is being captured: a baked host value
    would also be wrong on replay. Callers must use a static (shape-only)
    fallback in either case.

    Detection prefers the existing Habana helper
    ``bridge_config.get_pt_compile_only_mode()`` (the guard already used by
    the Qwen2.5-VL / Kimi vision paths). ``bc.env_setting`` flips that flag
    without touching ``os.environ``, so the env var alone would miss the
    real warmup; it is still checked first because it is how a CPU test (or
    a launch that exports the variable) simulates compile-only mode. The
    bridge and capture queries only run for HPU devices, and every probe is
    guarded, so CPU / non-Habana paths simply keep the exact fast path.
    """
    if os.environ.get("PT_COMPILE_ONLY_MODE", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    if device is not None and device.type != "hpu":
        return False
    try:
        from habana_frameworks.torch.internal import bridge_config

        if bridge_config.get_pt_compile_only_mode():
            return True
    except Exception:
        pass
    try:
        is_capturing = getattr(getattr(torch, "hpu", None), "is_current_stream_capturing", None)
        if callable(is_capturing) and is_capturing():
            return True
    except Exception:
        pass
    return False


def _ensure_dequant_op():
    """Resolve the indexed-dequant custom op, loading the extension if needed.

    Tests load the extension explicitly, but a ``vllm serve`` process never
    does; the first MoE forward would die on a missing op namespace entry.
    """
    namespace = torch.ops.custom_op
    if hasattr(namespace, _DEQUANT_OP):
        return getattr(namespace, _DEQUANT_OP)
    candidates = []
    override = os.environ.get(_DEQUANT_LIB_ENV)
    if override:
        candidates.append(Path(override))
    # packed_moe.py lives at <repo>/vllm-gaudi/vllm_gaudi/models/qwen4exp/;
    # the built extension sits in <repo>/vllm-gaudi/csrc/qwen4exp/pytorch/.
    here = Path(__file__).resolve()
    for base in (here.parents[3], Path.cwd()):
        candidates.extend(base.glob("csrc/qwen4exp/pytorch/hpu_qwen4exp_nvfp4_dequant_pt2*.so"))
    for candidate in candidates:
        if candidate.is_file():
            torch.ops.load_library(str(candidate))
            if hasattr(namespace, _DEQUANT_OP):
                return getattr(namespace, _DEQUANT_OP)
    raise RuntimeError(
        f"{_DEQUANT_OP} is not registered and no built extension was found; "
        "build it with `cd vllm-gaudi/csrc/qwen4exp/pytorch && "
        "python setup.py build_ext --inplace` (or point "
        f"{_DEQUANT_LIB_ENV} at the built .so)"
    )


def materialized_ids(ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return the expert-id list as a 1-D int32 tensor the TPC ops can consume.

    Both indexed kernels take ``ids`` as their first input and read it while the
    lazy graph is built. A *view* has no storage of its own, so ``reshape(-1)``
    of the ``[T, top_k]`` routing result - or anything derived from that view,
    even ``+ 0`` - makes the graph die at compile time with

        Empty tensor optional ... Strided Params Has Value: 1

    Eager mode never showed this because every eager call has a real tensor,
    which is why the layer tests passed while the lazy full-layer benchmark
    failed. Copying into a fresh buffer materializes the list; it is 4 bytes per
    slot (a few hundred bytes), one launch per projection, and it keeps the
    decode path graph-safe in both execution modes.
    """
    if (ids.dim() == 1 and ids.dtype == torch.int32 and ids.device == device and not ids._is_view()
            and not _is_hpu_lazy_mode()):
        return ids
    flat = ids.detach().to(device=device, dtype=torch.int32)
    if flat.dim() != 1:
        flat = flat.reshape(-1)
    out = torch.empty(flat.shape, dtype=torch.int32, device=device)
    out.copy_(flat)
    return out


def dequant_indexed(
    packed: torch.Tensor,
    scales: torch.Tensor,
    scale2: torch.Tensor,
    ids: torch.Tensor,
) -> torch.Tensor:
    """Dequant selected experts to ``[S, N, K]`` BF16, scale2 folded.

    ``ids`` indexes axis 0 of ``packed`` / ``scales`` / ``scale2``. The HPU
    path is the TPC kernel; the host path is the same nibble math with the
    scale folded after the BF16 cast, not before it.
    """
    if ids.numel() == 0:
        # Empty selection (zero-token or all-pad warmup steps) yields an
        # empty [0, N, K] result instead of failing the whole forward.
        return packed.new_empty((0, packed.shape[1], packed.shape[2] * 2), dtype=torch.bfloat16)
    if packed.device.type != "hpu":
        from vllm_gaudi.models.qwen4exp.weights import dequant_nvfp4

        ids_cpu = ids.detach().to(device="cpu", dtype=torch.int32)
        pieces = []
        for start in range(0, ids_cpu.numel(), MAX_DEQUANT_IDS):
            piece = ids_cpu[start:start + MAX_DEQUANT_IDS]
            weight = torch.stack([
                dequant_nvfp4(packed[int(expert)], scales[int(expert)], 1.0)
                for expert in piece.tolist()
            ])
            factor = scale2[piece.long()].to(device=weight.device, dtype=weight.dtype)
            pieces.append(weight * factor.view(-1, 1, 1))
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
    # Stay on the device. Copying ids and scale2 to the host drains the
    # pipeline once per projection per layer (144 drains per decode token).
    # scale2 is a per-expert scalar; indexing it on HPU is exact for ids
    # that fit in 32 bits, which every expert id does.
    op = _ensure_dequant_op()
    ids_dev = materialized_ids(ids, packed.device)
    # The kernel casts scale2 to BF16 and multiplies after the nibble decode,
    # which is what _fold_scale2 did as a second HBM pass.
    scale2_dev = scale2.detach().to(device=packed.device, dtype=torch.float32).contiguous()
    pieces = []
    if ids_dev.shape[0] <= MAX_DEQUANT_IDS:
        # No slice: a chunk view is exactly the input the bridge rejects.
        pieces.append(op(ids_dev, packed, scales, scale2_dev))
    else:
        for start in range(0, ids_dev.shape[0], MAX_DEQUANT_IDS):
            # A prefill chunk is a slice of the materialized list, so it needs
            # its own buffer for the same reason as ``ids`` itself. The copy is
            # 4 bytes per slot against a multi-GiB dequantized output.
            piece = materialized_ids(ids_dev[start:start + MAX_DEQUANT_IDS], packed.device)
            pieces.append(op(piece, packed, scales, scale2_dev))
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)


def _fold_scale2(weight: torch.Tensor, scale2: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Multiply the BF16 rows by the per-expert scale after the cast.

    ``(w * s2) @ x`` is what the kernel contract asks the caller to do. The
    fold stays in the weight (not the GEMM epilogue) so the numerics match
    the path the layer tests already checked.
    """
    if scale2.device == weight.device:
        factor = scale2.index_select(0, ids.to(dtype=torch.long))
    else:
        # Streaming loads may still hold scale2 on CPU. Index the small id
        # list there; this is the only host touch, and only until ``.to``.
        factor = scale2.index_select(0, ids.detach().to(device="cpu", dtype=torch.long))
        factor = factor.to(device=weight.device)
    return weight * factor.to(dtype=weight.dtype).view(-1, 1, 1)


def gate_up_rowperm_enabled() -> bool:
    """H31 (``VLLM_QWEN4EXP_MOE_ROWPERM``, default 1): gate/up rows stored in the k4 order of the down input.

    The down projection contracts over the intermediate axis that gate/up produce. Its k4 forms (the
    decode GEMV and the prefill dequant) read that axis permuted, ``x_k4[t*512 + j*128 + i] =
    x[t*512 + 4i + j]``; the decode graph permuted the activation for it every layer (a DMA transpose
    and two engine hand-offs per layer) and the prefill chunk graph could not afford to, so down kept
    the natural dequant, three times the TPC instructions per weight of the k4 one. Storing the gate
    and up rows in that order makes their output the down input as is. Rows are independent and the
    NVFP4 blocks run along K inside a row, so this moves whole rows and changes no value. Generic
    paths (``dequant``, ``gemv_slots``, ``swiglu_gemv_slots``, ``forward_proj``) give their gate/up
    results back in the natural order; only the decode and prefill graphs read the stored order.
    """
    return os.environ.get("VLLM_QWEN4EXP_MOE_ROWPERM", "1") != "0"


#: Set once any store has its gate/up rows in the k4 order (all stores of a model are permuted together).
_ROWPERM_ACTIVE: list = [False]
_ROWPERM_INV: list = [None]


def rowperm_active() -> bool:
    return _ROWPERM_ACTIVE[0]


def k4_row_order(rows: int, device=None) -> torch.Tensor:
    """``perm`` with ``stored[p] = natural[perm[p]]``: k4 order within whole 512-row tiles, the tail natural."""
    full = rows // 512 * 512
    head = torch.arange(full, dtype=torch.long).view(full // 512, 128, 4).transpose(-1, -2).reshape(-1)
    perm = torch.cat([head, torch.arange(full, rows, dtype=torch.long)])
    return perm.to(device) if device is not None else perm


def ensure_gate_up_rowperm(store) -> None:
    """Permute ``store``'s gate/up rows in place once, when its planes are on the card."""
    if getattr(store, "gate_up_row_inv", None) is not None or not gate_up_rowperm_enabled():
        return
    packed = getattr(store, "gate_proj_packed", None)
    if packed is None or packed.numel() == 0 or packed.device.type != "hpu" or packed.dim() != 3:
        return
    if _static_shape_mode(packed.device):
        return
    rows = int(packed.shape[1])
    perm = k4_row_order(rows, packed.device)
    with torch.no_grad():
        for kind in ("gate_proj", "up_proj"):
            for suffix in ("packed", "scale"):
                name = f"{kind}_{suffix}"
                setattr(store, name, getattr(store, name).index_select(1, perm).contiguous())
    store.gate_up_row_perm = perm
    store.gate_up_row_inv = torch.argsort(perm)
    _ROWPERM_ACTIVE[0] = True
    if _ROWPERM_INV[0] is None:
        _ROWPERM_INV[0] = store.gate_up_row_inv


def rowperm_inv() -> torch.Tensor | None:
    """The inverse row order shared by every permuted store (one intermediate size per model)."""
    return _ROWPERM_INV[0]


def gate_up_natural(store, out: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """A gate/up result in the stored row order back in the natural order (no-op without the permutation)."""
    inv = getattr(store, "gate_up_row_inv", None)
    return out if inv is None else out.index_select(dim, inv)


class PackedRoutedExperts(nn.Module):
    """The routed-expert planes of one layer, still packed in HBM.

    Storage is allocated on the first tensor of a group, so construction does
    not reserve 60 GiB and the shapes come from the checkpoint rather than
    from config arithmetic.
    """

    # Checkpoint projection -> (buffer prefix, scale2 attribute)
    _KINDS = ("gate_proj", "up_proj", "down_proj")

    def __init__(self, num_experts: int) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        for kind in self._KINDS:
            self.register_buffer(f"{kind}_packed", torch.empty(0, dtype=torch.uint8), persistent=False)
            self.register_buffer(f"{kind}_scale", torch.empty(0, dtype=torch.uint8), persistent=False)
            self.register_buffer(f"{kind}_scale2", torch.empty(0, dtype=torch.float32), persistent=False)
        self._pending: dict[int, dict[str, torch.Tensor]] = {}

    def update_expert_map(self) -> None:
        """EPLB hook. A single card keeps every expert resident."""

    @staticmethod
    def pack_header(memory_map, header_bytes: int = 512) -> dict:
        """Parse the JSON header of a ``tools/qwen4exp_flashnext/pack_experts.py`` pack."""
        import json

        blob = bytes(memory_map[:header_bytes])
        head = json.loads(blob.split(b"\x00", 1)[0].decode())
        if head.get("magic") != "STRATA-Q4E-PACK":
            raise ValueError("not a Qwen4Exp expert pack")
        return head

    def load_pack(self, memory_map, header: dict, slab_offset: int) -> None:
        """Fill every buffer from one pack slab (layer x kind major order).

        ``memory_map`` is a numpy memmap of the pack file; the three buffers
        become zero-copy views into it. The later ``.to("hpu")`` performs one
        bulk transfer per buffer, replacing 73,728 per-tensor loads.
        """
        import numpy as np

        p_bytes = s_bytes = None
        for kind in self._KINDS:
            geometry = header["geometry"][kind]
            packed_shape = (header["num_experts"], *geometry["packed"])
            scales_shape = (header["num_experts"], *geometry["scales"])
            if self.num_experts != header["num_experts"]:
                raise ValueError(
                    f"pack holds {header['num_experts']} experts, model has {self.num_experts}"
                )
            p_bytes = int(np.prod(packed_shape))
            s_bytes = int(np.prod(scales_shape))
            packed = torch.from_numpy(memory_map[slab_offset:slab_offset + p_bytes].reshape(
                *packed_shape).view(np.uint8))
            slab_offset += p_bytes
            scales = torch.from_numpy(memory_map[slab_offset:slab_offset + s_bytes].reshape(
                *scales_shape).view(np.uint8))
            slab_offset += s_bytes
            scale2 = torch.from_numpy(memory_map[slab_offset:slab_offset + self.num_experts * 4]
                                      .copy().view(np.float32))
            slab_offset += self.num_experts * 4
            setattr(self, f"{kind}_packed", packed)
            setattr(self, f"{kind}_scale", scales)
            setattr(self, f"{kind}_scale2", scale2)
            # Non-tensor attribute: the buffers above are zero-copy views of
            # this mapping, so it must outlive them (until .to(device) copies).
            self._pack_mmap = memory_map

    def _fill(self, kind: str, expert: int, weight: torch.Tensor, scale: torch.Tensor, scale2) -> None:
        packed = getattr(self, f"{kind}_packed")
        if packed.numel() == 0:
            packed = torch.empty(
                (self.num_experts, *weight.shape), dtype=torch.uint8, device=weight.device,
            )
            setattr(self, f"{kind}_packed", packed)
            setattr(self, f"{kind}_scale", torch.empty(
                (self.num_experts, *scale.shape), dtype=torch.uint8, device=weight.device,
            ))
            setattr(self, f"{kind}_scale2", torch.zeros(
                self.num_experts, dtype=torch.float32, device=weight.device,
            ))
        packed[expert] = weight
        getattr(self, f"{kind}_scale")[expert] = scale
        if scale2 is not None:
            getattr(self, f"{kind}_scale2")[expert] = float(scale2.reshape(-1)[0])

    def load_tensor(self, name: str, tensor: torch.Tensor) -> bool:
        """Consume ``experts.<i>.<kind>.weight[_scale[_2]]``. True when handled."""
        parts = name.split(".")
        if len(parts) < 4 or parts[0] != "experts" or not parts[1].isdigit():
            return False
        kind = parts[2]
        if kind not in self._KINDS:
            return False
        expert = int(parts[1])
        suffix = ".".join(parts[3:])
        slot = self._pending.setdefault(expert, {})
        if suffix == "weight":
            slot[kind] = tensor
        elif suffix == "weight_scale":
            slot[f"{kind}_scale"] = tensor.view(torch.uint8)
        elif suffix == "weight_scale_2":
            slot[f"{kind}_scale2"] = tensor
            # The group may already be allocated (weight and scale arrived
            # first), in which case nothing will call _fill again.
            current = getattr(self, f"{kind}_scale2")
            if current.numel() > 0:
                current[expert] = float(tensor.reshape(-1)[0])
                return True
        elif suffix == "input_scale":
            # Activation-quant input scale; inference never consumes it.
            # Rejecting it would leak the name into AutoWeightsLoader, which
            # has no ``experts.<i>`` child to place it under.
            return True
        else:
            return False
        weight = slot.get(kind)
        scale = slot.get(f"{kind}_scale")
        if weight is not None and scale is not None:
            self._fill(kind, expert, weight.contiguous(), scale.contiguous(), slot.get(f"{kind}_scale2"))
            slot.pop(kind, None)
            slot.pop(f"{kind}_scale", None)
        return True

    def dequant(self, kind: str, ids: torch.Tensor) -> torch.Tensor:
        ensure_gate_up_rowperm(self)
        weight = dequant_indexed(
            getattr(self, f"{kind}_packed"),
            getattr(self, f"{kind}_scale"),
            getattr(self, f"{kind}_scale2"),
            ids,
        )
        return gate_up_natural(self, weight, 1) if kind in ("gate_proj", "up_proj") else weight


def load_expert_pack(model, path) -> int:
    """Fill every ``PackedRoutedExperts`` from a pack file. Returns layers."""
    import numpy as np

    memory_map = np.memmap(path, dtype=np.uint8, mode="r")
    header = PackedRoutedExperts.pack_header(memory_map)
    experts_per_layer = header["num_experts"]
    stride = 0
    for kind in PackedRoutedExperts._KINDS:
        geometry = header["geometry"][kind]
        stride += (experts_per_layer * int(np.prod(geometry["packed"]))
                   + experts_per_layer * int(np.prod(geometry["scales"]))
                   + experts_per_layer * 4)
    offset = 512
    loaded = 0
    for layer in model.model.layers:
        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        if not isinstance(experts, PackedRoutedExperts):
            continue
        experts.load_pack(memory_map, header, offset)
        offset += stride
        loaded += 1
    if loaded != header["num_layers"]:
        raise ValueError(f"pack holds {header['num_layers']} layers, model has {loaded}")
    return loaded


def moe_h15_enabled() -> bool:
    """H15 MoE fusion master switch (``VLLM_QWEN4EXP_MOE_H15=0`` disables all).

    CPU / non-HPU tensors always keep the exact pre-H15 fallback regardless
    of this switch; every H15 helper below checks the device itself.
    """
    return os.environ.get("VLLM_QWEN4EXP_MOE_H15", "1") != "0"


def _h15_op(name: str):
    """Resolve an H15 custom op from the shared extension, or ``None``."""
    namespace = torch.ops.custom_op
    if hasattr(namespace, name):
        return getattr(namespace, name)
    try:
        _ensure_dequant_op()
    except Exception:
        return None
    return getattr(namespace, name, None)


def route_topk(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax-renormalized top-k. Returns ``(ids [T, K] int32, weights [T, K])``.

    H15: on HPU with E=512 the post-GEMM chain (topk + float cast + softmax
    + BF16/int32 casts) is one fused TPC kernel; ids are in descending-logit
    order with deterministic lowest-index tie-breaking and weights are the
    F32 softmax of the top-k values rounded to BF16. The selected value
    sequence is bit-identical to the chain below; on exact BF16 ties the
    HPU ``torch.topk`` order is partition-dependent (not an index order),
    so a differing id there is an exact logit tie, verified as such in
    ``tools/qwen4exp_flashnext/bench_moe_h15.py``.
    """
    logits = nn.functional.linear(hidden, gate_weight)
    if (moe_h15_enabled() and hidden.device.type == "hpu" and logits.shape[-1] == 512
            and 0 < top_k <= 64):
        op = _h15_op("custom_qwen4exp_moe_router_topk_bf16_gaudi2")
        if op is not None:
            ids, weights = op(_materialized(logits), int(top_k))
            return ids, weights
    values, indices = torch.topk(logits, top_k, dim=-1)
    weights = torch.softmax(values.float(), dim=-1).to(hidden.dtype)
    return indices.to(torch.int32), weights


def apply_routed_experts(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU experts. ``topk_ids`` are compact indices into ``gate``/``up``/``down``.

    ``gate`` and ``up`` are ``[U, intermediate, hidden]``; ``down`` is
    ``[U, hidden, intermediate]``. One matmul per selected expert, so a
    prefill does not materialize ``[T, K, N, K]``.
    """
    out = torch.zeros(hidden.shape[0], down.shape[1], dtype=hidden.dtype, device=hidden.device)
    for expert in range(gate.shape[0]):
        chosen = topk_ids == expert
        token_ids = chosen.any(dim=-1).nonzero().flatten()
        if token_ids.numel() == 0:
            continue
        tokens = hidden.index_select(0, token_ids)
        activated = nn.functional.silu(nn.functional.linear(tokens, gate[expert]))
        activated = activated * nn.functional.linear(tokens, up[expert])
        projected = nn.functional.linear(activated, down[expert])
        mix = torch.where(chosen, topk_weights, torch.zeros_like(topk_weights)).sum(dim=-1)
        out.index_add_(0, token_ids, projected * mix.index_select(0, token_ids).unsqueeze(-1))
    return out


# Slot-chunk size for the static-shape grouped path below. One chunk's
# bucket is ``num_experts * chunk`` rows; at E=512 / H=2560 a 512-slot chunk
# is a 1.25 GiB BF16 bucket (plus sub-GiB GEMM outputs), whereas a single
# bucket over a whole 2048-token prompt batch (20480 slots) would be
# 50 GiB and exhausts the HPU device-memory pool during compile-only
# warmup. The chunk count is a Python int, so every shape stays static.
_STATIC_GROUPED_SLOT_CHUNK = 512


def _add_grouped_bucket(
    out: torch.Tensor,
    hidden: torch.Tensor,
    flat_ids: torch.Tensor,
    weights: torch.Tensor,
    token_of_slot: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    experts: torch.Tensor,
    cap: int,
) -> None:
    """One expert-major bucket pass over ``flat_ids`` slots, added into ``out``.

    ``cap`` must be at least the largest per-expert slot count inside
    ``flat_ids`` (the caller guarantees this); rows past an expert's count
    stay zero and are never picked, so the result is the exact grouped mix
    for these slots.
    """
    slots = flat_ids.numel()
    num_experts, _, _ = gate.shape
    hidden_size = hidden.shape[-1]
    device = hidden.device
    counts = (flat_ids.unsqueeze(1) == experts).sum(dim=0)
    starts = torch.cumsum(counts, dim=0) - counts
    # Position of each slot inside its expert bucket. The slots are visited in
    # expert order (argsort, stable), so the rank is the sorted position minus
    # the expert's start; index_copy_ moves it back to slot order (an advanced
    # ``rank[order] = ...`` assignment is avoided on purpose).
    order = torch.argsort(flat_ids, stable=True)
    sorted_ids = flat_ids.index_select(0, order)
    rank_sorted = torch.arange(slots, device=device, dtype=torch.int64) \
        - starts.index_select(0, sorted_ids)
    rank = torch.empty_like(rank_sorted)
    rank.index_copy_(0, order, rank_sorted)
    cell = flat_ids * cap + rank
    x_slots = hidden.index_select(0, token_of_slot)
    bucket = hidden.new_zeros(num_experts * cap, hidden_size)
    bucket.index_copy_(0, cell, x_slots)
    del x_slots
    gate_out = torch.bmm(bucket.view(num_experts, cap, hidden_size), gate.transpose(1, 2))
    up_out = torch.bmm(bucket.view(num_experts, cap, hidden_size), up.transpose(1, 2))
    del bucket
    activated = nn.functional.silu(gate_out) * up_out
    del gate_out, up_out
    projected = torch.bmm(activated, down.transpose(1, 2))
    del activated
    picked = projected.view(num_experts * cap, hidden_size).index_select(0, cell)
    del projected
    out.index_add_(0, token_of_slot, picked * weights.unsqueeze(-1))


def apply_routed_experts_grouped(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU experts in expert-major buckets: three batched GEMMs, no loop.

    Strata's prefill shape (``u4_moe_bucket`` -> grouped GEMM -> ``u4_moe_join``)
    with only STATIC-SHAPE torch ops. The Gaudi2 graph compiler rejects every
    data-dependent-shape op this port tried for bucketing inside the engine
    (``nonzero``, ``bincount``, ``scatter_add_``, ``searchsorted`` all failed
    with synStatus 26 even though they run in a bare process), so the bucket is
    built from counts, a prefix sum, a gather and ``index_copy_``:

    * ``counts[e]``  one-hot reduction ``(flat_ids[:, None] == e).sum(0)``
    * ``starts``     exclusive prefix sum of the counts
    * ``rank``       slot order inside its expert bucket (gather + subtract)
    * ``cell``       ``flat_ids * cap + rank``: the slot's row in the bucket
    * bucket rows     ``index_copy_`` of the routed activations into a
                      ``[num_experts * cap, H]`` buffer (padding rows stay zero)
    * the three projections run as one ``bmm`` each over ``[U, cap, H]``
    * ``index_select`` with the same ``cell`` maps the results back to slots

    Only the busiest expert's bucket size ``cap`` is data-dependent, and it is
    read once per call (an int), never as a tensor shape inside a graph.
    In compile-only mode / HPU graph capture (``_static_shape_mode``) that
    read would see uninitialized storage, so the slots are processed in
    static chunks of ``_STATIC_GROUPED_SLOT_CHUNK`` and each chunk uses its
    own length as ``cap``: within a chunk one expert can receive at most
    that many slots, the extra bucket rows stay zero and are never picked,
    and the per-chunk results add into the same output, so an executed
    result is identical to the exact-cap result. (A single static bucket
    with ``cap == slots`` over all experts scales as ``num_experts * slots``
    rows -- tens of GiB at prompt-bucket sizes -- and exhausted the HPU
    device-memory pool in compile-only warmup.) Outside that mode the exact
    cap is kept, clamped into ``[0, slots]`` as defense-in-depth.
    """
    num_tokens, top_k = topk_ids.shape
    hidden_size = hidden.shape[-1]
    num_experts, intermediate, gate_k = gate.shape
    if gate_k != hidden_size:
        raise ValueError(f"expert gate expects hidden {gate_k}, got {hidden_size}")
    if up.shape != gate.shape:
        raise ValueError(f"gate/up disagree: {tuple(gate.shape)} vs {tuple(up.shape)}")
    if down.shape != (num_experts, hidden_size, intermediate):
        raise ValueError(
            f"down must be [{num_experts}, {hidden_size}, {intermediate}], "
            f"got {tuple(down.shape)}")
    device = hidden.device
    slots = num_tokens * top_k
    out = torch.zeros(num_tokens, hidden_size, dtype=hidden.dtype, device=device)
    if slots == 0 or num_experts == 0:
        return out
    if num_experts * slots > 268_435_456:
        # One-hot counting would allocate >1 GiB; keep the exact loop instead.
        return apply_routed_experts(
            hidden, topk_ids, topk_weights, gate, up, down)

    # Lazy mode: end the pending graph before the host cap read below.
    # The caller dequantized gate/up/down just before this call; those ops
    # are not ancestors of ``counts``, so a partial execution forced by
    # int(counts.max()) can leave them unexecuted and their (empty)
    # outputs then feed the bucket GEMMs as garbage/NaN. Marking first
    # executes them in queue order. No-op in eager mode. In static-shape
    # mode there is no host read to force that partial execution, so the
    # mark (and the read) are both skipped.
    static_cap = _static_shape_mode(device)
    if device.type == "hpu" and not static_cap:
        _lazy_mark_step()
    flat_ids = topk_ids.reshape(-1).to(dtype=torch.int64).contiguous()
    weights = topk_weights.reshape(-1).contiguous()
    token_of_slot = torch.arange(num_tokens, device=device).repeat_interleave(top_k)
    experts = torch.arange(num_experts, dtype=torch.int64, device=device)
    if static_cap:
        # Static shapes only: no host read, and each chunk's length is a
        # valid worst-case cap for that chunk (see the docstring).
        for start in range(0, slots, _STATIC_GROUPED_SLOT_CHUNK):
            end = min(start + _STATIC_GROUPED_SLOT_CHUNK, slots)
            _add_grouped_bucket(
                out,
                hidden,
                flat_ids[start:end],
                weights[start:end],
                token_of_slot[start:end],
                gate,
                up,
                down,
                experts,
                end - start,
            )
        return out
    counts = (flat_ids.unsqueeze(1) == experts).sum(dim=0)
    cap = min(slots, max(0, int(counts.max())))
    if cap == 0:
        return out
    _add_grouped_bucket(out, hidden, flat_ids, weights, token_of_slot, gate, up, down, experts, cap)
    return out


def prefill_grouped_enabled() -> bool:
    """Expert-major grouped prefill GEMM (default on; 0 restores the loop)."""
    return os.environ.get("VLLM_QWEN4EXP_MOE_PREFILL_GROUPED", "1") != "0"


# ---------------------------------------------------------------------------
# H18 prefill MoE: count-ordered expert chunks, one host transfer
# ---------------------------------------------------------------------------
def prefill_h18_enabled() -> bool:
    """H18 prefill MoE (default on; ``VLLM_QWEN4EXP_MOE_PREFILL_H18=0`` restores
    the H9(a) path exactly).

    What H18 changes, all of it lossless for the routed dot products:

    * the presence mask and the per-expert counts are computed ONCE (H9(a)
      built ``(flat[:, None] == experts)`` three times per layer: once in
      ``compact_expert_ids``, once for ``cap`` in
      ``apply_routed_experts_grouped``, once inside ``_add_grouped_bucket``);
    * the two host reads of the H9(a) path (``int(present.sum())`` and
      ``int(counts.max())``) collapse into ONE ``.tolist()`` that carries the
      whole static shape plan (present count + per-chunk caps + per-chunk slot
      offsets);
    * experts are ordered by descending count, then dequantized / bucketed /
      GEMMed in chunks. The per-chunk cap is the max count inside that chunk,
      so the bucket stops padding every expert up to the single busiest
      expert's cap (measured 90.8 % zero rows at T=512 on real layer-0
      routing), and the BF16 weight peak drops from ``U * 9.83 MB`` (4.2 GiB
      at U=427) to ``chunk * 9.83 MB``.
    """
    return os.environ.get("VLLM_QWEN4EXP_MOE_PREFILL_H18", "1") != "0"


# Chunking pays for the per-chunk launch dispatch only once the bucket is big
# enough to be padding-bound. Measured on real layer-0 routing: T=64 (640 slots)
# 13.36 -> 13.61 ms (+1.9%), T=128 (1280) 16.70 -> 16.82 (+0.7%, inside the
# run-to-run spread), T=512 (5120) 26.33 -> 23.46, T=2048 (20480) 41.63 -> 29.17.
# Below this many slots the H9(a) path is kept.
#
# H27 re-measured the boundary with the four-row dequant kernel (2.2-2.4x
# cheaper per launch) at chunk 64 vs 128 vs 256 vs 512, interleaved paired A/B,
# 9 pairs per cell, bitdiff 0 in every cell (``bench_h27_chunk.py``):
#
#   slots   L0 128/64   L3 128/64     (ratio > 1 means chunk 128 is faster)
#     640      1.033       1.039
#    1280      1.023       1.028
#    2560      1.045       1.039
#    5120      1.015       1.047
#   10240      0.981       1.023
#   20480      0.917       0.990
#
# so the crossover sits between 5120 and 10240 slots.
#
# The gate itself stays where H18 put it. With the H27 kernel the chunked path at
# 640 slots (T=64) is layer-dependent in the production block -- L0 1.0402
# (9/9 wins) but L3 0.9743 (6/15), reproduced -- so lowering the gate to 640
# would trade a measured L0 win for a measured L3 loss. Above it the chunk rule
# gains in every measured cell.
PREFILL_H18_MIN_SLOTS = 1024

#: Above this many slots the per-chunk tight ``cap`` is worth more than the
#: extra dequant launches, so the H18 chunk stays at 64 (see the table above:
#: 0.917-0.990 at 20480 slots). Below it, chunk 128 wins every measured cell.
PREFILL_H18_SMALL_SLOTS = 5120

#: Chunk size below ``PREFILL_H18_SMALL_SLOTS``. The BF16 peak of a chunk is
#: ``2 * chunk * 9.83 MB`` while gate+up run, so 128 costs at most +0.9 GiB over
#: 64 (measured max +868 MiB) -- chunk 256 leaves the 1.2 GiB budget at
#: T>=1024 (+1.3 GiB) and loses to 128 on L0 at T=256.
PREFILL_H18_SMALL_CHUNK = 128


def prefill_h18_chunk(n_experts: int, slots: int) -> int:
    """Experts per dequant+bucket chunk (``VLLM_QWEN4EXP_MOE_PREFILL_H18_CHUNK``).

    ``auto`` (the default) picks ``PREFILL_H18_SMALL_CHUNK`` for a small prefill
    and 64 for a big one -- see the crossover table above ``PREFILL_H18_MIN_SLOTS``.
    An integer forces that chunk size (the H18/H22 harnesses still do).
    """
    raw = os.environ.get("VLLM_QWEN4EXP_MOE_PREFILL_H18_CHUNK", "auto")
    try:
        chunk = int(raw)
    except (TypeError, ValueError):
        chunk = PREFILL_H18_SMALL_CHUNK if slots <= PREFILL_H18_SMALL_SLOTS else 64
    return max(1, min(int(n_experts), chunk))


def count_ordered_plan(
    topk_ids: torch.Tensor,
    n_experts: int,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
    """``(selected, local, plan)`` with the whole shape plan in one host read.

    ``selected`` is the present expert ids in descending slot-count order
    (ties by ascending id), ``local`` indexes that list, and ``plan`` carries
    the Python ints the chunk loop needs:

    ``count``      number of present experts (the dequantized id count)
    ``caps[c]``    max slot count among the experts of chunk ``c`` (>= 1)
    ``starts[c]``  first slot of chunk ``c`` in count-sorted slot order
    ``chunk``      experts per chunk
    ``flat``       local ids per slot, ``[slots]`` int64
    ``order``      argsort of ``flat`` (count-sorted slot order)
    ``prefix``     exclusive prefix sum of the count-sorted counts, ``[E+1]``

    Ordering by count is a permutation of the expert axis, so every routed
    slot computes the same dot products against the same weights; only the
    order in which the per-slot results are added into the output changes.
    ``None`` means "caller keeps the H9(a) path" (compile-only/graph capture,
    where the host read would see uninitialized storage).
    """
    flat = topk_ids.reshape(-1).to(dtype=torch.int64)
    device = flat.device
    slot_count = int(flat.numel())
    if flat.numel() == 0 or n_experts <= 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        return empty, torch.zeros_like(topk_ids, dtype=torch.int32), None
    if _static_shape_mode(device):
        return torch.empty(0, dtype=torch.int32, device=device), topk_ids.to(torch.int32), None
    experts = torch.arange(n_experts, dtype=torch.int64, device=device)
    # The one and only presence matrix of the layer. H27 measured the
    # alternatives on real layer-0/3 routing before keeping this form:
    #   * a ``torch.bincount(flat, minlength=E)`` is NOT faster on Gaudi2
    #     (0.43-0.83 ms vs 0.23-0.55 ms for the matrix over 640..20480 slots);
    #   * the fused ``sum(dim=0, dtype=torch.int32)`` is SLOWER than
    #     ``sum(dim=0).to(torch.int32)`` (paired A/B ratio 0.81-0.99, 0-1 wins of
    #     11), so the cast stays;
    #   * the plan is dispatch-bound, not sync-bound: at T=64 the device ops up to
    #     the meta vector take 0.599 ms and the single ``.tolist()`` host read adds
    #     < 0.03 ms (0.672 ms including it).
    counts = (flat.unsqueeze(1) == experts).sum(dim=0).to(torch.int32)
    counts_sorted, order = torch.sort(counts, descending=True, stable=True)
    n_chunks_max = -(-n_experts // chunk)
    padded = counts_sorted
    if n_chunks_max * chunk != n_experts:
        padded = torch.cat([
            counts_sorted,
            torch.zeros(n_chunks_max * chunk - n_experts, dtype=torch.int32, device=device),
        ])
    caps = padded.view(n_chunks_max, chunk).amax(dim=1)
    # dtype on the cumsum itself: one dispatch less than .to(int64) + cumsum
    # (paired A/B ratio 1.00-1.13, 6-8 wins of 11).
    prefix = torch.cat([
        torch.zeros(1, dtype=torch.int64, device=device),
        torch.cumsum(counts_sorted, dim=0, dtype=torch.int64),
    ])
    starts = prefix[::chunk]
    meta = torch.cat([torch.count_nonzero(counts).reshape(1).to(torch.int64),
                      caps.to(torch.int64),
                      starts])
    # Lazy mode: end the pending graph before the host read (H16 rule).
    if device.type == "hpu":
        _lazy_mark_step()
    values = meta.tolist()  # the single host transfer of the layer
    count = max(0, min(int(values[0]), n_experts))
    selected = order[:count].to(dtype=torch.int32)
    ranks = torch.zeros(n_experts, dtype=torch.int32, device=device)
    if count:
        ranks.index_copy_(
            0,
            selected.to(dtype=torch.int64),
            torch.arange(count, dtype=torch.int32, device=device),
        )
    # ``local`` is the count-sorted rank of every slot's expert (0..count-1),
    # NOT the checkpoint's expert id; ``selected`` maps a rank back to the id.
    # Everything downstream stays in sorted slot order -- the slots of expert
    # rank r are exactly the sorted positions ``prefix[r] .. prefix[r+1]``, so
    # a chunk is a contiguous slice and no scatter back to slot order is
    # needed (that scatter costs an index_copy_ plus a gather per layer).
    local = ranks.index_select(0, flat).view_as(topk_ids)
    # H27: the rank map is already int32, so the sort and the two gathers run on
    # int32 (the int64 copy was pure traffic: 20480 slots at T=2048). The final
    # ``rank_sorted`` stays int64 because it indexes the bucket rows.
    local_flat = local.reshape(-1)
    slot_order = torch.argsort(local_flat, stable=True)
    local_sorted = local_flat.index_select(0, slot_order)
    rank_sorted = torch.arange(slot_count, dtype=torch.int64, device=device) \
        - prefix.index_select(0, local_sorted.to(dtype=torch.int64))
    plan = {
        "count": count,
        "caps": [int(v) for v in values[1:1 + n_chunks_max]],
        "starts": [int(v) for v in values[1 + n_chunks_max:]],
        "chunk": int(chunk),
        # Per-rank slot count (rank order): the sparse-tier split needs to know
        # how many experts have ``count > K`` and where their slots end.
        "counts_sorted": counts_sorted,
        "local_sorted": local_sorted,
        "rank_sorted": rank_sorted,
        "slot_order": slot_order,
    }
    return selected, local, plan


def moe_host_plan_enabled() -> bool:
    """H30: compute the prefill routing plan on the host (``VLLM_QWEN4EXP_MOE_HOST_PLAN=0`` disables)."""
    return os.environ.get("VLLM_QWEN4EXP_MOE_HOST_PLAN", "1") != "0" and not _is_hpu_lazy_mode()


#: H31 static chunk graph buckets: experts per chunk and rows per expert.
_CHUNK_GRAPH_EXPERTS = (8, 16, 32, 64)


def moe_chunk_graph_enabled() -> bool:
    """H31 (``VLLM_QWEN4EXP_MOE_CHUNK_GRAPH``, default 1): each expert chunk of a prefill MoE as one
    compiled graph over a fixed-shape bucket.

    Eager runs the chunk op by op: dequant (TPC), bmm (MME), dequant, bmm, silu*mul, dequant, bmm.
    One graph lets the graph compiler overlap each projection's TPC dequant with another
    projection's MME bmm. Measured (tools/qwen4exp_flashnext/kernel_ab/bench_moe_chunk_h31.py, 64 experts): 1.44 ->
    1.15 ms at 64 rows, 1.82 -> 1.11 ms at 256, 4.29 -> 1.92 ms at 1024, same output bits. The bmm
    hides under the dequant, so padding the rows up to a power of two costs nearly nothing, and a
    few dozen graph shapes cover every routing.
    """
    return os.environ.get("VLLM_QWEN4EXP_MOE_CHUNK_GRAPH", "1") != "0"


def _cap_bucket(cap: int) -> int:
    bucket = 64
    while bucket < cap:
        bucket *= 2
    return bucket


def _experts_bucket(n: int) -> int:
    for bucket in _CHUNK_GRAPH_EXPERTS:
        if n <= bucket:
            return bucket
    return n


def _moe_chunk_body(bucketed, ids, gate_packed, gate_scale, gate_scale2, up_packed, up_scale, up_scale2,
                    down_packed, down_scale, down_scale2):
    op = torch.ops.custom_op.custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2
    gate_w = op(ids, gate_packed, gate_scale, gate_scale2)
    up_w = op(ids, up_packed, up_scale, up_scale2)
    gate_out = torch.bmm(bucketed, gate_w.transpose(1, 2))
    up_out = torch.bmm(bucketed, up_w.transpose(1, 2))
    activated = nn.functional.silu(gate_out) * up_out
    down_w = op(ids, down_packed, down_scale, down_scale2)
    return torch.bmm(activated, down_w.transpose(1, 2))


def _moe_chunk_body_k4(bucketed_k4, ids, gate_packed, gate_scale, gate_scale2, up_packed, up_scale, up_scale2,
                       down_packed, down_scale, down_scale2):
    """``_moe_chunk_body`` with gate/up dequantized in the k4 layout; the bucket rows come in that layout.

    The dequant_k4 values equal the natural ones permuted along K (checked on the card), so the
    gate/up products are the same up to the MME's K summation order (about 0.4 % of the BF16
    outputs move by one ulp). Down stays natural: permuting the activation for it cost more than it
    saved at 1,024 rows. Measured in one graph: 64 experts x 256 rows 1.11 -> 0.90 ms, x 1,024 equal.
    """
    op4 = torch.ops.custom_op.custom_qwen4exp_nvfp4_indexed_dequant_k4_bf16_gaudi2
    op = torch.ops.custom_op.custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2
    gate_w = op4(ids, gate_packed, gate_scale, gate_scale2)
    up_w = op4(ids, up_packed, up_scale, up_scale2)
    gate_out = torch.bmm(bucketed_k4, gate_w.transpose(1, 2))
    up_out = torch.bmm(bucketed_k4, up_w.transpose(1, 2))
    activated = nn.functional.silu(gate_out) * up_out
    down_w = op(ids, down_packed, down_scale, down_scale2)
    return torch.bmm(activated, down_w.transpose(1, 2))


def _moe_chunk_body_k4rp(bucketed_k4, ids, gate_packed, gate_scale, gate_scale2, up_packed, up_scale, up_scale2,
                         down_packed, down_scale, down_scale2):
    """``_moe_chunk_body_k4`` over gate/up rows stored in the k4 order (``VLLM_QWEN4EXP_MOE_ROWPERM``).

    gate/up then produce the activation already in the k4 order of the down input, so down takes
    the k4 dequant as well (a third of the natural dequant's TPC instructions per weight) with no
    activation permute. The output rows (hidden) are natural. The products are the same; only the
    MME's K summation order for down changes.
    """
    op4 = torch.ops.custom_op.custom_qwen4exp_nvfp4_indexed_dequant_k4_bf16_gaudi2
    gate_w = op4(ids, gate_packed, gate_scale, gate_scale2)
    up_w = op4(ids, up_packed, up_scale, up_scale2)
    gate_out = torch.bmm(bucketed_k4, gate_w.transpose(1, 2))
    up_out = torch.bmm(bucketed_k4, up_w.transpose(1, 2))
    activated = nn.functional.silu(gate_out) * up_out
    down_w = op4(ids, down_packed, down_scale, down_scale2)
    return torch.bmm(activated, down_w.transpose(1, 2))


def moe_dequant_k4_enabled(hidden_size: int) -> bool:
    """H31 (``VLLM_QWEN4EXP_MOE_DEQUANT_K4``, default 1): the chunk graph's gate/up dequant in the k4 layout."""
    return (os.environ.get("VLLM_QWEN4EXP_MOE_DEQUANT_K4", "1") != "0" and hidden_size % 512 == 0
            and hasattr(torch.ops.custom_op, "custom_qwen4exp_nvfp4_indexed_dequant_k4_bf16_gaudi2"))


_CHUNK_GRAPH_WARMED: set = set()


def _chunk_graph(k4: bool = False):
    from .decode_graph import _compiled

    if k4 and rowperm_active():
        return _compiled(_moe_chunk_body_k4rp)
    return _compiled(_moe_chunk_body_k4 if k4 else _moe_chunk_body)


def _warm_chunk_graphs(store: "PackedRoutedExperts", planes: tuple, hidden_size: int, max_cap: int,
                       device: torch.device, k4: bool = False) -> None:
    """Compile every (experts, rows) bucket once, at the first prefill (the launcher's warmup)."""
    key = (id(store), max_cap, k4, rowperm_active())
    if key in _CHUNK_GRAPH_WARMED or os.environ.get("VLLM_QWEN4EXP_MOE_CHUNK_GRAPH_WARM", "1") == "0":
        return
    _CHUNK_GRAPH_WARMED.add(key)
    graph = _chunk_graph(k4)
    cap = 64
    caps = []
    while cap <= _cap_bucket(max_cap):
        caps.append(cap)
        cap *= 2
    for n_b in _CHUNK_GRAPH_EXPERTS:
        ids = torch.zeros(n_b, dtype=torch.int32, device=device)
        for cap_b in caps:
            bucket = torch.zeros(n_b, cap_b, hidden_size, dtype=torch.bfloat16, device=device)
            graph(bucket, ids, *planes)
            del bucket


def _host_routing_plan(topk_ids: torch.Tensor, n_experts: int, chunk: int, bucket_caps: bool = False) -> dict:
    """The plan of :func:`count_ordered_plan`, computed from one read of the routed ids.

    The layer has to read the plan back to size its buckets anyway, so running the counts, the
    sorts and the rank arithmetic on the device only adds about 60 launches per layer. The stable
    sorts give the same order as the device path, so every chunk holds the same experts and every
    slot lands in the same bucket cell.
    """
    import numpy as np

    flat = topk_ids.reshape(-1).cpu().numpy()
    counts = np.bincount(flat, minlength=n_experts)
    order = np.argsort(-counts, kind="stable")
    counts_sorted = counts[order]
    count = int(np.count_nonzero(counts))
    prefix = np.concatenate(([0], np.cumsum(counts_sorted)))
    n_chunks = -(-n_experts // chunk)
    caps = np.maximum.reduceat(counts_sorted, np.arange(0, n_experts, chunk))
    if bucket_caps:
        # H31 chunk graphs: each chunk's rows per expert padded to a power of two (>= 64).
        caps = np.array([_cap_bucket(int(c)) if c > 0 else 0 for c in caps], dtype=caps.dtype)
    ranks = np.zeros(n_experts, dtype=np.int16)
    ranks[order[:count]] = np.arange(count, dtype=np.int16)
    local = ranks[flat]
    slot_order = np.argsort(local, kind="stable")
    local_sorted = local[slot_order].astype(np.int64)
    which = local_sorted // chunk
    cell = (local_sorted - which * chunk) * caps[which] + (np.arange(flat.size, dtype=np.int64) - prefix[local_sorted])
    return {
        "count": count,
        "chunk": int(chunk),
        "n_chunks": int(n_chunks),
        "caps": caps.tolist(),
        "starts": prefix[::chunk].tolist(),
        "selected": order[:count].astype(np.int32),
        "slot_order": slot_order.astype(np.int64),
        "cell": cell,
    }


def apply_routed_experts_host_plan(
    hidden: torch.Tensor,
    num_tokens: int,
    top_k: int,
    topk_weights: torch.Tensor,
    store: "PackedRoutedExperts",
    plan: dict,
) -> torch.Tensor:
    """:func:`apply_routed_experts_grouped_chunked` over a host plan.

    Same chunks, bucket cells and mix order, so the result is bit-identical. Padded bucket rows
    stay uninitialized: their GEMM outputs are never picked and a GEMM row depends only on its
    own input row.
    """
    import numpy as np

    hidden_size = hidden.shape[-1]
    device = hidden.device
    slots = num_tokens * top_k
    out = torch.zeros(num_tokens, hidden_size, dtype=hidden.dtype, device=device)
    slot_out = torch.empty(slots, hidden_size, dtype=hidden.dtype, device=device)
    chunk = plan["chunk"]
    count = plan["count"]
    caps = plan["caps"]
    starts = plan["starts"]
    selected = plan["selected"]
    slot_order_host = plan["slot_order"]
    slot_order = torch.from_numpy(slot_order_host).to(device)
    token_sorted = torch.from_numpy(slot_order_host // top_k).to(device)
    cell_all = torch.from_numpy(plan["cell"]).to(device)
    weight_sorted = topk_weights.reshape(-1).index_select(0, slot_order)
    ensure_gate_up_rowperm(store)
    use_graph = moe_chunk_graph_enabled() and chunk == 64
    k4 = use_graph and moe_dequant_k4_enabled(hidden_size)
    if use_graph and not k4 and rowperm_active():
        # The natural chunk body would read the stored gate/up order as natural; the eager branch
        # below goes through store.dequant, which hands gate/up back in the natural order.
        use_graph = False
    if k4:
        from .decode_graph import _k4_layout

        # The bucket rows feed the k4 gate/up dequant, so the tokens are permuted along K once here.
        x_sorted = _k4_layout(hidden).index_select(0, token_sorted)
    else:
        x_sorted = hidden.index_select(0, token_sorted)
    last = len(starts) - 1
    inflight = moe_prefill_inflight_chunks()
    events: list = []
    if use_graph:
        planes = (store.gate_proj_packed, store.gate_proj_scale, _scale2_on(store, "gate_proj", store.gate_proj_packed),
                  store.up_proj_packed, store.up_proj_scale, _scale2_on(store, "up_proj", store.gate_proj_packed),
                  store.down_proj_packed, store.down_proj_scale,
                  _scale2_on(store, "down_proj", store.gate_proj_packed))
        _warm_chunk_graphs(store, planes, hidden_size, int(os.environ.get("VLLM_QWEN4EXP_MOE_CHUNK_GRAPH_MAX_CAP",
                                                                          "4096")), device, k4)
        graph = _chunk_graph(k4)
    for index in range((count + chunk - 1) // chunk):
        n_c = min(chunk, count - index * chunk)
        cap_c = int(caps[index])
        s0 = int(starts[index])
        s1 = int(starts[min(index + 1, last)])
        if n_c <= 0 or cap_c <= 0 or s1 <= s0:
            continue
        if inflight and len(events) >= inflight:
            # H31: the host queues every chunk long before the card runs them, and a queued chunk's
            # BF16 weights and buckets stay allocated until it has run. Without this wait a layer
            # held all eight chunks at once: 4.6-11 GiB of transient per prefill chunk.
            events.pop(0).synchronize()
        cell = cell_all[s0:s1]
        if use_graph:
            # Fixed-shape bucket: n_b experts (the padding repeats the chunk's last expert and its rows
            # are never picked) by cap_c rows, already a power of two from the plan.
            n_b = _experts_bucket(n_c)
            chosen = selected[index * chunk:index * chunk + n_c]
            if n_b > n_c:
                chosen = np.concatenate([chosen, np.repeat(chosen[-1:], n_b - n_c)])
            ids_c = torch.from_numpy(np.ascontiguousarray(chosen, dtype=np.int32)).to(device)
            bucket = torch.empty(n_b * cap_c, hidden_size, dtype=hidden.dtype, device=device)
            bucket.index_copy_(0, cell, x_sorted[s0:s1])
            projected = graph(bucket.view(n_b, cap_c, hidden_size), ids_c, *planes)
            del bucket
            picked = projected.view(n_b * cap_c, hidden_size).index_select(0, cell)
            del projected
            slot_out[s0:s1] = picked * weight_sorted[s0:s1].unsqueeze(-1)
            if inflight:
                event = torch.hpu.Event()
                event.record()
                events.append(event)
            continue
        ids_c = torch.from_numpy(selected[index * chunk:index * chunk + n_c].copy()).to(device)
        bucket = torch.empty(n_c * cap_c, hidden_size, dtype=hidden.dtype, device=device)
        bucket.index_copy_(0, cell, x_sorted[s0:s1])
        bucketed = bucket.view(n_c, cap_c, hidden_size)
        gate_w = store.dequant("gate_proj", ids_c)
        up_w = store.dequant("up_proj", ids_c)
        gate_out = torch.bmm(bucketed, gate_w.transpose(1, 2))
        up_out = torch.bmm(bucketed, up_w.transpose(1, 2))
        del gate_w, up_w, bucket
        activated = nn.functional.silu(gate_out) * up_out
        del gate_out, up_out
        down_w = store.dequant("down_proj", ids_c)
        projected = torch.bmm(activated, down_w.transpose(1, 2))
        del activated, down_w
        picked = projected.view(n_c * cap_c, hidden_size).index_select(0, cell)
        del projected
        slot_out[s0:s1] = picked * weight_sorted[s0:s1].unsqueeze(-1)
        if inflight:
            event = torch.hpu.Event()
            event.record()
            events.append(event)
    out.index_add_(0, token_sorted, slot_out)
    return out


def moe_prefill_inflight_chunks() -> int:
    """Expert chunks of one prefill MoE layer allowed on the card's queue (``VLLM_QWEN4EXP_MOE_PREFILL_INFLIGHT``).

    0 lets the host queue them all (the H30 behaviour).
    """
    return int(os.environ.get("VLLM_QWEN4EXP_MOE_PREFILL_INFLIGHT", "2") or 0)


def apply_routed_experts_grouped_chunked(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    store: "PackedRoutedExperts",
    selected: torch.Tensor,
    plan: dict,
    mix: bool = True,
) -> torch.Tensor:
    """H18: dequant, bucket and GEMM one count-ordered expert chunk at a time.

    Same three batched GEMMs and the same bucket layout as
    ``apply_routed_experts_grouped``, restricted to ``chunk`` experts and to
    their own slot count. ``cap`` per chunk is a host int from the single
    ``.tolist()`` in :func:`count_ordered_plan`, so every tensor shape stays
    static for a given routing plan and compile-only mode never reaches here.
    """
    num_tokens, top_k = topk_ids.shape
    hidden_size = hidden.shape[-1]
    device = hidden.device
    slots = num_tokens * top_k
    out = torch.zeros(num_tokens, hidden_size, dtype=hidden.dtype, device=device)
    # Per-slot results are assembled in this buffer and mixed with ONE
    # ``index_add_`` at the end, exactly like the H9(a) path. Splitting the mix
    # into one ``index_add_`` per chunk would change the order in which a
    # token's top-k contributions are summed and move the output by ~1 BF16 ulp
    # (measured rel_max 1.35e-3 vs the exact H9(a) output); with the buffer the
    # result is bit-identical while the dequant/BF16 peak still drops to the
    # chunk size.
    slot_out = torch.empty(slots, hidden_size, dtype=hidden.dtype, device=device)
    chunk = int(plan["chunk"])
    count = int(plan["count"])
    caps = plan["caps"]
    starts = plan["starts"]
    local_sorted = plan["local_sorted"]
    rank_sorted = plan["rank_sorted"]
    slot_order = plan["slot_order"]
    # Sorted slot order: the token each slot writes back to, its mixing
    # weight, and its activation row -- gathered once, sliced per chunk.
    token_sorted = torch.arange(num_tokens, device=device).repeat_interleave(top_k) \
        .index_select(0, slot_order)
    weight_sorted = topk_weights.reshape(-1).index_select(0, slot_order)
    x_sorted = hidden.index_select(0, token_sorted)
    n_chunks = (count + chunk - 1) // chunk
    last = len(starts) - 1
    # H22: build the whole layer's bucket in ONE allocation (one cell vector, one
    # ``index_copy_``) instead of once per chunk, and leave the padded rows
    # uninitialized. Chunk-major order is untouched, so the dequant/bmm
    # interleaving that makes H18 fast stays: this only removes glue.
    #   eager, real layer-0/3: per-chunk bucket 0.77-1.25 ms vs 0.28 ms single.
    #   bit-exact: same rows, same cells, same bmm shapes.
    hoisted = prefill_h22_enabled() and n_chunks > 0
    if hoisted:
        _, rows, bases = _h22_chunk_geometry(count, chunk, caps)
        bucket_cell, cell_local = _h22_bucket_cells(plan, chunk, slots, bases, device)
        bucket_all = torch.empty(bases[-1], hidden_size, dtype=hidden.dtype, device=device)
        bucket_all.index_copy_(0, bucket_cell, x_sorted)
    for index in range(n_chunks):
        n_c = min(chunk, count - index * chunk)
        cap_c = int(caps[index])
        s0 = int(starts[index])
        s1 = int(starts[min(index + 1, last)])
        if n_c <= 0 or cap_c <= 0 or s1 <= s0:
            continue
        ids_c = materialized_ids(selected[index * chunk:index * chunk + n_c], device)
        if hoisted:
            # (count-sorted rank - first rank of this chunk) is the row group
            # inside the chunk's bucket, so ``cap_c`` is the only padding this
            # chunk pays; the cell vector already carries the bucket offset.
            bucketed = bucket_all[bases[index]:bases[index] + rows[index]].view(
                n_c, cap_c, hidden_size)
            cell = cell_local[s0:s1]
        else:
            cell = (local_sorted[s0:s1] - index * chunk) * cap_c + rank_sorted[s0:s1]
            bucket = hidden.new_zeros(n_c * cap_c, hidden_size)
            bucket.index_copy_(0, cell, x_sorted[s0:s1])
            bucketed = bucket.view(n_c, cap_c, hidden_size)
        # Two projections in flight, never three: the BF16 peak of a chunk is
        # 2 * chunk * 9.83 MB while gate+up run, and 1 * chunk while down runs.
        gate_w = store.dequant("gate_proj", ids_c)
        up_w = store.dequant("up_proj", ids_c)
        gate_out = torch.bmm(bucketed, gate_w.transpose(1, 2))
        up_out = torch.bmm(bucketed, up_w.transpose(1, 2))
        del gate_w, up_w
        if not hoisted:
            del bucket
        activated = nn.functional.silu(gate_out) * up_out
        del gate_out, up_out
        down_w = store.dequant("down_proj", ids_c)
        projected = torch.bmm(activated, down_w.transpose(1, 2))
        del activated, down_w
        picked = projected.view(n_c * cap_c, hidden_size).index_select(0, cell)
        del projected
        slot_out[s0:s1] = picked * weight_sorted[s0:s1].unsqueeze(-1)
    if hoisted:
        del bucket_all
    if not mix:
        return slot_out
    out.index_add_(0, token_sorted, slot_out)
    return out


def prefill_grouped_h18(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    store: "PackedRoutedExperts",
) -> torch.Tensor | None:
    """H18 prefill routed MoE. ``None`` -> the caller keeps the H9(a) path."""
    if not prefill_h18_enabled() or hidden.device.type != "hpu":
        return None
    if hidden.dtype != torch.bfloat16:
        return None
    num_tokens, top_k = topk_ids.shape
    n_experts = int(store.num_experts)
    slots = num_tokens * top_k
    if slots == 0 or n_experts == 0:
        return None
    if n_experts * slots > 268_435_456:
        # Same guard as apply_routed_experts_grouped: keep the exact loop.
        return None
    if slots < PREFILL_H18_MIN_SLOTS:
        # Small prefill: the per-chunk dispatch is bigger than the padding it
        # removes (see PREFILL_H18_MIN_SLOTS). H9(a) runs unchanged.
        return None
    if moe_host_plan_enabled() and not _static_shape_mode(hidden.device):
        plan = _host_routing_plan(topk_ids, n_experts, prefill_h18_chunk(n_experts, slots),
                                  bucket_caps=moe_chunk_graph_enabled() and prefill_h18_chunk(n_experts, slots) == 64)
        if plan["count"] == 0:
            return None
        return apply_routed_experts_host_plan(hidden, num_tokens, top_k, topk_weights, store, plan)
    selected, local, plan = count_ordered_plan(
        topk_ids, n_experts, prefill_h18_chunk(n_experts, slots))
    if plan is None or plan["count"] == 0:
        return None
    return apply_routed_experts_grouped_chunked(
        hidden, local, topk_weights, store, selected, plan)


# ---------------------------------------------------------------------------
# H22 prefill MoE: projection-first dequant, one bucket, per-chunk tight cap
# ---------------------------------------------------------------------------
def prefill_h22_enabled() -> bool:
    """H22 prefill MoE. **Opt-in** (``VLLM_QWEN4EXP_MOE_PREFILL_H22=1``); H18 is
    the default because every H22 mechanism measured neutral-to-negative in the
    real pipeline (see the end of this docstring).

    H18 cut the zero-padding by ordering experts by descending slot count and
    running ``chunk`` experts at a time, but it pays three costs this unit
    removes, all of them pure glue (the routed dot products are untouched):

    * ``3 * n_chunks`` dequant launches per layer (24 at chunk=64) and ~17 eager
      dispatches per chunk. The same BF16 bytes are written either way: H22
      dequantizes ONE projection for a whole ``prefill_h22_supers`` group of
      chunks and runs every chunk's tight-cap ``bmm`` before the next
      projection starts.
    * two BF16 projections resident at once. Projection-first keeps exactly one
      live (``U_dense * 3.125 MiB``), so the peak no longer doubles for gate+up.
    * one bucket allocation + ``index_copy_`` + ``new_zeros`` PER CHUNK. The
      per-chunk caps are still per chunk, but they concatenate into ONE bucket
      built by one cell vector and one gather (``_h22_bucket_cells``), and the
      padded rows are left uninitialized (``torch.empty``) because a ``bmm``
      output row depends only on its own input row and every padded row's
      output is discarded by the gather that follows.

    Every one of those is bit-exact against H18: the same expert weights meet
    the same activation rows in the same per-chunk ``cap_c`` bucket, the same
    ``slot_out`` buffer collects the same rows, and the single ``index_add_``
    at the end sums them in the same order.

    Measured verdict (``bench_h22_ab.py``, interleaved paired A/B against the
    H18 entry, real layer-0/3 planes, 9-11 pairs per cell, bitdiff 0 in every
    cell). Ratio = H18 ms / H22 ms, so > 1 means H22 is faster:

      projection-first, eager    L0 T>=512 0.4 % .. +2.9 %   (median +0.4 %)
                                 L3 T 512 +2.9 %, T 2048 -0.1 %
      projection-first, lazy     L0 T=512 0.964, T=2048 0.961 (SUPERS 1/2/8 all)
      bucket hoist only, eager   -1.3 % .. +1.0 % over T=128..2048 (L0 and L3),
                                 mean -0.25 %

    Two conclusions, both measured, neither flattering:

    * the sync-attributed pieces say the 24 launches cost 1.9-2.8 ms more than
      3 and that the per-chunk bucket costs 0.5-0.9 ms, but in the pipelined
      eager stream those ops are already overlapped with the next chunk's
      dequant/bmm, so the whole restructure is worth ~0.4 % at best;
    * in lazy mode the projection-first *order* loses 4 %: the graph scheduler
      no longer overlaps TPC dequant with MME bmm.

    Hence the default is H18 (this entry returns ``None``) and H22 is a
    documented dead end with a working, bit-exact implementation behind the
    switch. What it does buy, when switched on, is structure: 3-12 dequant
    launches instead of 24, one projection resident instead of two, one bucket
    per layer instead of eight.
    """
    return os.environ.get("VLLM_QWEN4EXP_MOE_PREFILL_H22", "0") != "0"


#: Below this many slots H22 keeps H18. Interleaved paired A/B on real layer-0
#: routing ("bench_h22_ab.py", 11 pairs, bit-identical at every point):
#:
#:   T=128  (1280 slots, U=293) H18 14.98 ms  H22 15.06 ms  ratio 0.995
#:   T=256  (2560 slots, U=374) H18 18.56 ms  H22 18.89 ms  ratio 0.988
#:   T=512  (5120 slots, U=426) H18 21.15 ms  H22 21.14 ms  ratio 1.004
#:   T=1024 (10240 slots, U=459) H18 23.72 ms H22 23.38 ms  ratio 1.022
#:   T=2048 (20480 slots, U=473) H18 26.51 ms H22 26.24 ms  ratio 1.004
#:
#: so the projection-first order only pays once the prefill is big enough to be
#: dequant-bound; below that the extra resident rows (one bucket for the whole
#: layer, every chunk's gate/activation) cost more than the launch saving.
PREFILL_H22_MIN_SLOTS = 5120


def prefill_h22_order() -> str:
    """Execution order (``VLLM_QWEN4EXP_MOE_PREFILL_H22_ORDER``).

    ``chunk`` (default) keeps H18's chunk-major loop - dequant of chunk ``c+1``
    overlaps the ``bmm`` of chunk ``c`` - and only hoists the layer's bucket into
    one allocation with uninitialized padding. ``projfirst`` runs the
    projection-first body (``apply_routed_experts_grouped_h22``): 3 dequant
    launches for the layer, one projection live, every chunk still on its own
    tight cap. Measured, real layer-0 planes, interleaved paired A/B:

      ``chunk``    eager  -1.3 % .. +1.0 % (mean -0.25 %): H18's op sequence,
                   one bucket per layer instead of eight
      ``projfirst`` eager  +0.4 % .. +2.9 % at T>=512, negative below
                            (``PREFILL_H22_MIN_SLOTS``); lazy **-4 %**

    Projection-first is therefore opt-in: in lazy mode the 3-phase structure
    loses the TPC/MME overlap that the chunk-major order gives the graph
    scheduler, and no ``SUPERS`` value recovers it (measured 1/2/8: 0.964 /
    0.948 / 0.961 ratios at T=512/2048).
    """
    raw = os.environ.get("VLLM_QWEN4EXP_MOE_PREFILL_H22_ORDER", "chunk").strip().lower()
    return raw if raw in ("chunk", "projfirst") else "chunk"


def prefill_h22_supers(n_chunks: int) -> int:
    """Expert groups per projection (``VLLM_QWEN4EXP_MOE_PREFILL_H22_SUPERS``).

    ``1`` = one dequant launch per projection (3 per layer, one BF16
    projection live). Larger values trade launches back for a smaller BF16
    peak; the value is clamped to ``n_chunks``, where it reproduces H18's
    launch count with half of H18's BF16 peak.
    """
    raw = os.environ.get("VLLM_QWEN4EXP_MOE_PREFILL_H22_SUPERS", "1")
    try:
        supers = int(raw)
    except (TypeError, ValueError):
        supers = 1
    return max(1, min(max(1, int(n_chunks)), supers))


def _h22_chunk_geometry(count: int, chunk: int, caps) -> tuple[int, list[int], list[int]]:
    """``(n_chunks, bucket_rows_per_chunk, bucket_row_base)`` for a plan."""
    n_chunks = (int(count) + chunk - 1) // chunk
    rows = [min(chunk, int(count) - index * chunk) * int(caps[index]) for index in range(n_chunks)]
    bases = [0]
    for row in rows:
        bases.append(bases[-1] + row)
    return n_chunks, rows, bases


def _h22_bucket_cells(plan: dict, chunk: int, slots: int, bases: list[int],
                      device) -> tuple[torch.Tensor, torch.Tensor]:
    """Bucket cells of the whole count-sorted slot range, in one expression.

    ``cell`` indexes the single concatenated bucket (``[sum_c n_c * cap_c, H]``);
    ``cell_local`` indexes the flattened ``[n_c * cap_c, H]`` projection of the
    chunk a slot belongs to, which is what the epilogue gathers with. ``caps``
    and ``bases`` become static int64 vectors, so this costs five elementwise
    ops on ``[slots]`` regardless of how many chunks the plan has.
    """
    rank = plan["local_sorted"][:slots]
    caps_t = torch.tensor([int(v) for v in plan["caps"][:len(bases) - 1]],
                          dtype=torch.int64, device=device)
    base_t = torch.tensor(bases[:-1], dtype=torch.int64, device=device)
    which = rank // chunk
    cap_v = caps_t.index_select(0, which)
    base_v = base_t.index_select(0, which)
    cell_local = (rank - which * chunk) * cap_v + plan["rank_sorted"][:slots]
    return cell_local + base_v, cell_local


def _h22_super_ranges(n_chunks: int) -> list[tuple[int, int]]:
    """Chunk ranges that share one dequant launch (see ``prefill_h22_supers``)."""
    if n_chunks <= 0:
        return []
    per_group = max(1, -(-n_chunks // prefill_h22_supers(n_chunks)))
    return [(lo, min(n_chunks, lo + per_group)) for lo in range(0, n_chunks, per_group)]


def apply_routed_experts_grouped_h22(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    store: "PackedRoutedExperts",
    selected: torch.Tensor,
    plan: dict,
    mix: bool = True,
) -> torch.Tensor:
    """H22: one bucket for the layer, one dequant launch per projection group.

    ``topk_ids`` carries the count-sorted *rank* of every slot (``plan``'s
    ``local``), exactly like ``apply_routed_experts_grouped_chunked``, and
    ``selected`` maps a rank back to the checkpoint expert id.
    """
    num_tokens, top_k = topk_ids.shape
    hidden_size = hidden.shape[-1]
    device = hidden.device
    slots = num_tokens * top_k
    count = int(plan["count"])
    chunk = int(plan["chunk"])
    caps = [int(v) for v in plan["caps"]]
    starts = [int(v) for v in plan["starts"][:max(1, (count + chunk - 1) // chunk)]] + [slots]
    n_chunks, rows, bases = _h22_chunk_geometry(count, chunk, caps)
    if n_chunks <= 0:
        # No routed slot has a present expert: every contribution is zero.
        return torch.zeros(num_tokens, hidden_size, dtype=hidden.dtype, device=device)
    slot_out = torch.empty(slots, hidden_size, dtype=hidden.dtype, device=device)
    # Sorted slot order: the token each slot writes back to, its mixing weight
    # and its activation row, gathered ONCE for the whole layer.
    token_sorted = torch.arange(num_tokens, device=device).repeat_interleave(top_k) \
        .index_select(0, plan["slot_order"])
    weight_sorted = topk_weights.reshape(-1).index_select(0, plan["slot_order"])
    x_sorted = hidden.index_select(0, token_sorted)
    cell, cell_local = _h22_bucket_cells(plan, chunk, slots, bases, device)
    # ``empty``: a padded row's bmm output is thrown away, so its input
    # never has to be zeroed (H18 zero-filled 8 buckets per layer).
    bucket = torch.empty(bases[-1], hidden_size, dtype=hidden.dtype, device=device)
    bucket.index_copy_(0, cell, x_sorted)
    groups = _h22_super_ranges(n_chunks)
    gates: list[torch.Tensor | None] = [None] * n_chunks
    activated: list[torch.Tensor | None] = [None] * n_chunks
    # 1) gate: one dequant launch per group, then that group's tight caps.
    for lo, hi in groups:
        gate_w = store.dequant("gate_proj", materialized_ids(
            selected[lo * chunk:min(count, hi * chunk)], device))
        for index in range(lo, hi):
            n_c = min(chunk, count - index * chunk)
            x_c = bucket[bases[index]:bases[index] + rows[index]].view(
                n_c, caps[index], hidden_size)
            gates[index] = torch.bmm(
                x_c, gate_w[(index - lo) * chunk:(index - lo) * chunk + n_c].transpose(1, 2))
        del gate_w
    # 2) up + SwiGLU. The gate rows die here, never held with down.
    for lo, hi in groups:
        up_w = store.dequant("up_proj", materialized_ids(
            selected[lo * chunk:min(count, hi * chunk)], device))
        for index in range(lo, hi):
            n_c = min(chunk, count - index * chunk)
            x_c = bucket[bases[index]:bases[index] + rows[index]].view(
                n_c, caps[index], hidden_size)
            up_out = torch.bmm(
                x_c, up_w[(index - lo) * chunk:(index - lo) * chunk + n_c].transpose(1, 2))
            activated[index] = nn.functional.silu(gates[index]) * up_out
            gates[index] = None
            del up_out
        del up_w
    # 3) down + weighted gather into the sorted slot buffer.
    for lo, hi in groups:
        down_w = store.dequant("down_proj", materialized_ids(
            selected[lo * chunk:min(count, hi * chunk)], device))
        for index in range(lo, hi):
            n_c = min(chunk, count - index * chunk)
            projected = torch.bmm(
                activated[index],
                down_w[(index - lo) * chunk:(index - lo) * chunk + n_c].transpose(1, 2))
            activated[index] = None
            picked = projected.view(rows[index], hidden_size) \
                .index_select(0, cell_local[starts[index]:starts[index + 1]])
            slot_out[starts[index]:starts[index + 1]] = picked * weight_sorted[
                starts[index]:starts[index + 1]].unsqueeze(-1)
            del projected
        del down_w
    del bucket
    if not mix:
        return slot_out
    out = torch.zeros(num_tokens, hidden_size, dtype=hidden.dtype, device=device)
    out.index_add_(0, token_sorted, slot_out)
    return out


def prefill_grouped_h22(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    store: "PackedRoutedExperts",
) -> torch.Tensor | None:
    """H22 prefill routed MoE. ``None`` -> the caller tries H18, then H9(a).

    The guards are H18's, in H18's order, and the H18 kill switch also disables
    this one: ``VLLM_QWEN4EXP_MOE_PREFILL_H18=0`` still restores the H9(a) path
    exactly, while ``..._H22=0`` restores H18.
    """
    if not prefill_h22_enabled() or not prefill_h18_enabled():
        return None
    if hidden.device.type != "hpu" or hidden.dtype != torch.bfloat16:
        return None
    num_tokens, top_k = topk_ids.shape
    n_experts = int(store.num_experts)
    slots = num_tokens * top_k
    if slots == 0 or n_experts == 0:
        return None
    if n_experts * slots > 268_435_456:
        return None
    order = prefill_h22_order()
    floor = PREFILL_H18_MIN_SLOTS
    if order == "projfirst":
        # Projection-first only pays where the prefill is dequant-bound; below
        # PREFILL_H22_MIN_SLOTS it loses 0.5-1.2 % (see the table there).
        floor = max(floor, PREFILL_H22_MIN_SLOTS)
    if slots < floor:
        return None
    selected, local, plan = count_ordered_plan(
        topk_ids, n_experts, prefill_h18_chunk(n_experts, slots))
    if plan is None or plan["count"] == 0:
        return None
    if order == "projfirst":
        return apply_routed_experts_grouped_h22(
            hidden, local, topk_weights, store, selected, plan)
    # Default: H18's chunk-major loop with the H22 bucket hoist (applied inside
    # ``apply_routed_experts_grouped_chunked`` when this unit is enabled).
    return apply_routed_experts_grouped_chunked(
        hidden, local, topk_weights, store, selected, plan)


def compact_expert_ids(
    topk_ids: torch.Tensor,
    n_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map expert ids to a dense ``0..U-1`` range without data-dependent shapes.

    ``torch.unique``, ``nonzero`` and ``scatter_`` either allocate a workspace
    proportional to the id domain or fail Gaudi2 graph compile inside the
    engine (synStatus 26) even though they run in a bare process. Ids are a
    closed domain ``[0, n_experts)``. Presence is a static ``[E]`` mask; the
    selected count is read once as a Python int, the same way the grouped
    bucket reads ``cap``, and the id list is a prefix of a static argsort.
    ``unique`` is the selected expert ids in ascending order; ``local``
    indexes that list. Absent ranks are never read.

    In compile-only mode / HPU graph capture (``_static_shape_mode``) the
    host read is skipped entirely and ``count`` is the static
    ``n_experts``: ``selected`` is then the whole argsort order (present
    experts first, ascending; absent experts after, also ascending). This
    is still exact if executed -- a present expert keeps the same rank it
    has in the exact-count list, so ``local`` for every routed slot is
    unchanged, and the extra absent experts dequantized downstream simply
    receive no tokens in ``apply_routed_experts[_grouped]``. Outside that
    mode the exact count is kept, clamped into ``[0, n_experts]`` as
    defense-in-depth against a stray unexecuted read.
    """
    flat = topk_ids.reshape(-1).to(dtype=torch.int64)
    if flat.numel() == 0 or n_experts <= 0:
        empty = torch.empty(0, dtype=torch.int32, device=topk_ids.device)
        return empty, torch.zeros_like(topk_ids, dtype=torch.int32)
    experts = torch.arange(n_experts, dtype=torch.int64, device=flat.device)
    present = (flat.unsqueeze(1) == experts).any(dim=0)
    # Stable descending sort: present experts (1) come first, and ties keep
    # ascending ids, which is what ``nonzero`` on the mask used to return.
    order = torch.argsort(present.to(torch.int32), descending=True, stable=True)
    if _static_shape_mode(flat.device):
        count = int(n_experts)
    else:
        count = min(n_experts, max(0, int(present.sum())))
    selected = order[:count].to(dtype=torch.int32)
    ranks = torch.zeros(n_experts, dtype=torch.int32, device=flat.device)
    if count:
        ranks.index_copy_(
            0,
            selected.to(dtype=torch.int64),
            torch.arange(count, dtype=torch.int32, device=flat.device),
        )
    return selected, ranks.index_select(0, flat).view_as(topk_ids)


def use_grouped_slots(num_tokens: int, top_k: int) -> bool:
    """Decode-sized batches group every selected slot into one GEMM per proj.

    Strata's speed on a single token is not a coarser quant: it is one launch
    for the whole expert group instead of one graph per expert. The weights
    here stay the bit-exact NVFP4→BF16 rows (E2M1 × per-16 E4M3, scale2 folded
    after the cast). IQ4 / FP8 resident GEMV would drop that scale granularity,
    so it is not used. Prefill keeps the unique-expert loop, which already
    GEMMs many tokens per expert.
    """
    if os.environ.get("VLLM_QWEN4EXP_MOE_GROUPED", "1") == "0":
        return False
    return 0 < num_tokens <= 8 and num_tokens * top_k <= DECODE_MAX_SLOTS


def apply_grouped_slots(
    hidden: torch.Tensor,
    weights: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU over per-token slots. ``gate``/``up`` are ``[T, K, I, H]``.

    ``down`` is ``[T, K, H, I]``. Each output element is the same dot as the
    per-expert ``linear`` in ``apply_routed_experts``; the K slots of one
    token share one matmul so the launch is not paid K times.
    """
    tokens, k = weights.shape
    intermediate = gate.shape[2]
    hidden_size = hidden.shape[-1]
    outs = []
    for t in range(tokens):
        token = hidden[t:t + 1]
        stacked_g = gate[t].reshape(k * intermediate, hidden_size)
        stacked_u = up[t].reshape(k * intermediate, hidden_size)
        activated = nn.functional.silu(nn.functional.linear(token, stacked_g))
        activated = activated.view(k, intermediate)
        activated = activated * nn.functional.linear(token, stacked_u).view(k, intermediate)
        projected = torch.bmm(activated.unsqueeze(1), down[t].transpose(1, 2)).squeeze(1)
        mix = weights[t].to(dtype=projected.dtype).unsqueeze(-1)
        outs.append((projected * mix).sum(dim=0, keepdim=True))
    return torch.cat(outs, dim=0)


def _gemv_op():
    """Fused NVFP4 GEMV, loaded from the same extension as indexed dequant."""
    namespace = torch.ops.custom_op
    name = "custom_qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2"
    if hasattr(namespace, name):
        return getattr(namespace, name)
    _ensure_dequant_op()
    return getattr(namespace, name, None)


def nvfp4_gemv_enabled() -> bool:
    """On by default: the kernel matches the exact BF16 decode and skips the materialize."""
    return os.environ.get("VLLM_QWEN4EXP_NVFP4_GEMV", "1") != "0"


def _scale2_on(experts: PackedRoutedExperts, kind: str, like: torch.Tensor) -> torch.Tensor:
    scale2 = getattr(experts, f"{kind}_scale2")
    # The GEMV contract takes [E] F32. ``module.to(dtype=...)`` casts every
    # floating buffer, so a model-wide bf16 cast turns scale2 into bf16; the
    # materialize path re-casts it, and this path must do the same.
    if scale2.device != like.device or scale2.dtype != torch.float32:
        scale2 = scale2.to(device=like.device, dtype=torch.float32)
        # In compile-only warmup / graph capture the cast never executes;
        # caching it would hand an empty tensor to the first real GEMV.
        if not _static_shape_mode(like.device):
            setattr(experts, f"{kind}_scale2", scale2)
    return scale2


def gemv_slots(
    experts: PackedRoutedExperts,
    kind: str,
    ids: torch.Tensor,
    activation: torch.Tensor,
) -> torch.Tensor:
    """``activation`` against selected experts. Returns ``[S, N]`` BF16.

    ``activation`` may be per-slot ``[S, K]`` or per-token ``[T, K]`` with
    ``S = T * top_k`` (H10): the kernel maps slot ``s`` to x row ``s`` in the
    first case and ``s // top_k`` in the second, so gate/up callers pass the
    token rows directly without replicating them per slot.

    scale2 is applied inside the kernel after the F32 dot, so the caller must
    not fold it again.
    """
    op = _gemv_op()
    if op is None:
        raise RuntimeError("NVFP4 GEMV custom op is not registered")
    slots = int(ids.numel())
    x_rows = int(activation.shape[0]) if activation.dim() >= 1 else 0
    if activation.dim() != 2 or x_rows < 1 or x_rows > slots or slots % x_rows != 0:
        raise ValueError(
            f"GEMV activation must be [S, K] or [T, K] with S % T == 0, "
            f"got {tuple(activation.shape)} for S={slots}")
    ensure_gate_up_rowperm(experts)
    packed = getattr(experts, f"{kind}_packed")
    out = op(
        materialized_ids(ids, packed.device),
        packed,
        getattr(experts, f"{kind}_scale"),
        _materialized(activation),
        _scale2_on(experts, kind, packed),
    )
    return gate_up_natural(experts, out) if kind in ("gate_proj", "up_proj") else out


#: Slot count above which the fused H15 SwiGLU GEMV is replaced by two
#: standalone GEMV launches (H22). Default 512 = never replace it.
#:
#: H21 Phase 2 left this as the next candidate because the fused kernel runs
#: two passes per member (gate, then up) and therefore has half the member
#: parallelism of two standalone launches. Measured on Gaudi2 with real layer-0
#: planes (``bench_h22_decode_swiglu.py``), S = slots:
#:
#:   S=10  fused 0.232 ms  2 x gemv 0.269 ms  (+16 %)   fused wins
#:   S=40  fused 0.556 ms  2 x gemv 0.523 ms  ( -6 %)   2 x gemv wins
#:   S=80  fused 1.038 ms  2 x gemv 0.946 ms  ( -9 %)   2 x gemv wins
#:
#: It is NOT bit-exact, which is why the default stays at the fused form. The
#: fused kernel evaluates silu in F32 on the BF16-rounded gate with an accurate
#: exp + Newton reciprocal and rounds the result to BF16, i.e. it reproduces
#: EXACT silu (0/25600 differing against a CPU reference, H15). The
#: ``F.silu`` that a two-launch chain must use is the device BF16 silu kernel,
#: whose sigmoid is the hardware approximation, and it moves ~29 % of the
#: activation elements by one BF16 ulp (measured 2321/6400 at S=10, 9293/25600
#: at S=40, 18610/51200 at S=80; 6475/10240 block outputs at T=4). An explicit
#: F32 emulation ``(silu(gate.float()) * up.float()).bf16()`` still differs on
#: 26 % of elements -- HPU's F32 sigmoid is the same approximation -- so no
#: torch-side form reproduces the kernel.
#:
#: Set ``VLLM_QWEN4EXP_MOE_SWIGLU_GEMV_MAX_SLOTS=10`` to take the 6-9 % at
#: S>=40 and accept the 1-ulp decode change; ``0`` always uses two launches.
SWIGLU_GEMV_MAX_SLOTS = 512


def swiglu_gemv_max_slots() -> int:
    """``VLLM_QWEN4EXP_MOE_SWIGLU_GEMV_MAX_SLOTS`` (default 512 = always fused)."""
    raw = os.environ.get("VLLM_QWEN4EXP_MOE_SWIGLU_GEMV_MAX_SLOTS", str(SWIGLU_GEMV_MAX_SLOTS))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return SWIGLU_GEMV_MAX_SLOTS


def swiglu_gemv_slots(
    experts: PackedRoutedExperts,
    ids: torch.Tensor,
    activation: torch.Tensor,
) -> torch.Tensor | None:
    """H15 fused gate+up SwiGLU GEMV. Returns ``[S, N]`` BF16 or ``None``.

    One kernel launch replaces gate GEMV + up GEMV + ``F.silu`` + ``*``:
    the activation tile is loaded once per K part and reused by the four
    gate and four up rows, gate/up are BF16-rounded in-kernel (matching the
    standalone GEMV stores) and ``silu(gate_bf16) * up_bf16`` is rounded to
    BF16 before the single output store. ``None`` means the op is not
    registered (old extension); the caller keeps the pre-H15 chain.
    """
    op = _h15_op("custom_qwen4exp_nvfp4_indexed_swiglu_gemv_bf16_gaudi2")
    if op is None:
        return None
    slots = int(ids.numel())
    x_rows = int(activation.shape[0]) if activation.dim() >= 1 else 0
    if activation.dim() != 2 or x_rows < 1 or x_rows > slots or slots % x_rows != 0:
        raise ValueError(
            f"SwiGLU GEMV activation must be [S, K] or [T, K] with S % T == 0, "
            f"got {tuple(activation.shape)} for S={slots}")
    ensure_gate_up_rowperm(experts)
    gate_packed = experts.gate_proj_packed
    return gate_up_natural(experts, op(
        materialized_ids(ids, gate_packed.device),
        gate_packed,
        experts.gate_proj_scale,
        _scale2_on(experts, "gate_proj", gate_packed),
        experts.up_proj_packed,
        experts.up_proj_scale,
        _scale2_on(experts, "up_proj", gate_packed),
        _materialized(activation),
    ))


def down_epilogue(
    slot_down: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor | None:
    """H15 fused ``slot_down * w`` + top-k sum. ``[S, H]``, ``[T, K]`` -> ``[T, H]``."""
    op = _h15_op("custom_qwen4exp_moe_down_epilogue_bf16_gaudi2")
    if op is None:
        return None
    weights = topk_weights
    if weights.dtype != slot_down.dtype:
        weights = weights.to(dtype=slot_down.dtype)
    return op(_materialized(slot_down), _materialized(weights))


def shared_expert_h15_parts(shared_expert, expert_gate=None):
    """Resolve the weights the fused shared expert needs, or ``None``.

    Works for both spellings the port uses:

    * ``Qwen2MoeMLP`` (imported as ``Qwen3NextMLP`` in ``qwen4_exp.py`` and
      built with ``expert_gate=``) keeps the scalar gate on the MLP as
      ``expert_gate``;
    * a block that keeps it beside the MLP passes its ``shared_expert_gate``
      as ``expert_gate``.

    The MLP attribute wins when both are present, so both spellings feed the
    same weights. Anything the fused form does not cover -- a quantized
    (Track B FP8/MXFP8) or otherwise non-BF16 weight, a bias, a gate whose
    width does not match the MLP, a ``[2I, H]``/``[H, I]`` shape mismatch --
    returns ``None`` and the caller keeps the stock MLP.
    """
    gate_up = getattr(shared_expert, "gate_up_proj", None)
    down = getattr(shared_expert, "down_proj", None)
    on_mlp = getattr(shared_expert, "expert_gate", None)
    if on_mlp is not None:
        expert_gate = on_mlp
    if gate_up is None or down is None or expert_gate is None:
        return None
    gate_w = getattr(gate_up, "weight", None)
    down_w = getattr(down, "weight", None)
    shared_w = getattr(expert_gate, "weight", None)
    if gate_w is None or down_w is None or shared_w is None:
        return None
    if getattr(gate_up, "bias", None) is not None or getattr(expert_gate, "bias", None) is not None:
        return None
    if (gate_w.dim() != 2 or shared_w.shape != (1, gate_w.shape[1])
            or down_w.shape != (gate_w.shape[1], gate_w.shape[0] // 2)
            or gate_w.dtype != torch.bfloat16 or down_w.dtype != torch.bfloat16
            or shared_w.dtype != torch.bfloat16):
        return None
    return gate_up, down, gate_w, down_w, shared_w


def shared_expert_forward_h15(shared_expert, hidden: torch.Tensor, expert_gate=None) -> torch.Tensor | None:
    """H15 shared expert in 3 launches, or ``None`` to keep the stock MLP.

    merged GEMM [1281, H] (gate_up ++ shared_expert_gate, concatenated once
    in memory and cached keyed by weight pointer/version -- zero disk
    rewrite) -> fused SwiGLU+sigmoid-scalar TPC epilogue -> down GEMM.
    The per-token ``sigmoid(shared_gate)`` scalar moves in front of the
    linear down projection (exact in real arithmetic; one BF16 rounding
    relocates, checked at MoE-block tolerance in bench_moe_h15.py).

    ``expert_gate`` carries the scalar gate when it lives beside the MLP;
    :func:`shared_expert_h15_parts` resolves both that and the MLP's own
    ``expert_gate`` and rejects any shape/bias/dtype it cannot cover, so a
    quantized (Track B) or host shared expert keeps the stock MLP.
    """
    if not moe_h15_enabled() or hidden.device.type != "hpu" or shared_expert is None:
        return None
    if hidden.dtype != torch.bfloat16:
        return None
    parts = shared_expert_h15_parts(shared_expert, expert_gate)
    if parts is None:
        return None
    down, gate_w, down_w, shared_w = parts[1:]
    op = _h15_op("custom_qwen4exp_moe_shared_swiglu_bf16_gaudi2")
    if op is None:
        return None
    key = (gate_w.data_ptr(), getattr(gate_w, "_version", 0), shared_w.data_ptr(),
           getattr(shared_w, "_version", 0), tuple(gate_w.shape), str(gate_w.device))
    merged = getattr(shared_expert, "_h15_merged_weight", None)
    from .merged_weights import views_of

    if merged is not None and views_of(merged, gate_w, shared_w):
        shared_expert._h15_merged_key = key  # current: the parameters are its row blocks
    elif merged is None or getattr(shared_expert, "_h15_merged_key", None) != key:
        # In compile-only warmup / graph capture the cat never executes;
        # caching it would hand an empty weight to the first real GEMM.
        if _static_shape_mode(gate_w.device):
            merged = torch.cat([gate_w.detach(), shared_w.detach()], dim=0).contiguous()
        else:
            from .merged_weights import merge_rows

            merged = merge_rows(gate_w, shared_w)
            if gate_w.data_ptr() == merged.data_ptr():
                # H31: gate_w/shared_w are the parameters, now views of ``merged``.
                key = (gate_w.data_ptr(), getattr(gate_w, "_version", 0), shared_w.data_ptr(),
                       getattr(shared_w, "_version", 0), tuple(gate_w.shape), str(gate_w.device))
            shared_expert._h15_merged_weight = merged
            shared_expert._h15_merged_key = key
    merged_out = nn.functional.linear(hidden, merged)
    activated = op(_materialized(merged_out))
    down_bias = getattr(down, "bias", None)
    return nn.functional.linear(activated, down_w, down_bias)


def grouped_gemv(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    experts: PackedRoutedExperts,
) -> torch.Tensor:
    """Decode SwiGLU without writing the BF16 expert rows.

    H10: gate/up receive ``hidden`` ``[T, H]`` directly -- the GEMV kernel
    maps each slot to its token row (``slot // top_k``), so the per-slot
    replication of the token rows is gone (five helper ops fewer per MoE
    layer than the replicated/scatter form). Down still receives the
    per-slot ``[S, I]`` activation. The mix is a contiguous ``view`` +
    ``sum`` over the slot axis rather than a zero-filled scatter-add.
    """
    token_count, top_k = topk_ids.shape
    hidden_size = hidden.shape[-1]
    flat_ids = topk_ids.reshape(-1).to(dtype=torch.int32).contiguous()
    if moe_h15_enabled() and hidden.device.type == "hpu":
        # H15: one fused SwiGLU GEMV (x tile shared by gate+up) and one fused
        # weighted-reduction epilogue after down. Both are BF16-bit-exact
        # against the pre-H15 chain below (bench_moe_h15.py); if an op is
        # missing from the extension the chain below runs unchanged.
        # H22: the fused kernel has half the member parallelism (two passes per
        # member), so above SWIGLU_GEMV_MAX_SLOTS the two standalone launches
        # win; they are bit-identical (see the constant).
        activated = None
        if int(flat_ids.numel()) <= swiglu_gemv_max_slots():
            activated = swiglu_gemv_slots(experts, flat_ids, hidden)
        if activated is None:
            activated = nn.functional.silu(
                gemv_slots(experts, "gate_proj", flat_ids, hidden)) \
                * gemv_slots(experts, "up_proj", flat_ids, hidden)
        slot_down = gemv_slots(experts, "down_proj", flat_ids, activated)
        reduced = down_epilogue(slot_down, topk_weights)
        if reduced is not None:
            return reduced
        flat_weights = topk_weights.reshape(-1, 1)
        slot_down = slot_down * flat_weights.to(dtype=slot_down.dtype)
        if token_count == 1:
            return slot_down.sum(dim=0, keepdim=True)
        return slot_down.view(token_count, top_k, hidden_size).sum(dim=1)
    flat_weights = topk_weights.reshape(-1, 1)
    slot_gate = gemv_slots(experts, "gate_proj", flat_ids, hidden)
    slot_up = gemv_slots(experts, "up_proj", flat_ids, hidden)
    activated = nn.functional.silu(slot_gate) * slot_up
    slot_down = gemv_slots(experts, "down_proj", flat_ids, activated)
    slot_down = slot_down * flat_weights.to(dtype=slot_down.dtype)
    if token_count == 1:
        return slot_down.sum(dim=0, keepdim=True)
    return slot_down.view(token_count, top_k, hidden_size).sum(dim=1)


def grouped_from_ids(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    experts: PackedRoutedExperts,
) -> torch.Tensor:
    """Dequant the top-k slots and mix them.

    On HPU the default is the fused GEMV: the same E2M1×E4M3 product, dotted
    in F32, with scale2 applied after the dot. ``VLLM_QWEN4EXP_NVFP4_GEMV=0``
    keeps the materialize-then-GEMM path. Duplicate slots stay separate, so
    the mix matches compacting the expert first and adding its weights.
    """
    if nvfp4_gemv_enabled() and hidden.device.type == "hpu" and _gemv_op() is not None:
        return grouped_gemv(hidden, topk_ids, topk_weights, experts)
    tokens, k = topk_ids.shape
    ids = topk_ids.reshape(-1).contiguous()
    gate = experts.dequant("gate_proj", ids)
    up = experts.dequant("up_proj", ids)
    down = experts.dequant("down_proj", ids)
    hidden_size = hidden.shape[-1]
    intermediate = gate.shape[1]
    gate = gate.view(tokens, k, intermediate, hidden_size)
    up = up.view(tokens, k, intermediate, hidden_size)
    down = down.view(tokens, k, hidden_size, down.shape[-1])
    return apply_grouped_slots(hidden, topk_weights, gate, up, down)
