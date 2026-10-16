"""Prefix caching for hybrid GDN models whose recurrent state lives in compact slots (VLLM_HPU_PREFIX_STATE_CACHE).

vLLM's own Mamba prefix caching ("align" mode) keeps a recurrent state in a KV-cache block at every cached block
boundary, and the model runner reads and writes states through those blocks. The HPU runner keeps GDN state in
compact per-request slots (plus one checkpoint per DFlash2 verification token) instead, and the decode kernels
depend on that layout. So here:

- attention KV (target and draft groups) goes through vLLM's prefix caching unchanged;
- GDN states are kept as a few device snapshots at chosen block boundaries, keyed by vLLM's block hash of the
  prefix ending there (``SnapshotRegistry``, scheduler side; ``GdnSnapshotStore``, runner side);
- the Mamba groups report a cache hit only where a snapshot exists, so the reconciled hit length always lands on
  one, and the request's first chunk starts from the restored state;
- prefill chunks stop at the snapshot positions: every ``interval`` tokens and the last block boundary before the
  prompt end, which is where a conversation that appends to its previous prompt resumes.

The scheduler attaches each step's copies to the ``SchedulerOutput`` (``hpu_prefix_state``); the runner applies
them in ``_update_states`` before the step's forward, so a save captures the state at the start of the chunk.
DFlash2 draft KV at position p depends only on the target features at p, so it needs no EAGLE-style last-block
drop and the draft group is reused like the target groups.
"""
from collections import OrderedDict

import torch

from vllm_gaudi import envs as gaudi_envs
from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()

STATE_ATTR = "hpu_prefix_state"
_DEBUG = [int(__import__("os").environ.get("VLLM_HPU_PREFIX_STATE_DEBUG", "0"))]  # log this many steps


def _debug(msg, *args):
    if _DEBUG[0] > 0:
        _DEBUG[0] -= 1
        logger.info("prefix-state debug: " + msg, *args)


def enabled() -> bool:
    return bool(gaudi_envs.VLLM_HPU_PREFIX_STATE_CACHE)


def block_state_caching(cache_config) -> bool:
    """Whether vLLM's own per-block Mamba state caching is active (off when snapshots replace it)."""
    return bool(getattr(cache_config, "enable_prefix_caching", False)) and not enabled()


def final_boundary(prefill_end: int, block_size: int) -> int:
    """Last block boundary strictly before the end of the prefill (prompt, or prompt + replayed output)."""
    return (prefill_end - 1) // block_size * block_size


def snapshot_positions(start: int, end: int, prefill_end: int, block_size: int, interval: int) -> list[int]:
    """Snapshot positions p with start < p < end (chunk stops)."""
    stops = []
    p = (start // interval + 1) * interval
    if p < end and p < prefill_end:
        stops.append(p)
    fb = final_boundary(prefill_end, block_size)
    if start < fb < end:
        stops.append(fb)
    return sorted(stops)


def is_snapshot_position(pos: int, prefill_end: int, block_size: int, interval: int) -> bool:
    return 0 < pos < prefill_end and pos % block_size == 0 and (pos % interval == 0
                                                               or pos == final_boundary(prefill_end, block_size))


class SnapshotRegistry:
    """Block hash at a boundary -> snapshot slot, least recently used first."""

    def __init__(self, num_slots: int):
        if num_slots <= 0:
            raise ValueError("VLLM_HPU_PREFIX_STATE_SLOTS must be positive")
        self.num_slots = num_slots
        self._free = list(range(num_slots - 1, -1, -1))
        self._slots: OrderedDict = OrderedDict()
        self.stats = {"hits": 0, "hit_tokens": 0, "saves": 0, "evictions": 0}

    def __contains__(self, key) -> bool:
        return key in self._slots

    def __len__(self) -> int:
        return len(self._slots)

    def slot(self, key):
        return self._slots.get(key)

    def touch(self, key) -> None:
        self._slots.move_to_end(key)

    def hit_blocks(self, block_hashes, max_blocks: int) -> int:
        """Largest n <= max_blocks such that the prefix of n blocks has a snapshot."""
        for i in range(min(max_blocks, len(block_hashes)) - 1, -1, -1):
            if block_hashes[i] in self._slots:
                return i + 1
        return 0

    def acquire(self, key, pinned: set) -> int | None:
        """Slot for a new snapshot of ``key``; None when it is already kept or every slot is pinned."""
        if key in self._slots:
            self._slots.move_to_end(key)
            return None
        if self._free:
            slot = self._free.pop()
        else:
            victim = next((k for k, s in self._slots.items() if s not in pinned), None)
            if victim is None:
                return None
            slot = self._slots.pop(victim)
            self.stats["evictions"] += 1
        self._slots[key] = slot
        self.stats["saves"] += 1
        return slot

    def drop_slot(self, slot: int) -> None:
        for key, s in list(self._slots.items()):
            if s == slot:
                del self._slots[key]
                self._free.append(slot)


_registry: SnapshotRegistry | None = None


def registry() -> SnapshotRegistry | None:
    return _registry


# --------------------------------------------------------------------------- scheduler side


def plan_step(scheduler, scheduler_output) -> dict | None:
    """Restores for requests whose first scheduled chunk starts at a cache hit, saves at snapshot positions.

    Runs before ``_update_after_schedule`` advances ``num_computed_tokens``, so ``request.num_computed_tokens`` is
    where this step's chunk starts."""
    reg = _registry
    if reg is None:
        _debug("no registry")
        return None
    # the scheduler's own block size: cache_config.block_size can be reset by a later config re-validation
    block_size = scheduler.block_size
    interval = gaudi_envs.VLLM_HPU_PREFIX_STATE_INTERVAL
    first = [data.req_id for data in scheduler_output.scheduled_new_reqs]
    cached = scheduler_output.scheduled_cached_reqs
    first += [req_id for req_id in cached.req_ids if req_id in cached.resumed_req_ids]
    restore, save, pinned = [], [], set()
    for req_id in first:
        request = scheduler.requests[req_id]
        start = request.num_computed_tokens
        if start <= 0:
            continue
        if start % block_size != 0 or start // block_size > len(request.block_hashes):
            raise RuntimeError(f"prefix hit of {start} tokens for {req_id} is not on a snapshot boundary")
        key = request.block_hashes[start // block_size - 1]
        slot = reg.slot(key)
        if slot is None:
            raise RuntimeError(f"prefix hit of {start} tokens for {req_id} has no GDN snapshot")
        reg.touch(key)
        pinned.add(slot)
        restore.append((req_id, slot))
        reg.stats["hits"] += 1
        reg.stats["hit_tokens"] += start
        logger.info("Prefix cache hit: %s resumes at %d of %d prompt tokens (snapshot slot %d; %s)", req_id, start,
                    request.num_prompt_tokens, slot, reg.stats)
    for req_id, n in scheduler_output.num_scheduled_tokens.items():
        request = scheduler.requests[req_id]
        start = request.num_computed_tokens
        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        if start < prefill_end:
            _debug("%s chunk [%d, %d) of %d, hashes %d, block %d", req_id, start, start + n, prefill_end,
                   len(request.block_hashes), block_size)
        if not is_snapshot_position(start, prefill_end, block_size, interval):
            continue
        if start // block_size > len(request.block_hashes):
            continue
        slot = reg.acquire(request.block_hashes[start // block_size - 1], pinned)
        if slot is not None:
            pinned.add(slot)
            save.append((req_id, slot))
            logger.info("GDN snapshot: %s at %d of %d prompt tokens -> slot %d (%d kept)", req_id, start,
                        request.num_prompt_tokens, slot, len(reg))
    if not restore and not save:
        return None
    return {"restore": restore, "save": save}


def _aligned_split(self, request, num_new_tokens: int, num_new_local_computed_tokens: int = 0,
                   num_external_computed_tokens: int = 0) -> int:
    """Stop prefill chunks at snapshot positions; elsewhere chunks end wherever the token budget allows."""
    start = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
    prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
    if start >= prefill_end:
        return num_new_tokens
    stops = snapshot_positions(start, start + num_new_tokens, prefill_end, self.block_size,
                               gaudi_envs.VLLM_HPU_PREFIX_STATE_INTERVAL)
    _debug("split %s start %d tokens %d stops %s", request.request_id, start, num_new_tokens, stops)
    return stops[0] - start if stops else num_new_tokens


def _mamba_find_longest_cache_hit(cls, block_hashes, max_length, kv_cache_group_ids, block_pool, kv_cache_spec,
                                  drop_eagle_block, alignment_tokens, dcp_world_size=1, pcp_world_size=1):
    """Mamba groups hit exactly where a GDN snapshot exists. The blocks are placeholders: the state is restored
    into the request's compact slot by the runner, not read from the KV cache."""
    block_size = kv_cache_spec.block_size
    reg = _registry
    num_blocks = 0
    if reg is not None and block_pool.hash_block_size == block_size:
        num_blocks = reg.hit_blocks(block_hashes, max_length // block_size)
    return tuple([block_pool.null_block] * num_blocks for _ in kv_cache_group_ids), num_blocks * block_size


def _init_scheduler(self) -> None:
    global _registry
    self._hpu_prefix_state_checked = True
    if not getattr(self, "has_mamba_layers", False) or not self.cache_config.enable_prefix_caching:
        logger.warning("GDN prefix state cache inactive: has_mamba_layers=%s enable_prefix_caching=%s",
                       getattr(self, "has_mamba_layers", None), self.cache_config.enable_prefix_caching)
        return
    if self.cache_config.mamba_cache_mode != "align":
        raise ValueError("VLLM_HPU_PREFIX_STATE_CACHE needs mamba_cache_mode='align' (got "
                         f"{self.cache_config.mamba_cache_mode!r})")
    block_size = self.block_size
    if self.hash_block_size != block_size:
        raise ValueError(f"VLLM_HPU_PREFIX_STATE_CACHE needs one block size; scheduler {block_size}, hash "
                         f"{self.hash_block_size}")
    interval = gaudi_envs.VLLM_HPU_PREFIX_STATE_INTERVAL
    if interval <= 0 or interval % block_size:
        raise ValueError(f"VLLM_HPU_PREFIX_STATE_INTERVAL ({interval}) must be a positive multiple of the block size "
                         f"({block_size})")
    _registry = SnapshotRegistry(gaudi_envs.VLLM_HPU_PREFIX_STATE_SLOTS)
    # DFlash2 needs no last-block drop (see module docstring); the coordinator's fallback would flag every group,
    # Mamba included, and disable reuse.
    coordinator = self.kv_cache_manager.coordinator
    coordinator.eagle_group_ids = set()
    for manager in coordinator.single_type_managers:
        manager.use_eagle = False
    if hasattr(coordinator, "attention_groups"):
        coordinator.attention_groups = [
            group._replace(use_eagle=False) if hasattr(group, "_replace") else group
            for group in coordinator.attention_groups
        ]
    logger.info("GDN prefix state cache: %d snapshot slots, block %d tokens, snapshot interval %d tokens",
                _registry.num_slots, block_size, interval)


def install() -> None:
    """Patch the vLLM scheduler and Mamba manager (idempotent)."""
    if not enabled():
        return
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager
    if getattr(Scheduler, "_hpu_prefix_state_installed", False):
        return
    original_init = Scheduler.__init__
    original_update = Scheduler._update_after_schedule

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _init_scheduler(self)

    def _update_after_schedule(self, scheduler_output):
        if not getattr(self, "_hpu_prefix_state_checked", False):
            # the instance may predate the patch (scheduler built before the model runner)
            _init_scheduler(self)
        plan = plan_step(self, scheduler_output)
        if plan is not None:
            setattr(scheduler_output, STATE_ATTR, plan)
        original_update(self, scheduler_output)

    Scheduler.__init__ = __init__
    Scheduler._update_after_schedule = _update_after_schedule
    Scheduler._mamba_block_aligned_split = _aligned_split
    MambaManager.find_longest_cache_hit = classmethod(_mamba_find_longest_cache_hit)
    Scheduler._hpu_prefix_state_installed = True
    logger.info("GDN prefix state cache: scheduler and Mamba manager patched")


# --------------------------------------------------------------------------- runner side


class GdnSnapshotStore:
    """Device copies of compact GDN states: slot s keeps one row per GDN group of every state tensor.

    Row ``s * groups + g`` of a snapshot tensor mirrors row ``(g * max_reqs + base) * slots_per_req + 1`` of the
    compact tensor, the base (non-checkpoint) state of group offset g for the request in compact slot ``base``."""

    def __init__(self, num_slots: int):
        self.num_slots = num_slots
        self._snapshots: list[torch.Tensor] | None = None
        self._sources: list[torch.Tensor] = []

    def _ensure(self, states: list[torch.Tensor], groups: int) -> None:
        if self._snapshots is not None and len(self._sources) == len(states) and all(
                a is b for a, b in zip(self._sources, states)):
            return
        self._sources = list(states)
        self._snapshots = [
            torch.zeros((self.num_slots * groups, *state.shape[1:]), dtype=state.dtype, device=state.device)
            for state in states
        ]
        total = sum(t.numel() * t.element_size() for t in self._snapshots)
        logger.info("GDN prefix state cache: %d snapshot slots, %.2f GiB on device", self.num_slots, total / 2**30)

    @staticmethod
    def compact_rows(base: int, groups: int, max_reqs: int, slots_per_req: int) -> list[int]:
        return [(g * max_reqs + base) * slots_per_req + 1 for g in range(groups)]

    def apply(self, runner, scheduler_output) -> None:
        plan = getattr(scheduler_output, STATE_ATTR, None)
        if not plan:
            return
        states = runner._compact_gdn_state_tensors
        groups = runner._num_gdn_groups
        if not states or groups <= 0:
            raise RuntimeError("VLLM_HPU_PREFIX_STATE_CACHE needs compact GDN state (VLLM_COMPACT_GDN=1)")
        self._ensure(states, groups)
        device = states[0].device
        for module_wait in _dma_waits(runner):
            module_wait()

        def rows(req_id):
            base = runner._gdn_req_to_base_slot[req_id]
            return torch.tensor(self.compact_rows(base, groups, runner._gdn_max_reqs, runner._gdn_state_slots_per_req),
                                dtype=torch.long, device=device)

        def slot_rows(slot):
            return torch.arange(slot * groups, (slot + 1) * groups, dtype=torch.long, device=device)

        for req_id, slot in plan["restore"]:
            dst, src = rows(req_id), slot_rows(slot)
            for state, snap in zip(states, self._snapshots):
                state.index_copy_(0, dst, snap.index_select(0, src))
        for req_id, slot in plan["save"]:
            src, dst = rows(req_id), slot_rows(slot)
            for state, snap in zip(states, self._snapshots):
                snap.index_copy_(0, dst, state.index_select(0, src))
        if device.type == "hpu":
            import habana_frameworks.torch as htorch
            htorch.hpu.synchronize()


def _dma_waits(runner):
    if not gaudi_envs.VLLM_HPU_GDN_ASYNC_STATE_DMA:
        return []
    from vllm_gaudi.v1.worker.hpu_model_runner import HpuModelAdapter
    return [
        module.gdn_dma_pipeline().wait_all for module in runner.model.modules()
        if isinstance(module, HpuModelAdapter) and module.gdn_dma_pipeline() is not None
    ]
