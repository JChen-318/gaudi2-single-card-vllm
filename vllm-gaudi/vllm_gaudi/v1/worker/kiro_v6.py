# SPDX-License-Identifier: Apache-2.0
"""v6 host-path helpers: private compile code objects, a low-overhead step
timer and the DFlash2 next-round lookahead. Every feature has its own
default-off switch in ``vllm_gaudi.envs``."""

from __future__ import annotations

import contextlib
import itertools
import time
import types
from types import SimpleNamespace

import torch
try:
    from vllm_gaudi.extension.logger import logger as _plugin_logger
    logger = _plugin_logger()
except ImportError:  # CPU tests without the plugin
    from vllm.logger import init_logger
    logger = init_logger(__name__)

_code_ids = itertools.count()


def _copy_function(function: types.FunctionType, tag: str) -> types.FunctionType:
    code = function.__code__
    name = f"{code.co_name}__{tag}"
    try:
        new_code = code.replace(co_name=name, co_qualname=f"{code.co_qualname}__{tag}")
    except TypeError:  # Python < 3.11 has no co_qualname
        new_code = code.replace(co_name=name)
    copied = types.FunctionType(new_code, function.__globals__, name, function.__defaults__, function.__closure__)
    copied.__kwdefaults__ = function.__kwdefaults__
    copied.__dict__.update(function.__dict__)
    copied.__module__ = function.__module__
    copied.__doc__ = function.__doc__
    return copied


def with_private_code(obj):
    """Give ``obj`` a forward (or function body) whose code object is its own.

    Dynamo keeps one cache per code object and checks the entries one by one.
    The 19 target layer groups (and the 5 draft layers) share one ``forward``,
    so every lookup first fails on the sibling regions' entries; measured on
    the server trace this is 0.3-0.9 ms per target group. A copied code object
    leaves the traced bytecode, guards and graphs unchanged.
    """
    tag = f"c{next(_code_ids)}"
    if isinstance(obj, torch.nn.Module):
        bound = obj.__dict__.get("forward")
        if bound is not None:
            if isinstance(bound, types.MethodType) and isinstance(bound.__func__, types.FunctionType):
                obj.__dict__["forward"] = types.MethodType(_copy_function(bound.__func__, tag), bound.__self__)
            return obj
        cls = type(obj)
        forward = getattr(cls, "forward", None)
        if not isinstance(forward, types.FunctionType):
            return obj
        namespace = {"forward": _copy_function(forward, tag), "__module__": cls.__module__,
                     "__qualname__": cls.__qualname__}
        try:
            obj.__class__ = type(cls)(cls.__name__, (cls, ), namespace)
        except TypeError:
            return obj
        return obj
    if isinstance(obj, types.MethodType) and isinstance(obj.__func__, types.FunctionType):
        return types.MethodType(_copy_function(obj.__func__, tag), obj.__self__)
    if isinstance(obj, types.FunctionType):
        return _copy_function(obj, tag)
    return obj


class RuntimeControl:
    """Same-process A/B switches for the runtime-only v6 features."""

    def __init__(self, path: str):
        self.path = path
        self.active = {"lookahead": True, "fast_topk": True, "presample": True, "delta": True, "predraft": True,
                       "guard_skip": True, "rng_pool": True, "early_readback": True}

    def refresh(self) -> dict:
        if self.path:
            import json
            try:
                with open(self.path, encoding="utf-8") as handle:
                    loaded = json.load(handle)
                self.active = {key: bool(loaded.get(key, True)) for key in self.active}
            except (OSError, ValueError) as exc:
                logger.warning("v6 control file %s unreadable (%s); keeping %s", self.path, exc, self.active)
        return self.active


class WindowProfiler:
    """cProfile calls number ``start`` to ``end`` of one call site, then dump the stats once."""

    def __init__(self, path: str, start: int = 300, end: int = 500):
        self.path, self.start, self.end = path, start, end
        self.calls = 0
        self.profile = None

    def run(self, function, *args, **kwargs):
        index = self.calls
        self.calls += 1
        if not self.path or not self.start <= index < self.end:
            return function(*args, **kwargs)
        import cProfile
        if self.profile is None:
            self.profile = cProfile.Profile()
        self.profile.enable()
        try:
            return function(*args, **kwargs)
        finally:
            self.profile.disable()
            if index == self.end - 1:
                self.profile.dump_stats(self.path)
                logger.info("Window profile written to %s", self.path)


class StepTimer:
    """Append one line per runner call: ``perf_counter kind decodes prefills tokens``.

    ``E`` marks entry to execute_model, ``S`` the return of sample_tokens.
    Consecutive ``E`` stamps of a steady single request give the round time
    without any profiler in the process.
    """

    def __init__(self, path: str):
        self._file = open(path, "a", buffering=1 << 16)
        self._lines = 0

    def mark(self, kind: str, decodes: int = -1, prefills: int = -1, tokens: int = -1) -> None:
        self._file.write(f"{time.perf_counter():.6f} {kind} {decodes} {prefills} {tokens}\n")
        self._lines += 1
        if self._lines % 64 == 0:
            self._file.flush()


class DFlashLookahead:
    """Queue the next round's target forward (and its sampling) before returning to the engine.

    At the end of round N the host knows how many tokens each request
    accepted, and the new drafts are queued on the device. For steady decode
    requests the next scheduler step is predictable: the same requests, 1 + K
    tokens each, drafts equal to ours. We build that round's inputs from the
    predicted state and submit the target forward right behind the drafter,
    followed by the rejection sampling of that round, so the device stays busy
    while the engine schedules and the runner prepares round N+1.

    The forward writes target KV and advances GDN checkpoints, so it must
    never be followed by a second forward of the same round. Launch therefore
    requires 1-2 decode requests with full blocks, every KV page allocated, no
    defragmenter, no structured output and no pending lookahead. Round N+1
    reuses the outputs only after checking requests, positions, drafts and
    pages. A finished, aborted or preempted request drops its rows; when the
    other request of a pair continues alone, its rows are selected from the
    pair's outputs instead of running the forward again. Any other mismatch is
    logged and turns the feature off.

    Sampling moves earlier in host time but not in device order, and every
    random draw still comes from the same generator, so the output
    distribution is unchanged (unseeded draws of a prefill sampled in the same
    step may interleave differently).
    """

    def __init__(self, runner, presample: bool = True, profile_path: str = "", delta_verify: int | None = None,
                 predraft_verify: int | None = None, early_readback_verify: int | None = None):
        self.runner = runner
        self.presample = presample
        self.delta = DeltaPrepare(runner, delta_verify) if delta_verify is not None else None
        # v11 pre-draft: queue the drafter of the predicted round right behind its sampling.
        self.predraft_verifier = None
        if predraft_verify is not None:
            from vllm_gaudi.v1.worker.kiro_v11 import PreDraftVerifier
            self.predraft_verifier = PreDraftVerifier(predraft_verify)
        self.predraft_disabled = predraft_verify is None
        self.ready_predraft = None
        # v11 early readback: copy the presampled tokens to the host before the pre-draft is queued.
        self.early_readback_left = early_readback_verify
        self.ready_sampled_cpu = None
        self._sampled_buffers: dict = {}
        self._sampled_flip = 0
        self.pending: dict | None = None
        self.disabled = False
        self.stats = {"launched": 0, "reused": 0, "subset": 0, "dropped": 0, "mismatch": 0, "predrafted": 0,
                      "predraft_used": 0, "early_readback": 0}
        self._copy_stream = None
        self._drafts_cpu = None
        self._drafts_event = None
        self._profile_path = profile_path
        self._profiler = None
        self._staged = None

    def _mark(self, kind: str) -> None:
        timer = self.runner._step_timer
        if timer is not None:
            timer.mark(kind)

    # ---- draft export -------------------------------------------------
    def take_drafts(self) -> list[list[int]] | None:
        """Host drafts from the side-stream copy, without waiting for the lookahead forward."""
        if self._drafts_cpu is None:
            return None
        self._drafts_event.synchronize()
        drafts = self._drafts_cpu.tolist()
        self._drafts_cpu = self._drafts_event = None
        pending = self.pending
        if pending is not None and pending["drafts_host"] is None:
            pending["drafts_host"] = {req_id: list(drafts[row]) for req_id, row in pending["rows"].items()}
        return drafts

    def _export_drafts(self, drafts: torch.Tensor) -> None:
        if self._copy_stream is None:
            self._copy_stream = torch.hpu.Stream()
        event = torch.hpu.Event()
        default_stream = torch.hpu.current_stream()
        with torch.hpu.stream(self._copy_stream):
            self._copy_stream.wait_stream(default_stream)
            self._drafts_cpu = drafts.to("cpu", non_blocking=True)
            event.record()
        self._drafts_event = event

    def _export_sampled(self, sampled: torch.Tensor):
        """D2H of the presampled tokens queued right behind the sampling and before the
        pre-draft, into pinned host memory, with an event; the next step's readback waits
        for that event instead of for everything queued so far. Two buffers alternate."""
        key = (tuple(sampled.shape), sampled.dtype)
        buffers = self._sampled_buffers.get(key)
        if buffers is None:
            buffers = [torch.empty(sampled.shape, dtype=sampled.dtype, pin_memory=True) for _ in range(2)]
            self._sampled_buffers[key] = buffers
        self._sampled_flip ^= 1
        host = buffers[self._sampled_flip]
        host.copy_(sampled, non_blocking=True)
        event = torch.hpu.Event()
        event.record()
        return host, event

    def pop_sampled_cpu(self, device_sampled):
        """Host copy of this step's reused sampled tokens, or None to read the device tensor."""
        ready, self.ready_sampled_cpu = self.ready_sampled_cpu, None
        if ready is None or self.early_readback_left is None:
            return None
        host, event = ready
        event.synchronize()
        self._mark("W")
        if self.early_readback_left > 0:
            self.early_readback_left -= 1
            if not torch.equal(host, device_sampled.to("cpu")):
                logger.error("DFlash2 early readback differs from the device tokens; early readback disabled")
                self.early_readback_left = None
                return None
            if self.early_readback_left == 0:
                logger.info("DFlash2 early readback verified")
        self.stats["early_readback"] += 1
        return host

    def _prepare(self, predicted, count: int):
        runner = self.runner
        launched = self.stats["launched"]
        if self._profile_path and 50 <= launched < 250:
            import cProfile
            if self._profiler is None:
                self._profiler = cProfile.Profile()
            self._profiler.enable()
            try:
                return runner._prepare_inputs(predicted, 0, count)
            finally:
                self._profiler.disable()
                if launched == 249:
                    self._profiler.dump_stats(self._profile_path)
                    logger.info("DFlash2 lookahead prepare profile written to %s", self._profile_path)
        return runner._prepare_inputs(predicted, 0, count)

    def _full_prepare(self, rows, computed, sampled, num_spec, width, num_decodes):
        """The runner's own prepare on the predicted batch state (restored afterwards)."""
        batch = self.runner.input_batch
        saved = []
        try:
            for req_id, row in rows.items():
                first_spec = computed[req_id] + 1
                saved.append((row, int(batch.num_computed_tokens_cpu[row]),
                              batch.token_ids_cpu[row, first_spec:first_spec + num_spec].copy(), int(batch.num_tokens[row])))
                batch.num_computed_tokens_cpu[row] = computed[req_id]
                batch.token_ids_cpu[row, first_spec:first_spec + num_spec] = 0
                batch.num_tokens[row] = first_spec + num_spec
            predicted = SimpleNamespace(num_scheduled_tokens={req_id: width for req_id in rows},
                                        scheduled_spec_decode_tokens={req_id: [0] * num_spec for req_id in rows},
                                        total_num_scheduled_tokens=width * num_decodes)
            _, (decode_data, _) = self._prepare(predicted, num_decodes)
            return decode_data
        finally:
            for row, num_computed, tokens, num_tokens in saved:
                first_spec = num_computed + len(sampled[row]) + 1
                batch.num_computed_tokens_cpu[row] = num_computed
                batch.token_ids_cpu[row, first_spec:first_spec + num_spec] = tokens
                batch.num_tokens[row] = num_tokens

    # ---- launch -------------------------------------------------------
    def stage(self, num_decodes: int, num_prefills: int, sampled: list[list[int]], warmup_mode: bool,
              structured_output: bool):
        """Prepare the predicted next round. Needs the accepted counts, not the new drafts, so it can run
        before the drafter is queued; the device then runs the drafter while the host queues the forward."""
        self._staged = None
        runner = self.runner
        batch = runner.input_batch
        if (self.disabled or self.pending is not None or warmup_mode or structured_output or num_prefills
                or num_decodes not in (1, 2) or batch.num_reqs != num_decodes):
            return None
        num_spec = runner.speculative_config.num_speculative_tokens
        width = num_spec + 1
        defragmenter = getattr(runner, "defragmenter", None)
        if defragmenter is not None and getattr(defragmenter, "enabled", False):
            return
        group_id = runner._get_attention_group_id_for_hybrid()
        block_size = runner.attn_block_size
        table = batch.block_table[group_id]
        blocks_per_row = getattr(table, "num_blocks_per_row", None)
        if blocks_per_row is None:
            return
        table_cpu = table.get_cpu_tensor()
        rows = {}
        computed = {}
        last = {}
        pages = {}
        for row in range(num_decodes):
            req_id = batch.req_ids[row]
            if not sampled[row]:
                return
            new_computed = int(batch.num_computed_tokens_cpu[row]) + len(sampled[row])
            if int(batch.num_tokens_no_spec[row]) != new_computed + 1 or new_computed + width + 1 > runner.max_model_len:
                return
            request = runner.requests.get(req_id)
            max_tokens = getattr(getattr(request, "sampling_params", None), "max_tokens", None)
            if request is None or (max_tokens is not None and len(request.output_token_ids) + width >= max_tokens):
                # The scheduler stops or trims near max_tokens; do not predict it.
                return
            needed = (new_computed + width + block_size - 1) // block_size
            if int(blocks_per_row[row]) < needed:
                return
            rows[req_id] = row
            computed[req_id] = new_computed
            last[req_id] = int(sampled[row][-1])
            pages[req_id] = table_cpu[row, :needed].tolist()

        req_list = list(rows)
        starts = [computed[req_id] for req_id in req_list]
        delta = self.delta if self.delta is not None and runner._v6_control.active.get("delta", True) else None
        decode_data = delta.build(req_list, starts, [last[req_id] for req_id in req_list], width) if delta else None
        if decode_data is None or delta.verify_left > 0:
            full = self._full_prepare(rows, computed, sampled, num_spec, width, num_decodes)
            if decode_data is None:
                decode_data = full
                if delta is not None:
                    delta.stats["full"] += 1
            elif not delta.check(decode_data, full):
                decode_data = full
        if delta is not None:
            delta.remember(req_list, starts, width, decode_data)
        self._mark("P")
        self._staged = {"rows": rows, "computed": computed, "last": last, "pages": pages, "group_id": group_id,
                        "decode_data": decode_data, "width": width, "num_decodes": num_decodes,
                        "req_ids": list(batch.req_ids[:num_decodes])}
        return self._staged

    def maybe_launch(self, num_decodes: int, num_prefills: int, sampled: list[list[int]], warmup_mode: bool,
                     structured_output: bool) -> None:
        runner = self.runner
        staged = getattr(self, "_staged", None)
        self._staged = None
        if staged is None or staged["req_ids"] != list(runner.input_batch.req_ids[:num_decodes]):
            staged = self.stage(num_decodes, num_prefills, sampled, warmup_mode, structured_output)
            self._staged = None
        if staged is None or self.pending is not None:
            return
        drafts = runner._draft_token_ids
        width = staged["width"]
        if not isinstance(drafts, torch.Tensor) or tuple(drafts.shape) != (num_decodes, width - 1):
            return
        rows, computed, last, pages = staged["rows"], staged["computed"], staged["last"], staged["pages"]
        group_id, decode_data = staged["group_id"], staged["decode_data"]
        self._export_drafts(drafts)
        if (decode_data is None or decode_data.token_ids is None
                or decode_data.token_ids.shape[0] % width or decode_data.token_ids.shape[0] // width < num_decodes):
            return
        padded = decode_data.token_ids.shape[0] // width
        # Each row's first position holds its last sampled token; the drafts follow.
        decode_data.token_ids.view(padded, width)[:num_decodes, 1:].copy_(drafts.to(decode_data.token_ids.dtype))
        control = runner._v6_control.active
        presample = (self.presample and control.get("presample", True) and decode_data.spec_decode_metadata is not None)
        starts = [computed[req_id] for req_id in rows]
        plan = (runner._dflash2_predraft_plan(starts, padded)
                if presample and not self.predraft_disabled and control.get("predraft", True) else None)
        guard_skip = getattr(runner, "_guard_skip", None) if control.get("guard_skip", True) else None
        key = None
        if guard_skip is not None:
            from vllm_gaudi.v1.worker.kiro_v11 import shape_key
            batch = runner.input_batch
            from vllm_gaudi.v1.sample import fast_topk_topp
            key = (shape_key(decode_data), tuple(sorted(control.items())), None if plan is None else plan[1],
                   batch.all_greedy, batch.all_random, batch.no_top_k, batch.no_top_p, fast_topk_topp.is_active())
        predraft = None
        with guard_skip.round(key) if guard_skip is not None else contextlib.nullcontext():
            outputs = runner._execute_model_generic(decode_data.token_ids,
                                                    decode_data.position_ids,
                                                    decode_data.attn_metadata,
                                                    decode_data.logits_indices,
                                                    runner.kv_caches,
                                                    None,
                                                    None,
                                                    warmup_mode=False,
                                                    fuse_greedy_sampling=False)
            self._mark("F")
            presampled = None
            sampled_cpu = None
            if presample:
                presampled = runner._dflash_lookahead_sample(decode_data, outputs[3], list(rows))
                if self.early_readback_left is not None and control.get("early_readback", True):
                    try:
                        sampled_cpu = self._export_sampled(presampled)
                    except Exception as exc:  # noqa: BLE001 - e.g. no pinned host allocator
                        self.early_readback_left = None
                        logger.error("DFlash2 early readback unavailable (%s: %s); disabled", type(exc).__name__, exc)
                self._mark("Q")
            if plan is not None and presampled is not None:
                try:
                    predraft = runner._dflash2_predraft(decode_data, outputs, presampled, starts, plan)
                except Exception as exc:  # noqa: BLE001 - drafts only steer acceptance; keep serving
                    # Nothing but draft KV may have been written; the host drafter rewrites it.
                    self.predraft_disabled = True
                    logger.error("DFlash2 pre-draft failed (%s: %s); pre-draft is now disabled",
                                 type(exc).__name__, exc)
                    predraft = None
                if predraft is not None:
                    self.stats["predrafted"] += 1
                    self._mark("G")
        self.pending = {
            "req_ids": list(rows),
            "rows": rows,
            "computed": computed,
            "last": last,
            "pages": pages,
            "group_id": group_id,
            "padded": padded,
            "width": width,
            "drafts_host": None,
            "outputs": outputs,
            "sampled": presampled,
            "decode_data": decode_data,
            "predraft": predraft,
            "sampled_cpu": sampled_cpu,
        }
        self.stats["launched"] += 1

    # ---- reuse --------------------------------------------------------
    def _checks(self, pending, scheduler_output, decode_ids, alive) -> dict:
        runner = self.runner
        batch = runner.input_batch
        width = pending["width"]
        table = batch.block_table[pending["group_id"]].get_cpu_tensor()
        drafts_host = pending["drafts_host"] or {}
        checks = {"requests": sorted(decode_ids) == sorted(alive)}
        for index, req_id in enumerate(decode_ids):
            if req_id not in pending["rows"]:
                checks["requests"] = False
                break
            computed = pending["computed"][req_id]
            checks[f"tokens:{index}"] = scheduler_output.num_scheduled_tokens.get(req_id) == width
            checks[f"drafts:{index}"] = (list(scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))
                                         == drafts_host.get(req_id))
            checks[f"computed:{index}"] = int(batch.num_computed_tokens_cpu[index]) == computed
            checks[f"last:{index}"] = int(batch.token_ids_cpu[index, computed]) == pending["last"][req_id]
            page_list = pending["pages"][req_id]
            checks[f"pages:{index}"] = table[index, :len(page_list)].tolist() == page_list
        return checks

    def peek_full_reuse(self, scheduler_output, num_decodes: int, prompt_req_ids):
        """The pending decode inputs, if this step will reuse every row unchanged and in order."""
        pending = self.pending
        if pending is None or pending["sampled"] is None:
            return None
        batch = self.runner.input_batch
        decode_ids = list(batch.req_ids[:num_decodes])
        if decode_ids != pending["req_ids"] or any(req_id in prompt_req_ids for req_id in decode_ids):
            return None
        if not all(self._checks(pending, scheduler_output, decode_ids, decode_ids).values()):
            return None
        return pending["decode_data"]

    def take(self, scheduler_output, decode_data, num_decodes: int, prompt_req_ids):
        """Return (forward outputs, presampled token ids or None) for this step's decodes, or None."""
        pending = self.pending
        self.ready_predraft = None
        self.ready_sampled_cpu = None
        if pending is None:
            return None
        runner = self.runner
        batch = runner.input_batch
        alive = [req_id for req_id in pending["req_ids"] if req_id in runner.requests and req_id not in prompt_req_ids]
        if not alive:
            # Finished, aborted, or preempted and recomputed from its prompt.
            self.pending = None
            self.stats["dropped"] += 1
            return None
        decode_ids = list(batch.req_ids[:num_decodes])
        if not any(req_id in decode_ids for req_id in alive):
            # Not scheduled as a decode this step; keep the result for later.
            return None
        self.pending = None
        width = pending["width"]
        checks = self._checks(pending, scheduler_output, decode_ids, alive)
        padded_now = decode_data.token_ids.shape[0] // width
        if not all(checks.values()):
            self.stats["mismatch"] += 1
            self.disabled = True
            logger.error(
                "DFlash2 lookahead mismatch for %s (%s); recurrent state was already advanced. "
                "Lookahead is now disabled. stats=%s", decode_ids, {k: v for k, v in checks.items() if not v},
                self.stats)
            return None
        outputs = pending["outputs"]
        sampled = pending["sampled"]
        predraft = pending.get("predraft")
        subset_before = self.stats["subset"]
        rows = [pending["rows"][req_id] for req_id in decode_ids]
        if rows != list(range(pending["padded"]))[:len(rows)] or padded_now != pending["padded"]:
            # The other request of the pair left: keep only the surviving rows.
            real_rows = list(rows)
            rows = rows + [rows[0]] * (padded_now - len(rows))
            device = outputs[3].device
            if predraft is not None:
                predraft = predraft.index_select(0, torch.tensor(real_rows, dtype=torch.long, device=device))
            token_rows = torch.tensor([row * width + offset for row in rows for offset in range(width)],
                                      dtype=torch.long, device=device)
            hidden, aux, sample_hidden, logits, fused = outputs
            outputs = (hidden.index_select(0, token_rows),
                       [states.index_select(0, token_rows) for states in aux] if aux is not None else None,
                       sample_hidden.index_select(0, token_rows), logits.index_select(0, token_rows), fused)
            if sampled is not None:
                sampled = sampled.index_select(0, torch.tensor(rows, dtype=torch.long, device=device))
            self.stats["subset"] += 1
        self.stats["reused"] += 1
        if self.stats["subset"] == subset_before:
            self.ready_sampled_cpu = pending.get("sampled_cpu")
        # The pre-draft belongs to exactly these rows; only a pure decode step consumes it.
        self.ready_predraft = predraft if not prompt_req_ids and len(decode_ids) == num_decodes else None
        if self.stats["reused"] in (1, 100) or self.stats["reused"] % 1000 == 0:
            logger.info("DFlash2 lookahead stats: %s", self.stats)
        return outputs, sampled

    def pop_predraft(self):
        """The pre-drafted next drafts for this step's decodes, if the reused round brought them."""
        predraft, self.ready_predraft = self.ready_predraft, None
        if predraft is None or self.predraft_disabled:
            return None
        self.stats["predraft_used"] += 1
        return predraft

    def needs_predraft_check(self) -> bool:
        return self.predraft_verifier is not None and self.predraft_verifier.left > 0

    def check_predraft(self, predraft, host) -> None:
        if not self.predraft_verifier.compare(predraft, host):
            self.predraft_disabled = True


class DeltaPrepare:
    """Build the next steady decode round's inputs by updating the previous ones.

    Used by the lookahead for the predicted round when the same requests stay
    in the same order, with full verification blocks, shared-prefix decode and
    an unchanged KV page list in every attention group. Then only positions,
    KV slots, page usage, accepted counts and token ids change; seq lens, GDN
    state indices, query start locations, logits and speculative indices and
    the page lists keep their device tensors. Anything else (a new page, a new
    request, another bucket) returns None and the caller runs the full
    prepare, whose result seeds the next delta.

    The first ``verify`` delta builds are compared field by field against the
    full prepare; one difference disables the delta path for the process.
    """

    def __init__(self, runner, verify: int):
        self.runner = runner
        self.verify_left = verify
        self.disabled = False
        self.prev = None
        self.stats = {"delta": 0, "full": 0, "verified": 0}

    # -- helpers -----------------------------------------------------------
    def _groups(self):
        runner = self.runner
        first = runner._get_attention_group_id_for_hybrid()
        others = [gid for gid in getattr(runner, "_legacy_attention_group_ids", ()) if gid != first]
        return first, others

    def _pages(self, req_ids, starts, width):
        runner = self.runner
        batch = runner.input_batch
        block_size = runner.attn_block_size
        first, others = self._groups()
        pages = {}
        for gid in [first] + others:
            table = batch.block_table[gid].get_cpu_tensor().numpy()
            pages[gid] = [table[batch.req_id_to_index[req_id], :(start + width + block_size - 1) // block_size].copy()
                          for req_id, start in zip(req_ids, starts)]
        return pages

    @staticmethod
    def _same_pages(a, b) -> bool:
        import numpy as np
        return a.keys() == b.keys() and all(
            len(a[gid]) == len(b[gid]) and all(np.array_equal(x, y) for x, y in zip(a[gid], b[gid])) for gid in a)

    def eligible(self) -> bool:
        runner = self.runner
        defragmenter = getattr(runner, "defragmenter", None)
        return (not self.disabled and runner._shared_prefix_decode_enabled()
                and (defragmenter is None or not getattr(defragmenter, "enabled", False))
                and not runner.use_async_scheduling)

    def remember(self, req_ids, starts, width, decode_data) -> None:
        """Seed the next delta with a full prepare's result."""
        self.prev = None
        if not self.eligible() or decode_data is None:
            return
        metadata = decode_data.attn_metadata
        groups = getattr(metadata, "attention_groups", None)
        first, others = self._groups()
        if groups is None or metadata.block_usage is None or metadata.block_list is None:
            return
        for gid in others:
            group = groups[gid]
            # Groups must share the first group's page usage and owners (shared masks).
            if group is None or group.block_usage is not metadata.block_usage or group.block_groups is not metadata.block_groups:
                return
        self.prev = {"req_ids": list(req_ids), "width": width, "decode_data": decode_data,
                     "pages": self._pages(req_ids, starts, width)}

    # -- delta build -------------------------------------------------------
    def build(self, req_ids, starts, last_tokens, width):
        prev = self.prev
        if prev is None or not self.eligible() or prev["req_ids"] != list(req_ids) or prev["width"] != width:
            return None
        pages = self._pages(req_ids, starts, width)
        if not self._same_pages(pages, prev["pages"]):
            return None
        import numpy as np
        from dataclasses import replace

        from vllm_gaudi.utils import async_h2d_copy

        runner = self.runner
        device = runner.device
        block_size = runner.attn_block_size
        decode_data = prev["decode_data"]
        metadata = decode_data.attn_metadata
        count = len(req_ids)
        padded = decode_data.token_ids.shape[0] // width
        if padded != count:
            return None
        first, others = self._groups()
        starts_np = np.asarray(starts, dtype=np.int64)
        positions = (starts_np[:, None] + np.arange(width, dtype=np.int64)[None, :]).reshape(-1)
        logical = positions // block_size
        offsets = positions % block_size

        def slots_for(gid):
            physical = np.concatenate([pages[gid][row].astype(np.int64)[logical[row * width:(row + 1) * width]]
                                       for row in range(count)])
            return torch.from_numpy((physical * block_size + offsets).astype(np.int32)).reshape(-1, 1)

        page_counts = np.asarray([len(page_list) for page_list in pages[first]], dtype=np.int64)
        total = int(page_counts.sum())
        owner = np.repeat(np.arange(count, dtype=np.int64), page_counts)
        logical_pages = np.arange(total, dtype=np.int64) - (np.cumsum(page_counts) - page_counts)[owner]
        target = metadata.block_usage.shape[0]
        usage = np.ones((target, width), dtype=np.int64)
        visible = (starts_np[owner][:, None] + np.arange(width, dtype=np.int64)[None, :] + 1
                   - logical_pages[:, None] * block_size)
        usage[:total] = np.clip(visible, 0, block_size)
        accepted = [runner._gdn_num_accepted_by_req.get(req_id, 1) for req_id in req_ids]
        tokens = np.zeros(count * width, dtype=np.int64)
        tokens[np.arange(count) * width] = last_tokens

        def h2d(array, like):
            return async_h2d_copy(torch.from_numpy(np.ascontiguousarray(array)).to(like.dtype).reshape(like.shape),
                                  device=device)

        usage_dev = h2d(usage, metadata.block_usage)
        accepted_dev = h2d(np.asarray(accepted, dtype=np.int64), metadata.num_accepted_tokens)
        # The K/V mirror tail moves with the context (v1/worker/decode_kv_mirror.py); same shapes, same decision.
        plan_mirror = getattr(runner, "_decode_kv_mirror_tail", None)
        mirror_tail = plan_mirror(req_ids, starts, padded, int(metadata.block_list.shape[0])) if plan_mirror else None
        if (mirror_tail is None) != (getattr(metadata, "kv_mirror_tail", None) is None):
            return None
        mirror = {} if mirror_tail is None else {
            "kv_mirror_tail": async_h2d_copy(torch.from_numpy(mirror_tail), device=device)}
        main = replace(metadata, slot_mapping=async_h2d_copy(slots_for(first).to(metadata.slot_mapping.dtype),
                                                             device=device),
                       block_usage=usage_dev, num_accepted_tokens=accepted_dev, attention_groups=None, **mirror)
        groups = list(metadata.attention_groups)
        groups[first] = replace(main, attention_groups=None)
        for gid in others:
            old = metadata.attention_groups[gid]
            groups[gid] = replace(main, slot_mapping=async_h2d_copy(slots_for(gid).to(old.slot_mapping.dtype),
                                                                    device=device),
                                  block_list=old.block_list, attention_groups=None)
        main.attention_groups = tuple(groups)
        position_like = decode_data.position_ids
        if position_like.ndim == 2 and position_like.shape[0] == 3 and position_like.shape[1] == count * width:
            # M-RoPE: every axis carries the position shifted by the request's delta (runner layout).
            shifts = np.repeat(np.asarray([int(getattr(runner.requests[req_id], "mrope_position_delta", 0) or 0)
                                           for req_id in req_ids], dtype=np.int64), width)
            positions = np.broadcast_to((positions + shifts)[None, :], (3, count * width))
        elif position_like.numel() != count * width:
            return None
        built = replace(decode_data, token_ids=h2d(tokens, decode_data.token_ids),
                        position_ids=h2d(positions, position_like), attn_metadata=main)
        self.stats["delta"] += 1
        return built

    # -- verification ------------------------------------------------------
    @staticmethod
    def _same(a, b, path, diffs):
        from dataclasses import fields, is_dataclass
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
                diffs.append(path)
            elif a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a.cpu(), b.cpu()):
                diffs.append(path)
            return
        if is_dataclass(a) and is_dataclass(b):
            for item in fields(a):
                DeltaPrepare._same(getattr(a, item.name), getattr(b, item.name), f"{path}.{item.name}", diffs)
            return
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            if len(a) != len(b):
                diffs.append(path)
                return
            for index, (x, y) in enumerate(zip(a, b)):
                DeltaPrepare._same(x, y, f"{path}[{index}]", diffs)
            return
        if a != b:
            diffs.append(path)

    def check(self, delta, full) -> bool:
        diffs: list[str] = []
        for name in ("token_ids", "position_ids", "logits_indices", "attn_metadata", "num_decodes"):
            self._same(getattr(delta, name), getattr(full, name), name, diffs)
        spec_delta, spec_full = delta.spec_decode_metadata, full.spec_decode_metadata
        for name in ("num_draft_tokens", "cu_num_draft_tokens", "cu_num_sampled_tokens", "target_logits_indices",
                     "bonus_logits_indices", "logits_indices", "draft_token_ids"):
            self._same(getattr(spec_delta, name), getattr(spec_full, name), f"spec.{name}", diffs)
        self.verify_left -= 1
        if diffs:
            self.disabled = True
            logger.error("Lookahead delta prepare differs from the full prepare at %s; delta disabled", diffs[:12])
            return False
        self.stats["verified"] += 1
        if self.verify_left == 0:
            logger.info("Lookahead delta prepare verified on %d rounds; stats=%s", self.stats["verified"], self.stats)
        return True
