# SPDX-License-Identifier: Apache-2.0
"""v11 host-path helpers (Kiro): steady-decode guard skipping, a pooled
sampling RNG and the DFlash2 pre-draft bridge. Every feature has its own
default-off switch in ``vllm_gaudi.envs``."""

from __future__ import annotations

import contextlib
import dataclasses
import functools

import torch
try:
    from vllm_gaudi.extension.logger import logger as _plugin_logger
    logger = _plugin_logger()
except ImportError:  # CPU tests without the plugin
    from vllm.logger import init_logger
    logger = init_logger(__name__)

_SKIP_MESSAGE = "skip_guard_eval_unsafe"


def _unique_graphs() -> int:
    from torch._dynamo.utils import counters
    return counters["stats"]["unique_graphs"]


class GuardSkip:
    """Run steady decode rounds under Dynamo's ``skip_guard_eval_unsafe`` stance.

    Under that stance Dynamo only evaluates the guards that failed before,
    i.e. the ones that tell a region's cache entries apart. An input that was
    never compiled can then silently reuse a graph built for other shapes, so
    the stance is used only for a round key (every input shape, metadata flag
    and switch, see ``shape_key``) that already completed ``warm`` guarded
    rounds without compiling a new graph. Every ``recheck`` skipped rounds the
    key runs guarded once more. A region that still needs a new graph under the
    stance raises before it runs; ``protect`` retries exactly that call with
    full guards, so no region is ever executed twice.
    """

    def __init__(self, warm: int = 3, recheck: int = 1024):
        self.warm = max(1, warm)
        self.recheck = max(2, recheck)
        self.clean: dict = {}
        self.skipped_since: dict = {}
        self.disabled = False
        self.retried = False
        self.stats = {"skipped": 0, "guarded": 0, "retried": 0, "keys": 0}

    @contextlib.contextmanager
    def round(self, key):
        """Context for one steady round; yields True when guards are skipped."""
        if self.disabled or key is None:
            yield False
            return
        if self.clean.get(key, 0) >= self.warm:
            count = self.skipped_since.get(key, 0) + 1
            if count < self.recheck:
                self.skipped_since[key] = count
                self.retried = False
                with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
                    yield True
                self.stats["skipped"] += 1
                if self.retried:
                    # A region recompiled under the stance: earn the key again.
                    self.clean[key] = 0
                if self.stats["skipped"] in (1, 1000) or self.stats["skipped"] % 20000 == 0:
                    logger.info("Guard skip stats: %s", self.stats)
                return
            # Periodic recheck: one guarded round, then the key qualifies again.
            self.skipped_since[key] = 0
            self.clean[key] = self.warm - 1
        before = _unique_graphs()
        yield False
        self.stats["guarded"] += 1
        if key not in self.clean:
            self.stats["keys"] += 1
        self.clean[key] = self.clean.get(key, 0) + 1 if _unique_graphs() == before else 0

    def on_retry(self) -> None:
        self.retried = True
        self.stats["retried"] += 1
        if self.stats["retried"] <= 3:
            logger.warning("A compiled region needed a new graph under the guard-skip stance; "
                           "retried it with full guards. stats=%s", self.stats)


def protect(compiled, skip: GuardSkip):
    """Retry a compiled region with full guards when the skip stance asks it to recompile."""

    def call(inner, args, kwargs):
        try:
            return inner(*args, **kwargs)
        except RuntimeError as exc:
            if _SKIP_MESSAGE not in str(exc):
                raise
            skip.on_retry()
            with torch.compiler.set_stance("default"):
                return inner(*args, **kwargs)

    if isinstance(compiled, torch.nn.Module):
        inner_forward = compiled.forward

        def forward(*args, **kwargs):
            return call(inner_forward, args, kwargs)

        object.__setattr__(compiled, "forward", forward)
        return compiled

    @functools.wraps(compiled)
    def wrapper(*args, **kwargs):
        return call(compiled, args, kwargs)

    return wrapper


def shape_key(value, depth: int = 0):
    """Hashable summary of everything that selects a compiled graph: tensor
    shapes and dtypes, scalar flags and the structure around them. Integer
    lists (per-request lengths) contribute only their length."""
    if isinstance(value, torch.Tensor):
        return (tuple(value.shape), value.dtype)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth > 6:
        return type(value).__name__
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (type(value).__name__, ) + tuple(
            shape_key(getattr(value, field.name), depth + 1) for field in dataclasses.fields(value))
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            return ("ints", len(value))
        return tuple(shape_key(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(name), shape_key(item, depth + 1)) for name, item in value.items()))
    return type(value).__name__


class RngPool:
    """Uniform and exponential draws for rejection sampling, generated ``rounds`` at a time.

    Every eager random op on HPU costs about 1-2 ms of host time, mostly fixed
    overhead. One draw per round becomes one large draw every ``rounds`` rounds;
    each round consumes its own slice (cloned into owned storage), so the draws
    stay independent and identically distributed. Requests with their own
    generator never use the pool.
    """

    def __init__(self, rounds: int):
        self.rounds = rounds
        self.active = True
        self._pools: dict = {}
        self.stats = {"refills": 0, "draws": 0}

    def take(self, kind: str, shape, dtype: torch.dtype, device) -> torch.Tensor:
        key = (kind, tuple(shape), dtype, str(device))
        entry = self._pools.get(key)
        if entry is None or entry[1] >= self.rounds:
            pool = torch.empty((self.rounds, *shape), dtype=dtype, device=device)
            if kind == "uniform":
                pool.uniform_()
            else:
                pool.exponential_()
            entry = [pool, 0]
            self._pools[key] = entry
            self.stats["refills"] += 1
        out = entry[0][entry[1]].clone()
        entry[1] += 1
        self.stats["draws"] += 1
        return out


RNG_POOL: RngPool | None = None


def pooled_draw(kind: str, shape, dtype: torch.dtype, device, generators) -> torch.Tensor | None:
    """A pooled draw, or None when the pool is off or a request brings its own generator."""
    pool = RNG_POOL
    if pool is None or not pool.active or generators:
        return None
    return pool.take(kind, shape, dtype, device)


def predraft_meta(starts, padded: int, width: int, block_size: int, window: int | None, table_width: int,
                  bucket=None):
    """Host inputs of the device-side draft bridge for rows ``0..len(starts)-1``.

    Returns the packed int64 ``[4, padded]`` tensor (base positions, active
    flags, first and last context pages) and the static context page count.
    The pages cover every possible accepted length: from the earliest query
    start (``start + 1``) back by the sliding window, to the last verified
    position (``start + width - 1``); the attention bias masks the rest.
    """
    meta = torch.zeros((4, padded), dtype=torch.int64)
    meta[3].fill_(-1)
    blocks = 1
    for row, start in enumerate(starts):
        first_position = 0 if window is None else max(0, start + 1 - (window - 1))
        first, last = first_position // block_size, (start + width - 1) // block_size
        meta[0, row], meta[1, row], meta[2, row], meta[3, row] = start, 1, first, last
        blocks = max(blocks, last - first + 1)
    if bucket is not None:
        blocks = bucket(blocks)
    return meta, min(blocks, table_width)


class PreDraftVerifier:
    """Compare the first pre-drafts with the host drafter's result for the same round.

    Both paths write the same draft KV values, so running both is harmless.
    Drafts only steer acceptance, never the sampled distribution; numerically
    different padding can flip an occasional argmax, so the gate is a row
    agreement rate rather than exact equality.
    """

    def __init__(self, rounds: int, min_agreement: float = 0.75):
        self.left = rounds
        self.min_agreement = min_agreement
        self.rows = 0
        self.equal = 0

    def compare(self, predraft: torch.Tensor, host: torch.Tensor) -> bool:
        """Record one round; returns False when the pre-draft must be disabled."""
        if tuple(predraft.shape) != tuple(host.shape):
            logger.error("Pre-draft shape %s differs from the host drafter %s; pre-draft disabled",
                         tuple(predraft.shape), tuple(host.shape))
            self.left = 0
            return False
        same = (predraft.to("cpu") == host.to("cpu")).all(dim=-1)
        self.rows += int(same.numel())
        self.equal += int(same.sum())
        self.left -= 1
        if self.left > 0:
            return True
        agreement = self.equal / max(1, self.rows)
        if agreement < self.min_agreement:
            logger.error("Pre-draft agreed with the host drafter on %d/%d rows (%.1f%%); pre-draft disabled",
                         self.equal, self.rows, 100 * agreement)
            return False
        logger.info("Pre-draft verified: %d/%d rows equal to the host drafter (%.1f%%)", self.equal, self.rows,
                    100 * agreement)
        return True


class ThreadIsolation:
    """Give the engine thread a physical core of its own (VLLM_HPU_PIN_MAIN_CPU).

    The runtime keeps one thread busy-polling at 100% CPU and several pipeline
    threads busy; whenever the scheduler puts one of them on the other
    hyperthread of the engine thread's core, every host step of a round slows
    down by about a quarter. Applied after warmup from the engine thread: that
    thread is pinned to ``cpu``; every other thread of the process may run
    anywhere except ``cpu`` and its hyperthread siblings. Threads created later
    are moved off the same way by ``refresh``.
    """

    def __init__(self, cpu: int):
        self.cpu = cpu
        self.main_tid = None
        self.others = None
        self.known: set[int] = set()
        self.failed = False

    @staticmethod
    def _siblings(cpu: int) -> set[int]:
        path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        cpus: set[int] = set()
        try:
            with open(path, encoding="utf-8") as handle:
                for part in handle.read().strip().split(","):
                    lo, _, hi = part.partition("-")
                    cpus.update(range(int(lo), int(hi or lo) + 1))
        except OSError:
            cpus.add(cpu)
        return cpus | {cpu}

    def _move_others(self) -> int:
        import os
        moved = 0
        for name in os.listdir("/proc/self/task"):
            tid = int(name)
            if tid == self.main_tid or tid in self.known:
                continue
            self.known.add(tid)
            try:
                os.sched_setaffinity(tid, self.others)
                moved += 1
            except OSError:  # the thread may have exited
                pass
        return moved

    def apply(self) -> None:
        import os
        import threading
        if self.failed or self.main_tid is not None:
            return
        try:
            allowed = os.sched_getaffinity(0)
            reserved = self._siblings(self.cpu)
            self.others = allowed - reserved
            if self.cpu not in allowed or not self.others:
                raise ValueError(f"cpu {self.cpu} not usable with affinity {sorted(allowed)}")
            self.main_tid = threading.get_native_id()
            moved = self._move_others()
            os.sched_setaffinity(0, {self.cpu})
            logger.info("Engine thread %d pinned to cpu %d; %d other threads kept off cpus %s", self.main_tid,
                        self.cpu, moved, sorted(reserved))
        except (OSError, ValueError) as exc:
            self.failed = True
            logger.warning("Thread isolation skipped: %s", exc)

    def refresh(self) -> None:
        if self.main_tid is None or self.failed:
            return
        try:
            self._move_others()
        except OSError:
            pass


class CompileTimers:
    """Where warmup compile time goes (VLLM_HPU_COMPILE_TIMES): Habana backend passes and the
    Synapse graph compile (which the recipe disk cache serves), next to Dynamo's own metrics."""

    PASSES = ("optimize_pre_placement", "optimize_pre_partitioner", "partition_module", "optimize_post_partitioner")

    def __init__(self):
        self.stats: dict = {}
        self.installed = False

    def _wrap(self, module, name, label):
        import time
        original = getattr(module, name, None)
        if original is None:
            return
        stats = self.stats

        @functools.wraps(original)
        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                entry = stats.setdefault(label, [0, 0.0])
                entry[0] += 1
                entry[1] += time.perf_counter() - start

        setattr(module, name, timed)

    def install(self) -> None:
        try:
            from habana_frameworks.torch.dynamo.compile_backend import _recipe_compiler_C, compilers
        except ImportError as exc:
            logger.warning("Compile timers unavailable: %s", exc)
            return
        self._wrap(compilers, "hpu_compiler_inner", "habana backend passes (total)")
        for name in self.PASSES:
            self._wrap(compilers, name, "  " + name)
        self._wrap(_recipe_compiler_C, "graph_compile", "synapse graph_compile (recipe cache)")
        self.installed = True

    def report(self, warmup_seconds: float) -> None:
        from torch._dynamo.utils import compile_times, counters
        lines = [f"Compile time breakdown over warmup ({warmup_seconds:.0f} s):"]
        for label, (calls, seconds) in self.stats.items():
            lines.append(f"  {label:45s} calls={calls:6d} total={seconds:8.1f} s")
        lines.append(f"  dynamo unique_graphs={counters['stats'].get('unique_graphs', 0)} "
                     f"frames={dict(counters['frames'])}")
        lines.append(compile_times(repr="str", aggregate=True))
        logger.info("\n".join(lines))


class NoGcMemoryProfiler:
    """HabanaMemoryProfiler without the full gc.collect() on entry and exit.

    Warmup wraps every bucket in a memory profiler only to log the graph
    memory total; each forced collection walks millions of Dynamo/FX objects.
    Exposes the same attributes the warmup loop reads."""

    def __enter__(self):
        from vllm_gaudi.extension.profiler import HabanaMemoryProfiler
        self._prof = HabanaMemoryProfiler
        self.initial_device_memory = HabanaMemoryProfiler.current_device_memory_usage()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.final_device_memory = self._prof.current_device_memory_usage()
        self.consumed_device_memory = self.final_device_memory - self.initial_device_memory
