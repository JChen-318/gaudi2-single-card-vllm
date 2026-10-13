# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: BLE001
"""Bounded host/stream timing outside compiled functions; never a full trace.

No per-phase synchronization. Event elapsed includes stream gaps, not kernel
busy time. A pending round remains open for the separate draft-export RPC.
"""

import json
import logging
import math
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

import torch
from vllm_gaudi import envs as gaudi_envs

_PATH = gaudi_envs.VLLM_HPU_MTP_HOST_TIMING
_EVENTS = gaudi_envs.VLLM_HPU_MTP_HOST_EVENTS
_ACTIVE = ContextVar('mtp_host_timing', default=None)
_LOGGER = logging.getLogger('vllm')
_SKIP = 2
_LIMIT = 12
_MARKER = 'mtp-host-check-'
_DEVICE_STAGES = frozenset(('target_model_forward', 'target_logits', 'target_rejection_sampling',
                          'sampling_bonus', 'sampling_constraints', 'sampling_correction',
                          'draft_head_forward', 'draft_logits_selection'))


class _BonusObserver(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, *args, **kwargs):
        with phase('sampling_bonus'):
            return self.inner(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__('inner'), name)


class HostCapture:
    def __init__(self, runner, request_id):
        self.runner = runner
        self.request_id = request_id
        self.seen = 0
        self.finished = False
        self.current = None
        self.rounds = []
        self.stack = []
        self.errors = []
        self.events_enabled = _EVENTS
        self.original_bonus = None
        self.observer = None
        self.path = Path(_PATH)
        self.reserved = False
        self.identity = {}

    def error(self, exc):
        if len(self.errors) < 8:
            self.errors.append(type(exc).__name__ + ': ' + str(exc)[:200])

    def begin(self, scheduled):
        if self.current is not None:
            self.seal(False)
        if self.finished:
            return False
        self.seen += 1
        if self.seen <= _SKIP:
            return False
        if not self.reserved:
            if not self.path.is_absolute() or self.path.suffix != '.json' or not self.path.parent.is_dir():
                raise ValueError('Host timing requires a new absolute .json in an existing directory')
            with self.path.open('x', encoding='utf-8'):
                pass
            self.reserved = True
            sampler = self.runner.rejection_sampler.sampler
            self.identity = {
                'bonus_is_runner_sampler_before_observer': sampler is self.runner.sampler,
                'bonus_is_runner_sampler_orig_mod': sampler is getattr(self.runner.sampler, '_orig_mod', None),
                'bonus_type': type(sampler).__name__, 'runner_sampler_type': type(self.runner.sampler).__name__,
            }
            self.original_bonus = sampler
            self.observer = _BonusObserver(sampler)
            self.runner.rejection_sampler.sampler = self.observer
        self.current = {'index': self.seen, 'scheduled_tokens': scheduled, 'stages': {},
                        'events': [], 'sample_start_ns': time.perf_counter_ns()}
        return True

    @contextmanager
    def span(self, name):
        pair = None
        if self.events_enabled and name in _DEVICE_STAGES:
            try:
                pair = (torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True))
                pair[0].record()
            except Exception as exc:
                self.error(exc)
                self.events_enabled = False
                pair = None
        frame = [name, time.perf_counter_ns(), 0]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - frame[1]
            self.stack.pop()
            if self.stack:
                self.stack[-1][2] += elapsed
            if self.current is not None:
                stats = self.current['stages'].setdefault(name, {'calls': 0, 'host_ns': 0, 'exclusive_host_ns': 0})
                stats['calls'] += 1
                stats['host_ns'] += elapsed
                stats['exclusive_host_ns'] += max(0, elapsed - frame[2])
                if pair is not None:
                    try:
                        pair[1].record()
                        self.current['events'].append((name, pair))
                    except Exception as exc:
                        self.error(exc)
                        self.events_enabled = False

    def seal(self, exported):
        if self.current is None:
            return
        item = self.current
        self.current = None
        item['draft_export_completed'] = exported
        item['sample_through_export_ms'] = (time.perf_counter_ns() - item.pop('sample_start_ns')) / 1e6
        for name, pair in item.pop('events'):
            stats = item['stages'][name]
            try:
                if not pair[0].query() or not pair[1].query():
                    stats['stream_unavailable_calls'] = stats.get('stream_unavailable_calls', 0) + 1
                    continue
                value = float(pair[0].elapsed_time(pair[1]))
                if not math.isfinite(value) or value < 0:
                    raise ValueError('Device event returned an invalid elapsed time')
                stats['stream_ms'] = stats.get('stream_ms', 0.0) + value
                stats['stream_measured_calls'] = stats.get('stream_measured_calls', 0) + 1
            except Exception as exc:
                self.error(exc)
                stats['stream_unavailable_calls'] = stats.get('stream_unavailable_calls', 0) + 1
        self.rounds.append(item)
        if len(self.rounds) >= _LIMIT:
            self.finish()

    def finish(self):
        if self.finished:
            return
        self.finished = True
        if self.observer is not None and self.runner.rejection_sampler.sampler is self.observer:
            self.runner.rejection_sampler.sampler = self.original_bonus
        self.observer = self.original_bonus = None
        if not self.reserved:
            return
        result = {'kind': 'bounded_mtp_host_stream_timing', 'request_id': self.request_id,
                  'skipped_rounds': _SKIP, 'round_limit': _LIMIT, 'rounds': self.rounds,
                  'sampler_identity_before_observer': self.identity, 'diagnostic_errors': self.errors,
                  'full_profiler_used': False, 'per_phase_synchronize_used': False,
                  'stream_elapsed_is_not_kernel_busy': True, 'host_and_stream_times_are_not_additive': True}
        with self.path.open('w', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False)
        _LOGGER.info('MTP_HOST_TIMING saved=%s rounds=%d full_profiler=False', self.path, len(self.rounds))


def _disable(capture, exc):
    # Backend event/export failures vary by runtime; diagnostics must never
    # replace an inference result with a version-specific exception.
    capture.error(exc)
    try:
        if capture.current is not None:
            capture.seal(False)
        capture.finish()
    except Exception as cleanup_error:
        capture.error(cleanup_error)
        capture.finished = True


def phase(name):
    capture = _ACTIVE.get() if _PATH else None
    if capture is None or capture.finished or capture.current is None:
        return nullcontext()
    return capture.span(name)


def profile_stage(name):
    def decorate(function):
        if not _PATH:
            return function

        @wraps(function)
        def wrapped(*args, **kwargs):
            capture = _ACTIVE.get()
            export = name == 'cpu_draft_export_d2h'
            if capture is None and export and args:
                capture = getattr(args[0], '_mtp_host_capture', None)
            if capture is None or capture.finished or capture.current is None:
                return function(*args, **kwargs)
            if export and list(capture.runner.input_batch.req_ids) != [capture.request_id]:
                capture.seal(False)
                capture.finish()
                return function(*args, **kwargs)
            # The caller's exception/output must survive all diagnostic errors.
            try:
                with capture.span(name):
                    result = function(*args, **kwargs)
            except BaseException:
                _disable(capture, RuntimeError('inference stage failed'))
                raise
            if export:
                try:
                    capture.seal(True)
                except Exception as exc:
                    _disable(capture, exc)
            return result

        return wrapped

    return decorate


def profile_round(function):
    if not _PATH:
        return function

    @wraps(function)
    def wrapped(runner, *args, **kwargs):
        capture = getattr(runner, '_mtp_host_capture', None)
        eligible = (runner._is_qwen_mtp and not runner.warmup_mode and runner.scheduler_output is not None
                    and runner.input_batch.num_reqs == 1)
        request_id = runner.input_batch.req_ids[0] if eligible else ''
        eligible = eligible and _MARKER in request_id and not runner._is_prompt(0, runner.scheduler_output)
        if not eligible:
            if capture is not None and not capture.finished:
                try:
                    capture.seal(False)
                    capture.finish()
                except Exception as exc:
                    _disable(capture, exc)
            return function(runner, *args, **kwargs)
        if capture is None:
            capture = runner._mtp_host_capture = HostCapture(runner, request_id)
        if capture.finished:
            return function(runner, *args, **kwargs)
        if capture.request_id != request_id:
            _disable(capture, RuntimeError('request changed'))
            return function(runner, *args, **kwargs)
        try:
            selected = capture.begin(int(runner.scheduler_output.num_scheduled_tokens[request_id]))
        except Exception as exc:
            _disable(capture, exc)
            return function(runner, *args, **kwargs)
        if not selected:
            return function(runner, *args, **kwargs)
        token = _ACTIVE.set(capture)
        try:
            with capture.span('decode_round'):
                return function(runner, *args, **kwargs)
        except BaseException:
            _disable(capture, RuntimeError('inference round failed'))
            raise
        finally:
            _ACTIVE.reset(token)

    return wrapped
