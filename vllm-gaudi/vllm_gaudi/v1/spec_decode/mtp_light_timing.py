# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: BLE001
"""Eight coarse rounds. No profiler, sampler observer, or extra synchronization."""
import json
import logging
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

import torch
from vllm_gaudi import envs as gaudi_envs

_PATH = gaudi_envs.VLLM_HPU_MTP_LIGHT_TIMING
_ACTIVE = ContextVar('mtp_light_timing', default=None)
_DEVICE_STAGES = frozenset(('target_model_forward', 'target_rejection_sampling', 'draft_generation'))
_LOGGER = logging.getLogger('vllm')


class LightCapture:
    def __init__(self, request_id):
        self.path = Path(_PATH)
        if not self.path.is_absolute() or not self.path.parent.is_dir() or self.path.suffix != '.json':
            raise ValueError('Light timing needs a new absolute JSON path in an existing directory')
        with self.path.open('x', encoding='utf-8'):
            pass
        self.request_id = request_id
        self.seen = 0
        self.rounds = []
        self.current = None
        self.events = []
        self.finished = False
        self.errors = []

    def error(self, exc):
        if len(self.errors) < 8:
            self.errors.append(type(exc).__name__ + ': ' + str(exc)[:160])

    @contextmanager
    def span(self, name):
        pair = None
        if name in _DEVICE_STAGES:
            try:
                pair = (torch.hpu.Event(enable_timing=True), torch.hpu.Event(enable_timing=True))
                pair[0].record()
            except Exception as exc:
                self.error(exc)
                pair = None
        wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
        try:
            yield
        finally:
            wall, cpu = time.perf_counter_ns() - wall, time.thread_time_ns() - cpu
            stats = self.current['stages'].setdefault(name, {'calls': 0, 'wall_ns': 0, 'thread_cpu_ns': 0})
            stats['calls'] += 1
            stats['wall_ns'] += wall
            stats['thread_cpu_ns'] += cpu
            if pair is not None:
                try:
                    pair[1].record()
                    self.events.append((name, pair))
                except Exception as exc:
                    self.error(exc)

    def seal(self, runner):
        if self.current is None:
            return
        for name, pair in self.events:
            stats = self.current['stages'][name]
            stats['stream_elapsed_ms_includes_gaps'] = None
            try:
                if pair[0].query() and pair[1].query():
                    stats['stream_elapsed_ms_includes_gaps'] = pair[0].elapsed_time(pair[1])
            except Exception as exc:
                self.error(exc)
        self.events = []
        pool = getattr(runner.drafter, '_mtp_metadata_pool', None)
        if pool is not None:
            self.current['metadata_pool_cumulative_stats'] = dict(pool.stats)
        self.rounds.append(self.current)
        self.current = None
        if len(self.rounds) >= 8:
            self.finished = True
            data = {'request_id': self.request_id, 'rounds': self.rounds, 'errors': self.errors,
                'full_profiler': False, 'sampler_rebound': False, 'per_phase_synchronize': False,
                'limitations': ['Diagnostic request is not an undisturbed speed benchmark',
                             'Nested stage times are not additive',
                             'thread CPU measures only the calling thread, not backend workers',
                             'wall minus thread CPU includes scheduling and dependency waits',
                             'stream elapsed includes submission gaps, not summed kernel busy time']}
            try:
                self.path.write_text(json.dumps(data, indent=2), encoding='utf-8')
            except Exception as exc:
                self.error(exc)
                _LOGGER.warning('MTP light diagnostic could not save: %s', exc)


def phase(name):
    capture = _ACTIVE.get()
    return capture.span(name) if capture is not None and capture.current is not None else nullcontext()


def profile_stage(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            capture = _ACTIVE.get()
            if capture is None and name == 'cpu_draft_export_d2h' and args:
                capture = getattr(args[0], '_mtp_light_capture', None)
            if capture is None or capture.finished or capture.current is None:
                return function(*args, **kwargs)
            with capture.span(name):
                result = function(*args, **kwargs)
            if name == 'cpu_draft_export_d2h':
                capture.seal(args[0])
            return result
        return wrapped
    return decorate


def profile_round(function):
    @wraps(function)
    def wrapped(runner, *args, **kwargs):
        if (getattr(runner, '_mtp_light_disabled', False) or not runner._is_qwen_mtp
                or runner.warmup_mode or runner.scheduler_output is None
                or runner.input_batch.num_reqs != 1
                or 'mtp-submit-check-' not in runner.input_batch.req_ids[0]
                or runner._is_prompt(0, runner.scheduler_output)):
            return function(runner, *args, **kwargs)
        capture = getattr(runner, '_mtp_light_capture', None)
        if capture is None:
            try:
                capture = runner._mtp_light_capture = LightCapture(runner.input_batch.req_ids[0])
            except Exception as exc:
                runner._mtp_light_disabled = True
                _LOGGER.warning('MTP light diagnostic disabled: %s', exc)
                return function(runner, *args, **kwargs)
        if capture.finished or capture.request_id != runner.input_batch.req_ids[0]:
            return function(runner, *args, **kwargs)
        capture.seen += 1
        if capture.seen <= 2:
            return function(runner, *args, **kwargs)
        if capture.current is not None:
            capture.error(RuntimeError('Previous diagnostic draft export was not completed'))
            capture.finished = True
            return function(runner, *args, **kwargs)
        capture.current = {'index': capture.seen, 'stages': {}}
        token = _ACTIVE.set(capture)
        try:
            with capture.span('decode_round'):
                return function(runner, *args, **kwargs)
        finally:
            _ACTIVE.reset(token)
    return wrapped
