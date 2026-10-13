# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: BLE001
"""One bounded decode-round trace; clocks across ordinary runner RPCs.

Only host scopes are inserted. No sampler rebinding, per-module synchronization,
HPU Event recording, tensor retention, model hooks, or inference algorithm edits.
"""
import json
import logging
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

import torch
from vllm_gaudi import envs

_ACTIVE = ContextVar('mtp_idle_diagnostic', default=None)
_PATH = envs.VLLM_HPU_MTP_IDLE_DIAGNOSTIC
_CPU_ONLY = getattr(envs, 'VLLM_HPU_MTP_IDLE_CPU_ONLY', False)
_LOG = logging.getLogger('vllm')


class IdleCapture:
    def __init__(self, request_id):
        self.path = Path(_PATH)
        if not self.path.is_absolute() or not self.path.parent.is_dir():
            raise ValueError('Idle diagnostic requires a new absolute path')
        with self.path.open('x', encoding='utf-8'):
            pass
        self.trace_path = self.path.with_name('idle-one-round-trace.json')
        if self.trace_path.exists():
            raise FileExistsError(self.trace_path)
        self.request_id = request_id
        self.cpu_only = _CPU_ONLY
        self.seen = 0
        self.recording = False
        self.finished = False
        self.failed = False
        self.profiler = None
        self.trace_started = False
        self.spans = []
        self.timeline = []
        self.errors = []
        self.export = {}

    def clock(self, name, function, *args, **kwargs):
        wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
        try:
            return function(*args, **kwargs)
        finally:
            self.timeline.append({'name': name, 'round_seen': self.seen, 'start_ns': wall,
                                  'end_ns': time.perf_counter_ns(),
                                  'thread_cpu_ns': time.thread_time_ns() - cpu})

    @contextmanager
    def span(self, name):
        wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
        # record_function annotates outside already-compiled callees. No events
        # are manually inserted into a stream at each module boundary.
        try:
            with nullcontext() if self.cpu_only else torch.profiler.record_function('mtp_idle/' + name):
                yield
        finally:
            self.spans.append({'name': name, 'round_seen': self.seen,
                               'start_ns': wall, 'end_ns': time.perf_counter_ns(),
                               'thread_cpu_ns': time.thread_time_ns() - cpu})

    def start(self):
        if self.cpu_only:
            self.recording = True
            return
        # RAM is context, not a hard eligibility threshold. Bound capture to
        # one round; disable shapes/stacks/allocations. A private hardware
        # buffer bounds device trace storage, not total profiler host memory.
        try:
            info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            self.export['mem_available_before_bytes'] = int(info['MemAvailable'].split()[0]) * 1024
        except (OSError, KeyError, ValueError) as exc:
            self.export['meminfo_error'] = repr(exc)[:300]
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU],
            record_shapes=False, with_stack=False, profile_memory=False)
        started = time.perf_counter_ns()
        self.profiler.start()
        self.trace_started = True
        self.export['profiler_start_wall_ns'] = time.perf_counter_ns() - started
        self.recording = True

    def save(self):
        data = {'request_id': self.request_id, 'selected_round': 16,
                'selected_clock_rounds': [9, 16] if self.cpu_only else [16, 16],
                'configured_trace_rounds': 0 if self.cpu_only else 1,
                'active_trace_rounds': int(self.trace_started), 'cpu_only': self.cpu_only,
                'memory_threshold_skip': False,
                'spans': self.spans, 'runner_timeline': self.timeline, 'errors': self.errors,
                'export': self.export, 'sampler_rebound': False, 'per_module_sync': False,
                'manual_device_events': False, 'record_shapes': False, 'with_stack': False,
                'profiling_is_not_a_throughput_benchmark': True,
                'clock_ns': 'perf_counter_ns',
                'limitations': ['Host spans include runtime work and waits; CPU is calling thread only',
                                'Nested spans are not additive',
                                'Trace start/stop/export overhead is outside the selected inference spans',
                                'Gap between worker RPCs includes scheduling and IPC; attribution needs trace/OS evidence']}
        try:
            self.path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        except OSError:
            _LOG.exception('MTP idle diagnostic could not save its own artifact')

    def stop(self):
        self.recording = False
        self.finished = True
        if self.cpu_only:
            self.save()
            return
        started = time.perf_counter_ns()
        try:
            self.profiler.stop()
            self.export['profiler_stop_wall_ns'] = time.perf_counter_ns() - started
            started = time.perf_counter_ns()
            self.profiler.export_chrome_trace(str(self.trace_path))
            self.export['trace_export_wall_ns'] = time.perf_counter_ns() - started
            self.export['trace_bytes'] = self.trace_path.stat().st_size
            self.export['trace_exceeds_download_limit'] = self.export['trace_bytes'] > 96 * 1024**2
            _LOG.info('MTP_IDLE_DIAGNOSTIC trace complete: rounds=1 bytes=%d', self.export['trace_bytes'])
        except Exception as exc:
            self.errors.append(repr(exc)[:300])
            _LOG.exception('Bounded MTP diagnostic export failed')
        finally:
            self.profiler = None
            self.save()


def phase(name):
    capture = _ACTIVE.get()
    return capture.span(name) if capture is not None and capture.recording else nullcontext()


def profile_stage(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            capture = _ACTIVE.get()
            if capture is not None and capture.recording:
                with capture.span(name):
                    return function(*args, **kwargs)
            runner = args[0] if args else None
            capture = getattr(runner, '_mtp_idle_capture', None)
            if capture is None or capture.failed or capture.seen >= 24:
                return function(*args, **kwargs)
            if name not in ('cpu_scheduler_state_update', 'cpu_draft_export_d2h'):
                return function(*args, **kwargs)
            if name == 'cpu_draft_export_d2h' and capture.recording:
                with capture.span(name):
                    result = capture.clock(name, function, *args, **kwargs)
                # This method performs the existing final D2H, so submitted
                # draft work has completed before stopping the sole trace.
                if not capture.cpu_only or capture.seen == 16:
                    capture.stop()
                return result
            return capture.clock(name, function, *args, **kwargs)
        return wrapped
    return decorate


def profile_round(function):
    @wraps(function)
    def wrapped(runner, *args, **kwargs):
        if (getattr(runner, '_mtp_idle_disabled', False) or not runner._is_qwen_mtp
                or runner.warmup_mode or runner.scheduler_output is None
                or runner.input_batch.num_reqs != 1
                or 'mtp-idle-check-' not in runner.input_batch.req_ids[0]
                or runner._is_prompt(0, runner.scheduler_output)):
            return function(runner, *args, **kwargs)
        capture = getattr(runner, '_mtp_idle_capture', None)
        if capture is None:
            try:
                capture = runner._mtp_idle_capture = IdleCapture(runner.input_batch.req_ids[0])
            except Exception:
                runner._mtp_idle_disabled = True
                _LOG.exception('MTP idle diagnostic initialization failed')
                return function(runner, *args, **kwargs)
        if capture.request_id != runner.input_batch.req_ids[0] or capture.failed or capture.seen >= 24:
            return function(runner, *args, **kwargs)
        capture.seen += 1
        if capture.seen == (9 if capture.cpu_only else 16):
            try:
                capture.start()
            except Exception as exc:
                capture.failed = True
                capture.errors.append(repr(exc)[:300])
                if capture.profiler is not None:
                    try:
                        capture.profiler.stop()
                    except Exception:
                        _LOG.exception('MTP idle profiler failed during startup cleanup')
                    capture.profiler = None
                capture.save()
                _LOG.exception('MTP idle capture did not start; generation continues')
        token = _ACTIVE.set(capture) if capture.recording else None
        try:
            if capture.recording:
                with capture.span('decode_round'):
                    return capture.clock('sample_tokens', function, runner, *args, **kwargs)
            return capture.clock('sample_tokens', function, runner, *args, **kwargs)
        except BaseException:
            if capture.recording:
                capture.stop()
            raise
        finally:
            if token is not None:
                _ACTIVE.reset(token)
            if capture.seen == 24:
                capture.save()
    return wrapped
