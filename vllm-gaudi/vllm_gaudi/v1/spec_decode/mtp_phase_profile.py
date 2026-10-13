# SPDX-License-Identifier: Apache-2.0
"""Opt-in, bounded CPU+HPU trace outside compiled MTP/model functions.

Host spans measure host work, not completed HPU compute. Attribute device
events using the exported profiler correlations; never add asynchronous
kernel durations to host span totals. Profiling is not a throughput benchmark.
"""

import logging
from contextlib import nullcontext
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

import torch
from vllm_gaudi import envs as gaudi_envs

_TRACE_PATH = gaudi_envs.VLLM_HPU_MTP_PHASE_TRACE
_SKIP_ROUNDS = 2
_ACTIVE_ROUNDS = 12
_REQUEST_MARKER = 'mtp-phase-check-'
_ACTIVE = ContextVar('mtp_phase_capture', default=None)
_LOGGER = logging.getLogger('vllm')


class MTPPhaseCapture:
    def __init__(self, path):
        self.path = Path(path)
        self.seen = 0
        self.collected = 0
        self.finished = False
        self.recording = False
        self.profiler = None
        self.request_id = None

    def enter_round(self):
        if self.finished:
            return False
        self.seen += 1
        if self.seen <= _SKIP_ROUNDS:
            return False
        if self.profiler is None:
            if not self.path.is_absolute() or self.path.suffix != '.json' or not self.path.parent.is_dir():
                self.finished = True
                raise ValueError('MTP phase trace needs a new absolute .json path in an existing directory')
            # Reserve this exact file exclusively. Export may replace only our
            # own newly-created placeholder, never an existing user artifact.
            self.finished = True
            with self.path.open('x', encoding='utf-8'):
                pass
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU],
                record_shapes=False, with_stack=False, profile_memory=False)
            self.profiler.start()
            self.recording = True
            self.finished = False
        return True

    def finish(self):
        if not self.recording:
            return
        self.recording = False
        self.finished = True
        self.profiler.stop()
        self.profiler.export_chrome_trace(str(self.path))
        _LOGGER.info('MTP_PHASE_TRACE saved=%s request_id=%s decode_rounds=%d skipped_rounds=%d '
                     'host_spans_are_not_device_times=True', self.path, self.request_id, self.collected, _SKIP_ROUNDS)

    def exit_round(self):
        self.collected += 1
        self.profiler.step()
        if self.collected >= _ACTIVE_ROUNDS:
            self.finish()


def phase(name):
    if not _TRACE_PATH:
        return nullcontext()
    capture = _ACTIVE.get()
    if capture is None or not capture.recording:
        return nullcontext()
    return torch.profiler.record_function('mtp_phase/' + name)


def _disable_capture(capture):
    """A diagnostic startup/export error must not replace inference output."""
    _LOGGER.exception('MTP_PHASE_TRACE disabled after diagnostic failure; inference continues without trace')
    try:
        capture.finish()
    except Exception:
        _LOGGER.exception('MTP_PHASE_TRACE cleanup failed')
    capture.finished = True
    capture.recording = False


def _finish_capture(capture):
    try:
        capture.finish()
    except Exception:
        _disable_capture(capture)


def profile_stage(name):
    def decorate(function):
        if not _TRACE_PATH:
            return function

        @wraps(function)
        def wrapped(*args, **kwargs):
            capture = _ACTIVE.get()
            if capture is None and args:
                capture = getattr(args[0], '_mtp_phase_capture', None)
            if capture is None or not capture.recording:
                return function(*args, **kwargs)
            with torch.profiler.record_function('mtp_phase/' + name):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def profile_round(function):
    if not _TRACE_PATH:
        return function

    @wraps(function)
    def wrapped(runner, *args, **kwargs):
        capture = getattr(runner, '_mtp_phase_capture', None)
        if (not runner._is_qwen_mtp or runner.warmup_mode or runner.scheduler_output is None
                or runner.input_batch.num_reqs != 1):
            if capture is not None and capture.recording:
                _finish_capture(capture)
            return function(runner, *args, **kwargs)
        request_id = runner.input_batch.req_ids[0]
        if _REQUEST_MARKER not in request_id:
            if capture is not None and capture.recording:
                _finish_capture(capture)
            return function(runner, *args, **kwargs)
        # This runner classifies prompt/decode with _is_prompt(), not an
        # InputBatch.num_decodes field. Use the same predicate as sampling,
        # including multi-token preempted prompts and speculative windows.
        if runner._is_prompt(0, runner.scheduler_output):
            if capture is not None and capture.recording:
                _finish_capture(capture)
            return function(runner, *args, **kwargs)
        if capture is None:
            capture = runner._mtp_phase_capture = MTPPhaseCapture(_TRACE_PATH)
        if capture.request_id is None:
            capture.request_id = request_id
        elif capture.request_id != request_id:
            _finish_capture(capture)
            capture.finished = True
            return function(runner, *args, **kwargs)
        try:
            selected = capture.enter_round()
        except Exception:
            _disable_capture(capture)
            return function(runner, *args, **kwargs)
        if not selected:
            return function(runner, *args, **kwargs)
        token = _ACTIVE.set(capture)
        try:
            with phase('decode_round'):
                result = function(runner, *args, **kwargs)
        except BaseException:
            try:
                capture.finish()
            except Exception:
                _LOGGER.exception('MTP_PHASE_TRACE failed to finalize after inference failure')
            raise
        else:
            try:
                capture.exit_round()
            except Exception:
                _disable_capture(capture)
            return result
        finally:
            _ACTIVE.reset(token)

    return wrapped


# Select the bounded observer at import time, keeping all compiled children
# and disabled decorators unchanged. Never construct the heavy profiler here.
if gaudi_envs.VLLM_HPU_MTP_HOST_TIMING:
    if _TRACE_PATH:
        raise ValueError('Select host timing or full phase trace, not both')
    from vllm_gaudi.v1.spec_decode.mtp_host_timing import phase, profile_round, profile_stage

if gaudi_envs.VLLM_HPU_MTP_LIGHT_TIMING:
    if _TRACE_PATH or gaudi_envs.VLLM_HPU_MTP_HOST_TIMING:
        raise ValueError('Light timing is mutually exclusive with other MTP profilers')
    from vllm_gaudi.v1.spec_decode.mtp_light_timing import phase, profile_round, profile_stage

if gaudi_envs.VLLM_HPU_MTP_IDLE_DIAGNOSTIC:
    if _TRACE_PATH or gaudi_envs.VLLM_HPU_MTP_HOST_TIMING or gaudi_envs.VLLM_HPU_MTP_LIGHT_TIMING:
        raise ValueError('Idle diagnosis is mutually exclusive with older observers')
    from vllm_gaudi.v1.spec_decode.mtp_idle_diagnostic import phase, profile_round, profile_stage
