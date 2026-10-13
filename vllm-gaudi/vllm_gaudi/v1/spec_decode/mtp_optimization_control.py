# SPDX-License-Identifier: Apache-2.0
"""Bounded local software control, sampled only when the request set changes."""
import json
import logging

from vllm_gaudi import envs

_FEATURES = ('handoff', 'sampling_cache', 'single_head')
_ACTIVE = dict.fromkeys(_FEATURES, True)
_REQUESTS = None
logger = logging.getLogger(__name__)


def refresh(request_ids):
    global _REQUESTS, _ACTIVE
    key = tuple(request_ids)
    if key == _REQUESTS:
        return
    _REQUESTS = key
    path = envs.VLLM_HPU_MTP_OPT_CONTROL
    if not path:
        return
    try:
        with open(path, encoding='utf-8') as stream:
            raw = stream.read(4097)
        if len(raw) > 4096:
            raise ValueError('Control exceeds the bounded size')
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != set(_FEATURES):
            raise ValueError('Control must contain only the three software switches')
        if any(type(flag) is not bool for flag in value.values()):
            raise ValueError('Control switches must be booleans')
        _ACTIVE = value
        logger.info('MTP software selection: %s', value)
    except (OSError, ValueError, TypeError) as exc:
        _ACTIVE = dict.fromkeys(_FEATURES, False)
        logger.warning('Invalid MTP software control; retain legacy paths: %s', exc)


def enabled(feature):
    flags = {'handoff': envs.VLLM_HPU_MTP_HANDOFF_OPT,
             'sampling_cache': envs.VLLM_HPU_MTP_SAMPLING_CACHE,
             'single_head': envs.VLLM_HPU_MTP_SINGLE_HEAD_COMPILE}
    return bool(flags[feature] and _ACTIVE[feature])
