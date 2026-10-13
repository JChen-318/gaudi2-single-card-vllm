# SPDX-License-Identifier: Apache-2.0
"""Keep unported request modes out of the experimental HPU MTP engine."""

from dflash2_request_guard import DFlash2RequestGuard


class MTPRequestGuard(DFlash2RequestGuard):
    """MTP shares the rectangular HPU sampler's current API restrictions."""

    @staticmethod
    def _error(message):
        message = message.replace('DFlash2 on Gaudi 1.24.1', 'Qwen MTP initial HPU port')
        message = message.replace('DFlash2', 'Qwen MTP initial port')
        return DFlash2RequestGuard._error(message)
