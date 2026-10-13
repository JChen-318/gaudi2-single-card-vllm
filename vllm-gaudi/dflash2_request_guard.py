"""Reject request modes that the Gaudi DFlash2 runner cannot safely handle."""

from __future__ import annotations

import json
from typing import Any

from starlette.responses import JSONResponse


class DFlash2RequestGuard:
    """ASGI middleware protecting EngineCore from unsupported request modes."""

    _PATHS = {"/v1/chat/completions", "/v1/completions", "/v1/responses"}
    _STRUCTURED_KEYS = {
        "guided_choice",
        "guided_grammar",
        "guided_json",
        "guided_regex",
        "structural_tag",
        "structured_outputs",
    }
    _MULTIMODAL_TYPES = {
        "audio",
        "audio_url",
        "image",
        "image_url",
        "input_audio",
        "input_image",
        "input_video",
        "video",
        "video_url",
    }
    _MULTIMODAL_KEYS = {
        "audio",
        "audio_url",
        "image",
        "image_url",
        "input_audio",
        "input_image",
        "input_video",
        "video",
        "video_url",
    }

    def __init__(self, app: Any) -> None:
        self.app = app

    @staticmethod
    def _error(message: str) -> JSONResponse:
        return JSONResponse(
            {
                "error": {
                    "message": message,
                    "type": "BadRequestError",
                    "param": None,
                    "code": 400,
                }
            },
            status_code=400,
        )

    @classmethod
    def _contains_multimodal_input(cls, value: Any) -> bool:
        if isinstance(value, list):
            return any(cls._contains_multimodal_input(item) for item in value)
        if not isinstance(value, dict):
            return False

        part_type = value.get("type")
        if isinstance(part_type, str) and part_type.lower() in cls._MULTIMODAL_TYPES:
            return True
        if any(key in value and value[key] is not None for key in cls._MULTIMODAL_KEYS):
            return True
        return any(cls._contains_multimodal_input(item) for item in value.values())

    @classmethod
    def _uses_structured_output(cls, payload: dict[str, Any]) -> bool:
        if any(payload.get(key) is not None for key in cls._STRUCTURED_KEYS):
            return True

        response_format = payload.get("response_format")
        if isinstance(response_format, dict):
            response_type = response_format.get("type")
            if response_type not in (None, "text"):
                return True

        # The Responses API nests its response format under text.format.
        text_config = payload.get("text")
        if isinstance(text_config, dict):
            text_format = text_config.get("format")
            if isinstance(text_format, dict):
                format_type = text_format.get("type")
                if format_type not in (None, "text"):
                    return True
        return False

    @staticmethod
    def _uses_unsupported_sampling_processor(payload: dict[str, Any]) -> bool:
        """Reject processors that still assume packed speculative rows."""
        numeric_defaults = {
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "min_p": 0.0,
            "min_tokens": 0.0,
        }
        for key, default in numeric_defaults.items():
            value = payload.get(key)
            if value is None:
                continue
            try:
                if float(value) != default:
                    return True
            except (TypeError, ValueError):
                # Leave malformed values to vLLM's normal request validation.
                continue

        for key in ("allowed_token_ids", "bad_words", "logit_bias"):
            if payload.get(key):
                return True
        # These use packed-token indexing or Triton logprob assembly in the
        # core sampler and have not been adapted to HPU rectangular rows.
        for key in ("prompt_logprobs", "thinking_token_budget"):
            if payload.get(key) is not None:
                return True
        logprobs = payload.get("logprobs")
        if logprobs is not None and logprobs is not False:
            return True
        if payload.get("logprob_token_ids") or payload.get("top_logprobs"):
            return True
        if payload.get("include") and "message.output_text.logprobs" in payload["include"]:
            return True
        return False

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in self._PATHS
        ):
            await self.app(scope, receive, send)
            return

        messages: list[dict[str, Any]] = []
        body_parts: list[bytes] = []
        while True:
            message = await receive()
            messages.append(message)
            if message.get("type") != "http.request":
                break
            body_parts.append(message.get("body", b""))
            if not message.get("more_body", False):
                break

        try:
            payload = json.loads(b"".join(body_parts))
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = None

        if isinstance(payload, dict):
            tool_choice = payload.get("tool_choice")
            if tool_choice == "required" or isinstance(tool_choice, dict):
                response = self._error(
                    "HPU DFlash2 on Gaudi 1.24.1 does not safely support forced "
                    "structured tool choice; use tool_choice='auto'."
                )
                await response(scope, receive, send)
                return

            if self._uses_structured_output(payload):
                response = self._error(
                    "HPU DFlash2 on Gaudi 1.24.1 does not safely support "
                    "structured-output or guided-decoding requests."
                )
                await response(scope, receive, send)
                return

            if self._uses_unsupported_sampling_processor(payload):
                response = self._error(
                    "HPU DFlash2 sampling currently supports temperature, "
                    "top_p, and top_k, but not penalty, min_p/min_tokens, "
                    "allowed-token, bad-word, logit-bias, thinking-budget, "
                    "or logprobs requests."
                )
                await response(scope, receive, send)
                return

            if (
                payload.get("prompt_embeds") is not None
                or payload.get("multi_modal_data") is not None
                or self._contains_multimodal_input(payload.get("messages"))
                or self._contains_multimodal_input(payload.get("input"))
            ):
                response = self._error(
                    "HPU DFlash2 currently supports text-only requests; "
                    "multimodal inputs and prompt embeddings are disabled."
                )
                await response(scope, receive, send)
                return

        replay_index = 0

        async def replay_receive() -> dict[str, Any]:
            nonlocal replay_index
            if replay_index < len(messages):
                message = messages[replay_index]
                replay_index += 1
                return message
            # Once the buffered request body has been replayed, delegate back to
            # Uvicorn.  Returning an empty http.request forever makes Starlette's
            # disconnect listener spin and can monopolize the API event loop.
            return await receive()

        await self.app(scope, replay_receive, send)
