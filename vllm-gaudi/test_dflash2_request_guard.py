from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any

from dflash2_request_guard import DFlash2RequestGuard


class GuardHarness:
    def __init__(self) -> None:
        self.downstream_messages: list[dict[str, Any]] = []
        self.sent: list[dict[str, Any]] = []
        self.downstream_called = False

    async def run(
        self,
        payload: Any,
        *,
        path: str = "/v1/chat/completions",
        method: str = "POST",
        raw_body: bytes | None = None,
    ) -> tuple[bool, list[dict[str, Any]], list[dict[str, Any]]]:
        body = raw_body if raw_body is not None else json.dumps(payload).encode()
        incoming = [
            {"type": "http.request", "body": body, "more_body": False},
            {"type": "http.disconnect"},
        ]

        async def receive() -> dict[str, Any]:
            return incoming.pop(0)

        async def send(message: dict[str, Any]) -> None:
            self.sent.append(message)

        async def app(scope: dict[str, Any], recv: Any, sender: Any) -> None:
            self.downstream_called = True
            self.downstream_messages.append(await recv())
            self.downstream_messages.append(await recv())
            await sender({"type": "http.response.start", "status": 204, "headers": []})
            await sender({"type": "http.response.body", "body": b""})

        guard = DFlash2RequestGuard(app)
        scope = {"type": "http", "method": method, "path": path}
        await asyncio.wait_for(guard(scope, receive, send), timeout=1)
        return self.downstream_called, self.downstream_messages, self.sent


class DFlash2RequestGuardTests(unittest.IsolatedAsyncioTestCase):
    async def assert_passes(self, payload: Any, **kwargs: Any) -> None:
        called, received, sent = await GuardHarness().run(payload, **kwargs)
        self.assertTrue(called)
        self.assertEqual(received[0]["type"], "http.request")
        self.assertEqual(received[1]["type"], "http.disconnect")
        self.assertEqual(sent[0]["status"], 204)

    async def assert_rejected(self, payload: Any, **kwargs: Any) -> None:
        called, _received, sent = await GuardHarness().run(payload, **kwargs)
        self.assertFalse(called)
        self.assertEqual(sent[0]["type"], "http.response.start")
        self.assertEqual(sent[0]["status"], 400)

    async def test_safe_text_requests_pass_and_replay_once(self) -> None:
        cases = [
            {"messages": [{"role": "user", "content": "hello"}]},
            {"messages": [{"role": "user", "content": "hello"}], "temperature": 0},
            {"messages": [{"role": "user", "content": "hello"}], "temperature": 0.7},
            {
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
            },
            {
                "messages": [{"role": "user", "content": "hello"}],
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "repetition_penalty": 1,
                "min_p": 0,
                "min_tokens": 0,
                "allowed_token_ids": [],
                "bad_words": [],
                "logit_bias": {},
            },
            {"messages": [{"role": "user", "content": "hello"}], "tool_choice": "auto"},
            {"messages": [{"role": "user", "content": "hello"}], "tool_choice": "none"},
            {"messages": [{"role": "user", "content": "hello"}], "response_format": {"type": "text"}},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                await self.assert_passes(payload)

        await self.assert_passes(
            {"input": [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]},
            path="/v1/responses",
        )

    async def test_invalid_json_is_left_to_vllm_validation(self) -> None:
        await self.assert_passes(None, raw_body=b"{not-json")

    async def test_non_post_or_unrelated_route_bypasses_guard(self) -> None:
        await self.assert_passes({}, method="GET")
        await self.assert_passes({}, path="/health")

    async def test_forced_tool_choice_is_rejected(self) -> None:
        cases = [
            {"tool_choice": "required"},
            {"tool_choice": {"type": "function", "function": {"name": "f"}}},
            {"tool_choice": {"type": "function", "name": "f"}},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                await self.assert_rejected(payload)

    async def test_structured_output_variants_are_rejected(self) -> None:
        cases = [
            {"structured_outputs": {"json": {"type": "object"}}},
            {"guided_json": {"type": "object"}},
            {"guided_regex": "[0-9]+"},
            {"guided_choice": ["yes", "no"]},
            {"guided_grammar": "root ::= 'yes'"},
            {"response_format": {"type": "json_object"}},
            {"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}},
            {"text": {"format": {"type": "json_schema", "schema": {}}}},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                await self.assert_rejected(payload)

    async def test_packed_row_sampling_processors_are_rejected(self) -> None:
        cases = [
            {"frequency_penalty": 0.1},
            {"presence_penalty": -0.1},
            {"repetition_penalty": 1.1},
            {"min_p": 0.05},
            {"min_tokens": 2},
            {"allowed_token_ids": [1, 2]},
            {"bad_words": ["x"]},
            {"logit_bias": {"1": 10}},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                await self.assert_rejected(payload)

    async def test_multimodal_and_prompt_embeddings_are_rejected(self) -> None:
        cases = [
            {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]},
            {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "x"}}]}]},
            {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "x"}]}]},
            {"multi_modal_data": {"image": "x"}},
            {"prompt_embeds": "AA=="},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                await self.assert_rejected(payload)


if __name__ == "__main__":
    unittest.main()
