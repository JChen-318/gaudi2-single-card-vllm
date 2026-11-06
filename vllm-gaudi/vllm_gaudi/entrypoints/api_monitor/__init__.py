"""API request monitor: the last requests, the engine and the card on one page (API_MONITOR=1 in serve.sh).

ASGI middleware for vLLM's `--middleware`, outside every other middleware, so it sees each request as the client
sent it and each response as the client receives it:

    GET /api-monitor              the page (web/monitor.html)
    GET /api/requests[?id=...]    the requests kept (newest first), or one with its input and output
    GET /api/monitor/status       the engine (vLLM's own Prometheus counters) and the card (hl-smi)

The last VLLM_API_MONITOR_KEEP (100) requests to the generation endpoints are kept in this process's memory with
their prompts and answers (each capped at 256K characters), and are gone after a restart. With --api-key the
/api/* routes need the same key; the page itself holds no data.

Responses reach the client as vLLM sends them. One request is changed on the way in: a streaming chat or
completion request that did not ask for usage gets stream_options.include_usage, and the usage chunk that adds
is taken out of the stream again, so the monitor has exact token counts.

Queue, prefill and decode times, cached prompt tokens and draft acceptance per request are differences of vLLM's
counters between the request's start and end. They belong to the request only while it runs alone (the focus of
this deployment); a request that overlapped another one shows none of them.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

WEB = Path(__file__).with_name("web")
CAP = 262144                       # characters kept per field (input, output, reasoning, response)
KEEP = max(1, int(os.environ.get("VLLM_API_MONITOR_KEEP", "100")))
TRACKED = ("/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages")
USAGE_INJECT = ("/v1/chat/completions", "/v1/completions")
ASSETS = {
    "monitor.js": "text/javascript; charset=utf-8",
    "tokens.css": "text/css; charset=utf-8",
    "favicon.svg": "image/svg+xml",
    "fonts/outfit-latin-wght.woff2": "font/woff2",
    "fonts/outfit-latin-ext-wght.woff2": "font/woff2",
}
LIST_HIDDEN = ("input", "output", "reasoning", "response")

# vLLM's Prometheus families the monitor reads (only these are collected: a full collect walks every histogram)
COUNTERS = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens", "vllm:spec_decode_num_accepted_tokens",
            "vllm:spec_decode_num_accepted_tokens_per_pos", "vllm:prefix_cache_queries", "vllm:prefix_cache_hits",
            "vllm:prompt_tokens_cached")
HISTOGRAMS = ("vllm:request_queue_time_seconds", "vllm:request_prefill_time_seconds",
              "vllm:request_decode_time_seconds", "vllm:time_to_first_token_seconds")
GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc")


def _registry():
    try:
        from vllm.v1.metrics.prometheus import get_prometheus_registry
        return get_prometheus_registry()
    except Exception:
        try:
            from prometheus_client import REGISTRY
            return REGISTRY
        except Exception:
            return None


def engine_counters(registry: Any = None) -> dict[str, float]:
    """vLLM's counters, gauges and histogram sums/counts summed over labels (per position for draft acceptance)."""
    registry = registry if registry is not None else _registry()
    if registry is None:
        return {}
    names = getattr(registry, "_names_to_collectors", None)
    try:
        if isinstance(names, dict):
            collectors, seen = [], set()
            for base in COUNTERS + HISTOGRAMS + GAUGES:
                collector = names.get(base)
                if collector is not None and id(collector) not in seen:
                    seen.add(id(collector))
                    collectors.append(collector)
            families = [family for collector in collectors for family in collector.collect()]
        else:
            families = list(registry.collect())
    except Exception:
        return {}
    values: dict[str, float] = {}
    for family in families:
        for sample in family.samples:
            name = sample.name
            if name.endswith(("_created", "_bucket")):
                continue
            if "position" in sample.labels:
                name = f"{name}#{sample.labels['position']}"
            values[name] = values.get(name, 0.0) + float(sample.value)
    return values


def engine_status(values: dict[str, float]) -> dict[str, Any] | None:
    """What the page's engine cards show, from engine_counters()."""
    if not values:
        return None
    drafts = values.get("vllm:spec_decode_num_drafts_total", 0.0)
    drafted = values.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    accepted = values.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    queries = values.get("vllm:prefix_cache_queries_total", 0.0)
    per_pos = sorted((int(k.rsplit("#", 1)[1]), v) for k, v in values.items()
                     if k.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total#"))
    return {
        "running": int(values.get("vllm:num_requests_running", 0)),
        "waiting": int(values.get("vllm:num_requests_waiting", 0)),
        "kv_cache_usage": values.get("vllm:kv_cache_usage_perc"),
        "prefix_cache_hit_rate": (round(values.get("vllm:prefix_cache_hits_total", 0.0) / queries, 4)
                                  if queries else None),
        "prefix_cache_queried_tokens": int(queries),
        "spec": {
            "rounds": int(drafts),
            "mean_acceptance_length": round(1 + accepted / drafts, 3) if drafts else None,
            "acceptance_rate": round(accepted / drafted, 4) if drafted else None,
            # share of verify rounds that accepted draft position i (0 = the first draft token)
            "per_position": [round(v / drafts, 4) for _, v in per_pos] if drafts else [],
        },
    }


def request_engine_figures(before: dict[str, float], after: dict[str, float]) -> dict[str, Any]:
    """One request's share of the counters: valid only when no other request ran at the same time."""
    def delta(key: str) -> float:
        return after.get(key, 0.0) - before.get(key, 0.0)

    out: dict[str, Any] = {}
    for name, key in (("queue_s", "vllm:request_queue_time_seconds"),
                      ("prefill_s", "vllm:request_prefill_time_seconds"),
                      ("decode_s", "vllm:request_decode_time_seconds"),
                      ("ttft_s", "vllm:time_to_first_token_seconds")):
        if delta(key + "_count") == 1:
            out[name] = round(delta(key + "_sum"), 4)
    if "vllm:prompt_tokens_cached_total" in after:
        out["cached_tokens"] = int(delta("vllm:prompt_tokens_cached_total"))
    rounds = delta("vllm:spec_decode_num_drafts_total")
    if rounds > 0:
        accepted = delta("vllm:spec_decode_num_accepted_tokens_total")
        drafted = delta("vllm:spec_decode_num_draft_tokens_total")
        out["spec"] = {"rounds": int(rounds), "accepted": int(accepted), "drafted": int(drafted),
                       "mean_acceptance_length": round(1 + accepted / rounds, 3),
                       "acceptance_rate": round(accepted / drafted, 4) if drafted else None}
    return out


class DeviceTelemetry:
    """The card's temperature, power, memory and utilization from hl-smi (about 0.2 s per call): read in a thread,
    at most every VLLM_API_MONITOR_HL_SMI_INTERVAL seconds (2) and only while the page is open (a reading is asked
    for), so an idle server never runs hl-smi."""

    FIELDS = ("name", "temperature.aip", "power.draw", "memory.used", "memory.total", "utilization.aip",
              "clocks.current.soc")

    def __init__(self, binary: str | None = "auto", interval: float | None = None) -> None:
        """binary: the hl-smi to run; "auto" finds it on PATH or in /usr/bin, None turns the readings off."""
        if binary == "auto":
            binary = shutil.which("hl-smi") or ("/usr/bin/hl-smi" if os.path.exists("/usr/bin/hl-smi") else None)
        self.binary = binary
        if interval is None:
            interval = float(os.environ.get("VLLM_API_MONITOR_HL_SMI_INTERVAL", "2"))
        self.interval = max(0.5, interval)
        self.value: dict[str, Any] | None = None
        self._at = 0.0
        self._lock = threading.Lock()

    @staticmethod
    def parse(text: str) -> dict[str, Any] | None:
        line = next((ln for ln in text.splitlines() if ln.strip()), "")
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < len(DeviceTelemetry.FIELDS):
            return None

        def number(cell: str) -> float | None:
            try:
                return float(cell.split()[0])
            except (ValueError, IndexError):
                return None

        name, temp, power, used, total, util, clock = cells[:7]
        return {"name": name, "temperature_c": number(temp), "power_w": number(power),
                "memory_used_mib": number(used), "memory_total_mib": number(total),
                "utilization_pct": number(util), "soc_clock_mhz": number(clock)}

    def _read(self) -> None:
        try:
            done = subprocess.run([self.binary, "-Q", ",".join(self.FIELDS), "-f", "csv,noheader"],
                                  capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL)
            value = self.parse(done.stdout) if done.returncode == 0 else None
            if value is not None:
                value["read_at"] = time.time()
            self.value = value
        except Exception:
            self.value = None
        finally:
            self._lock.release()

    def current(self) -> dict[str, Any] | None:
        """The last reading; starts a new one in the background when it is older than the interval."""
        if self.binary is None:
            return None
        now = time.monotonic()
        if now - self._at >= self.interval and self._lock.acquire(blocking=False):
            self._at = now
            threading.Thread(target=self._read, daemon=True, name="api-monitor-hl-smi").start()
        return self.value


def _shorten_data_urls(value: Any) -> Any:
    """Images sent inline (data: URLs) shown by type and size instead of megabytes of base64."""
    if isinstance(value, str):
        if value.startswith("data:") and len(value) > 512 and ";base64," in value[:100]:
            head = value.split(",", 1)[0]
            return f"{head},<{len(value) - len(head) - 1} base64 characters not kept by the monitor>"
        return value
    if isinstance(value, list):
        return [_shorten_data_urls(v) for v in value]
    if isinstance(value, dict):
        return {k: _shorten_data_urls(v) for k, v in value.items()}
    return value


def _capped(record: dict[str, Any], field: str, text: str) -> None:
    record[field] = text[:CAP]
    record[field + "_truncated"] = len(text) > CAP


def _append(record: dict[str, Any], field: str, text: str) -> None:
    if not text:
        return
    current = record[field]
    if len(current) >= CAP:
        record[field + "_truncated"] = True
        return
    combined = current + text
    record[field] = combined[:CAP]
    record[field + "_truncated"] = record[field + "_truncated"] or len(combined) > CAP


def _usage_numbers(usage: dict[str, Any] | None) -> tuple[int | None, int | None, int | None]:
    """(prompt, output, cached prompt) tokens from an OpenAI, Responses or Anthropic usage object."""
    if not isinstance(usage, dict):
        return None, None, None
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    output = usage.get("completion_tokens", usage.get("output_tokens"))
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    if cached is None:
        cached = usage.get("cache_read_input_tokens")
    return prompt, output, cached


def _error_of(obj: Any) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    error = obj.get("error")
    if isinstance(error, dict):
        return {"message": str(error.get("message") or error), "type": error.get("type")}
    if isinstance(error, str):
        return {"message": error}
    if obj.get("object") == "error" or obj.get("type") == "error":
        return {"message": str(obj.get("message") or obj.get("detail") or obj)}
    return None


class _Capture:
    """One request's record and the parsing of its response, streamed (SSE) or not."""

    def __init__(self, record: dict[str, Any], drop_usage_chunk: bool) -> None:
        self.record = record
        self.drop_usage_chunk = drop_usage_chunk
        self.sse = False
        self.buffer = b""
        self.body = bytearray()
        self.body_truncated = False
        self.raw: list[str] = []
        self.raw_len = 0
        self.last_token_clock: float | None = None

    # -- streamed responses ------------------------------------------------------------------------------------
    def feed(self, chunk: bytes, final: bool) -> bytes:
        """Bytes to pass on for this body chunk: whole events, minus the usage chunk the monitor asked for."""
        self.buffer += chunk
        out = bytearray()
        while True:
            end = self.buffer.find(b"\n\n")
            if end < 0:
                break
            event, self.buffer = self.buffer[:end + 2], self.buffer[end + 2:]
            if self._event(event):
                out += event
        if final and self.buffer:
            if self._event(self.buffer):
                out += self.buffer
            self.buffer = b""
        return bytes(out)

    def _event(self, event: bytes) -> bool:
        """Record one SSE event; False to leave it out of the stream."""
        try:
            text = event.decode("utf-8")
        except UnicodeDecodeError:
            return True
        name, data = None, []
        for line in text.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
        payload = "\n".join(data)
        obj = None
        if payload and payload != "[DONE]":
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                obj = None
        if (self.drop_usage_chunk and isinstance(obj, dict) and obj.get("choices") == []
                and obj.get("usage") is not None):
            self.record["usage"] = obj["usage"]
            return False
        self._raw(text)
        if isinstance(obj, dict):
            self._absorb(obj, name)
        return True

    def _raw(self, text: str) -> None:
        if self.raw_len < CAP:
            self.raw.append(text)
        self.raw_len += len(text)

    def _token(self) -> None:
        now = time.perf_counter()
        if self.record["first_token_s"] is None:
            self.record["first_token_s"] = round(now - self.record["_clock"], 3)
            if self.record["state"] == "running":
                self.record["state"] = "generating"
        self.last_token_clock = now

    def _absorb(self, obj: dict[str, Any], event: str | None) -> None:
        record = self.record
        error = _error_of(obj)
        if error is not None:
            record["error"] = error
            record["state"] = "error"
            return
        kind = obj.get("type") or event
        content, reasoning, activity = "", "", False
        for choice in obj.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            content += delta.get("content") or choice.get("text") or ""
            reasoning += delta.get("reasoning") or delta.get("reasoning_content") or ""
            activity = activity or bool(delta.get("tool_calls"))
            if choice.get("finish_reason"):
                record["finish_reason"] = choice["finish_reason"]
        if kind == "response.output_text.delta":
            content += obj.get("delta") or ""
        elif kind in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
            reasoning += obj.get("delta") or ""
        elif kind == "response.function_call_arguments.delta":
            activity = True
        elif kind in ("response.completed", "response.incomplete", "response.failed"):
            response = obj.get("response") or {}
            if response.get("usage"):
                record["usage"] = response["usage"]
            record["finish_reason"] = response.get("status") or kind.rsplit(".", 1)[1]
            if kind == "response.failed":
                record["error"] = _error_of(response) or {"message": "response failed"}
                record["state"] = "error"
        elif kind == "content_block_delta":
            delta = obj.get("delta") or {}
            content += delta.get("text") or ""
            reasoning += delta.get("thinking") or ""
            activity = activity or bool(delta.get("partial_json"))
        elif kind == "message_start":
            usage = (obj.get("message") or {}).get("usage")
            if isinstance(usage, dict):
                record["usage"] = dict(usage)
        elif kind == "message_delta":
            usage = obj.get("usage")
            if isinstance(usage, dict):
                record["usage"] = {**(record.get("usage") or {}), **usage}
            stop = (obj.get("delta") or {}).get("stop_reason")
            if stop:
                record["finish_reason"] = stop
        if obj.get("usage") and "choices" in obj:
            record["usage"] = obj["usage"]
        if content or reasoning or activity:
            self._token()
        _append(record, "output", content)
        _append(record, "reasoning", reasoning)

    # -- whole responses ---------------------------------------------------------------------------------------
    def keep(self, chunk: bytes) -> None:
        room = 4 * CAP - len(self.body)
        if room > 0:
            self.body += chunk[:room]
        self.body_truncated = self.body_truncated or len(chunk) > room

    def finish_body(self) -> None:
        record = self.record
        if self.sse:
            _capped(record, "response", "".join(self.raw))
            record["response_truncated"] = self.raw_len > CAP
            return
        text = self.body.decode("utf-8", errors="replace")
        try:
            obj = json.loads(text) if not self.body_truncated else None
        except json.JSONDecodeError:
            obj = None
        if obj is None:
            _capped(record, "response", text)
            record["response_truncated"] = record["response_truncated"] or self.body_truncated
            return
        _capped(record, "response", json.dumps(obj, ensure_ascii=False, indent=2))
        error = _error_of(obj)
        if error is not None:
            record["error"] = error
            return
        if isinstance(obj.get("usage"), dict):
            record["usage"] = obj["usage"]
        output, reasoning = [], []
        for choice in obj.get("choices") or []:
            message = choice.get("message") or {}
            output.append(message.get("content") or choice.get("text") or "")
            reasoning.append(message.get("reasoning") or message.get("reasoning_content") or "")
            if message.get("tool_calls"):
                output.append(json.dumps(message["tool_calls"], ensure_ascii=False, indent=2))
            if choice.get("finish_reason"):
                record["finish_reason"] = choice["finish_reason"]
        for item in obj.get("output") or []:                    # Responses API
            if not isinstance(item, dict):
                continue
            parts = (item.get("content") or []) + (item.get("summary") or [])
            target = reasoning if item.get("type") == "reasoning" else output
            target.extend(p.get("text") or "" for p in parts if isinstance(p, dict))
            if item.get("type") == "function_call":
                output.append(json.dumps(item, ensure_ascii=False, indent=2))
        if obj.get("type") == "message":                        # Anthropic Messages
            for block in obj.get("content") or []:
                if block.get("type") == "text":
                    output.append(block.get("text") or "")
                elif block.get("type") == "thinking":
                    reasoning.append(block.get("thinking") or "")
                elif block.get("type") == "tool_use":
                    output.append(json.dumps(block, ensure_ascii=False, indent=2))
            record["finish_reason"] = obj.get("stop_reason")
        if obj.get("object") == "response" or "output" in obj:
            record["finish_reason"] = obj.get("status") or record.get("finish_reason")
        _capped(record, "output", "\n".join(s for s in output if s))
        _capped(record, "reasoning", "\n".join(s for s in reasoning if s))


class ApiMonitor:
    """vLLM `--middleware vllm_gaudi.entrypoints.api_monitor.ApiMonitor`."""

    def __init__(self, app: Any, *, counters: Any = engine_counters, device: DeviceTelemetry | None = None) -> None:
        self.app = app
        self.records: deque[dict[str, Any]] = deque(maxlen=KEEP)
        self.active: dict[str, dict[str, Any]] = {}
        self.counters = counters
        self.device = device if device is not None else DeviceTelemetry()
        self.started_at = time.time()
        self._tokens: list[bytes] | None = None

    # -- routing -----------------------------------------------------------------------------------------------
    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "").removeprefix(scope.get("root_path", "")) or "/"
        method = scope.get("method")
        if method == "GET" and (path in ("/api-monitor", "/api/requests", "/api/monitor/status")
                                or path.startswith("/api-monitor/")):
            await self._serve(scope, send, path)
        elif method == "POST" and path in TRACKED:
            await self._track(scope, receive, send, path)
        else:
            await self.app(scope, receive, send)

    # -- the page and its data ---------------------------------------------------------------------------------
    @staticmethod
    async def _respond(send: Any, status: int, body: bytes, content_type: str) -> None:
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", content_type.encode()), (b"content-length", str(len(body)).encode()),
                                (b"cache-control", b"no-store")]})
        await send({"type": "http.response.body", "body": body})

    async def _json(self, send: Any, status: int, obj: Any) -> None:
        await self._respond(send, status, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    @staticmethod
    def _file(name: str) -> bytes:
        """Read on each request (a few small files, loaded once per page view), so a page fix needs no restart."""
        return (WEB / name).read_bytes()

    def _api_tokens(self, scope: dict[str, Any]) -> list[bytes]:
        """The server's API keys, hashed (vLLM's --api-key, else VLLM_API_KEY), as vLLM's own check uses them."""
        if self._tokens is None:
            args = getattr(getattr(scope.get("app"), "state", None), "args", None)
            keys = getattr(args, "api_key", None)
            keys = [keys] if isinstance(keys, str) else list(keys or [])
            if not keys and os.environ.get("VLLM_API_KEY"):
                keys = [os.environ["VLLM_API_KEY"]]
            self._tokens = [hashlib.sha256(k.encode()).digest() for k in keys if k]
        return self._tokens

    def _authorized(self, scope: dict[str, Any]) -> bool:
        tokens = self._api_tokens(scope)
        if not tokens:
            return True
        header = next((v.decode("latin-1") for k, v in scope.get("headers", []) if k.lower() == b"authorization"), "")
        scheme, _, key = header.partition(" ")
        if scheme.lower() != "bearer":
            return False
        digest = hashlib.sha256(key.encode()).digest()
        match = False
        for token in tokens:
            match |= secrets.compare_digest(digest, token)
        return match

    def _server_facts(self, scope: dict[str, Any]) -> dict[str, Any]:
        args = getattr(getattr(scope.get("app"), "state", None), "args", None)
        names = getattr(args, "served_model_name", None) or getattr(args, "model", None)
        names = [names] if isinstance(names, str) else list(names or [])
        spec = getattr(args, "speculative_config", None)
        if isinstance(spec, str):
            try:
                spec = json.loads(spec)
            except json.JSONDecodeError:
                spec = None
        return {"model": names[0] if names else None, "max_model_len": getattr(args, "max_model_len", None),
                "max_num_seqs": getattr(args, "max_num_seqs", None),
                "num_speculative_tokens": spec.get("num_speculative_tokens") if isinstance(spec, dict) else None}

    async def _serve(self, scope: dict[str, Any], send: Any, path: str) -> None:
        if path == "/api-monitor":
            await self._respond(send, 200, self._file("monitor.html"), "text/html; charset=utf-8")
            return
        if path.startswith("/api-monitor/"):
            name = path[len("/api-monitor/"):]
            if name in ASSETS:
                await self._respond(send, 200, self._file(name), ASSETS[name])
            else:
                await self._json(send, 404, {"error": {"message": "not found"}})
            return
        if not self._authorized(scope):
            await self._json(send, 401, {"error": {"message": "Unauthorized: this server needs its API key"}})
            return
        if path == "/api/monitor/status":
            await self._json(send, 200, self.status(scope))
            return
        query = dict(part.split("=", 1) for part in scope.get("query_string", b"").decode().split("&") if "=" in part)
        if query.get("id"):
            record = next((r for r in self.records if r["id"] == query["id"]), None)
            if record is None:
                await self._json(send, 404, {"error": {"message": "request no longer retained"}})
            else:
                await self._json(send, 200, self._view(record, full=True))
            return
        await self._json(send, 200, {"requests": [self._view(r) for r in reversed(self.records)],
                                     "retention": KEEP, "persistent": False})

    def status(self, scope: dict[str, Any]) -> dict[str, Any]:
        return {**self._server_facts(scope), "started_at": self.started_at, "in_flight": len(self.active),
                "engine": engine_status(self.counters()), "device": self.device.current(),
                "device_reader": self.device.binary is not None}

    @staticmethod
    def _view(record: dict[str, Any], full: bool = False) -> dict[str, Any]:
        view = {k: v for k, v in record.items() if not k.startswith("_") and (full or k not in LIST_HIDDEN)}
        if record["wallclock_s"] is None:
            view["wallclock_s"] = round(time.perf_counter() - record["_clock"], 3)
        return view

    # -- recording a request -----------------------------------------------------------------------------------
    async def _track(self, scope: dict[str, Any], receive: Any, send: Any, path: str) -> None:
        messages: list[dict[str, Any]] = []
        parts: list[bytes] = []
        while True:
            message = await receive()
            messages.append(message)
            if message.get("type") != "http.request":
                break
            parts.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        body = b"".join(parts)
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = None

        # the input as the client sent it (before the usage request below), inline images shortened
        input_text = (json.dumps(_shorten_data_urls(payload), ensure_ascii=False, indent=2) if payload is not None
                      else body.decode("utf-8", errors="replace"))
        options = payload.get("stream_options") if isinstance(payload, dict) else None
        inject = bool(isinstance(payload, dict) and payload.get("stream") is True and path in USAGE_INJECT
                      and (options is None or isinstance(options, dict)) and not (options or {}).get("include_usage"))
        if inject:
            body = json.dumps({**payload, "stream_options": {**(options or {}), "include_usage": True}},
                              ensure_ascii=False).encode()
            messages = [{"type": "http.request", "body": body, "more_body": False}] + \
                [m for m in messages if m.get("type") != "http.request"]
            scope = {**scope, "headers": [(k, v) for k, v in scope.get("headers", []) if k.lower() != b"content-length"]
                     + [(b"content-length", str(len(body)).encode())]}

        record: dict[str, Any] = {
            "id": uuid.uuid4().hex[:12], "path": path,
            "model": payload.get("model") if isinstance(payload, dict) else None,
            "started_at": time.time(), "state": "running",
            "stream": bool(isinstance(payload, dict) and payload.get("stream")),
            "http_status": None, "wallclock_s": None, "first_token_s": None, "finish_reason": None,
            "usage": None, "prompt_tokens": None, "output_tokens": None, "cached_tokens": None,
            "prefill_tok_s": None, "decode_tok_s": None, "engine": None, "overlapped": bool(self.active), "error": None,
            "input": "", "input_truncated": False, "output": "", "output_truncated": False,
            "reasoning": "", "reasoning_truncated": False, "response": "", "response_truncated": False,
            "_clock": time.perf_counter(), "_before": self.counters(),
        }
        _capped(record, "input", input_text)
        for other in self.active.values():
            other["overlapped"] = True
        self.active[record["id"]] = record
        self.records.append(record)
        capture = _Capture(record, drop_usage_chunk=inject)
        done = False

        replay = 0

        async def replay_receive() -> dict[str, Any]:
            nonlocal replay
            if replay < len(messages):
                message = messages[replay]
                replay += 1
            else:
                message = await receive()
            if message.get("type") == "http.disconnect" and not done:
                record["state"] = "disconnected"
            return message

        async def monitored_send(message: dict[str, Any]) -> None:
            nonlocal done
            kind = message.get("type")
            if kind == "http.response.start":
                record["http_status"] = message.get("status")
                content_type = next((v for k, v in message.get("headers", []) if k.lower() == b"content-type"), b"")
                capture.sse = b"text/event-stream" in content_type
                if record["http_status"] and record["http_status"] >= 400:
                    record["state"] = "error"
            elif kind == "http.response.body":
                final = not message.get("more_body", False)
                if capture.sse:
                    message = {**message, "body": capture.feed(message.get("body", b""), final)}
                else:
                    capture.keep(message.get("body", b""))
                if final:
                    done = True
            try:
                await send(message)
            except Exception:
                record["state"] = "disconnected"
                raise
            if done and record["wallclock_s"] is None:
                self._finish(record, capture)

        try:
            await self.app(scope, replay_receive, monitored_send)
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError) or record["state"] == "disconnected":
                record["state"] = "disconnected"
            else:
                record["state"] = "error"
                record["error"] = record["error"] or {"message": f"{type(exc).__name__}: {exc}"}
            raise
        finally:
            if record["wallclock_s"] is None:
                self._finish(record, capture)

    def _finish(self, record: dict[str, Any], capture: _Capture) -> None:
        record["wallclock_s"] = round(time.perf_counter() - record["_clock"], 3)
        try:
            capture.finish_body()
        except Exception as exc:                                # the monitor never breaks a response
            record["error"] = record["error"] or {"message": f"monitor could not read the response: {exc}"}
        if record["state"] not in ("error", "disconnected"):
            record["state"] = "error" if (record["error"] or (record["http_status"] or 500) >= 400) else "completed"
        prompt, output, cached = _usage_numbers(record["usage"])
        record["prompt_tokens"], record["output_tokens"] = prompt, output
        engine = None
        if not record["overlapped"] and record["state"] != "error":   # a rejected request moved no counters
            engine = request_engine_figures(record.pop("_before", {}) or {}, self.counters())
        record.pop("_before", None)
        record["engine"] = engine or None
        record["cached_tokens"] = cached if cached is not None else (engine or {}).get("cached_tokens")
        # prefill speed as a client measures it (llm_speedtest and the like): prompt tokens over the time to the first
        # token, here taken at the server (the stream's first token, else vLLM's own TTFT). Only the tokens the engine
        # computed count: a prefix-cache hit is not read again. Needs the request alone, for the cached count.
        first = record["first_token_s"] if record["first_token_s"] is not None else (engine or {}).get("ttft_s")
        computed = (prompt or 0) - (record["cached_tokens"] or 0)
        if engine and first and computed > 0:
            record["prefill_tok_s"] = round(computed / first, 1)
        decode_s = (engine or {}).get("decode_s")
        if output and output > 1 and decode_s:
            record["decode_tok_s"] = round((output - 1) / decode_s, 1)
        elif output and output > 1 and capture.last_token_clock and record["first_token_s"] is not None:
            span = capture.last_token_clock - record["_clock"] - record["first_token_s"]
            if span > 0:
                record["decode_tok_s"] = round((output - 1) / span, 1)
        self.active.pop(record["id"], None)
