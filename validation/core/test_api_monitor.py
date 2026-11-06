"""API request monitor (API_MONITOR=1): what the client receives is unchanged, and what the page shows.

    python -m pytest -q validation/core/test_api_monitor.py

The middleware is plain ASGI: these tests drive it with a stand-in for vLLM's app (no vLLM, no device).
"""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(os.environ.get("VLLM_GAUDI_SRC", ROOT / "vllm-gaudi" / "vllm_gaudi"))
_spec = importlib.util.spec_from_file_location("api_monitor_under_test", SRC / "entrypoints" / "api_monitor" / "__init__.py")
mon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mon)


class Counters:
    """vLLM's counters as the monitor reads them; tests move them as the engine would."""

    def __init__(self):
        self.values = {"vllm:num_requests_running": 0.0, "vllm:num_requests_waiting": 0.0,
                       "vllm:kv_cache_usage_perc": 0.0, "vllm:prompt_tokens_cached_total": 0.0}

    def __call__(self):
        return dict(self.values)

    def add(self, **kw):
        for k, v in kw.items():
            name = {"drafts": "vllm:spec_decode_num_drafts_total", "drafted": "vllm:spec_decode_num_draft_tokens_total",
                    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
                    "cached": "vllm:prompt_tokens_cached_total"}.get(k, k)
            self.values[name] = self.values.get(name, 0.0) + v

    def finish_request(self, queue, prefill, decode, ttft):
        for name, v in (("queue_time", queue), ("prefill_time", prefill), ("decode_time", decode)):
            self.add(**{f"vllm:request_{name}_seconds_sum": v, f"vllm:request_{name}_seconds_count": 1})
        self.add(**{"vllm:time_to_first_token_seconds_sum": ttft, "vllm:time_to_first_token_seconds_count": 1})


def sse(obj):
    return f"data: {json.dumps(obj)}\n\n".encode()


def chat_chunk(content=None, reasoning=None, finish=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning"] = reasoning
    return sse({"id": "chatcmpl-1", "object": "chat.completion.chunk", "choices": [
        {"index": 0, "delta": delta, "finish_reason": finish}]})


USAGE = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}


class FakeVllm:
    """Replies like vLLM: SSE when the request streams (with the usage chunk when asked), JSON otherwise."""

    def __init__(self, counters=None, status=200, raise_exc=None, gate=None):
        self.counters, self.status, self.raise_exc, self.gate = counters, status, raise_exc, gate
        self.bodies, self.headers = [], []

    async def __call__(self, scope, receive, send):
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        self.bodies.append(body)
        self.headers.append(dict(scope["headers"]))
        if self.raise_exc is not None:
            raise self.raise_exc
        req = json.loads(body)
        if self.status != 200:
            out = json.dumps({"error": {"message": "bad sampling params", "type": "BadRequestError", "code": 400}}).encode()
            await send({"type": "http.response.start", "status": self.status,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": out})
            return
        if self.counters is not None:
            self.counters.add(drafts=2, drafted=14, accepted=3, cached=4)
            self.counters.finish_request(0.01, 0.2, 0.5, 0.25)
        if self.gate is not None:
            await self.gate.wait()
        if req.get("stream"):
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"text/event-stream; charset=utf-8")]})
            await asyncio.sleep(0.02)                                # time to the first token
            for part in (chat_chunk(reasoning="think"), chat_chunk(content="Hel"), chat_chunk(content="lo", finish="stop")):
                await send({"type": "http.response.body", "body": part, "more_body": True})
            if (req.get("stream_options") or {}).get("include_usage"):
                await send({"type": "http.response.body", "more_body": True,
                            "body": sse({"id": "chatcmpl-1", "object": "chat.completion.chunk", "choices": [], "usage": USAGE})})
            await send({"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": False})
        else:
            out = json.dumps({"id": "chatcmpl-1", "object": "chat.completion", "usage": USAGE, "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hello", "reasoning": "think"}}]}).encode()
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(out)).encode())]})
            await send({"type": "http.response.body", "body": out})


def scope_for(path, method="POST", headers=(), app_state=None, query=b""):
    return {"type": "http", "method": method, "path": path, "root_path": "", "query_string": query,
            "headers": [(b"content-type", b"application/json"), *headers],
            "app": SimpleNamespace(state=SimpleNamespace(args=app_state)) if app_state is not None else None}


async def call(middleware, scope, body=b""):
    incoming = [{"type": "http.request", "body": body, "more_body": False}]
    sent = []

    async def receive():
        if incoming:
            return incoming.pop(0)
        await asyncio.sleep(3600)

    async def send(message):
        sent.append(message)

    await middleware(scope, receive, send)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    data = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, data, sent


def monitor(app, counters=None):
    return mon.ApiMonitor(app, counters=counters or Counters(), device=mon.DeviceTelemetry(binary=None))


def run(coro):
    return asyncio.run(coro)


def test_stream_gets_usage_for_the_monitor_but_the_client_sees_the_original_stream():
    counters = Counters()
    app = FakeVllm(counters)
    m = monitor(app, counters)
    req = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    status, data, _ = run(call(m, scope_for("/v1/chat/completions"), json.dumps(req).encode()))
    assert status == 200
    sent_on = json.loads(app.bodies[0])
    assert sent_on["stream_options"] == {"include_usage": True}
    assert app.headers[0][b"content-length"] == str(len(app.bodies[0])).encode()
    # exactly what vLLM sends without include_usage
    assert data == chat_chunk(reasoning="think") + chat_chunk(content="Hel") + chat_chunk(content="lo", finish="stop") + \
        b"data: [DONE]\n\n"
    record = m.records[0]
    assert record["state"] == "completed" and record["output"] == "Hello" and record["reasoning"] == "think"
    assert (record["prompt_tokens"], record["output_tokens"]) == (12, 5)
    assert record["finish_reason"] == "stop" and record["first_token_s"] is not None
    assert '"stream_options"' not in record["input"]                 # the input as the client sent it
    assert b'"usage"' not in record["response"].encode()
    engine = record["engine"]
    assert engine["queue_s"] == 0.01 and engine["prefill_s"] == 0.2 and engine["decode_s"] == 0.5
    assert record["cached_tokens"] == 4
    assert engine["spec"] == {"rounds": 2, "accepted": 3, "drafted": 14, "mean_acceptance_length": 2.5,
                              "acceptance_rate": round(3 / 14, 4)}
    assert record["decode_tok_s"] == 8.0                             # (5 - 1) tokens / 0.5 s
    assert record["prefill_tok_s"] == round((12 - 4) / record["first_token_s"], 1)   # new tokens / first token
    assert not m.active


def test_a_client_that_asked_for_usage_keeps_its_usage_chunk_and_body():
    app = FakeVllm()
    m = monitor(app)
    raw = json.dumps({"stream": True, "stream_options": {"include_usage": True}, "messages": []}).encode()
    _, data, _ = run(call(m, scope_for("/v1/chat/completions"), raw))
    assert app.bodies[0] == raw
    assert b'"usage"' in data and m.records[0]["output_tokens"] == 5


def test_whole_response_passes_unchanged_and_is_read():
    app = FakeVllm()
    m = monitor(app)
    raw = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
    status, data, sent = run(call(m, scope_for("/v1/chat/completions"), raw))
    assert app.bodies[0] == raw
    assert json.loads(data)["choices"][0]["message"]["content"] == "Hello"
    start = next(s for s in sent if s["type"] == "http.response.start")
    assert dict(start["headers"])[b"content-length"] == str(len(data)).encode()
    record = m.records[0]
    assert record["output"] == "Hello" and record["reasoning"] == "think" and record["output_tokens"] == 5
    assert record["state"] == "completed" and json.loads(record["response"])["id"] == "chatcmpl-1"
    assert record["first_token_s"] is None and record["prefill_tok_s"] is None   # no counters in this test


def test_error_response_is_an_error_record():
    m = monitor(FakeVllm(status=400))
    status, _, _ = run(call(m, scope_for("/v1/chat/completions"), b'{"messages": []}'))
    assert status == 400
    record = m.records[0]
    assert record["state"] == "error" and record["error"]["message"] == "bad sampling params"


def test_overlapping_requests_show_no_engine_figures():
    counters = Counters()
    gate = asyncio.Event()

    async def both():
        app = FakeVllm(counters, gate=gate)
        m = monitor(app, counters)
        first = asyncio.create_task(call(m, scope_for("/v1/chat/completions"), b'{"messages": []}'))
        second = asyncio.create_task(call(m, scope_for("/v1/chat/completions"), b'{"messages": []}'))
        await asyncio.sleep(0.01)
        assert len(m.active) == 2
        gate.set()
        await asyncio.gather(first, second)
        return m

    m = run(both())
    assert all(r["overlapped"] and r["engine"] is None and r["state"] == "completed" for r in m.records)
    assert all(r["output_tokens"] == 5 for r in m.records)           # usage is per request, still shown


def test_alone_after_an_overlap_is_measured_again():
    counters = Counters()
    m = monitor(FakeVllm(counters), counters)
    run(call(m, scope_for("/v1/chat/completions"), b'{"messages": []}'))
    run(call(m, scope_for("/v1/chat/completions"), b'{"messages": []}'))
    assert [r["overlapped"] for r in m.records] == [False, False]
    assert m.records[1]["engine"]["decode_s"] == 0.5
    assert m.records[1]["prefill_tok_s"] == 32.0              # whole response: (12 - 4) tokens / vLLM's TTFT 0.25 s


def test_disconnect_and_failure_states():
    m = monitor(FakeVllm(raise_exc=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        run(call(m, scope_for("/v1/chat/completions"), b'{"messages": []}'))
    assert m.records[0]["state"] == "disconnected"
    m = monitor(FakeVllm(raise_exc=RuntimeError("engine dead")))
    with pytest.raises(RuntimeError):
        run(call(m, scope_for("/v1/chat/completions"), b'{"messages": []}'))
    assert m.records[0]["state"] == "error" and "engine dead" in m.records[0]["error"]["message"]
    assert not m.active


def _stream_app(events, content_type=b"text/event-stream"):
    async def app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", content_type)]})
        for i, e in enumerate(events):
            await send({"type": "http.response.body", "body": e, "more_body": i < len(events) - 1})
    return app


def test_responses_api_stream():
    def ev(kind, **data):
        return f"event: {kind}\ndata: {json.dumps({'type': kind, **data})}\n\n".encode()
    events = [ev("response.created", response={}), ev("response.reasoning_text.delta", delta="hmm"),
              ev("response.output_text.delta", delta="Hi"), ev("response.output_text.delta", delta="!"),
              ev("response.completed", response={"status": "completed", "usage": {
                  "input_tokens": 9, "output_tokens": 3, "input_tokens_details": {"cached_tokens": 8}}})]
    m = monitor(_stream_app(events))
    _, data, _ = run(call(m, scope_for("/v1/responses"), b'{"stream": true, "input": "x"}'))
    assert data == b"".join(events)
    r = m.records[0]
    assert (r["output"], r["reasoning"], r["prompt_tokens"], r["output_tokens"]) == ("Hi!", "hmm", 9, 3)
    assert r["cached_tokens"] == 8 and r["finish_reason"] == "completed"


def test_anthropic_messages_stream():
    def ev(kind, **data):
        return f"event: {kind}\ndata: {json.dumps({'type': kind, **data})}\n\n".encode()
    events = [ev("message_start", message={"usage": {"input_tokens": 20, "output_tokens": 1}}),
              ev("content_block_delta", delta={"type": "thinking_delta", "thinking": "plan"}),
              ev("content_block_delta", delta={"type": "text_delta", "text": "Done"}),
              ev("message_delta", delta={"stop_reason": "end_turn"}, usage={"output_tokens": 7}), ev("message_stop")]
    m = monitor(_stream_app(events))
    _, data, _ = run(call(m, scope_for("/v1/messages"), b'{"stream": true, "messages": []}'))
    assert data == b"".join(events)
    r = m.records[0]
    assert (r["output"], r["reasoning"], r["prompt_tokens"], r["output_tokens"], r["finish_reason"]) == \
        ("Done", "plan", 20, 7, "end_turn")


def test_events_split_across_body_messages_are_reassembled():
    whole = chat_chunk(content="ab") + chat_chunk(content="cd", finish="stop") + b"data: [DONE]\n\n"
    pieces = [whole[:7], whole[7:50], whole[50:]]
    m = monitor(_stream_app(pieces))
    _, data, _ = run(call(m, scope_for("/v1/chat/completions"), b'{"stream": true, "stream_options": {"include_usage": true}}'))
    assert data == whole and m.records[0]["output"] == "abcd"


def test_inline_images_are_kept_short():
    image = "data:image/png;base64," + "A" * 100000
    m = monitor(FakeVllm())
    body = json.dumps({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}}]}]})
    run(call(m, scope_for("/v1/chat/completions"), body.encode()))
    assert len(m.records[0]["input"]) < 1000 and "100000 base64 characters" in m.records[0]["input"]


def test_other_routes_are_untouched():
    seen = []

    async def app(scope, receive, send):
        seen.append(scope["path"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    m = monitor(app)
    assert run(call(m, scope_for("/health", method="GET")))[1] == b"ok"
    assert run(call(m, scope_for("/v1/models", method="GET")))[1] == b"ok"
    assert seen == ["/health", "/v1/models"] and not m.records


def test_page_assets_and_data_routes():
    args = SimpleNamespace(api_key=None, served_model_name=["qwen"], max_model_len=262144, max_num_seqs=2,
                           speculative_config='{"method": "dflash", "num_speculative_tokens": 7}')
    m = monitor(FakeVllm())
    run(call(m, scope_for("/v1/chat/completions", app_state=args), b'{"messages": []}'))
    status, page, _ = run(call(m, scope_for("/api-monitor", "GET", app_state=args)))
    assert status == 200 and b'src="/api-monitor/monitor.js"' in page
    for asset in mon.ASSETS:
        assert run(call(m, scope_for(f"/api-monitor/{asset}", "GET", app_state=args)))[0] == 200
    assert run(call(m, scope_for("/api-monitor/../__init__.py", "GET", app_state=args)))[0] == 404
    status, data, _ = run(call(m, scope_for("/api/requests", "GET", app_state=args)))
    listing = json.loads(data)
    assert status == 200 and listing["retention"] == mon.KEEP and "input" not in listing["requests"][0]
    rid = listing["requests"][0]["id"]
    one = json.loads(run(call(m, scope_for("/api/requests", "GET", app_state=args, query=f"id={rid}".encode())))[1])
    assert one["output"] == "Hello" and "_clock" not in one
    assert run(call(m, scope_for("/api/requests", "GET", app_state=args, query=b"id=gone")))[0] == 404
    status = json.loads(run(call(m, scope_for("/api/monitor/status", "GET", app_state=args)))[1])
    assert (status["model"], status["max_model_len"], status["num_speculative_tokens"]) == ("qwen", 262144, 7)
    assert status["in_flight"] == 0 and status["device"] is None and status["device_reader"] is False


def test_data_routes_need_the_api_key_when_the_server_has_one():
    args = SimpleNamespace(api_key=["secret"], served_model_name=None)
    m = monitor(FakeVllm())
    assert run(call(m, scope_for("/api/requests", "GET", app_state=args)))[0] == 401
    assert run(call(m, scope_for("/api/monitor/status", "GET", app_state=args,
                                 headers=[(b"authorization", b"Bearer wrong")])))[0] == 401
    assert run(call(m, scope_for("/api/requests", "GET", app_state=args,
                                 headers=[(b"authorization", b"Bearer secret")])))[0] == 200
    assert run(call(m, scope_for("/api-monitor", "GET", app_state=args)))[0] == 200   # the page holds no data


def test_engine_status_from_counters():
    values = {"vllm:spec_decode_num_drafts_total": 100.0, "vllm:spec_decode_num_draft_tokens_total": 700.0,
              "vllm:spec_decode_num_accepted_tokens_total": 230.0,
              "vllm:spec_decode_num_accepted_tokens_per_pos_total#0": 70.0,
              "vllm:spec_decode_num_accepted_tokens_per_pos_total#1": 50.0,
              "vllm:prefix_cache_queries_total": 1000.0, "vllm:prefix_cache_hits_total": 900.0,
              "vllm:num_requests_running": 1.0, "vllm:num_requests_waiting": 0.0, "vllm:kv_cache_usage_perc": 0.25}
    s = mon.engine_status(values)
    assert s["spec"]["mean_acceptance_length"] == 3.3 and s["spec"]["acceptance_rate"] == round(230 / 700, 4)
    assert s["spec"]["per_position"] == [0.7, 0.5]
    assert s["prefix_cache_hit_rate"] == 0.9 and s["running"] == 1 and s["kv_cache_usage"] == 0.25
    assert mon.engine_status({}) is None


def test_engine_counters_read_vllm_style_prometheus_metrics():
    prometheus_client = pytest.importorskip("prometheus_client")
    registry = prometheus_client.CollectorRegistry()
    labels = ["model_name", "engine"]
    drafts = prometheus_client.Counter("vllm:spec_decode_num_drafts", "", labels, registry=registry)
    per_pos = prometheus_client.Counter("vllm:spec_decode_num_accepted_tokens_per_pos", "", labels + ["position"],
                                        registry=registry)
    queue = prometheus_client.Histogram("vllm:request_queue_time_seconds", "", labels, registry=registry)
    running = prometheus_client.Gauge("vllm:num_requests_running", "", labels, registry=registry)
    prometheus_client.Counter("vllm:unrelated", "", registry=registry).inc(5)
    drafts.labels("m", "0").inc(10)
    per_pos.labels("m", "0", "0").inc(7)
    per_pos.labels("m", "0", "1").inc(4)
    queue.labels("m", "0").observe(0.5)
    running.labels("m", "0").set(1)
    v = mon.engine_counters(registry)
    assert v["vllm:spec_decode_num_drafts_total"] == 10
    assert v["vllm:spec_decode_num_accepted_tokens_per_pos_total#0"] == 7
    assert v["vllm:spec_decode_num_accepted_tokens_per_pos_total#1"] == 4
    assert v["vllm:request_queue_time_seconds_sum"] == 0.5 and v["vllm:request_queue_time_seconds_count"] == 1
    assert v["vllm:num_requests_running"] == 1
    assert not any("unrelated" in k or k.endswith(("_bucket", "_created")) for k in v)


def test_hl_smi_reading():
    line = "HL-225, 33 C, 63 W, 91299 MiB, 98304 MiB, 0 %, 1650 MHz\n"
    d = mon.DeviceTelemetry.parse(line)
    assert d == {"name": "HL-225", "temperature_c": 33.0, "power_w": 63.0, "memory_used_mib": 91299.0,
                 "memory_total_mib": 98304.0, "utilization_pct": 0.0, "soc_clock_mhz": 1650.0}
    assert mon.DeviceTelemetry.parse("") is None
    assert mon.DeviceTelemetry(binary=None).current() is None
