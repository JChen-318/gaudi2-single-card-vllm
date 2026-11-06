"use strict";
// The vLLM API monitor (vllm_gaudi/entrypoints/api_monitor): the Strata monitor page, reading this server's
// /api/monitor/status and /api/requests every 2 seconds.
const $ = id => document.getElementById(id);
let selected = location.hash.slice(1) || null, detail = null, pane = "input", records = [], stamp = "";
const store = {get(k, d) { try { return localStorage.getItem(k) ?? d; } catch (_) { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (_) {} }};
let apiKey = ""; try { apiKey = sessionStorage.getItem("vllm.monitor.key") || ""; } catch (_) {}
document.documentElement.dataset.theme = store.get("vllm.monitor.theme", "dark");
const num = n => typeof n === "number" && isFinite(n);
const seconds = n => num(n) ? (n < 10 ? `${n.toFixed(2)} s` : `${n.toFixed(1)} s`) : "—";
const speed = n => num(n) ? n.toFixed(1) : "—";
const count = n => num(n) ? n.toLocaleString() : "—";
const pct = n => num(n) ? `${(n * 100).toFixed(n >= 0.995 || n < 0.1 ? 1 : 0)}%` : "—";
const active = r => !["completed", "error", "disconnected"].includes(r.state);
function text(id, value) { const el = $(id); if (el.textContent !== String(value)) el.textContent = value; }
function html(id, value) { const el = $(id); if (el.innerHTML !== value) el.innerHTML = value; }
function meter(id, share, warnAt, hotAt) {
  const el = $(id); const v = num(share) ? Math.max(0, Math.min(1, share)) : 0;
  el.firstElementChild.style.width = `${(v * 100).toFixed(1)}%`;
  el.className = `meter${v >= hotAt ? " hot" : v >= warnAt ? " warn" : ""}`;
}
async function api(path) {
  const response = await fetch(path, {cache: "no-store", headers: apiKey ? {Authorization: `Bearer ${apiKey}`} : {}});
  let value = null; try { value = await response.json(); } catch (_) {}
  if (!response.ok) {
    if (response.status === 401) $("auth").hidden = false;
    throw new Error(value?.error?.message || `HTTP ${response.status}`);
  }
  return value;
}
function notice(message) { $("error").hidden = !message; text("error", message || ""); }
function list() {
  const query = $("search").value.toLowerCase(), filter = $("filter").value;
  const visible = records.filter(r => `${r.id} ${r.path} ${r.model || ""}`.toLowerCase().includes(query) &&
    (filter === "all" || (filter === "active" ? active(r) : r.state === filter)));
  const nextStamp = JSON.stringify([visible, selected]);
  if (stamp === nextStamp) return;
  stamp = nextStamp;
  const focusId = document.activeElement?.dataset.request;
  $("requests").replaceChildren();
  if (!visible.length) {
    const empty = document.createElement("div"); empty.className = "empty";
    empty.textContent = records.length ? "No matching requests." : "No requests yet. API calls will appear here automatically.";
    $("requests").append(empty);
  }
  for (const r of visible) {
    const button = document.createElement("button"); button.className = `request${selected === r.id ? " selected" : ""}`;
    button.dataset.request = r.id; button.setAttribute("aria-label", `Request ${r.id}`); button.setAttribute("aria-pressed", String(selected === r.id));
    const row = document.createElement("div"); row.className = "row";
    const time = document.createElement("span"); time.textContent = new Date(r.started_at * 1000).toLocaleTimeString();
    const tag = document.createElement("span"); tag.className = `tag ${r.state}`; tag.textContent = r.state;
    row.append(time, tag);
    const endpoint = document.createElement("div"); endpoint.className = "endpoint"; endpoint.textContent = `POST ${r.path}`;
    const meta = document.createElement("div"); meta.className = "small";
    const tokens = num(r.prompt_tokens) ? ` · ${count(r.prompt_tokens)} → ${count(r.output_tokens)} tokens` : "";
    meta.textContent = `${r.id} · ${seconds(r.wallclock_s)} · ${r.stream ? "stream" : "JSON response"}${tokens}${r.http_status ? ` · HTTP ${r.http_status}` : ""}`;
    button.append(row, endpoint, meta);
    button.onclick = async () => { selected = r.id; detail = null; history.replaceState(null, "", `#${r.id}`); list(); await showDetail(); };
    $("requests").append(button);
    if (focusId === r.id) button.focus({preventScroll: true});
  }
}
function content() {
  if (!detail) return;
  let value = detail[pane] || "No content.";
  if (pane === "output") { try { value = JSON.stringify(JSON.parse(value), null, 2); } catch (_) {} }
  text("content", value);
  const note = pane === "input" ? "Request body as the client sent it; inline images are shown by size only."
    : pane === "output" ? (detail.error ? "Model output received before the error." : "Model answer. JSON is formatted here for readability.")
    : pane === "reasoning" ? "Separate reasoning content, when the reasoning parser returns it."
    : detail.stream ? "The event stream as the client received it." : "API response body.";
  text("content-note", note + (detail[`${pane}_truncated`] ? " Monitor capture truncated at 256K characters; the API response was not shortened." : ""));
}
async function showDetail() {
  if (!selected) return;
  const id = selected;
  const value = await api(`/api/requests?id=${encodeURIComponent(id)}`);
  if (selected !== id) return;
  detail = value; $("selection").hidden = false; $("no-selection").hidden = true;
  const e = value.engine || {};
  text("detail-title", `${value.id} · ${value.path}`);
  text("detail-status", value.state); $("detail-status").className = `tag ${value.state}`;
  text("detail-time", `${new Date(value.started_at * 1000).toLocaleString()}${value.model ? ` · ${value.model}` : ""}`);
  text("detail-wall", seconds(value.wallclock_s));
  text("detail-queue", seconds(e.queue_s)); text("detail-prefill", seconds(e.prefill_s)); text("detail-first", seconds(value.first_token_s ?? e.ttft_s));
  text("detail-prompt", count(value.prompt_tokens)); text("detail-tokens", count(value.output_tokens));
  text("detail-cached", num(value.cached_tokens) ? `${count(value.cached_tokens)}${value.prompt_tokens ? ` · ${pct(value.cached_tokens / value.prompt_tokens)}` : ""}` : "—");
  text("detail-speed", num(value.decode_tok_s) ? `${speed(value.decode_tok_s)} tok/s` : "—");
  text("detail-accept", num(e.spec?.mean_acceptance_length) ? e.spec.mean_acceptance_length.toFixed(2) : "—");
  text("detail-rounds", count(e.spec?.rounds)); text("detail-finish", value.finish_reason || "—");
  text("detail-prefill-speed", num(value.prefill_tok_s) ? `${speed(value.prefill_tok_s)} tok/s` : "—");
  text("detail-note", value.overlapped ? "Ran alongside another request: queue, prefill, cache and acceptance figures are shared, so they are not shown."
    : active(value) ? "Engine figures appear when the request finishes." : value.stream ? "First token measured at the server, as the client's stream saw it." : "");
  $("detail-error").hidden = !value.error; text("detail-error", value.error?.message || ""); content();
}
function engineCards(s) {
  const e = s.engine, d = s.device;
  const spec = e?.spec;
  text("accept", num(spec?.mean_acceptance_length) ? spec.mean_acceptance_length.toFixed(2) : "—");
  text("accept-note", num(spec?.acceptance_rate)
    ? `Tokens per verify round · ${pct(spec.acceptance_rate)} of ${s.num_speculative_tokens || "the"} drafts kept`
    : "Tokens per verify round, since start");
  const bars = (spec?.per_position || []).map((v, i) => `<i style="height:${Math.max(4, v * 100).toFixed(0)}%" title="Draft ${i + 1}: accepted in ${pct(v)} of rounds"></i>`).join("");
  html("positions", bars);
  text("prefix", pct(e?.prefix_cache_hit_rate));
  text("prefix-note", num(e?.prefix_cache_queried_tokens) ? `Of ${count(e.prefix_cache_queried_tokens)} prompt tokens looked up` : "Prompt tokens found in the cache");
  text("kv", pct(e?.kv_cache_usage)); meter("kv-meter", e?.kv_cache_usage, 0.75, 0.92);
  text("kv-note", e ? `${e.running} running · ${e.waiting} waiting` : "Engine counters unavailable");
  if (d) {
    text("device-name", d.name ? `Gaudi2 · ${d.name}` : "Gaudi2");
    html("temp", `${num(d.temperature_c) ? d.temperature_c.toFixed(0) : "—"}<small>°C</small> ${num(d.power_w) ? d.power_w.toFixed(0) : "—"}<small>W</small>`);
    const used = d.memory_used_mib, total = d.memory_total_mib;
    text("device-note", `HBM ${num(used) ? (used / 1024).toFixed(1) : "—"} / ${num(total) ? (total / 1024).toFixed(0) : "—"} GiB · AIP ${num(d.utilization_pct) ? d.utilization_pct.toFixed(0) : "—"}%`);
    meter("hbm-meter", num(used) && total ? used / total : null, 0.97, 0.995);
  } else { text("temp", "—"); text("device-note", s.device_reader ? "Reading hl-smi…" : "hl-smi not available in this process"); }
}
async function refresh() {
  try {
    const [status, listing] = await Promise.all([api("/api/monitor/status"), api("/api/requests")]);
    records = listing.requests;
    if (selected && !records.some(r => r.id === selected)) {
      selected = detail = null; history.replaceState(null, "", location.pathname); $("selection").hidden = true; $("no-selection").hidden = false;
    }
    const running = status.engine ? status.engine.running + status.engine.waiting : 0;
    const inflight = Math.max(status.in_flight, running);
    const generating = records.some(r => r.state === "generating");
    // a streamed request shows its first token; a JSON one is only seen when it ends, so it is just "Running"
    const prefill = records.some(r => r.state === "running" && r.stream);
    text("state", inflight ? (generating ? "Generating" : prefill ? "Prefill" : "Running") : "Idle");
    $("engine-state").className = `value ${inflight ? "busy" : "loaded"}`;
    text("model", `${status.model || "—"}${status.max_model_len ? ` · ${count(status.max_model_len)}-token context` : ""}`);
    text("count", records.length);
    text("active", inflight ? `${inflight} active / queued` : "No active requests");
    const last = records.find(r => !active(r)), lastSpeed = records.find(r => num(r.decode_tok_s));
    const lastPrefill = records.find(r => num(r.prefill_tok_s));
    text("wall", last ? seconds(last.wallclock_s) : "—"); text("speed", speed(lastSpeed?.decode_tok_s));
    text("prefill-speed", speed(lastPrefill?.prefill_tok_s));
    text("prefill-note", lastPrefill ? `${count(lastPrefill.prompt_tokens - (lastPrefill.cached_tokens || 0))} new tokens · first token ${seconds(lastPrefill.first_token_s ?? lastPrefill.engine?.ttft_s)}` : "New prompt tokens / second");
    text("wall-note", last && num(last.output_tokens) ? `${count(last.prompt_tokens)} → ${count(last.output_tokens)} tokens` : "Includes queue and prefill");
    engineCards(status);
    text("retention", `Last ${listing.retention} requests kept in memory until restart. Inputs and outputs stay on this machine.`);
    list();
    const current = records.find(r => r.id === selected);
    if (selected && (!detail || active(detail) || (current && current.state !== detail.state))) await showDetail();
    text("updated", `Live · updated ${new Date().toLocaleTimeString()}`);
    notice("");
  } catch (error) { notice(error.message); text("updated", "Disconnected · retrying"); $("engine-state").className = "value down"; text("state", "Unreachable"); }
}
$("key").onclick = () => { $("auth").hidden = !$("auth").hidden; if (!$("auth").hidden) $("api-key").focus(); };
$("auth").onsubmit = async event => { event.preventDefault(); apiKey = $("api-key").value.trim(); try { sessionStorage.setItem("vllm.monitor.key", apiKey); } catch (_) {} $("auth").hidden = true; await refresh(); };
$("theme").onclick = () => { const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; document.documentElement.dataset.theme = theme; store.set("vllm.monitor.theme", theme); };
$("search").oninput = list; $("filter").onchange = list;
document.querySelectorAll("[data-pane]").forEach(button => { button.onclick = () => { pane = button.dataset.pane; document.querySelectorAll("[data-pane]").forEach(b => b.setAttribute("aria-selected", String(b === button))); content(); }; });
$("copy").onclick = async () => { try { await navigator.clipboard.writeText($("content").textContent); text("copy", "Copied"); setTimeout(() => text("copy", "Copy"), 1500); } catch (_) { notice("Clipboard unavailable; select the text to copy it."); } };
async function poll() { await refresh(); setTimeout(poll, 2000); }
poll();
