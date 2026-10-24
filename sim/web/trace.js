/* ============================================================================
   TRACE · observability viewer — sim/web/trace.js
   Vanilla JS, no framework, no build step, no external request.

   Panes: ① timeline (virtualised event feed) ② runs (waterfall + span tree)
          ③ lineage (hand-rolled SVG PROV graph) ④ database browser ⑤ stats.

   Data contract: docs/observability-design.md §9.2. The same shapes are served
   by the inline fixture in demo mode (?demo=1, or any file:// load), so the
   page can be reviewed and screenshotted with no backend at all.
   ========================================================================== */
(function () {
"use strict";

/* ── vocabularies (design doc §4.4 / §4.8 / §4.9, closed enums) ─────────── */
var LEVELS = ["DEBUG", "DEFAULT", "WARNING", "ERROR"];
var SPAN_KINDS = ["AGENT","LLM","RETRIEVER","TOOL","ACTION","PERCEPTION","LOOP","FEEDBACK","EVALUATOR","MAINT"];
var EVENT_TYPES = [
  "run.started","run.ended","command.received","retrieval.performed","llm.call","plan.produced",
  "plan.repaired","step.started","step.finished","action.executed","scene.reset","kb.write",
  "memory.write","file.write","feedback.received","perception.run","loop.closed","eval.scored",
  "audit.line","console.log","trace.warning"
];
var RELS = ["wasGeneratedBy","used","wasRevisionOf","wasDerivedFrom","wasAttributedTo",
            "wasAssociatedWith","wasInformedBy","wasInvalidatedBy"];
var VERDICT_SIGN = { insert: "+", reinforce: "~", update: "*", delete: "−", noop: "·" };
var ROW_H = 28;                 /* must equal --row-h in trace.css */
var MAX_ROWS = 2000;            /* client window cap (§11.3) */
var TAIL_MS = 1200;

/* ── tiny DOM helpers ──────────────────────────────────────────────────── */
function h(tag, attrs) {
  var e = document.createElement(tag), k, v, i, kid;
  if (attrs) for (k in attrs) {
    v = attrs[k];
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k === "dataset") { for (var d in v) if (v[d] != null) e.dataset[d] = v[d]; }
    else if (k.slice(0, 2) === "on") e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (i = 2; i < arguments.length; i++) {
    kid = arguments[i];
    if (kid === null || kid === undefined || kid === false) continue;
    if (Array.isArray(kid)) { kid.forEach(function (c) { if (c != null && c !== false) e.appendChild(typeof c === "object" ? c : document.createTextNode(String(c))); }); }
    else e.appendChild(typeof kid === "object" ? kid : document.createTextNode(String(kid)));
  }
  return e;
}
function svgEl(tag, attrs) {
  var e = document.createElementNS("http://www.w3.org/2000/svg", tag), k;
  if (attrs) for (k in attrs) { if (attrs[k] === null || attrs[k] === undefined) continue; e.setAttribute(k, attrs[k]); }
  for (var i = 2; i < arguments.length; i++) { var kid = arguments[i]; if (kid) e.appendChild(kid); }
  return e;
}
function $(id) { return document.getElementById(id); }
function clear(el) { while (el && el.firstChild) el.removeChild(el.firstChild); return el; }
function on(el, ev, fn) { if (el) el.addEventListener(ev, fn); }
function debounce(fn, ms) { var t; return function () { var a = arguments, s = this; clearTimeout(t); t = setTimeout(function () { fn.apply(s, a); }, ms); }; }

/* ── formatting ────────────────────────────────────────────────────────── */
function pad2(n) { return (n < 10 ? "0" : "") + n; }
function clock(ts) { if (!ts) return "—"; var d = new Date(ts * 1000); return pad2(d.getHours()) + ":" + pad2(d.getMinutes()) + ":" + pad2(d.getSeconds()); }
function isoTs(ts) {
  if (!ts) return "—";
  var d = new Date(ts * 1000);
  return d.getFullYear() + "-" + pad2(d.getMonth() + 1) + "-" + pad2(d.getDate()) + " " +
         pad2(d.getHours()) + ":" + pad2(d.getMinutes()) + ":" + pad2(d.getSeconds());
}
function dur(ms) {
  if (ms === null || ms === undefined) return "—";
  if (ms < 1000) return Math.round(ms) + " ms";
  if (ms < 60000) return (ms / 1000).toFixed(1) + " s";
  return Math.floor(ms / 60000) + "m " + Math.round((ms % 60000) / 1000) + "s";
}
function clip(s, n) { s = s === null || s === undefined ? "" : String(s); return s.length > n ? s.slice(0, n - 1) + "…" : s; }
function bytes(n) {
  if (n == null) return "—";
  if (n < 1024) return n + " B";
  if (n < 1048576) return (n / 1024).toFixed(1) + " KB";
  return (n / 1048576).toFixed(1) + " MB";
}
function short(id, n) { id = String(id || ""); return id.length > (n || 10) ? id.slice(0, n || 10) + "…" : id; }
function kindVar(kind) { return SPAN_KINDS.indexOf(kind) >= 0 ? "var(--k-" + kind + ")" : "var(--n-500)"; }

/* ── app state ─────────────────────────────────────────────────────────── */
var S = {
  demo: false,
  view: "timeline",
  route: { view: "timeline", arg: null, arg2: null, params: {} },
  drawerOpen: false,
  lastFocus: null,
  live: false,
  liveTimer: null,
  health: null
};

/* ── data endpoints ────────────────────────────────────────────────────── */
function qstr(params) {
  var parts = [], k;
  for (k in params) {
    var v = params[k];
    if (v === null || v === undefined || v === "") continue;
    parts.push(encodeURIComponent(k) + "=" + encodeURIComponent(v));
  }
  return parts.length ? "?" + parts.join("&") : "";
}
/* Every endpoint answers with an {"error": ...} envelope rather than a status
   code (house contract), so a rejected fetch is normalised into one too. */
function api(name, params) {
  if (S.demo) {
    try { return Promise.resolve(DEMO.endpoint(name, params || {})); }
    catch (err) { return Promise.resolve({ error: "demo fixture: " + err.message }); }
  }
  return fetch("/" + name + qstr(params || {}), { headers: { Accept: "application/json" } })
    .then(function (r) {
      if (!r.ok) return { error: "HTTP " + r.status + " from /" + name };
      return r.json().catch(function () { return { error: "malformed JSON from /" + name }; });
    })
    .catch(function (e) {
      return { error: "cannot reach /" + name + " (" + e.message + ") — is sim/webapp.py running?" };
    });
}

/* A missing endpoint is the expected state on a webapp build without the trace
   routes — say so, and point at the fixture, instead of showing a bare 404. */
function hint(err) {
  var s = String(err || "");
  if (S.demo) return s;
  if (/404|cannot reach|not available/i.test(s))
    return s + " — the /trace_* routes are not served by this build. Append ?demo=1 to the URL to " +
      "preview the viewer against inline fixture data.";
  return s;
}

/* ── loading / empty / error surfaces (§11.4: never a dead spinner) ────── */
function stateBox(kind, title, msg, action) {
  var box = h("div", { class: "state " + kind });
  box.appendChild(kind === "loading" ? h("div", { class: "state-icon" })
    : h("div", { class: "state-icon", "aria-hidden": "true" },
        kind === "error" ? "⚠" : kind === "empty" ? "◌" : "✦"));
  box.appendChild(h("div", { class: "state-title", text: title }));
  if (msg) box.appendChild(h("p", { class: "state-msg", text: msg }));
  if (action) box.appendChild(action);
  return box;
}
function showState(el, kind, title, msg, action) {
  clear(el).appendChild(stateBox(kind, title, msg, action));
  return el;
}
/* Uniform handling of the error envelope + the "nothing here yet" case. */
function guard(el, res, emptyTest, emptyTitle, emptyMsg) {
  if (!res || res.error) {
    showState(el, "error", "could not load", hint((res && res.error) || "no response"),
      h("button", { class: "btn", onclick: renderRoute }, "retry"));
    return false;
  }
  if (emptyTest && emptyTest(res)) {
    showState(el, "empty", emptyTitle || "nothing recorded yet", emptyMsg || "");
    return false;
  }
  return true;
}

/* ── shape normalisers ────────────────────────────────────────────────────
   The server returns writes/causes as FLAT sqlite rows (before_json,
   run_command, span_name, …). The design sketch nests them. Accept both, so
   the viewer keeps working whichever side is refactored first. */
function jparse(v) {
  if (v === null || v === undefined) return null;
  if (typeof v !== "string") return v;
  try { return JSON.parse(v); } catch (e) { return v; }
}
function normWrite(w) {
  if (!w) return w;
  var o = {}, k;
  for (k in w) o[k] = w[k];
  if (o.before === undefined) o.before = jparse(o.before_json);
  if (o.after === undefined) o.after = jparse(o.after_json);
  if (o.write_id === undefined) o.write_id = o.id;
  if (!o.run && (o.run_command || o.run_scene || o.run_arm || o.run_kind))
    o.run = { run_id: o.run_id, command: o.run_command, scene: o.run_scene, arm: o.run_arm,
      kind: o.run_kind, decision_id: o.decision_id, started_ts: o.run_started_ts };
  if (!o.span && (o.span_name || o.span_kind))
    o.span = { span_id: o.span_id, name: o.span_name, kind: o.span_kind, op: o.span_op };
  return o;
}
function normUse(u) {
  var o = {}, k;
  for (k in u) o[k] = u[k];
  if (!o.span_id && typeof o.node === "string" && o.node.indexOf("span:") === 0) o.span_id = o.node.slice(5);
  o.name = o.name || o.span_name || o.node || o.span_id;
  o.kind = o.kind || o.span_kind || o.span_op || "span";
  return o;
}

/* ── chips ─────────────────────────────────────────────────────────────── */
function chip(text, cls, title) { return h("span", { class: "chip " + (cls || ""), title: title || null, text: text }); }
function levelDot(level) { return h("span", { class: "lvl-dot", dataset: { l: level || "DEFAULT" }, title: level || "DEFAULT" }); }
function statusChip(status) {
  var cls = status === "OK" ? "chip-ok" : status === "ERROR" ? "chip-err" : "chip-live";
  return chip(status || "UNSET", cls);
}
function writeChip(w, onClick) {
  var kind = w.table_or_kind || w.kind || "?";
  var label = (VERDICT_SIGN[w.verdict] || "") + kind + " #" + clip(w.pointer, 22);
  var b = h("button", { class: "w-chip", type: "button", dataset: { v: w.verdict },
    title: (w.verdict || "") + " " + (w.artifact_id || "") + (w.why ? "\n" + w.why : "") },
    h("span", { class: "sig", text: label }));
  if (w.summary) b.appendChild(h("span", { class: "muted", dir: "auto", text: clip(w.summary, 26) }));
  on(b, "click", function (ev) { ev.stopPropagation(); (onClick || defaultWriteClick)(w); });
  return b;
}
function defaultWriteClick(w) { if (w.artifact_id) openEntity("art:" + w.artifact_id); }

/* ── routing (deep links, back/forward) ────────────────────────────────── */
function parseHash() {
  var raw = location.hash.replace(/^#\/?/, "");
  var qi = raw.indexOf("?"), params = {}, path = raw;
  if (qi >= 0) {
    path = raw.slice(0, qi);
    raw.slice(qi + 1).split("&").forEach(function (kv) {
      if (!kv) return;
      var i = kv.indexOf("="), k = i < 0 ? kv : kv.slice(0, i);
      params[decodeURIComponent(k)] = i < 0 ? "" : decodeURIComponent(kv.slice(i + 1).replace(/\+/g, " "));
    });
  }
  var seg = path.split("/").filter(function (x) { return x !== ""; }).map(decodeURIComponent);
  var view = seg[0] || "timeline";
  if (["timeline", "runs", "lineage", "db", "stats"].indexOf(view) < 0) view = "timeline";
  return { view: view, arg: seg[1] || null, arg2: seg.slice(2).join("/") || null, params: params };
}
function nav(hash, replace) {
  if (replace) history.replaceState(null, "", hash);
  else location.hash = hash;
  if (replace) renderRoute();
}
function routeHash(view, arg, arg2, params) {
  var s = "#/" + view;
  if (arg) s += "/" + encodeURIComponent(arg);
  if (arg2) s += "/" + encodeURIComponent(arg2);
  var q = qstr(params || {});
  return s + (q ? q.replace("?", "?") : "");
}

function setView(view) {
  S.view = view;
  ["timeline", "runs", "lineage", "db", "stats"].forEach(function (v) {
    var panel = $("view-" + v), tab = $("tab-" + v);
    if (panel) panel.hidden = v !== view;
    if (tab) tab.setAttribute("aria-selected", v === view ? "true" : "false");
  });
}

function renderRoute() {
  var r = parseHash();
  S.route = r;
  setView(r.view);
  if (r.view === "timeline") renderTimeline(r);
  else if (r.view === "runs") renderRuns(r);
  else if (r.view === "lineage") renderLineage(r);
  else if (r.view === "db") renderDb(r);
  else if (r.view === "stats") renderStats(r);
  syncLiveTimer();
}

/* ── shared detail drawer ──────────────────────────────────────────────── */
function openDrawer(title, build) {
  var d = $("drawer"), body = $("drawer-body");
  if (!S.drawerOpen) S.lastFocus = document.activeElement;
  $("drawer-title").textContent = title;
  clear(body);
  d.hidden = false;
  S.drawerOpen = true;
  build(body);
  $("drawer-close").focus();
  if (S.view === "lineage") setTimeout(fitZoom, 0);   /* the surface just narrowed */
}
function closeDrawer() {
  var d = $("drawer");
  if (!d || d.hidden) return;
  d.hidden = true;
  S.drawerOpen = false;
  if (S.view === "lineage") setTimeout(fitZoom, 0);
  if (S.lastFocus && document.contains(S.lastFocus)) S.lastFocus.focus();
}
function sec(title, node) {
  var s = h("div", { class: "sec" }, h("h3", { text: title }));
  if (node) s.appendChild(node);
  return s;
}
function kv(pairs) {
  var dl = h("dl", { class: "kv" });
  pairs.forEach(function (p) {
    if (p[1] === null || p[1] === undefined || p[1] === "") return;
    dl.appendChild(h("dt", { text: p[0] }));
    var dd = h("dd", { dir: "auto" });
    if (typeof p[1] === "object" && p[1].nodeType) dd.appendChild(p[1]);
    else { dd.textContent = String(p[1]); if (p[2]) dd.className = "wrap"; }
    dl.appendChild(dd);
  });
  return dl;
}
function pre(obj) {
  var txt = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
  return h("pre", { class: "pre", dir: "auto", text: txt });
}
function linkBtn(label, fn, title) {
  return h("button", { class: "lnk", type: "button", title: title || null, onclick: fn }, label);
}

/* ══════════════════════════════════════════════════════════════════════════
   DEMO FIXTURE — the exact shapes of docs/observability-design.md §9.2,
   generated in the browser so ?demo=1 (and any file:// review) renders the
   whole page with no backend. Every row here is synthetic.
   ══════════════════════════════════════════════════════════════════════════ */
var DEMO = {
  built: false, runs: [], spans: [], events: [], writes: [], artifacts: [], edges: [],
  ART: {}, TBL: {}, seq: 0, build: demoBuild, endpoint: demoEndpoint
};
var ART_KIND = { episodes: "episode", spatial: "spatial", failures: "failure", nav_cells: "nav_cell",
  perception_labels: "perception_label", procedure: "procedure", lesson: "lesson", fact: "fact",
  "loop_reports.jsonl": "report" };
var SRC_ACTOR = { human: "human:arash", demo: "human:arash", environment: "env:ai2thor", prior: "system:seed" };

function demoBuild() {
  if (DEMO.built) return;
  DEMO.built = true;
  var runs = DEMO.runs, spans = DEMO.spans, events = DEMO.events, writes = DEMO.writes,
      arts = DEMO.artifacts, edges = DEMO.edges;
  var wid = 0, spanSeq = {}, rs = 0x2f6e2b13;
  var T = Math.floor(Date.now() / 1000) - 3300;
  DEMO.TBL = { "kb.procedures": [], "kb.lessons": [], "kb.facts": [], "kb.audit": [],
    "mem.episodes": [], "mem.spatial": [], "mem.failures": [], "mem.nav_cells": [],
    "mem.perception_labels": [], "mem.bandit_arms": [], "mem.bandit_decisions": [],
    "file.loop_reports": [] };

  function rnd() { rs = (rs + 0x6D2B79F5) | 0; var t = Math.imul(rs ^ (rs >>> 15), 1 | rs);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }
  function hex(n) { var s = ""; while (s.length < n) s += Math.floor(rnd() * 16).toString(16); return s; }

  function ev(o) {
    DEMO.seq++;
    events.push({ seq: DEMO.seq, event_id: hex(32), ts: o.ts, run_id: o.run_id || null,
      span_id: o.span_id || null, type: o.type, actor: o.actor || null, level: o.level || "DEFAULT",
      scene: o.scene || null, summary: o.summary || "", payload: o.payload || {},
      origin: o.origin || "webapp", pid: 9120, thread: o.thread || "plan" });
    return DEMO.seq;
  }
  function edge(src, dst, rel, note, ts) {
    for (var i = 0; i < edges.length; i++)
      if (edges[i].src === src && edges[i].dst === dst && edges[i].rel === rel) return;
    edges.push({ id: edges.length + 1, src: src, dst: dst, rel: rel, note: note || null,
      ts: ts || null, seq: DEMO.seq });
  }

  function mkRun(o) {
    var r = { run_id: o.id || hex(32), parent_run_id: o.parent || null, kind: o.kind || "command",
      origin: o.origin || "webapp", started_ts: o.t, ended_ts: o.end === undefined ? null : o.end,
      status: o.status || "UNSET", scene: o.scene || null, level: o.level || null, lang: o.lang || null,
      command: o.cmd || null, arm: o.arm || null, decision_id: o.decision_id == null ? null : o.decision_id,
      agent_id: o.model || "gemma3:4b", seed: 7, code_version: "0d28354", error: o.error || null,
      config: { model: o.model || "gemma3:4b", endpoint: "http://172.20.144.1:11434", provider: "ollama",
        rag_enabled: true, loop_enabled: o.loop !== false, bandit_enabled: true, env_feedback_enabled: true,
        perception_mode: "oracle", auto_retry: false, arm: o.arm || null, scene: o.scene || null,
        level: o.level || null, lang: o.lang || null, attempt: o.attempt || 0, resolution: "600x600",
        schema_version: 1 },
      config_hash: o.hash || "9f2c1b77aa30de41", n_spans: 0, n_events: 0, n_writes: 0 };
    runs.push(r);
    spanSeq[r.run_id] = 0;
    ev({ type: "run.started", ts: o.t, run_id: r.run_id, actor: "system:webapp", scene: r.scene,
      origin: r.origin, summary: r.kind + " run started" + (r.command ? " — " + r.command : ""),
      payload: { config: r.config, origin: r.origin } });
    if (r.command) ev({ type: "command.received", ts: o.t + 0.04, run_id: r.run_id, actor: "human:arash",
      scene: r.scene, origin: r.origin, summary: "command received: " + r.command,
      payload: { command: r.command, level: r.level, lang: r.lang, scene: r.scene } });
    return r;
  }
  function endRun(r) {
    if (r.ended_ts == null) return;
    ev({ type: "run.ended", ts: r.ended_ts, run_id: r.run_id, actor: "system:webapp", scene: r.scene,
      origin: r.origin, level: r.status === "ERROR" ? "ERROR" : "DEFAULT",
      summary: "run ended " + r.status + " in " + dur((r.ended_ts - r.started_ts) * 1000),
      payload: { status: r.status, error: r.error } });
  }
  function mkSpan(o) {
    var id = hex(16), depth = 0, i;
    if (o.parent) for (i = spans.length - 1; i >= 0; i--)
      if (spans[i].span_id === o.parent) { depth = spans[i].depth + 1; break; }
    var sp = { span_id: id, run_id: o.run.run_id, parent_span_id: o.parent || null,
      seq: spanSeq[o.run.run_id]++, depth: depth, name: o.name, op: o.op, kind: o.kind,
      step_ix: o.step == null ? null : o.step, start_ts: o.t, end_ts: o.t + o.ms / 1000,
      duration_ms: o.ms, status: o.status || "OK", level: o.level || "DEFAULT", error: o.error || null,
      scene: o.run.scene, actor: o.actor || "system:webapp", repeat: o.repeat || 1,
      input_json: o.input ? JSON.stringify(o.input) : null,
      output_json: o.output ? JSON.stringify(o.output) : null,
      attrs_json: o.attrs ? JSON.stringify(o.attrs) : null };
    spans.push(sp);
    edge("span:" + id, o.parent ? "span:" + o.parent : "run:" + o.run.run_id, "wasInformedBy", null, o.t);
    edge("span:" + id, "agent:" + sp.actor, "wasAssociatedWith", null, o.t);
    o.run.n_spans++;
    return sp;
  }
  function used(span, aid, note, ts) { edge("span:" + span.span_id, "art:" + aid, "used", note, ts); }

  function wr(o) {
    var kind = ART_KIND[o.table] || o.table;
    var aid = o.store + ":" + kind + ":" + String(o.pointer).replace(/\//g, "%2F");
    var type = o.store === "kb" ? "kb.write" : o.store === "mem" ? "memory.write" : "file.write";
    var op = o.verdict === "insert" ? "ADD" : o.verdict === "reinforce" ? "UPSERT"
           : o.verdict === "delete" ? "PRUNE" : "UPDATE";
    var audit = o.audit || (isoTs(o.ts) + "  [" + op + " " + kind + "]  <" + (o.source || "environment") +
      ">  " + clip(o.summary || String(o.pointer), 70) + "   WHY: " + (o.why || ""));
    var s = ev({ type: type, ts: o.ts, run_id: o.run ? o.run.run_id : null,
      span_id: o.span ? o.span.span_id : null, scene: o.run ? o.run.scene : null,
      actor: o.actor || (o.store === "kb" ? "system:improve" : "system:memory"),
      level: o.store === "file" ? "DEBUG" : "DEFAULT", origin: o.run ? o.run.origin : "webapp",
      summary: o.verdict + " " + kind + " #" + o.pointer + (o.summary ? " — " + clip(o.summary, 58) : ""),
      payload: { kind: kind, id: o.pointer, why: o.why, dedup_key: o.dedup || null,
        before: o.before || null, after: o.after || null, audit_line: audit } });
    var a = DEMO.ART[aid];
    if (!a) {
      a = { artifact_id: aid, store: o.store, kind: kind, ref: String(o.pointer), created_seq: s,
        created_ts: o.ts, updated_ts: o.ts, content_hash: hex(16), summary: o.summary || null,
        source: o.source || "environment", scope: o.scope || "global",
        confidence: o.confidence == null ? null : o.confidence, revision: 1, occurrences: 1,
        dedup_key: o.dedup || null, alive: 1, n_causes: 0 };
      arts.push(a); DEMO.ART[aid] = a;
      edge("art:" + aid, "agent:" + (SRC_ACTOR[a.source] || "env:ai2thor"), "wasAttributedTo", null, o.ts);
    } else {
      a.updated_ts = o.ts;
      if (o.verdict === "reinforce") { a.revision++; a.occurrences++; }
      else if (o.verdict === "update") a.revision++;
      else if (o.verdict === "delete") a.alive = 0;
      if (o.confidence != null) a.confidence = o.confidence;
      if (o.summary) a.summary = o.summary;
    }
    a.n_causes++;
    /* keep the live store row consistent with the ledger: a reinforce bumps the
       entry's own counters, exactly as knowledge.upsert_* does on disk */
    if (o.verdict !== "insert" && o.after && KIND_TABLE[kind] && DEMO.TBL[KIND_TABLE[kind]]) {
      var pkCol = (catEntry(KIND_TABLE[kind]) || {}).pk;
      DEMO.TBL[KIND_TABLE[kind]].forEach(function (row) {
        if (pkCol && String(row[pkCol]) === String(o.pointer))
          for (var f in o.after) if (row[f] !== undefined) row[f] = o.after[f];
      });
    }
    if (o.span) edge("art:" + aid, "span:" + o.span.span_id,
      o.verdict === "insert" ? "wasGeneratedBy" : o.verdict === "delete" ? "wasInvalidatedBy" : "wasRevisionOf",
      o.why || null, o.ts);
    writes.push({ id: ++wid, event_seq: s, run_id: o.run ? o.run.run_id : null,
      span_id: o.span ? o.span.span_id : null, step_ix: o.step == null ? null : o.step, ts: o.ts,
      store: o.store, table_or_kind: o.table, pointer: String(o.pointer), artifact_id: aid,
      verdict: o.verdict, dedup_key: o.dedup || null,
      before_json: o.before ? JSON.stringify(o.before) : null,
      after_json: o.after ? JSON.stringify(o.after) : null, why: o.why || null,
      source: o.source || "environment", audit_line: audit, summary: o.summary || null,
      origin: o.run ? o.run.origin : "webapp" });
    if (o.run) o.run.n_writes++;
    if (o.store === "kb") DEMO.TBL["kb.audit"].push({ line_no: DEMO.TBL["kb.audit"].length + 1,
      time: isoTs(o.ts), op: op, kind: kind, source: o.source || "environment", line: audit });
    if (o.row && o.tbl) DEMO.TBL[o.tbl].push(o.row);
    return a;
  }

  /* one planning+execution run: retrieval → chat → steps(+actions) → close_loop */
  function commandRun(o) {
    var r = mkRun(o), t = o.t, i, j;
    var root = mkSpan({ run: r, name: "invoke_agent " + clip(o.cmd, 44), op: "invoke_agent",
      kind: "AGENT", t: t, ms: (o.end - o.t) * 1000, status: o.status, actor: "system:webapp",
      attrs: { "agent.level": o.level, "agent.lang": o.lang, "agent.arm": o.arm,
               "agent.decision_id": o.decision_id, "gen_ai.agent.id": o.model || "gemma3:4b" } });
    var rt = mkSpan({ run: r, parent: root.span_id, name: "retrieval hybrid k=5", op: "retrieval",
      kind: "RETRIEVER", t: t + 0.2, ms: 440, actor: "system:rag",
      attrs: { "gen_ai.retrieval.documents": (o.hits || []).length, "agent.rag_enabled": true } });
    ev({ type: "retrieval.performed", ts: t + 0.6, run_id: r.run_id, span_id: rt.span_id,
      actor: "system:rag", scene: r.scene, origin: r.origin,
      summary: "retrieved " + (o.hits || []).length + " memory hits for: " + clip(o.cmd, 40),
      payload: { hits: o.hits || [], strategy: "hybrid", k: 5 } });
    (o.hits || []).forEach(function (aid) { used(rt, aid, "injected into planner prompt", t + 0.6); });
    var chat = mkSpan({ run: r, parent: root.span_id, name: "chat " + (o.model || "gemma3:4b"),
      op: "chat", kind: "LLM", t: t + 0.7, ms: o.llm_ms || 8700, actor: "llm:" + (o.model || "gemma3:4b"),
      input: { prompt_chars: 2480, hits: (o.hits || []).length },
      output: { plan: (o.steps || []).map(function (s) { return s.name; }) },
      attrs: { "gen_ai.operation.name": "chat", "gen_ai.provider.name": "ollama",
        "gen_ai.request.model": o.model || "gemma3:4b", "gen_ai.usage.input_chars": 2480,
        "gen_ai.usage.output_chars": 316 } });
    ev({ type: "llm.call", ts: t + 0.7, run_id: r.run_id, span_id: chat.span_id,
      actor: "llm:" + (o.model || "gemma3:4b"), scene: r.scene, origin: r.origin,
      summary: "planning call " + (o.model || "gemma3:4b") + " · " + dur(o.llm_ms || 8700),
      payload: { prompt: "You are an embodied kitchen agent…\nCOMMAND: " + o.cmd, chars_in: 2480, chars_out: 316 } });
    ev({ type: "plan.produced", ts: t + 0.7 + (o.llm_ms || 8700) / 1000, run_id: r.run_id,
      span_id: chat.span_id, actor: "llm:" + (o.model || "gemma3:4b"), scene: r.scene, origin: r.origin,
      summary: "plan: " + (o.steps || []).map(function (s) { return s.name; }).join(" → "),
      payload: { plan: (o.steps || []).map(function (s) { return { action: s.name, args: s.args || {} }; }) } });
    var cursor = t + 1.0 + (o.llm_ms || 8700) / 1000, stepSpans = [];
    (o.steps || []).forEach(function (st, ix) {
      var sp = mkSpan({ run: r, parent: root.span_id, name: "execute_tool " + st.name, op: "execute_tool",
        kind: "TOOL", step: ix, t: cursor, ms: st.ms, status: st.ok === false ? "ERROR" : "OK",
        error: st.error || null, actor: "env:ai2thor", input: st.args || {},
        output: { success: st.ok !== false, error: st.error || null },
        attrs: { "gen_ai.tool.name": st.name, "thor.scene": r.scene,
                 "thor.object_type": (st.args && st.args.object) || null, "thor.success": st.ok !== false } });
      ev({ type: "step.started", ts: cursor, run_id: r.run_id, span_id: sp.span_id, actor: "env:ai2thor",
        scene: r.scene, origin: r.origin, summary: "step " + ix + " ▸ " + st.name +
          (st.args && st.args.object ? " " + st.args.object : ""), payload: { step: ix, args: st.args || {} } });
      var ac = cursor + 0.1;
      (st.actions || []).forEach(function (a) {
        var asp = mkSpan({ run: r, parent: sp.span_id, name: a.a + (a.obj ? " " + a.obj : ""),
          op: "thor_action", kind: "ACTION", t: ac, ms: a.ms || 180, level: "DEBUG",
          status: a.ok === false ? "ERROR" : "OK", error: a.err || null, actor: "env:ai2thor",
          repeat: a.repeat || 1,
          attrs: { "thor.action": a.a, "thor.object_id": a.obj ? a.obj + "|+01.2|+00.9|-00.4" : null,
                   "thor.success": a.ok !== false, "thor.error": a.err || null } });
        ev({ type: "action.executed", ts: ac, run_id: r.run_id, span_id: asp.span_id, level: "DEBUG",
          actor: "env:ai2thor", scene: r.scene, origin: r.origin,
          summary: a.a + (a.obj ? "(" + a.obj + ")" : "()") + " → " + (a.ok === false ? "FAIL: " + a.err : "ok") +
            (a.repeat > 1 ? "  ×" + a.repeat : ""),
          payload: { action: a.a, object: a.obj, success: a.ok !== false, error: a.err || null,
            repeat: a.repeat || 1, agent_pos: { x: 1.25, y: 0.9, z: -0.5 } } });
        ac += (a.ms || 180) / 1000 * (a.repeat || 1);
      });
      ev({ type: "step.finished", ts: cursor + st.ms / 1000, run_id: r.run_id, span_id: sp.span_id,
        actor: "env:ai2thor", scene: r.scene, origin: r.origin,
        level: st.ok === false ? "ERROR" : "DEFAULT",
        summary: "step " + ix + " " + (st.ok === false ? "FAILED — " + st.error : "ok") + " · " + dur(st.ms),
        payload: { step: ix, success: st.ok !== false, error: st.error || null } });
      stepSpans.push(sp);
      cursor += st.ms / 1000 + 0.2;
    });
    var loop = null;
    if (o.loop !== false) {
      loop = mkSpan({ run: r, parent: root.span_id, name: "close_loop", op: "close_loop", kind: "LOOP",
        t: o.end - 1.5, ms: 1200, actor: "system:improve",
        attrs: { "agent.loop_enabled": true, "kb.write.action": o.status === "OK" ? "upsert" : "lesson" } });
    }
    return { run: r, root: root, retrieval: rt, chat: chat, steps: stepSpans, loop: loop, end: o.end };
  }

  demoScenario({ T: T, hex: hex, ev: ev, edge: edge, used: used, wr: wr, mkRun: mkRun, endRun: endRun,
    mkSpan: mkSpan, commandRun: commandRun });

  /* denormalised counters the runs table carries */
  runs.forEach(function (r) {
    r.n_events = events.filter(function (e) { return e.run_id === r.run_id; }).length;
  });
}
/* The scripted history: two successful commands (one English, one Persian) that
   teach the KB, a failing L3 run that writes a lesson, a human teaching run, a
   Persian run that CONSUMES what the earlier runs wrote (the self-improving
   cycle, closed), an eval sweep, a maintenance prune, a live run, and a noisy
   exploration run so the timeline has realistic volume. */
function demoScenario(F) {
  var T = F.T, TBL = DEMO.TBL, i;

  function epi(o) {
    return F.wr({ run: o.run, span: o.span, step: o.step, ts: o.ts, store: "mem", table: "episodes",
      pointer: o.id, verdict: "insert", summary: o.action + (o.obj ? " " + o.obj : ""),
      why: "env feedback: step " + o.step + " logged (" + (o.ok ? "success" : "failure") + ")",
      source: "environment", scope: o.run.scene, confidence: o.ok ? 0.9 : 0.3,
      after: { success: !!o.ok, rating: null },
      tbl: "mem.episodes", row: { id: o.id, ts: o.ts, scene: o.run.scene, command: o.run.command,
        level: o.run.level, action: o.action, args: JSON.stringify(o.args || {}), success: o.ok ? 1 : 0,
        error: o.err || null, state_before: JSON.stringify({ holding: null, fridge: "closed" }),
        state_after: JSON.stringify({ holding: o.ok ? (o.obj || null) : null, fridge: "open" }),
        rating: null } });
  }

  /* ── R1 · English L2, first time: everything is learned from scratch ─── */
  var A = F.commandRun({ t: T, end: T + 41.3, cmd: "Put the apple in the fridge.", lang: "en", level: "L2",
    scene: "FloorPlan1", arm: "reuse_memory", decision_id: 417, status: "OK", hits: [], llm_ms: 8700,
    steps: [
      { name: "navigate_to", args: { object: "Apple" }, ms: 9200, actions: [
        { a: "RotateRight", ms: 160 }, { a: "MoveAhead", ms: 150, repeat: 6 },
        { a: "RotateLeft", ms: 160 }, { a: "MoveAhead", ms: 150, repeat: 3 }] },
      { name: "pick_and_place", args: { object: "Apple", to: "Fridge" }, ms: 18400, actions: [
        { a: "PickupObject", obj: "Apple", ms: 210 }, { a: "RotateRight", ms: 160 },
        { a: "MoveAhead", ms: 150, repeat: 4 }, { a: "OpenObject", obj: "Fridge", ms: 240 },
        { a: "PutObject", obj: "Fridge", ms: 230 }, { a: "CloseObject", obj: "Fridge", ms: 220 }] }
    ] });
  F.wr({ run: A.run, span: A.steps[0], step: 0, ts: T + 12.0, store: "mem", table: "spatial", pointer: 44,
    verdict: "insert", summary: "Apple @ FloorPlan1 (1.25, 0.90, -0.50)", source: "environment",
    scope: "FloorPlan1", confidence: 0.72, dedup: "FloorPlan1|Apple|counter",
    after: { observations: 1, confidence: 0.72 }, why: "observed while navigating to Apple",
    tbl: "mem.spatial", row: { id: 44, scene: "FloorPlan1", object_type: "Apple", x: 1.25, y: 0.9, z: -0.5,
      confidence: 0.72, observations: 1, track: "counter", last_ts: T + 12.0 } });
  epi({ run: A.run, span: A.steps[0], step: 0, id: 301, ts: T + 12.2, action: "navigate_to", obj: "Apple",
    args: { object: "Apple" }, ok: 1 });
  epi({ run: A.run, span: A.steps[1], step: 1, id: 302, ts: T + 31.0, action: "pick_and_place", obj: "Apple",
    args: { object: "Apple", to: "Fridge" }, ok: 1 });
  F.wr({ run: A.run, span: A.steps[0], step: 0, ts: T + 12.4, store: "mem", table: "nav_cells",
    pointer: "FloorPlan1:3:-2", verdict: "insert", summary: "cell (3,-2) unknown → free",
    source: "environment", scope: "FloorPlan1", before: { state: "unknown" }, after: { state: "free" },
    why: "MoveAhead succeeded through the cell", dedup: "FloorPlan1|3|-2",
    tbl: "mem.nav_cells", row: { scene: "FloorPlan1", gx: 3, gz: -2, state: "free", ts: T + 12.4 } });
  F.wr({ run: A.run, span: A.loop, ts: T + 40.6, store: "kb", table: "procedure", pointer: 12,
    verdict: "insert", summary: "Put the apple in the fridge.", source: "environment", scope: "global",
    confidence: 0.9, dedup: "put the apple in the fridge|((navigate_to,apple),(pick_and_place,apple))",
    after: { occurrences: 1, uses: 0 },
    why: "loop closure: command succeeded (goal signal, conf 0.9)",
    tbl: "kb.procedures", row: { id: 12, time: isoTs(T + 40.6), source: "environment", ts: T + 40.6,
      confidence: 0.9, scope: "global", command: "Put the apple in the fridge.",
      plan: JSON.stringify([{ action: "navigate_to", object: "Apple" },
        { action: "pick_and_place", object: "Apple", to: "Fridge" }]),
      scene: "FloorPlan1", tags: JSON.stringify(["auto-loop"]), uses: 0, occurrences: 1 } });
  F.edge("art:kb:procedure:12", "art:mem:episode:302", "wasDerivedFrom", "distilled from the successful episode", T + 40.6);
  F.edge("art:kb:procedure:12", "art:mem:episode:301", "wasDerivedFrom", "distilled from the successful episode", T + 40.6);
  F.wr({ run: A.run, span: A.loop, ts: T + 40.9, store: "file", table: "loop_reports.jsonl",
    pointer: "loop_reports.jsonl#1", verdict: "insert", summary: "loop report #1 (1 write)",
    source: "environment", why: "journal of the loop-closure decision",
    tbl: "file.loop_reports", row: { line_no: 1, ts: T + 40.9, run_id: A.run.run_id,
      command: "Put the apple in the fridge.", success: true,
      writes: JSON.stringify([{ kind: "procedure", id: 12, action: "insert" }]),
      notes: "goal signal satisfied" } });
  F.ev({ type: "loop.closed", ts: T + 41.0, run_id: A.run.run_id, span_id: A.loop.span_id,
    actor: "system:improve", scene: "FloorPlan1", summary: "loop closed: 1 procedure, 2 episodes written",
    payload: { writes: [{ kind: "procedure", id: 12, action: "insert" }], goal_check: "pass" } });
  F.endRun(A.run);

  /* ── R2 · the same task in Persian, now grounded by procedure #12 ────── */
  var B = F.commandRun({ t: T + 300, end: T + 322.1, cmd: "سیب را در یخچال بگذار", lang: "fa", level: "L2",
    scene: "FloorPlan1", arm: "reuse_memory", decision_id: 418, status: "OK", llm_ms: 5100,
    hits: ["kb:procedure:12", "mem:spatial:44"],
    steps: [{ name: "pick_and_place", args: { object: "Apple", to: "Fridge" }, ms: 12600, actions: [
      { a: "MoveAhead", ms: 150, repeat: 4 }, { a: "PickupObject", obj: "Apple", ms: 205 },
      { a: "OpenObject", obj: "Fridge", ms: 245 }, { a: "PutObject", obj: "Fridge", ms: 228 },
      { a: "CloseObject", obj: "Fridge", ms: 219 }] }] });
  epi({ run: B.run, span: B.steps[0], step: 0, id: 305, ts: T + 318.0, action: "pick_and_place",
    obj: "Apple", args: { object: "Apple", to: "Fridge" }, ok: 1 });
  F.wr({ run: B.run, span: B.loop, ts: T + 321.4, store: "kb", table: "procedure", pointer: 12,
    verdict: "reinforce", summary: "Put the apple in the fridge.", source: "environment", scope: "global",
    confidence: 0.93, dedup: "put the apple in the fridge|((navigate_to,apple),(pick_and_place,apple))",
    before: { occurrences: 1, uses: 1, confidence: 0.9 }, after: { occurrences: 2, uses: 1, confidence: 0.93 },
    why: "loop closure: Persian paraphrase succeeded with the retrieved plan" });
  F.wr({ run: B.run, span: B.steps[0], step: 0, ts: T + 318.4, store: "mem", table: "spatial", pointer: 44,
    verdict: "reinforce", summary: "Apple @ FloorPlan1 (1.25, 0.90, -0.50)", source: "environment",
    scope: "FloorPlan1", confidence: 0.85, dedup: "FloorPlan1|Apple|counter",
    before: { observations: 1, confidence: 0.72 }, after: { observations: 2, confidence: 0.85 },
    why: "re-observed at the same pose" });
  F.endRun(B.run);

  /* ── R3 · an L3 abstract goal that FAILS → a lesson is written ───────── */
  var C = F.commandRun({ t: T + 700, end: T + 741.8, cmd: "Slice the tomato and put it on the counter.",
    lang: "en", level: "L3", scene: "FloorPlan2", arm: "explore", decision_id: 419, status: "ERROR",
    error: "PickupObject failed: object not visible", llm_ms: 11200, hits: ["kb:procedure:12"],
    steps: [
      { name: "navigate_to", args: { object: "Tomato" }, ms: 8100, actions: [
        { a: "RotateRight", ms: 165 }, { a: "MoveAhead", ms: 150, repeat: 5 }] },
      { name: "slice_object", args: { object: "Tomato" }, ms: 6400, ok: false,
        error: "PickupObject failed: object not visible", actions: [
        { a: "PickupObject", obj: "Knife", ms: 240 },
        { a: "PickupObject", obj: "Tomato", ms: 260, ok: false, err: "object not visible" }] }
    ] });
  var per = F.mkSpan({ run: C.run, parent: C.root.span_id, name: "observe yolo FloorPlan2", op: "observe",
    kind: "PERCEPTION", t: T + 730, ms: 640, level: "DEBUG", actor: "system:perception",
    attrs: { "thor.scene": "FloorPlan2", detections: 7 } });
  F.ev({ type: "perception.run", ts: T + 730, run_id: C.run.run_id, span_id: per.span_id, level: "DEBUG",
    actor: "system:perception", scene: "FloorPlan2", summary: "yolo: 7 detections (Tomato conf 0.41)",
    payload: { detections: 7, best: { label: "Tomato", conf: 0.41 } } });
  epi({ run: C.run, span: C.steps[1], step: 1, id: 306, ts: T + 736.0, action: "slice_object", obj: "Tomato",
    args: { object: "Tomato" }, ok: 0, err: "PickupObject failed: object not visible" });
  F.wr({ run: C.run, span: C.steps[1], step: 1, ts: T + 736.3, store: "mem", table: "failures",
    pointer: "PickupObject|pickupobject failed not visible|Tomato", verdict: "insert",
    summary: "PickupObject · not visible · Tomato", source: "environment", scope: "FloorPlan2",
    after: { count: 1 }, why: "step 1 failed", dedup: "PickupObject|pickupobject failed not visible|Tomato",
    tbl: "mem.failures", row: { signature: "PickupObject|pickupobject failed not visible|Tomato",
      action: "PickupObject", error_class: "not visible", object_type: "Tomato", count: 1,
      last_ts: T + 736.3, lesson_id: 7 } });
  F.wr({ run: C.run, span: C.loop, ts: T + 741.0, store: "kb", table: "lesson", pointer: 7,
    verdict: "insert", summary: "Move closer and face the object before PickupObject.",
    source: "environment", scope: "global", confidence: 0.8,
    dedup: "slice the tomato and put it on the counter|not visible",
    after: { occurrences: 1 }, why: "failure lesson: PickupObject failed — object not visible",
    tbl: "kb.lessons", row: { id: 7, time: isoTs(T + 741.0), source: "environment", ts: T + 741.0,
      confidence: 0.8, scope: "global", command: "Slice the tomato and put it on the counter.",
      problem: "PickupObject failed", error: "object not visible", error_key: "not visible",
      lesson: "Move closer and face the object before PickupObject.", scene: "FloorPlan2", occurrences: 1 } });
  F.edge("art:kb:lesson:7", "art:mem:failure:PickupObject|pickupobject failed not visible|Tomato",
    "wasDerivedFrom", "distilled from the failure counter", T + 741.0);
  F.ev({ type: "loop.closed", ts: T + 741.4, run_id: C.run.run_id, span_id: C.loop.span_id,
    actor: "system:improve", scene: "FloorPlan2", level: "WARNING",
    summary: "loop closed after failure: 1 lesson written",
    payload: { writes: [{ kind: "lesson", id: 7, action: "insert" }], goal_check: "fail" } });
  F.endRun(C.run);

  /* ── R4 · human teaching (source=human) + a rating on an old episode ─── */
  var t4 = T + 1100;
  var R4 = F.mkRun({ t: t4, end: t4 + 6.2, kind: "teach", cmd: "when I say tidy up, first close the fridge",
    lang: "en", level: "L2", scene: "FloorPlan1", status: "OK", arm: "human" });
  var fb = F.mkSpan({ run: R4, name: "feedback teach", op: "feedback", kind: "FEEDBACK", t: t4, ms: 6200,
    actor: "human:arash", input: { mode: "inline-teaching" } });
  F.ev({ type: "feedback.received", ts: t4 + 0.3, run_id: R4.run_id, span_id: fb.span_id, actor: "human:arash",
    scene: "FloorPlan1", summary: "teaching: tidy up ⇒ close the fridge first",
    payload: { mode: "teach", text: "when I say tidy up, first close the fridge" } });
  F.wr({ run: R4, span: fb, ts: t4 + 4.1, store: "kb", table: "procedure", pointer: 13, verdict: "insert",
    summary: "tidy up ⇒ close_object Fridge, then put items away", source: "human", scope: "global",
    confidence: 1.0, actor: "human:arash", dedup: "tidy up|((close_object,fridge))",
    after: { occurrences: 1, uses: 0 }, why: "taught by the operator in the console",
    tbl: "kb.procedures", row: { id: 13, time: isoTs(t4 + 4.1), source: "human", ts: t4 + 4.1,
      confidence: 1.0, scope: "global", command: "tidy up",
      plan: JSON.stringify([{ action: "close_object", object: "Fridge" }]), scene: "FloorPlan1",
      tags: JSON.stringify(["human"]), uses: 0, occurrences: 1 } });
  F.wr({ run: R4, span: fb, ts: t4 + 5.0, store: "mem", table: "episodes", pointer: 302, verdict: "update",
    summary: "rating +1 on episode #302", source: "human", actor: "human:arash",
    before: { rating: null }, after: { rating: 1 }, why: "operator rated the plan good (decision 417)" });
  F.endRun(R4);

  /* ── R5 · Persian command that CONSUMES procedure #12 and lesson #7 ──── */
  var D = F.commandRun({ t: T + 1500, end: T + 1518.4, cmd: "در یخچال را باز کن", lang: "fa", level: "L1",
    scene: "FloorPlan1", arm: "reuse_memory", decision_id: 421, status: "OK", llm_ms: 4200,
    hits: ["kb:procedure:12", "kb:lesson:7", "mem:spatial:44"],
    steps: [{ name: "open_object", args: { object: "Fridge" }, ms: 9100, actions: [
      { a: "MoveAhead", ms: 150, repeat: 3 }, { a: "OpenObject", obj: "Fridge", ms: 250 }] }] });
  epi({ run: D.run, span: D.steps[0], step: 0, id: 311, ts: T + 1514.0, action: "open_object", obj: "Fridge",
    args: { object: "Fridge" }, ok: 1 });
  F.wr({ run: D.run, span: D.loop, ts: T + 1517.6, store: "kb", table: "procedure", pointer: 12,
    verdict: "reinforce", summary: "Put the apple in the fridge.", source: "environment", scope: "global",
    confidence: 0.95, before: { occurrences: 2, uses: 2, confidence: 0.93 },
    after: { occurrences: 3, uses: 3, confidence: 0.95 },
    dedup: "put the apple in the fridge|((navigate_to,apple),(pick_and_place,apple))",
    why: "loop closure: the retrieved plan generalised to the fridge sub-goal" });
  F.wr({ run: D.run, span: D.loop, ts: T + 1517.9, store: "kb", table: "lesson", pointer: 7,
    verdict: "reinforce", summary: "Move closer and face the object before PickupObject.",
    source: "environment", scope: "global", confidence: 0.88,
    before: { occurrences: 1, uses: 0 }, after: { occurrences: 2, uses: 1 },
    why: "lesson applied successfully: agent moved closer before acting" });
  F.endRun(D.run);

  /* ── R6 · an eval sweep with one child eval_record run ───────────────── */
  var t6 = T + 1900;
  var R6 = F.mkRun({ t: t6, end: t6 + 96.4, kind: "eval", origin: "eval", scene: "FloorPlan1",
    status: "OK", arm: "reuse_memory", hash: "1c77aa30de419f2c" });
  var evs = F.mkSpan({ run: R6, name: "evaluate 12 records", op: "evaluate", kind: "EVALUATOR", t: t6,
    ms: 96400, actor: "system:eval", output: { SR: 0.58, PC: 0.71, TDS: 0.44 },
    attrs: { "gen_ai.evaluation.name": "SR/PC/TDS", "gen_ai.evaluation.score.value": 0.58 } });
  F.ev({ type: "eval.scored", ts: t6 + 96.0, run_id: R6.run_id, span_id: evs.span_id, actor: "system:eval",
    scene: "FloorPlan1", origin: "eval", summary: "eval finished · SR 0.58 · PC 0.71 · TDS 0.44",
    payload: { SR: 0.58, PC: 0.71, TDS: 0.44, n: 12, model: "gemma3:4b" } });
  var R6a = F.mkRun({ t: t6 + 4, end: t6 + 12.2, kind: "eval_record", origin: "eval", parent: R6.run_id,
    cmd: "Open the microwave.", lang: "en", level: "L1", scene: "FloorPlan1", status: "OK" });
  F.mkSpan({ run: R6a, name: "invoke_agent Open the microwave.", op: "invoke_agent", kind: "AGENT",
    t: t6 + 4, ms: 8200, actor: "system:eval" });
  F.endRun(R6a); F.endRun(R6);

  /* ── R7 · maintenance: a prune that invalidates a stale belief ───────── */
  var t7 = T + 2100;
  var R7 = F.mkRun({ t: t7, end: t7 + 1.4, kind: "maint", origin: "cli", scene: "FloorPlan1", status: "OK" });
  var mnt = F.mkSpan({ run: R7, name: "prune spatial (decayed < 0.2)", op: "maintenance", kind: "MAINT",
    t: t7, ms: 1400, actor: "system:memory" });
  F.wr({ run: R7, span: mnt, ts: t7 + 0.9, store: "mem", table: "spatial", pointer: 12, verdict: "delete",
    summary: "Mug @ FloorPlan1 (stale)", source: "environment", scope: "FloorPlan1",
    before: { confidence_raw: 0.31, confidence_decayed: 0.11 }, after: { deleted: true },
    why: "prune: decayed confidence 0.11 < 0.2 after 41 days",
    tbl: "mem.spatial", row: { id: 12, scene: "FloorPlan1", object_type: "Mug", x: -0.4, y: 0.92, z: 1.1,
      confidence: 0.31, observations: 1, track: "counter", last_ts: T - 300000 } });
  F.endRun(R7);

  /* ── R8 · a run still in flight (open span, no ended_ts) ─────────────── */
  var t8 = Math.floor(Date.now() / 1000) - 14;
  var R8 = F.mkRun({ t: t8, kind: "command", cmd: "Make a coffee and bring it to the table.", lang: "en",
    level: "L3", scene: "FloorPlan3", arm: "explore", decision_id: 422 });
  var r8root = F.mkSpan({ run: R8, name: "invoke_agent Make a coffee…", op: "invoke_agent", kind: "AGENT",
    t: t8, ms: 14000, status: "UNSET", actor: "system:webapp" });
  r8root.end_ts = null; r8root.duration_ms = null;
  F.mkSpan({ run: R8, parent: r8root.span_id, name: "chat gemma3:4b", op: "chat", kind: "LLM", t: t8 + 0.6,
    ms: 6200, actor: "llm:gemma3:4b" });
  F.ev({ type: "console.log", ts: t8 + 7, run_id: R8.run_id, level: "DEBUG", actor: "system:webapp",
    scene: "FloorPlan3", summary: "[plan] 4 steps queued, worker busy", payload: { line: "4 steps queued" } });

  /* ── R9 · a noisy exploration scan: volume for the virtualised feed ──── */
  var t9 = T + 2300;
  var R9 = F.mkRun({ t: t9, end: t9 + 190, kind: "explore", origin: "cli", scene: "FloorPlan2",
    status: "OK", arm: "explore" });
  var r9root = F.mkSpan({ run: R9, name: "scan FloorPlan2 reachable grid", op: "observe", kind: "AGENT",
    t: t9, ms: 180000, actor: "system:webapp" });
  var acts = ["MoveAhead", "RotateRight", "MoveAhead", "RotateLeft", "LookDown", "LookUp"];
  for (i = 0; i < 210; i++) {
    var a = acts[i % acts.length], ok = (i % 37) !== 0, ts9 = t9 + i * 0.82;
    var sp9 = F.mkSpan({ run: R9, parent: r9root.span_id, name: a, op: "thor_action", kind: "ACTION",
      t: ts9, ms: 140 + (i % 5) * 12, level: "DEBUG", status: ok ? "OK" : "ERROR",
      error: ok ? null : "collision", actor: "env:ai2thor", repeat: (i % 11 === 0) ? 4 : 1 });
    F.ev({ type: "action.executed", ts: ts9, run_id: R9.run_id, span_id: sp9.span_id, level: "DEBUG",
      actor: "env:ai2thor", scene: "FloorPlan2", origin: "cli",
      summary: a + "() → " + (ok ? "ok" : "FAIL: collision") + ((i % 11 === 0) ? "  ×4" : ""),
      payload: { action: a, success: ok, cell: [i % 9, Math.floor(i / 9) - 4] } });
    if (i % 30 === 0)
      F.wr({ run: R9, span: sp9, ts: ts9, store: "mem", table: "nav_cells",
        pointer: "FloorPlan2:" + (i % 9) + ":" + (Math.floor(i / 9) - 4), verdict: "insert",
        summary: "cell (" + (i % 9) + "," + (Math.floor(i / 9) - 4) + ") → " + (ok ? "free" : "blocked"),
        source: "environment", scope: "FloorPlan2", before: { state: "unknown" },
        after: { state: ok ? "free" : "blocked" }, why: "grid scan",
        tbl: "mem.nav_cells", row: { scene: "FloorPlan2", gx: i % 9, gz: Math.floor(i / 9) - 4,
          state: ok ? "free" : "blocked", ts: ts9 } });
  }
  F.ev({ type: "action.skipped_summary", ts: t9 + 181, run_id: R9.run_id, level: "DEBUG",
    actor: "system:webapp", scene: "FloorPlan2", origin: "cli",
    summary: "filtered 640 Pass/Done primitives from the trace", payload: { Pass: 610, Done: 30 } });
  F.ev({ type: "scene.reset", ts: t9 + 182, run_id: R9.run_id, level: "WARNING", actor: "env:ai2thor",
    scene: "FloorPlan2", origin: "cli", summary: "scene reset → FloorPlan2 (object state reloaded)",
    payload: { scene: "FloorPlan2", reason: "set_sensors" } });
  F.endRun(R9);

  /* rows that pre-date the trace store: real stores contain history with no cause */
  TBL["mem.perception_labels"] = [
    { id: 1, scene: "FloorPlan1", label: "Apple", conf: 0.91, box: "[210,180,58,54]", ts: T - 90000, corrected: 0 },
    { id: 2, scene: "FloorPlan1", label: "Mug", conf: 0.44, box: "[88,301,40,44]", ts: T - 89000, corrected: 1 },
    { id: 3, scene: "FloorPlan2", label: "Tomato", conf: 0.41, box: "[402,222,49,47]", ts: T + 730, corrected: 0 }
  ];
  TBL["mem.bandit_arms"] = [
    { arm: "reuse_memory", n: 41, reward_sum: 27.4, A: "<blob 512 bytes>", b: "<blob 64 bytes>", updated: T + 1517 },
    { arm: "explore", n: 18, reward_sum: 7.1, A: "<blob 512 bytes>", b: "<blob 64 bytes>", updated: T + 741 },
    { arm: "ask_human", n: 4, reward_sum: 3.0, A: "<blob 512 bytes>", b: "<blob 64 bytes>", updated: T + 1100 }
  ];
  TBL["mem.bandit_decisions"] = [
    { id: 417, ts: T, arm: "reuse_memory", context: "[0.1,0.9,0.0]", reward: 1.0, run: "R1" },
    { id: 418, ts: T + 300, arm: "reuse_memory", context: "[0.4,0.9,0.1]", reward: 1.0, run: "R2" },
    { id: 419, ts: T + 700, arm: "explore", context: "[0.0,0.2,0.8]", reward: 0.0, run: "R3" },
    { id: 421, ts: T + 1500, arm: "reuse_memory", context: "[0.6,0.9,0.1]", reward: 1.0, run: "R5" },
    { id: 422, ts: t8, arm: "explore", context: "[0.0,0.3,0.7]", reward: null, run: "R8" }
  ];
  TBL["kb.facts"] = [];
}
/* ── the fixture's query engine: one function per route in §9.1, returning
      exactly the §9.2 payloads (so live and demo code paths are identical) ── */
var CATALOG = [
  { id: "kb.procedures", store: "kb", label: "Procedures", pk: "id",
    columns: ["id","time","source","ts","confidence","scope","command","plan","scene","tags","uses","occurrences"],
    provenance: ["source","ts","confidence","scope"], artifact: { store: "kb", kind: "procedure", ref_col: "id" } },
  { id: "kb.lessons", store: "kb", label: "Lessons", pk: "id",
    columns: ["id","time","source","ts","confidence","scope","command","problem","error","error_key","lesson","scene","occurrences"],
    provenance: ["source","ts","confidence","scope"], artifact: { store: "kb", kind: "lesson", ref_col: "id" } },
  { id: "kb.facts", store: "kb", label: "Facts", pk: "id",
    columns: ["id","time","source","ts","confidence","scope","fact","scene"],
    provenance: ["source","ts","confidence","scope"], artifact: { store: "kb", kind: "fact", ref_col: "id" } },
  { id: "kb.audit", store: "kb", label: "Audit log", pk: "line_no",
    columns: ["line_no","time","op","kind","source","line"], provenance: ["time","source"], artifact: null },
  { id: "mem.episodes", store: "mem", label: "Episodes", pk: "id",
    columns: ["id","ts","scene","command","level","action","args","success","error","state_before","state_after","rating"],
    provenance: ["ts"], artifact: { store: "mem", kind: "episode", ref_col: "id" } },
  { id: "mem.spatial", store: "mem", label: "Spatial beliefs", pk: "id",
    columns: ["id","scene","object_type","x","y","z","confidence","observations","track","last_ts"],
    provenance: ["confidence","last_ts","scene"], artifact: { store: "mem", kind: "spatial", ref_col: "id" } },
  { id: "mem.failures", store: "mem", label: "Failure counters", pk: "signature",
    columns: ["signature","action","error_class","object_type","count","last_ts","lesson_id"],
    provenance: ["last_ts"], artifact: { store: "mem", kind: "failure", ref_col: "signature" } },
  { id: "mem.nav_cells", store: "mem", label: "Nav cells", pk: "scene",
    columns: ["scene","gx","gz","state","ts"], provenance: ["ts","scene"], artifact: null },
  { id: "mem.perception_labels", store: "mem", label: "Perception labels", pk: "id",
    columns: ["id","scene","label","conf","box","ts","corrected"], provenance: ["ts","conf","scene"],
    artifact: { store: "mem", kind: "perception_label", ref_col: "id" } },
  { id: "mem.bandit_arms", store: "mem", label: "Bandit arms", pk: "arm",
    columns: ["arm","n","reward_sum","A","b","updated"], provenance: ["updated"], artifact: null },
  { id: "mem.bandit_decisions", store: "mem", label: "Bandit decisions", pk: "id",
    columns: ["id","ts","arm","context","reward","run"], provenance: ["ts"], artifact: null },
  { id: "trace.runs", store: "trace", label: "Runs", pk: "run_id",
    columns: ["run_id","parent_run_id","kind","origin","started_ts","ended_ts","status","scene","level","lang","command","arm","decision_id","agent_id","config_hash","n_spans","n_events","n_writes"],
    provenance: ["started_ts","origin"], artifact: null },
  { id: "trace.spans", store: "trace", label: "Spans", pk: "span_id",
    columns: ["span_id","run_id","parent_span_id","seq","depth","name","op","kind","step_ix","start_ts","end_ts","duration_ms","status","level","error","actor","repeat"],
    provenance: ["start_ts","actor"], artifact: null },
  { id: "trace.events", store: "trace", label: "Events", pk: "seq",
    columns: ["seq","ts","run_id","span_id","type","actor","level","scene","summary","origin","thread"],
    provenance: ["ts","actor","origin"], artifact: null },
  { id: "trace.artifacts", store: "trace", label: "Artifacts", pk: "artifact_id",
    columns: ["artifact_id","store","kind","ref","created_ts","updated_ts","summary","source","scope","confidence","revision","occurrences","alive","n_causes"],
    provenance: ["source","created_ts","confidence","scope"], artifact: null },
  { id: "trace.writes", store: "trace", label: "Writes ledger", pk: "id",
    columns: ["id","event_seq","run_id","span_id","step_ix","ts","store","table_or_kind","pointer","artifact_id","verdict","why","source","audit_line"],
    provenance: ["ts","source"], artifact: null },
  { id: "trace.edges", store: "trace", label: "PROV edges", pk: "id",
    columns: ["id","src","dst","rel","note","ts"], provenance: ["ts"], artifact: null },
  { id: "file.loop_reports", store: "file", label: "Loop reports", pk: "line_no",
    columns: ["line_no","ts","run_id","command","success","writes","notes"], provenance: ["ts"],
    artifact: { store: "file", kind: "report", ref_col: "line_no" } }
];
var KIND_TABLE = { procedure: "kb.procedures", lesson: "kb.lessons", fact: "kb.facts",
  episode: "mem.episodes", spatial: "mem.spatial", failure: "mem.failures", nav_cell: "mem.nav_cells",
  perception_label: "mem.perception_labels", report: "file.loop_reports" };

function lvlIx(l) { var i = LEVELS.indexOf(l); return i < 0 ? 1 : i; }
function findRun(id) { for (var i = 0; i < DEMO.runs.length; i++) if (DEMO.runs[i].run_id === id) return DEMO.runs[i]; return null; }
function findSpan(id) { for (var i = 0; i < DEMO.spans.length; i++) if (DEMO.spans[i].span_id === id) return DEMO.spans[i]; return null; }
function evPublic(e) {
  var w = null;
  for (var i = 0; i < DEMO.writes.length; i++) if (DEMO.writes[i].event_seq === e.seq) {
    var x = DEMO.writes[i];
    w = { store: x.store, table_or_kind: x.table_or_kind, pointer: x.pointer, verdict: x.verdict,
          artifact_id: x.artifact_id, summary: x.summary };
    break;
  }
  return { seq: e.seq, ts: e.ts, iso: isoTs(e.ts), run_id: e.run_id, span_id: e.span_id, type: e.type,
    actor: e.actor, level: e.level, scene: e.scene, summary: e.summary, origin: e.origin, thread: e.thread,
    has_payload: !!(e.payload && Object.keys(e.payload).length), write: w };
}
function demoCauses(aid) {
  var out = [];
  DEMO.writes.forEach(function (w) {
    if (w.artifact_id !== aid) return;
    var r = w.run_id ? findRun(w.run_id) : null, s = w.span_id ? findSpan(w.span_id) : null;
    out.push({ write_id: w.id, ts: w.ts, verdict: w.verdict, why: w.why, source: w.source,
      before: w.before_json ? JSON.parse(w.before_json) : null,
      after: w.after_json ? JSON.parse(w.after_json) : null, audit_line: w.audit_line,
      event_seq: w.event_seq, step_ix: w.step_ix,
      run: r ? { run_id: r.run_id, command: r.command, scene: r.scene, arm: r.arm,
        decision_id: r.decision_id, started_ts: r.started_ts, kind: r.kind } : null,
      span: s ? { span_id: s.span_id, name: s.name, kind: s.kind, op: s.op } : null });
  });
  return out.sort(function (a, b) { return a.ts - b.ts; });
}
function demoEffects(spanId) {
  return DEMO.writes.filter(function (w) { return w.span_id === spanId; }).map(function (w) {
    return { store: w.store, table_or_kind: w.table_or_kind, pointer: w.pointer, verdict: w.verdict,
      artifact_id: w.artifact_id, summary: w.summary, why: w.why, ts: w.ts };
  });
}
function nodeMeta(id, depth) {
  var kind = id.slice(0, id.indexOf(":")), rest = id.slice(id.indexOf(":") + 1);
  if (kind === "art") {
    var a = DEMO.ART[rest];
    if (!a) return { id: id, type: "artifact", label: rest, sublabel: "(unknown artifact)", depth: depth };
    return { id: id, type: "artifact", label: a.kind + " #" + clip(a.ref, 18), sublabel: a.summary || "",
      depth: depth, store: a.store, kind: a.kind, source: a.source, ts: a.created_ts, alive: a.alive,
      n_causes: a.n_causes, revision: a.revision };
  }
  if (kind === "span") {
    var s = findSpan(rest);
    if (!s) return { id: id, type: "span", label: rest, sublabel: "(unknown span)", depth: depth };
    return { id: id, type: "span", label: clip(s.name, 26), sublabel: s.kind + " · " + dur(s.duration_ms),
      depth: depth, kind: s.kind, ts: s.start_ts, run_id: s.run_id, status: s.status };
  }
  if (kind === "run") {
    var r = findRun(rest);
    return { id: id, type: "run", label: "run " + short(rest, 8), depth: depth,
      sublabel: r ? clip(r.command || r.kind, 34) : "(unknown run)", ts: r ? r.started_ts : null,
      status: r ? r.status : null };
  }
  return { id: id, type: "agent", label: rest, depth: depth,
    sublabel: rest.indexOf("human:") === 0 ? "prov:Person" : "prov:SoftwareAgent" };
}
function demoLineage(node, direction, maxDepth, limit) {
  var depths = {}, keep = {}, order = [], frontier, next, i, e, hit = false;
  depths[node] = 0; keep[node] = true; order.push(node);
  function walk(forward) {
    frontier = [node];
    for (var d = 1; d <= maxDepth; d++) {
      next = [];
      for (i = 0; i < DEMO.edges.length; i++) {
        e = DEMO.edges[i];
        var from = forward ? e.src : e.dst, to = forward ? e.dst : e.src;
        if (frontier.indexOf(from) < 0) continue;
        if (depths[to] === undefined) {
          if (order.length >= limit) { hit = true; continue; }
          depths[to] = d; keep[to] = true; order.push(to); next.push(to);
        }
      }
      if (!next.length) break;
      frontier = next;
    }
  }
  if (direction === "up" || direction === "both") walk(true);
  if (direction === "down" || direction === "both") walk(false);
  var edges = DEMO.edges.filter(function (x) { return keep[x.src] && keep[x.dst]; })
    .map(function (x) { return { src: x.src, dst: x.dst, rel: x.rel, note: x.note, ts: x.ts }; });
  return { root: node, direction: direction, truncated: hit, error: null,
    nodes: order.map(function (n) { return nodeMeta(n, depths[n]); }), edges: edges };
}
function demoTableRows(id) {
  if (DEMO.TBL[id]) return DEMO.TBL[id];
  if (id === "trace.runs") return DEMO.runs;
  if (id === "trace.spans") return DEMO.spans;
  if (id === "trace.events") return DEMO.events;
  if (id === "trace.artifacts") return DEMO.artifacts;
  if (id === "trace.writes") return DEMO.writes;
  if (id === "trace.edges") return DEMO.edges;
  return [];
}
function catEntry(id) { for (var i = 0; i < CATALOG.length; i++) if (CATALOG[i].id === id) return CATALOG[i]; return null; }
function projectRow(cat, row) {
  var out = {};
  cat.columns.forEach(function (c) { out[c] = row[c] === undefined ? null : row[c]; });
  if (cat.artifact) {
    var ref = row[cat.artifact.ref_col];
    if (ref !== undefined && ref !== null) {
      var aid = cat.artifact.store + ":" + cat.artifact.kind + ":" + String(ref).replace(/\//g, "%2F");
      out._artifact_id = aid;
      out._causes = DEMO.ART[aid] ? DEMO.ART[aid].n_causes : 0;
    }
  }
  return out;
}

/* live-tail simulator: the open run keeps producing events while you watch */
function demoTick() {
  var open = null, i;
  for (i = 0; i < DEMO.runs.length; i++) if (DEMO.runs[i].ended_ts === null) open = DEMO.runs[i];
  if (!open) return;
  var now = Date.now() / 1000;
  var acts = ["MoveAhead", "RotateRight", "OpenObject CoffeeMachine", "ToggleObjectOn CoffeeMachine", "PickupObject Mug"];
  var a = acts[DEMO.seq % acts.length];
  DEMO.seq++;
  DEMO.events.push({ seq: DEMO.seq, event_id: String(DEMO.seq), ts: now, run_id: open.run_id,
    span_id: null, type: "action.executed", actor: "env:ai2thor", level: "DEBUG", scene: open.scene,
    summary: a + "() → ok", payload: { action: a, success: true }, origin: "webapp", pid: 9120,
    thread: "worker" });
}

function demoEndpoint(name, p) {
  demoBuild();
  var i, out, rows, lim = Math.min(1000, Math.max(1, parseInt(p.limit, 10) || 200));

  if (name === "trace_events") {
    rows = DEMO.events.filter(function (e) {
      if (p.run_id && e.run_id !== p.run_id) return false;
      if (p.span_id && e.span_id !== p.span_id) return false;
      if (p.actor && e.actor !== p.actor) return false;
      if (p.scene && e.scene !== p.scene) return false;
      if (p.level && e.level !== p.level) return false;
      if (p.min_level && lvlIx(e.level) < lvlIx(p.min_level)) return false;
      if (p.type && p.type.split(",").indexOf(e.type) < 0) return false;
      if (p.q && (e.summary || "").toLowerCase().indexOf(String(p.q).toLowerCase()) < 0) return false;
      if (p.after && e.seq <= +p.after) return false;
      if (p.before && e.seq >= +p.before) return false;
      return true;
    });
    var total = rows.length;
    rows = rows.sort(function (a, b) { return b.seq - a.seq; });
    var page = rows.slice(0, lim);
    if (p.order === "asc") page = page.slice().reverse();
    return { events: page.map(evPublic), total: total, has_more: total > page.length,
      cursor: { first_seq: page.length ? page[page.length - 1].seq : null,
                last_seq: page.length ? page[0].seq : null }, error: null };
  }

  if (name === "trace_event") {
    var seq = +p.seq, e = null;
    for (i = 0; i < DEMO.events.length; i++) if (DEMO.events[i].seq === seq) e = DEMO.events[i];
    if (!e) return { event: null, writes: [], run: null, span: null, error: "no such event: " + p.seq };
    var r = e.run_id ? findRun(e.run_id) : null, s = e.span_id ? findSpan(e.span_id) : null;
    var full = evPublic(e);
    full.payload = e.payload; full.event_id = e.event_id; full.pid = e.pid;
    return { event: full,
      writes: DEMO.writes.filter(function (w) { return w.event_seq === seq; }).map(function (w) {
        return { id: w.id, store: w.store, table_or_kind: w.table_or_kind, pointer: w.pointer,
          artifact_id: w.artifact_id, verdict: w.verdict, why: w.why, source: w.source,
          before: w.before_json ? JSON.parse(w.before_json) : null,
          after: w.after_json ? JSON.parse(w.after_json) : null, audit_line: w.audit_line }; }),
      run: r ? { run_id: r.run_id, command: r.command, scene: r.scene, arm: r.arm, kind: r.kind } : null,
      span: s ? { span_id: s.span_id, name: s.name, kind: s.kind, op: s.op } : null, error: null };
  }

  if (name === "trace_runs") {
    rows = DEMO.runs.filter(function (r) {
      if (p.kind && r.kind !== p.kind) return false;
      if (p.scene && r.scene !== p.scene) return false;
      if (p.level && r.level !== p.level) return false;
      if (p.lang && r.lang !== p.lang) return false;
      if (p.status && r.status !== p.status) return false;
      if (p.q && (r.command || "").toLowerCase().indexOf(String(p.q).toLowerCase()) < 0) return false;
      return true;
    }).sort(function (a, b) { return b.started_ts - a.started_ts; });
    lim = Math.min(200, parseInt(p.limit, 10) || 50);
    return { runs: rows.slice(0, lim).map(function (r) {
      var o = {}; for (var k in r) if (k !== "config") o[k] = r[k];
      o.duration_ms = r.ended_ts ? (r.ended_ts - r.started_ts) * 1000 : null;
      return o; }), has_more: rows.length > lim, error: null };
  }

  if (name === "trace_run") {
    var run = findRun(p.run_id);
    if (!run) return { run: null, spans: [], roots: [], error: "no such run: " + p.run_id };
    var sp = DEMO.spans.filter(function (s) { return s.run_id === run.run_id; })
      .sort(function (a, b) { return a.seq - b.seq; });
    var t0 = run.started_ts, t1 = run.ended_ts || (sp.length ? Math.max.apply(null, sp.map(function (s) {
      return s.end_ts || s.start_ts; })) : t0 + 1);
    var kids = {};
    sp.forEach(function (s) { if (s.parent_span_id) (kids[s.parent_span_id] = kids[s.parent_span_id] || []).push(s.span_id); });
    var minl = p.min_level || "DEBUG";
    return { run: run, roots: sp.filter(function (s) { return !s.parent_span_id; }).map(function (s) { return s.span_id; }),
      spans: sp.map(function (s) {
        var o = {}; for (var k in s) o[k] = s[k];
        o.children = kids[s.span_id] || [];
        o.writes = demoEffects(s.span_id);
        o.n_events = DEMO.events.filter(function (e) { return e.span_id === s.span_id && lvlIx(e.level) >= lvlIx(minl); }).length;
        return o; }),
      timeline: { t0: t0, t1: t1 }, truncated: false, error: null };
  }

  if (name === "trace_entity") {
    var node = p.node || ("art:" + p.store + ":" + p.kind + ":" + p.ref);
    var aid = node.indexOf("art:") === 0 ? node.slice(4) : node;
    var a = DEMO.ART[aid];
    if (!a) return { artifact: null, causes: [], uses: [], error: "unknown entity: " + node };
    var tbl = KIND_TABLE[a.kind], live = null;
    if (tbl && DEMO.TBL[tbl]) {
      var cat = catEntry(tbl);
      DEMO.TBL[tbl].forEach(function (row) {
        if (cat && String(row[cat.pk]) === String(a.ref)) live = row;
      });
    }
    var uses = DEMO.edges.filter(function (x) { return x.rel === "used" && x.dst === "art:" + aid; })
      .map(function (x) {
        var s = findSpan(x.src.slice(5)), r = s ? findRun(s.run_id) : null;
        return { span_id: s ? s.span_id : x.src, name: s ? s.name : x.src, kind: s ? s.kind : null,
          run_id: r ? r.run_id : null, command: r ? r.command : null, ts: x.ts }; });
    return { artifact: a, live: live, causes: demoCauses(aid), uses: uses,
      lineage: demoLineage("art:" + aid, "both", 6, 300), error: null };
  }

  if (name === "trace_span_effects")
    return { span_id: p.span_id, writes: demoEffects(p.span_id), error: null };

  if (name === "trace_lineage")
    return demoLineage(p.node, p.dir || p.direction || "both",
      Math.min(10, Math.max(1, parseInt(p.depth, 10) || 6)), Math.min(500, parseInt(p.limit, 10) || 300));

  if (name === "trace_db")
    return { tables: CATALOG.map(function (c) {
        var o = {}; for (var k in c) o[k] = c[k];
        o.rows = demoTableRows(c.id).length; o.error = null; return o; }),
      stores: { kb: "sim/knowledge_base.json", mem: "sim/memory.db (read-only)",
        trace: "sim/trace.db", file: "sim/loop_reports.jsonl" }, error: null };

  if (name === "trace_db_rows") {
    var c = catEntry(p.table);
    if (!c) return { table: p.table, columns: [], rows: [], total: 0, error: "unknown table" };
    rows = demoTableRows(c.id).map(function (r) { return projectRow(c, r); });
    if (p.q) {
      var needle = String(p.q).toLowerCase();
      rows = rows.filter(function (r) { return JSON.stringify(r).toLowerCase().indexOf(needle) >= 0; });
    }
    if (p.where_col && p.where_val)
      rows = rows.filter(function (r) { return String(r[p.where_col]) === String(p.where_val); });
    if (p.order_by && c.columns.indexOf(p.order_by) >= 0) {
      var descOrder = String(p.desc) === "1" || p.desc === true;
      rows = rows.slice().sort(function (x, y) {
        var a2 = x[p.order_by], b2 = y[p.order_by];
        if (a2 === null || a2 === undefined) return 1;
        if (b2 === null || b2 === undefined) return -1;
        var v = (typeof a2 === "number" && typeof b2 === "number") ? a2 - b2 : String(a2).localeCompare(String(b2));
        return descOrder ? -v : v; });
    }
    var off = Math.max(0, parseInt(p.offset, 10) || 0);
    lim = Math.min(500, Math.max(1, parseInt(p.limit, 10) || 100));
    return { table: c.id, columns: c.columns, rows: rows.slice(off, off + lim), total: rows.length,
      offset: off, limit: lim, error: null };
  }

  if (name === "trace_db_row") {
    var cc = catEntry(p.table);
    if (!cc) return { table: p.table, row: null, error: "unknown table" };
    var found = null;
    demoTableRows(cc.id).forEach(function (r) { if (String(r[cc.pk]) === String(p.pk)) found = r; });
    if (!found) return { table: cc.id, pk: p.pk, row: null, error: "row not found" };
    var prov = {};
    cc.provenance.forEach(function (k) { if (found[k] !== undefined) prov[k] = found[k]; });
    if (prov.ts) prov.time = isoTs(prov.ts);
    var aid2 = null;
    if (cc.artifact) aid2 = cc.artifact.store + ":" + cc.artifact.kind + ":" +
      String(found[cc.artifact.ref_col]).replace(/\//g, "%2F");
    var art = aid2 ? DEMO.ART[aid2] : null;
    if (art) { prov.occurrences = art.occurrences; prov.revision = art.revision; prov.source = prov.source || art.source; }
    var effects = [];
    if (cc.id === "trace.spans") effects = demoEffects(found.span_id);
    if (cc.id === "trace.runs") effects = DEMO.writes.filter(function (w) { return w.run_id === found.run_id; })
      .map(function (w) { return { store: w.store, table_or_kind: w.table_or_kind, pointer: w.pointer,
        verdict: w.verdict, artifact_id: w.artifact_id, summary: w.summary, ts: w.ts }; });
    var causes = aid2 ? demoCauses(aid2) : [];
    return { table: cc.id, pk: String(p.pk), row: found, provenance: prov, artifact_id: aid2,
      causes: causes, effects: effects,
      audit_lines: causes.map(function (x) { return x.audit_line; }).filter(Boolean), error: null };
  }

  if (name === "trace_stats") {
    out = { events_by_type: {}, events_by_level: {}, spans_by_kind: {}, writes_by_store_verdict: [],
      artifacts_by_source: {}, artifacts_by_store_kind: [],
      runs: { total: DEMO.runs.length, ok: 0, error: 0, open: 0 },
      orphans: { writes: 0, events: 0 }, top_artifacts: [], recent_rate: [],
      db: { path: "sim/trace.db (demo fixture — nothing on disk)", size_bytes: 262144, schema_version: 1 },
      tracer: { enabled: true, degraded: false, reason: null, db_path: "sim/trace.db (demo)",
        origin: "webapp", schema_version: 1, api_version: 1, queue_depth: 0, dropped: 0, write_errors: 0,
        events: DEMO.events.length, runs: DEMO.runs.length, spans: DEMO.spans.length,
        writes: DEMO.writes.length, artifacts: DEMO.artifacts.length, edges: DEMO.edges.length,
        skipped_actions: 640, metrics: { emit_us_p50: 41.2, emit_us_p99: 188.6, flush_ms_p99: 3.9 } },
      error: null };
    DEMO.events.forEach(function (e) {
      out.events_by_type[e.type] = (out.events_by_type[e.type] || 0) + 1;
      out.events_by_level[e.level] = (out.events_by_level[e.level] || 0) + 1;
      if (!e.run_id) out.orphans.events++;
    });
    DEMO.spans.forEach(function (s) { out.spans_by_kind[s.kind] = (out.spans_by_kind[s.kind] || 0) + 1; });
    var agg = {};
    DEMO.writes.forEach(function (w) {
      if (!w.run_id) out.orphans.writes++;
      var k = w.store + "|" + w.table_or_kind + "|" + w.verdict;
      agg[k] = (agg[k] || 0) + 1;
    });
    out.writes_by_store_verdict = Object.keys(agg).map(function (k) {
      var s = k.split("|"); return { store: s[0], table_or_kind: s[1], verdict: s[2], n: agg[k] };
    }).sort(function (a, b) { return b.n - a.n; });
    var byKind = {};
    DEMO.artifacts.forEach(function (a) {
      out.artifacts_by_source[a.source || "?"] = (out.artifacts_by_source[a.source || "?"] || 0) + 1;
      var k2 = a.store + "|" + a.kind;
      byKind[k2] = byKind[k2] || { store: a.store, kind: a.kind, n: 0, alive: 0 };
      byKind[k2].n++; byKind[k2].alive += a.alive;
    });
    out.artifacts_by_store_kind = Object.keys(byKind).map(function (k) { return byKind[k]; });
    DEMO.runs.forEach(function (r) {
      if (r.ended_ts === null) out.runs.open++;
      else if (r.status === "OK") out.runs.ok++;
      else if (r.status === "ERROR") out.runs.error++;
    });
    out.top_artifacts = DEMO.artifacts.slice().sort(function (a, b) { return b.n_causes - a.n_causes; })
      .slice(0, 12).map(function (a) { return { artifact_id: a.artifact_id, summary: a.summary,
        n_causes: a.n_causes, revision: a.revision, source: a.source, store: a.store, kind: a.kind }; });
    var buckets = {}, nowMin = Math.floor(Date.now() / 60000);
    DEMO.events.forEach(function (e) {
      var m = Math.floor(e.ts / 60);
      if (nowMin - m <= 60) buckets[m] = (buckets[m] || 0) + 1;
    });
    Object.keys(buckets).sort(function (a, b) { return a - b; }).forEach(function (m) {
      out.recent_rate.push({ minute: isoTs(m * 60), n: buckets[m] });
    });
    return out;
  }

  if (name === "trace_tail") {
    demoTick();
    var after = parseInt(p.after, 10) || 0;
    rows = DEMO.events.filter(function (e) {
      return e.seq > after && lvlIx(e.level) >= lvlIx(p.min_level || "DEBUG");
    }).sort(function (a, b) { return a.seq - b.seq; }).slice(0, lim);
    return { events: rows.map(evPublic), last_seq: DEMO.seq,
      counts: { events: DEMO.events.length, runs: DEMO.runs.length, writes: DEMO.writes.length },
      tracer: { queue_depth: 0, dropped: 0, degraded: false }, error: null };
  }

  if (name === "trace_search") {
    var q = String(p.q || "").toLowerCase();
    lim = Math.min(200, parseInt(p.limit, 10) || 40);
    return { q: p.q, error: null,
      runs: DEMO.runs.filter(function (r) { return (r.command || "").toLowerCase().indexOf(q) >= 0; })
        .slice(0, lim).map(function (r) { return { run_id: r.run_id, command: r.command,
          started_ts: r.started_ts, status: r.status, scene: r.scene, kind: r.kind }; }),
      events: DEMO.events.filter(function (e) { return (e.summary || "").toLowerCase().indexOf(q) >= 0; })
        .slice(-lim).reverse().map(function (e) { return { seq: e.seq, type: e.type, summary: e.summary,
          ts: e.ts, level: e.level, run_id: e.run_id }; }),
      artifacts: DEMO.artifacts.filter(function (a) {
        return ((a.summary || "") + " " + a.ref + " " + a.artifact_id).toLowerCase().indexOf(q) >= 0; })
        .slice(0, lim).map(function (a) { return { artifact_id: a.artifact_id, summary: a.summary,
          n_causes: a.n_causes, store: a.store, kind: a.kind, source: a.source }; }),
      writes: DEMO.writes.filter(function (w) { return (w.why || "").toLowerCase().indexOf(q) >= 0; })
        .slice(0, lim).map(function (w) { return { id: w.id, artifact_id: w.artifact_id,
          verdict: w.verdict, why: w.why, ts: w.ts }; }) };
  }

  return { error: "unknown endpoint /" + name };
}

/* ══════════════════════════════════════════════════════════ ① TIMELINE ══
   A dense feed of EVERY event. Capped at MAX_ROWS in memory and virtualised
   (spacer + absolutely positioned rows), so 10^5 events never enter the DOM. */
var TL = { rows: [], view: [], total: 0, has_more: false, sel: null, ready: false,
  f: { types: [], min_level: "DEFAULT", actor: "", scene: "", run: "", q: "", window: "" } };

function initTimeline() {
  var box = $("tl-types");
  EVENT_TYPES.forEach(function (t) {
    var c = h("button", { class: "chip chip-btn", type: "button", "aria-pressed": "false",
      dataset: { type: t }, text: t });
    on(c, "click", function () {
      var on_ = c.getAttribute("aria-pressed") === "true";
      c.setAttribute("aria-pressed", on_ ? "false" : "true");
      pushTimelineFilters();
    });
    box.appendChild(c);
  });
  ["tl-level", "tl-actor", "tl-scene", "tl-window"].forEach(function (id) {
    on($(id), "change", pushTimelineFilters);
  });
  on($("tl-q"), "input", debounce(pushTimelineFilters, 260));
  on($("tl-run"), "input", debounce(pushTimelineFilters, 260));
  on($("tl-clear"), "click", function () { nav("#/timeline"); });
  on($("tl-older"), "click", tlLoadOlder);
  on($("tl-scroll"), "scroll", tlPaint);
  on($("tl-scroll"), "keydown", function (ev) {
    if (ev.key !== "ArrowDown" && ev.key !== "ArrowUp" && ev.key !== "Enter") return;
    var ix = TL.view.findIndex(function (e) { return e.seq === TL.sel; });
    if (ev.key === "Enter") { if (ix >= 0) openEvent(TL.view[ix].seq); return; }
    ev.preventDefault();
    ix = ev.key === "ArrowDown" ? Math.min(TL.view.length - 1, ix + 1) : Math.max(0, ix - 1);
    if (TL.view[ix]) {
      TL.sel = TL.view[ix].seq;
      $("tl-scroll").scrollTop = Math.max(0, ix * ROW_H - $("tl-scroll").clientHeight / 2);
      tlPaint();
    }
  });
  window.addEventListener("resize", debounce(tlPaint, 120));
}

function pushTimelineFilters() {
  var types = [];
  Array.prototype.forEach.call($("tl-types").children, function (c) {
    if (c.getAttribute("aria-pressed") === "true") types.push(c.dataset.type);
  });
  var p = {};
  if (types.length) p.type = types.join(",");
  if ($("tl-level").value !== "DEFAULT") p.min_level = $("tl-level").value;
  if ($("tl-actor").value) p.actor = $("tl-actor").value;
  if ($("tl-scene").value) p.scene = $("tl-scene").value;
  if ($("tl-window").value) p.window = $("tl-window").value;
  if ($("tl-q").value.trim()) p.q = $("tl-q").value.trim();
  if ($("tl-run").value.trim()) p.run_id = $("tl-run").value.trim();
  nav(routeHash("timeline", null, null, p), true);
}

function renderTimeline(r) {
  var p = r.params || {};
  TL.f = { types: p.type ? p.type.split(",") : [], min_level: p.min_level || "DEFAULT",
    actor: p.actor || "", scene: p.scene || "", run: p.run_id || "", q: p.q || "", window: p.window || "" };
  Array.prototype.forEach.call($("tl-types").children, function (c) {
    c.setAttribute("aria-pressed", TL.f.types.indexOf(c.dataset.type) >= 0 ? "true" : "false");
  });
  $("tl-level").value = TL.f.min_level;
  $("tl-window").value = TL.f.window;
  if ($("tl-q").value !== TL.f.q) $("tl-q").value = TL.f.q;
  if ($("tl-run").value !== TL.f.run) $("tl-run").value = TL.f.run;
  tlLoad();
}

function tlParams(extra) {
  var p = { limit: 400, min_level: TL.f.min_level };
  if (TL.f.types.length) p.type = TL.f.types.join(",");
  if (TL.f.actor) p.actor = TL.f.actor;
  if (TL.f.scene) p.scene = TL.f.scene;
  if (TL.f.run) p.run_id = TL.f.run;
  if (TL.f.q) p.q = TL.f.q;
  for (var k in (extra || {})) p[k] = extra[k];
  return p;
}

function tlLoad() {
  showState($("tl-state"), "loading", "loading events…", "");
  $("tl-state").hidden = false;
  api("trace_events", tlParams()).then(function (res) {
    if (!res || res.error) {
      showState($("tl-state"), "error", "no event feed", hint((res && res.error) ||
        "unknown error"), h("button", { class: "btn", onclick: tlLoad }, "retry"));
      $("tl-state").hidden = false;
      clear($("tl-rows"));
      return;
    }
    TL.rows = res.events || [];
    TL.total = res.total == null ? TL.rows.length : res.total;
    TL.has_more = !!res.has_more;
    tlFillSelects();
    tlApply();
  });
}
function tlLoadOlder() {
  if (!TL.rows.length) return tlLoad();
  var before = TL.rows[TL.rows.length - 1].seq;
  $("tl-older").disabled = true;
  api("trace_events", tlParams({ before: before })).then(function (res) {
    $("tl-older").disabled = false;
    if (!res || res.error || !res.events) return;
    TL.rows = TL.rows.concat(res.events).slice(0, MAX_ROWS);
    TL.has_more = !!res.has_more;
    tlApply();
  });
}
function tlFillSelects() {
  [["tl-actor", "actor"], ["tl-scene", "scene"]].forEach(function (pair) {
    var sel = $(pair[0]), want = pair[0] === "tl-actor" ? TL.f.actor : TL.f.scene, seen = {};
    TL.rows.forEach(function (e) { if (e[pair[1]]) seen[e[pair[1]]] = 1; });
    if (want) seen[want] = 1;
    var opts = Object.keys(seen).sort();
    clear(sel).appendChild(h("option", { value: "" }, pair[1] === "actor" ? "any actor" : "any scene"));
    opts.forEach(function (o) { sel.appendChild(h("option", { value: o, text: o })); });
    sel.value = want;
  });
}
/* the time window is a client-side lens over the loaded page (the endpoint
   pages by seq cursor, not by timestamp) — labelled as such in the footer */
function tlApply() {
  var cutoff = TL.f.window ? (Date.now() / 1000 - parseFloat(TL.f.window)) : null;
  TL.view = cutoff ? TL.rows.filter(function (e) { return e.ts >= cutoff; }) : TL.rows.slice();
  var sc = $("tl-scroll");
  sc.setAttribute("aria-rowcount", String(TL.view.length));
  $("tl-spacer").style.height = (TL.view.length * ROW_H) + "px";
  $("tl-state").hidden = TL.view.length > 0;
  if (!TL.view.length) {
    clear($("tl-rows"));
    showState($("tl-state"), "empty", "no events match",
      TL.rows.length ? "The filters exclude all " + TL.rows.length + " loaded events."
        : "Nothing has been traced yet. Run a command in the console, or open this page with ?demo=1 for a worked example.");
    $("tl-state").hidden = false;
  }
  $("tl-count").textContent = TL.view.length + " shown · " + TL.rows.length + " loaded · " +
    TL.total + " total" + (TL.f.window ? " · window applied to loaded rows" : "");
  $("tl-older").disabled = !TL.has_more;
  tlPaint();
}
function tlPaint() {
  var sc = $("tl-scroll"), host = $("tl-rows");
  if (!sc || !TL.view.length) { clear(host); return; }
  var first = Math.max(0, Math.floor(sc.scrollTop / ROW_H) - 4);
  var count = Math.ceil(sc.clientHeight / ROW_H) + 8;
  clear(host);
  for (var i = first; i < Math.min(TL.view.length, first + count); i++) host.appendChild(tlRow(TL.view[i], i));
}
function tlRow(e, ix) {
  var row = h("div", { class: "tl-row" + (e._new ? " new" : ""), role: "row", tabindex: "-1",
    "aria-selected": TL.sel === e.seq ? "true" : "false",
    style: "top:" + (ix * ROW_H) + "px;border-left-color:" +
      (e.write ? "var(--st-live)" : e.level === "ERROR" ? "var(--st-fail)" : "transparent") });
  row.appendChild(h("span", { class: "c-seq", text: e.seq }));
  row.appendChild(h("span", { class: "c-time", title: e.iso || isoTs(e.ts), text: clock(e.ts) }));
  row.appendChild(h("span", { class: "c-lvl" }, levelDot(e.level)));
  row.appendChild(h("span", { class: "c-type", text: e.type }));
  row.appendChild(h("span", { class: "c-actor", title: e.actor || "", text: e.actor || "—" }));
  row.appendChild(h("span", { class: "c-scene", text: e.scene || "—" }));
  row.appendChild(h("span", { class: "c-sum", dir: "auto", title: e.summary || "", text: e.summary || "" }));
  var w = h("span", { class: "c-write" });
  if (e.write) w.appendChild(writeChip(e.write));
  row.appendChild(w);
  on(row, "click", function () { TL.sel = e.seq; tlPaint(); openEvent(e.seq); });
  return row;
}
function tlPrepend(events) {
  if (!events || !events.length) return;
  var known = {};
  TL.rows.forEach(function (e) { known[e.seq] = 1; });
  var fresh = events.filter(function (e) { return !known[e.seq]; });
  if (!fresh.length) return;
  fresh.forEach(function (e) { e._new = true; });
  TL.rows = fresh.slice().reverse().concat(TL.rows).slice(0, MAX_ROWS);
  TL.total += fresh.length;
  tlApply();
}
function syncLiveTimer() {
  if (S.liveTimer) { clearInterval(S.liveTimer); S.liveTimer = null; }
  if (!S.live) return;
  S.liveTimer = setInterval(function () {
    var after = TL.rows.length ? Math.max.apply(null, TL.rows.map(function (e) { return e.seq; })) : 0;
    api("trace_tail", { after: after, limit: 200, min_level: TL.f.min_level }).then(function (res) {
      if (!res || res.error) return;
      if (S.view === "timeline") tlPrepend(res.events);
      var el = $("health-chip");
      if (res.counts) el.textContent = "live · " + res.counts.events + " ev · " + res.counts.runs +
        " runs · " + res.counts.writes + " writes";
      if (res.tracer) el.classList.toggle("bad", !!res.tracer.degraded);
    });
  }, TAIL_MS);
}

/* ── drawer: one event, with the writes it caused ──────────────────────── */
function openEvent(seq) {
  openDrawer("event #" + seq, function (body) {
    body.appendChild(stateBox("loading", "loading event…", ""));
    api("trace_event", { seq: seq }).then(function (res) {
      clear(body);
      if (!res || res.error || !res.event) {
        body.appendChild(stateBox("error", "event unavailable", (res && res.error) || "not found"));
        return;
      }
      var e = res.event;
      body.appendChild(sec("event", kv([
        ["type", e.type], ["level", e.level], ["time", isoTs(e.ts)], ["actor", e.actor],
        ["scene", e.scene], ["origin", e.origin], ["thread", e.thread], ["summary", e.summary, true]
      ])));
      var links = h("div", { class: "row-links" });
      if (e.run_id) links.appendChild(linkBtn("run " + short(e.run_id, 10) + " ▸",
        function () { nav("#/runs/" + e.run_id); closeDrawer(); }, e.run_id));
      if (res.run && res.run.command) links.appendChild(chip(clip(res.run.command, 30)));
      if (e.span_id) links.appendChild(chip("span " + short(e.span_id, 8)));
      if (links.children.length) body.appendChild(links);
      if (res.span) body.appendChild(sec("span", kv([
        ["name", res.span.name], ["kind", res.span.kind], ["op", res.span.op],
        ["span id", res.span.span_id]])));
      if (res.writes && res.writes.length) {
        var wbox = h("div");
        res.writes.forEach(function (w) { wbox.appendChild(writeCard(w)); });
        body.appendChild(sec("writes caused by this event", wbox));
      }
      if (e.payload && Object.keys(e.payload).length)
        body.appendChild(sec("payload", pre(e.payload)));
    });
  });
}
function writeCard(w) {
  w = normWrite(w);
  var card = h("div", { class: "cause-card" });
  var head = h("div", { class: "head" },
    chip(w.verdict, w.verdict === "insert" ? "chip-ok" : w.verdict === "delete" ? "chip-err" : "chip-live"),
    chip((w.store || "") + " · " + (w.table_or_kind || "")),
    w.source ? chip(w.source) : null);
  card.appendChild(head);
  if (w.artifact_id) card.appendChild(linkBtn(w.artifact_id + " ▸", function () { openEntity("art:" + w.artifact_id); }));
  if (w.summary) card.appendChild(h("p", { class: "why", dir: "auto", text: w.summary }));
  if (w.why) card.appendChild(h("p", { class: "why", dir: "auto", text: "why: " + w.why }));
  if (w.before || w.after) card.appendChild(h("div", { class: "delta" },
    JSON.stringify(w.before) + " → ", h("b", { text: JSON.stringify(w.after) })));
  if (w.audit_line) card.appendChild(h("code", { class: "audit-line", dir: "auto", text: w.audit_line }));
  return card;
}

/* ── drawer: one artifact — causes, uses, live content, lineage jump ───── */
function openEntity(node) {
  openDrawer(node.replace(/^art:/, ""), function (body) {
    body.appendChild(stateBox("loading", "loading entity…", ""));
    api("trace_entity", { node: node }).then(function (res) {
      clear(body);
      if (!res || res.error || !res.artifact) {
        body.appendChild(stateBox("error", "entity unavailable", (res && res.error) || "not found"));
        return;
      }
      var a = res.artifact;
      body.appendChild(h("div", { class: "row-links" },
        linkBtn("open lineage graph ▸", function () { nav("#/lineage/" + encodeURIComponent(node)); closeDrawer(); }),
        KIND_TABLE[a.kind] ? linkBtn("open DB row ▸", function () {
          nav("#/db/" + KIND_TABLE[a.kind] + "/" + encodeURIComponent(a.ref)); closeDrawer(); }) : null));
      body.appendChild(sec("provenance", kv([
        ["summary", a.summary, true], ["store", a.store + " · " + a.kind], ["ref", a.ref],
        ["source", a.source], ["scope", a.scope],
        ["confidence", a.confidence == null ? null : a.confidence],
        ["revision", a.revision], ["occurrences", a.occurrences], ["causes", a.n_causes],
        ["created", isoTs(a.created_ts)], ["updated", isoTs(a.updated_ts)],
        ["alive", a.alive ? "yes" : "no — invalidated"], ["dedup key", a.dedup_key, true]
      ])));
      var cbox = h("div");
      (res.causes || []).forEach(function (c) { cbox.appendChild(causeCard(c)); });
      body.appendChild(sec("caused by (" + (res.causes || []).length + ")",
        (res.causes || []).length ? cbox : stateBox("empty", "no recorded cause",
          "This entry pre-dates the trace store, or was written by a path with no run context (an orphan write).")));
      if (res.uses && res.uses.length) {
        var ubox = h("div");
        res.uses.forEach(function (u) {
          u = normUse(u);
          ubox.appendChild(h("div", { class: "cause-card" },
            h("div", { class: "head" }, chip("used", "chip-acc"), chip(u.kind || "span"),
              chip(isoTs(u.ts))),
            h("p", { class: "why", dir: "auto", text: u.command || u.name }),
            u.run_id ? linkBtn("run " + short(u.run_id, 10) + " ▸", function () {
              nav("#/runs/" + u.run_id); closeDrawer(); }) : null));
        });
        body.appendChild(sec("later runs that used it (" + res.uses.length + ")", ubox));
      }
      if (res.live) body.appendChild(sec("live content in the store", pre(res.live)));
    });
  });
}
function causeCard(c) {
  c = normWrite(c);
  var card = h("div", { class: "cause-card" });
  card.appendChild(h("div", { class: "head" },
    chip(c.verdict, c.verdict === "insert" ? "chip-ok" : c.verdict === "delete" ? "chip-err" : "chip-live"),
    chip(isoTs(c.ts)), c.source ? chip(c.source) : null,
    c.span ? chip(c.span.kind + " · " + clip(c.span.name, 22)) : null));
  if (c.run) {
    card.appendChild(h("p", { class: "why", dir: "auto",
      text: c.run.command || ("(" + c.run.kind + " run)") }));
    card.appendChild(h("div", { class: "row-links" },
      linkBtn("run " + short(c.run.run_id, 10) + " ▸", function () { nav("#/runs/" + c.run.run_id); closeDrawer(); }),
      c.run.scene ? chip(c.run.scene) : null,
      c.run.arm ? chip("arm " + c.run.arm) : null,
      c.run.decision_id != null ? chip("decision " + c.run.decision_id) : null));
  }
  if (c.why) card.appendChild(h("p", { class: "why", dir: "auto", text: "why: " + c.why }));
  if (c.before || c.after) card.appendChild(h("div", { class: "delta" },
    JSON.stringify(c.before) + " → ", h("b", { text: JSON.stringify(c.after) })));
  if (c.audit_line) card.appendChild(h("code", { class: "audit-line", dir: "auto", text: c.audit_line }));
  return card;
}

/* ── drawer: cross-store search ────────────────────────────────────────── */
function openSearch(q) {
  openDrawer("search · " + clip(q, 26), function (body) {
    body.appendChild(stateBox("loading", "searching…", ""));
    api("trace_search", { q: q, limit: 40 }).then(function (res) {
      clear(body);
      if (!res || res.error) { body.appendChild(stateBox("error", "search failed", (res && res.error) || "")); return; }
      var n = (res.runs || []).length + (res.events || []).length + (res.artifacts || []).length + (res.writes || []).length;
      if (!n) { body.appendChild(stateBox("empty", "no matches", "Nothing in runs, events, artifacts or writes matches “" + q + "”.")); return; }
      if ((res.runs || []).length) {
        var rb = h("div");
        res.runs.forEach(function (r) {
          rb.appendChild(h("div", { class: "cause-card" },
            h("p", { class: "why", dir: "auto", text: r.command || r.kind }),
            h("div", { class: "row-links" }, chip(r.status || "—"), chip(r.scene || "—"),
              linkBtn("open run ▸", function () { nav("#/runs/" + r.run_id); closeDrawer(); }))));
        });
        body.appendChild(sec("runs (" + res.runs.length + ")", rb));
      }
      if ((res.artifacts || []).length) {
        var ab = h("div");
        res.artifacts.forEach(function (a) {
          ab.appendChild(h("div", { class: "cause-card" },
            h("div", { class: "head" }, chip(a.artifact_id, "chip-acc"),
              chip((a.n_causes || 0) + " causes")),
            h("p", { class: "why", dir: "auto", text: a.summary || "" }),
            linkBtn("entity ▸", function () { openEntity("art:" + a.artifact_id); })));
        });
        body.appendChild(sec("artifacts (" + res.artifacts.length + ")", ab));
      }
      if ((res.events || []).length) {
        var eb = h("div");
        res.events.forEach(function (e) {
          eb.appendChild(h("div", { class: "cause-card" },
            h("div", { class: "head" }, chip(e.type), chip(isoTs(e.ts))),
            h("p", { class: "why", dir: "auto", text: e.summary }),
            linkBtn("event #" + e.seq + " ▸", function () { openEvent(e.seq); })));
        });
        body.appendChild(sec("events (" + res.events.length + ")", eb));
      }
      if ((res.writes || []).length) {
        var wb = h("div");
        res.writes.forEach(function (w) { wb.appendChild(writeCard(w)); });
        body.appendChild(sec("writes (" + res.writes.length + ")", wb));
      }
    });
  });
}

/* ══════════════════════════════════════════════════════════════ ② RUNS ══
   run → command → plan → step → executed action → the writes it caused.
   Waterfall is hand-drawn CSS; the tree is native <details> (free keyboard). */
var RUNS = { sel: null };

function initRuns() {
  on($("run-kind"), "change", function () { loadRunList(); });
  on($("run-q"), "input", debounce(function () { loadRunList(); }, 260));
}
function renderRuns(r) {
  RUNS.sel = r.arg || null;
  loadRunList();
  if (RUNS.sel) loadRunDetail(RUNS.sel);
  else showState($("run-detail"), "empty", "pick a run",
    "Select a run on the left to see its plan, its waterfall, and every knowledge-base write it caused.");
}
function loadRunList() {
  var host = $("run-list");
  showState(host, "loading", "loading runs…", "");
  api("trace_runs", { limit: 60, kind: $("run-kind").value || null, q: $("run-q").value.trim() || null })
    .then(function (res) {
      if (!guard(host, res, function (x) { return !x.runs || !x.runs.length; }, "no runs recorded",
        "Runs appear here as soon as the agent executes a command with tracing enabled.")) return;
      clear(host);
      res.runs.forEach(function (r) {
        var b = h("button", { class: "run-item", type: "button", role: "option",
          "aria-selected": r.run_id === RUNS.sel ? "true" : "false" },
          h("span", { class: "cmd", dir: "auto", text: r.command || "(" + r.kind + " run)" }),
          h("span", { class: "meta" },
            statusChip(r.ended_ts ? r.status : "RUNNING"),
            chip(r.kind),
            r.level ? h("span", { class: "chip", dataset: { lvl: r.level }, text: r.level }) : null,
            r.lang ? chip(r.lang) : null,
            r.scene ? chip(r.scene) : null,
            chip(dur(r.duration_ms)),
            r.n_writes ? chip(r.n_writes + " writes", "chip-live") : null,
            chip(clock(r.started_ts))));
        on(b, "click", function () { nav("#/runs/" + r.run_id); });
        host.appendChild(b);
      });
    });
}
function loadRunDetail(runId) {
  var host = $("run-detail");
  showState(host, "loading", "loading run…", "");
  api("trace_run", { run_id: runId, min_level: "DEBUG" }).then(function (res) {
    if (!guard(host, res, function (x) { return !x.run; }, "run not found",
      "This run id is not in the trace store.")) return;
    clear(host);
    var run = res.run, spans = res.spans || [];
    var evBySpan = res.events_by_span || {};
    spans.forEach(function (s) { if (s.n_events == null) s.n_events = evBySpan[s.span_id] || 0; });
    var cfg = run.config || (run.config_json ? JSON.parse(run.config_json) : null);

    host.appendChild(h("h2", { class: "run-cmd", dir: "auto",
      text: run.command || "(" + run.kind + " run)" }));
    host.appendChild(h("div", { class: "run-strip" },
      statusChip(run.ended_ts ? run.status : "RUNNING"),
      chip(run.kind), run.level ? h("span", { class: "chip", dataset: { lvl: run.level }, text: run.level }) : null,
      run.lang ? chip("lang " + run.lang) : null,
      run.scene ? chip(run.scene) : null,
      chip("arm " + (run.arm || "—")),
      run.decision_id != null ? chip("decision " + run.decision_id) : null,
      chip("model " + (run.agent_id || "—")),
      chip("config " + short(run.config_hash, 10), null, "runs with the same config_hash are comparable by construction"),
      chip(dur(run.ended_ts ? (run.ended_ts - run.started_ts) * 1000 : null)),
      chip(isoTs(run.started_ts)),
      chip(short(run.run_id, 12), null, run.run_id)));
    if (run.error) host.appendChild(h("p", { class: "why", dir: "auto",
      style: "color:var(--st-fail)", text: run.error }));

    if (!spans.length) {
      host.appendChild(stateBox("empty", "no spans", "This run recorded events but no spans."));
      return;
    }

    /* A scan run can hold hundreds of raw AI2-THOR primitives. They are traced
       (they are the ground truth) but folded away by default so the plan stays
       readable; the toggle brings them back. */
    var prims = spans.filter(function (s) { return s.kind === "ACTION"; }).length;
    var fold = prims > 40 && !RUNS.showPrim;
    var hiddenBy = {};
    if (fold) {
      spans = spans.filter(function (s) {
        if (s.kind !== "ACTION") return true;
        hiddenBy[s.parent_span_id] = (hiddenBy[s.parent_span_id] || 0) + 1;
        return false;
      });
    }

    /* ── waterfall ── */
    var t0 = res.timeline ? res.timeline.t0 : run.started_ts;
    var t1 = res.timeline ? res.timeline.t1 : (run.ended_ts || t0 + 1);
    var span_ms = Math.max(1, (t1 - t0) * 1000);
    var wf = h("div", { class: "wf" });
    spans.forEach(function (s) {
      var st = (s.start_ts - t0) * 1000;
      var w = s.duration_ms == null ? (span_ms - st) : s.duration_ms;
      var bar = h("div", { class: "wf-bar", dataset: { status: s.status },
        title: s.name + " · " + dur(s.duration_ms) + " · " + s.kind,
        style: "left:" + (st / span_ms * 100).toFixed(3) + "%;width:" +
          Math.max(0.4, w / span_ms * 100).toFixed(3) + "%;background:" + kindVar(s.kind) +
          (s.duration_ms == null ? ";opacity:.5" : "") });
      if (s.repeat > 1) bar.appendChild(h("span", { class: "rep", text: "×" + s.repeat }));
      var lab = h("span", { class: "wf-label", tabindex: "0", role: "button",
        title: s.name, style: "padding-left:" + (s.depth * 10) + "px",
        text: (s.step_ix != null ? "[" + s.step_ix + "] " : "") + clip(s.name, 44) });
      on(lab, "click", function () { openSpan(s, run); });
      on(lab, "keydown", function (ev) { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); openSpan(s, run); } });
      on(bar, "click", function () { openSpan(s, run); });
      wf.appendChild(h("div", { class: "wf-row" }, lab, h("div", { class: "wf-track" }, bar)));
    });
    var title = h("div", { class: "card-title" },
      h("span", { text: "waterfall · " + spans.length + " spans" }));
    if (prims > 40) title.appendChild(h("button", { class: "btn", type: "button",
      style: "margin-left:auto", onclick: function () { RUNS.showPrim = !fold; loadRunDetail(runId); } },
      fold ? "show " + prims + " primitives" : "fold primitives"));
    var card = h("div", { class: "card" }, title, wf,
      h("div", { class: "wf-axis" }, h("span", { text: "0 ms" }), h("span", { text: dur(span_ms) })));
    host.appendChild(card);

    /* ── span tree with the writes each step caused ── */
    var byId = {}, roots = [];
    spans.forEach(function (s) { byId[s.span_id] = s; });
    spans.forEach(function (s) { if (!s.parent_span_id || !byId[s.parent_span_id]) roots.push(s); });
    var tree = h("div", { class: "tree" });
    roots.forEach(function (s) { tree.appendChild(spanNode(s, byId, run, 0, hiddenBy)); });
    host.appendChild(h("div", { class: "card" },
      h("div", { class: "card-title" }, "plan tree · click a span for its detail"), tree));

    /* ── everything this run wrote, in one ledger ── */
    var all = [];
    spans.forEach(function (s) { (s.writes || []).forEach(function (w) { all.push(w); }); });
    var wbox = h("div");
    if (all.length) {
      var chips = h("div", { class: "tree" }, h("div", { class: "writes" },
        all.map(function (w) { return writeChip(w); })));
      wbox.appendChild(chips);
    } else {
      wbox.appendChild(stateBox("empty", "this run wrote nothing",
        "No knowledge-base or memory mutation was attributed to this run."));
    }
    host.appendChild(h("div", { class: "card" },
      h("div", { class: "card-title" }, "writes caused by this run (" + all.length + ")"), wbox));

    if (cfg) {
      var d = h("details", null, h("summary", { class: "pane-title", style: "cursor:pointer" },
        "config snapshot · " + short(run.config_hash, 12)), pre(cfg));
      host.appendChild(h("div", { class: "card" }, d));
    }
  });
}
function spanNode(s, byId, run, depth, hiddenBy) {
  var kids = (s.children || []).map(function (id) { return byId[id]; }).filter(Boolean);
  var hidden = (hiddenBy && hiddenBy[s.span_id]) || 0;
  var det = h("details", depth < 2 ? { open: true } : null);
  var sum = h("summary", { class: kids.length ? "" : "leaf" },
    h("span", { class: "kind-bar", style: "background:" + kindVar(s.kind) }),
    h("span", { class: "sname", dir: "auto",
      text: (s.step_ix != null ? "[" + s.step_ix + "] " : "") + s.name }),
    chip(s.kind), s.repeat > 1 ? chip("×" + s.repeat, "chip-live") : null,
    s.status === "ERROR" ? chip("ERROR", "chip-err") : null,
    h("span", { class: "sdur", text: dur(s.duration_ms) }),
    s.n_events ? chip(s.n_events + " ev") : null,
    hidden ? chip(hidden + " primitives folded") : null);
  on(sum, "click", function (ev) {
    if (ev.target.closest(".w-chip")) return;
    if (ev.altKey) { ev.preventDefault(); openSpan(s, run); }
  });
  on(sum, "dblclick", function () { openSpan(s, run); });
  det.appendChild(sum);
  if (s.error) det.appendChild(h("p", { class: "serr", dir: "auto", text: s.error }));
  if (s.writes && s.writes.length)
    det.appendChild(h("div", { class: "writes" }, s.writes.map(function (w) { return writeChip(w); })));
  if (kids.length) {
    var box = h("div", { class: "kids" });
    kids.forEach(function (k) { box.appendChild(spanNode(k, byId, run, depth + 1, hiddenBy)); });
    det.appendChild(box);
  }
  return det;
}
function openSpan(s, run) {
  openDrawer("span · " + clip(s.name, 24), function (body) {
    body.appendChild(sec("span", kv([
      ["name", s.name, true], ["kind", s.kind], ["op", s.op], ["status", s.status],
      ["step", s.step_ix], ["duration", dur(s.duration_ms)], ["start", isoTs(s.start_ts)],
      ["actor", s.actor], ["repeat", s.repeat > 1 ? "×" + s.repeat : null],
      ["span id", s.span_id], ["parent", s.parent_span_id], ["error", s.error, true]
    ])));
    body.appendChild(h("div", { class: "row-links" },
      linkBtn("lineage of this span ▸", function () {
        nav("#/lineage/" + encodeURIComponent("span:" + s.span_id)); closeDrawer(); }),
      run ? linkBtn("events of this run ▸", function () {
        nav(routeHash("timeline", null, null, { run_id: run.run_id, min_level: "DEBUG" })); closeDrawer(); }) : null));
    if (s.writes && s.writes.length) {
      var box = h("div");
      s.writes.forEach(function (w) { box.appendChild(writeCard(w)); });
      body.appendChild(sec("what this span wrote (" + s.writes.length + ")", box));
    } else {
      body.appendChild(sec("what this span wrote", stateBox("empty", "no writes", "This activity mutated nothing.")));
    }
    ["input_json", "output_json", "attrs_json"].forEach(function (k) {
      if (!s[k]) return;
      var v = s[k];
      try { v = JSON.parse(v); } catch (e) {}
      body.appendChild(sec(k.replace("_json", ""), pre(v)));
    });
  });
}

/* ═══════════════════════════════════════════════════════════ ③ LINEAGE ══
   The PROV graph, drawn by hand in SVG. Layer = distance from the queried
   node (the recursive CTE already returns it); order inside a layer comes
   from two barycentre sweeps. Causes sit LEFT of the root, effects RIGHT.
   Nodes are real DOM elements: clickable, hoverable, focusable. */
var LIN = { scale: 1, sel: null, data: null };
var NW = 184, NH = 46, GX = 78, GY = 18, PAD = 22;

function initLineageControls() {
  on($("lin-depth"), "input", function () { $("lin-depth-out").value = $("lin-depth").value; });
  on($("lin-depth"), "change", pushLineage);
  on($("lin-dir"), "change", pushLineage);
  on($("lin-go"), "click", pushLineage);
  on($("lin-node"), "keydown", function (ev) { if (ev.key === "Enter") { ev.preventDefault(); pushLineage(); } });
  on($("lin-zoom-in"), "click", function () { LIN.scale = Math.min(2.2, LIN.scale * 1.2); applyZoom(); });
  on($("lin-zoom-out"), "click", function () { LIN.scale = Math.max(0.35, LIN.scale / 1.2); applyZoom(); });
  on($("lin-zoom-fit"), "click", fitZoom);
  var lg = $("lin-legend");
  RELS.forEach(function (rel) {
    var sw = svgEl("svg", { class: "dag", width: 22, height: 8, "aria-hidden": "true" });
    var ln = svgEl("line", { x1: 1, y1: 4, x2: 21, y2: 4, class: "edge " + rel });
    sw.appendChild(ln);
    var b = h("button", { class: "lg", type: "button", "aria-pressed": "true", dataset: { rel: rel } },
      sw, h("span", { text: rel }));
    on(b, "click", function () {
      var vis = b.getAttribute("aria-pressed") === "true";
      b.setAttribute("aria-pressed", vis ? "false" : "true");
      var hidden = [];
      Array.prototype.forEach.call(lg.children, function (x) {
        if (x.getAttribute("aria-pressed") === "false") hidden.push(x.dataset.rel);
      });
      $("lin-wrap").setAttribute("data-hide", hidden.join(" "));
    });
    lg.appendChild(b);
  });
}
function pushLineage() {
  var node = $("lin-node").value.trim();
  if (!node) return;
  nav(routeHash("lineage", node, null, { dir: $("lin-dir").value, depth: $("lin-depth").value }));
}
function renderLineage(r) {
  var wrap = $("lin-wrap"), node = r.arg;
  var p = r.params || {};
  if (p.dir) $("lin-dir").value = p.dir;
  if (p.depth) { $("lin-depth").value = p.depth; $("lin-depth-out").value = p.depth; }
  fillLineageSuggestions();
  if (!node) {
    $("lin-note").textContent = "—";
    showState(wrap, "empty", "pick a node to trace",
      "Type an id above — art:kb:procedure:12, span:…, run:… — or start from one of the most-caused entries.");
    api("trace_stats", {}).then(function (res) {
      if (!res || res.error || !res.top_artifacts || !res.top_artifacts.length) return;
      var box = h("div", { class: "row-links", style: "justify-content:center" });
      res.top_artifacts.slice(0, 6).forEach(function (a) {
        box.appendChild(h("button", { class: "chip chip-btn", type: "button",
          onclick: function () { nav(routeHash("lineage", "art:" + a.artifact_id, null,
            { dir: $("lin-dir").value, depth: $("lin-depth").value })); } },
          a.artifact_id + " · " + (a.n_causes || 0) + " causes"));
      });
      var st = wrap.querySelector(".state");
      if (st) st.appendChild(box);
    });
    return;
  }
  $("lin-node").value = node;
  showState(wrap, "loading", "walking the graph…", "");
  api("trace_lineage", { node: node, dir: $("lin-dir").value, depth: $("lin-depth").value, limit: 300 })
    .then(function (res) {
      if (!res || res.error) {
        showState(wrap, "error", "lineage unavailable", hint(res && res.error));
        return;
      }
      if (!res.nodes || res.nodes.length <= 1) {
        showState(wrap, "empty", "no lineage recorded for this node",
          "Nothing points at " + node + " and it points at nothing. Orphan writes look like this — " +
          "they are the instrumentation-coverage signal, not an error.");
        $("lin-note").textContent = node + " · 0 edges";
        return;
      }
      LIN.data = res;
      drawGraph(res);
    });
}
function fillLineageSuggestions() {
  var dl = $("lin-suggest");
  if (dl.children.length) return;
  api("trace_stats", {}).then(function (res) {
    if (!res || res.error || !res.top_artifacts) return;
    res.top_artifacts.forEach(function (a) { dl.appendChild(h("option", { value: "art:" + a.artifact_id })); });
  });
}
function layoutGraph(data) {
  var nodes = data.nodes.map(function (n) { var o = {}; for (var k in n) o[k] = n[k]; return o; });
  var byId = {}, out = {}, inn = {};
  nodes.forEach(function (n) { byId[n.id] = n; });
  data.edges.forEach(function (e) {
    if (!byId[e.src] || !byId[e.dst]) return;
    (out[e.src] = out[e.src] || []).push(e.dst);
    (inn[e.dst] = inn[e.dst] || []).push(e.src);
  });
  function bfs(adj) {
    var d = {}, q = [data.root], c, i;
    d[data.root] = 0;
    while (q.length) {
      c = q.shift();
      var nb = adj[c] || [];
      for (i = 0; i < nb.length; i++) if (d[nb[i]] === undefined) { d[nb[i]] = d[c] + 1; q.push(nb[i]); }
    }
    return d;
  }
  var up = bfs(out), down = bfs(inn);       /* up = causes (src→dst), down = effects */
  nodes.forEach(function (n) {
    if (n.id === data.root) { n.lane = 0; return; }
    var u = up[n.id], dn = down[n.id];
    n.lane = (u !== undefined && (dn === undefined || u <= dn)) ? -u
           : (dn !== undefined ? dn : (n.depth || 1));
  });
  var lanes = {};
  nodes.forEach(function (n) { (lanes[n.lane] = lanes[n.lane] || []).push(n); });
  var keys = Object.keys(lanes).map(Number).sort(function (a, b) { return a - b; });
  keys.forEach(function (k) { lanes[k].forEach(function (n, i) { n.order = i; }); });
  for (var pass = 0; pass < 3; pass++) {
    keys.forEach(function (k) {
      lanes[k].forEach(function (n) {
        var nb = (out[n.id] || []).concat(inn[n.id] || []).map(function (id) { return byId[id]; })
          .filter(function (x) { return x && x.lane !== n.lane; });
        n.bc = nb.length ? nb.reduce(function (s, x) { return s + x.order; }, 0) / nb.length : n.order;
      });
      lanes[k].sort(function (a, b) { return a.bc - b.bc; });
      lanes[k].forEach(function (n, i) { n.order = i; });
    });
  }
  var maxCount = 1;
  keys.forEach(function (k) { maxCount = Math.max(maxCount, lanes[k].length); });
  keys.forEach(function (k, ki) {
    var arr = lanes[k], off = (maxCount - arr.length) * (NH + GY) / 2;
    arr.forEach(function (n, i) {
      n.x = PAD + ki * (NW + GX); n.y = PAD + off + i * (NH + GY); n.w = NW; n.h = NH;
    });
  });
  return { nodes: nodes, byId: byId,
    W: PAD * 2 + keys.length * (NW + GX) - GX, H: PAD * 2 + maxCount * (NH + GY) - GY };
}
function drawGraph(data) {
  var wrap = clear($("lin-wrap"));
  var L = layoutGraph(data);
  var svg = svgEl("svg", { class: "dag", viewBox: "0 0 " + L.W + " " + L.H, role: "img",
    "aria-label": "Causal lineage graph of " + data.root });
  var defs = svgEl("defs");
  [["mk-gen", "var(--st-ok)"], ["mk-used", "var(--acc-500)"], ["mk-rev", "var(--st-live)"],
   ["mk-der", "var(--n-200)"], ["mk-inv", "var(--st-fail)"], ["mk-def", "var(--n-500)"]].forEach(function (m) {
    var mk = svgEl("marker", { id: m[0], viewBox: "0 0 8 8", refX: 7, refY: 4, markerWidth: 6,
      markerHeight: 6, orient: "auto-start-reverse" });
    mk.appendChild(svgEl("path", { d: "M0,0 L8,4 L0,8 z", fill: m[1] }));
    defs.appendChild(mk);
  });
  svg.appendChild(defs);
  var MK = { wasGeneratedBy: "mk-gen", used: "mk-used", wasRevisionOf: "mk-rev",
    wasDerivedFrom: "mk-der", wasInvalidatedBy: "mk-inv" };
  var gEdges = svgEl("g"), gNodes = svgEl("g"), paths = {};
  data.edges.forEach(function (e) {
    var a = L.byId[e.src], b = L.byId[e.dst];
    if (!a || !b) return;
    var a1 = { x: b.x >= a.x ? a.x + a.w : a.x, y: a.y + a.h / 2 };
    var b1 = { x: b.x >= a.x ? b.x : b.x + b.w, y: b.y + b.h / 2 };
    var dx = Math.max(28, Math.abs(b1.x - a1.x) * 0.45) * (b1.x >= a1.x ? 1 : -1);
    var p = svgEl("path", { class: "edge " + e.rel,
      d: "M" + a1.x + "," + a1.y + " C" + (a1.x + dx) + "," + a1.y + " " +
         (b1.x - dx) + "," + b1.y + " " + b1.x + "," + b1.y,
      "marker-end": "url(#" + (MK[e.rel] || "mk-def") + ")" });
    p.appendChild(svgEl("title", null, document.createTextNode(
      e.src + " —" + e.rel + "→ " + e.dst + (e.note ? "\n" + e.note : ""))));
    gEdges.appendChild(p);
    (paths[e.src] = paths[e.src] || []).push(p);
    (paths[e.dst] = paths[e.dst] || []).push(p);
  });
  svg.appendChild(gEdges);

  L.nodes.forEach(function (n) {
    var cls = "node " + n.type + (n.id === data.root ? " root" : "") +
      (n.alive === 0 ? " dead" : "") + (LIN.sel === n.id ? " selected" : "");
    var g = svgEl("g", { class: cls, tabindex: "0", role: "button",
      "aria-label": n.type + " " + n.label + (n.sublabel ? ", " + n.sublabel : "") +
        ", press Enter to inspect, Shift+Enter to re-root" });
    var rx = n.type === "agent" ? NH / 2 : n.type === "artifact" ? 7 : 3;
    g.appendChild(svgEl("rect", { class: "n-box", x: n.x, y: n.y, width: n.w, height: n.h, rx: rx, ry: rx }));
    if (n.type === "span") {
      g.appendChild(svgEl("rect", { x: n.x + 1, y: n.y + 1, width: 4, height: n.h - 2, rx: 2,
        fill: kindVar(n.kind) }));
    }
    if (n.type === "artifact" && n.store) {
      g.appendChild(svgEl("rect", { x: n.x + 8, y: n.y + n.h - 15, width: 26, height: 11, rx: 3,
        fill: "var(--n-500)" }));
      var badge = svgEl("text", { class: "n-badge", x: n.x + 21, y: n.y + n.h - 6.5,
        "text-anchor": "middle" });
      badge.appendChild(document.createTextNode(n.store));
      g.appendChild(badge);
    }
    var lab = svgEl("text", { class: "n-label", x: n.x + 10, y: n.y + 18 });
    lab.appendChild(document.createTextNode(clip(n.label || n.id, 26)));
    g.appendChild(lab);
    if (n.sublabel) {
      var sub = svgEl("text", { class: "n-sub", x: n.x + (n.type === "artifact" ? 40 : 10), y: n.y + 33 });
      sub.appendChild(document.createTextNode(clip(n.sublabel, n.type === "artifact" ? 22 : 28)));
      g.appendChild(sub);
    }
    g.appendChild(svgEl("title", null, document.createTextNode(
      n.id + (n.sublabel ? "\n" + n.sublabel : "") + "\ndepth " + n.depth)));
    function inspect() { LIN.sel = n.id; LIN.keepZoom = true; drawGraph(data); openNode(n.id); }
    function reroot() { nav(routeHash("lineage", n.id, null,
      { dir: $("lin-dir").value, depth: $("lin-depth").value })); }
    on(g, "click", inspect);
    on(g, "dblclick", function (ev) { ev.preventDefault(); reroot(); });
    on(g, "keydown", function (ev) {
      if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); ev.shiftKey ? reroot() : inspect(); }
    });
    on(g, "mouseenter", function () { (paths[n.id] || []).forEach(function (p) { p.classList.add("hl"); }); });
    on(g, "mouseleave", function () { (paths[n.id] || []).forEach(function (p) { p.classList.remove("hl"); }); });
    gNodes.appendChild(g);
  });
  svg.appendChild(gNodes);
  wrap.appendChild(svg);
  LIN.W = L.W; LIN.H = L.H;
  if (LIN.keepZoom) { LIN.keepZoom = false; applyZoom(); } else fitZoom();
  $("lin-note").textContent = data.nodes.length + " nodes · " + data.edges.length + " edges · dir " +
    data.direction + (data.truncated ? "  ⚠ graph truncated at the depth/limit — widen with care" : "");
}
function applyZoom() {
  var svg = $("lin-wrap").querySelector("svg.dag");
  if (!svg) return;
  svg.setAttribute("width", Math.round(LIN.W * LIN.scale));
  svg.setAttribute("height", Math.round(LIN.H * LIN.scale));
}
function fitZoom() {
  var wrap = $("lin-wrap");
  if (!LIN.W) return;
  LIN.scale = Math.max(0.35, Math.min(1.6, (wrap.clientWidth - 24) / LIN.W));
  applyZoom();
}
function openNode(id) {
  if (id.indexOf("art:") === 0) return openEntity(id);
  if (id.indexOf("run:") === 0) { nav("#/runs/" + id.slice(4)); return; }
  if (id.indexOf("span:") === 0) {
    var sid = id.slice(5), meta = null;
    (LIN.data ? LIN.data.nodes : []).forEach(function (n) { if (n.id === id) meta = n; });
    openDrawer("span · " + short(sid, 10), function (body) {
      body.appendChild(sec("activity", kv([
        ["label", meta && meta.label], ["detail", meta && meta.sublabel],
        ["kind", meta && meta.kind], ["status", meta && meta.status],
        ["time", meta && meta.ts ? isoTs(meta.ts) : null], ["span id", sid]])));
      if (meta && meta.run_id) body.appendChild(h("div", { class: "row-links" },
        linkBtn("open the run tree ▸", function () { nav("#/runs/" + meta.run_id); closeDrawer(); })));
      var box = h("div");
      body.appendChild(sec("what this activity wrote", box));
      box.appendChild(stateBox("loading", "loading effects…", ""));
      api("trace_span_effects", { span_id: sid, descendants: 1 }).then(function (res) {
        clear(box);
        if (!res || res.error) {
          box.appendChild(stateBox("empty", "effects unavailable", hint(res && res.error)));
          return;
        }
        if (!res.writes || !res.writes.length) {
          box.appendChild(stateBox("empty", "no writes", "This activity mutated nothing."));
          return;
        }
        res.writes.forEach(function (w) { box.appendChild(writeCard(w)); });
      });
    });
    return;
  }
  openDrawer("agent · " + id.slice(6), function (body) {
    body.appendChild(sec("PROV agent", kv([["id", id.slice(6)],
      ["class", id.indexOf("agent:human") === 0 ? "prov:Person" : "prov:SoftwareAgent"]])));
    body.appendChild(h("p", { class: "state-msg",
      text: "Everything attributed to this agent points here with wasAttributedTo or wasAssociatedWith." }));
  });
}

/* ══════════════════════════════════════════════════════════ ④ DATABASE ══
   The stores themselves — knowledge base, memory.db, the trace store and the
   loop-report file — paged, sorted, filtered, and joined to their causes.
   Strictly read-only: there is no edit affordance anywhere in this pane. */
var DB = { table: null, pk: null, offset: 0, limit: 100, order_by: null, desc: true, q: "", cat: null };

function initDb() {
  on($("db-q"), "input", debounce(function () { DB.q = $("db-q").value.trim(); DB.offset = 0; loadDbRows(); }, 260));
  on($("db-prev"), "click", function () { DB.offset = Math.max(0, DB.offset - DB.limit); loadDbRows(); });
  on($("db-next"), "click", function () { DB.offset += DB.limit; loadDbRows(); });
}
function renderDb(r) {
  var table = r.arg, pk = r.arg2;
  if (table !== DB.table) { DB.offset = 0; DB.order_by = null; DB.q = ""; $("db-q").value = ""; }
  DB.table = table; DB.pk = pk;
  loadCatalog();
  if (table) loadDbRows();
  else {
    $("db-table-title").textContent = "—";
    showState($("db-rows"), "empty", "pick a table",
      "Choose a store on the left. Rows are read straight from knowledge_base.json, memory.db (opened read-only) and trace.db.");
  }
  if (table && pk) loadDbRow(table, pk);
  else showState($("db-detail"), "empty", "no row selected",
    "Click a row to see its provenance and everything that caused it.");
}
function loadCatalog() {
  var host = $("db-catalog");
  if (DB.cat) return paintCatalog(DB.cat);
  showState(host, "loading", "loading catalog…", "");
  api("trace_db", {}).then(function (res) {
    if (!guard(host, res, function (x) { return !x.tables || !x.tables.length; }, "no tables",
      "No store could be opened.")) return;
    DB.cat = res;
    paintCatalog(res);
  });
}
function paintCatalog(res) {
  var host = clear($("db-catalog")), groups = {};
  res.tables.forEach(function (t) { (groups[t.store] = groups[t.store] || []).push(t); });
  var LABEL = { kb: "knowledge base", mem: "memory.db", trace: "trace.db", file: "files" };
  Object.keys(groups).forEach(function (store) {
    var g = h("div", { class: "cat-group" },
      h("div", { class: "cat-title", text: LABEL[store] || store }));
    groups[store].forEach(function (t) {
      var b = h("button", { class: "cat-item" + (t.error ? " bad" : ""), type: "button",
        "aria-selected": t.id === DB.table ? "true" : "false", title: t.error || t.id },
        h("span", { text: t.label }),
        h("span", { class: "n", text: t.error ? "error" : (t.rows == null ? "—" : t.rows) }));
      on(b, "click", function () { nav("#/db/" + t.id); });
      g.appendChild(b);
    });
    host.appendChild(g);
  });
  if (res.stores) {
    var f = h("div", { class: "cat-group" }, h("div", { class: "cat-title", text: "on disk" }));
    Object.keys(res.stores).forEach(function (k) {
      f.appendChild(h("div", { class: "mono muted", style: "padding:2px 8px", text: res.stores[k] }));
    });
    host.appendChild(f);
  }
}
function loadDbRows() {
  var host = $("db-rows"), cat = null;
  if (DB.cat) DB.cat.tables.forEach(function (t) { if (t.id === DB.table) cat = t; });
  $("db-table-title").textContent = DB.table || "—";
  if (DB.cat) paintCatalog(DB.cat);
  showState(host, "loading", "loading rows…", "");
  api("trace_db_rows", { table: DB.table, limit: DB.limit, offset: DB.offset, q: DB.q || null,
    order_by: DB.order_by, desc: DB.desc ? 1 : 0 }).then(function (res) {
    if (!res || res.error) {
      showState(host, "error", "cannot read this table", hint(res && res.error));
      $("db-range").textContent = "—";
      return;
    }
    $("db-range").textContent = res.total ? (res.offset + 1) + "–" +
      Math.min(res.total, res.offset + (res.rows || []).length) + " of " + res.total : "0";
    $("db-prev").disabled = res.offset <= 0;
    $("db-next").disabled = res.offset + DB.limit >= res.total;
    if (!res.rows || !res.rows.length) {
      showState(host, "empty", DB.q ? "no rows match the filter" : "this table is empty",
        DB.q ? "Clear the filter to see all " + res.total + " rows."
             : "The store exists but holds no rows yet — a fresh knowledge base looks exactly like this.");
      return;
    }
    clear(host).appendChild(dbTable(res, cat));
  });
}
function dbTable(res, cat) {
  var wrap = h("div", { class: "dtable-wrap" });
  var table = h("table", { class: "dtable" });
  var thead = h("thead"), trh = h("tr");
  var cols = res.columns || Object.keys(res.rows[0]);
  var hasCauses = res.rows.some(function (r) { return r._causes !== undefined; });
  cols.forEach(function (c) {
    var th = h("th", { scope: "col", tabindex: "0", title: "sort by " + c },
      h("span", { text: c }),
      DB.order_by === c ? h("span", { class: "ord", text: DB.desc ? "▼" : "▲" }) : null);
    function sort() {
      if (DB.order_by === c) DB.desc = !DB.desc; else { DB.order_by = c; DB.desc = true; }
      DB.offset = 0; loadDbRows();
    }
    on(th, "click", sort);
    on(th, "keydown", function (ev) { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); sort(); } });
    trh.appendChild(th);
  });
  if (hasCauses) trh.appendChild(h("th", { scope: "col", title: "recorded causes in the writes ledger" }, "causes"));
  thead.appendChild(trh);
  table.appendChild(thead);
  var tbody = h("tbody");
  var pk = (cat && cat.pk) || cols[0];
  res.rows.forEach(function (row) {
    var tr = h("tr", { tabindex: "0", "aria-selected": String(row[pk]) === String(DB.pk) ? "true" : "false" });
    cols.forEach(function (c) { tr.appendChild(dbCell(row[c])); });
    if (hasCauses) tr.appendChild(h("td", null,
      row._causes ? h("span", { class: "causes-badge", text: row._causes + " ▸" }) : h("span", { class: "muted", text: "—" })));
    function open() { nav("#/db/" + res.table + "/" + encodeURIComponent(row[pk])); }
    on(tr, "click", open);
    on(tr, "keydown", function (ev) { if (ev.key === "Enter") { ev.preventDefault(); open(); } });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
  return wrap;
}
function dbCell(v) {
  if (v === null || v === undefined) return h("td", { class: "muted" }, "—");
  if (typeof v === "number") return h("td", { class: "num", text: String(v) });
  var s = String(v);
  if (s.indexOf("<blob") === 0) return h("td", null, h("span", { class: "blob", text: s }));
  var looksJson = (s.charAt(0) === "{" || s.charAt(0) === "[") && s.length > 12;
  if (looksJson) {
    var td = h("td", { class: "json" });
    var btn = h("span", { class: "exp", role: "button", tabindex: "0",
      title: "click to expand", text: clip(s, 40) });
    function toggle(ev) {
      ev.stopPropagation();
      var open = td.classList.toggle("expanded");
      btn.textContent = open ? s : clip(s, 40);
    }
    on(btn, "click", toggle);
    on(btn, "keydown", function (ev) { if (ev.key === "Enter") toggle(ev); });
    td.appendChild(btn);
    return td;
  }
  return h("td", { dir: "auto", title: s, text: s });
}
function loadDbRow(table, pk) {
  var host = $("db-detail");
  showState(host, "loading", "loading row…", "");
  api("trace_db_row", { table: table, pk: pk }).then(function (res) {
    if (!res || res.error || !res.row) {
      showState(host, "error", "row unavailable", hint((res && res.error) || "not found"));
      return;
    }
    clear(host);
    var p = res.provenance || {};
    var prov = h("div", { class: "prov" },
      p.source ? chip(p.source, "chip-acc", "prov:wasAttributedTo") : null,
      p.time || p.ts ? chip(p.time || isoTs(p.ts)) : null,
      p.scope ? chip(p.scope) : null,
      p.occurrences != null ? chip("×" + p.occurrences + " occurrences", "chip-live") : null,
      p.revision != null ? chip("rev " + p.revision) : null);
    if (p.confidence != null) {
      prov.appendChild(chip("conf " + Number(p.confidence).toFixed(2)));
      prov.appendChild(h("span", { class: "meter", title: "confidence " + p.confidence },
        h("i", { style: "width:" + Math.round(Math.max(0, Math.min(1, p.confidence)) * 100) + "%" })));
    }
    host.appendChild(sec("provenance", prov));
    if (res.artifact_id) host.appendChild(h("div", { class: "row-links" },
      linkBtn("lineage ▸", function () { nav("#/lineage/" + encodeURIComponent("art:" + res.artifact_id)); }),
      linkBtn("entity ▸", function () { openEntity("art:" + res.artifact_id); }),
      chip(res.artifact_id)));

    var pairs = [];
    Object.keys(res.row).forEach(function (k) {
      if (k.charAt(0) === "_") return;
      var v = res.row[k];
      pairs.push([k, v === null || v === undefined ? null : String(v), true]);
    });
    host.appendChild(sec("row", kv(pairs)));

    var causes = res.causes || [];
    var cbox = h("div");
    causes.forEach(function (c) { cbox.appendChild(causeCard(c)); });
    host.appendChild(sec("caused by (" + causes.length + ")", causes.length ? cbox :
      stateBox("empty", "no recorded cause",
        "Either this row pre-dates the trace store, or it was written outside any run context.")));

    if (res.effects && res.effects.length) {
      var ebox = h("div", { class: "tree" }, h("div", { class: "writes" },
        res.effects.map(function (w) { return writeChip(w); })));
      host.appendChild(sec("effects — what this activity wrote (" + res.effects.length + ")", ebox));
    }
    if (res.audit_lines && res.audit_lines.length) {
      var abox = h("div");
      res.audit_lines.forEach(function (l) { abox.appendChild(h("code", { class: "audit-line", dir: "auto", text: l })); });
      host.appendChild(sec("legacy audit trail", abox));
    }
  });
}

/* ═════════════════════════════════════════════════════════════ ⑤ STATS ══ */
function barList(entries, onPick) {
  var max = 1;
  entries.forEach(function (e) { max = Math.max(max, e[1]); });
  var box = h("div", { class: "bars" });
  entries.forEach(function (e) {
    var row = h("div", { class: "bar-row" + (onPick ? " clickable" : ""), tabindex: onPick ? "0" : null,
      role: onPick ? "button" : null, title: e[0] + ": " + e[1] },
      h("span", { class: "lbl", text: e[0] }),
      h("span", { class: "track" }, h("i", { class: "fill",
        style: "width:" + Math.max(2, e[1] / max * 100) + "%" + (e[2] ? ";background:" + e[2] : "") })),
      h("span", { class: "val", text: e[1] }));
    if (onPick) {
      on(row, "click", function () { onPick(e); });
      on(row, "keydown", function (ev) { if (ev.key === "Enter") onPick(e); });
    }
    box.appendChild(row);
  });
  return box;
}
function sparkline(points) {
  var w = 280, hgt = 56, n = points.length;
  var svg = svgEl("svg", { class: "spark", viewBox: "0 0 " + w + " " + hgt,
    preserveAspectRatio: "none", role: "img", "aria-label": "events per minute, last hour" });
  if (!n) return svg;
  var max = 1;
  points.forEach(function (p) { max = Math.max(max, p.n); });
  var step = n > 1 ? w / (n - 1) : w;
  var d = "", area = "";
  points.forEach(function (p, i) {
    var x = i * step, y = hgt - 4 - (p.n / max) * (hgt - 10);
    d += (i ? " L" : "M") + x.toFixed(1) + "," + y.toFixed(1);
  });
  area = d + " L" + w + "," + hgt + " L0," + hgt + " Z";
  svg.appendChild(svgEl("path", { class: "ar", d: area }));
  svg.appendChild(svgEl("path", { class: "ln", d: d }));
  svg.appendChild(svgEl("line", { class: "base", x1: 0, y1: hgt - 1, x2: w, y2: hgt - 1 }));
  return svg;
}
function kpi(label, value, cls) {
  return h("div", { class: "kpi" }, h("div", { class: "v " + (cls || ""), text: String(value) }),
    h("div", { class: "k", text: label }));
}
function renderStats() {
  var host = $("stats-grid");
  showState(clear(host), "loading", "loading aggregates…", "");
  api("trace_stats", {}).then(function (res) {
    if (!guard(host, res)) return;
    clear(host);
    var t = res.tracer || {};

    host.appendChild(h("div", { class: "card" }, h("div", { class: "card-title" }, "runs"),
      h("div", { class: "kpis" },
        kpi("total", res.runs.total), kpi("ok", res.runs.ok), kpi("error", res.runs.error, res.runs.error ? "warn" : ""),
        kpi("open", res.runs.open))));

    host.appendChild(h("div", { class: "card" }, h("div", { class: "card-title" }, "store"),
      h("div", { class: "kpis" },
        kpi("events", t.events != null ? t.events : "—"), kpi("spans", t.spans != null ? t.spans : "—"),
        kpi("writes", t.writes != null ? t.writes : "—"), kpi("artifacts", t.artifacts != null ? t.artifacts : "—"),
        kpi("edges", t.edges != null ? t.edges : "—"))));

    var cov = h("div", { class: "card" }, h("div", { class: "card-title" }, "instrumentation coverage"),
      h("div", { class: "kpis" },
        kpi("orphan writes", res.orphans.writes, res.orphans.writes ? "warn" : ""),
        kpi("orphan events", res.orphans.events, res.orphans.events ? "warn" : ""),
        kpi("filtered primitives", t.skipped_actions != null ? t.skipped_actions : "—")),
      h("p", { class: "state-msg", style: "margin-top:8px;text-align:left",
        text: "A non-zero orphan count means a write path exists that no run context covers — the diagnostic that makes “we log everything” checkable." }));
    host.appendChild(cov);

    host.appendChild(h("div", { class: "card" }, h("div", { class: "card-title" }, "events / minute · last hour"),
      sparkline(res.recent_rate || [])));

    var wv = (res.writes_by_store_verdict || []).map(function (r) {
      var color = r.verdict === "insert" ? "var(--st-ok)" : r.verdict === "reinforce" ? "var(--st-live)"
        : r.verdict === "delete" ? "var(--st-fail)" : "var(--acc-600)";
      return [r.store + "." + r.table_or_kind + " · " + r.verdict, r.n, color];
    });
    host.appendChild(h("div", { class: "card" },
      h("div", { class: "card-title" }, "writes by store × verdict"),
      wv.length ? barList(wv) : stateBox("empty", "no writes yet", "The agent has not modified its memory."),
      h("p", { class: "state-msg", style: "margin-top:8px;text-align:left",
        text: "insert vs. reinforce IS the self-improvement signal — reinforcement accumulates causes instead of overwriting." })));

    var src = Object.keys(res.artifacts_by_source || {}).map(function (k) {
      return [k, res.artifacts_by_source[k]]; });
    host.appendChild(h("div", { class: "card" }, h("div", { class: "card-title" }, "artifacts by provenance"),
      src.length ? barList(src) : stateBox("empty", "no artifacts", "")));

    var byType = Object.keys(res.events_by_type || {}).map(function (k) { return [k, res.events_by_type[k]]; })
      .sort(function (a, b) { return b[1] - a[1]; });
    host.appendChild(h("div", { class: "card" }, h("div", { class: "card-title" }, "events by type · click to filter"),
      byType.length ? barList(byType, function (e) { nav(routeHash("timeline", null, null,
        { type: e[0], min_level: "DEBUG" })); }) : stateBox("empty", "no events", "")));

    var byKind = Object.keys(res.spans_by_kind || {}).map(function (k) {
      return [k, res.spans_by_kind[k], kindVar(k)]; }).sort(function (a, b) { return b[1] - a[1]; });
    host.appendChild(h("div", { class: "card" }, h("div", { class: "card-title" }, "spans by kind"),
      byKind.length ? barList(byKind) : stateBox("empty", "no spans", "")));

    var top = (res.top_artifacts || []).filter(function (a) { return a.n_causes > 0; });
    host.appendChild(h("div", { class: "card" },
      h("div", { class: "card-title" }, "most-reinforced memories · click for lineage"),
      top.length ? barList(top.map(function (a) {
        return [clip((a.summary || a.artifact_id), 34), a.n_causes]; }), function (e, i) {
        var pick = top[top.findIndex(function (a) { return clip((a.summary || a.artifact_id), 34) === e[0]; })];
        if (pick) nav(routeHash("lineage", "art:" + pick.artifact_id, null, { dir: "both", depth: "6" }));
      }) : stateBox("empty", "nothing reinforced yet", "Entries gain causes as the loop re-learns them.")));

    var m = t.metrics || {};
    host.appendChild(h("div", { class: "card" }, h("div", { class: "card-title" }, "tracer health"),
      kv([["enabled", String(t.enabled) + (t.pending ? " (store not created yet)" : "")],
        ["degraded", String(t.degraded) + (t.reason ? " — " + t.reason : "")],
        ["queue depth", t.queue_depth], ["dropped", t.dropped], ["write errors", t.write_errors],
        ["emit p50", m.emit_us_p50 != null ? m.emit_us_p50 + " µs" : "—"],
        ["emit p99", m.emit_us_p99 != null ? m.emit_us_p99 + " µs" : "—"],
        ["flush p99", m.flush_ms_p99 != null ? m.flush_ms_p99 + " ms" : "—"],
        ["db", res.db.path, true], ["size", bytes(res.db.size_bytes)],
        ["schema", "v" + res.db.schema_version], ["api", "v" + (t.api_version || "?")]])));
  });
}

/* ══════════════════════════════════════════════════════════ bootstrap ══ */
function initShell() {
  /* demo mode: explicit ?demo=1, or any file:// review with no backend */
  var qp = new URLSearchParams(location.search);
  S.demo = qp.get("demo") === "1" || qp.get("demo") === "true" || location.protocol === "file:";
  if (S.demo) { $("mode-chip").hidden = false; DEMO.build(); }

  /* theme: system -> light -> dark, the SAME localStorage key as the console ("thor.theme"), so both pages match */
  var THEMES = ["system", "light", "dark"];
  var curTheme = function () { var t = document.documentElement.getAttribute("data-theme"); return t === "light" || t === "dark" ? t : "system"; };
  var applyTheme = function (t) {
    if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
    else document.documentElement.removeAttribute("data-theme");
    $("theme-toggle").textContent = "◐ " + (t === "light" ? "Light" : t === "dark" ? "Dark" : "System");
    if (S.view === "lineage") renderLineage(S.route);
    if (S.view === "stats") renderStats(S.route);
  };
  var theme = null;
  try { theme = localStorage.getItem("thor.theme"); } catch (e) {}
  applyTheme(theme === "light" || theme === "dark" ? theme : "system");
  on($("theme-toggle"), "click", function () {
    var next = THEMES[(THEMES.indexOf(curTheme()) + 1) % THEMES.length];
    try { if (next === "system") localStorage.removeItem("thor.theme"); else localStorage.setItem("thor.theme", next); } catch (e) {}
    applyTheme(next);
  });

  /* rail */
  Array.prototype.forEach.call(document.querySelectorAll(".rail-btn"), function (b) {
    on(b, "click", function () { nav("#/" + b.dataset.view); });
  });

  /* global search */
  on($("search-form"), "submit", function (ev) {
    ev.preventDefault();
    var q = $("q-global").value.trim();
    if (q) openSearch(q);
  });

  /* live tail */
  on($("live-toggle"), "click", function () {
    S.live = !S.live;
    $("live-toggle").setAttribute("aria-pressed", S.live ? "true" : "false");
    syncLiveTimer();
  });
  on($("refresh"), "click", renderRoute);
  on($("drawer-close"), "click", closeDrawer);

  /* keyboard: 1–5 switch views, / focuses search, Esc closes the drawer */
  on(document, "keydown", function (ev) {
    var t = ev.target, tag = t && t.tagName;
    var typing = tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || (t && t.isContentEditable);
    if (ev.key === "Escape") { closeDrawer(); return; }
    if (typing || ev.ctrlKey || ev.metaKey || ev.altKey) return;
    var map = { "1": "timeline", "2": "runs", "3": "lineage", "4": "db", "5": "stats" };
    if (map[ev.key]) { ev.preventDefault(); nav("#/" + map[ev.key]); }
    else if (ev.key === "/") { ev.preventDefault(); $("q-global").focus(); }
  });

  on(window, "hashchange", renderRoute);
  on(window, "resize", debounce(function () { if (S.view === "lineage") fitZoom(); }, 160));

  initTimeline();
  initRuns();
  initLineageControls();
  initDb();

  if (!location.hash) history.replaceState(null, "", "#/timeline");
  renderRoute();
  refreshHealth();
}

function refreshHealth() {
  api("trace_stats", {}).then(function (res) {
    var el = $("health-chip");
    if (!res || res.error) { el.textContent = "tracer unavailable"; el.classList.add("bad"); return; }
    S.health = res;
    var t = res.tracer || {};
    var bad = t.degraded || (t.dropped || 0) > 0 || (t.write_errors || 0) > 0;
    el.classList.toggle("bad", !!bad);
    /* `pending` = nothing has been written yet; a read-only view must not create the store to find out. */
    el.textContent = (t.pending ? "idle" : (t.enabled === false ? "off" : "on")) +
      " · " + (res.runs ? res.runs.total : 0) + " runs" +
      " · " + (t.events != null ? t.events : "?") + " ev" +
      (t.dropped ? " · " + t.dropped + " dropped" : "") +
      (t.degraded ? " · degraded" : "");
    el.title = "schema v" + (t.schema_version || "?") + " · " + (t.db_path || "");
  });
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initShell);
else initShell();

})();
