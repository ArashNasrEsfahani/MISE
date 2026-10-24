"use strict";
/* AI2-THOR embodied agent console — behaviour layer (shared + oracle mode).
   Console v2: docs/console-v2-spec.md. Honest mode lives in honest.js (loaded after this file), which this
   file reaches only through window.HONEST. Every element id and endpoint the v1 console used is kept.
   Non-module on purpose: index.html's inline on* handlers resolve these top-level names. */
const $ = (id) => document.getElementById(id);

/* ----------------------------------------------------------- mode (oracle | honest)
   The server is the truth: /status.honest.on sets body[data-mode]; body[data-acting] is present while an honest
   job holds the execution lease. Handlers check these too (hidden buttons are not a gate). */
function honestOn() { return document.body.dataset.mode === "honest"; }
function honestActing() { return document.body.hasAttribute("data-acting"); }
function applyMode(s) {
  const b = document.body;
  const on = !!(s.honest && s.honest.on);
  const mode = on ? "honest" : "oracle";
  const owner = s.lease && s.lease.owner;
  const acting = on && (owner === "honest" || owner === "mode-switch");
  if (b.dataset.mode !== mode) {
    b.dataset.mode = mode;
    if (typeof stopDrive === "function" && driveTimer) stopDrive();   // a mode change never keeps a key held
    hideActPopup();
    markSeg("agent-mode", "data-amode", mode);
    selectFirstVisibleTab(document.querySelector(".mindtabs"), true);
  }
  if (acting !== b.hasAttribute("data-acting")) {
    if (acting) { b.setAttribute("data-acting", ""); if (driveTimer || heldKeys.length) stopDrive(); }
    else b.removeAttribute("data-acting");
  }
  const lock = $("manual-lock");
  if (lock) lock.hidden = !acting;
  document.querySelectorAll(".manual button, #gamemode").forEach(el => { el.disabled = acting; });
}

/* ----------------------------------------------------------- toast: one line, the server's own words */
let _toastTimer = null;
function toast(msg, kind, ms) {
  const t = $("toast"); if (!t) return;
  clearTimeout(_toastTimer);
  if (!msg) { t.hidden = true; return; }
  t.textContent = msg;
  t.className = "toast" + (kind ? " is-" + kind : "");
  t.hidden = false;
  _toastTimer = setTimeout(() => { t.hidden = true; }, ms || 6000);
}

/* ----------------------------------------------------------- shared helpers */
function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, m =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));
}
/* write only when the value actually changed — the 400ms poll used to repaint every node, every tick */
function setText(el, v) {
  if (!el) return;
  const s = (v === null || v === undefined) ? "" : String(v);
  if (el.textContent !== s) el.textContent = s;
}
function setClass(el, s) { if (el && el.className !== s) el.className = s; }   // full overwrite, but only when it differs
function setHTML(el, html, cache, key) {           // innerHTML with a change signature (no thrash, no lost animations)
  if (!el) return;
  if (cache[key] === html) return;
  cache[key] = html;
  el.innerHTML = html;
}
const SIG = Object.create(null);                   // signature cache for the innerHTML writers
function isTextTarget(el) {                        // §10.4 — INPUT / SELECT / TEXTAREA / contenteditable
  if (!el) return false;
  if (el.isContentEditable) return true;
  const t = el.tagName;
  return t === "INPUT" || t === "SELECT" || t === "TEXTAREA";
}
function inWidget(el) {                            // arrow keys belong to the segmented controls / tabs when focused there
  return !!(el && el.closest && el.closest(".seg, .tabs, .actpop"));
}

/* ----------------------------------------------------- connection reporting */
/* A failed request must never be swallowed: /status staleness and one-off request failures
   are both surfaced through #conn (role="status"), which is the console's single truth line. */
const STALE_AFTER = 2;                             // consecutive /status misses before the UI declares itself stale
const NET_ERR_MS = 6000;
let failCount = 0, lastOkPoll = 0, NET_ERR = null;
let CONN = { model: "", note: "connecting…" };

function renderConn() {
  const el = $("conn"); if (!el) return;
  let txt, ok = false;
  if (failCount >= STALE_AFTER) {
    txt = "disconnected — retrying";
  } else if (NET_ERR && Date.now() - NET_ERR.ts < NET_ERR_MS) {
    txt = NET_ERR.msg;
  } else {
    txt = CONN.note || (CONN.model ? "ollama · " + CONN.model : "set a model");
    ok = !!CONN.model && !CONN.note;
  }
  setText(el, txt);
  el.classList.toggle("ok", ok);
}
function setConn(model, note) { CONN = { model: model, note: note }; renderConn(); }
function noteNetError(path, why) {
  NET_ERR = { msg: "request failed · " + String(path).split("?")[0] + (why ? " (" + why + ")" : ""), ts: Date.now() };
  renderConn();
  toast(NET_ERR.msg, "bad");
}
/* every fire-and-forget call goes through here — errors surface in #conn and a toast instead of vanishing.
   A refusal (409 from the honest gate or the lease) carries its reason in JSON: show those words. */
const api = (path) => fetch(path)
  .then(r => {
    if (!r.ok) {
      r.clone().json().then(j => {
        const msg = (j && (j.error || (j.errors && j.errors[0]))) || "";
        if (msg) { NET_ERR = { msg: msg, ts: Date.now() }; renderConn(); toast(msg, r.status === 409 ? "warn" : "bad"); }
        else noteNetError(path, "HTTP " + r.status);
      }).catch(() => noteNetError(path, "HTTP " + r.status));
    }
    return r;
  })
  .catch(() => { noteNetError(path, "network"); });

/* ----------------------------------------------------------- live telemetry */
let lastView = "rgb", lastRes = "";
const STALE_FIELDS = ["fps", "gpu", "hud-fps", "hud-gpu", "t-pos", "t-yaw", "t-pitch", "t-stance"];

/* ----------------------------------------------------------- server/UI version handshake
   sim/web/* is re-read from disk on every request; sim/webapp.py is loaded ONCE at process start. So editing
   the server and reloading the browser silently pairs a new console with an old backend: the new controls 404,
   the new /status fields are simply absent, and every new feature looks broken rather than absent. Bump
   UI_API in step with WEBAPP_API in webapp.py. */
const UI_API = 7;     // 7: honest mode (/honest*, /status.honest|lease); 4: /general, /general_save|delete|review, /general_export, eval `dataset=`; 5: /llm, /status.llm; 6: /inspect, /status.detectors
let _apiWarned = null;
function checkApi(serverApi) {
  const stale = !(serverApi >= UI_API);            // absent (old server, no field) counts as stale
  if (_apiWarned === stale) return;
  _apiWarned = stale;
  let el = $("api-warn");
  if (!stale) { if (el) el.remove(); return; }
  if (!el) {
    el = document.createElement("div");
    el.id = "api-warn";
    el.setAttribute("role", "alert");
    el.textContent = "This console is newer than the running server (ui v" + UI_API + " vs server v"
      + (serverApi || "≤1") + "). Restart it — new controls will 404 and new panels will stay empty until you do.";
    document.body.insertBefore(el, document.body.firstChild);
  }
}

function pollFailed() {                            // §10.1 — a dead simulator must never look like a live one
  failCount++;
  if (failCount < STALE_AFTER) return;
  document.body.classList.add("stale");
  STALE_FIELDS.forEach(id => setText($(id), "—"));
  const c = $("conn");
  if (c) { setText(c, "disconnected — retrying"); c.classList.remove("ok"); }
}

let _polling = false, _pollSeq = 0, _pollApplied = 0;
async function poll() {
  if (_polling) return;                            // one /status in flight at a time: no out-of-order paints
  _polling = true;
  const seq = ++_pollSeq;
  let s;
  try {
    const ac = new AbortController();
    const tmo = setTimeout(() => ac.abort(), 5000);          // a hung /status must not freeze the poll forever
    const res = await fetch("/status", { signal: ac.signal }).finally(() => clearTimeout(tmo));
    if (!res.ok) throw new Error("HTTP " + res.status);   // an HTML error page must not be parsed as telemetry
    s = await res.json();
  } catch (e) { _polling = false; pollFailed(); return; }
  _polling = false;
  if (seq < _pollApplied) return;
  _pollApplied = seq;
  failCount = 0; lastOkPoll = Date.now();
  document.body.classList.remove("stale");
  checkApi(s.api);                                 // before anything else reads a field the server may not have
  applyMode(s);
  if (s.llm) renderLlm(s.llm);
  renderConn();
  renderSimHealth(s);

  setText($("fps"), s.idle ? "idle" : s.fps);          // idle cadence: the loop renders at 5 Hz on purpose
  setText($("gpu"), s.gpu || "—");
  $("chip-gpu").classList.toggle("bad", /cpu/i.test(s.gpu || ""));
  const fpsN = Number(s.fps);
  const cf = $("chip-fps");                               // §10.13 — the low-FPS warning the markup always anticipated
  if (cf) cf.classList.toggle("bad", !s.idle && Number.isFinite(fpsN) && fpsN < 10);
  setText($("scene"), (s.scene || "").replace("_physics", ""));
  setText($("held"), s.held === "none" ? "nothing" : s.held);
  setText($("held-name"), s.held === "none" ? "empty hand" : s.held);
  // viewport HUD
  setText($("hud-scene"), (s.scene || "").replace("_physics", ""));
  setText($("hud-fps"), s.idle ? "idle" : s.fps + " fps");
  setText($("hud-gpu"), s.gpu || "");
  setText($("hud-view"), (s.view || "rgb").toUpperCase());
  setText($("hud-res"), s.resolution || "");
  $("hud-rec").hidden = !s.recording;
  if ($("rec").checked !== !!s.recording) $("rec").checked = !!s.recording;
  // stream aspect ratio -> .vp-media (§10.8): this is what stops the feed letterboxing inside the bezel
  if (s.resolution && s.resolution !== lastRes) {
    lastRes = s.resolution;
    const m = /^\s*(\d+)\s*x\s*(\d+)\s*$/.exec(s.resolution);
    if (m) setCamAr(m[1], m[2]);
  }
  // telemetry strip
  const p = s.pose || {};
  lastPose = p;                                   // shared with the nav-map canvas (agent marker + heading)
  if (s.perception) markSeg("perc", "data-perc", s.perception);
  paintPerception(s);
  paintDetectors(s.detectors);
  renderMotion(s.motion);
  renderLoop(s.loop);
  renderTrain(s.train);
  renderEval(s.eval);
  setText($("t-pos"), (p.x ?? "—") + ", " + (p.z ?? "—"));
  setText($("t-yaw"), (p.rot ?? "—") + "°");
  setText($("t-pitch"), (p.horizon ?? "—") + "°");
  setText($("t-stance"), p.standing === false ? "crouched" : "standing");
  // reflect active view in the segmented control
  if (s.view !== lastView) { lastView = s.view; markSeg("views", "data-view", s.view); }
  // stage + plan: ONE owner per mode. In honest mode honest.js paints the ticket, the stage and the holding
  // belief; the oracle renderers must not overwrite them every tick.
  if (honestOn() && window.HONEST) {
    HONEST.onStatus(s);                            // it owns the stage, the topbar state and the bezel
  } else {
    if (window.HONEST) HONEST.onStatus(s);         // it keeps its availability line + model list current
    renderStage(s);
    renderPlan(s.plan);
    renderRating(s);
  }
  renderDemo(s.demo);
  renderLog(s.log);
}

/* Setup › Simulator: health in words (generation, restarts, render) */
function renderSimHealth(s) {
  const el = $("sim-health"); if (!el) return;
  const sim = s.sim || {}, r = s.render || {};
  const st = sim.state || "ok";
  const word = { ok: "running", busy: "busy (the agent is computing)", "waiting-llm": "waiting for the planner",
                 stalled: "not responding", recovering: "restarting after a crash", starting: "starting" }[st] || st;
  let m = `Simulator ${word} · generation ${sim.generation ?? "—"} · ${sim.recoveries || 0} crash restarts · ` +
          `${r.restarts || 0} render restarts`;
  if (r.glitch) m += ` · last render problem: ${typeof r.glitch === "string" ? r.glitch : JSON.stringify(r.glitch)}`;
  if (sim.waiting) m += ` · ${sim.waiting}`;
  setText(el, m);
  el.classList.toggle("bad", st === "stalled" || st === "recovering");
}

/* transcript — real block elements, bounded, and never destroyed under a live selection (§10.5) */
const LOG_CAP = 300;
let _logSig = null;
/* §5.7 BIDI. `dir="auto"` resolves from the first strong character of the WHOLE node, and
   the server prepends an ASCII speaker prefix ("you: " / "agent: "), so «آیا در یخچال...؟»
   rendered LTR with its «؟» thrown to the visual left. Split the prefix off, and pick the
   line's direction from the MESSAGE alone. */
const SPEAKER_RE = /^\s*(you|agent|error|sys|system)\s*:\s*/i;
const STRONG_RE = /[\u0591-\u08FF\uFB1D-\uFDFD\uFE70-\uFEFC]|[A-Za-z\u00C0-\u02AF\u0370-\u058F]/;
function firstStrongDir(s) {                       // the dir="auto" rule, applied to our own slice
  const m = STRONG_RE.exec(s || "");
  if (!m) return "ltr";
  const c = m[0].charCodeAt(0);
  return (c >= 0x0591 && c <= 0x08FF) || c >= 0xFB1D ? "rtl" : "ltr";
}
function renderLog(lines) {
  const t = $("log"); if (!t) return;
  const arr = (lines || []).slice(-LOG_CAP);
  const sig = arr.join("\n");
  if (sig === _logSig) return;                     // identical transcript -> leave the DOM alone
  const sel = window.getSelection();
  if (sel && sel.rangeCount && !sel.isCollapsed && t.contains(sel.anchorNode)) return;  // user is copying — do not rebuild
  _logSig = sig;
  const atBottom = t.scrollTop + t.clientHeight >= t.scrollHeight - 30;
  const frag = document.createDocumentFragment();
  arr.forEach(line => {
    let c = "";
    if (line.includes("you:")) c = "you"; else if (line.includes("agent:")) c = "ag";
    if (line.includes("error")) c = "er";
    const sp = document.createElement("span");
    sp.className = c ? "ln " + c : "ln";
    // bilingual transcript: the line takes the direction of the MESSAGE, never of the
    // ASCII speaker prefix the server prepends (which would force every Persian line LTR)
    const m = SPEAKER_RE.exec(line);
    const body = m ? line.slice(m[0].length) : line;
    sp.dir = firstStrongDir(body);
    if (m) {
      const who = document.createElement("span");
      who.className = "spk"; who.dir = "ltr"; who.textContent = m[0];
      const msg = document.createElement("span");
      msg.className = "msg"; msg.textContent = body;
      sp.appendChild(who); sp.appendChild(msg);
    } else {
      sp.textContent = line;
    }
    frag.appendChild(sp);
  });
  t.textContent = "";
  t.appendChild(frag);
  if (atBottom) t.scrollTop = t.scrollHeight;
}

/* Simulator health is NOT the job phase: busy (an honest step computing) and waiting-llm are healthy states.
   Returns the failure sentence, or "" when the simulator is fine. Shared with honest.js. */
function simTrouble(s) {
  const sim = s.sim || {};
  const st = sim.state || "ok";
  if (st === "ok" || st === "busy" || st === "waiting-llm") return "";
  return st === "stalled" ? `simulator not responding (${sim.frame_age_s}s)` :
         st === "recovering" ? "simulator crashed — restarting…" : "simulator starting…";
}
function setAgentState(agent) {
  if (document.body.dataset.agent !== agent) document.body.dataset.agent = agent;   // §10.6 — bezel + state chip
}
function renderStage(s) {
  const el = $("stage"), txt = $("stage-text");
  el.className = "stage";                          // full overwrite (contract) — state lives on <body> instead
  let agent = "idle";
  const bad = simTrouble(s);
  if (bad) {                                       // a dead / hung / restarting simulator is never shown as "ready"
    el.classList.add("fail"); setText(txt, bad); setAgentState("fail");
    setText($("tb-state"), bad);
    return;
  }
  if (s.busy) {
    if (s.plan && s.plan.active >= 0) {
      el.classList.add("exec"); agent = "exec";
      setText(txt, `executing · step ${s.plan.active + 1}/${s.plan.steps.length}`);
    } else if (s.lease && s.lease.owner && s.lease.owner !== "oracle-cmd") {
      el.classList.add("think"); agent = "think"; setText(txt, `busy: ${s.lease.owner}`);
    } else { el.classList.add("think"); agent = "think"; setText(txt, "thinking…"); }
  } else if (s.plan && s.plan.steps && s.plan.steps.length) {
    const ok = s.plan.steps.filter(x => x.status === "done").length;
    const fail = s.plan.steps.some(x => x.status === "failed");
    el.classList.add(fail ? "fail" : "done"); agent = fail ? "fail" : "done";
    setText(txt, `${ok}/${s.plan.steps.length} steps ${fail ? "— some failed" : "done"}`);
  } else { setText(txt, "ready"); }
  setAgentState(agent);
  setText($("tb-state"), txt.textContent);         // topbar mirror, readable while looking at the feed
  renderLevel(s.plan);
  renderOracleTicket(s);
}
/* The level is never guessed from the plan's shape (spec §3.3). Oracle mode shows the planner's own claim
   (/status.plan.level) when it made one; honest mode shows the dataset level of a picked item. */
function levelOf(raw) {
  if (raw == null) return null;
  const m = /L?([123])/.exec(String(raw).toUpperCase());
  return m ? "L" + m[1] : null;
}
function renderLevel(plan) {
  showLevel(plan ? levelOf(plan.level) : null, plan && levelOf(plan.level) ? "planner" : "none");
}
const LEVEL_WORD = { L1: "atomic — one command, one function",
                     L2: "sequential — an ordered chain of functions",
                     L3: "abstract goal — decomposed into a new plan" };
const LEVEL_SHORT = { L1: "L1 atomic", L2: "L2 sequential", L3: "L3 abstract" };
/* source: "dataset" | "planner" | "none". Shared with honest.js. */
function showLevel(lvl, source) {
  const b = document.body;
  if (lvl) { if (b.dataset.level !== lvl) b.dataset.level = lvl; }
  else if (b.dataset.level) delete b.dataset.level;
  const t = $("ticket"); if (t && t.dataset.lvl !== (lvl || "")) t.dataset.lvl = lvl || "";
  setText($("t-level"), lvl ? LEVEL_SHORT[lvl] : "level: not classified");
  setText($("t-levelsrc"), !lvl ? "" : source === "dataset" ? "dataset level" : "planner's claim");
  sayLevel(lvl);
}
/* The spine is the thesis's "single most important design fact" — it may not be
   colour-only, and it may not be silent. aria-current marks the active segment and
   #level-say (role=status) states it in words for anyone not looking at the tint. */
function sayLevel(lvl) {
  const spine = $("level-spine"); if (!spine) return;
  spine.querySelectorAll(".lvseg").forEach(seg => {
    const on = seg.dataset.lvl === lvl;
    if (on) seg.setAttribute("aria-current", "step");
    else if (seg.hasAttribute("aria-current")) seg.removeAttribute("aria-current");
  });
  setText($("level-say"), lvl ? `command level: ${lvl} ${LEVEL_WORD[lvl] || ""}`
                              : "command level: not classified yet");
}
/* Oracle ticket: header, provenance and the three stamps. Oracle runs are never audited, so that stamp says
   so instead of pretending to a result; the Goal stamp shows the executor's own goal check (plan.goal). */
function stampHTML(cls, title, body) {
  return `<div class="stamp is-${cls}"><b>${esc(title)}</b>${body}</div>`;
}
/* The oracle executor does run a goal check after the plan (plan.goal, the same check that writes the
   "goal: ..." step). Show its verdict; say "not checked" only when there is none. */
function goalStamp(g, live) {
  if (!g) return stampHTML(live ? "live" : "none", "Goal", live ? "checked after the plan" : "not checked here");
  if (g.error) return stampHTML("none", "Goal", `check failed: ${esc(g.error)}`);
  if (g.applicable === false) return stampHTML("none", "Goal", "no checkable goal");
  if (g.ok === true) return stampHTML("ok", "Goal", "met" + (g.target ? ` · <bdi>${esc(g.target)}</bdi>` : ""));
  if (g.ok === false) return stampHTML("fail", "Goal", "not met" + (g.target ? ` · <bdi>${esc(g.target)}</bdi>` : ""));
  return stampHTML("none", "Goal", "not checked here");
}
function renderOracleTicket(s) {
  const plan = s.plan;
  const cmd = plan && plan.command;
  setHTML($("t-cmd"), cmd ? `<bdi dir="auto">${esc(cmd)}</bdi>` : `<span class="dim">No instruction yet</span>`, SIG, "tcmd");
  setText($("t-prov"), "simulator truth");
  setClass($("t-prov"), "badge prov truth");
  let st = "";
  if (plan && plan.steps && plan.steps.length) {
    const n = plan.steps.length, ok = plan.steps.filter(x => x.status === "done").length;
    const fail = plan.steps.some(x => x.status === "failed");
    const live = s.busy;
    st = stampHTML(live ? "live" : (fail || ok < n) ? "fail" : "ok", "Executed", `<bdi>${ok}</bdi> of <bdi>${n}</bdi> steps ok`) +
         goalStamp(plan.goal, live) +
         stampHTML("none", "Audit", "not audited (oracle)");
  }
  setHTML($("t-stamps"), st, SIG, "tstamps");
  const p = $("t-planner");
  let line = "";
  if (plan && plan.say) line = `Planner: ${plan.say}` + (plan.repaired ? " · repaired by the console" : "");
  setText(p, line);
  if (p) p.dir = "auto";
}
function renderPlan(plan) {
  const box = $("plan");
  if (!plan || !plan.steps || !plan.steps.length) { setHTML(box, "", SIG, "plan"); return; }
  const icon = { pending: "·", running: "…", done: "✓", failed: "✗" };
  setHTML(box, plan.steps.map((st, i) =>
    `<div class="pstep is-${esc(st.status)}"><span class="n">${i + 1}</span><span class="st">${icon[st.status] || "·"}</span>` +
    `<span class="v" dir="auto">${esc(st.label)}</span></div>`
  ).join(""), SIG, "plan");
}

/* ----------------------------------------------------------- plan rating (Phase 6 bandit feedback) */
const RATED = Object.create(null);   // decision_id -> true once a thumb was sent (survives re-polls)
let RATE_ID = null;                  // decision_id of the finished plan currently on screen

function planDecisionId(s) {         // read decision_id from the plan JSON, falling back to status top level
  try {
    const v = (s.plan && s.plan.decision_id != null) ? s.plan.decision_id
            : (s.decision_id != null) ? s.decision_id : null;
    return (v != null && Number.isFinite(+v)) ? +v : null;
  } catch (e) { return null; }
}
function renderRating(s) {
  const box = $("rate"); if (!box) return;
  try {
    const finished = !s.busy && s.plan && s.plan.steps && s.plan.steps.length;
    const did = finished ? planDecisionId(s) : null;
    if (did == null) { box.hidden = true; RATE_ID = null; return; }
    if (did !== RATE_ID) {           // a different plan arrived — reset the control for it
      RATE_ID = did;
      const done = !!RATED[did];
      $("rate-up").disabled = done; $("rate-down").disabled = done;
      setText($("rate-note"), done ? "thanks, learning" : "");
    }
    box.hidden = false;
  } catch (e) { box.hidden = true; }
}
function sendRating(up) {            // exception-safe: a failed POST never breaks the console
  try {
    if (RATE_ID == null || RATED[RATE_ID]) return;
    RATED[RATE_ID] = true;
    $("rate-up").disabled = true; $("rate-down").disabled = true;
    setText($("rate-note"), "thanks, learning");
    fetch("/rating?decision_id=" + RATE_ID + "&rating=" + (up ? 1 : -1))
      .catch(() => noteNetError("/rating", "network"));
  } catch (e) {}
}
/* Phase-7 self-improving loop: toggle state, last write-back summary, and the RQ3 ledger strip */
function renderLoop(lp) {
  try {
    if (!lp) return;
    if ($("loop").checked !== !!lp.enabled) $("loop").checked = !!lp.enabled;
    const el = $("loop-stat"); if (!el) return;
    const last = lp.last;
    if (!last) {
      setText(el, lp.enabled ? "loop on · nothing learned yet" : "loop off");
      setHTML($("loop-ledger"), "", SIG, "ledger");
      return;
    }
    let m = last.success ? "✓" : "✗";
    const writes = last.writes || [];
    const w = writes[0];
    if (w) m += ` ${w.action === "reinforce" ? "reinforced" : "learned"} ${w.kind} #${w.id}`;
    else m += lp.enabled ? " no write-back" : " loop off — nothing written";
    if (last.reindexed != null) m += ` · ${last.reindexed} embedded`;
    setText(el, "last: " + m);
    el.title = last.command || "";
    // §10.16 — up to three chips, one per knowledge-base write the loop just made
    setHTML($("loop-ledger"), writes.slice(0, 3).map(x =>
      `<span class="ledger-chip" data-kind="${esc(x.kind)}">${x.action === "reinforce" ? "↻" : "+"}${esc(x.kind)} #${esc(x.id)}</span>`
    ).join(""), SIG, "ledger");
  } catch (e) {}
}
function renderDemo(d) {
  try {
    const active = d && d.active;
    $("demo-start").disabled = !!active;
    $("demo-stop").disabled = !active;
    $("demo-cancel").disabled = !active;
    setText($("demo-stat"), active ? `${d.steps || 0} steps` : "");
  } catch (e) {}
}

/* ----------------------------------------------------------- objects */
let OBJS = [], SEL = null, LIQUIDS = ["water"];
async function loadObjs() {
  if (document.hidden) return;                     // §10.17 — no background repaint work on a hidden tab
  try { OBJS = await (await fetch("/objects")).json(); renderObjs(); } catch (e) {}
}
function setHudObj(type) { setText($("hud-obj"), type || ""); }   // §10.13 — object under the cursor / current selection
let _objSig = null;
function renderObjs() {
  const q = $("objq").value.toLowerCase(), vo = $("objvis").checked, box = $("objs");
  const list = OBJS.filter(o => (!q || o.type.toLowerCase().includes(q)) && (!vo || o.visible)).slice(0, 100);
  // /objects is polled every 1.5s; rebuilding an unchanged list would blow away focus (rows are tabbable) and selection
  const sig = JSON.stringify([SEL, list.map(o => [o.id, o.type, o.dist, o.flags, !!o.visible])]);
  if (sig !== _objSig) {
    _objSig = sig;
    box.innerHTML = "";
    list.forEach(o => {
      const d = document.createElement("div");
      d.className = "orow" + (SEL === o.id ? " sel" : "");
      d.tabIndex = 0;                                                  // §10.19 — reachable without a mouse
      d.innerHTML = `<span class="od ${o.visible ? "vis" : ""}"></span><span class="ot">${esc(o.type)}</span>` +
                    `<span class="of">${esc(o.dist)}m ${esc(o.flags)}</span>`;
      const pick = (cx, cy, kbd) => { SEL = o.id; setHudObj(o.type); renderObjs(); renderActs(o); showActPopup(o, cx, cy, kbd); };
      d.onclick = (ev) => pick(ev.clientX, ev.clientY);
      // d.click() from the keyboard synthesises clientX/clientY = 0, which the popup's
      // `cx || 80` fallback then read as "no position" and parked in the top-left corner,
      // a screen away from the row. Anchor to the ROW's own box instead.
      d.onkeydown = (ev) => {
        if (ev.key !== "Enter" && ev.key !== " ") return;
        ev.preventDefault();
        const r = d.getBoundingClientRect();
        pick(r.right - 24, r.bottom, true);
      };
      box.appendChild(d);
    });
  }
  if (SEL) { const s = OBJS.find(o => o.id === SEL); if (s) { renderActs(s); setHudObj(s.type); } }
}
function actButtons(container, o, openId, liquidId) {
  container.innerHTML = "";
  ["Go To", "Look At"].concat(o.actions || []).forEach(lbl => {
    const b = document.createElement("button"); b.type = "button"; b.textContent = lbl;
    b.onclick = () => {
      const p = { id: o.id, action: lbl };
      if (lbl === "Open" && $(openId)) p.openness = $(openId).value;
      if (lbl === "Fill" && $(liquidId)) p.liquid = $(liquidId).value;
      api("/act_obj?" + new URLSearchParams(p));
      // the popup's buttons are a snapshot of the object's state at the click (Open stays "Open" after the door
      // opens); close it so a stale action is never offered — the side panel re-renders from /objects
      if (container.id === "ap-acts") hideActPopup();
    };
    container.appendChild(b);
  });
}
let _actsSig = null;
function renderActs(o) {
  const box = $("objacts"); if (!box) return;
  const sig = o.id + "|" + (o.actions || []).join(",");
  if (sig === _actsSig && box.querySelector("button")) return;   // same object, same skills — leave the buttons alone
  _actsSig = sig;
  actButtons(box, o, "p-open", "p-liquid");
}

/* floating action popup — pops up right at the click so actions aren't buried in the side panel */
function showActPopup(o, cx, cy, focusIt) {
  const pop = $("actpop"); if (!pop) return;
  const honest = honestOn();                       // honest mode: inspect only — object actions use simulator truth
  const acts = honest ? [] : (o.actions || []);
  let html = `<div class="ap-head"><b>${esc(o.type)}</b>` +
    `<span class="ap-x" role="button" tabindex="0" aria-label="close" title="close" onclick="hideActPopup()"` +
    ` onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();hideActPopup();}">×</span></div>`;
  if (honest) {
    html += `<p class="ap-note"><span class="badge truth">simulator truth</span> display only. ` +
      `Object actions are in Oracle mode; the honest agent never sees this.</p>`;
    if (o.flags || o.dist != null) html += `<p class="ap-note dim">${esc(o.dist != null ? o.dist + " m " : "")}${esc(o.flags || "")}</p>`;
    pop.innerHTML = html;
  } else {
    if (acts.includes("Open")) html += `<label class="ap-ctl">openness <input type="range" id="ap-open" min="0" max="1" step="0.1" value="1"></label>`;
    if (acts.includes("Fill")) html += `<label class="ap-ctl">liquid <select id="ap-liquid">${LIQUIDS.map(l => `<option>${esc(l)}</option>`).join("")}</select></label>`;
    html += `<div class="ap-acts" id="ap-acts"></div>`;
    pop.innerHTML = html;
    actButtons($("ap-acts"), o, "ap-open", "ap-liquid");
  }
  pop.hidden = false;
  const pw = pop.offsetWidth || 230, ph = pop.offsetHeight || 150;
  let x = (cx || 80) + 14, y = (cy || 80) + 8;
  x = Math.min(x, window.innerWidth - pw - 10); y = Math.min(y, window.innerHeight - ph - 10);
  pop.style.left = Math.max(8, x) + "px"; pop.style.top = Math.max(8, y) + "px";
  // opened from the keyboard: #actpop is the last node in the document, so leaving focus
  // on the (now rebuilt) row would put ~85 tab stops between the user and these buttons
  if (focusIt) { const first = pop.querySelector("button,[tabindex]"); if (first) first.focus(); }
}
function hideActPopup() { const p = $("actpop"); if (p) p.hidden = true; }

/* ----------------------------------------------------------- models */
let MODELS = [];
async function loadModels(preferred) {
  const sel = $("model"), ep = $("endpoint").value.trim();
  const prev = sel.value;                       // keep the current pick across a refresh
  setConn("", "loading models…");
  let d;
  try { d = await (await fetch("/models?endpoint=" + encodeURIComponent(ep))).json(); }
  catch (e) { d = { ok: false, models: [] }; }
  MODELS = d.models || [];
  sel.innerHTML = "";
  if (d.off) {
    sel.add(new Option("— LLM is off —", ""));
    setConn("", LLM_OFF_NOTE);
    return;
  }
  if (!MODELS.length) {
    sel.add(new Option("— no models found —", ""));
    setConn("", d.error ? "endpoint unreachable" : "no models");
    return;
  }
  MODELS.forEach(m => sel.add(new Option(m, m)));
  const want = MODELS.includes(prev) ? prev
             : MODELS.includes(preferred) ? preferred : MODELS[0];
  sel.value = want;
  setConn(want);
  if (!honestOn()) setText($("cmd-model"), want);
  if (window.HONEST) HONEST.models(MODELS);
}
$("model").addEventListener("change", () => {
  const m = $("model").value;
  setConn(m);
  if (!honestOn()) setText($("cmd-model"), m || "—");
  api("/setmodel?" + new URLSearchParams({ model: m, endpoint: $("endpoint").value }));
});
$("model-refresh").addEventListener("click", () => loadModels());
$("endpoint").addEventListener("change", () => loadModels());

/* ----------------------------------------------------------- LLM switch */
/* /status.llm is the truth; the chip only mirrors it. Off = the server never contacts Ollama, so the
   model list is not fetched and commands are refused here rather than silently dropped. */
const LLM_OFF_NOTE = "LLM off · drive by hand";
let LLM_ON = null;
function renderLlm(llm) {
  const c = $("chip-llm"); if (!c || c.classList.contains("busy")) return;
  const on = !!llm.on, was = LLM_ON;
  LLM_ON = on;
  c.classList.toggle("off", !on);
  c.setAttribute("aria-pressed", String(on));
  setText($("llm-state"), on ? "on" : "off");
  c.title = on ? "The LLM planner is ON (Ollama). Click to turn it off — manual driving keeps working."
               : "The LLM planner is OFF: nothing contacts Ollama. Click to turn it on (starts Ollama if needed)."
                 + (llm.note ? "\n" + llm.note : "");
  if (was === null || was === on) return;          // boot: loadModels() from /defaults already ran
  loadModels();                                    // off -> the server answers {off:true} and the note is set
}
$("chip-llm").addEventListener("click", async () => {
  const c = $("chip-llm"), want = !LLM_ON;
  c.classList.add("busy");
  setText($("llm-state"), want ? "starting…" : "stopping…");
  let d;
  try { d = await (await fetch("/llm?" + new URLSearchParams({ on: want ? "1" : "0", endpoint: $("endpoint").value }))).json(); }
  catch (e) { d = { ok: false, on: LLM_ON, error: "network" }; }
  c.classList.remove("busy");
  renderLlm({ on: d.on, note: d.error || "" });
  if (!d.ok) noteNetError("/llm", d.error);
});

/* ----------------------------------------------------------- chat */
const TEACH_RE = /^\s*(teach|correction|correct|remember|fact|lesson|spatial)\s*:/i;   // mirrors /cmd in webapp.py
async function send() {
  if (honestOn() && window.HONEST) { HONEST.run(); return; }   // Run and Enter follow the selected agent mode
  const text = $("cmd").value.trim(); if (!text) return;
  if (LLM_ON === false && !TEACH_RE.test(text)) {  // keep the text: the user only has to flip the chip and resend
    NET_ERR = { msg: "LLM is off — click the LLM chip in the topbar to plan commands", ts: Date.now() };
    renderConn();
    return;
  }
  $("cmd").value = "";
  api("/cmd?" + new URLSearchParams({ text, model: $("model").value, endpoint: $("endpoint").value }));
}
async function sendTeach() {
  const text = $("teach-text").value.trim(); if (!text) return;
  $("teach-text").value = "";
  api("/teach?" + new URLSearchParams({ text }));
}
function demoStart() {
  api("/demo?" + new URLSearchParams({ on: "1", command: $("demo-cmd").value.trim() }));
}
function demoStop(save) {
  api("/demo?" + new URLSearchParams({ on: "0", save: save ? "1" : "0", command: $("demo-cmd").value.trim() }));
}
function sendPerceptionFeedback() {
  const object = $("pf-obj").value.trim(); if (!object) return;
  const receptacle = $("pf-rec").value.trim();
  const bbox = $("pf-bbox").value.trim();
  api("/perception_feedback?" + new URLSearchParams({ object, receptacle, bbox }));
  $("pf-obj").value = ""; $("pf-rec").value = ""; $("pf-bbox").value = "";
}

/* ----------------------------------------------------------- camera click */
$("cam").addEventListener("click", async (e) => {
  if (e.shiftKey && honestOn()) {                  // shift-click = pickup with simulator truth: refused here, not just hidden
    toast("Shift-click pickup acts with simulator truth. Switch the agent to Oracle to use it.", "warn");
    return;
  }
  const r = e.target.getBoundingClientRect();
  const x = Math.min(.999, Math.max(0, (e.clientX - r.left) / r.width));
  const y = Math.min(.999, Math.max(0, (e.clientY - r.top) / r.height));
  let res;
  try {
    res = await (await fetch("/click?" + new URLSearchParams(
      { x: x.toFixed(3), y: y.toFixed(3), shift: e.shiftKey ? "1" : "0" }))).json();
  } catch (err) { noteNetError("/click", "network"); return; }
  const hit = res && res.hit;
  if (!hit) { setHudObj(""); hideActPopup(); return; }   // clicked empty space — nothing under the cursor
  // select the clicked object and POP UP its actions right at the click point (no longer buried in the side panel)
  SEL = hit.id;
  setHudObj(hit.type);
  let o = OBJS.find(ob => ob.id === hit.id);
  if (!o) { await loadObjs(); o = OBJS.find(ob => ob.id === hit.id); }
  if (!o) o = { id: hit.id, type: hit.type, actions: [] };   // fallback: at least Go To / Look At
  renderObjs(); renderActs(o);
  showActPopup(o, e.clientX, e.clientY);
});

/* ----------------------------------------------------------- segmented controls */
/* signature is FROZEN: markSeg("views"|"res"|"perc", attr, value) — see ui-contract.md §1b */
function markSeg(id, attr, val) {
  const box = $(id); if (!box) return;
  const btns = box.querySelectorAll("button");
  let any = false;
  btns.forEach(b => {
    const on = b.getAttribute(attr) === val;
    if (on) any = true;
    b.classList.toggle("on", on);
    b.setAttribute("aria-checked", on ? "true" : "false");
    b.tabIndex = on ? 0 : -1;                      // roving tabindex: the group is ONE tab stop
  });
  if (!any && btns.length) btns[0].tabIndex = 0;   // never strand the group outside the tab order
}
/* one delegated arrow-key handler per radiogroup (§10.9) */
document.querySelectorAll(".seg").forEach(seg => {
  seg.addEventListener("keydown", e => {
    const btns = Array.prototype.slice.call(seg.querySelectorAll("button"));
    const i = btns.indexOf(document.activeElement); if (i < 0) return;
    let j = null;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") j = (i + 1) % btns.length;
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") j = (i - 1 + btns.length) % btns.length;
    else if (e.key === "Home") j = 0;
    else if (e.key === "End") j = btns.length - 1;
    if (j === null) return;
    e.preventDefault(); btns[j].focus();
    if (seg.id === "agent-mode") return;           // switching the agent reloads the scene: Enter/Space only
    btns[j].click();
  });
});
$("views").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  const prev = lastView;
  markSeg("views", "data-view", b.dataset.view);
  api("/view?mode=" + b.dataset.view).then(r => { if (!r || !r.ok) markSeg("views", "data-view", prev); });  // refused: roll back
});
function setCamAr(w, h) {                          // §10.8 — .vp-media carries the stream's real aspect ratio
  const a = Number(w) / Number(h);
  if (isFinite(a) && a > 0) document.documentElement.style.setProperty("--cam-ar", String(a));
}
$("res").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  const prevRes = lastRes;
  markSeg("res", "data-res", b.dataset.res);
  const [w, h] = b.dataset.res.split("x");
  api(`/resolution?w=${w}&h=${h}`).then(r => {         // the aspect ratio follows the server, not the click
    if (r && r.ok) { setCamAr(w, h); return; }
    const m = /^\s*(\d+)\s*x\s*(\d+)\s*$/.exec(prevRes || "");
    if (m) markSeg("res", "data-res", m[1] + "x" + m[2]);
  });
});

/* ----------------------------------------------------------- nav + sliders + toggles */
document.querySelectorAll(".dpad button, button[data-act]").forEach(b => {
  if (b.dataset.act) b.onclick = () => {
    if (honestActing()) { toast("The agent is acting. Stop first.", "warn"); return; }
    markDriving(true); api("/action?name=" + b.dataset.act); markDriving(false);
  };
});
$("s-move").oninput = e => { setText($("v-move"), e.target.value); api("/setparam?move=" + e.target.value); };
$("s-rot").oninput = e => { setText($("v-rot"), e.target.value); api("/setparam?rotate=" + e.target.value); };
$("rec").onchange = e => api("/record?on=" + (e.target.checked ? "1" : "0"));
$("loop").onchange = e => api("/loop?on=" + (e.target.checked ? "1" : "0"));

/* game-mode keyboard drive — HOLD a movement key for smooth continuous motion (server steps it per frame) */
const DRIVE_KEYS = { w: "move_ahead", s: "move_back", a: "move_left", d: "move_right",
  arrowleft: "rotate_left", arrowright: "rotate_right", arrowup: "look_up", arrowdown: "look_down",
  q: "rotate_left", e: "rotate_right" };
const TAP_KEYS = { c: "crouch", g: "drop" };
const heldKeys = [];                                            // movement keys currently down (last = active dir)
let driveTimer = null, driveOffTimer = null;
function markDriving(on) {                                      // body[data-drive="on"] -> ◆ DRIVE badge + d-pad ring (§6)
  if (on) {
    if (driveOffTimer) { clearTimeout(driveOffTimer); driveOffTimer = null; }
    if (document.body.dataset.drive !== "on") document.body.dataset.drive = "on";
    return;
  }
  if (driveOffTimer) clearTimeout(driveOffTimer);
  driveOffTimer = setTimeout(() => { delete document.body.dataset.drive; driveOffTimer = null; }, 1500);
}
function curDrive() { return heldKeys.length ? DRIVE_KEYS[heldKeys[heldKeys.length - 1]] : null; }
function startDrive() {                                         // keepalive every 150ms refreshes the server watchdog
  markDriving(true);
  if (driveTimer) return;
  driveTimer = setInterval(() => { const a = curDrive(); if (a) api("/drive?action=" + a + "&on=1"); else stopDrive(); }, 150);
}
function stopDrive() {                                          // ALWAYS able to stop — clears timer + tells the server
  if (driveTimer) { clearInterval(driveTimer); driveTimer = null; }
  heldKeys.length = 0;
  markDriving(false);
  api("/drive?on=0");
}
function releaseKey(k) {
  const i = heldKeys.indexOf(k); if (i < 0) return;             // never emit /drive?on=0 for a key we never held
  heldKeys.splice(i, 1);
  if (!heldKeys.length) stopDrive();
}
window.addEventListener("keydown", e => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;               // §10.3 — Ctrl+S must reach the browser, not the robot
  if (!$("gamemode").checked) return;
  if (isTextTarget(e.target) || inWidget(e.target)) return;     // §10.4 — typing must never drive
  if (document.body.dataset.viewMode !== "run") return;         // keys drive only where the camera is
  const k = e.key.toLowerCase();
  if ((DRIVE_KEYS[k] || TAP_KEYS[k]) && honestActing()) {       // the lease is the agent's: no manual input
    e.preventDefault(); if (!e.repeat) toast("The agent is acting. Stop first.", "warn"); return;
  }
  if (k === "g" && honestOn()) {                                // drop is refused in honest mode (it would change the
    e.preventDefault(); if (!e.repeat) toast("Drop acts with simulator truth. Switch the agent to Oracle to use it.", "warn");
    return;                                                     // hand behind the agent's back)
  }
  if (DRIVE_KEYS[k]) {
    e.preventDefault();
    if (e.repeat) return;                                      // OS auto-repeat -> ignore (the keepalive interval drives)
    if (!heldKeys.includes(k)) heldKeys.push(k);
    api("/drive?action=" + DRIVE_KEYS[k] + "&on=1");           // immediate response, then the interval keeps it alive
    startDrive();
  } else if (TAP_KEYS[k] && !e.repeat) {
    e.preventDefault(); api("/action?name=" + TAP_KEYS[k]);
  }
});
window.addEventListener("keyup", e => {
  const k = (e.key || "").toLowerCase();
  // A key we are actually holding must ALWAYS be releasable — otherwise the 150ms keepalive drives
  // forever if a modifier goes down mid-hold or focus lands in a field before the release.
  if (heldKeys.includes(k)) { releaseKey(k); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;               // §10.3
  if (!$("gamemode").checked) return;                           // §10.2
  if (isTextTarget(e.target) || inWidget(e.target)) return;     // §10.2 — 'a' typed in a text box no longer cancels motion
  if (!DRIVE_KEYS[k]) return;
  releaseKey(k);
});
// belt-and-suspenders: losing focus / hiding the tab can swallow keyup — stop driving so nothing hangs
window.addEventListener("blur", stopDrive);
document.addEventListener("visibilitychange", () => { if (document.hidden) stopDrive(); });
$("gamemode").addEventListener("change", e => { if (!e.target.checked && driveTimer) stopDrive(); });

/* ----------------------------------------------------------- tabs + theme */
/* §10.10 — generic tablist: [role=tab][data-tab] drives [data-pane] visibility. */
/* A tab hidden by the current mode (.honest-only / .oracle-only) has no layout box: skip it. */
function tabVisible(t) { return !!(t && (t.offsetWidth || t.offsetHeight || t.getClientRects().length)); }
const _tabSelect = new Map();                       // tablist -> its select(tab, focus)
function selectFirstVisibleTab(list, force) {
  if (!list || !_tabSelect.has(list)) return;
  const tabs = Array.prototype.slice.call(list.querySelectorAll('[role="tab"][data-tab]'));
  const cur = tabs.find(t => t.getAttribute("aria-selected") === "true");
  if (!force && cur && tabVisible(cur)) return;     // the open tab exists in this mode too: keep it
  const first = tabs.find(tabVisible);
  if (first) _tabSelect.get(list)(first, false);
}
function initTabs() {
  document.querySelectorAll('[role="tablist"]').forEach(list => {
    const tabs = Array.prototype.slice.call(list.querySelectorAll('[role="tab"][data-tab]'));
    if (!tabs.length) return;
    const isViewSwitch = list.classList.contains("viewtabs");
    const select = (tab, focus) => {
      tabs.forEach(t => {
        const on = t === tab;
        t.setAttribute("aria-selected", on ? "true" : "false");
        t.tabIndex = on ? 0 : -1;
        const pane = document.querySelector('[data-pane="' + t.dataset.tab + '"]');
        if (pane) pane.hidden = !on;               // every #ev-*/#ds-* node stays in the DOM (§3.1)
      });
      if (isViewSwitch) {
        document.body.dataset.viewMode = tab.dataset.tab;
        placeCamera(tab.dataset.tab);
        if (tab.dataset.tab !== "run" && (driveTimer || heldKeys.length)) stopDrive();
      }
      list.dispatchEvent(new CustomEvent("tabchange", { detail: tab.dataset.tab }));
      if (focus) tab.focus();
    };
    _tabSelect.set(list, select);
    tabs.forEach(t => t.addEventListener("click", () => select(t, false)));
    list.addEventListener("keydown", e => {
      const vis = tabs.filter(tabVisible);           // arrow keys skip the tabs this mode hides
      const i = vis.indexOf(document.activeElement); if (i < 0) return;
      let j = null;
      if (e.key === "ArrowRight" || e.key === "ArrowDown") j = (i + 1) % vis.length;
      else if (e.key === "ArrowLeft" || e.key === "ArrowUp") j = (i - 1 + vis.length) % vis.length;
      else if (e.key === "Home") j = 0;
      else if (e.key === "End") j = vis.length - 1;
      if (j === null) return;
      e.preventDefault(); select(vis[j], true);
    });
    select(tabs.filter(t => t.getAttribute("aria-selected") === "true")[0] || tabs[0], false);
  });
}
/* ONE MJPEG <img>: the camera node moves into Setup while Setup is open, never duplicated (spec §4.6). */
function placeCamera(view) {
  const cam = $("viewport"), home = $("cam-home"), setup = $("cam-setup");
  if (!cam || !home || !setup) return;
  const want = view === "setup" ? setup : home;
  if (cam.parentNode !== want) want.appendChild(cam);
}
/* Theme: system (follows the OS) → light → dark, kept in localStorage and shared with the trace viewer.
   data-theme on <html> overrides the media query; no attribute = system. Restored pre-paint in <head>. */
const THEME_KEY = "thor.theme";
const THEMES = ["system", "light", "dark"];
function curTheme() { const t = document.documentElement.dataset.theme; return t === "light" || t === "dark" ? t : "system"; }
function applyTheme(t) {
  if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
  else delete document.documentElement.dataset.theme;
  setText($("theme-say"), t === "light" ? "Light" : t === "dark" ? "Dark" : "System");
  drawNavmap();                                    // the canvases are painted, not styled — repaint them
  if (window.HONEST) HONEST.redraw();
}
(function initTheme() {
  const btn = $("theme-btn"); if (!btn) return;
  btn.addEventListener("click", () => {
    const next = THEMES[(THEMES.indexOf(curTheme()) + 1) % THEMES.length];
    try { if (next === "system") localStorage.removeItem(THEME_KEY); else localStorage.setItem(THEME_KEY, next); } catch (e) {}
    applyTheme(next);
  });
  setText($("theme-say"), curTheme() === "light" ? "Light" : curTheme() === "dark" ? "Dark" : "System");
  try {                                            // "system" must follow the OS live, not only at load
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      if (curTheme() === "system") { drawNavmap(); if (window.HONEST) HONEST.redraw(); }
    });
  } catch (e) {}
})();

/* ----------------------------------------------------------- nav map + perception (Phase 4) */
let lastPose = {};
function scanScene() {                                          // server defaults: target=40 positions=24 states=on
  // The container pass opens every cabinet/fridge/drawer and photographs what is INSIDE — without it, anything
  // stored behind a door is never in the dataset at all. It roughly triples scan time, hence the toggle.
  const box = $("scan-containers");
  api("/scan?do_containers=" + (box && !box.checked ? "0" : "1"));
}
function trainScene() { api("/train"); }
function exploreScene() { api("/explore"); }                   // drive the whole reachable area -> full discovered map
function renderTrain(tr) {
  const el = $("train-stat"), btn = $("train-btn");
  const running = !!tr && tr.status === "running";
  if (btn) {                                                   // §10.13 — the button reports the job it started
    btn.disabled = running;
    if (running) btn.setAttribute("data-busy", ""); else btn.removeAttribute("data-busy");
  }
  if (!el) return;
  el.classList.toggle("bad", !!tr && tr.status === "error");
  if (!tr) { setText(el, ""); return; }
  let m = "detector: " + tr.status;
  if (tr.status === "done" && tr.weights_path) m += " ✓";
  if (tr.status === "done" && tr.metrics) {
    const map = tr.metrics["metrics/mAP50(B)"] ?? tr.metrics["metrics/mAP50-95(B)"];
    if (map != null) m += ` · mAP50 ${(+map).toFixed(3)}`;
    const nc = tr.per_class_ap ? Object.keys(tr.per_class_ap).length : 0;
    if (nc) m += ` · ${nc} cls`;
    if (tr.worst_recall && tr.worst_recall.length)
      m += " · worst recall " + tr.worst_recall.map(([c, r]) => `${c} ${(+r).toFixed(2)}`).join(", ");
  }
  if (tr.status === "error") m += " — " + (tr.error || "").slice(0, 40);
  setText(el, m);
}
$("perc").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  setPerception(b.dataset.perc);
});

/* ----------------------------------------------------------- reset
   `everything` is irreversible, so it is never one click away: the button arms, and the two scopes are the
   confirmation. No confirm() — a modal blocks the whole page (and, under browser automation, the extension). */
function armReset(on) {
  $("reset-arm").hidden = !on;
  $("reset-btn").setAttribute("aria-expanded", on ? "true" : "false");
  $("reset-btn").hidden = !!on;
  if (on) $("reset-session").focus();
}
async function doReset(scope) {
  armReset(false);
  const btns = ["reset-session", "reset-all"].map($);
  btns.forEach(b => b && (b.disabled = true));
  const res = await api("/reset?scope=" + encodeURIComponent(scope));
  btns.forEach(b => b && (b.disabled = false));
  // The transcript and plan are rendered from cached signatures; clear them or a now-empty log repaints as
  // "unchanged" and the panel keeps showing a run that no longer exists.
  _logSig = null; lastMotion = null; _apiWarned = null;
  Object.keys(RATED).forEach(k => delete RATED[k]);   // a wiped bandit has no decisions left to have rated
  renderLog([]); renderPlan(null);
  if (res && !res.ok) {
    let msg = "reset failed";
    try { msg = (await res.clone().json()).error || msg; } catch (e) {}
    noteNetError("/reset", msg);
  }
}
$("reset-btn").addEventListener("click", () => armReset(true));
$("reset-cancel").addEventListener("click", () => { armReset(false); $("reset-btn").focus(); });
$("reset-session").addEventListener("click", () => doReset("session"));
$("reset-all").addEventListener("click", () => doReset("all"));
$("resetbox").addEventListener("keydown", e => { if (e.key === "Escape") { armReset(false); $("reset-btn").focus(); } });

/* ----------------------------------------------------------- the senses switch (perception axis)
   ONE entry point for both controls: the prominent button under the composer and the #perc segmented
   control in Scene & setup. The server is the source of truth — poll() re-reads /status.perception and
   re-paints both, so a mode set from the other control (or a degraded fallback) always shows up here. */
function setPerception(mode) {
  markSeg("perc", "data-perc", mode);
  paintPerception({ perception: mode });          // optimistic, corrected by the next poll if the server disagrees
  api("/perception?mode=" + encodeURIComponent(mode));
}
function togglePerception() {
  setPerception($("perc-toggle").getAttribute("aria-pressed") === "true" ? "oracle" : "perceived");
}
function paintPerception(s) {
  const btn = $("perc-toggle"); if (!btn) return;
  const on = s.perception === "perceived";
  btn.setAttribute("aria-pressed", on ? "true" : "false");
  btn.classList.toggle("on", on);
  setText($("perc-toggle-text"), on ? "scene info: OFF — detector only" : "scene info: full (oracle)");
  const note = $("perc-note"); if (!note) return;
  // A degraded run is the one thing that must never be quiet: it means the planner was handed the oracle
  // list while this button still said "detector only".
  if (on && s.perception_degraded) {
    note.className = "sense-note bad";
    setText(note, "degraded → oracle: " + String(s.perception_degraded).slice(0, 90));
  } else if (on && s.perception_ready === false) {
    note.className = "sense-note bad";
    setText(note, "no trained detector for this scene — scan & learn, then train");
  } else if (on) {
    note.className = "sense-note";
    setText(note, "planner sees only YOLO + its own search");
  } else {
    note.className = "sense-note";
    setText(note, "");
  }
}

/* ----------------------------------------------------------- the two detectors (step 3)
   Primary YOLO26s on every frame; the ambiguity / inspection specialist YOLO26m takes over its 10 look-alike
   classes while the agent holds an object up (auto). In the "detect" view its boxes are amber and starred. */
$("insp").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  markSeg("insp", "data-insp", b.dataset.insp);
  api("/inspect?mode=" + encodeURIComponent(b.dataset.insp));
});
function paintDetectors(d) {
  const el = $("det-stack"); if (!el || !d) return;
  markSeg("insp", "data-insp", d.inspect_mode || "auto");
  if (!d.primary) { setText(el, "detectors: legacy single model (no step-3 weights for this scene)"); return; }
  const spec = d.specialist ? d.specialist.replace(".pt", "") : "not trained yet";
  const live = d.inspecting ? " — INSPECTING now" : (d.holding && d.inspect_mode === "auto" ? " — holding (switch to detect view)" : "");
  setText(el, "primary " + d.primary.replace(".pt", "") + " · inspection " + spec + live);
}

/* ----------------------------------------------------------- motion detail
   How finely the agent's OWN plans animate. The server renders one frame per generator yield, so this is
   the frame rate of the robot's movement, not a playback speed. */
$("motion").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  markSeg("motion", "data-motion", b.dataset.motion);
  api("/motion?speed=" + encodeURIComponent(b.dataset.motion));
});
let lastMotion = null;
function renderMotion(m) {
  if (!m) return;
  if (m.speed !== lastMotion) { lastMotion = m.speed; markSeg("motion", "data-motion", m.speed); }
  const chip = $("hud-motion"); if (!chip) return;
  const lab = (m.label || "").trim().replace(/_/g, " ");   // "move_ahead" is a wire name, not a caption
  chip.hidden = !lab;
  if (lab) setText(chip, lab);
}
async function drawNavmap() {
  if (document.hidden) return;                                 // §10.17
  const cv0 = $("navmap");
  if (!cv0 || !cv0.offsetWidth || honestOn()) return;          // closed tab / honest mode: no /navmap worker job
  let j;
  try { j = await (await fetch("/navmap")).json(); } catch (e) { return; }
  const cv = $("navmap"); if (!cv) return;
  if (!cv.offsetWidth) return;                                 // its tab is closed: nothing to paint
  // drawn at CSS size × devicePixelRatio so lines and labels stay crisp at any zoom (spec §7.3)
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth, cssH = Math.round(cssW * 0.78);
  if (cv.width !== Math.round(cssW * dpr) || cv.height !== Math.round(cssH * dpr)) {
    cv.width = Math.round(cssW * dpr); cv.height = Math.round(cssH * dpr); cv.style.height = cssH + "px";
  }
  const ctx = cv.getContext("2d"), W = cssW, H = cssH, pad = 14;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  // §10.12 — the canvas is the one surface CSS cannot theme; read the tokens instead of hard-coding them
  const cs = getComputedStyle(document.documentElement);
  const tok = (name, fb) => ((cs.getPropertyValue(name) || "").trim() || fb);
  ctx.fillStyle = tok("--well", "#0C1013"); ctx.fillRect(0, 0, W, H);
  const b = j.bounds || { minx: 0, maxx: 0, minz: 0, maxz: 0 };
  let minx = b.minx, maxx = b.maxx, minz = b.minz, maxz = b.maxz;
  const px = lastPose.x, pz = lastPose.z, hasA = typeof px === "number" && typeof pz === "number";
  if (hasA) { minx = Math.min(minx, px); maxx = Math.max(maxx, px); minz = Math.min(minz, pz); maxz = Math.max(maxz, pz); }
  const spanx = (maxx - minx) || 1, spanz = (maxz - minz) || 1;
  const sc = Math.min((W - 2 * pad) / spanx, (H - 2 * pad) / spanz);
  const X = x => pad + (x - minx) * sc, Y = z => H - pad - (z - minz) * sc;   // flip z so +z is up
  // oracle baseline. Was .45 alpha on --d-oracle = 1.78:1 against the canvas — the layer
  // the whole discovered-vs-oracle comparison exists for was effectively invisible.
  ctx.globalAlpha = .85;
  ctx.fillStyle = tok("--d-oracle", "#93A2B2");
  (j.oracle || []).forEach(o => { ctx.beginPath(); ctx.arc(X(o.x), Y(o.z), 1.5, 0, 7); ctx.fill(); });
  ctx.globalAlpha = 1;
  // discovered cells
  const cell = Math.max(3, (j.cell_size || 0.25) * sc);
  const COL = { free: tok("--d-free", "#8FEBC8"), blocked: tok("--d-blocked", "#C9704F"), unknown: tok("--d-unknown", "#59646E") };
  ctx.globalAlpha = 1;                                         // no alpha wash: the tokens are already ramped by lightness
  (j.cells || []).forEach(c => {
    ctx.fillStyle = COL[c.s] || COL.unknown;
    ctx.fillRect(X(c.x) - cell / 2, Y(c.z) - cell / 2, cell, cell);
  });
  ctx.globalAlpha = 1;
  // live agent marker + heading (yaw 0 = +z = up; clockwise)
  if (hasA) {
    const ax = X(px), ay = Y(pz), yaw = (lastPose.rot || 0) * Math.PI / 180;
    const agentCol = tok("--d-agent", "#FFB454");
    ctx.strokeStyle = agentCol; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ax + Math.sin(yaw) * 13, ay - Math.cos(yaw) * 13); ctx.stroke();
    ctx.fillStyle = agentCol; ctx.beginPath(); ctx.arc(ax, ay, 4, 0, 7); ctx.fill();
  }
  const cells = j.cells || [];
  const free = cells.filter(c => c.s === "free").length, blk = cells.filter(c => c.s === "blocked").length;
  setText($("navmap-stat"), `${free} free · ${blk} blocked · ${(j.oracle || []).length} oracle`);
}

/* ----------------------------------------------------------- scene + gallery */
function loadScene() { api("/scene?name=" + $("scene-sel").value); }
async function loadGallery() {
  const g = $("gallery"); g.innerHTML = "";
  try {
    const files = await (await fetch("/gallery")).json();
    files.forEach(n => {
      const im = document.createElement("img");
      im.src = "/cap_img?name=" + encodeURIComponent(n); im.title = n; im.alt = n; im.loading = "lazy";
      im.tabIndex = 0;
      im.onclick = () => window.open(im.src, "_blank");
      im.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); im.click(); } };
      g.appendChild(im);
    });
  } catch (e) {}
}

/* ----------------------------------------------------------- boot */
$("cmd").addEventListener("keydown", e => { if (e.key === "Enter" && !e.isComposing) send(); });
fetch("/defaults").then(r => r.json()).then(d => {
  $("endpoint").value = d.endpoint;
  setText($("cmd-model"), d.model || "—");
  const sc = $("scene-sel"); (d.scenes || []).forEach(n => sc.add(new Option(n, n)));
  LIQUIDS = d.liquids || ["water"];
  const lq = $("p-liquid"); LIQUIDS.forEach(n => lq.add(new Option(n, n)));
  loadModels(d.model);   // populate the dropdown from the endpoint, preselect the saved model
}).catch(() => noteNetError("/defaults", "network"));
window.addEventListener("keydown", e => { if (e.key === "Escape") hideActPopup(); });
$("cmd-model").addEventListener("click", () => {     // the model lives in Setup › Planner; this is a shortcut
  const t = document.querySelector('.viewtabs [data-tab="setup"]'); if (t) t.click();
  const f = honestOn() ? $("h-model") : $("model"); if (f) f.focus();
});
window.addEventListener("resize", () => { clearTimeout(window._rzT); window._rzT = setTimeout(() => {
  drawNavmap(); if (window.HONEST) HONEST.redraw(); }, 150); });
document.querySelector(".mindtabs").addEventListener("tabchange", e => {
  if (e.detail === "m-nav") drawNavmap();
  if (window.HONEST) HONEST.redraw();
});

/* ----------------------------------------------------------- Phase 8/9 evaluation panel */
const f3 = (v) => (v === null || v === undefined) ? "—" : Number(v).toFixed(3);

function goldText(g) {
  g = g || {};
  if (g.sequence) return "seq: " + g.sequence.join("  ;  ");
  if (g.actions) return "actions: " + g.actions.join(",  ");
  if (g.goal) return "goal: " + g.goal.map(a => a.type + "." + a.key + "=" +
    (a.value !== undefined ? a.value : ("∋" + a.contains))).join(",  ");
  return "";
}

/* §10.15b — SR / PC / TDS hero. The three numbers the thesis exists to report, tied to their level. */
const HERO = [["SR", "success rate", "L1", "SR"], ["PC", "procedural correctness", "L2", "PC"],
              ["TDS", "task decomposition", "L3", "TDS"]];
function renderHero(overall) {
  const box = $("ev-hero"); if (!box) return;
  if (!overall) { setHTML(box, "", SIG, "hero"); return; }
  setHTML(box, HERO.map(([key, sub, lvl, field]) => {
    const raw = overall[field];
    const v = (raw === null || raw === undefined) ? null : Number(raw);
    const filled = (v === null || !isFinite(v)) ? 0 : Math.max(0, Math.min(12, Math.ceil(v * 12)));
    let bars = "";
    for (let i = 0; i < 12; i++) bars += `<i class="${i < filled ? "f" : ""}"></i>`;
    return `<div class="metric" data-lvl="${lvl}"><div class="metric-key">${key}</div>` +
      `<div class="metric-val">${f3(raw)}</div>` +
      `<div class="metric-sub">${esc(sub)} · ${lvl}</div>` +
      `<div class="mbar" aria-hidden="true">${bars}</div></div>`;
  }).join(""), SIG, "hero");
}

function renderAgg(agg, config) {
  const box = $("ev-results");
  if (!agg) { setHTML(box, "", SIG, "agg"); renderHero(null); return; }
  renderHero(agg.overall);
  // skipped = not applicable in that scene (object absent / goal already true); never inside a mean
  const skip = b => b.n_skipped ? ` <span class="dim" title="not applicable in the scene: left out of every mean">+${b.n_skipped} skipped</span>` : "";
  const row = (name, b, lvl) => `<tr${lvl ? ` data-lvl="${esc(name)}"` : ""}><td>${esc(name)}</td><td class="n">${b.n}${skip(b)}</td>` +
    `<td class="n">${f3(b.SR)}</td><td class="n">${f3(b.PC)}</td><td class="n">${f3(b.TDS)}</td></tr>`;
  const sect = (title, obj, lvl) => {
    if (!obj || !Object.keys(obj).length) return "";
    let r = `<tr class="sec"><td colspan="5">${title}</td></tr>`;
    for (const k of Object.keys(obj).sort()) r += row(k, obj[k], lvl);
    return r;
  };
  const cfg = config ? `<div class="dim evcfg">${esc(config.models)} &middot; rag=${esc(config.rag)} ` +
    `&middot; loop=${esc(config.loop)} &middot; ${esc((config.scenes || []).join ? config.scenes.join(",") : config.scenes)}</div>` : "";
  setHTML(box, cfg + `<div class="evaltbl"><table><thead><tr><th>group</th><th class="n">n</th>` +
    `<th class="n">SR</th><th class="n">PC</th><th class="n">TDS</th></tr></thead><tbody>` +
    row("OVERALL", agg.overall) + sect("by level", agg.by_level, true) + sect("by model", agg.by_model) +
    sect("by scene", agg.by_scene) + sect("by lang", agg.by_lang) + `</tbody></table></div>`, SIG, "agg");
}

let _evShownTs = null;
function renderEval(ev) {
  if (!ev) return;
  const b = $("eval-badge");
  // NB: still a FULL className overwrite (ui-contract §5a) — markup must not park extra classes on #eval-badge
  if (ev.running) { setText(b, `running ${ev.done}/${ev.total || "?"}`); setClass(b, "dim eval-badge run"); }
  else if (ev.error) { setText(b, "error"); setClass(b, "dim eval-badge err"); }
  else if (ev.results) { setText(b, `done ${ev.results.n_records || ""}`); setClass(b, "dim eval-badge ok"); }
  else { setText(b, ""); setClass(b, "dim eval-badge"); }
  const prog = $("ev-prog");
  if (ev.running) {
    const pct = ev.total ? Math.round(100 * ev.done / ev.total) : 0;
    setHTML(prog, `<div class="pbar"><div class="pfill" style="width:${pct}%"></div></div>` +
      `<span class="dim">${ev.done}/${ev.total || "?"} records &middot; ${esc(ev.config ? ev.config.models : "")}</span>`,
      SIG, "prog");
  } else if (ev.error) { setHTML(prog, `<span class="bad">eval: ${esc(ev.error)}</span>`, SIG, "prog"); }
  else { setHTML(prog, "", SIG, "prog"); }
  // Render a just-finished run's results once (keyed on its start stamp), and refresh the file picker.
  if (!ev.running && ev.results && ev.results.aggregates && _evShownTs !== ev.started) {
    _evShownTs = ev.started;
    renderAgg(ev.results.aggregates, ev.results.config);
    loadEvalResults();
  }
}

async function runEval() {
  const q = new URLSearchParams({
    models: $("ev-models").value || "gemma3:4b", rag: $("ev-rag").value, loop: $("ev-loop").value,
    lang: $("ev-lang").value, scenes: $("ev-scenes").value, limit: $("ev-limit").value,
    dataset: $("ev-dataset").value,
  });
  let r;
  try { r = await (await fetch("/eval_run?" + q)).json(); }
  catch (e) { noteNetError("/eval_run", "network"); SIG.prog = null; $("ev-prog").innerHTML = `<span class="bad">could not reach the server</span>`; return; }
  if (!r.ok) { SIG.prog = null; $("ev-prog").innerHTML = `<span class="bad">${esc(r.error || "failed to start")}</span>`; }
}

async function loadEvalResults() {
  const file = $("ev-file").value || "";
  let r;
  try { r = await (await fetch("/eval_results?file=" + encodeURIComponent(file))).json(); }
  catch (e) { noteNetError("/eval_results", "network"); return; }
  if (r.files) {
    $("ev-file").innerHTML = r.files.length
      ? r.files.map(x => `<option ${x.file === r.selected ? "selected" : ""}>${esc(x.file)}</option>`).join("")
      : `<option value="">no result files yet</option>`;
  }
  if (r.aggregates) renderAgg(r.aggregates, r.config);
  else if (r.error) { SIG.agg = null; $("ev-results").innerHTML = `<span class="bad">${esc(r.error)}</span>`; renderHero(null); }
}

async function loadDataset() {
  const q = new URLSearchParams({ scene: $("ds-scene").value, level: $("ds-level").value, lang: $("ds-lang").value, limit: "300" });
  let r;
  try { r = await (await fetch("/dataset?" + q)).json(); }
  catch (e) { noteNetError("/dataset", "network"); $("ds-list").innerHTML = `<span class="bad">could not reach the server</span>`; return; }
  if (r.error) { $("ds-list").innerHTML = `<span class="bad">${esc(r.error)}</span>`; return; }
  const c = r.counts;
  setText($("ds-counts"), `${c.total} records · L1 ${c.by_level.L1 || 0} / L2 ${c.by_level.L2 || 0} / ` +
    `L3 ${c.by_level.L3 || 0} · ${Object.keys(c.by_scene).length} scenes · showing ${r.shown}`);
  $("ds-list").innerHTML = r.items.map(it =>
    `<div class="dsrow"><span class="tag lv lv-${esc(it.level)}">${esc(it.level)}</span><span class="tag lang">${esc(it.lang)}</span>` +
    `<span class="tag sc">${esc((it.scene || "").replace("FloorPlan", "FP"))}</span> ` +
    `<span class="cmd" dir="auto">${esc(it.command)}</span><div class="dim gold">${esc(goldText(it.gold))}</div></div>`).join("");
}

$("ev-run").addEventListener("click", runEval);

/* ----------------------------------------------------------- general instruction set (editable) */
/* One row = one instruction in BOTH languages. Goals and plans travel as one-line text -- `Apple in Fridge`,
   `fill Mug = water` -- and the SERVER parses and validates them (sim/general_set.py), so the rules live in one
   place and a rejected save comes back naming the line that is wrong. */
const GEN = { items: [], editing: null, loaded: false, level: "", undo: null, undoTimer: null };

function genRowHTML(it) {
  const ap = it.applies || {};
  let apTag = "";
  if (ap.total) {
    const all = ap.n === ap.total;
    apTag = `<span class="tag ${all ? "ap-all" : "ap-some"}" title="${all
      ? "every object this instruction needs exists in all scanned kitchens"
      : "skipped (not failed) where the kitchen has no " + esc((ap.missing || []).join(", "))}">` +
      `${ap.n}/${ap.total} kitchens${all ? "" : " · no " + esc((ap.missing || []).join(", "))}</span>`;
  }
  const pre = it.presatisfied
    ? `<span class="tag ap-some" title="the goal is already true at the start there, so the record is skipped">starts true in ${it.presatisfied}</span>` : "";
  const line = (k, v) => v ? `<b>${k}</b><span>${esc(v.replace(/\n/g, "  ;  "))}</span>` : "";
  return `<div class="genrow${it.reviewed ? " checked" : ""}" data-id="${esc(it.id)}">` +
    `<div class="genrow-head"><span class="tag lv lv-${esc(it.level)}">${esc(it.level)}</span>` +
    `<span class="genid">${esc(it.id)}</span>${apTag}${pre}` +
    `<span class="genrow-acts">` +
    `<button type="button" class="ghost" data-gen="check" aria-pressed="${it.reviewed ? "true" : "false"}" ` +
    `title="mark this instruction as checked — both languages read and correct">${it.reviewed ? "✓ checked" : "check"}</button>` +
    `<button type="button" class="ghost" data-gen="edit">edit</button>` +
    `<button type="button" class="ghost" data-gen="dup" title="start a new instruction from a copy of this one">duplicate</button>` +
    `<button type="button" class="ghost" data-gen="del">delete</button></span></div>` +
    `<div class="genpair"><div class="gen-en" lang="en" dir="ltr">${esc(it.en)}</div>` +
    `<div class="gen-fa" lang="fa" dir="rtl">${esc(it.fa)}</div></div>` +
    `<div class="gengold">${line("setup", it.setup_text)}${line("plan", it.plan_text)}${line("goal", it.goal_text)}</div></div>`;
}

function genEditorHTML(it, asNew) {
  const isNew = !it || !!asNew;                      // asNew: "duplicate" — a prefilled form that ADDS
  it = it || { id: "", level: GEN.level || "L1", en: "", fa: "", setup_text: "", plan_text: "", goal_text: "" };
  const lv = ["L1", "L2", "L3"].map(l => `<option${l === it.level ? " selected" : ""}>${l}</option>`).join("");
  return `<form class="genedit" data-original="${esc(isNew ? "" : it.id)}" autocomplete="off">` +
    `<div class="genedit-top"><label><span class="fld-label">id</span>` +
    `<input name="id" value="${esc(it.id)}" placeholder="g-l1-open-oven" spellcheck="false" dir="ltr"></label>` +
    `<label><span class="fld-label">level</span><select name="level">${lv}</select></label></div>` +
    `<div class="genedit-pair"><label><span class="fld-label">English</span>` +
    `<textarea name="en" rows="2" lang="en" dir="ltr">${esc(it.en)}</textarea></label>` +
    `<label><span class="fld-label">فارسی</span>` +
    `<textarea name="fa" rows="2" lang="fa" dir="rtl">${esc(it.fa)}</textarea></label></div>` +
    `<div class="genedit-gold"><label><span class="fld-label">setup · runs before the agent plans</span>` +
    `<textarea name="setup_text" class="code" rows="3" spellcheck="false">${esc(it.setup_text)}</textarea></label>` +
    `<label><span class="fld-label">reference plan · gold labels come from this</span>` +
    `<textarea name="plan_text" class="code" rows="3" spellcheck="false">${esc(it.plan_text)}</textarea></label>` +
    `<label><span class="fld-label">goal · what must be true at the end</span>` +
    `<textarea name="goal_text" class="code" rows="3" spellcheck="false">${esc(it.goal_text)}</textarea></label></div>` +
    `<div class="genhint">one per line &nbsp;·&nbsp; step: <code>pick_and_place Apple -&gt; Fridge</code> ` +
    `<code>fill Mug = water</code> <code>toggle_on Toaster</code> &nbsp;·&nbsp; goal: <code>Apple in Fridge</code> ` +
    `<code>Fridge.isOpen = false</code> <code>holding Knife</code> <code>any Cabinet.isOpen = true</code> ` +
    `<code>all Drawer.isOpen = false</code> <code>Fridge.distance &lt;= 1.5</code></div>` +
    `<div class="genmsg"></div>` +
    `<div class="row"><button type="submit" class="primary">${isNew ? "add instruction" : "save changes"}</button>` +
    `<button type="button" class="ghost" data-gen="cancel">cancel</button>` +
    `<span class="genhint">Ctrl+Enter saves · Esc cancels</span></div></form>`;
}

function renderGeneral() {
  const list = $("gen-list");
  if (!GEN.items.length) { list.innerHTML = `<span class="dim">no instructions match</span>`; return; }
  list.innerHTML = GEN.items.map(it => it.id === GEN.editing ? genEditorHTML(it) : genRowHTML(it)).join("");
}

const GEN_LEVEL_NAMES = { "": "all", L1: "L1 atomic", L2: "L2 sequential", L3: "L3 abstract" };

function renderGenCounts(c, shown) {
  const lv = c.by_level || {}, ck = c.checked_by_level || {};
  setText($("gen-counts"), `${c.total} instructions × 2 languages = ${c.total * 2} records · showing ${shown}` +
    (c.kitchens ? ` · checked against ${c.kitchens} kitchens` : ""));
  setText($("gen-tabcount"), String(c.total));
  document.querySelectorAll("#gen-levels button").forEach(b => {
    const l = b.dataset.level;
    b.innerHTML = `${esc(GEN_LEVEL_NAMES[l])}<i>${l ? (lv[l] || 0) : c.total}</i>`;
    b.title = l ? `${ck[l] || 0} of ${lv[l] || 0} checked` : `${c.checked} of ${c.total} checked`;
    b.classList.toggle("on", l === GEN.level);
  });
  $("gen-prog").hidden = !c.total;
  $("gen-prog-fill").style.width = (c.total ? Math.round(100 * c.checked / c.total) : 0) + "%";
  setText($("gen-prog-text"), `${c.checked} of ${c.total} checked` +
    (c.checked < c.total ? ` · ${c.total - c.checked} to go` : " · all done"));
  const opt = $("ev-dataset").querySelector('option[value="general"]');
  if (opt) opt.textContent = `general instructions · ${c.total * 2} records`;
}

async function loadGeneral() {
  const q = new URLSearchParams({ level: GEN.level, q: $("gen-q").value, show: $("gen-show").value });
  let r;
  try { r = await (await fetch("/general?" + q)).json(); }
  catch (e) { noteNetError("/general", "network"); $("gen-list").innerHTML = `<span class="bad">could not reach the server</span>`; return; }
  if (r.error) { $("gen-list").innerHTML = `<span class="bad">${esc(r.error)}</span>`; return; }
  GEN.items = r.items; GEN.loaded = true;
  renderGenCounts(r.counts, r.shown);
  renderGeneral();
}

function genMessages(form, res) {
  const box = form.querySelector(".genmsg");
  box.innerHTML = (res.errors || []).map(e => `<span class="bad">${esc(e)}</span>`).join("") +
    (res.warnings || []).map(w => `<span class="warn">${esc(w)}</span>`).join("");
}

async function genPost(path, body) {
  try {
    return await (await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body) })).json();
  } catch (e) { noteNetError(path, "network"); return { ok: false, errors: ["could not reach the server"] }; }
}

function genToast(html, ms) {
  const t = $("gen-toast");
  clearTimeout(GEN.undoTimer);
  if (!html) { t.hidden = true; t.innerHTML = ""; return; }
  t.innerHTML = html; t.hidden = false;
  GEN.undoTimer = setTimeout(() => { t.hidden = true; GEN.undo = null; }, ms || 9000);
}

function genCloseEditors() { GEN.editing = null; $("gen-new").innerHTML = ""; renderGeneral(); }

$("gencard").addEventListener("submit", async e => {
  const form = e.target.closest(".genedit"); if (!form) return;
  e.preventDefault();
  const body = { original_id: form.dataset.original || null };
  new FormData(form).forEach((v, k) => { body[k] = v; });
  const btn = form.querySelector('button[type="submit"]'); btn.disabled = true;
  const res = await genPost("/general_save", body);
  btn.disabled = false;
  if (!res.ok) { genMessages(form, res); return; }
  GEN.editing = null; $("gen-new").innerHTML = "";
  await loadGeneral();
  const row = document.querySelector(`.genrow[data-id="${CSS.escape(res.item.id)}"]`);
  if (row) {
    row.scrollIntoView({ block: "nearest" });
    // a saved instruction may carry warnings (e.g. "already true at the start in 5 kitchens") — keep them visible
    if ((res.warnings || []).length) row.insertAdjacentHTML("beforeend", `<div class="genmsg">` +
      res.warnings.map(w => `<span class="warn">${esc(w)}</span>`).join("") + `</div>`);
  }
});

$("gencard").addEventListener("click", async e => {
  const b = e.target.closest("[data-gen]"); if (!b) return;
  const act = b.dataset.gen;
  if (act === "cancel") { genCloseEditors(); return; }
  const row = b.closest(".genrow"); if (!row) return;
  const id = row.dataset.id;
  const it = GEN.items.find(x => x.id === id);
  if (act === "edit") { GEN.editing = id; $("gen-new").innerHTML = ""; renderGeneral(); return; }
  if (act === "dup" && it) {
    GEN.editing = null; renderGeneral();
    $("gen-new").innerHTML = genEditorHTML(Object.assign({}, it, { id: it.id + "-copy" }), true);
    const f = $("gen-new").querySelector('input[name="id"]');
    $("gen-new").scrollIntoView({ block: "nearest" }); f.focus(); f.select();
    return;
  }
  if (act === "check" && it) {
    b.disabled = true;
    const res = await genPost("/general_review", { id, reviewed: !it.reviewed });
    b.disabled = false;
    if (res.ok) await loadGeneral();
    return;
  }
  if (act === "del") {
    // two clicks, no modal: the first arms the button for 4 s, the second deletes (same idea as topbar reset)
    if (!b.classList.contains("armed")) {
      b.classList.add("armed"); b.textContent = "really delete?";
      setTimeout(() => { if (b.isConnected) { b.classList.remove("armed"); b.textContent = "delete"; } }, 4000);
      return;
    }
    b.disabled = true;
    const res = await genPost("/general_delete", { id });
    if (!res.ok) { b.disabled = false; b.textContent = (res.errors || ["failed"])[0]; return; }
    GEN.undo = it || null;
    await loadGeneral();
    if (GEN.undo) genToast(`deleted <code>${esc(id)}</code> <button type="button" class="ghost" id="gen-undo">undo</button>`);
  }
});

// undo = re-add the instruction exactly as it was (the audit log records both the delete and the re-add)
$("gen-toast").addEventListener("click", async e => {
  if (!e.target.closest("#gen-undo") || !GEN.undo) return;
  const it = GEN.undo; GEN.undo = null;
  const res = await genPost("/general_save", { id: it.id, level: it.level, en: it.en, fa: it.fa,
    setup_text: it.setup_text, plan_text: it.plan_text, goal_text: it.goal_text });
  if (res.ok && it.reviewed) await genPost("/general_review", { id: it.id, reviewed: true });
  genToast(res.ok ? `restored <code>${esc(it.id)}</code>` : esc((res.errors || ["could not restore"])[0]), 3500);
  await loadGeneral();
});

// Ctrl+Enter saves, Esc cancels — 288 rows is a lot of reaching for the mouse
$("gencard").addEventListener("keydown", e => {
  const form = e.target.closest(".genedit"); if (!form) return;
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); form.requestSubmit(); }
  else if (e.key === "Escape") { e.preventDefault(); genCloseEditors(); }
});

$("gen-add").addEventListener("click", () => {
  GEN.editing = null; renderGeneral();
  $("gen-new").innerHTML = genEditorHTML(null);
  $("gen-new").querySelector('input[name="id"]').focus();
});
$("gen-levels").addEventListener("click", e => {
  const b = e.target.closest("button[data-level]"); if (!b) return;
  GEN.level = b.dataset.level; loadGeneral();
});
$("gen-show").addEventListener("change", loadGeneral);
let _genQTimer = null;
$("gen-q").addEventListener("input", () => { clearTimeout(_genQTimer); _genQTimer = setTimeout(loadGeneral, 220); });
$("ev-dataset").addEventListener("change", () => {
  $("ev-scenes").placeholder = $("ev-dataset").value === "general" ? "FloorPlan1  (comma-separated; each runs the whole set)" : "all in dataset";
});
// load when the Evaluate view is first opened, not at boot: 288 rows nobody is looking at are not worth the parse
document.querySelector('.viewtabs [data-tab="eval"]').addEventListener("click", () => { if (!GEN.loaded) loadGeneral(); });
document.querySelector('[data-tab="ev-instr"]').addEventListener("click", () => { if (!GEN.loaded) loadGeneral(); });
// the scene-grounded browser used to need its "load" button pressed; open the tab and it is simply there
document.querySelector('[data-tab="ev-scene"]').addEventListener("click", () => { if (!$("ds-list").children.length) loadDataset(); });

/* ----------------------------------------------------------- timers */
initTabs();
setInterval(poll, 400); poll();                    // keeps running while hidden — it is what detects reconnection
// one request of each in flight at a time: on a slow link (the Windows -> WSL NAT can stall a connect for
// seconds) un-guarded 1.5 s polls pile up and take every one of the browser's ~6 sockets for this origin
const oneAtATime = (fn) => { let busy = false; return async () => { if (busy) return; busy = true; try { await fn(); } finally { busy = false; } }; };
const loadObjsOnce = oneAtATime(loadObjs), drawNavmapOnce = oneAtATime(drawNavmap);
setInterval(loadObjsOnce, 1500); loadObjsOnce();
setInterval(drawNavmapOnce, 1500); drawNavmapOnce();
// the state chip may be cut with an ellipsis at laptop width: keep its full text as the tooltip
(() => { const t = $("tb-state"); if (!t) return;
  new MutationObserver(() => { t.parentElement.title = t.textContent; }).observe(t, { childList: true, characterData: true, subtree: true }); })();
// the camera's Display menu (and any other <details class="menu">) closes on a click outside it or on Escape
document.addEventListener("click", (e) => {
  document.querySelectorAll("details.menu[open]").forEach(d => { if (!d.contains(e.target)) d.open = false; });
});
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  document.querySelectorAll("details.menu[open]").forEach(d => { d.open = false; const s = d.querySelector("summary"); if (s) s.focus(); });
});
document.addEventListener("visibilitychange", () => { if (!document.hidden) { loadObjs(); drawNavmap(); } });
loadGallery();
loadEvalResults();
