"use strict";
/* Honest mode — the console side of sim/honest_console.py (docs/console-v2-spec.md §2.6, §2.7, §3).
   Display only: everything here reads /honest (the agent's own beliefs, plan, map and audit) or /status, and
   sends the six honest routes. Nothing in this file reaches the agent's decisions.
   Loaded after app.js; app.js calls window.HONEST.{onStatus, run, redraw, models}. */
(function () {
  const ACTIVE = ["prepare", "sweep", "planning", "executing", "checking", "stopping"];
  const FINAL = ["done", "failed", "rejected", "cancelled"];
  const CAUSE_WORDS = {
    not_detected: "not seen by the camera", wrong_target: "found the wrong object",
    unreachable: "could not get there", action_refused: "the simulator refused the action",
    state_unrecognised: "could not tell its state", retry_exhausted: "retries used up",
    episode_budget: "primitive budget used up",
  };
  const PHASE_WORDS = {
    idle: "ready", prepare: "getting ready", sweep: "looking around", planning: "Gemma is planning",
    executing: "executing", checking: "checking the goal (report only)", stopping: "Stopping…",
    done: "done", failed: "failed", rejected: "plan rejected, not executed", cancelled: "stopped",
  };
  const H = {
    snap: null, rev: -1, fetching: false, again: false, fetchedAt: 0, seq: 0, applied: 0,
    gridRev: -1, grid: null, gridKey: "",
    view: null, viewTicket: null, viewReq: 0,           // a reopened earlier ticket, held apart from live state
    pipSeq: -1, pipOn: true,
    picked: null,                                        // {id, lang, text, level} bound to the exact text
    setItems: null, setLang: "en",
    sort: { key: "step", dir: -1 }, openSteps: new Set(),
    printed: "", switching: false, stopping: false, lastAvail: 0, models: [],
    zoom: 1, panX: 0, panY: 0, mapGeom: null, mapPts: [],
  };
  const $ = (id) => document.getElementById(id);
  const isActive = (p) => ACTIVE.indexOf(p) >= 0;
  const isFinal = (p) => FINAL.indexOf(p) >= 0;
  const bdi = (v) => `<bdi>${esc(v)}</bdi>`;
  const fa = (lang, text) => lang === "fa" || /[؀-ۿ]/.test(text || "");

  /* ------------------------------------------------------------------ fetching */
  function fetchT(url, ms) {                               // a hung request must not block every later refresh
    const ac = new AbortController();
    const t = setTimeout(() => ac.abort(), ms || 8000);
    return fetch(url, { signal: ac.signal }).finally(() => clearTimeout(t));
  }
  async function fetchHonest(full) {
    if (H.fetching) { H.again = true; return; }
    H.fetching = true;
    const seq = ++H.seq;
    let d = null;
    try {
      const r = await fetchT("/honest" + (full ? "?full=1" : ""), 8000);
      d = await r.json();
    } catch (e) { d = null; }
    H.fetching = false;
    // session identity first: a new session (or a restarted server) starts its revisions again from 0
    const sessionChanged = !!d && (!H.snap || (d.session && d.session.id) !== (H.snap.session && H.snap.session.id));
    if (d && d.ok !== false && seq >= H.applied && (H.snap == null || sessionChanged || !d.on || d.rev >= H.snap.rev || d.on !== H.snap.on)) {
      H.applied = seq;
      if (sessionChanged) {
        H.grid = null; H.gridRev = -1; H.viewReq++; H.view = null; H.viewTicket = null; H.printed = "";
        H.openSteps.clear(); H.pipSeq = -1;
      }
      const hadModel = H.snap && H.snap.model;
      H.snap = d; H.rev = d.rev; H.fetchedAt = Date.now();
      if (!H.modelUser && d.model && d.model !== hadModel) {        // the honest default (Gemma 2B) until the user picks
        models(H.models);
        if (honestOn()) setText($("cmd-model"), $("h-model").value || d.model);
      }
      if (d.map && d.map.free) { H.grid = d.map.free; H.gridRev = d.map.rev; }
      render();
      if (d.on && d.map && (H.grid == null || d.map.rev !== H.gridRev) && !full) { fetchHonest(true); return; }
    } else if (!d) {
      H.rev = -1;                                         // failed: the next status tick retries
    }
    if (H.again) { H.again = false; fetchHonest(false); }
  }

  function onStatus(s) {
    const hs = s.honest || {};
    const now = Date.now();
    if (!hs.on) {
      if (H.snap && H.snap.on) { H.snap = Object.assign({}, H.snap, { on: false }); }
      if (now - H.lastAvail > 5000) { H.lastAvail = now; fetchHonest(false); }   // keep "available?" current
      paintAvail();
      return;
    }
    if (hs.rev !== H.rev) fetchHonest(false);
    tickStage(s);
  }

  /* ------------------------------------------------------------------ mode switch */
  async function switchMode(mode) {
    const want = mode === "honest";
    if (H.switching || want === honestOn()) { markSeg("agent-mode", "data-amode", honestOn() ? "honest" : "oracle"); return; }
    if (want && H.snap && H.snap.available && !H.snap.available.ok) {
      toast(H.snap.available.why || "honest mode is not available", "warn");
      markSeg("agent-mode", "data-amode", "oracle");
      return;
    }
    H.switching = true;
    const seg = $("agent-mode");
    seg.classList.add("is-busy");
    seg.querySelectorAll("button").forEach(b => { b.disabled = true; });
    setText($("h-avail"), want ? "Switching to Honest: the scene reloads once with the depth sensor on…"
                               : "Switching to Oracle: the sensors go back to what the view needs…");
    let r = null, d = null;
    try { r = await fetch("/honest?on=" + (want ? "1" : "0")); d = await r.json(); } catch (e) { d = { ok: false, error: "network" }; }
    H.switching = false;
    seg.classList.remove("is-busy");
    seg.querySelectorAll("button").forEach(b => { b.disabled = false; });
    setText($("h-avail"), "");
    if (!r || !r.ok || d.ok === false) {
      toast("Could not switch: " + ((d && d.error) || "HTTP " + (r && r.status)), "bad", 9000);
      markSeg("agent-mode", "data-amode", honestOn() ? "honest" : "oracle");
      return;
    }
    toast(want ? "Honest mode on. Fresh world, empty hand." : "Back to the oracle agent.", "ok");
    if (want) { H.rev = -1; fetchHonest(true); loadSet(); }
    poll();
  }
  $("agent-mode").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-amode]"); if (!b) return;
    switchMode(b.dataset.amode);
  });

  function paintAvail() {
    const el = $("h-avail"); if (!el || H.switching) return;
    const a = H.snap && H.snap.available;
    const btn = document.querySelector('#agent-mode [data-amode="honest"]');
    if (a && !a.ok) {
      setText(el, honestOn() ? "" : "Honest mode unavailable: " + (a.why || "?"));
      if (btn) btn.title = "Unavailable: " + (a.why || "?");
    } else {
      setText(el, "");
      if (btn) btn.title = "FloorPlan1 only. The agent decides from its camera + depth, its own walked map and its memory. Switching reloads the scene once.";
    }
  }

  /* ------------------------------------------------------------------ run / stop / look / forget */
  async function call(path, okMsg) {
    let r, d;
    try { r = await fetch(path); d = await r.json(); } catch (e) { toast("request failed · " + path.split("?")[0] + " (network)", "bad"); return null; }
    if (!r.ok || d.ok === false) { toast(d.error || ("HTTP " + r.status), r.status === 409 ? "warn" : "bad", 8000); return null; }
    if (okMsg) toast(okMsg, "ok", 2500);
    fetchHonest(false);
    return d;
  }
  async function run() {
    const box = $("cmd");
    const text = box.value.trim();
    if (!text) { box.focus(); return; }
    if (honestActing()) { toast("The agent is acting. Stop first.", "warn"); return; }
    const p = H.picked && H.picked.text === text ? H.picked : null;
    const q = new URLSearchParams({
      text, item: p ? p.id : "", lang: p ? p.lang : (fa("", text) ? "fa" : "en"),
      model: $("h-model").value || (H.snap && H.snap.model) || "", endpoint: $("endpoint").value,
    });
    backToLive();
    const d = await call("/honest_run?" + q);
    if (d) { box.value = ""; unpick(); H.stopping = false; }
  }
  $("h-stop").addEventListener("click", async () => {
    H.stopping = true; paintButtons();
    const d = await call("/honest_stop");
    if (!d || !d.stopping) { H.stopping = false; paintButtons(); }
  });
  $("h-look").addEventListener("click", () => { backToLive(); call("/honest_look"); });
  $("h-forget").addEventListener("click", () => call("/honest_forget", "The next instruction starts a new memory episode. The hand is kept."));

  /* ------------------------------------------------------------------ From the set */
  async function loadSet(force) {
    if (H.setItems && !force) return;
    try {
      const d = await (await fetch("/general")).json();
      H.setItems = (d.items || []);
    } catch (e) { H.setItems = null; setHTML($("h-setlist"), `<span class="bad">could not load the set</span>`, SIG, "hset"); return; }
    paintSet();
  }
  function fp1Note(it) {
    const ap = it.applies || {};
    if ((ap.missing_in || []).indexOf("FloorPlan1") >= 0) return "not in FloorPlan1";
    if ((it.presatisfied_in || []).indexOf("FloorPlan1") >= 0) return "already true at the start in FloorPlan1";
    return "";
  }
  function paintSet() {
    const list = $("h-setlist"); if (!list || !H.setItems) return;
    const q = ($("h-setq").value || "").trim().toLowerCase();
    const lang = H.setLang;
    const rows = H.setItems.filter(it => !q || it.id.indexOf(q) >= 0 || (it.en || "").toLowerCase().indexOf(q) >= 0 || (it.fa || "").indexOf(q) >= 0)
      .slice(0, 160).map(it => {
        const setup = (it.setup || []).length > 0;
        const fp = fp1Note(it);
        const dis = setup || fp ? " disabled" : "";
        const why = setup ? "needs setup — run via step4_gemma_trial" : fp;
        const text = lang === "fa" ? it.fa : it.en;
        return `<button type="button" class="setrow" data-id="${esc(it.id)}"${dis}>` +
          `<span class="lvl" data-lvl="${esc(it.level)}">${esc(it.level)}</span>` +
          `<span class="setrow-t" dir="${lang === "fa" ? "rtl" : "ltr"}" lang="${lang}">${esc(text)}</span>` +
          `<span class="setrow-id dim">${bdi(it.id)}${why ? " · " + esc(why) : ""}</span></button>`;
      });
    setHTML(list, rows.join("") || `<span class="dim">nothing matches</span>`, SIG, "hset");
  }
  function openSet(open) {
    const pop = $("h-setpop"), b = $("h-set");
    pop.hidden = !open; b.setAttribute("aria-expanded", open ? "true" : "false");
    if (open) { paintSet(); loadSet(true); $("h-setq").focus(); }   // fresh: the set may have been edited
  }
  $("h-set").addEventListener("click", () => openSet($("h-setpop").hidden));
  $("h-setq").addEventListener("input", paintSet);
  $("h-setlang").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-lang]"); if (!b) return;
    H.setLang = b.dataset.lang; markSeg("h-setlang", "data-lang", H.setLang); paintSet();
  });
  $("h-setlist").addEventListener("click", (e) => {
    const b = e.target.closest(".setrow"); if (!b || b.disabled) return;
    const it = (H.setItems || []).find(x => x.id === b.dataset.id); if (!it) return;
    const text = (H.setLang === "fa" ? it.fa : it.en) || "";
    $("cmd").value = text;
    H.picked = { id: it.id, lang: H.setLang, text: text.trim(), level: it.level };
    paintPicked();
    openSet(false);
    $("cmd").focus();
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("h-setpop").hidden) { openSet(false); $("h-set").focus(); } });
  document.addEventListener("click", (e) => { if (!$("h-setpop").hidden && !e.target.closest("#h-setbox")) openSet(false); });
  function unpick() { H.picked = null; paintPicked(); }
  function paintPicked() {
    const el = $("cmd-item"); const p = H.picked;
    el.hidden = !p || !honestOn();
    if (p) el.innerHTML = `${bdi(p.id)} · ${esc(p.level)} · dataset <button type="button" class="pip-x" id="cmd-unpick" aria-label="use as free text" title="use as free text (no goal check)">×</button>`;
  }
  $("cmd").addEventListener("input", () => {               // editing the text unbinds the dataset item
    if (H.picked && $("cmd").value.trim() !== H.picked.text) unpick();
  });
  $("cmd-item").addEventListener("click", (e) => { if (e.target.closest("#cmd-unpick")) unpick(); });

  /* ------------------------------------------------------------------ history (earlier rail) */
  async function openTicket(id) {
    const req = ++H.viewReq;
    let d = null;
    try { d = await (await fetch("/honest?ticket=" + encodeURIComponent(id))).json(); } catch (e) {}
    if (req !== H.viewReq) return;                        // a newer selection won
    if (!d || !d.ticket) { toast("That ticket is no longer kept (only the last 20 are).", "warn"); return; }
    H.view = id; H.viewTicket = d.ticket; H.printed = "";
    render();
  }
  function backToLive() { H.viewReq++; H.view = null; H.viewTicket = null; render(); }
  $("h-live").addEventListener("click", backToLive);
  $("h-railitems").addEventListener("click", (e) => {
    const b = e.target.closest("[data-ticket]"); if (!b) return;
    if (H.view === b.dataset.ticket) backToLive(); else openTicket(b.dataset.ticket);
  });

  /* ------------------------------------------------------------------ render */
  function render() {
    const d = H.snap;
    paintAvail();
    if (!d || !d.on) { paintPicked(); return; }
    if (!honestOn()) return;                              // /status has not flipped the body yet
    const live = { job: d.job, planner: d.planner, steps: d.steps, outcome: d.outcome };
    const t = H.viewTicket || live;
    $("h-histbar").hidden = !H.view;
    paintTicket(t, !H.view, d);
    paintHolding(d);
    paintSession(d);
    paintRail(d);
    paintMemory(d);
    paintPlaces(d);
    paintAudit(d);
    paintPip(d);
    paintButtons();
    paintPicked();
    drawMap();
  }

  function paintHolding(d) {
    const h = d.holding;
    setText($("h-holding"), h == null ? "nothing" : h === "?" ? "unknown (?)" : h);
  }

  function stageText(job, live) {
    if (!job) return "ready";
    const p = job.phase;
    let t = PHASE_WORDS[p] || p;
    if (p === "executing" && H.snap) {
      const st = H.snap.steps || [], i = st.findIndex(x => x.status === "running");
      if (i >= 0) t = `executing · step ${i + 1}/${st.length}`;
    }
    if (p === "failed" && job.error) t = "failed: " + job.error;
    if (p === "cancelled" && job.error) t = "stopped: " + job.error;
    if (live && isActive(p)) {
      const el = (job.elapsed_s || 0) + (Date.now() - H.fetchedAt) / 1000;
      t += ` · ${el.toFixed(0)} s`;
    } else if (job.elapsed_s != null && isFinal(p)) t += ` · ${(+job.elapsed_s).toFixed(1)} s`;
    return t;
  }
  function stageClass(p) {
    return isActive(p) ? (p === "executing" || p === "sweep" ? "exec" : "think")
         : p === "done" ? "done" : (p === "failed" || p === "rejected") ? "fail" : p === "cancelled" ? "fail" : "";
  }
  /* every /status tick: the stage line (phase, elapsed, Gemma waiting) without waiting for a /honest refetch */
  function tickStage(s) {
    if (!honestOn() || !H.snap) return;
    const bad = simTrouble(s);
    const el = $("stage"), txt = $("stage-text");
    const live = H.snap.job;
    el.className = "stage";
    if (bad) { el.classList.add("fail"); setText(txt, bad); setText($("tb-state"), bad); setAgentState("fail"); return; }
    let lt = stageText(live, true);
    const w = (s.honest && s.honest.waiting) || (s.sim && s.sim.waiting);
    if (w && live && isActive(live.phase)) lt += " · " + w;
    const lc = live ? stageClass(live.phase) : "";
    setText($("tb-state"), lt);                              // topbar + camera bezel always follow the LIVE job
    setAgentState({ exec: "exec", think: "think", done: "done", fail: "fail" }[lc] || "idle");
    const vj = H.view ? (H.viewTicket || {}).job : null;
    const c = H.view ? (vj ? stageClass(vj.phase) : "") : lc;
    if (c) el.classList.add(c);
    setText(txt, H.view ? "earlier: " + stageText(vj, false) : lt);
    paintButtons();
  }

  function stamp(cls, title, body) { return `<div class="stamp is-${cls}"><b>${esc(title)}</b>${body}</div>`; }
  function paintTicket(t, live, d) {
    const job = t.job, steps = t.steps || [], out = t.outcome || {}, pl = t.planner;
    // header
    if (job) {
      const f = fa(job.lang, job.command);
      setHTML($("t-cmd"), `<bdi dir="${f ? "rtl" : "ltr"}" lang="${f ? "fa" : "en"}">${esc(job.command)}</bdi>`, SIG, "tcmd");
    } else setHTML($("t-cmd"), `<span class="dim">No instruction yet. Type one above, or pick one from the set.</span>`, SIG, "tcmd");
    showLevel(job && job.level_source === "dataset" ? levelOf(job.level) : null, "dataset");
    SIG.tlsrc = null;                                    // showLevel() just wrote this node as text
    if (job && job.item) setHTML($("t-levelsrc"), `dataset level · ${bdi(job.item)}`, SIG, "tlsrc");
    else { if (job && job.kind !== "look") setText($("t-levelsrc"), "free text, so the goal is not checked"); else setText($("t-levelsrc"), ""); }
    setText($("t-prov"), "honest");
    setClass($("t-prov"), "badge prov honest");
    // stamps
    let html = "";
    if (job && job.kind !== "look") {
      const n = steps.length, ok = steps.filter(x => x.status === "ok").length;
      const anyBad = steps.some(x => x.status === "failed" || x.status === "cancelled");
      const p = job.phase;
      let ex;
      if (!n) ex = isActive(p) ? stamp("live", "Executed", p === "planning" ? "waiting for the plan" : "not started") :
                  p === "rejected" ? stamp("fail", "Executed", "nothing: plan rejected") : stamp("none", "Executed", "no steps");
      else ex = stamp(isActive(p) ? "live" : (anyBad || ok < n) ? "fail" : "ok", "Executed", `${bdi(ok)} of ${bdi(n)} steps ok`);
      let gl;
      const g = out.goal;
      if (g && g.checked) gl = g.ok ? stamp("ok", "Goal", "met (report only)") :
          stamp("fail", "Goal", "unmet" + ((g.unmet || []).length ? ": " + esc((g.unmet || []).slice(0, 2).join("; ")) : ""));
      else if (job.item) gl = stamp("none", "Goal", isActive(p) ? "checked after the plan" : "not checked");
      else gl = stamp("none", "Goal", "not checked (free text)");
      let au;
      const a = out.audit || (live ? auditLive(d) : null);
      if (!a) au = stamp("none", "Audit", "not available");
      else if (!a.valid) au = stamp("fail", "Audit", `INVALID · ${bdi(a.violations)} violation${a.violations === 1 ? "" : "s"}`);
      else au = stamp(isActive(p) ? "live" : "ok", "Audit", `valid · ${bdi(a.violations || 0)} recorded violations`);
      // tidiness (2026-10-07): what the agent closed again at the end of the run; anything left open is shown
      let td = "";
      const ty = out.tidy;
      if (ty) {
        const left = (ty.left_open || []).concat(ty.failed || []);
        const kept = ty.kept_open || [];
        const shut = (ty.closed_during || []).concat(ty.closed || []);
        const body = (shut.length ? `closed again: ${esc(shut.join(", "))}` : "nothing opened") +
          (left.length ? `<br>still open: ${esc(left.join(", "))}` : "") +
          (kept.length ? `<br>kept open (asked): ${esc(kept.join(", "))}` : "");
        td = stamp(left.length ? "fail" : "ok", "Tidy", body);
      }
      html = ex + gl + au + td;
      // the one orchestrated moment: a finished ticket prints its stamps once (scale only, never opacity)
      const key = job.id + ":" + p;
      const box = $("t-stamps");
      if (isFinal(p) && H.printed !== key && live) {
        H.printed = key; SIG.tstamps = null; setHTML(box, html, SIG, "tstamps");
        box.classList.remove("is-print"); void box.offsetWidth; box.classList.add("is-print");
      } else setHTML(box, html, SIG, "tstamps");
    } else if (job && job.kind === "look") {
      setHTML($("t-stamps"), stamp(isActive(job.phase) ? "live" : job.phase === "done" ? "ok" : "fail", "Look around",
        isActive(job.phase) ? "sweeping" : esc(job.phase)) + stamp("none", "Memory", "kept") +
        stamp("none", "Audit", "own small episode"), SIG, "tstamps");
    } else setHTML($("t-stamps"), "", SIG, "tstamps");
    // steps
    paintSteps(steps, job);
    // planner line
    let line = "";
    if (pl) {
      const secs = pl.secs != null ? `${(+pl.secs).toFixed(1)} s` : "";
      if (pl.rejected || !pl.ok) line = `Gemma ${secs} · rejected: ${(pl.errors || [])[0] || "invalid plan"} — not executed`;
      else if (pl.repaired) line = `Gemma ${secs} · repaired once: ${(pl.errors || [])[0] || "first plan was invalid"} · ${(pl.plan || []).length} steps`;
      else line = `Gemma ${secs} · ${(pl.normalised || []).length ? "accepted after argument normalisation" : "valid on the first pass"} · ${(pl.plan || []).length} step${(pl.plan || []).length === 1 ? "" : "s"}`;
      if (pl.model) line += ` · ${pl.model}`;
      if (pl.say) line += ` — “${pl.say}”`;
      // the validator's normalisations ("put_in takes no 'to'; 'Fridge' ignored") change what runs: never hide them
      if (pl.notes && pl.notes.length) line += ` · validator: ${pl.notes.join("; ")}`;
    } else if (job && job.phase === "planning") line = "Gemma is planning with the honest tools…";
    const p = $("t-planner"); setText(p, line); p.dir = "auto";
  }
  function auditLive(d) {
    const a = d.audit, j = d.job;
    if (!a || !j || a.episode !== j.id) return null;           // an older instruction's audit is not this one's
    return { valid: !a.invalid, violations: a.violations || 0 };
  }

  function stepVerb(st) {
    let v = esc(st.action || "?");
    if (st.object) v += ` <em>${esc(st.object)}</em>`;
    if (st.to) v += ` <span class="dim">→</span> <em>${esc(st.to)}</em>`;
    if (st.arg) v += ` <span class="dim">= ${esc(st.arg)}</span>`;
    return v;
  }
  const GLYPH = { pending: "·", running: "…", ok: "✓", failed: "✗", skipped: "–", cancelled: "■" };
  function paintSteps(steps, job) {
    const box = $("h-steps");
    if (!steps.length) {
      setHTML(box, job && job.phase === "planning" ? `<li class="step-empty dim">Waiting for Gemma's plan…</li>` :
        job && job.phase === "sweep" ? `<li class="step-empty dim">Looking around before planning…</li>` : "", SIG, "hsteps");
      return;
    }
    const html = steps.map(st => {
      const meta = st.status === "pending" ? "pending" : st.status === "running" ? "running" :
        `${st.prims || 0} prims${st.secs != null ? " · " + (+st.secs).toFixed(1) + " s" : ""}`;
      let why = "";
      if (st.status === "failed" || st.status === "cancelled" || st.status === "skipped" || st.retries) {
        const c = st.cause ? (CAUSE_WORDS[st.cause] || st.cause) : "";
        const u = st.under && st.under !== st.cause ? " — underneath: " + (CAUSE_WORDS[st.under] || st.under) : "";
        const w = st.why ? (c ? ` (${st.why})` : st.why) : "";
        const r = st.retries ? `<b>${st.retries} retr${st.retries === 1 ? "y" : "ies"}</b>${c || w ? ": " : ""}` : "";
        if (c || w || r) why = `<span class="why" dir="auto">${r}${esc(c)}${esc(u)}<span class="dim">${esc(w)}</span></span>`;
      }
      const ev = (st.events || []).map(e => esc(typeof e === "string" ? e : JSON.stringify(e))).join("\n");
      const ver = (st.verify || []).map(e => esc(typeof e === "string" ? e : JSON.stringify(e))).join("\n");
      let srch = "";
      const s = st.search;
      if (s) {
        srch = `<p class="srch"><b>Search for ${esc(s.target)}</b>: ` +
          `ranked ${esc((s.ranked || []).slice(0, 6).join(", ") || "—")} → visited ${esc((s.visited || []).join(", ") || "—")}` +
          ` → opened ${esc((s.opened || []).join(", ") || "—")} · ${bdi(s.viewpoints || 0)} viewpoints · ` +
          `${s.found ? (s.found === "here" ? "found here" : "found") : "not found"}</p>`;
        // the explicit search route: numbered stops, in visiting order (2026-10-08, "systematic search route")
        const plan = s.plan || [];
        if (plan.length) {
          const PG = { planned: "", running: " ▶", done: " ✓", found: " ★", unreachable: " ✗" };
          srch += `<p class="srch plan"><b>Search route</b>${(s.plans || 0) > 1 ? ` <span class="dim">(re-planned ${bdi(s.plans - 1)}×)</span>` : ""}: ` +
            plan.map(p => `<span class="p-${esc(p.status)}">${bdi(p.n)}${PG[p.status] || ""} ${esc(p.label)}` +
              `${p.split ? ` <span class="dim">(split ${bdi(p.split)})</span>` : ""}</span>`).join(" → ") + `</p>`;
        }
        // honest inside-visible verdict per container the search opened (2026-10-07)
        const ins = (s.inspections || []).filter(Boolean);
        if (ins.length) {
          const W = { visible: "inside visible ✓", partly: "inside partly visible ~", found: "target seen inside ✓",
                      unsure: "inside not visible ✗" };
          srch += `<p class="srch insp">` + ins.map(i =>
            `<span class="v-${esc(i.verdict)}">${esc(i.type)} ${bdi(i.n)}: ${esc(W[i.verdict] || i.verdict)}` +
            `${i.reason ? " (" + esc(i.reason) + ")" : ""}${i.verdict === "unsure" ? " · unsure, kept as a candidate" : ""}` +
            (i.pose ? ` <span class="dim">[${i.crouched ? "crouched" : "standing"}, ${bdi(i.pose.dist)} m, ` +
              `${bdi(i.pose.off)}° off its front, pitch ${bdi(i.pose.pitch)}°]</span>` : "") + `</span>`
          ).join(" · ") + `</p>`;
        }
      }
      if (st.chooser) {
        const c = st.chooser;
        srch += `<p class="srch dim">Gemma chose where to look${c.secs != null ? " in " + (+c.secs).toFixed(1) + " s" : ""}: ` +
          `${esc((c.places || []).join(", ") || "—")}${c.error ? " · error: " + esc(c.error) : ""}</p>`;
      }
      const hasDet = ev || ver || srch;
      const open = H.openSteps.has(st.i) ? " open" : "";
      const det = hasDet ? `<details data-i="${st.i}"${open}><summary>events${ev ? " (" + (st.events || []).length + ")" : ""}${srch ? " · search" : ""}</summary>` +
        `${srch}${ev ? `<pre>${ev}</pre>` : ""}${ver ? `<pre class="verify">${ver}</pre>` : ""}</details>` : "";
      return `<li class="step is-${esc(st.status)}"><span class="n">${(st.i || 0) + 1}</span>` +
        `<span class="st" aria-label="${esc(st.status)}">${GLYPH[st.status] || "·"}</span>` +
        `<span class="v">${stepVerb(st)}${st.retries ? ` <span class="retry" title="retries">↻${st.retries}</span>` : ""}</span>` +
        `<span class="m">${esc(meta)}</span>${why}${det}</li>`;
    }).join("");
    setHTML(box, html, SIG, "hsteps");
  }
  $("h-steps").addEventListener("toggle", (e) => {          // keep expanded steps open across repaints
    const d = e.target; if (!d || d.tagName !== "DETAILS") return;
    const i = +d.dataset.i;
    if (d.open) H.openSteps.add(i); else H.openSteps.delete(i);
  }, true);

  function paintSession(d) {
    const s = d.session, el = $("h-session");
    if (!s) { setText(el, ""); return; }
    const iv = (s.interventions || []).map(x => `${x.kind} ×${x.n}`).join(", ");
    let m = `session ${s.id} · world ${s.world_gen} · memory ${s.episode ? s.episode.split(":").pop() : "not started"} · ` +
            `${s.instructions || 0} instruction${s.instructions === 1 ? "" : "s"}`;
    if (iv) m += ` · by hand: ${iv}`;
    if (s.tainted) m += " · TAINTED: the next run starts a new memory episode; Reset gives a clean start";
    setText(el, m);
    el.classList.toggle("bad", !!s.tainted);
  }

  function paintRail(d) {
    const h = (d.history || []).slice().reverse();
    if (!h.length) { setHTML($("h-railitems"), `<span class="dim small">Finished instructions appear here. Click one to reopen its ticket.</span>`, SIG, "hrail"); return; }
    setHTML($("h-railitems"), h.map(x => {
      const ok = x.phase === "done" && x.valid !== false;
      const lv = levelOf(x.level) || "";
      const f = fa(x.lang, x.command);
      return `<button type="button" class="railt${H.view === x.id ? " is-sel" : ""}" data-ticket="${esc(x.id)}" data-lvl="${lv}" ` +
        `title="${esc(x.phase)}${x.error ? ": " + esc(x.error) : ""}${x.valid === false ? " · audit INVALID" : ""}">` +
        `<span class="${ok ? "ok" : "fail"}">${ok ? "✓" : "✗"}</span>${lv ? `<span class="small dim">${lv}</span>` : ""}` +
        `<span class="railt-t" dir="${f ? "rtl" : "ltr"}"${f ? ' lang="fa"' : ""}>${esc(x.command)}</span>` +
        `${x.valid === false ? '<span class="fail small">INVALID</span>' : ""}</button>`;
    }).join(""), SIG, "hrail");
  }

  function paintMemory(d) {
    const m = d.memory || { tracks: [] };
    const q = ($("h-memq").value || "").trim().toLowerCase();
    let rows = (m.tracks || []).filter(t => !q || [t.type, t.state, t.status, t.place].join(" ").toLowerCase().indexOf(q) >= 0);
    const k = H.sort.key, dir = H.sort.dir;
    rows.sort((a, b) => {
      const va = a[k], vb = b[k];
      if (typeof va === "number" || typeof vb === "number") return dir * ((+va || 0) - (+vb || 0));
      return dir * String(va || "").localeCompare(String(vb || ""));
    });
    setText($("h-memn"), (m.tracks || []).length ? String((m.tracks || []).length) : "");
    document.querySelectorAll("#h-mem .sortbtn").forEach(b => {
      const on = b.dataset.sort === k;
      b.setAttribute("aria-sort", on ? (dir > 0 ? "ascending" : "descending") : "none");
      b.closest("th").setAttribute("aria-sort", on ? (dir > 0 ? "ascending" : "descending") : "none");
    });
    const body = rows.length ? rows.map(t => {
      const conf = t.conf != null ? Math.max(0, Math.min(1, +t.conf)) : null;
      const where = t.status === "held" ? "in hand" :
        `${t.place ? esc(t.place).replace(/([a-z])([A-Z])/g, "$1<wbr>$2") : "—"}${t.x != null ? `<span class="xy">${bdi((+t.x).toFixed(2) + ", " + (+t.z).toFixed(2))}</span>` : ""}`;
      const wb = (x) => esc(x).replace(/([a-z])([A-Z])/g, "$1<wbr>$2");   // CounterTop -> Counter|Top may wrap
      return `<tr><td>${wb(t.type)}</td><td>${esc(t.state || "?")}</td>` +
        `<td><span class="mstat s-${esc(t.status)}">${esc(t.status)}</span></td>` +
        `<td>${conf == null ? "—" : `<span class="conf"><i style="width:${Math.round(conf * 100)}%"></i></span>${conf.toFixed(2)}`}</td>` +
        `<td>${where}</td><td class="n">${bdi(t.step ?? "—")}</td><td class="n">${bdi(t.seen ?? "—")}</td></tr>`;
    }).join("") : `<tr><td colspan="7" class="dim">${(m.tracks || []).length ? "Nothing matches the filter." : "Nothing remembered yet. Run an instruction or press Look around."}</td></tr>`;
    setHTML($("h-membody"), body, SIG, "hmem");
  }
  $("h-memq").addEventListener("input", () => { if (H.snap) paintMemory(H.snap); });
  $("h-mem").addEventListener("click", (e) => {
    const b = e.target.closest(".sortbtn"); if (!b) return;
    const k = b.dataset.sort;
    H.sort = { key: k, dir: H.sort.key === k ? -H.sort.dir : (k === "type" || k === "status" || k === "state" ? 1 : -1) };
    if (H.snap) paintMemory(H.snap);
  });

  function paintPlaces(d) {
    const ps = d.places_seen || [];
    const by = new Map();
    ps.forEach(p => {
      const e = by.get(p.type) || { n: 0, opened: false };
      e.n++; e.opened = e.opened || !!p.opened; by.set(p.type, e);
    });
    setText($("h-places-h"), ps.length >= 60 ? "Places seen (latest 60)" : "Places seen");
    setHTML($("h-places"), by.size ? Array.from(by.entries()).map(([t, e]) =>
      `<span class="chip">${esc(t)}${e.n > 1 ? " ×" + e.n : ""}${e.opened ? " (opened)" : ""}</span>`).join("")
      : `<span class="dim small">none yet</span>`, SIG, "hplaces");
  }

  const REASONS = [["pose", "pose (proprioception)"], ["actuator", "actuator: did it work"], ["action-resolve", "screen point → object"],
                   ["map-build", "map building"], ["scoring", "scoring (report only)"], ["oracle", "oracle reads"]];
  function paintAudit(d) {
    const a = d.audit;
    const box = $("h-audit");
    if (!a) { setHTML(box, `<p class="dim">No audit yet. It starts with the first instruction or Look around.</p>`, SIG, "haud"); setText($("h-audn"), ""); return; }
    const v = a.violations || 0;
    setText($("h-audn"), a.invalid ? "!" : String(v));
    $("h-audn").className = "tabn " + (a.invalid ? "is-fail" : "is-ok");
    const counts = a.counts || {};
    const errs = a.errors_logged || [];
    const vl = a.violation_list || [];
    const hist = (d.history || []).slice().reverse();
    const html =
      `<div class="audit-big ${a.invalid ? "is-fail" : "is-ok"}">${a.invalid ? "INVALID" : bdi(v) + " recorded violation" + (v === 1 ? "" : "s")}</div>` +
      `<p class="dim small">under the implemented checks · audit episode ${bdi(a.episode || "—")}${a.invalid ? " · this run's beliefs do not carry over" : ""}</p>` +
      `<div class="counts">${REASONS.map(([k, w]) => `<div class="${k === "oracle" && counts[k] ? "is-fail" : ""}"><b>${bdi(counts[k] || 0)}</b><span>${esc(w)}</span></div>`).join("")}</div>` +
      (vl.length ? `<p class="fld-label">Violations</p><ul class="assume bad">${vl.map(x => `<li>${esc(x)}</li>`).join("")}</ul>` : "") +
      `<p class="fld-label">Actuator errors logged: ${bdi(a.errors || 0)} <span class="dim">(failure breakdown only, never decision input)</span></p>` +
      (errs.length ? `<details class="errs"><summary>show the last ${errs.length}</summary><pre>${errs.map(e => esc(Array.isArray(e) ? `[${e[0]}] ${e[1]}: ${e[2]}` : String(e))).join("\n")}</pre></details>` : "") +
      `<p class="fld-label">Standing assumptions</p><ul class="assume">${(d.assumptions || []).map(x => `<li>${esc(x)}</li>`).join("")}</ul>` +
      (hist.length ? `<p class="fld-label">This session</p><table class="histtbl"><tbody>${hist.map(x =>
        `<tr><td>${bdi(x.id)}</td><td dir="auto">${esc(x.command)}</td><td>${esc(x.phase)}</td>` +
        `<td class="${x.valid === false ? "fail" : "ok"}">${x.valid === false ? "INVALID" : "valid"}</td></tr>`).join("")}</tbody></table>` : "");
    setHTML(box, html, SIG, "haud");
  }

  function paintPip(d) {
    const fig = $("h-pip"), lk = d.last_look;
    const show = H.pipOn && !!lk;
    fig.hidden = !show;
    if (!lk) return;
    if (lk.seq !== H.pipSeq) {
      H.pipSeq = lk.seq;
      $("h-pip-img").src = "/honest_last_look.jpg?seq=" + encodeURIComponent(lk.seq);
    }
    const cap = `last look · step ${lk.step} · ${lk.n_dets} box${lk.n_dets === 1 ? "" : "es"}${lk.inspecting ? " · inspection specialist" : ""}`;
    setText($("h-pip-cap"), cap); $("h-pip-cap").title = cap;
  }
  $("h-pip-x").addEventListener("click", () => { H.pipOn = false; $("h-pip-on").checked = false; if (H.snap) paintPip(H.snap); });
  $("h-pip-on").addEventListener("change", (e) => { H.pipOn = e.target.checked; if (H.snap) paintPip(H.snap); });

  function paintButtons() {
    const job = H.snap && H.snap.job;
    const active = !!(job && isActive(job.phase));
    const stopping = active && (job.phase === "stopping" || H.stopping);
    if (!active) H.stopping = false;
    const stop = $("h-stop");
    stop.disabled = !active || stopping;
    setText(stop, stopping ? "Stopping…" : "Stop");
    const acting = honestActing();
    $("h-look").disabled = acting; $("h-forget").disabled = acting;
    if (honestOn()) $("send").disabled = acting;
  }

  /* ------------------------------------------------------------------ the agent's own map */
  function tok(cs, name, fb) { return ((cs.getPropertyValue(name) || "").trim() || fb); }
  function drawMap() {
    const cv = $("h-map"); if (!cv || !cv.offsetWidth || !H.snap || !H.snap.map) return;
    const m = H.snap.map, free = H.grid || [];
    if (!free.length) { setText($("h-mapstat"), "Loading the walked map…"); return; }
    const dpr = window.devicePixelRatio || 1;
    const xs = free.map(c => c[0]), zs = free.map(c => c[1]);
    const x0 = Math.min(...xs) - 1, x1 = Math.max(...xs) + 1, z0 = Math.min(...zs) - 1, z1 = Math.max(...zs) + 1;
    const cols = x1 - x0 + 1, rows = z1 - z0 + 1;
    const W = Math.min(cv.parentNode.clientWidth, 560);
    const Hh = Math.max(220, Math.min(460, Math.round(W * rows / cols)));
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(Hh * dpr)) {
      cv.width = Math.round(W * dpr); cv.height = Math.round(Hh * dpr);
      cv.style.width = W + "px"; cv.style.height = Hh + "px";
    }
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const cs = getComputedStyle(document.documentElement);
    const C = {
      bed: tok(cs, "--well", "#0C1013"), free: tok(cs, "--map-free", "#33424A"), edge: tok(cs, "--map-edge", "#E0704F"),
      ink: tok(cs, "--well-ink", "#D9E2E5"), agent: tok(cs, "--map-agent", "#FFC861"), ok: tok(cs, "--map-ok", "#5BD38B"),
      placed: tok(cs, "--map-placed", "#86AEFF"), fail: tok(cs, "--map-fail", "#FF8277"), dim: tok(cs, "--map-dim", "#8A989E"),
    };
    g.fillStyle = C.bed; g.fillRect(0, 0, W, Hh);
    const s0 = Math.min(W / cols, Hh / rows), s = s0 * H.zoom;
    const ox = (W - s0 * cols) / 2 + H.panX - (s - s0) * cols / 2, oz = (Hh - s0 * rows) / 2 + H.panY - (s - s0) * rows / 2;
    const P = (gx, gz) => [ox + (gx - x0) * s, oz + (z1 - gz) * s];          // north (+z) up
    H.mapGeom = { P, s, x0, z1 };
    g.fillStyle = C.free;
    const gap = s > 6 ? 1 : 0;
    for (const [x, z] of free) { const [px, pz] = P(x, z); g.fillRect(px + gap, pz + gap, s - 2 * gap, s - 2 * gap); }
    // learned blocked edges: a bar on the border between the two cells
    g.strokeStyle = C.edge; g.lineWidth = Math.max(2, s * 0.14); g.lineCap = "round";
    for (const [a, b] of (m.learned_blocked || [])) {
      const [ax, az] = P(a[0], a[1]), [bx, bz] = P(b[0], b[1]);
      const mx = (ax + bx) / 2 + s / 2, mz = (az + bz) / 2 + s / 2;
      const vert = a[1] === b[1];
      g.beginPath();
      if (vert) { g.moveTo(mx, mz - s / 2); g.lineTo(mx, mz + s / 2); } else { g.moveTo(mx - s / 2, mz); g.lineTo(mx + s / 2, mz); }
      g.stroke();
    }
    // the step in focus: running, else the last that ran
    const st = H.snap.steps || [];
    const cur = st.find(x => x.status === "running") || st.slice().reverse().find(x => x.status !== "pending" && x.status !== "skipped");
    const target = cur ? ((cur.search && cur.search.target) || cur.object) : null;
    const visited = cur && cur.search ? (cur.search.visited || []) : [];
    const origin = m.origin || [0, 0], cell = m.cell || 0.25;
    const toCell = (x, z) => [Math.round((x - origin[0]) / cell), Math.round((z - origin[1]) / cell)];
    const tracks = ((H.snap.memory || {}).tracks || []).filter(t => t.x != null && t.status !== "held");
    const r = Math.max(3, s * 0.24);
    H.mapPts = [];
    const visitMark = new Map();
    const route = cur && cur.search ? (cur.search.plan || []) : [];
    if (!route.length) visited.forEach((v, i) => { if (!visitMark.has(v)) visitMark.set(v, i + 1); });
    const labels = [];
    for (const t of tracks) {
      const [gx, gz] = toCell(t.x, t.z), [px, pz] = P(gx, gz), cx = px + s / 2, cz = pz + s / 2;
      H.mapPts.push({ x: cx, y: cz, t });
      g.lineWidth = 2;
      if (t.status === "trusted") { g.fillStyle = C.ok; g.beginPath(); g.arc(cx, cz, r, 0, 7); g.fill(); }
      else if (t.status === "placed") { g.strokeStyle = C.placed; g.beginPath(); g.arc(cx, cz, r, 0, 7); g.stroke(); }
      else if (t.status === "contradicted") {
        g.strokeStyle = C.fail; g.beginPath(); g.moveTo(cx - r, cz - r); g.lineTo(cx + r, cz + r); g.moveTo(cx + r, cz - r); g.lineTo(cx - r, cz + r); g.stroke();
      } else { g.fillStyle = C.dim; g.beginPath(); g.arc(cx, cz, r * 0.8, 0, 7); g.fill(); }
      if (target && t.type === target) {
        g.strokeStyle = C.fail; g.lineWidth = 2; g.setLineDash([4, 3]); g.beginPath(); g.arc(cx, cz, r * 2, 0, 7); g.stroke(); g.setLineDash([]);
        labels.push([t.type, cx, cz]);
      }
      if (visitMark.has(t.type)) { labels.push([String(visitMark.get(t.type)), cx - r * 2.6, cz + r * 2.4, true]); visitMark.delete(t.type); }
    }
    // the search route: numbered stops joined in visiting order (done = dim, found = green, current = agent colour)
    if (route.length) {
      const pts = route.map(p => { const [px, pz] = P(p.cell[0], p.cell[1]); return [px + s / 2, pz + s / 2, p]; });
      g.strokeStyle = C.dim; g.lineWidth = 1.5; g.setLineDash([5, 4]); g.beginPath();
      pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.stroke(); g.setLineDash([]);
      const RR = Math.max(7, s * 0.34);
      g.font = "700 11px " + tok(cs, "--f-latin", "sans-serif"); g.textAlign = "center"; g.textBaseline = "middle";
      for (const [x, y, p] of pts) {
        const col = p.status === "found" ? C.ok : p.status === "running" ? C.agent : p.status === "unreachable" ? C.fail
          : p.status === "done" ? C.dim : C.ink;
        g.fillStyle = C.bed; g.beginPath(); g.arc(x, y, RR, 0, 7); g.fill();
        g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); g.arc(x, y, RR, 0, 7); g.stroke();
        g.fillStyle = col; g.fillText(String(p.n), x, y + 0.5);
        H.mapPts.push({ x, y, stop: `search stop ${p.n} · ${p.label} · ${p.status} · stand at (${p.x}, ${p.z})` });
      }
      g.textAlign = "start";
    }
    // agent arrow, from the last pose the agent itself read (never an extra audit read)
    const ag = m.agent;
    if (ag) {
      const [px, pz] = P(ag.gx, ag.gz), cx = px + s / 2, cz = pz + s / 2, R = Math.max(7, s * 0.42);
      const a = (ag.yaw || 0) * Math.PI / 180;                // yaw 0 = +z (up), 90 = +x (right)
      const dx = Math.sin(a), dz = -Math.cos(a);
      g.fillStyle = ag.stale ? C.dim : C.agent;
      g.beginPath();
      g.moveTo(cx + dx * R, cz + dz * R);
      g.lineTo(cx - dx * R * 0.6 + dz * R * 0.6, cz - dz * R * 0.6 - dx * R * 0.6);
      g.lineTo(cx - dx * R * 0.6 - dz * R * 0.6, cz - dz * R * 0.6 + dx * R * 0.6);
      g.closePath(); g.fill();
      g.strokeStyle = C.bed; g.lineWidth = 1.5; g.stroke();
      H.mapPts.push({ x: cx, y: cz, agent: ag });
    }
    g.font = "600 13px " + tok(cs, "--f-latin", "sans-serif");
    g.textBaseline = "middle";
    for (const [txt, x, y, isVisit] of labels) {
      const w = g.measureText(txt).width;
      const lx = Math.min(Math.max(4, x + r * 1.6), W - w - 6), ly = Math.min(Math.max(10, y - r * 1.8), Hh - 10);
      g.fillStyle = "rgba(12,16,19,.78)"; g.fillRect(lx - 3, ly - 9, w + 6, 18);
      g.fillStyle = isVisit ? C.dim : C.ink; g.fillText(txt, lx, ly);
    }
    const stat = `${free.length} free cells · ${(m.learned_blocked || []).length} learned blocked edges · ${tracks.length} remembered objects on the map` +
      (ag ? ` · agent as of step ${ag.step}${ag.stale ? " (moved by hand since; the agent has not read its pose yet)" : ""}` : " · agent pose not read yet") +
      (H.zoom !== 1 ? ` · zoom ${H.zoom.toFixed(1)}×` : "");
    setText($("h-mapstat"), stat);
  }
  // zoom (wheel / pinch / buttons) and pan (drag)
  (function mapInput() {
    const cv = $("h-map"), tip = $("h-maptip");
    const zoomAt = (f, x, y) => {
      const z = Math.max(1, Math.min(6, H.zoom * f));
      if (z === H.zoom) return;
      const k = z / H.zoom;
      const W = cv.clientWidth, Hh = cv.clientHeight;
      H.panX = (H.panX - (x - W / 2)) * k + (x - W / 2); H.panY = (H.panY - (y - Hh / 2)) * k + (y - Hh / 2);
      H.zoom = z; if (z === 1) { H.panX = 0; H.panY = 0; }
      drawMap();
    };
    cv.addEventListener("wheel", (e) => {
      e.preventDefault(); const b = cv.getBoundingClientRect();
      zoomAt(e.deltaY < 0 ? 1.2 : 1 / 1.2, e.clientX - b.left, e.clientY - b.top);
    }, { passive: false });
    const ptrs = new Map(); let pinch0 = 0, drag = null;
    cv.addEventListener("pointerdown", (e) => { cv.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, [e.clientX, e.clientY]); drag = [e.clientX, e.clientY]; if (ptrs.size === 2) { const [a, b] = Array.from(ptrs.values()); pinch0 = Math.hypot(a[0] - b[0], a[1] - b[1]); } });
    cv.addEventListener("pointermove", (e) => {
      const b = cv.getBoundingClientRect(), x = e.clientX - b.left, y = e.clientY - b.top;
      if (ptrs.has(e.pointerId)) {
        ptrs.set(e.pointerId, [e.clientX, e.clientY]);
        if (ptrs.size === 2) {
          const [p, q] = Array.from(ptrs.values()), d = Math.hypot(p[0] - q[0], p[1] - q[1]);
          if (pinch0) zoomAt(d / pinch0, (p[0] + q[0]) / 2 - b.left, (p[1] + q[1]) / 2 - b.top); pinch0 = d; return;
        }
        if (drag && H.zoom > 1) { H.panX += e.clientX - drag[0]; H.panY += e.clientY - drag[1]; drag = [e.clientX, e.clientY]; drawMap(); return; }
      }
      // hover tooltip: the nearest remembered object (or the agent) within 12 px
      let best = null, bd = 144;
      for (const p of H.mapPts) { const d2 = (p.x - x) ** 2 + (p.y - y) ** 2; if (d2 < bd) { bd = d2; best = p; } }
      if (!best) { tip.hidden = true; return; }
      if (best.agent) tip.textContent = `agent · yaw ${best.agent.yaw}° · from its own pose read at step ${best.agent.step}`;
      else if (best.stop) tip.textContent = best.stop;
      else { const t = best.t; tip.textContent = `${t.type} · ${t.status} · ${t.state || "?"} · conf ${t.conf != null ? (+t.conf).toFixed(2) : "—"}${t.place ? " · on " + t.place : ""} · step ${t.step} · seen ${t.seen}`; }
      tip.hidden = false;
      tip.style.left = Math.min(x + 12, cv.clientWidth - tip.offsetWidth - 4) + "px";
      tip.style.top = Math.max(4, y - 30) + "px";
    });
    const up = (e) => { ptrs.delete(e.pointerId); if (ptrs.size < 2) pinch0 = 0; if (!ptrs.size) drag = null; };
    cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", up);
    cv.addEventListener("pointerleave", () => { tip.hidden = true; });
    $("h-map-in").addEventListener("click", () => zoomAt(1.4, cv.clientWidth / 2, cv.clientHeight / 2));
    $("h-map-out").addEventListener("click", () => zoomAt(1 / 1.4, cv.clientWidth / 2, cv.clientHeight / 2));
    $("h-map-reset").addEventListener("click", () => { H.zoom = 1; H.panX = 0; H.panY = 0; drawMap(); });
  })();

  /* ------------------------------------------------------------------ models (honest planner) */
  function models(list) {
    H.models = list || [];
    const sel = $("h-model"); if (!sel) return;
    const want = (H.modelUser && sel.value) || (H.snap && H.snap.model) || sel.value || "";
    sel.innerHTML = "";
    const names = H.models.slice();
    if (want && names.indexOf(want) < 0) names.unshift(want);
    names.forEach(n => sel.add(new Option(n, n)));
    if (want) sel.value = want;
  }
  $("h-model").addEventListener("change", () => { H.modelUser = true; if (honestOn()) setText($("cmd-model"), $("h-model").value); });
  let _lastMode = null;
  setInterval(() => {                                  // the command bar names the model the selected agent uses
    const m = honestOn() ? "honest" : "oracle";
    if (m === _lastMode) return;
    _lastMode = m;
    if (m === "honest") {
      if (!$("h-model").value && H.snap && H.snap.model) models(H.models);
      setText($("cmd-model"), $("h-model").value || (H.snap && H.snap.model) || "—");
      $("cmd").placeholder = "Tell the honest agent what to do, in English or فارسی (FloorPlan1)";
      if (H.snap) render();
      loadSet();
    } else {
      setText($("cmd-model"), $("model").value || "—");
      $("cmd").placeholder = "Tell the agent what to do, in English or فارسی";
      $("send").disabled = false;
      paintPicked();
    }
  }, 300);

  window.HONEST = { onStatus, run, redraw: drawMap, models, switchMode };
  fetchHonest(false);
})();
