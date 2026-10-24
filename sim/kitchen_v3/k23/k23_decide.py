"""K23 supervisor decisions, in Python so every failure path is testable offline (Astra K23-PREP r1 2-5, 10; r2 2-6).

capture_k23.ps1 only executes; this module decides. Commands (each prints ONE json line; exit 0 when it could decide,
2 when not -- the supervisor treats 2 and any unreadable output as STOP):

  auth  --scene S --pipeline P [--check-only]   the dated, scoped authorization record (authorization.json) is valid
                                                 (without --check-only: valid AND now inside the signed window)
  pre   --scene S --pipeline P --deadline D     may a batch launch now? -> launch (with a fresh invocation id, recorded
                                                 as OUTSTANDING before the launch) | done | stop
  post  --scene S --pipeline P --deadline D --invocation ID --wsl-rc N [--first-batch-only]
                                                 validate the receipt and decide -> continue | retry | done | stop
  reset-done / snapshot / receipt / template     helpers (receipt + snapshot run in WSL)

Durable state per scene and day: sim/kitchen_v3/k23/<scene>/session_<YYYYMMDD>.json = {outstanding, terminal, budgets,
history}. An OUTSTANDING invocation without a decided receipt, or a TERMINAL stop without a valid person-written
resolution (session_resolution_<n>.json), blocks every later launch. A thermal pause is terminal for the window.
r3 (Astra K23-PREP r3 1, 2, 4): a REAL pause marker of the day (thermal/pause_<day>-*.json, incl. an unreadable or
half-written one), a pause in the receipt or a WSL exit code 86 ends the window on EVERY path -- before receipt
validation, with no receipt, and in `pre`; only the pause test's own induced trigger is exempt. A pending_<stamp>.json
(a batch that could not write its receipt before its absolute end) blocks every later launch.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.dirname(SIM)
LEDGER = os.path.join(REPO, "docs", "final-test", "progress.md")
AUTH = os.path.join(HERE, "authorization.json")
SCENES = ("FloorPlan15", "FloorPlan17")
PIPELINES = ("probe", "thermaltest", "main", "topup", "close", "amb", "parity")
CAPTURING = ("main", "topup", "close", "amb", "parity")   # K2-AMB 2026-10-17: amb writes its own store (kitchen_v3_amb)      # K2-TOPUP 2026-10-17: pipelines that commit episodes (QC + receipt rules of main)
TEST_MODE = os.environ.get("K23_TEST_MODE") == "1"
MAX_RESETS = 16              # WSL resets per authorized window (person-invoked, counted)
MAX_IDLE = 2                 # main runs in a row that commit nothing
MAX_HANG_RETRIES = 1
MAX_BURST_RETRIES = 1
QC_BUDGET_S = 470            # QC (timeout 370 s + KILL 30 s) + receipt (60 s + KILL 10 s); capture_inner_k23.sh
GRACE_S = 300                # capture KILL-after 30 s + guard end <= 75 s + group stop 16 s + thermal post-stop 110 s
                             # + OUTER_RESERVE_S: capture_inner_k23.sh ends by deadline - OUTER_RESERVE_S, and
OUTER_RESERVE_S = 60         # capture_one_k23.sh keeps these last 60 s for its own last-resort cleanup (r3 4)
THERMAL_RESUME = "pause_ends_window"
# capture exit codes (k23_capture.py + the WSL scripts)
RC_OK, RC_ABORT, RC_REFUSED, RC_CLASSES = 0, 3, 4, 5
RC_LOCK, RC_DEADLINE, RC_FOREIGN_SIM, RC_SNAPSHOT, RC_PREQC = 9, 20, 21, 22, 23
RC_PAUSE, RC_GUARD_START, RC_GUARD_DIED, RC_DIED = 86, 87, 88, 99
RC_YIELD = 24                # the batch found another job waiting for the sim lock: it started nothing (A28 15.9)
RC_TIMEOUT = (124, 137, 143)
QC_PASS, QC_FAIL, QC_ERR, QC_WARN, QC_NONE, QC_OPEN = 0, 1, 2, 3, 4, 5
SCRIPTS = ("k23_capture.py", "capture_k23.ps1", "capture_one_k23.sh", "capture_inner_k23.sh", "preflight_k23.sh",
           "thermal_guard.py", "k23_qc.py", "qc_thresholds_k1.json", "layout_fingerprint.py", "k23_classes.py",
           "expected_classes.json", "k1_unchanged.py", "k23_decide.py", "emergency_stop_k23.sh", "k23_own.sh",
           "k23_procs.py", "k1_priority_k23.py")
A28_ITEMS = ("i_signoff", "ii_resume", "iii_barrier", "iv_agent_side", "v_deferral")
# the registered basis of the capture's Kitchen 1 precondition: A28 15.2 (a) (after the freeze, Arash read its summary)
# or A28 15.9 (Arash's dated decision to capture in parallel before the freeze, Kitchen 1 always first)
PRECONDITIONS = ("freeze_summary", "parallel_k1_priority", "k23_approvals")
# Arash 2026-10-17 ("Approve 1–5." item 1: Kitchen 3 capture; "I approve all 6." item 6: FloorPlan15 / FloorPlan17 windows
# through 2026-10-25): the basis for both kitchens once the Kitchen 1 locked test is complete.
K23_APPROVAL_TITLES = ("ARASH DECISIONS: Approve 1-5", "ARASH DECISIONS: I approve all 6")
K23_APPROVALS_LAST_DAY = "2026-10-25"
PARALLEL_DECISION_TITLE = "ARASH DECISION: start Kitchen 2 data capture now, in parallel, when the GPU is free"


def sha_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def scene_dir(scene: str) -> str:
    if TEST_MODE and os.environ.get("K23_OUTDIR"):
        return os.environ["K23_OUTDIR"]                 # integration test only
    return os.path.join(HERE, scene)


def overrides_outside_test() -> list:
    return [] if TEST_MODE else sorted(k for k in os.environ if k.startswith("K23_"))


def session_path(scene: str, day: str) -> str:
    return os.path.join(scene_dir(scene), f"session_{day.replace('-', '')}.json")


def new_session() -> dict:
    return {"resets": 0, "idle_runs": 0, "hang_retries": 0, "burst_retries": 0, "batches": 0, "last_done": None,
            "outstanding": None, "terminal": None, "resolutions": [], "history": []}


def load_session(scene: str, day: str) -> dict:
    p = session_path(scene, day)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            s = json.load(fh)
        for k, v in new_session().items():
            s.setdefault(k, v)
        return s
    return new_session()


def save_session(scene: str, day: str, s: dict) -> None:
    p = session_path(scene, day)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(s, fh, indent=1)
    os.replace(p + ".tmp", p)


def valid_resolution(path: str, need_astra: bool = True) -> list:
    """A person-written resolution: {who, when, finding, decision: "resume", astra_file (existing)}."""
    try:
        r = json.load(open(path, encoding="utf-8"))
    except Exception as exc:
        return [f"resolution unreadable: {exc!r}"]
    bad = [f"resolution field {k} missing" for k in ("who", "when", "finding", "decision") if not r.get(k)]
    if r.get("decision") != "resume":
        bad.append("resolution decision is not 'resume'")
    if need_astra and not (r.get("astra_file") and os.path.exists(os.path.join(REPO, r["astra_file"]))):
        bad.append("resolution names no existing Astra review file")
    return bad


# ======================================================================================================================
# auth: the dated, scoped authorization record.
# ======================================================================================================================
def ledger_headings(path: str = LEDGER) -> list:
    with open(path, encoding="utf-8") as fh:
        return [l.rstrip("\n") for l in fh if l.startswith("## ")]


def window_of(rec: dict) -> tuple:
    """(start, end) of the signed window. K2-CAPTURE: a window may cross midnight -- `window.end_date` (YYYY-MM-DD, the
    day after `date` at most) dates its end; a window is never longer than 24 h and always ends after it starts."""
    start = dt.datetime.strptime(f"{rec['date']} {rec['window']['start']}", "%Y-%m-%d %H:%M")
    end_day = rec["window"].get("end_date", rec["date"])
    end = dt.datetime.strptime(f"{end_day} {rec['window']['end']}", "%Y-%m-%d %H:%M")
    if not start < end <= start + dt.timedelta(hours=24):
        raise ValueError(f"window {start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M} must end after it starts, within 24 h")
    return start, end


def check_auth(rec: dict, scene: str, pipeline: str, now: dt.datetime, headings: list, here: str = HERE,
               check_only: bool = False) -> list:
    """Every reason the record does NOT authorize this run now (empty list = authorized)."""
    if not isinstance(rec, dict):
        return ["authorization record unreadable"]
    bad = []
    if rec.get("template") is not False:
        bad.append("template must be false (this is still the TEMPLATE)")
    basis = rec.get("precondition", "freeze_summary")
    if basis not in PRECONDITIONS:
        return [f"precondition {basis!r} is not one of {PRECONDITIONS}"]
    pre_fields = (("freeze_summary_entry", "freeze_summary_read_by_arash") if basis == "freeze_summary"
                  else ("approval_entries",) if basis == "k23_approvals"
                  else ("parallel_decision_entry", "kitchen1_priority"))
    for k in ("signed_by", "ledger_entry", "date", "window", "scenes", "pipelines") + pre_fields + (
            "a28_items", "thermal", "script_sha256"):
        if rec.get(k) in (None, "", [], {}):
            bad.append(f"missing field {k}")
    if bad:
        return bad
    if rec["signed_by"] != "Arash":
        bad.append("signed_by is not Arash")
    try:
        start, end = window_of(rec)
        if not start.date() <= now.date() <= end.date():
            bad.append(f"dated {rec['date']} (window to {end:%Y-%m-%d}), today is {now:%Y-%m-%d}")
        if not check_only and not (start <= now < end):
            bad.append(f"now {now:%H:%M} is outside the signed window {rec['window']['start']}-{rec['window']['end']}")
    except Exception as exc:
        bad.append(f"window unreadable: {exc}")
    if scene not in rec["scenes"]:
        bad.append(f"{scene} not authorized ({rec['scenes']})")
    if pipeline not in rec["pipelines"]:
        bad.append(f"pipeline {pipeline} not authorized ({rec['pipelines']})")
    if basis == "freeze_summary" and rec["freeze_summary_read_by_arash"] is not True:
        bad.append("Arash has not confirmed reading the Kitchen 1 freeze summary")
    if basis == "parallel_k1_priority":
        # A28 15.9 (Arash 2026-10-14 ~14:25): capture before the Kitchen 1 freeze, in parallel, Kitchen 1 always first.
        # The record must name his decision entry and state the priority rule; the freeze-summary fields do not apply.
        if rec["kitchen1_priority"] is not True:
            bad.append("kitchen1_priority must be true (A28 15.9: the capture yields to every Kitchen 1 stage)")
        if scene != "FloorPlan15" or rec.get("scenes") != ["FloorPlan15"]:
            bad.append("the parallel basis (A28 15.9) covers FloorPlan15 only: scenes must be exactly ['FloorPlan15']")
        if PARALLEL_DECISION_TITLE not in str(rec["parallel_decision_entry"]):
            bad.append(f"parallel_decision_entry does not name Arash's decision '{PARALLEL_DECISION_TITLE}'")
    if basis == "k23_approvals":
        ents = list(rec["approval_entries"])
        for title in K23_APPROVAL_TITLES:
            if not any(title in e for e in ents):
                bad.append(f"approval_entries do not name Arash's '{title}'")
        for e in ents:
            if f"## {e}" not in headings:
                bad.append(f"approval entry: no ledger heading is exactly '## {e}'")
        if not set(rec["scenes"]) <= set(SCENES):
            bad.append(f"scenes {rec['scenes']} outside {SCENES}")
        try:
            if window_of(rec)[1].strftime("%Y-%m-%d") > K23_APPROVALS_LAST_DAY:
                bad.append(f"the k23_approvals basis covers windows ending by {K23_APPROVALS_LAST_DAY}")
        except Exception:
            pass
    for k in (("ledger_entry",) if basis == "k23_approvals" else
              ("ledger_entry", "freeze_summary_entry" if basis == "freeze_summary" else "parallel_decision_entry")):
        t = rec[k]
        if f"## {t}" not in headings:
            bad.append(f"{k}: no ledger heading is exactly '## {t}'")
        elif rec["date"] not in t:
            bad.append(f"{k}: the heading is not dated {rec['date']}")
    items = rec["a28_items"]
    for k in A28_ITEMS:
        if not str(items.get(k) or "").strip():
            bad.append(f"A28 item 15.6 ({k}) not closed")
    th = rec["thermal"] if isinstance(rec["thermal"], dict) else {}
    for k, v in thermal_policy().items():          # every implemented rule value, explicitly signed (r2 3)
        if th.get(k) != v:
            bad.append(f"thermal.{k} must be {v!r} (the implemented guard rule; change the guard, re-hash, re-sign), "
                       f"got {th.get(k)!r}")
    for name, h in rec["script_sha256"].items():
        p = os.path.join(here, name)
        if not os.path.exists(p):
            bad.append(f"authorized script missing: {name}")
        elif sha_file(p) != h:
            bad.append(f"script changed after the authorization: {name}")
    missing = [s for s in SCRIPTS if s not in rec["script_sha256"]]
    if missing:
        bad.append(f"authorization does not pin {missing}")
    return bad


def thermal_policy() -> dict:
    """The thermal rule exactly as thermal_guard.py implements it (what Arash signs under A28 15.6 (ii))."""
    import thermal_guard as tg
    return {"resume": THERMAL_RESUME, "pause_c": tg.TEMP_MAX_C, "sm_min_mhz": tg.SM_MIN_MHZ,
            "util_full_pct": tg.UTIL_FULL, "consecutive_samples": tg.CONSECUTIVE, "interval_s": tg.INTERVAL_S,
            "no_telemetry_s": tg.NO_TELEMETRY_S}


def load_auth():
    return json.load(open(AUTH, encoding="utf-8"))


def cmd_auth(a) -> dict:
    try:
        rec = load_auth()
    except Exception as exc:
        return {"ok": False, "why": [f"cannot read {AUTH}: {exc!r}"]}
    bad = check_auth(rec, a.scene, a.pipeline, dt.datetime.now(), ledger_headings(), check_only=a.check_only)
    return {"ok": not bad, "why": bad or ["authorized"], "ledger_entry": rec.get("ledger_entry"),
            "precondition": rec.get("precondition", "freeze_summary"),
            "deadline": f"{(rec.get('window') or {}).get('end_date', rec.get('date'))} "
                        f"{(rec.get('window') or {}).get('end')}"}          # r8c: the window's END day (display only)


def template() -> dict:
    return {"template": True,
            "what": "K23 capture authorization: copy to authorization.json ONLY after Arash's dated ledger entry; fill "
                    "every field from that entry; set template false. The supervisor verifies all of it.",
            "signed_by": "Arash", "ledger_entry": "<exact heading text after '## ' of Arash's dated entry>",
            "date": "2026-10-14", "window": {"start": "07:30", "end": "10:30"}, "scenes": ["FloorPlan15"],
            "pipelines": ["probe", "thermaltest", "main"],
            "precondition": "parallel_k1_priority  (or freeze_summary; A28 15.9 / 15.2 (a))",
            "parallel_decision_entry": "<exact heading text after '## ' of Arash's decision entry (A28 15.9)>",
            "kitchen1_priority": False,
            "freeze_summary_entry": "<only with precondition freeze_summary: the freeze-summary heading>",
            "freeze_summary_read_by_arash": False,
            "a28_items": {k: "" for k in A28_ITEMS},
            "thermal": thermal_policy(),
            "script_sha256": {s: sha_file(os.path.join(HERE, s)) for s in SCRIPTS if os.path.exists(os.path.join(HERE, s))}}


# ======================================================================================================================
# thermal closure (Astra K23-PREP r3 1): a REAL pause ends the window on EVERY path -- before the receipt is validated,
# when the receipt is missing, and in `pre`. Only the pause TEST's own trigger (thermal_guard marker induced + test +
# TEST TRIGGER reason) is not a real pause.
# ======================================================================================================================
MARKER_RE = re.compile(r"pause_(\d{8})-(\d{6})-([0-9A-Za-z_]{1,8})\.json(\.tmp)?")


def marker_induced(m) -> bool:
    return (isinstance(m, dict) and m.get("test") is True and m.get("induced") is True
            and str(m.get("reason") or "").startswith("TEST TRIGGER"))


# A28-ii-SM (Arash 2026-10-19 ~00:27, AskUserQuestion "Resume tonight, alone (Recommended)"; Astra K23-A28SM-r1 REJECT
# -> r2): a slow-clock pause with the GPU below 80 C and no hardware heat flag does not end the window. Narrow and
# record-bound: only the ONE pause marker pinned (with its guard CSV) by thermal/a28sm_<inv8>.json, whose conditions
# RECOMPUTE true from those immutable files, with Arash's heading and a pinned affirmative Astra review of THIS code.
# 87 C x5 and no-telemetry pauses are never exempt; stop records are lifted only if ALL their evidence is that pause.
A28SM_ENTRY = ("2026-10-19 ~00:27 IST -- ARASH DECISION: A28 (ii) amendment A28-ii-SM (a slow-clock pause below 80 C "
               "without a hardware heat flag does not end the window)")
A28SM_TEMP_C = 80.0
A28SM_REASON = "SM clock < 1000 MHz with utilization >= 90 % in 5 consecutive samples"
A28SM_ASTRA = os.path.join("docs", "final-test", "_astra", "K23-A28SM-r2.txt")
A28SM_PREFIXES = ("THERMAL PAUSE (persisted first): ", "THERMAL PAUSE: a pause ends this capture window (no restart): ")


def _finite_num(x) -> bool:
    import math
    return type(x) in (int, float) and math.isfinite(x)


def _a28sm_csv_ok(csv_path: str, samples: list) -> list:
    """The guard CSV is complete and agrees with the marker: every telemetry row is followed by a readable flag pair
    with the hardware flag 'Not Active'; no NO_TELEMETRY row; each pause sample is a CSV row (same ts/temp/sm/util)."""
    rows, bad = [], []
    lines = open(csv_path, encoding="utf-8", errors="replace").read().splitlines()

    def is_row(line):
        return line[:1].isdigit()

    for i, line in enumerate(lines):
        if is_row(line):
            f = [x.strip() for x in line.split(",")]
            try:
                complete = len(f) >= 6 and all(_finite_num(float(f[k])) for k in (2, 3, 5))
            except ValueError:
                complete = False
            if not complete:                       # NO_TELEMETRY, empty or non-numeric rule fields
                bad.append(f"telemetry row {i} incomplete")
                continue
            if not (i + 1 < len(lines) and lines[i + 1].startswith("# corun ")):
                bad.append(f"row {i}: no flag line follows")
            rows.append(f)
        elif line.startswith("# corun "):          # EVERY flag line: right after a complete row, readable, hw inactive
            mm = re.fullmatch(r"# corun \S+ thermal_flags '([^']*)' -> \S+(?: hw \S+)?", line)
            parts = [x.strip().lower() for x in mm.group(1).split(",")] if mm else []
            if i == 0 or not is_row(lines[i - 1]):
                bad.append(f"flag line {i} does not follow a telemetry row")
            if len(parts) != 2 or any(x not in ("active", "not active") for x in parts):
                bad.append(f"flag line {i}: no readable flag pair")
            elif parts[1] != "not active":
                bad.append(f"flag line {i}: hardware heat flag active")
    for s in samples:
        hit = [f for f in rows if f[1] == str(s.get("ts"))]
        try:
            same = len(hit) == 1 and float(hit[0][2]) == s["temp"] and float(hit[0][3]) == s["sm"] \
                and float(hit[0][5]) == s["util"]
        except (TypeError, ValueError, KeyError):
            same = False
        if not same:
            bad.append(f"pause sample {s.get('ts')} is not one CSV row with the same values")
    if len(rows) < 5:
        bad.append("fewer than 5 telemetry rows")
    return bad


def a28sm_exempt(scene: str, inv8: str, tdir: str | None = None, headings: list | None = None,
                 marker_file: str | None = None) -> list:
    """Every reason the pause of invocation `inv8` (and, if given, exactly `marker_file`) is NOT exempt under A28-ii-SM
    (empty list = exempt). Fail closed: any error is a reason."""
    try:
        d = tdir or os.path.join(scene_dir(scene), "thermal")
        p = os.path.join(d, f"a28sm_{inv8}.json")
        if len(inv8) != 8 or not os.path.exists(p):
            return ["no A28-ii-SM record"]
        r = json.load(open(p, encoding="utf-8"))
        bad = []
        hs = ledger_headings() if headings is None else headings
        if f"## {A28SM_ENTRY}" not in hs:
            bad.append("Arash's A28-ii-SM ledger heading is missing")
        mf, cf = str(r.get("marker") or ""), str(r.get("guard_csv") or "")
        if marker_file is not None and marker_file != mf:
            bad.append("another marker of this invocation (only the pinned marker can be exempt)")
        m = MARKER_RE.fullmatch(mf)
        if not m or m.group(4) or m.group(3) != inv8:
            bad.append("the pinned marker is not this invocation's pause marker")
        if not re.fullmatch(r"session_\d{8}-\d{6}-" + re.escape(inv8) + r"\.csv", cf):
            bad.append("the pinned guard CSV is not this invocation's")
        for name, key in ((mf, "marker_sha256"), (cf, "guard_csv_sha256")):
            fp = os.path.join(d, name)
            if not name or not os.path.isfile(fp) or sha_file(fp) != r.get(key):
                bad.append(f"{key} does not match the file")
        ap = os.path.join(REPO, A28SM_ASTRA)
        if os.path.normpath(str(r.get("astra_file") or "")) != os.path.normpath(A28SM_ASTRA) or not os.path.isfile(ap) \
                or sha_file(ap) != r.get("astra_sha256"):
            bad.append(f"the record does not pin the A28-ii-SM review {A28SM_ASTRA}")
        else:
            txt = open(ap, encoding="utf-8", errors="replace").read()
            code = [sha_file(os.path.abspath(__file__)), sha_file(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                                "thermal_guard.py"))]
            if not any(l.startswith("VERDICT: APPROVE") for l in txt.splitlines()) or any(h not in txt for h in code):
                bad.append("the review is not an affirmative review of THIS k23_decide.py + thermal_guard.py")
        if bad:
            return bad
        mk = json.load(open(os.path.join(d, mf), encoding="utf-8"))
        if mk.get("test") is not False or mk.get("induced") is not False:
            bad.append("a pause-test marker")
        if not str(mk.get("reason") or "").startswith(A28SM_REASON):
            bad.append("not the slow-clock rule (87 C and no-telemetry pauses are never exempt)")
        ls = mk.get("last_samples") or []
        if len(ls) < 5 or any(not _finite_num(s.get("temp")) or s["temp"] >= A28SM_TEMP_C for s in ls):
            bad.append(f"a pause sample at or above {A28SM_TEMP_C:.0f} C, non-finite or missing")
        bad += _a28sm_csv_ok(os.path.join(d, cf), ls)
        return bad
    except Exception as exc:
        return [f"A28-ii-SM check failed ({exc!r})"]


def a28sm_evidence_only(text: str, marker_file: str, prefixed: bool, scene: str = "", tdir: str | None = None) -> bool:
    """`text` (a stop's why; segments chained by ' | earlier: ') holds ONLY the evidence lines of this exact pause, each
    an EXACT copy of what pause_markers / real_pauses write for the pinned marker (reason[:120]), each segment naming the
    marker (duplicate persistence of one pause is allowed; any other cause or any extra text is not)."""
    try:
        d = tdir or os.path.join(scene_dir(scene), "thermal")
        reason = str(json.load(open(os.path.join(d, marker_file), encoding="utf-8")).get("reason") or "")[:120]
    except Exception:
        return False
    if not reason.startswith(A28SM_REASON):
        return False
    allowed = (f"marker {marker_file}: {reason}",
               f"the receipt carries a pause marker that is not the pause-test trigger ({reason})",
               "the receipt / exit code (wsl rc 86) shows a pause that is not a proven pause-test trigger")
    segs = str(text or "").split(" | earlier: ")
    for seg in segs:
        if prefixed:
            p = next((x for x in A28SM_PREFIXES if seg.startswith(x)), None)
            if p is None:
                return False
            seg = seg[len(p):]
        comps = seg.split("; ")
        if allowed[0] not in comps or any(c not in allowed for c in comps):
            return False
    return bool(segs)


def _a28sm_marker_of(scene: str, inv: str) -> str | None:
    """The pinned marker file of an exempt invocation (None if the invocation is not exempt)."""
    if len(inv) < 8 or a28sm_exempt(scene, inv[:8]):
        return None
    try:
        return json.load(open(os.path.join(scene_dir(scene), "thermal", f"a28sm_{inv[:8]}.json"),
                              encoding="utf-8"))["marker"]
    except Exception:
        return None


def closure_exempt(scene: str, day: str) -> bool:
    """The day's closure file holds ONLY the evidence of A28-ii-SM-exempt pauses (each event bound to an exempt
    invocation, its text nothing but that pause's evidence)."""
    try:
        ev = json.load(open(closure_path(scene, day), encoding="utf-8")).get("events") or []
        if not ev:
            return False
        for e in ev:
            mf = _a28sm_marker_of(scene, str(e.get("invocation") or ""))
            if not mf or not a28sm_evidence_only(e.get("why"), mf, prefixed=False, scene=scene):
                return False
        return True
    except Exception:
        return False


def pause_markers(scene: str, day: str, tdir: str | None = None) -> list:
    """Every pause marker of this scene's thermal folder for this day (YYYY-MM-DD). A marker whose name cannot be read,
    a leftover .tmp (the guard was stopped while writing it) or an unreadable marker counts as a REAL pause."""
    d = tdir or os.path.join(scene_dir(scene), "thermal")
    out = []
    if not os.path.isdir(d):
        return out
    want = day.replace("-", "")
    # K2-CAPTURE: a window may cross midnight (window_of), so a marker stamped the day BEFORE the deadline day belongs
    # to it too (fail closed: such a marker always counts)
    prev = (dt.datetime.strptime(want, "%Y%m%d") - dt.timedelta(days=1)).strftime("%Y%m%d")
    for f in sorted(os.listdir(d)):
        if not f.startswith("pause_"):
            continue
        m = MARKER_RE.fullmatch(f)
        if m and m.group(1) not in (want, prev):
            continue                                     # another day = another window
        rec = {"file": f, "inv8": m.group(3) if m else None, "induced": False}
        if not m:
            rec["why"] = "unrecognised marker name"
        elif m.group(4):
            rec["why"] = "marker left half-written (.tmp)"
        else:
            try:
                mk = json.load(open(os.path.join(d, f), encoding="utf-8"))
                rec["induced"] = marker_induced(mk)
                rec["why"] = str(mk.get("reason") or "no reason")[:120]
                if not rec["induced"] and not a28sm_exempt(scene, m.group(3), d, marker_file=f):
                    rec["exempt"] = True                  # A28-ii-SM: this pinned pause does not end the window
            except Exception as exc:
                rec["why"] = f"unreadable ({type(exc).__name__})"
        out.append(rec)
    return out


def real_pauses(markers: list, rec, pipeline: str, wsl_rc, invocation: str = "") -> list:
    """The evidence of a REAL pause (empty = none). Never raises (r4 1: it runs BEFORE any fallible validation; an
    internal error is itself counted as a real pause, fail closed). REAL evidence dominates every induced claim (r4 1):
    any non-induced file marker of the day; a receipt marker copy that is present and not the test trigger (whatever
    the pipeline and whatever the file markers say); and a pause shown by the receipt / exit code is induced ONLY in
    the thermaltest pipeline when the receipt's copy (if any) is the trigger AND every file marker of this invocation is
    the trigger AND at least one of them proves it."""
    try:
        out = [f"marker {m.get('file')}: {m.get('why')}" for m in (markers or []) if not (isinstance(m, dict) and (
            m.get("induced") is True or m.get("exempt") is True))]
        is_dict = isinstance(rec, dict)
        pm = rec.get("pause_marker") if is_dict else None
        if is_dict and "pause_marker" in rec and not marker_induced(pm):
            out.append(f"the receipt carries a pause marker that is not the pause-test trigger "
                       f"({str((pm or {}).get('reason') if isinstance(pm, dict) else pm)[:120]})")
        shown = (wsl_rc == RC_PAUSE or (is_dict and (rec.get("thermal") is True or rec.get("capture_rc") == RC_PAUSE))
                 or (is_dict and rec.get("thermal") not in (True, False)))
        if shown:
            mine = [m for m in (markers or []) if invocation and isinstance(m, dict)
                    and m.get("inv8") == invocation[:8]]
            pm_ok = (pm is None and "pause_marker" not in rec) if is_dict else True
            pm_ok = pm_ok or marker_induced(pm)
            all_mine = all(m.get("induced") is True for m in mine)
            induced = pipeline == "thermaltest" and pm_ok and all_mine and (marker_induced(pm) or bool(mine))
            if not induced:
                out.append(f"the receipt / exit code (wsl rc {wsl_rc}) shows a pause that is not a proven pause-test "
                           f"trigger")
        return out
    except Exception as exc:                              # fail closed: an unreadable pause claim is a real pause
        return [f"pause evidence could not be evaluated ({exc!r}): treated as a REAL pause"]


def closure_path(scene: str, day: str) -> str:
    return os.path.join(scene_dir(scene), f"window_ended_{day.replace('-', '')}.json")


def early_closure(scene: str, day: str, pipeline: str, wsl_rc, invocation: str, now_epoch: float) -> dict:
    """r5 1: read the pause evidence (markers + receipt) and, on a REAL pause, write the durable closure file FIRST
    (window_ended_<day>.json, appended per event; `pre` refuses while it exists). Never raises."""
    out = {"markers": [], "rec": None, "rec_error": None, "real": []}
    try:
        out["markers"] = pause_markers(scene, day)
    except Exception as exc:
        out["markers"] = [{"file": "thermal/", "induced": False, "why": f"pause markers unreadable ({exc!r})"}]
    try:
        out["rec"] = json.load(open(os.path.join(scene_dir(scene), "last_batch.json"), encoding="utf-8"))
    except Exception as exc:
        out["rec_error"] = exc
    out["real"] = real_pauses(out["markers"], out["rec"], pipeline, wsl_rc, invocation)
    if out["real"]:
        try:
            p = closure_path(scene, day)
            old = []
            if os.path.exists(p):
                try:
                    old = json.load(open(p, encoding="utf-8")).get("events") or []
                except Exception:
                    old = [{"why": "earlier closure file unreadable (kept closed)"}]
            ev = old + [{"t": int(now_epoch), "invocation": invocation, "why": "; ".join(out["real"][:3])[:1000]}]
            with open(p + ".tmp", "w", encoding="utf-8") as fh:
                json.dump({"window_ended": True, "events": ev}, fh, indent=1)
            os.replace(p + ".tmp", p)
        except Exception as exc:
            out["real"].append(f"closure file could not be written ({exc!r})")
    return out


def close_window(sess: dict, why: str, now_epoch: float, invocation: str) -> None:
    old = sess.get("terminal") or {}
    sess["terminal"] = {"why": (why if not old.get("why") else f"{why} | earlier: {old.get('why')}")[:2000],
                        "t": int(now_epoch), "invocation": invocation, "window_ended": True}
    sess.setdefault("pause_evidence", []).append({"t": int(now_epoch), "invocation": invocation, "why": why[:500]})


# ======================================================================================================================
# pre: may a batch launch now?
# ======================================================================================================================
def decide_pre(sess: dict, now_epoch: float, deadline_epoch: float, batch_minutes: int, qc_open: dict | None,
               pipeline: str, signed_end_epoch: float | None = None, qc_bad: list | None = None,
               markers: list | None = None, pending: list | None = None) -> dict:
    real = [f"marker {m['file']}: {m.get('why')}" for m in (markers or []) if not m.get("induced")
            and m.get("exempt") is not True]
    if real:
        if not (sess.get("terminal") or {}).get("window_ended"):
            close_window(sess, "THERMAL PAUSE (found by pre): " + "; ".join(real[:3]), now_epoch, "")
        return {"action": "stop", "why": "a thermal pause ended this capture window (no restart): " + "; ".join(real[:3])}
    if pending:
        return {"action": "stop", "why": f"a batch left work PENDING (no receipt could be written): {pending[:3]}"}
    if signed_end_epoch is None or abs(deadline_epoch - signed_end_epoch) > 1:
        return {"action": "stop", "why": "the -Deadline does not equal the end of the signed window"}
    if sess.get("outstanding"):
        return {"action": "stop", "why": f"invocation {sess['outstanding'][:8]} has no decided receipt (missing / "
                                         f"unreadable output): its committed episodes may be unchecked"}
    if sess.get("terminal"):
        return {"action": "stop", "why": f"terminal stop not resolved: {sess['terminal'].get('why')}"}
    if qc_open:
        return {"action": "stop", "why": f"QC batch {qc_open.get('batch')} ended {qc_open.get('verdict')} and is "
                                         f"unresolved (no valid qc/resolution_batch_NNN.json)"}
    if qc_bad:
        return {"action": "stop", "why": f"QC report(s) not PASS and unresolved: {qc_bad[:3]}"}
    left = deadline_epoch - now_epoch
    need = batch_minutes * 60 if pipeline in CAPTURING else 600
    if left < need + QC_BUDGET_S + GRACE_S:
        return {"action": "done", "why": f"HARD STOP: {int(left // 60)} min to the deadline < {need // 60} min batch "
                                         f"+ QC + grace; no new batch (capture state is saved per episode)"}
    if sess["idle_runs"] >= MAX_IDLE:
        return {"action": "stop", "why": f"{sess['idle_runs']} runs in a row without a committed episode"}
    inv = uuid.uuid4().hex
    sess["outstanding"] = inv               # recorded BEFORE the launch (the caller saves the session)
    return {"action": "launch", "invocation": inv, "deadline_epoch": int(deadline_epoch)}


# ======================================================================================================================
# post: validate the receipt and decide.
# ======================================================================================================================
REQUIRED = {"invocation": str, "scene": str, "pipeline": str, "capture_rc": int, "qc_rc": int, "thermal": bool,
            "done_before": int, "done_after": int, "failed_after": int, "guard_ok": bool, "new_episodes": list,
            "new_skipped": list, "skipped_not_train": list, "skipped_unexplained": list}


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def validate_receipt(rec, invocation: str, scene: str, pipeline: str) -> list:
    if not isinstance(rec, dict):
        return ["receipt is not a json object"]
    bad = [f"field {k} missing or not {t.__name__}" for k, t in REQUIRED.items()
           if not isinstance(rec.get(k), t) or (t is int and not _is_int(rec.get(k)))]
    if bad:
        return bad
    if rec["invocation"] != invocation:
        bad.append(f"receipt is from another invocation ({rec['invocation'][:8]} != {invocation[:8]}): stale")
    if rec["scene"] != scene or rec["pipeline"] != pipeline:
        bad.append(f"receipt scene/pipeline {rec['scene']}/{rec['pipeline']} != {scene}/{pipeline}")
    ne, sk = rec["new_episodes"], rec["new_skipped"]
    for k in ("skipped_not_train", "skipped_unexplained"):
        if not all(_is_int(e) for e in rec[k]):
            bad.append(f"{k} must list integers")
    if not all(_is_int(e) for e in ne + sk):          # checked BEFORE set(): an unhashable entry must not raise (r4 1)
        return bad + ["new_episodes / new_skipped must be unique integers"]
    if len(set(ne)) != len(ne) or len(set(sk)) != len(sk):
        bad.append("new_episodes must be unique integers")
    if len(ne) != rec["done_after"] - rec["done_before"]:
        bad.append(f"{len(ne)} new episodes listed != done {rec['done_before']} -> {rec['done_after']}")
    if not set(sk) <= set(ne):
        bad.append("skipped episodes not among the new ones")
    if rec["skipped_not_train"]:
        bad.append(f"episodes without a record that are not quota-skipped TRAIN episodes: {rec['skipped_not_train']}")
    if rec["skipped_unexplained"]:
        bad.append(f"episodes 'done' without a record and without the capture's quota-skip log line: "
                   f"{rec['skipped_unexplained']}")
    if pipeline not in CAPTURING and ne:
        bad.append(f"pipeline {pipeline} must commit no episode, the receipt lists {ne[:5]}")
    return bad


LINE_STATUSES = ("PASS", "NOTE", "INFO", "WARN", "FAIL")
# check lines every K23 QC batch report carries (k23_qc.verdict); a report without them is not evidence (r4 2)
REQUIRED_CHECKS = ("class_ids", "failed_episodes", "label_meta", "pairing", "dup_cross_split", "dup_exact",
                   "check_capture", "layout", "kitchen1_unchanged", "cc_pace", "mask_sizes", "classes_seen")


SKIP_LINE = "{scene}: ep {e:3d} [train] skipped: every train target already met"


SKIP_EVIDENCE = "skip_evidence_{e:03d}.json"


def quota_sha(quota) -> str:
    return hashlib.sha256(json.dumps(quota, sort_keys=True).encode()).hexdigest()


def write_skip_evidence(qdir: str, scene: str, e: int, quota, first_seen_batch: int) -> dict:
    """r7 (Astra r6 2): RETAIN the evidence of a quota skip at first sight -- qc/skip_evidence_<ep>.json = the
    capture_state quota as stored then + the capture's skip line + the batch -- and return the provenance entry that
    binds it by sha256. Written once (an existing file is never replaced: the entry must then match it)."""
    name = SKIP_EVIDENCE.format(e=int(e))
    p = os.path.join(qdir, name)
    body = {"scene": scene, "episode": int(e), "first_seen_batch": int(first_seen_batch),
            "log_line": SKIP_LINE.format(scene=scene, e=int(e)), "quota": quota, "quota_sha256": quota_sha(quota),
            "t": time.strftime("%Y-%m-%d %H:%M:%S")}
    if not os.path.exists(p):
        with open(p + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(body, fh, sort_keys=True)
        os.replace(p + ".tmp", p)
    else:
        body = json.load(open(p, encoding="utf-8"))
    return {"episode": int(e), "first_seen_batch": body.get("first_seen_batch"), "log_line": body.get("log_line"),
            "quota_sha256": body.get("quota_sha256"), "evidence": name, "evidence_sha256": sha_file(p), "t": body.get("t")}


def skip_entry_problems(key, entry, scene: str, log_text: str | None = None, qdir: str | None = None,
                        batches: int | None = None) -> list:
    """r5 2 + r7: a quota-skip provenance entry is evidence only when it is a dict {episode (== its key, int),
    first_seen_batch (int >= 1, and <= batches + 1 when the QC batch count is given), log_line (exactly the capture's
    skip line for THIS scene and episode), quota_sha256 (64 hex), evidence (= skip_evidence_<ep>.json),
    evidence_sha256, t}, its RETAINED evidence file in ``qdir`` still has that sha256 and agrees with the entry (scene,
    episode, batch, line, quota hash = the hash of the quota it holds), that quota says every train target is met
    (the only state in which stage_capture skips), and -- when the capture log is given -- the line is still in the
    append-only log. No ``qdir`` = the evidence cannot be re-read = not evidence (fail closed)."""
    try:
        e = int(key)
    except (TypeError, ValueError):
        return [f"skip provenance key {key!r} is not an episode number"]
    if not isinstance(entry, dict):
        return [f"skip provenance of episode {e} is not a record ({str(entry)[:40]})"]
    out = []
    if entry.get("episode") != e or not _is_int(entry.get("episode")):
        out.append(f"skip provenance of episode {e} names episode {entry.get('episode')!r}")
    fsb = entry.get("first_seen_batch")
    if not (_is_int(fsb) and fsb >= 1):
        out.append(f"skip provenance of episode {e}: first_seen_batch missing")
    elif batches is not None and (not _is_int(batches) or fsb > batches + 1):
        out.append(f"skip provenance of episode {e}: first_seen_batch {fsb} is after the QC's batch {batches} + 1")
    want = SKIP_LINE.format(scene=scene, e=e)
    if entry.get("log_line") != want:
        out.append(f"skip provenance of episode {e}: log_line is not this scene's skip line")
    q = entry.get("quota_sha256")
    if not (isinstance(q, str) and re.fullmatch(r"[0-9a-f]{64}", q)):
        out.append(f"skip provenance of episode {e}: quota_sha256 missing")
    if log_text is not None and want not in log_text:
        out.append(f"skip provenance of episode {e}: its log line is no longer in the capture log")
    # r7: the retained evidence
    name = SKIP_EVIDENCE.format(e=e)
    if entry.get("evidence") != name:
        out.append(f"skip provenance of episode {e}: evidence is not {name}")
        return out
    if qdir is None:
        out.append(f"skip provenance of episode {e}: its evidence folder is unknown (cannot re-read {name})")
        return out
    p = os.path.join(qdir, name)
    if not os.path.isfile(p) or os.path.islink(p):
        out.append(f"skip provenance of episode {e}: retained evidence {name} is missing")
        return out
    if sha_file(p) != entry.get("evidence_sha256"):
        out.append(f"skip provenance of episode {e}: retained evidence {name} changed (sha256)")
        return out
    try:
        body = json.load(open(p, encoding="utf-8"))
        assert isinstance(body, dict)
    except Exception:
        return out + [f"skip provenance of episode {e}: retained evidence {name} unreadable"]
    if body.get("scene") != scene or body.get("episode") != e or body.get("log_line") != want             or body.get("first_seen_batch") != fsb:
        out.append(f"skip provenance of episode {e}: retained evidence disagrees with the entry")
    try:
        qs = quota_sha(body.get("quota"))
    except Exception:
        qs = None
    if qs is None or qs != body.get("quota_sha256") or qs != q:
        out.append(f"skip provenance of episode {e}: the retained quota does not hash to the entry's quota_sha256")
    try:
        sys.path.insert(0, SIM)
        import kitchen_data as kdat
        met = bool(kdat.Quota.from_dict(body["quota"]).split_done("train"))
    except Exception:
        met = False
    if not met:
        out.append(f"skip provenance of episode {e}: the retained quota does not show every train target met")
    return out


_PLAN = {}


def _parity_plan(scene, kdat) -> list:
    """K2-PARITY (Astra K2-PARITY-r1 5): the parity TRAIN plan (k23_plans, single source of truth)."""
    import k23_plans
    return k23_plans.parity_plan(scene, kdat)


def plan_splits(scene, eps) -> dict | None:
    """{split: count} of these episodes in the registered main + top-up plan of ``scene`` (None = cannot tell)."""
    if scene not in SCENES:
        return None
    if scene not in _PLAN:
        try:
            sys.path.insert(0, SIM)
            import kitchen_data as kdat
            _PLAN[scene] = {e["episode"]: e["split"] for e in kdat.episode_plan(scene, kdat.CAPTURE["episodes"])
                            + kdat.topup_train_plan(scene, kdat.TOPUP_TRAIN_MAX) + kdat.close_plan(scene, kdat.CLOSE_EPISODES)
                            + kdat.amb_plan(scene) + _parity_plan(scene, kdat)}
        except Exception:
            return None
    out = {}
    for e in eps:
        sp = _PLAN[scene].get(e)
        if sp is None:
            return None
        out[sp] = out.get(sp, 0) + 1
    return out


TRAIN_CHECKS = ("blur", "exposure", "exposure_extreme", "corruption", "near_dup", "mask_vs_label", "pixel_medians",
                "class_balance")


def report_content_problems(r) -> list:
    """Why a QC batch report's CONTENTS are not valid evidence (empty = valid). Astra K23-PREP r4 2: a PASS report with
    no check lines was accepted. Every line must be {status in LINE_STATUSES, check: non-empty str, detail: str}; every
    REQUIRED_CHECKS line must be present exactly once; the stored verdict must be the one the lines imply (FAIL if any
    FAIL, else WARN if any WARN, else PASS); the episode list must be unique integers."""
    if not isinstance(r, dict):
        return ["report is not a json object"]
    out = []
    lines = r.get("lines")
    if not isinstance(lines, list) or not lines:
        return ["report has no check lines"]
    bad = [i for i, x in enumerate(lines) if not (isinstance(x, dict) and x.get("status") in LINE_STATUSES
                                                  and isinstance(x.get("check"), str) and x.get("check")
                                                  and isinstance(x.get("detail"), str))]
    if bad:
        return [f"malformed check line(s) at {bad[:5]}"]
    names = [x["check"] for x in lines]
    miss = [c for c in REQUIRED_CHECKS if names.count(c) != 1]
    if miss:
        out.append(f"required check line(s) missing or repeated: {miss[:6]}")
    eps = r.get("episodes")
    ebs = r.get("episodes_by_split")
    if not isinstance(ebs, dict) or not all(_is_int(v) and v >= 0 for v in ebs.values()):
        out.append("episodes_by_split missing / unreadable")
        ebs = {}
    if isinstance(eps, list) and eps and all(_is_int(e) for e in eps):
        # r6 2: the split counts must be those of the REGISTERED plan for these episode ids (no empty / wrong split map)
        want = plan_splits(r.get("scene"), eps)
        if want is None:
            out.append("episodes_by_split cannot be reconciled (episode not in the registered plan / scene unknown)")
        else:
            if {k: v for k, v in ebs.items() if v} != want:
                out.append(f"episodes_by_split {ebs} != the registered plan's {want}")
            ebs = dict(want)                            # the TRAIN-line rule follows the plan, not the stored map
    if ebs.get("train", 0) > 0:                         # r5 2: a batch with TRAIN episodes carries the pixel checks
        tmiss = [c for c in TRAIN_CHECKS if names.count(c) != 1]
        if tmiss:
            out.append(f"TRAIN quality line(s) missing or repeated: {tmiss[:6]}")
    st = {x["status"] for x in lines}
    implied = "FAIL" if "FAIL" in st else ("WARN" if "WARN" in st else "PASS")
    if r.get("verdict") != implied:
        out.append(f"stored verdict {r.get('verdict')!r} != the verdict its lines imply ({implied})")
    eps = r.get("episodes")
    if not (isinstance(eps, list) and eps and all(_is_int(e) for e in eps) and len(set(eps)) == len(eps)):
        out.append("episode list missing / empty / not unique integers")
    return out


def qc_report_ok(rec: dict, qc_dir: str | None = None) -> list:
    """The receipt's QC report must be a batch report in this scene's qc folder, exist, match its sha256, be PASS
    with no FAIL / WARN line, and cover every new committed episode (quota-skipped TRAIN episodes have no rows)."""
    rel, h = rec.get("qc_report"), rec.get("qc_report_sha256")
    if not isinstance(rel, str) or not isinstance(h, str) or not rel or len(h) != 64:
        return ["receipt names no QC report / hash"]
    p = os.path.join(REPO, rel)        # repo-relative: the receipt is written in WSL, read on Windows
    qd = os.path.normcase(os.path.realpath(qc_dir or os.path.join(scene_dir(rec["scene"]), "qc")))
    if os.path.normcase(os.path.dirname(os.path.realpath(p))) != qd or not os.path.basename(p).startswith("batch_"):
        return [f"QC report {rel} is not a batch report of this scene's qc folder"]
    if not os.path.exists(p):
        return [f"QC report {rel} missing"]
    if sha_file(p) != h:
        return ["QC report changed after the receipt"]
    try:
        r = json.load(open(p, encoding="utf-8"))
    except Exception as exc:
        return [f"QC report unreadable: {exc!r}"]
    if r.get("scene") != rec["scene"]:
        return ["QC report is for another scene"]
    if r.get("verdict") != "PASS":
        return [f"QC report verdict {r.get('verdict')!r}"]
    lines = r.get("lines")
    if not isinstance(lines, list) or not lines:
        return ["QC report has no check lines"]
    worse = [x.get("check") if isinstance(x, dict) else x for x in lines
             if not isinstance(x, dict) or x.get("status") not in ("PASS", "NOTE", "INFO")]
    if worse:
        return [f"QC report says PASS but has FAIL / WARN / malformed lines {worse[:5]}"]
    cp = report_content_problems(r)
    if cp:
        return [f"QC report contents invalid: {'; '.join(cp)}"]
    need = set(rec["new_episodes"]) - set(rec["new_skipped"])
    have = set(r.get("episodes") or [])
    if not need <= have:
        return [f"QC report does not cover the new episodes {sorted(need - have)[:5]}"]
    return []


def thermal_test_ok(rec: dict) -> list:
    bad = []
    m = rec.get("pause_marker") or {}
    rb = m.get("running_before") or {}
    if not rec["thermal"]:
        bad.append("the pause did not fire")
    if not m.get("test"):
        bad.append("the marker is not a test marker")
    if not marker_induced(m):
        bad.append("the marker is not the induced pause-test trigger")
    if not (rb.get("job", 0) >= 1 and rb.get("simulator", 0) >= 1):
        bad.append(f"the capture job and the simulator were not both running in the owned group before the stop ({rb})")
    ps = rec.get("post_stop") or {}
    if ps.get("jobs_left") != 0 or ps.get("simulators_left") != 0:
        bad.append(f"processes left after the stop: {ps}")
    u = ps.get("util_after_stop")
    if not (isinstance(u, list) and len(u) == 3 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                         for x in u)):
        bad.append(f"need 3 valid GPU utilization samples after the stop, got {u!r}")
    elif any(x > 20 for x in u):
        bad.append(f"GPU utilization after the stop {u} (> 20 % in some sample)")
    return bad


def decide_post(rec, invocation: str, scene: str, pipeline: str, wsl_rc: int, sess: dict, now_epoch: float,
                deadline_epoch: float, first_batch_only: bool, qc_bad: list | None = None,
                qc_dir: str | None = None, markers: list | None = None) -> dict:
    def stop(why, terminal=True, window_ended=False):
        if window_ended:
            close_window(sess, why, now_epoch, invocation)
        elif terminal:
            old = sess.get("terminal") or {}
            sess["terminal"] = {"why": why, "t": int(now_epoch), "invocation": invocation,
                                "window_ended": bool(old.get("window_ended"))}
        return {"action": "stop", "code": 1, "why": why}

    # 0. thermal closure FIRST, before any validation that could fail (r3 1, r4 1): files, receipt and exit code, valid
    #    receipt or not. real_pauses never raises; the closure is recorded in the session before anything else runs (the
    #    caller saves it even if a later step raises: main() persists a closure found here on every path).
    real = real_pauses(markers or [], rec, pipeline, wsl_rc, invocation)
    if real:
        close_window(sess, "THERMAL PAUSE: a pause ends this capture window (no restart): " + "; ".join(real[:3]),
                     now_epoch, invocation)
        try:
            bad = validate_receipt(rec, invocation, scene, pipeline)
        except Exception as exc:
            bad = [f"receipt validation crashed: {exc!r}"]
        if not bad:
            sess["outstanding"] = None
            sess["history"].append({"t": int(now_epoch), "invocation": invocation, "pipeline": pipeline,
                                    "capture_rc": rec["capture_rc"], "qc_rc": rec["qc_rc"],
                                    "new": len(rec["new_episodes"])})
        return {"action": "stop", "code": 1,
                "why": "THERMAL PAUSE: a pause ends this capture window (no restart): " + "; ".join(real[:3])
                       + (f"; receipt invalid: {'; '.join(bad)}" if bad else "")}
    try:
        bad = validate_receipt(rec, invocation, scene, pipeline)
    except Exception as exc:                              # a malformed receipt is invalid, never a crash (r4 1)
        bad = [f"receipt validation crashed: {exc!r}"]
    if bad:
        # the outstanding invocation stays recorded: nothing may launch until a person reconciles it
        return stop("receipt invalid: " + "; ".join(bad))
    sess["outstanding"] = None
    crc, qrc = rec["capture_rc"], rec["qc_rc"]
    new = len(rec["new_episodes"])
    if wsl_rc != crc:
        return stop(f"WSL exit code {wsl_rc} != receipt capture_rc {crc}")
    if crc == RC_YIELD:
        if new == 0 and rec["failed_after"] == 0 and not qc_bad and rec["qc_rc"] in (-1, QC_NONE):
            sess["history"].append({"t": int(now_epoch), "invocation": invocation, "pipeline": pipeline,
                                    "capture_rc": crc, "qc_rc": qrc, "new": 0})
            return {"action": "done", "code": 0, "why": "YIELD: another job waits for the sim lock (Kitchen 1 first, "
                                                        "A28 15.9); nothing started; resume later"}
        return stop(f"a yield receipt with committed / failed episodes or an open QC (rc {qrc}): investigate")
    sess["last_done"] = rec["done_after"]
    if pipeline in CAPTURING:
        sess["idle_runs"] = 0 if new > 0 else sess["idle_runs"] + 1
    sess["history"].append({"t": int(now_epoch), "invocation": invocation, "pipeline": pipeline, "capture_rc": crc,
                            "qc_rc": qrc, "new": new})
    # 1. thermal (test or real) first
    if pipeline == "thermaltest":
        tb = thermal_test_ok(rec)
        if tb:
            return stop("THERMAL TEST FAILED: " + "; ".join(tb))
        return {"action": "done", "code": 0, "why": "THERMAL TEST PASS"}
    if rec["thermal"] or crc == RC_PAUSE:
        return stop("THERMAL PAUSE: a pause ends this capture window (no restart)", window_ended=True)
    if rec.get("guard_killed") is True:
        return stop("the thermal guard had to be killed during cleanup: its outcome is unknown; no capture without it")
    # nothing started (the lock was busy / a foreign simulator ran): a plain stop, no guard involved
    if crc in (RC_LOCK, RC_FOREIGN_SIM) and new == 0 and rec["failed_after"] == 0 and not qc_bad:
        return stop("sim lock busy: another job holds /home/arash/sim.lock" if crc == RC_LOCK else
                    "a simulator not started by this batch was running: nothing started", terminal=False)
    if crc in (RC_GUARD_START, RC_GUARD_DIED) or not rec["guard_ok"]:
        return stop("the thermal guard did not start or died: no capture without it")
    # 2a. any non-PASS QC report still unresolved stops, whatever this receipt says (r2 6)
    if qc_bad:
        return stop(f"QC report(s) not PASS and unresolved: {qc_bad[:3]}")
    if crc == RC_PREQC:
        return stop(f"earlier committed episodes were re-checked before the launch and QC did not pass (rc {qrc}): "
                    f"nothing captured; read qc/summary.log")
    # 2. QC before anything else: every committed episode checked and PASS; nothing new -> QC must say so (4)
    if pipeline in CAPTURING:
        if new > 0:
            if qrc != QC_PASS:
                names = {1: "FAIL", 2: "could not run", 3: "quality WARN", 4: "nothing new", 5: "unresolved earlier "
                         "batch", -1: "not run", 124: "timed out", 137: "killed"}
                return stop(f"QC rc {qrc} ({names.get(qrc, 'unknown')}) on {new} newly committed episode(s): read "
                            f"qc/summary.log, consult Astra")
            qb = qc_report_ok(rec, qc_dir)
            if qb:
                return stop("QC report invalid: " + "; ".join(qb))
        elif crc not in (RC_LOCK, RC_DEADLINE, RC_FOREIGN_SIM, RC_GUARD_START, RC_GUARD_DIED, RC_SNAPSHOT) \
                and qrc != QC_NONE:
            return stop(f"no new episode but QC rc {qrc} (expected 4 = nothing new)")
    if rec["failed_after"] > 0:
        return stop(f"{rec['failed_after']} episode(s) recorded failed by the capture: decide before continuing")
    # 3. the capture's own outcome
    if crc == RC_LOCK:
        return stop("sim lock busy: another job holds /home/arash/sim.lock", terminal=False)
    if crc == RC_FOREIGN_SIM:
        return stop("a simulator not started by this batch was running: nothing started", terminal=False)
    if crc in (RC_REFUSED, RC_CLASSES, RC_SNAPSHOT, RC_DIED):
        return stop(f"capture refused / class list mismatch / snapshot failed / batch died (rc {crc})")
    if crc == RC_DEADLINE:
        return {"action": "done", "code": 0, "why": "HARD STOP: the batch did not start (deadline)"}
    if crc in RC_TIMEOUT:
        if deadline_epoch - now_epoch < GRACE_S + QC_BUDGET_S + 60:
            return {"action": "done", "code": 0, "why": "HARD STOP: the running batch was ended at the deadline; the "
                                                        "partial episode is discarded and redone on resume"}
        if new > 0 and sess["hang_retries"] < MAX_HANG_RETRIES:
            sess["hang_retries"] += 1
            return {"action": "retry", "code": 0, "why": "batch timeout (hang) after progress, QC PASS: one resume "
                                                         "with a fresh simulator process"}
        return stop("batch timeout (hang) without progress or twice")
    if crc == RC_ABORT:
        if rec.get("guard_burst") and sess["burst_retries"] < MAX_BURST_RETRIES and sess["idle_runs"] < MAX_IDLE:
            sess["burst_retries"] += 1
            return {"action": "retry", "code": 0, "why": "render-guard burst, QC of committed episodes PASS: one resume "
                                                         "with a fresh simulator process (a WSL reset only by a person)"}
        return stop("fail-fast abort (not a first render-guard burst)")
    if crc != RC_OK:
        return stop(f"capture exited {crc}")
    if pipeline == "probe":
        return {"action": "done", "code": 0, "why": "probe done: record the class list in the ledger before capture"}
    sess["batches"] += 1
    if rec.get("complete"):
        return {"action": "done", "code": 0, "why": f"{pipeline} capture COMPLETE"}
    if new == 0:
        return stop(f"a {pipeline} batch committed nothing")
    if first_batch_only:
        return {"action": "done", "code": 0, "why": "first batch done, QC PASS: stop for the deep check"}
    return {"action": "continue", "code": 0, "why": f"batch {sess['batches']} QC PASS ({new} new episodes)"}


QC_DIRS = ("qc", "qc_amb")      # K2-AMB: the amb store keeps its own QC evidence; stops are checked scene-wide


def qc_dir_for(scene: str, pipeline: str = "main") -> str:
    return os.path.join(scene_dir(scene), "qc_amb" if pipeline == "amb" else "qc")


def qc_open(scene: str) -> dict | None:
    for sub in QC_DIRS:
        p = os.path.join(scene_dir(scene), sub, "qc_state.json")
        if not os.path.exists(p):
            continue
        st = json.load(open(p, encoding="utf-8"))
        op = st.get("open")
        if not op:
            continue
        res = os.path.join(scene_dir(scene), sub, f"resolution_batch_{op['batch']:03d}.json")
        if os.path.exists(res) and not valid_resolution(res):
            continue
        return dict(op, qc_dir=sub)
    return None


def qc_unresolved(scene: str, qc_dir: str | None = None) -> list:
    """Every QC batch report that is not PASS (or cannot be read) and has no valid resolution_batch_NNN.json; without
    qc_dir: scene-wide over QC_DIRS (amb entries prefixed 'qc_amb/')."""
    if qc_dir is None:
        out = []
        for sub in QC_DIRS:
            out += [(f"{sub}/" if sub != "qc" else "") + x
                    for x in qc_unresolved(scene, os.path.join(scene_dir(scene), sub))]
        return out
    qd = qc_dir
    out = []
    if not os.path.isdir(qd):
        return out
    for f in sorted(os.listdir(qd)):
        if not (f.startswith("batch_") and f.endswith(".json")):
            continue
        try:
            r = json.load(open(os.path.join(qd, f), encoding="utf-8"))
            v, n = r.get("verdict"), int(r.get("batch"))
        except Exception as exc:
            out.append(f"{f}: unreadable ({exc!r})")
            continue
        cp = report_content_problems(r)
        if v == "PASS" and not cp:
            continue
        res = os.path.join(qd, f"resolution_batch_{n:03d}.json")
        if not os.path.exists(res) or valid_resolution(res):
            out.append(f"{f}: {v}" + (f" (contents invalid: {cp[:2]})" if cp else ""))
    return out


def pending_files(scene: str) -> list:
    """pending_<stamp>.json: a batch could not write its receipt in time (work left durably PENDING, r3 4). Each one
    blocks every later launch until a person reconciles it and moves it out of the scene folder."""
    d = scene_dir(scene)
    return sorted(f for f in os.listdir(d) if f.startswith("pending_") and f.endswith(".json")) if os.path.isdir(d) else []


def resolve_terminal(scene: str, sess: dict) -> None:
    """A terminal stop is lifted only by a valid session_resolution_<inv8>.json bound to it; a thermal pause never in this window."""
    t = sess.get("terminal")
    if t and t.get("window_ended"):
        # A28-ii-SM: a window-ending stop whose EVERY segment is the evidence of one exempt pause (duplicate persistence
        # of that pause included) is lifted by its record; any other or additional cause keeps the window closed
        inv = str(t.get("invocation") or "")
        mf = _a28sm_marker_of(scene, inv)
        if mf and a28sm_evidence_only(t.get("why"), mf, prefixed=True, scene=scene):
            p = os.path.join(scene_dir(scene), "thermal", f"a28sm_{inv[:8]}.json")
            sess["resolutions"].append({"file": p, "sha256": sha_file(p), "lifted": t, "rule": "A28-ii-SM"})
            sess["terminal"] = None
        return
    if not t:
        return
    # K2-TOPUP 2026-10-17 (Astra K23-PARALLEL-r2): sessions are keyed by the window's end day, so a per-session counter
    # (session_resolution_<n>.json) re-read OLD, unrelated files in every new window. A resolution is now BOUND to the stop it
    # lifts: session_resolution_<first 8 of the stopped invocation>.json whose "invocation" field equals it in full.
    inv = str(t.get("invocation") or "")
    if len(inv) < 8:
        return
    p = os.path.join(scene_dir(scene), f"session_resolution_{inv[:8]}.json")
    try:
        bound = json.load(open(p, encoding="utf-8")).get("invocation") == inv if os.path.exists(p) else False
    except Exception:
        bound = False
    if bound and not valid_resolution(p):
        sess["resolutions"].append({"file": p, "sha256": sha_file(p), "lifted": t})
        sess["terminal"] = None
        if (t.get("why") or "").startswith("receipt invalid") or "no decided receipt" in (t.get("why") or ""):
            sess["outstanding"] = None


# ======================================================================================================================
# WSL side: snapshot before the batch, receipt after it (run by capture_inner_k23.sh with the thor-env python).
# ======================================================================================================================
def store_root(pipeline: str = "main") -> str:
    """The store a pipeline writes: amb -> kitchen_v3_amb (K2-AMB, Astra K2-AMB-r1 2), every other -> kitchen_v3."""
    if TEST_MODE and os.environ.get("K23_TEST_STORE_ROOT"):
        r = os.environ["K23_TEST_STORE_ROOT"]               # integration test only
        return r + "_amb" if pipeline == "amb" else r
    return os.path.join(os.path.expanduser("~"), "thor-data", "kitchen_v3_amb" if pipeline == "amb" else "kitchen_v3")


def store_state(scene: str, full: bool = False, pipeline: str = "main") -> dict:
    p = os.path.join(store_root(pipeline), scene, "capture_state.json")
    if not os.path.exists(p):
        return {} if full else {"done": [], "failed": {}}
    st = json.load(open(p, encoding="utf-8"))
    if full:
        return st
    return {"done": list(st.get("done_episodes") or []), "failed": dict(st.get("failed_episodes") or {})}


def recorded_episodes(scene: str, pipeline: str = "main") -> set:
    p = os.path.join(store_root(pipeline), scene, "frames.jsonl")
    out = set()
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith('{"episode_record"'):
                    out.add(json.loads(line)["episode_record"]["episode"])
    return out


def write_receipt(a) -> dict:
    before = json.load(open(a.snapshot, encoding="utf-8"))
    after = store_state(a.scene, pipeline=a.pipeline)
    new = sorted(set(after["done"]) - set(before["done"]))
    rec_eps = recorded_episodes(a.scene, a.pipeline) if new else set()
    skipped = [e for e in new if e not in rec_eps]           # 'done' without a record = quota-skipped TRAIN episodes
    sys.path.insert(0, SIM)
    import kitchen_data as kdat
    split = {e["episode"]: e["split"] for e in kdat.episode_plan(a.scene, kdat.CAPTURE["episodes"])
             + kdat.topup_train_plan(a.scene, kdat.TOPUP_TRAIN_MAX) + kdat.close_plan(a.scene, kdat.CLOSE_EPISODES)
             + kdat.amb_plan(a.scene) + _parity_plan(a.scene, kdat)}
    rec = {"invocation": a.invocation, "scene": a.scene, "pipeline": a.pipeline, "stamp": a.stamp,
           "capture_rc": int(a.capture_rc), "qc_rc": int(a.qc_rc), "guard_ok": a.guard_ok == "1" and a.guard_killed != "1",
           "guard_rc": a.guard_rc, "guard_killed": a.guard_killed == "1", "phase": a.phase,
           "done_before": len(before["done"]), "done_after": len(after["done"]), "new_episodes": new,
           "new_skipped": skipped, "skipped_not_train": [e for e in skipped if split.get(e) != "train"],
           "failed_after": len(after["failed"]), "failed_episodes": after["failed"],
           "thermal": bool(a.marker and (os.path.exists(a.marker) or os.path.exists(a.marker + ".tmp"))),
           "ended": time.strftime("%Y-%m-%d %H:%M:%S")}
    if rec["thermal"]:
        try:
            rec["pause_marker"] = json.load(open(a.marker, encoding="utf-8"))
        except Exception as exc:                          # still a pause (fail closed); never induced
            rec["pause_marker"] = {"unreadable": repr(exc)[:200]}
    if a.post_stop:
        rec["post_stop"] = json.loads(a.post_stop)
    try:
        with open(a.log, encoding="utf-8", errors="replace") as fh:
            fh.seek(int(a.log_offset))
            part = fh.read()
        rec["guard_burst"] = "ABORTED (fail-fast)" in part and "render guard" in part
        rec["complete"] = f"COMPLETE {a.pipeline}" in part if a.pipeline in CAPTURING else False
    except Exception:
        part = ""
        rec["guard_burst"], rec["complete"] = False, False
    # a 'done' episode without a record must be one the capture itself logged as quota-skipped (r2 6)
    logged = {int(m) for m in re.findall(r"ep\s+(\d+) \[train\] skipped: every train target already met", part)}
    # ... AND the stored quota must demonstrate it (r3 2): the same rule as layout_fingerprint / k23_qc
    import layout_fingerprint as lf
    und = lf.quota_skip_unexplained(store_state(a.scene, full=True, pipeline=a.pipeline), skipped, a.scene) if skipped else {}
    qst = os.path.join(qc_dir_for(a.scene, a.pipeline), "qc_state.json")       # K2-AMB r2 1: the pipeline's QC
    # ... AND the QC must have recorded its per-episode skip provenance when it first saw it (r4 2)
    try:
        prov = (json.load(open(qst, encoding="utf-8")).get("skips") or {}) if skipped else {}
    except Exception:
        prov = {}
    try:
        qb = json.load(open(qst, encoding="utf-8")).get("batches") if skipped else None
    except Exception:
        qb = -1
    rec["skipped_unexplained"] = [e for e in skipped if e not in logged or e in und or str(e) not in prov
                                  or skip_entry_problems(str(e), prov.get(str(e)), a.scene, part,
                                                         qdir=os.path.dirname(qst), batches=qb)]
    if new and os.path.exists(qst):
        n = json.load(open(qst, encoding="utf-8")).get("batches", 0)
        rp = os.path.join(qc_dir_for(a.scene, a.pipeline), f"batch_{n:03d}.json")
        if os.path.exists(rp):
            rec["qc_report"], rec["qc_report_sha256"] = os.path.relpath(rp, REPO).replace("\\", "/"), sha_file(rp)
    out = os.path.join(scene_dir(a.scene), "last_batch.json")
    with open(out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(out + ".tmp", out)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sn = sub.add_parser("snapshot")
    sn.add_argument("--scene", choices=SCENES, required=True)
    sn.add_argument("--out", required=True)
    sn.add_argument("--pipeline", choices=PIPELINES, default="main")      # K2-AMB: the snapshot reads the pipeline's store
    rc = sub.add_parser("receipt")
    for k in ("scene", "pipeline", "invocation", "stamp", "capture-rc", "qc-rc", "guard-ok", "snapshot", "log",
              "log-offset"):
        rc.add_argument("--" + k, required=True)
    rc.add_argument("--marker", default="")
    rc.add_argument("--post-stop", default="")
    rc.add_argument("--guard-rc", default="")
    rc.add_argument("--guard-killed", default="0")
    rc.add_argument("--phase", default="normal")
    for name in ("auth", "pre", "post", "template", "reset-done"):
        p = sub.add_parser(name)
        p.add_argument("--scene", choices=SCENES, default="FloorPlan15")
        p.add_argument("--pipeline", choices=PIPELINES, default="main")
        p.add_argument("--deadline", default=None)
        p.add_argument("--batch-minutes", type=int, default=30)
        p.add_argument("--invocation", default=None)
        p.add_argument("--wsl-rc", type=int, default=None)
        p.add_argument("--first-batch-only", action="store_true")
        p.add_argument("--check-only", action="store_true")
    a = ap.parse_args(argv)
    try:
        ov = overrides_outside_test()
        if ov and a.cmd not in ("template",):
            print(json.dumps({"action": "stop", "code": 1, "ok": False, "why": [f"K23_* overrides set outside test "
                                                                             f"mode: {ov}"]}))
            return 2
        if a.cmd == "snapshot":
            with open(a.out, "w", encoding="utf-8") as fh:
                json.dump(store_state(a.scene, pipeline=a.pipeline), fh)
            return 0
        if a.cmd == "receipt":
            r = write_receipt(a)
            print(json.dumps({k: r[k] for k in ("capture_rc", "qc_rc", "done_before", "done_after", "thermal")}))
            return 0
        if a.cmd == "template":
            out = template()
            with open(os.path.join(HERE, "authorization.TEMPLATE.json"), "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=1)
            print(json.dumps({"written": "authorization.TEMPLATE.json"}))
            return 0
        if a.cmd == "auth":
            print(json.dumps(cmd_auth(a)))
            return 0
        deadline = dt.datetime.strptime(a.deadline, "%Y-%m-%d %H:%M")
        day = deadline.strftime("%Y-%m-%d")
        now = time.time()
        early = None
        if a.cmd == "post":
            # r5 1: the thermal evidence is read and a real pause PERSISTED (closure file) before anything else --
            # before the session file is read, before any listing or diagnostic that could fail
            early = early_closure(a.scene, day, a.pipeline, a.wsl_rc, a.invocation or "", now)
        sess = load_session(a.scene, day)
        if a.cmd == "post" and early and early.get("real") and not (sess.get("terminal") or {}).get("window_ended"):
            close_window(sess, "THERMAL PAUSE (persisted first): " + "; ".join(early["real"][:3]), now,
                         a.invocation or "")
            save_session(a.scene, day, sess)
        if a.cmd == "reset-done":             # a PERSON ran the WSL reset after a passed preflight (-ResetWslNow)
            sess["resets"] += 1
            save_session(a.scene, day, sess)
            print(json.dumps({"resets": sess["resets"], "cap": MAX_RESETS}))
            return 0
        if a.cmd == "pre":
            rec = load_auth()
            bad = check_auth(rec, a.scene, a.pipeline, dt.datetime.now(), ledger_headings())
            if bad:
                print(json.dumps({"action": "stop", "why": "not authorized now: " + "; ".join(bad)}))
                return 0
            if os.path.exists(closure_path(a.scene, day)) and not closure_exempt(a.scene, day):
                if not (sess.get("terminal") or {}).get("window_ended"):
                    close_window(sess, "THERMAL PAUSE (closure file)", now, "")
                    save_session(a.scene, day, sess)
                print(json.dumps({"action": "stop", "why": "a thermal pause ended this capture window (closure file "
                                                           f"{os.path.basename(closure_path(a.scene, day))}; no restart)"}))
                return 0
            resolve_terminal(a.scene, sess)
            out = decide_pre(sess, now, deadline.timestamp(), a.batch_minutes, qc_open(a.scene), a.pipeline,
                             window_of(rec)[1].timestamp(), qc_unresolved(a.scene), pause_markers(a.scene, day),
                             pending_files(a.scene))
            save_session(a.scene, day, sess)          # the OUTSTANDING invocation is durable before the launch
            print(json.dumps(out))
            return 0
        mk, rec, rexc = early["markers"], early["rec"], early["rec_error"]
        if rexc is not None:
            # missing output = STOP; the outstanding invocation stays; a pause shown by a marker or the exit code
            # still ends the window (r3 1) -- already persisted above (r5 1); the diagnostics below may fail safely
            real = early["real"]
            try:
                pend = pending_files(a.scene)[:2]
            except Exception as exc2:
                pend = [f"unreadable ({exc2!r})"]
            why = f"no readable receipt ({rexc!r}); pending: {pend}"
            if real:
                close_window(sess, "THERMAL PAUSE: " + "; ".join(real[:3]) + "; " + why, now, a.invocation or "")
            else:
                old = sess.get("terminal") or {}
                sess["terminal"] = {"why": why, "t": int(now), "invocation": a.invocation,
                                    "window_ended": bool(old.get("window_ended"))}
            save_session(a.scene, day, sess)
            print(json.dumps({"action": "stop", "code": 1, "why": ("THERMAL PAUSE; " if real else "") + why
                              + ": missing output = STOP"}))
            return 0
        # r4 1: the thermal closure is PERSISTED before any fallible step (QC listing, validation, the decision)
        real0 = real_pauses(mk, rec, a.pipeline, a.wsl_rc, a.invocation or "")
        if real0 and not (sess.get("terminal") or {}).get("window_ended"):
            close_window(sess, "THERMAL PAUSE (persisted before validation): " + "; ".join(real0[:3]), now,
                         a.invocation or "")
            save_session(a.scene, day, sess)
        try:
            out = decide_post(rec, a.invocation or "", a.scene, a.pipeline, a.wsl_rc, sess, now, deadline.timestamp(),
                              a.first_batch_only, qc_unresolved(a.scene), qc_dir=qc_dir_for(a.scene, a.pipeline),
                              markers=mk)
        except Exception as exc:
            old = sess.get("terminal") or {}
            sess["terminal"] = {"why": f"decide post failed: {exc!r}"[:2000], "t": int(now),
                                "invocation": a.invocation, "window_ended": bool(old.get("window_ended") or real0)}
            out = {"action": "stop", "code": 1, "why": f"decide post failed: {exc!r}"}
        save_session(a.scene, day, sess)
        print(json.dumps(out))
        return 0
    except Exception as exc:
        print(json.dumps({"action": "stop", "code": 1, "ok": False, "why": [f"decide failed: {exc!r}"]}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
