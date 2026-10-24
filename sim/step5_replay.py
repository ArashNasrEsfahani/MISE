"""Step 5 / A5-replay - convert step-4 run logs into event-contract streams and replay them through step-5 memory.

Offline. Standard library plus the step-5 modules; no simulator, no WSL, no Ollama, no network. The source JSONL files
are COPIED into a temp directory first (sha256 recorded before and after; the originals are only ever read).

WHAT IS TAKEN FROM A STEP-4 ROW (explicit extraction, never a dict copy - this is the whitelist):
    status, level, steps[i].{action, object, to, ok, cause, under, retries, prims}, retries, prims, exec_ok,
    goal_ok (only as the post-terminal `admission_verdict`, the declared privileged use), audit.invalid /
    audit.violations (integrity booleans only).
EVERYTHING ELSE IS DROPPED: audit.errors_logged / counts / episode, pre, detail, why, verify, holding, events (pixel
points), search (visited / opened / plan lists), hash, ts, elapsed_s, seed_tracks, sweep_prims, told_holding, first_fail,
and the row id (replaced by an opaque content-addressed episode id; the source id lives only in the evaluator-side
manifest). Missing = unknown: the logs carry no world coordinates, no per-attempt causes and no recovery kinds, so no
sighting / inspection / open / attempt events are produced and `where_is` stays explore-only.

The only text that comes from outside the log is the PUBLIC instruction (sim/dataset/general.json en/fa), joined by id,
whose declared level must equal the row's level. Reference plans, goals, setup and `reviewed` are never read.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)                # append, not insert: sim/trace.py must not shadow the stdlib `trace`

from ltm_store import canon, sha256_hex    # noqa: E402

REPO = Path(HERE).parent
DATASET = Path(HERE) / "dataset" / "general.json"
REPLAY_VERSION = "1"
KITCHEN, FRAME, MAP_REV = "FloorPlan1", "f0", "m0"

#: the step-4 action space (honest_skills) as it appears in the logs; anything else rejects the row
VERBS = frozenset({"pickup", "drop", "put_in", "pick_and_place", "open", "close", "close_all", "toggle_on", "toggle_off",
                   "slice", "fill", "empty", "heat", "cook", "wash", "clean", "dirty", "break", "go_to", "search_for",
                   "crouch", "stand"})
CAUSES = frozenset({"not_detected", "wrong_target", "unreachable", "action_refused", "state_unrecognised",
                    "retry_exhausted", "no_cutting_tool"})   # no_cutting_tool: A26 / K-B2b 2026-10-13
ITEM_STATUS = frozenset({"ok", "FAIL", "skipped"})
NAME_RE = re.compile(r"^[A-Za-z]{1,32}$")
#: raw row fields that are NEVER read (documented, and checked by the perturbation test)
EXCLUDED_ROW_FIELDS = ("id", "hash", "detail", "cause", "first_fail", "pre", "audit.errors_logged", "audit.counts",
                       "audit.episode", "audit.errors", "told_holding", "seed_tracks", "sweep_prims", "elapsed_s", "ts",
                       "mode")
EXCLUDED_STEP_FIELDS = ("why", "verify", "holding", "search", "events", "cause_text")


class ReplayError(ValueError):
    """A source file or row cannot be converted (the row is rejected and counted; it never reaches memory)."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------------- sources
def default_sources(repo=REPO) -> list:
    """The logs named by the task: the console re-run and every final-fixes checkpoint run (relative to the repo)."""
    repo = Path(repo)
    out = []
    main = repo / "sim" / "step4" / "rerun-2026-10-09-console" / "refplans_honest.jsonl"
    if main.exists():
        out.append(main)
    ck = repo / "sim" / "step4" / "_checkpoints" / "final-fixes"
    if ck.exists():
        out.extend(sorted(ck.glob("*/refplans_honest.jsonl")))
    return out


def snapshot_sources(dest, paths=None, repo=REPO) -> list:
    """Copy each source into `dest` (read-only use of the original), recording sha256 of the original before the copy,
    of the copy, and of the original again afterwards (a concurrent writer shows up as `source_changed`)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    repo = Path(repo)
    out = []
    for n, p in enumerate(paths if paths is not None else default_sources(repo)):
        p = Path(p)
        try:
            rel = p.resolve().relative_to(repo.resolve()).as_posix()
        except ValueError:
            rel = p.name
        before = sha256_file(p)
        tag = f"src{n:03d}"
        tgt = dest / f"{tag}.jsonl"
        shutil.copyfile(p, tgt)
        after = sha256_file(p)
        out.append({"tag": tag, "source": rel, "sha256_before": before, "sha256_after": after,
                    "sha256_copy": sha256_file(tgt), "copy": str(tgt), "bytes": tgt.stat().st_size,
                    "copy_matches_source": sha256_file(tgt) == before, "source_changed": before != after})
    return out


def read_rows(path) -> tuple:
    """(rows, bad_lines). A torn or non-JSON line is counted, not fatal (resumable logs can end mid-write)."""
    rows, bad = [], 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip().lstrip("﻿")
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(r, dict):
                rows.append(r)
            else:
                bad += 1
    return rows, bad


def load_items(path=DATASET) -> dict:
    """id -> {level, en, fa}: the PUBLIC part of the dataset. Plans, goals, setup and review marks are not read."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {i["id"]: {"level": i["level"], "en": i["en"], "fa": i["fa"]} for i in d["items"]}


def load_archived(path=DATASET) -> frozenset:
    """A26 (K-B2b 2026-10-13): ids the successor dataset archived (Knife items). Their historical rows are SKIPPED, never replayed
    into memory and never counted as rejections."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return frozenset(a["id"] for a in d.get("archived") or [])


# --------------------------------------------------------------------------------------------------- conversion
def _bool(v, what):
    if not isinstance(v, bool):
        raise ReplayError(f"{what}: expected bool")
    return v


def _count(v, what, hi):
    if isinstance(v, bool) or not isinstance(v, int) or not (0 <= v <= hi):
        raise ReplayError(f"{what}: expected int in [0, {hi}]")
    return v


def _concepts(text):
    try:
        from ltm_concepts import parse
    except ImportError:                  # pragma: no cover
        from sim.ltm_concepts import parse
    p = parse(text)
    return {"roles": p["roles"], "ok": bool(p["ok"]), "parser": str(p.get("parser", "")),
            "lexicon": str(p.get("lexicon", ""))}


def row_events(row: dict, item: dict, *, episode: str, kitchen=KITCHEN, epoch=0, frame=FRAME, map_rev=MAP_REV,
               lang="en", mode="practice"):
    """Whitelisted events for ONE row, or None for a skipped (inapplicable) row. Raises ReplayError on a malformed row."""
    status = row.get("status")
    if status == "skipped":
        return None
    if status not in ITEM_STATUS:
        raise ReplayError(f"row status {status!r} is not a completed item")
    level = row.get("level")
    if level not in ("L1", "L2", "L3") or item is None or item["level"] != level:
        raise ReplayError("level missing or inconsistent with the public dataset")
    steps_in = row.get("steps")
    if not isinstance(steps_in, list) or len(steps_in) > 32:
        raise ReplayError("steps missing")
    audit = row.get("audit") if isinstance(row.get("audit"), dict) else {}
    invalid = bool(audit.get("invalid") is True or (isinstance(audit.get("violations"), int) and audit["violations"] > 0))
    base = {"schema": "s5.ev/1", "kitchen": kitchen, "epoch": epoch, "frame": frame, "map_rev": map_rev, "episode": episode}
    evs: list = []

    def add(kind, **kw):
        evs.append(dict(base, ev=len(evs) + 1, kind=kind, t=len(evs) + 1, **kw))

    add("episode_start", mode=mode, world="reset")
    text = item[lang]
    add("task_context", level=level, lang=lang, text=text, concepts=_concepts(text) if lang == "en" else None)
    if evs[-1]["concepts"] is None:
        del evs[-1]["concepts"]
    for pos, s in enumerate(steps_in):
        if not isinstance(s, dict):
            raise ReplayError("step is not an object")
        verb = s.get("action")
        if verb not in VERBS:
            raise ReplayError(f"verb {str(verb)[:20]!r} outside the action space")
        obj, to = s.get("object"), s.get("to")
        if obj is not None and not (isinstance(obj, str) and NAME_RE.match(obj)):
            raise ReplayError("object is not a type name")
        if to is not None and not (isinstance(to, str) and NAME_RE.match(to)):
            raise ReplayError("destination is not a type name")
        ok = _bool(s.get("ok"), "step.ok")
        cause = s.get("cause")
        cause = cause if cause in CAUSES else None          # unknown / free text -> unknown
        under = s.get("under")
        under = under if under in CAUSES else None
        i = pos                                              # the log's i must equal the position; else positions rule
        kw = {"i": i, "verb": verb}
        if obj:
            kw["otype"] = obj
        if to:
            kw["to"] = to
        add("step_start", **kw)
        so = {"i": i, "ok": ok, "retries": _count(s.get("retries", 0), "step.retries", 50),
              "prims": _count(s.get("prims", 0), "step.prims", 100000)}
        if not ok and cause:
            so["cause"] = cause
        if not ok and under:
            so["under"] = under
        add("step_outcome", **so)
    term = {"exec_ok": bool(row.get("exec_ok") is True), "prims": _count(row.get("prims", 0), "row.prims", 1000000),
            "invalid": invalid}
    add("terminal", **term)
    goal = row.get("goal_ok")
    if isinstance(goal, bool):
        add("admission_verdict", goal_ok=goal, tag="privileged-terminal-goal-check")
    add("episode_end")
    return evs


def episode_id(events: list) -> str:
    """Opaque, content-addressed id: identical whitelisted streams (the same run copied into two checkpoint folders)
    collapse to one episode instead of being counted twice."""
    body = [{k: v for k, v in e.items() if k != "episode"} for e in events]
    return "rp:" + sha256_hex(canon(body))[:12]


def convert(snapshots: list, items: dict | None = None, lang="en", per_source_cap=None) -> tuple:
    """-> (episodes, manifest). `episodes` = [{'episode', 'events'}] in source order with content-duplicates removed.
    `manifest` is evaluator-side only (source ids, hashes, counts, rejections); it is never fed to memory."""
    archived = load_archived()
    items = items if items is not None else load_items()
    episodes, seen = [], {}
    man = {"replay_version": REPLAY_VERSION, "lang": lang, "sources": [], "rejected": [], "totals": {}}
    tot = dict(rows=0, converted=0, skipped_inapplicable=0, rejected=0, duplicate_content=0, bad_lines=0, invalid=0)
    for sn in snapshots:
        rows, bad = read_rows(sn["copy"])
        tot["bad_lines"] += bad
        conv = 0
        for r in rows[:per_source_cap] if per_source_cap else rows:
            tot["rows"] += 1
            if r.get("id") in archived and r.get("id") not in items:
                tot["skipped_archived"] = tot.get("skipped_archived", 0) + 1     # A26: an archived (Knife) item
                continue
            try:
                evs = row_events(r, items.get(r.get("id")), episode="rp:pending", lang=lang)
            except ReplayError as e:
                tot["rejected"] += 1
                man["rejected"].append({"source": sn["tag"], "row_id": str(r.get("id"))[:60], "why": str(e)})
                continue
            if evs is None:
                tot["skipped_inapplicable"] += 1
                continue
            eid = episode_id(evs)
            evs = [dict(e, episode=eid) for e in evs]
            if eid in seen:
                tot["duplicate_content"] += 1
                continue
            seen[eid] = (sn["tag"], r.get("id"))
            if any(e["kind"] == "terminal" and e.get("invalid") for e in evs):
                tot["invalid"] += 1
            episodes.append({"episode": eid, "events": evs})
            conv += 1
        man["sources"].append({k: sn[k] for k in ("tag", "source", "sha256_before", "sha256_after", "sha256_copy",
                                                  "bytes", "source_changed")} | {"rows_converted": conv, "bad_lines": bad})
    tot["converted"] = len(episodes)
    man["totals"] = tot
    man["episode_sources"] = {eid: {"source": s, "row_id": rid} for eid, (s, rid) in seen.items()}   # evaluator-only
    return episodes, man


# --------------------------------------------------------------------------------------------------- replay
def replay_into(ltm, episodes: list, extensions=(), order=None) -> dict:
    """Feed each episode through WorkingMemory (event by event) and `ltm.end_episode`. Returns a tally."""
    from ltm_wm import WorkingMemory
    tally: dict = {}
    seq = list(episodes)
    if order is not None:
        seq = [seq[i] for i in order]
    results = []
    for ep in seq:
        wm = WorkingMemory(ep["episode"])
        for e in ep["events"]:
            wm.ingest(e)
        res = ltm.end_episode(wm, extensions=list(extensions))
        tally[res["status"]] = tally.get(res["status"], 0) + 1
        results.append((ep["episode"], res["status"]))
    return {"tally": tally, "results": results}


# --------------------------------------------------------------------------------------------------- leak scan
def _walk_strings(x, out):
    if isinstance(x, str):
        out.append(x)
    elif isinstance(x, dict):
        for k, v in x.items():
            _walk_strings(k, out)
            _walk_strings(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _walk_strings(v, out)


def forbidden_strings(raw_rows: list) -> list:
    """Strings from the raw rows that must never reach memory: the row id, hash, timestamps, the audit episode, and every
    free-text value (detail, why, error texts, event lines, search lists joined) of at least 8 characters that is not a
    bare type/cause token."""
    out = set()
    for r in raw_rows:
        for key in ("id", "hash", "ts", "detail"):
            v = r.get(key)
            if isinstance(v, str) and len(v) >= 4:
                out.add(v)
        a = r.get("audit") if isinstance(r.get("audit"), dict) else {}
        for v in a.get("errors_logged", []) or []:
            s: list = []
            _walk_strings(v, s)
            out.update(x for x in s if len(x) >= 8)
        if isinstance(a.get("episode"), str):
            out.add(a["episode"])
        for st in r.get("steps", []) or []:
            if not isinstance(st, dict):
                continue
            for key in EXCLUDED_STEP_FIELDS:
                s = []
                _walk_strings(st.get(key), s)
                out.update(x for x in s if len(x) >= 8 and (" " in x or "(" in x or "[" in x))
    return sorted(x for x in out if x)


def _all_values(db_path):
    con = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            cur = con.execute(f'SELECT * FROM "{name}"')
            cols = [c[0] for c in cur.description]
            for row in cur:
                for c, v in zip(cols, row):
                    if v is not None:
                        yield name, c, v if isinstance(v, str) else str(v)
    finally:
        con.close()


def scan_store(db_path, forbidden=(), extra_tokens=()) -> list:
    """Every stored value (all tables, all columns) against the forbidden strings and token patterns. [] = clean."""
    try:
        from ltm_store import SIM_ID_RE
    except ImportError:                  # pragma: no cover
        from sim.ltm_store import SIM_ID_RE
    bad = []
    low = [t.lower() for t in extra_tokens]
    for tbl, col, val in _all_values(db_path):
        if SIM_ID_RE.search(val):
            bad.append((tbl, col, "sim-id-like"))
        lv = val.lower()
        for f in forbidden:
            if f in val:
                bad.append((tbl, col, f"forbidden:{f[:24]}"))
        for t in low:
            if t in lv:
                bad.append((tbl, col, f"token:{t}"))
    return bad


def scan_events(episodes: list, forbidden=(), extra_tokens=()) -> list:
    bad = []
    low = [t.lower() for t in extra_tokens]
    for ep in episodes:
        s = canon(ep["events"])
        for f in forbidden:
            if f in s:
                bad.append((ep["episode"], f"forbidden:{f[:24]}"))
        ls = s.lower()
        for t in low:
            if t in ls:
                bad.append((ep["episode"], f"token:{t}"))
    return bad


def shuffled_order(n, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return idx


def main(argv=None) -> int:         # pragma: no cover - CLI convenience: python sim/step5_replay.py
    tmp = tempfile.mkdtemp(prefix="s5replay_")
    snaps = snapshot_sources(Path(tmp) / "src")
    eps, man = convert(snaps)
    print(json.dumps({"episodes": len(eps), "totals": man["totals"], "tmp": tmp}, indent=1))
    return 0


if __name__ == "__main__":          # pragma: no cover
    sys.exit(main())
