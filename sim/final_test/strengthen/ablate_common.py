"""Task A (Scene Memory ablations): shared helpers. Exploratory, post-hoc; never writes into a sealed namespace.

Records: docs/final-test/strengthen/A/   Work files (token cache, response cache, ledger, locks): sim/final_test/strengthen/work/A/
Reads (never writes): docs/final-test/matrix/test-v1/{capture,plan,exec,outcome,events,score}.jsonl, units.json, the frozen stores.

A recorded full-memory block (capture.jsonl blocks["S*"]) is one line per kept record, in ltm_package.SECTIONS order, each section with its
fixed header (the example section has none), followed by the need bundle (homeowner inventory + uses table + need instruction) when the
NEED rule fired. `parse_block` / `render_block` invert that exactly (checked on all 223 non-empty development-kitchen blocks).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True        # never drop a __pycache__ beside the pinned modules this package imports
os.environ["TRACE_ENABLED"] = "0"     # the thesis trace store (sim/trace.db) is never opened by this task's processes
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for p in (os.path.join(ROOT, "sim", "final_test", "c3"), os.path.join(ROOT, "sim"), os.path.join(ROOT, "sim", "final_test", "matrix")):
    if p not in sys.path:
        sys.path.insert(0, p)

NS = "test-v1"                                    # the development kitchen (the default); the held-out kitchens are supported too
KITCHENS = {"test-v1": "FloorPlan1", "test-k2-v1": "FloorPlan15", "test-k3-v1": "FloorPlan17"}
DOCS = os.path.join(ROOT, "docs", "final-test", "strengthen", "A")
WORK = os.path.join(HERE, "work", "A")


def main_dir(ns: str = NS) -> str:
    return os.path.join(ROOT, "docs", "final-test", "matrix", ns)


MAIN = main_dir(NS)
LEDGER_MD = os.path.join(ROOT, "docs", "final-test", "strengthen", "LEDGER.md")
RESUME_MD = os.path.join(DOCS, "RESUME.md")
SEED = 20261010
OPUS_CAP_USD = 25.0

SECTIONS = ("briefing", "profile", "procedure", "example", "template", "uncertainty")
HEADERS = {"briefing": "This kitchen:", "profile": "Seen in practice:", "procedure": "Steps that work here:",
           "template": "Plan outline:", "uncertainty": "Not known:"}
BUNDLE_START = "Kitchen inventory: the homeowner reports"
REQUEST_SUFFIX = "/request-types"
BUNDLE_SUFFIXES = ("/homeowner-inventory", "/uses-table", "/need-instruction")

#: the arms (A1 design). kind: "edit" = recorded block with records removed; "kitchen" = another kitchen's frozen store, same query;
#: "irrelevant" = records retrieved for other instructions, length-matched; "replan" = the full block planned again (noise control).
ARMS = {
    "rep":           {"kind": "replan", "label": "full Scene Memory, planned again (noise control)"},
    "no_request":    {"kind": "edit", "label": "without the English restatement (procedure/*/request-types)"},
    "no_guide":      {"kind": "edit", "label": "without the kitchen guide ('This kitchen:' briefing records)"},
    "no_procedures": {"kind": "edit", "label": "without the procedures (procedure records except the restatement)"},
    "no_inventory":  {"kind": "edit", "label": "without the inventory (the NEED bundle appended outside the budget)"},
    "kitchen_fp1":   {"kind": "kitchen", "label": "FloorPlan1's frozen Scene Memory, same query", "ns": "test-v1"},
    "kitchen_fp15":  {"kind": "kitchen", "label": "FloorPlan15's frozen Scene Memory, same query", "ns": "test-k2-v1"},
    "kitchen_fp17":  {"kind": "kitchen", "label": "FloorPlan17's frozen Scene Memory, same query", "ns": "test-k3-v1"},
    "irrelevant":    {"kind": "irrelevant", "label": "records retrieved for other instructions, same token length"},
    "no_examples_templates": {"kind": "edit", "label": "without the solved examples and templates (supplementary)"},
}


def arms_for(ns: str = NS) -> list:
    """Every arm of a kitchen: the other two kitchens' memories are its 'another kitchen' arms (never its own)."""
    return [a for a, v in ARMS.items() if not (v["kind"] == "kitchen" and v["ns"] == ns)]


def opus_arms_for(ns: str = NS) -> list:
    return ["no_procedures"] + [a for a in arms_for(ns) if ARMS[a]["kind"] == "kitchen"]


# ------------------------------------------------------------------------------------------------ io
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _fsync_replace(tmp: str, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    for k in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if k == 7:
                raise
            time.sleep(0.25)


def atomic_write_text(path: str, text: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    _fsync_replace(tmp, path)


def atomic_write_json(path: str, obj):
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n")


def atomic_write_jsonl(path: str, rows):
    atomic_write_text(path, "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def read_jsonl(path: str) -> list:
    """Tolerates a torn LAST line (a stop mid-append); anything else unparsable is an error."""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        lines = f.read().split(b"\n")
    out = []
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln.decode("utf-8")))
        except ValueError:
            if i == len(lines) - 1:
                continue
            raise RuntimeError(f"{path}: unparsable line {i + 1}")
    return out


def append_jsonl(path: str, rec: dict):
    """One fsync'd line. A torn tail left by an earlier stop is cut first, so the journal never holds a broken line in the middle."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path):
        with open(path, "rb+") as f:
            data = f.read()
            if data and not data.endswith(b"\n"):
                cut = data.rfind(b"\n") + 1
                try:
                    json.loads(data[cut:].decode("utf-8"))
                    f.seek(0, 2)
                    f.write(b"\n")
                except ValueError:
                    f.seek(cut)
                    f.truncate()
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def ledger_line(text: str):
    """One line in docs/final-test/strengthen/LEDGER.md (append; shared with tasks R / L / D)."""
    with open(LEDGER_MD, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M')} {text}\n")
        f.flush()
        os.fsync(f.fileno())


def opus_spent() -> float:
    """Cumulative committed Opus spend of task A across ALL runs: settled actual + still-open reservations in the one private ledger
    (sim/final_test/strengthen/work/A/opus_ledger.db; the same total CostLedger.spent() enforces the 25 USD ceiling on). Read only."""
    import sqlite3
    p = os.path.join(WORK, "opus_ledger.db")
    if not os.path.exists(p):
        return 0.0
    c = sqlite3.connect("file:" + p.replace(os.sep, "/") + "?mode=ro", uri=True)
    try:
        v = c.execute("SELECT COALESCE(SUM(CASE WHEN state='open' THEN reserved WHEN state='settled' THEN actual ELSE 0 END),0) "
                      "FROM ledger").fetchone()[0]
    finally:
        c.close()
    return round(float(v), 6)


def update_resume(subphase: str, done: list, opus_usd: float | None, resume_cmd: str, notes: list = ()):
    opus_usd = opus_spent()                    # always the ledger's cumulative total (never a caller's per-run counter)
    lines = ["# Task A (Scene Memory ablations): resume point", "",
             f"Updated {time.strftime('%Y-%m-%d %H:%M:%S')}.", "",
             f"- Current subphase: {subphase}",
             f"- Opus spend so far: {opus_usd:.4f} USD (cap {OPUS_CAP_USD:.0f} USD; ledger sim/final_test/strengthen/work/A/opus_ledger.db)",
             "- Done:"] + [f"  - {d}" for d in done] + ["", "Resume from the last checkpoint (repo root, Windows):", "", "```", resume_cmd, "```", ""]
    if notes:
        lines += ["Notes:"] + [f"- {n}" for n in notes] + [""]
    atomic_write_text(RESUME_MD, "\n".join(lines))


# ------------------------------------------------------------------------------------------------ inputs
def load_capture(ns: str = NS) -> dict:
    """pres -> capture row (canonical world, capture run 1, latest row wins, status ok), as plan.py reads it; every presentation must
    have a scored G+M r0 plan row in the main journal (the paired set)."""
    out = {}
    for r in read_jsonl(os.path.join(main_dir(ns), "capture.jsonl")):
        if r.get("namespace") != ns:
            raise RuntimeError(f"foreign namespace row in the {ns} capture")
        if int(r.get("capture_run", 1)) == 1 and r.get("world") == "canon":
            out[r["pres"]] = r
    want = {p for (a, p) in main_plans(ns) if a == "G+M" and p.startswith("canon:")}
    if set(out) != want or not out:
        raise RuntimeError(f"{ns}: {len(out)} canonical captures vs {len(want)} G+M r0 plan rows")
    return out


def load_types(ns: str = NS):
    """(types, source): plan.py's frozen list for Kitchen 1, the kitchen's own list for a K23 namespace (k23_run.k23_load_types)."""
    import plan as PL
    if ns == "test-v1":
        return PL.load_types(None)
    import k23_run as R
    return R.k23_load_types(ns)()


def family_map() -> dict:
    import common as C
    m = C.load_manifest(None, verify=False)
    return {i: f["family"] for f in (m.get("families") or []) for i in f.get("items") or ()}


_PLANS = {}


def main_plans(ns: str = NS) -> dict:
    """(arm, pres) -> the r = 0 plan row of the main journal (latest row per key, block main)."""
    if ns in _PLANS:
        return _PLANS[ns]
    lat = {}
    for r in read_jsonl(os.path.join(main_dir(ns), "plan.jsonl")):
        lat[r["key"]] = r
    _PLANS[ns] = {(r["arm"], r["pres"]): r for r in lat.values() if r.get("r") == 0 and r.get("block") == "main"}
    return _PLANS[ns]


# ------------------------------------------------------------------------------------------------ block grammar
def comp_of(rid: str) -> str:
    return rid.split("/", 1)[0]


def is_bundle(rid: str) -> bool:
    return rid.endswith(BUNDLE_SUFFIXES)


def parse_block(text: str, ids: list) -> dict:
    """-> {"records": [{"id", "comp", "line"}...] (section order), "bundle": [lines]}. Raises if the text does not follow the grammar."""
    if not text:
        if ids:
            raise ValueError("empty text with record ids")
        return {"records": [], "bundle": []}
    lines = text.split("\n")
    sec_ids = [i for i in ids if not is_bundle(i)]
    has_bundle = any(is_bundle(i) for i in ids)
    k = next((j for j, l in enumerate(lines) if l.startswith(BUNDLE_START)), None)
    if has_bundle != (k is not None):
        raise ValueError("need bundle ids and text disagree")
    body, bundle = (lines[:k], lines[k:]) if k is not None else (lines, [])
    recs, pos = [], 0
    for s in SECTIONS:
        si = [i for i in sec_ids if comp_of(i) == s]
        if not si:
            continue
        if s in HEADERS:
            if pos >= len(body) or body[pos] != HEADERS[s]:
                raise ValueError(f"missing header of section {s}")
            pos += 1
        for rid in si:
            if pos >= len(body):
                raise ValueError("fewer lines than records")
            recs.append({"id": rid, "comp": s, "line": body[pos]})
            pos += 1
    if pos != len(body):
        raise ValueError("more lines than records")
    if [r["id"] for r in recs] != sec_ids:
        raise ValueError("record order differs from the section order")
    out = {"records": recs, "bundle": bundle}
    if render_block(out) != text:
        raise ValueError("render(parse(text)) != text")
    return out


def render_block(parsed: dict) -> str:
    """The builder's own layout (ltm_package.PackageBuilder._section_text / build): a section with no record has no header."""
    parts = []
    for s in SECTIONS:
        rs = [r for r in parsed["records"] if r["comp"] == s]
        if not rs:
            continue
        body = "\n".join(r["line"] for r in rs)
        parts.append((HEADERS[s] + "\n" + body) if s in HEADERS else body)
    body = "\n".join(parts)
    bundle = "\n".join(parsed.get("bundle") or [])
    if bundle:
        body = (body + "\n" + bundle) if body else bundle
    return body


def edit_block(parsed: dict, arm: str) -> tuple:
    """-> (new parsed block, removed ids) for an 'edit' arm."""
    recs, bundle, removed = list(parsed["records"]), list(parsed["bundle"]), []

    def drop(pred):
        nonlocal recs
        keep = []
        for r in recs:
            (removed.append(r["id"]) if pred(r) else keep.append(r))
        recs = keep

    if arm == "no_request":
        drop(lambda r: r["id"].endswith(REQUEST_SUFFIX))
    elif arm == "no_guide":
        drop(lambda r: r["comp"] == "briefing")
    elif arm == "no_procedures":
        drop(lambda r: r["comp"] == "procedure" and not r["id"].endswith(REQUEST_SUFFIX))
    elif arm == "no_inventory":
        if bundle:
            removed += ["need-bundle"]
        bundle = []
    elif arm == "no_examples_templates":
        drop(lambda r: r["comp"] in ("example", "template"))
    else:
        raise ValueError(f"{arm} is not an edit arm")
    return {"records": recs, "bundle": bundle}, removed


def bundle_ids(ids: list) -> list:
    return [i for i in ids if is_bundle(i)]
