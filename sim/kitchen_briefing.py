"""B2: the welcome briefing generator (stage H-B2-briefing). The ONLY declared privileged channel of the assistance package.

What it is: a fixed, kitchen-independent script that writes a homeowner-style tour of a kitchen's FIXED layout (fixture types, counts, coarse
quantised positions, adjacency, openable / toggleable) plus the USUAL storage of categories as frozen house conventions (never read from where
movable objects sit at reset). It never contains a movable object's position or any object state. Arash reads and approves every briefing
before the freeze; his edits are versioned.

    python sim/kitchen_briefing.py --extract --scene FloorPlan1     # WSL + simulator, ONE reset, no action; under flock + hang guard
    python sim/kitchen_briefing.py --render  --scene FloorPlan1     # offline: .en.md, .fa.md, .records.jsonl
    python sim/kitchen_briefing.py --show    --scene FloorPlan1     # offline: print the records

Honesty: simulator metadata is read in exactly one function, ``extract_layout`` (the field whitelist below). Nothing else here sees it.
Generality: no kitchen constant, id, coordinate or scene-name branch; the kitchen name is data. Python 3.10 syntax (WSL imports it).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "docs", "final-test", "kitchen_memory", "briefings")
CONV_PATH = os.path.join(HERE, "step5", "briefing_conventions.json")
TPL_PATH = os.path.join(HERE, "step5", "briefing_templates.json")
DET_PATH = os.path.join(ROOT, "docs", "final-test", "matrix", "detector_types.json")

QUANT = 0.5            # metres: positions are quantised to this grid
ADJ_RADIUS = 1.25      # metres: "next to"
ADJ_MAX = 3            # at most this many neighbours per anchor type
#: the ONLY simulator fields read (section 5.1). parentReceptacles (ids) is NOT read (Astra H-B2: omit it); parent_type stays in the schema, always null.
FIELDS_WHITELIST = ["objectType", "position.x", "position.z", "openable", "toggleable", "receptacle", "pickupable"]


#: K-B2b (A26 / D2): the one global type filter. Conventions never name an excluded type in a rendered record.
try:
    import type_scope as TS
except ImportError:                                                            # pragma: no cover
    from sim import type_scope as TS

#: K-B2b (Arash decision (7), 2026-10-13): the approved natural-English tour of each kitchen, pinned by sha in the overrides file.
NATURAL_SUFFIX = ".en.natural.md"
NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
             "eighteen": 18, "nineteen": 19, "twenty": 20}
#: plain English words a homeowner uses for a fixture without naming it in brackets (generic, kitchen-independent)
NATURAL_ALIASES = {"stove": "StoveBurner", "stove knobs": "StoveKnob", "light": "LightSwitch"}


class BriefingNotApproved(Exception):
    pass


def _canon(o) -> str:
    return json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def self_sha(doc: dict) -> str:
    return sha256_text(_canon({k: v for k, v in doc.items() if k != "sha256"}))


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_conventions(path: str = CONV_PATH) -> dict:
    return _load(path)


def load_templates(path: str = TPL_PATH) -> dict:
    return _load(path)


def detector_vocab(path: str = DET_PATH):
    try:
        return set(_load(path)["types"])
    except (OSError, KeyError, ValueError):
        return None


# --------------------------------------------------------------------------- extraction (the privileged read)
def _q(x: float) -> float:
    return round(round(float(x) / QUANT) * QUANT, 3)


def _type_of_id(oid: str) -> str:
    return str(oid).split("|", 1)[0]


def fixture_types_allowed(conv: dict, vocab) -> list:
    return [t for t in conv["fixture_types"] if vocab is None or t in vocab]


def extract_layout(objects: list, scene: str, conv: dict = None, vocab=None, extracted: str = None) -> dict:
    """THE ONLY READ OF SIMULATOR METADATA. ``objects`` = metadata['objects'] of ONE reset of the canonical scene.

    Keeps an object only if it is NOT pickupable AND its type is in the fixed fixture list (intersected with the detector vocabulary). The
    ``moveable`` flag is deliberately NOT the filter: in AI2-THOR it is true for the Microwave, CoffeeMachine, Toaster and GarbageCan (they can
    be pushed), so a ``moveable == false`` filter would drop real fixtures; the type allow-list keeps stools, tables, bags and every
    pickupable object out. Only the whitelisted fields are read. Object ids are dropped. isOpen / isToggled / controlledObjects are never read."""
    conv = conv or load_conventions()
    allowed = set(fixture_types_allowed(conv, vocab))
    groups = {}
    for o in objects:
        t = o.get("objectType")
        if t not in allowed or o.get("pickupable", False):
            continue
        pos = o.get("position") or {}
        g = groups.setdefault(t, {"cells": [], "n": 0, "openable": False, "toggleable": False, "receptacle": False})
        g["n"] += 1
        g["cells"].append((_q(pos.get("x", 0.0)), _q(pos.get("z", 0.0))))
        g["openable"] = g["openable"] or bool(o.get("openable"))
        g["toggleable"] = g["toggleable"] or bool(o.get("toggleable"))
        g["receptacle"] = g["receptacle"] or bool(o.get("receptacle"))
    fixtures = []
    for t in sorted(groups):
        g = groups[t]
        fixtures.append({"type": t, "n": g["n"], "cells": [list(c) for c in sorted(set(g["cells"]))],
                         "parent_type": None,
                         "openable": g["openable"], "toggleable": g["toggleable"], "receptacle": g["receptacle"]})
    doc = {"schema": "pkg.layout/1", "scene": scene, "extracted": extracted or time.strftime("%Y-%m-%dT%H:%M:%S"),
           "fields_whitelist": list(FIELDS_WHITELIST), "quant_m": QUANT, "fixtures": fixtures}
    doc["sha256"] = self_sha(doc)
    return doc


# --------------------------------------------------------------------------- layout -> facts (pure)
def _by_type(layout: dict) -> dict:
    return {f["type"]: f for f in layout["fixtures"]}


def _extent(layout: dict):
    pts = [tuple(c) for f in layout["fixtures"] for c in f["cells"]]
    if not pts:
        return (0.0, 0.0, 0.0, 0.0)
    xs, zs = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), max(xs), min(zs), max(zs))


def _dir_of(x: float, z: float, ext) -> str:
    x0, x1, z0, z1 = ext
    cx, cz = (x0 + x1) / 2.0, (z0 + z1) / 2.0
    hx, hz = max((x1 - x0) / 2.0, 1e-9), max((z1 - z0) / 2.0, 1e-9)
    ew = "" if abs(x - cx) < 0.25 * hx else ("east" if x > cx else "west")
    ns = "" if abs(z - cz) < 0.25 * hz else ("north" if z > cz else "south")
    if not ew and not ns:
        return "middle"
    return (ns + ew) if (ns and ew) else (ns or ew)


def place_of(f: dict, ext) -> tuple:
    """-> ('one'|'many'|'spread', dirkey). Types whose cells fall in 3+ distinct directions are 'spread'."""
    cells = f["cells"]
    if len({_dir_of(c[0], c[1], ext) for c in cells}) >= 3:
        return ("spread", None)
    mx = sum(c[0] for c in cells) / len(cells)
    mz = sum(c[1] for c in cells) / len(cells)
    return ("one" if f["n"] == 1 else "many", _dir_of(mx, mz, ext))


def _dist(f, g) -> float:
    return min(math.hypot(a[0] - b[0], a[1] - b[1]) for a in f["cells"] for b in g["cells"])


def adjacency(layout: dict, conv: dict) -> dict:
    """anchor type -> up to ADJ_MAX nearest other types within ADJ_RADIUS (quantised cells, deterministic; parent/child pairs skipped)."""
    fx = _by_type(layout)
    excl = set(conv.get("adjacency_exclude", []))
    skip = {frozenset(p) for p in conv.get("adjacency_skip_pairs", [])}
    out = {}
    for a in conv["anchor_types"]:
        if a not in fx:
            continue
        cand = []
        for t, g in fx.items():
            if t == a or t in excl or frozenset((a, t)) in skip or fx[a].get("parent_type") == t or g.get("parent_type") == a:
                continue
            d = _dist(fx[a], g)
            if d <= ADJ_RADIUS:
                cand.append((round(d, 3), t))
        cand.sort()
        if cand:
            out[a] = [t for _, t in cand[:ADJ_MAX]]
    return out


def instantiated_conventions(layout: dict, conv: dict, vocab=None) -> list:
    """Per category: the receptacle types that EXIST in this kitchen and the category's types that are in the detector vocabulary."""
    have = set(_by_type(layout))
    out = []
    for cid in sorted(conv["categories"]):
        c = conv["categories"][cid]
        recs = [r for r in c["receptacles"] if r in have]
        types = [t for t in TS.filter_types(c["types"]) if vocab is None or t in vocab]       # K-B2b: D2 filter (no Knife)
        if recs and types:
            out.append({"id": cid, "name": c["name"], "types": types, "receptacles": recs})
    return out


# --------------------------------------------------------------------------- rendering
def _join(items: list, lang: str, tpl: dict) -> str:
    j = tpl["join"][lang]
    if len(items) <= 1:
        return "".join(items)
    return j["sep"].join(items[:-1]) + j["last"] + items[-1]


def _fill(s: str, **kw) -> str:
    for k, v in kw.items():
        s = s.replace("{" + k + "}", str(v))
    return s


def _rec(scene, layout_sha, kind, local, prio, keys, text, rel_types, authored_by):
    return {"schema": "pkg.rec/1", "id": "briefing/%s/%s" % (scene, local), "component": "briefing", "kitchen": scene,
            "keys": keys, "lex": {"en": " ".join(sorted(set(rel_types))).lower(), "fa": ""}, "text": text,
            "provenance": {"source": "briefing:" + layout_sha[:16], "evidence": [], "authored_by": authored_by, "support": []},
            "created": 0, "recency": 0, "confidence": 1.0, "privileged": True, "prio": prio, "kind": kind}


def build_records(layout: dict, conv: dict = None, tpl: dict = None, vocab=None) -> list:
    """The fact lines as pkg.rec/1 records (EN + FA). Deterministic. Core ids ('core:<Type>', prio 0) are the parser-failure fallback (D5)."""
    conv = conv or load_conventions()
    tpl = tpl or load_templates()
    scene, lsha = layout["scene"], layout["sha256"]
    who = "script:kitchen_briefing@" + sha256_text(_canon(conv) + _canon(tpl))[:16]
    fx = _by_type(layout)
    ext = _extent(layout)
    core = set(conv["core_types"])
    adj = adjacency(layout, conv)
    K = tpl["kinds"]
    recs = []

    def both(key, **kw):
        return {l: _fill(K[key][l], **kw) for l in ("en", "fa")}

    for t in sorted(fx):
        f = fx[t]
        text = both("exist_one", T=t) if f["n"] == 1 else both("exist_many", T=t, n=f["n"])
        is_core = t in core
        recs.append(_rec(scene, lsha, "core" if is_core else "exist", ("core:" if is_core else "exist:") + t, 0 if is_core else 1,
                         {"types": [t], "ops": [], "rels": []}, text, [t], who))
    for t in sorted(fx):
        mode, d = place_of(fx[t], ext)
        if mode == "spread":
            text = both("place_spread", T=t)
        else:
            text = {l: _fill(K["place_one" if mode == "one" else "place_many"][l], T=t, dir=tpl["dirs"][d][l]) for l in ("en", "fa")}
        recs.append(_rec(scene, lsha, "place", "place:" + t, 2, {"types": [t], "ops": [], "rels": []}, text, [t], who))
    for t in sorted(adj):
        nb = adj[t]
        text = {l: _fill(K["adj"][l], T=t, list=_join(nb, l, tpl)) for l in ("en", "fa")}
        recs.append(_rec(scene, lsha, "adj", "adj:" + t, 3, {"types": [t] + nb, "ops": [], "rels": ["near:" + x for x in nb]}, text, [t] + nb, who))
    for t in sorted(fx):
        f = fx[t]
        if not (f["openable"] or f["toggleable"]):
            continue
        key = "open_both" if f["openable"] and f["toggleable"] else ("open_open" if f["openable"] else "open_toggle")
        ops = (["open", "close"] if f["openable"] else []) + (["toggle_on", "toggle_off"] if f["toggleable"] else [])
        rels = (["opens:" + t] if f["openable"] else []) + (["toggles:" + t] if f["toggleable"] else [])
        recs.append(_rec(scene, lsha, "open", "open:" + t, 4, {"types": [t], "ops": ops, "rels": rels}, both(key, T=t), [t], who))
    for c in instantiated_conventions(layout, conv, vocab):
        text = {}
        for l in ("en", "fa"):
            where = tpl["or"][l].join(_fill(tpl["prep"].get(r, tpl["prep"]["default"])[l], T=r) for r in c["receptacles"])
            text[l] = _fill(K["conv"][l], cat=c["name"][l], types=tpl["join"][l]["sep"].join(c["types"]), where=where)
        recs.append(_rec(scene, lsha, "conv", "conv:" + c["id"], 5,
                         {"types": c["types"] + c["receptacles"], "ops": [], "rels": ["stores:" + r for r in c["receptacles"]]},
                         text, c["types"] + c["receptacles"], who))
    recs.sort(key=lambda r: (r["prio"], r["id"]))
    return recs


def render_markdown(layout: dict, records: list, lang: str, tpl: dict = None, status: str = "DRAFT", approved_at: str = None) -> str:
    tpl = tpl or load_templates()
    if approved_at:
        hdr = {"en": "# Welcome briefing: %s (approved by the homeowner on %s)" % (layout["scene"], approved_at),
               "fa": "# معرفی آشپزخانه: %s (تأییدشده در %s)" % (layout["scene"], approved_at)}[lang]
    else:
        hdr = {"en": "# Welcome briefing: %s (%s, not yet approved by the homeowner)" % (layout["scene"], status),
               "fa": "# معرفی آشپزخانه: %s (%s، هنوز تأیید نشده)" % (layout["scene"], status)}[lang]
    lines = [hdr, "", tpl["intro"][lang], "",
             "<!-- layout sha256 %s; generated by sim/kitchen_briefing.py; edits go in %s.overrides.json -->" % (layout["sha256"][:16], layout["scene"]), ""]
    for kind in ("core", "exist", "place", "adj", "open", "conv"):
        rs = [r for r in records if r["kind"] == kind]
        if rs:
            lines += ["## " + tpl["sections"][kind][lang], ""] + ["- " + r["text"][lang] for r in rs] + [""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- files, overrides, loading
def _paths(scene: str, out_dir: str = None) -> dict:
    d = out_dir or OUT_DIR
    return {"layout": os.path.join(d, scene + ".layout.json"), "en": os.path.join(d, scene + ".en.md"), "fa": os.path.join(d, scene + ".fa.md"),
            "records": os.path.join(d, scene + ".records.jsonl"), "overrides": os.path.join(d, scene + ".overrides.json"),
            "natural": os.path.join(d, scene + NATURAL_SUFFIX), "history": os.path.join(d, "_history")}


def _write(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def render_scene(scene: str, out_dir: str = None, layout: dict = None) -> dict:
    p = _paths(scene, out_dir)
    layout = layout or _load(p["layout"])
    if layout.get("sha256") != self_sha(layout):
        raise ValueError("layout sha256 mismatch for " + scene)
    recs = build_records(layout, vocab=detector_vocab())
    ov = _load(p["overrides"]) if os.path.exists(p["overrides"]) else {}
    when = ov.get("approved_at") if ov.get("approved_by") else None
    _write(p["en"], render_markdown(layout, recs, "en", approved_at=when))
    _write(p["fa"], render_markdown(layout, recs, "fa", approved_at=when))
    _write(p["records"], "".join(_canon(r) + "\n" for r in recs))
    if not os.path.exists(p["overrides"]):
        _write(p["overrides"], json.dumps({"version": 0, "approved_by": None, "approved_at": None, "edits": {}}, indent=1) + "\n")
    return {"scene": scene, "records": len(recs), "layout_sha16": layout["sha256"][:16]}


def save_overrides(scene: str, doc: dict, out_dir: str = None) -> str:
    """Write a new overrides version; every previous version is kept in briefings/_history/."""
    p = _paths(scene, out_dir)
    if os.path.exists(p["overrides"]):
        old = _load(p["overrides"])
        _write(os.path.join(p["history"], "%s.overrides.v%s.json" % (scene, old.get("version", 0))),
               json.dumps(old, indent=1, ensure_ascii=False) + "\n")
    _write(p["overrides"], json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return p["overrides"]


def load_records(scene: str, allow_unapproved: bool = False, out_dir: str = None) -> list:
    """pkg.rec/1 records with Arash's overrides applied. Refuses unless approved (dev and practice pass allow_unapproved=True; the caller logs it)."""
    p = _paths(scene, out_dir)
    ov = _load(p["overrides"]) if os.path.exists(p["overrides"]) else {"version": 0, "approved_by": None, "edits": {}}
    if not ov.get("approved_by") and not allow_unapproved:
        raise BriefingNotApproved("briefing for %s is not approved (approved_by is empty); allow_unapproved=True is for dev/practice only" % scene)
    if not allow_unapproved and (ov.get("natural") or os.path.exists(p["natural"])):
        load_natural(scene, out_dir=out_dir)          # K-B2b: the approved path also verifies the pinned natural tour (raises if not)
    with open(p["records"], encoding="utf-8") as f:
        recs = [json.loads(ln) for ln in f if ln.strip()]
    out = []
    for r in recs:
        e = ov.get("edits", {}).get(r["id"], "absent")
        if e is None:
            continue
        if e != "absent":
            r = dict(r, text={"en": e.get("en", r["text"]["en"]), "fa": e.get("fa", r["text"]["fa"])},
                     provenance=dict(r["provenance"], authored_by="arash-override:v%s" % ov.get("version")))
        out.append(r)
    if ov.get("additions"):
        out = sorted(out + addition_records(scene, ov, recs), key=lambda r: (r["prio"], r["id"]))
    return out


def addition_records(scene: str, ov: dict, recs: list) -> list:
    """K-B2b: homeowner-authored fact lines that the generator cannot derive from the whitelisted layout fields (the knob line of
    decision (7): which knob drives which burner is controlledObjects, never read). Each addition must name only fixture types that
    the generated records already list, must not collide with a generated id, and is authored by the override version."""
    have = {(r.get("keys") or {}).get("types", [""])[0] for r in recs if r.get("kind") in ("core", "exist")}     # fixtures only
    ids = {r["id"] for r in recs}
    src = (recs[0]["provenance"]["source"] if recs else "briefing")
    out = []
    for a in ov.get("additions") or []:
        rid = "briefing/%s/%s" % (scene, a["local"])
        types = list(a["types"])
        targets = {str(x).split(":", 1)[1] for x in a.get("rels") or [] if ":" in str(x)}
        if (rid in ids or not types or set(types) - have or targets - set(types) or len(targets) != len(a.get("rels") or [])
                or TS.mentions_excluded(a["text"]["en"]) or set(TS.EXCLUDED_TYPES) & set(types)):
            raise ValueError("briefing addition %s is invalid (id clash, non-fixture / excluded type or bad relation)" % rid)
        ids.add(rid)                                                                                  # no duplicate additions
        out.append({"schema": "pkg.rec/1", "id": rid, "component": "briefing", "kitchen": scene,
                    "keys": {"types": types, "ops": list(a.get("ops") or []), "rels": list(a.get("rels") or [])},
                    "lex": {"en": " ".join(sorted(set(types))).lower(), "fa": ""}, "text": {"en": a["text"]["en"], "fa": a["text"]["fa"]},
                    "provenance": {"source": src, "evidence": [], "authored_by": "arash-override:v%s" % ov.get("version"), "support": []},
                    "created": 0, "recency": 0, "confidence": 1.0, "privileged": True, "prio": int(a.get("prio", 4)), "kind": a["kind"]})
    return out


def load_natural(scene: str, allow_unapproved: bool = False, out_dir: str = None) -> str:
    """K-B2b: the approved natural-English tour, verified against the sha pinned in the overrides file (and that file's layout)."""
    p = _paths(scene, out_dir)
    ov = _load(p["overrides"]) if os.path.exists(p["overrides"]) else {}
    nat = ov.get("natural") or {}
    data = open(p["natural"], "rb").read()
    if not allow_unapproved:
        if not ov.get("approved_by"):
            raise BriefingNotApproved("natural briefing for %s is not approved" % scene)
        if nat.get("sha256") != hashlib.sha256(data).hexdigest():
            raise BriefingNotApproved("natural briefing for %s differs from the approved version" % scene)
        lay = _load(p["layout"])
        if nat.get("layout_sha16") != lay["sha256"][:16]:
            raise BriefingNotApproved("natural briefing for %s was approved for another layout" % scene)
    return data.decode("utf-8")


# --------------------------------------------------------------------------- natural text vs the generated facts (K-B2b gate)
_CAMEL = re.compile(r"(?<![A-Za-z])([A-Z][a-z]+(?:[A-Z][a-z]+)+|[A-Z][a-z]+)(?![A-Za-z])")


def _section(text: str, title: str) -> list:
    m = re.search(r"\*\*%s\.?\*\*(.*?)(?=\n\*\*|\Z)" % re.escape(title), text, re.S)
    return [ln[2:].strip() for ln in (m.group(1).splitlines() if m else []) if ln.startswith("- ")]


def _nouns(text: str, fixture_types: set) -> dict:
    """noun -> Type, learned from the text's own 'nine cabinets (Cabinet)' pairs, plus singular / plural forms and NATURAL_ALIASES."""
    out = dict(NATURAL_ALIASES)
    for m in re.finditer(r"((?:[A-Za-z-]+ ){1,3})\(([A-Z][A-Za-z]+)", text):
        t = m.group(2)
        if t not in fixture_types:
            continue
        k = len(re.findall(r"[A-Z][a-z]+", t))
        words = [w for w in m.group(1).split()[-k:] if w.lower() not in NUM_WORDS and w.lower() != "the"]
        if not words:
            continue
        n = " ".join(words).lower()
        forms = {n, n[:-1] if n.endswith("s") else n + "s"}
        if n.endswith("ves"):
            forms.add(n[:-3] + "f")
        if n.endswith("es"):
            forms.add(n[:-2])
        for f in forms:
            out.setdefault(f, t)
    return out


def _types_in(s: str, nouns: dict) -> list:
    s, found = " " + s.lower() + " ", []
    for n in sorted(nouns, key=len, reverse=True):
        rx = re.compile(r"(?<![a-z])%s(?![a-z])" % re.escape(n))
        if rx.search(s):
            found.append(nouns[n])
            s = rx.sub(" ", s)
    return found


def natural_facts_check(text: str, layout: dict, records: list, conv: dict = None) -> list:
    """Findings where the natural text disagrees with the generated facts ([] = consistent). Checks: every bracketed type is a fixture
    of this layout and every fixture is named; counts; no excluded type; compass sectors; 'next to' pairs (either direction);
    openable / switchable sets. Pure text + layout; no simulator."""
    conv = conv or load_conventions()
    fx = _by_type(layout)
    ext = _extent(layout)
    adj = adjacency(layout, conv)
    F = []
    named = set()
    body = re.sub(r"<!--.*?-->", "", "\n".join(ln for ln in text.split("\n") if not ln.startswith("#")), flags=re.S)
    for m in re.finditer(r"\(([^)]*)\)", body):
        toks = _CAMEL.findall(m.group(1))                      # every capitalised word in brackets is a type name in these texts
        for i, t in enumerate(toks):
            named.add(t)
            if t not in fx:
                F.append({"kind": "unknown_type", "detail": t})
                continue
            if i == 0:
                before = body[:m.start()].split()[-4:]
                nums = [NUM_WORDS[w.lower()] for w in before if w.lower() in NUM_WORDS]
                if nums and nums[-1] != fx[t]["n"]:
                    F.append({"kind": "count", "detail": "%s: text %d, layout %d" % (t, nums[-1], fx[t]["n"])})
    for t in sorted(set(fx) - named):
        F.append({"kind": "fixture_not_named", "detail": t})
    if TS.mentions_excluded(text) or re.search(r"(?<!butter )\bknif|\bknives\b", text, re.I):
        F.append({"kind": "excluded_type", "detail": "Knife"})
    nouns = _nouns(text, set(fx))
    for ln in _section(text, "Where things are"):
        if ":" not in ln:
            continue
        lab, rest = ln.split(":", 1)
        lab = lab.strip().lower().replace("-", "")
        for t in _types_in(rest, nouns):
            mode, d = place_of(fx[t], ext)
            if mode == "spread":
                F.append({"kind": "sector", "detail": "%s listed under %s but is spread around the kitchen" % (t, lab)})
            elif d != lab:
                F.append({"kind": "sector", "detail": "%s: text %s, layout %s" % (t, lab, d)})
    for ln in _section(text, "What sits next to what"):
        if "next to" in ln:
            left, right = ln.split("next to", 1)
            subj, nb = _types_in(left, nouns), _types_in(right, nouns)
        elif "side by side" in ln:
            left, right = ln.split("side by side", 1)
            subj = _types_in(left, nouns)
            nb = _types_in(right, nouns) + subj
        else:
            continue
        for s in subj:
            for n in nb:
                if n != s and n not in adj.get(s, []) and s not in adj.get(n, []):
                    F.append({"kind": "adjacency", "detail": "%s next to %s is not in the generated adjacency" % (s, n)})
    lines = _section(text, "What opens and what switches on")
    opens = set(_types_in(" ".join(l for l in lines if l.lower().startswith("you can open")), nouns))
    togg = set(_types_in(" ".join(l for l in lines if l.lower().startswith("you can switch")), nouns))
    want_o = {t for t, f in fx.items() if f["openable"]}
    want_t = {t for t, f in fx.items() if f["toggleable"]}
    if opens != want_o:
        F.append({"kind": "openable", "detail": "text %s, layout %s" % (sorted(opens), sorted(want_o))})
    if togg != want_t:
        F.append({"kind": "switchable", "detail": "text %s, layout %s" % (sorted(togg), sorted(want_t))})
    F += _habit_findings(_section(text, "How we usually keep things"), records or [], nouns, set(fx))
    return F


def _words(t: str) -> set:
    out = set()
    for w in (x.lower() for x in re.findall(r"[A-Z][a-z]+", t)):
        out |= {w, w + "s", w + "es"}
    return out


def _habit_findings(lines: list, records: list, nouns: dict, fixtures: set) -> list:
    """Each natural habit line against the generated conv record it describes: the same movable types (by their name words) and the
    same receptacles (Sink and SinkBasin count as one place). Every conv record needs a line and every line a record."""
    same = lambda s: {"SinkBasin" if t == "Sink" else t for t in s}                                   # noqa: E731
    conv = []
    for r in records:
        if r.get("kind") != "conv":
            continue
        rec_at = {x.split(":", 1)[1] for x in r["keys"].get("rels", []) if x.startswith("stores:")}
        conv.append((r["id"], [t for t in r["keys"]["types"] if t not in rec_at], same(rec_at)))
    F, used = [], set()
    for ln in lines:
        low = set(re.findall(r"[a-z]+", ln.lower()))
        best = max(conv, key=lambda c: sum(1 for t in c[1] if _words(t) & low), default=None)
        if not best or not any(_words(t) & low for t in best[1]):
            F.append({"kind": "habit_extra", "detail": ln})
            continue
        used.add(best[0])
        miss = [t for t in best[1] if not (_words(t) & low)]
        if miss:
            F.append({"kind": "habit_member", "detail": "%s: %s not in '%s'" % (best[0], miss, ln)})
        at = same(t for t in _types_in(ln, nouns) if t in fixtures)
        if at != best[2]:
            F.append({"kind": "habit_place", "detail": "%s: text %s, record %s" % (best[0], sorted(at), sorted(best[2]))})
    for cid, _m, _a in conv:
        if cid not in used:
            F.append({"kind": "habit_missing", "detail": cid})
    return F


def type_tokens(text: str, vocab) -> set:
    return {w for w in re.findall(r"[A-Za-z]+", text) if w in vocab}


def fa_missing_tokens(records: list, vocab) -> int:
    """Number of records whose text.fa lacks a detector-type token that text.en has (must be 0)."""
    return sum(1 for r in records if type_tokens(r["text"]["en"], vocab) - type_tokens(r["text"]["fa"], vocab))


# --------------------------------------------------------------------------- natural fixture lines (I-F3, precision package)
_COUNT = re.compile(r"^There (?:are (\d+)|is one)(?![A-Za-z])")


def fixture_facts(records: list) -> dict:
    """{Type: {n, ops, near, ids}} read from the briefing records themselves (id kind + keys + count in the text). Nothing is
    re-derived from the layout, so it cannot disagree with the approved records. `ids` lists the records the facts came from."""
    out = {}
    for r in records or []:
        rid = str(r.get("id", ""))
        kind = rid.split("/")[-1].split(":", 1)[0]
        types = list((r.get("keys") or {}).get("types") or [])
        if not types:
            continue
        t = types[0]
        f = out.setdefault(t, {"n": None, "ops": [], "near": [], "ids": []})
        if kind in ("core", "exist"):
            m = _COUNT.match(str((r.get("text") or {}).get("en", "")))
            if m:
                f["n"] = int(m.group(1)) if m.group(1) else 1
                f["ids"].append(r["id"])
        elif kind == "open":
            f["ops"] = sorted(set(f["ops"]) | set((r.get("keys") or {}).get("ops") or []))
            f["ids"].append(r["id"])
        elif kind == "adj":
            f["near"] = list(types[1:])
            f["ids"].append(r["id"])
        elif kind == "ctrl":                                   # K-B2b: homeowner addition 'controls:<Type>'; kept apart from 'ids'
            f.setdefault("controls", [])                       # so the default kitchen line (and its evidence) is unchanged
            f.setdefault("ctrl_ids", [])
            f["controls"] = sorted(set(f["controls"]) | {x.split(":", 1)[1] for x in (r.get("keys") or {}).get("rels", [])
                                                         if x.startswith("controls:")})
            f["ctrl_ids"].append(r["id"])
    return out


def _can(ops: list) -> str:
    o, t = ("open" in ops or "close" in ops), ("toggle_on" in ops or "toggle_off" in ops)
    if o and t:
        return "it opens and closes, and switches on and off"
    if o:
        return "it opens and closes"
    if t:
        return "it switches on and off"
    return ""


def natural_fixture_line(facts: dict, types: list, with_near: bool = False, prefix: str = "In this kitchen: ",
                         with_controls: bool = False) -> str:
    """One plain-English sentence for the fixtures a request needs, e.g.
    'In this kitchen: one Microwave (it opens and closes, and switches on and off); 3 CounterTop.'
    No compass directions (they never change a plan); 'next to' only when with_near is set. '' when no fixture is known.
    K-B2b: with_controls (default False, so the default line is byte-identical) adds 'it controls the StoveBurner' from a
    homeowner 'ctrl' addition."""
    parts = []
    for t in types:
        f = facts.get(t)
        if not f or f.get("n") is None:
            continue
        n = "one" if f["n"] == 1 else str(f["n"])
        ctl = ("it controls the " + ", ".join(f["controls"])) if with_controls and f.get("controls") else ""
        extra = [x for x in (_can(f.get("ops") or []), ("next to " + ", ".join(f["near"][:3])) if with_near and f.get("near") else "", ctl)
                 if x]
        parts.append("%s %s%s" % (n, t, " (" + "; ".join(extra) + ")" if extra else ""))
    if not parts:
        return ""
    body = "; ".join(parts) + "."
    return prefix + body if prefix else body[0].upper() + body[1:]


# --------------------------------------------------------------------------- CLI
def _extract_cli(scene: str, out_dir: str = None) -> dict:
    """WSL only. ONE reset of the canonical scene, no action. Run under flock + hang guard by the wrapper script."""
    from ai2thor.controller import Controller
    c = Controller(scene=scene, width=300, height=300)
    try:
        layout = extract_layout(c.last_event.metadata["objects"], scene, vocab=detector_vocab())
    finally:
        c.stop()
    p = _paths(scene, out_dir)
    _write(p["layout"], json.dumps(layout, indent=1, ensure_ascii=False) + "\n")
    return {"scene": scene, "fixtures": len(layout["fixtures"]), "sha16": layout["sha256"][:16], "file": p["layout"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="B2 welcome briefing generator")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--extract", action="store_true")
    g.add_argument("--render", action="store_true")
    g.add_argument("--show", action="store_true")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    if a.extract:
        print(json.dumps(_extract_cli(a.scene, a.out_dir)))
    elif a.render:
        print(json.dumps(render_scene(a.scene, a.out_dir)))
    else:
        for r in load_records(a.scene, allow_unapproved=True, out_dir=a.out_dir):
            print(r["id"], "|", r["text"]["en"], "|", r["text"]["fa"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
