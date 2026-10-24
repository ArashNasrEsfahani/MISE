"""The GENERAL instruction set -- bilingual commands that are not tied to one kitchen.

`sim/dataset/commands.jsonl` (Phase 9) grounds every record in ONE scene: its gold was written against that
kitchen's inventory, and it cannot be run anywhere else. This module holds the complementary set: instructions
phrased generally enough to be given in any iTHOR kitchen ("open the fridge", "wash the mug and put it on the
counter", "I want a chilled apple"). A record here carries NO `scene`; `eval.py` already routes scene-less records
into whichever scenes a run asks for.

One ITEM is one instruction in BOTH languages, because the pair shares a gold and is edited as a unit:

    {"id": "g-l2-fridge-apple", "level": "L2",
     "en": "Put the apple in the fridge and close the door.", "fa": "سیب را داخل یخچال بگذار و درش را ببند.",
     "setup": [],                                   # steps run BEFORE the agent plans (see below)
     "plan":  [{"action": "pick_and_place", "object": "Apple", "to": "Fridge", "arg": null},
               {"action": "close", "object": "Fridge", "to": null, "arg": null}],
     "goal":  [{"type": "Apple", "key": "parentReceptacles", "contains": "Fridge"},
               {"type": "Fridge", "key": "isOpen", "value": false}]}

Three design points, each of which exists because the naive version silently produces wrong scores:

* **`plan` is the reference plan, and the gold labels are DERIVED from it** through `eval.labels` -- the same
  function that labels a generated plan. A hand-typed label that drifts from the scorer's vocabulary
  ("fill Mug" vs "fill_with_liquid Mug") scores every correct answer as wrong, and nothing reports it.
  It also makes the item executable, so `validate_general.py` can run the reference plan in the simulator and
  prove the goal is reachable at all.
* **`setup`** puts the world in the state the instruction presupposes. Every kitchen starts with every door
  closed, every appliance off and nothing dirty (`scene_inventory.json`), so "close the fridge" with goal
  `Fridge.isOpen = false` is true before the agent moves -- an agent that does nothing would score SR = 1.
  `setup: [open Fridge]` makes the instruction mean something.
* **An instruction need not apply everywhere.** A kitchen without a Drawer cannot score "open the drawer"; the
  harness reports such a record as skipped ("object does not exist in this scene") instead of failing the
  model for the dataset's choice. `applicability()` predicts this offline from the inventory.

Goals and plans are edited as one-line text (`Apple in Fridge`, `fill Mug = water`), not raw JSON -- the
console's editor is a text box, and a DSL that reads like the instruction is what makes 250 records reviewable
by a person. `parse_*` / `format_*` round-trip exactly.

Pure stdlib. Safe to import anywhere (eval is imported lazily, only to derive labels).
"""
from __future__ import annotations

import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "dataset", "general.json")
AUDIT_PATH = os.path.join(HERE, "dataset", "general_audit.jsonl")
INVENTORY_PATH = os.path.join(HERE, "dataset", "scene_inventory.json")

LEVELS = ("L1", "L2", "L3")
QUANTIFIERS = ("any", "all")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,79}$")
_ARABIC_RE = re.compile(r"[؀-ۿ]")

# The pseudo-type for assertions about the robot itself (`Agent.isStanding = false`); never looked up in a scene.
AGENT = "Agent"


# ---------------------------------------------------------------------------------------------------------------
# STEP DSL        action [Object] [-> To] [= arg]
# ---------------------------------------------------------------------------------------------------------------
def parse_step(line) -> dict:
    """`pick_and_place Apple -> Fridge` / `fill Mug = water` / `crouch` -> a planner-shaped step dict."""
    s = str(line or "").strip()
    if not s:
        raise ValueError("empty step")
    arg = None
    if "=" in s:
        s, a = s.split("=", 1)
        arg = _scalar(a.strip())
    to = None
    if "->" in s:
        s, t = s.split("->", 1)
        to = t.strip() or None
    parts = s.split()
    if not parts or len(parts) > 2:
        raise ValueError(f"step must read `action [Object] [-> To] [= arg]`, got {line!r}")
    return {"action": parts[0].strip().lower(), "object": (parts[1] if len(parts) == 2 else None),
            "to": to, "arg": arg}


def format_step(step) -> str:
    s = str(step.get("action") or "")
    if step.get("object"):
        s += f" {step['object']}"
    if step.get("to"):
        s += f" -> {step['to']}"
    if step.get("arg") is not None:
        s += f" = {_show(step['arg'])}"
    return s


# ---------------------------------------------------------------------------------------------------------------
# GOAL DSL        [any|all] Type.key = value   |   Type.key <= n  (>= n)   |   Type in Receptacle   |   holding Type
# ---------------------------------------------------------------------------------------------------------------
def parse_goal(line) -> dict:
    s = str(line or "").strip()
    if not s:
        raise ValueError("empty goal")
    q = None
    head = s.split(None, 1)
    if len(head) == 2 and head[0].lower() in QUANTIFIERS:
        q, s = head[0].lower(), head[1].strip()
    low = s.lower()
    if low.startswith("holding "):
        a = {"type": s[8:].strip(), "key": "isPickedUp", "value": True}
    elif " in " in s and "=" not in s:
        t, r = s.split(" in ", 1)
        a = {"type": t.strip(), "key": "parentReceptacles", "contains": r.strip()}
    elif ("<=" in s or ">=" in s) and "." in re.split(r"[<>]=", s, maxsplit=1)[0]:
        op = "<=" if "<=" in s else ">="
        lhs, v = s.split(op, 1)
        t, k = lhs.strip().split(".", 1)
        try:
            a = {"type": t.strip(), "key": k.strip(), ("max" if op == "<=" else "min"): float(v)}
        except ValueError:
            raise ValueError(f"`{op}` needs a number, got {v.strip()!r}") from None
    elif "=" in s and "." in s.split("=", 1)[0]:
        lhs, v = s.split("=", 1)
        t, k = lhs.strip().split(".", 1)
        a = {"type": t.strip(), "key": k.strip(), "value": _scalar(v.strip())}
    else:
        raise ValueError(f"goal must read `Type.key = value`, `Type in Receptacle` or `holding Type`, got {line!r}")
    if not a["type"] or not a["key"]:
        raise ValueError(f"goal is missing its type or key: {line!r}")
    if q:
        a["q"] = q
    return a


def format_goal(a) -> str:
    q = f"{a['q']} " if a.get("q") else ""
    if "contains" in a and a.get("key") == "parentReceptacles":
        return f"{q}{a['type']} in {a['contains']}"
    if a.get("key") == "isPickedUp" and a.get("value") is True and not q:
        return f"holding {a['type']}"
    for bound, op in (("max", "<="), ("min", ">=")):
        if bound in a:
            return f"{q}{a['type']}.{a['key']} {op} {a[bound]:g}"
    return f"{q}{a['type']}.{a['key']} = {_show(a.get('value'))}"


def _scalar(v):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", ""):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v.strip("\"'")


def _show(v) -> str:
    if v is True:
        return "true"
    if v is False:
        return "false"
    return "null" if v is None else str(v)


def parse_lines(text, fn) -> list:
    """Parse a text box: one entry per line, blank lines and `#` comments ignored. Errors name the line."""
    out = []
    for i, ln in enumerate(str(text or "").splitlines(), 1):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        try:
            out.append(fn(ln))
        except ValueError as exc:
            raise ValueError(f"line {i}: {exc}") from exc
    return out


# ---------------------------------------------------------------------------------------------------------------
# GOLD + RECORDS
# ---------------------------------------------------------------------------------------------------------------
def plan_labels(plan) -> list:
    """The scored label sequence of a reference plan -- by construction the scorer's own vocabulary."""
    import eval as _eval                     # lazy: eval imports this module to load the set
    return _eval.labels(plan or [])


def gold_of(item) -> dict:
    """Level-appropriate gold: the goal (SR, every level) + sequence (L2, PC) or action set (L3, TDS)."""
    gold = {"goal": list(item.get("goal") or [])}
    if item.get("level") == "L2":
        gold["sequence"] = plan_labels(item.get("plan"))
    elif item.get("level") == "L3":
        gold["actions"] = sorted(set(plan_labels(item.get("plan"))))
    return gold


def to_records(doc, lang="all") -> list:
    """Expand items into scene-less CommandRecord dicts (`<id>-en`, `<id>-fa`), EN/FA interleaved per item so a
    `--limit` cut keeps the languages balanced."""
    rows = []
    for it in (doc.get("items") or []):
        gold = gold_of(it)
        for lg in ("en", "fa"):
            if lang not in ("all", lg) or not it.get(lg):
                continue
            rows.append({"id": f"{it['id']}-{lg}", "level": it["level"], "lang": lg, "command": it[lg],
                         "gold": gold, "setup": list(it.get("setup") or []), "general": True})
    return rows


def is_general_doc(obj) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("items"), list)


# ---------------------------------------------------------------------------------------------------------------
# INVENTORY-BASED CHECKS (offline predictions of what the harness decides live)
# ---------------------------------------------------------------------------------------------------------------
def load_inventory(path=None):
    try:
        with open(path or INVENTORY_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def referenced_types(item) -> list:
    """Every scene object type an item depends on: goal types, plan + setup objects and destinations."""
    types = []
    for a in (item.get("goal") or []):
        types.append(a.get("type"))
        if a.get("key") == "parentReceptacles":
            types.append(a.get("contains"))
    for st in list(item.get("setup") or []) + list(item.get("plan") or []):
        types += [st.get("object"), st.get("to")]
    return sorted({t for t in types if t and t != AGENT})


def applicability(item, inventory) -> dict:
    """In how many scanned kitchens does every referenced type exist? `missing` names the absences.

    A PREDICTION, from the inventory scan; the harness makes the real decision against the live scene, with
    the same rule (`eval.missing_types`: exact type, so a Potato never stands in for a missing Pot)."""
    if not inventory:
        return {"n": None, "total": None, "missing": {}}
    import eval as _eval
    scenes = inventory.get("scenes") or []
    per = inventory.get("per_scene") or {}
    need = referenced_types(item)
    missing = {}
    for s in scenes:
        gone = _eval.missing_types(need, list(per.get(s) or {}))
        if gone:
            missing[s] = gone
    return {"n": len(scenes) - len(missing), "total": len(scenes), "missing": missing}


def presatisfied_scenes(item, inventory) -> list:
    """Scenes where the goal is ALREADY TRUE at reset, so doing nothing would score SR = 1.

    Only decidable offline for an item with no `setup` and only for state the scan records; `visible` and
    `Agent.*` assertions are left to the live gate. For a multi-instance type an un-quantified or `any`
    assertion is met by any one instance, `all` needs every instance."""
    if not inventory or item.get("setup"):
        return []
    out = []
    for s in (inventory.get("scenes") or []):
        inv = (inventory.get("per_scene") or {}).get(s) or {}
        ok = bool(item.get("goal"))
        for a in (item.get("goal") or []):
            t = inv.get(a.get("type"))
            if not t:
                ok = False
                break
            if "contains" in a:
                hit = any(str(a["contains"]).lower() in p.lower() for p in t.get("parents") or [])
            else:
                vals = (t.get("initial") or {}).get(a.get("key"))
                if vals is None:
                    ok = False
                    break
                hit = (vals == [a.get("value")]) if a.get("q") == "all" else (a.get("value") in vals)
            if not hit:
                ok = False
                break
        if ok:
            out.append(s)
    return out


# ---------------------------------------------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------------------------------------------
def known_actions() -> set:
    """Every action name a plan step may use: the skills manifest, the planner's surface verbs, nav helpers."""
    import eval as _eval
    import skills
    return ({s["name"] for s in skills.skills_manifest()} | set(_eval._ACTION_ALIASES)
            | set(_eval._NAV_HELPERS) | {"search_for"})


def validate_item(item, actions=None, inventory=None):
    """-> (errors, warnings). Errors block a save; warnings are shown but allowed."""
    errs, warns = [], []
    iid = str(item.get("id") or "")
    if not _ID_RE.match(iid):
        errs.append("id must be 2-80 chars of a-z 0-9 . _ - (start with a letter or digit)")
    if item.get("level") not in LEVELS:
        errs.append(f"level must be one of {', '.join(LEVELS)}")
    for lg, name in (("en", "English"), ("fa", "Persian")):
        if not str(item.get(lg) or "").strip():
            errs.append(f"{name} text is required -- the set is bilingual")
    if item.get("fa") and not _ARABIC_RE.search(item["fa"]):
        warns.append("the Persian text contains no Persian letters")
    if item.get("en") and _ARABIC_RE.search(item["en"]):
        warns.append("the English text contains Persian letters")

    goal = item.get("goal") or []
    if not goal:
        errs.append("at least one goal line is required -- Success Rate is scored against it")
    for a in goal:
        if not isinstance(a, dict) or not a.get("type") or not a.get("key"):
            errs.append(f"malformed goal assertion: {a!r}")
        elif not any(k in a for k in ("value", "contains", "max", "min")):
            errs.append(f"goal on {a['type']}.{a['key']} has no value, `contains` or bound to check")
        elif a.get("q") and a["q"] not in QUANTIFIERS:
            errs.append(f"goal quantifier must be any/all, got {a['q']!r}")

    actions = actions if actions is not None else known_actions()
    for field in ("setup", "plan"):
        for st in (item.get(field) or []):
            if not isinstance(st, dict) or not st.get("action"):
                errs.append(f"malformed {field} step: {st!r}")
            elif st["action"] not in actions:
                errs.append(f"{field} step uses unknown action {st['action']!r}")

    if not errs:
        n = len(plan_labels(item.get("plan")))
        lvl = item["level"]
        if lvl in ("L2", "L3") and n == 0:
            errs.append(f"{lvl} needs a reference plan -- its "
                        f"{'sequence (PC)' if lvl == 'L2' else 'action set (TDS)'} is derived from it")
        elif lvl == "L1" and n > 1:
            warns.append(f"L1 is one command -> one function, but the plan has {n} scored steps (L2?)")
        elif lvl == "L2" and n < 2:
            warns.append("L2 is an ordered chain, but the plan has a single scored step")

    if inventory and not errs:
        cov = inventory.get("coverage") or {}
        import eval as _eval
        for t in _eval.missing_types(referenced_types(item), list(cov)):
            warns.append(f"{t} exists in none of the {len(inventory.get('scenes') or [])} scanned kitchens")
        pre = presatisfied_scenes(item, inventory)
        if pre:
            warns.append(f"goal is already true at the start in {len(pre)} kitchen(s) "
                         f"({', '.join(s.replace('FloorPlan', 'FP') for s in pre[:6])}"
                         f"{'...' if len(pre) > 6 else ''}) -- skipped there; add a `setup` if that is not intended")
    return errs, warns


# ---------------------------------------------------------------------------------------------------------------
# STORE (load / save / upsert / delete, every write audited)
# ---------------------------------------------------------------------------------------------------------------
def load(path=None) -> dict:
    p = path or PATH
    if not os.path.exists(p):
        return {"version": 1, "items": [], "deleted": []}
    with open(p, encoding="utf-8") as fh:
        doc = json.load(fh)
    doc.setdefault("items", [])
    doc.setdefault("deleted", [])
    return doc


def save(doc, path=None):
    """Atomic replace: a crash mid-write must not leave the evaluation dataset half-written."""
    p = path or PATH
    doc["updated_ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def _audit(op, iid, before, after, actor, path=None):
    """One JSON line per edit. The dataset is an evaluation INSTRUMENT: a score is only interpretable if one can
    say which instructions existed, in which wording, when it was produced."""
    try:
        with open(path or AUDIT_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "op": op, "id": iid, "actor": actor,
                                 "before": before, "after": after}, ensure_ascii=False) + "\n")
    except Exception:
        pass


def normalize(item) -> dict:
    """The stored shape of an item: known keys only, fixed order, steps fully keyed."""
    step = lambda s: {"action": s.get("action"), "object": s.get("object"), "to": s.get("to"), "arg": s.get("arg")}
    out = {"id": str(item.get("id") or "").strip(), "level": item.get("level"),
           "en": str(item.get("en") or "").strip(), "fa": str(item.get("fa") or "").strip(),
           "setup": [step(s) for s in (item.get("setup") or [])],
           "plan": [step(s) for s in (item.get("plan") or [])],
           "goal": list(item.get("goal") or [])}
    if item.get("note"):
        out["note"] = str(item["note"]).strip()
    if item.get("reviewed"):
        out["reviewed"] = True
    return out


def upsert(item, original_id=None, actor="user", path=None, audit_path=None):
    """Add a new item or replace an existing one. -> {"ok", "errors", "warnings", "item", "op"}.

    `original_id` is the id the editor opened (so a rename replaces instead of duplicating); omit it to add."""
    item = normalize(item)
    errs, warns = validate_item(item, inventory=load_inventory())
    doc = load(path)
    ids = {it["id"] for it in doc["items"]}
    if original_id and original_id not in ids:
        errs.append(f"{original_id!r} no longer exists -- reload the list")
    if item["id"] in ids and item["id"] != original_id:
        errs.append(f"id {item['id']!r} is already used by another instruction")
    if errs:
        return {"ok": False, "errors": errs, "warnings": warns}
    before = None
    if original_id:
        idx = next(i for i, it in enumerate(doc["items"]) if it["id"] == original_id)
        before = doc["items"][idx]
        if before.get("reviewed"):            # the editor does not post the mark; an edit must not un-check it
            item["reviewed"] = True
        doc["items"][idx] = item
    else:
        doc["items"].append(item)
    doc["deleted"] = [d for d in doc["deleted"] if d != item["id"]]
    save(doc, path)
    op = "update" if original_id else "add"
    _audit(op, item["id"], before, item, actor, audit_path)
    return {"ok": True, "errors": [], "warnings": warns, "item": item, "op": op}


def delete(iid, actor="user", path=None, audit_path=None):
    """Remove an item. The id is remembered in `deleted` so re-running the seed generator does not resurrect it."""
    doc = load(path)
    before = next((it for it in doc["items"] if it["id"] == iid), None)
    if before is None:
        return {"ok": False, "errors": [f"{iid!r} not found"]}
    doc["items"] = [it for it in doc["items"] if it["id"] != iid]
    if iid not in doc["deleted"]:
        doc["deleted"].append(iid)
    save(doc, path)
    _audit("delete", iid, before, None, actor, audit_path)
    return {"ok": True, "errors": []}


def set_reviewed(iid, flag, actor="user", path=None, audit_path=None):
    """Mark an instruction as checked by a person (or un-mark it). Reviewing 288 bilingual records is a job done
    over several sittings; the mark is what lets the next sitting start where the last one stopped."""
    doc = load(path)
    it = next((x for x in doc["items"] if x["id"] == iid), None)
    if it is None:
        return {"ok": False, "errors": [f"{iid!r} not found"]}
    was = bool(it.get("reviewed"))
    if flag:
        it["reviewed"] = True
    else:
        it.pop("reviewed", None)
    if was != bool(flag):
        save(doc, path)
        _audit("review" if flag else "unreview", iid, None, None, actor, audit_path)
    return {"ok": True, "errors": [], "reviewed": bool(flag)}


def export_csv(doc) -> str:
    """The set as a flat bilingual table (for the thesis appendix, or for reading away from the console)."""
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["id", "level", "english", "persian", "setup", "reference_plan", "goal", "checked"])
    for it in doc.get("items") or []:
        w.writerow([it["id"], it["level"], it["en"], it["fa"],
                    " ; ".join(format_step(s) for s in it.get("setup") or []),
                    " ; ".join(format_step(s) for s in it.get("plan") or []),
                    " ; ".join(format_goal(a) for a in it.get("goal") or []),
                    "yes" if it.get("reviewed") else ""])
    return buf.getvalue()


def view(item, inventory=None) -> dict:
    """An item as the console shows it: the stored fields + the DSL text the editor round-trips + derived gold."""
    out = dict(item)
    out["setup_text"] = "\n".join(format_step(s) for s in item.get("setup") or [])
    out["plan_text"] = "\n".join(format_step(s) for s in item.get("plan") or [])
    out["goal_text"] = "\n".join(format_goal(a) for a in item.get("goal") or [])
    try:
        out["labels"] = plan_labels(item.get("plan"))
    except Exception:
        out["labels"] = []
    if inventory:
        ap = applicability(item, inventory)
        out["applies"] = {"n": ap["n"], "total": ap["total"],
                          "missing": sorted({t for ts in ap["missing"].values() for t in ts}),
                          "missing_in": sorted(ap["missing"])}       # the scenes that lack a needed type
        pre = presatisfied_scenes(item, inventory)
        out["presatisfied"] = len(pre)
        out["presatisfied_in"] = sorted(pre)
    return out


def from_editor(payload) -> dict:
    """The editor posts DSL text; turn it back into an item (raises ValueError naming the offending line)."""
    field = lambda name, fn: parse_lines(payload.get(name), fn)
    try:
        setup = field("setup_text", parse_step)
    except ValueError as exc:
        raise ValueError(f"setup, {exc}") from exc
    try:
        plan = field("plan_text", parse_step)
    except ValueError as exc:
        raise ValueError(f"plan, {exc}") from exc
    try:
        goal = field("goal_text", parse_goal)
    except ValueError as exc:
        raise ValueError(f"goal, {exc}") from exc
    return {"id": payload.get("id"), "level": payload.get("level"), "en": payload.get("en"),
            "fa": payload.get("fa"), "note": payload.get("note"), "setup": setup, "plan": plan, "goal": goal}
