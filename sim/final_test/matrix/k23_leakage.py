"""K23: the practice-snapshot leakage gate and test sentences that are FRAGMENTS of the snapshot's own practice instructions -- ONLY under
a recorded decision. The registered gate (practice_k.leakage_gate, prereg 6.3 + A30) is not edited and still runs first.

The case (practice-k3-v1, 2026-10-20, the K = 10 snapshot): Gemma's free-text reply ("say") of practice attempt 8
(g-l2-slice-bread-tomato, EN: "Take the butter knife, slice the bread, then slice the tomato.") reads "I will slice the bread and then
the tomato." Normalised, it contains "slice the bread", the whole text of test item g-l2-slice-bread and of test-uncapped item
g-l1-slice-bread ("Slice the bread."). A30 exempts only the row's validated `instruction` field and its validated G1 line; every other
field is a substring hit, so the gate fails. A30 itself names this practice instruction as one that embeds those test sentences
("not derived from a test item; practice-own text"). Kitchen 1's attempt 8 said "Taking the butter knife and slicing the bread and
tomato." ("slicing" does not contain "slice the bread"), so Kitchen 1 passed by wording only; Kitchen 2 (same practice order) can meet
the same failure.

The rule implemented here (decision K23-PRACTICE-LEAK, choice "own-fragments"; narrower than A31's whole-segment rule):
  * the registered gate runs first, unchanged; if it passes, or if ANY hit is not an instruction-text hit (ids, dump unreadable, invalid
    exemption), its result stands;
  * otherwise "own fragments" = the test / test-uncapped / reserved instruction needles (normalised, >= MIN_NEEDLE characters) that are
    substrings of the normalised PINNED instruction of a snapshot row that passes A30 (a) (schedule position, item and language, no
    test / reserved id, instruction = pinned dataset text byte for byte);
  * in those A30-valid rows only, a string field other than `instruction` (any nesting) is blanked when its normalised text contains an
    own fragment, contains NO other needle once the own fragments are removed, contains no raw needle and no whole test / reserved id;
    each blanked field is audited {row, field path, practice item, own-fragment items, sha of the original value};
  * the registered gate then runs again on a temporary copy of the snapshot whose rows file carries those blanked fields; everything else
    (the store dump, every other file, every other row, ids on the unmasked bytes) is checked exactly as before. Pass only if that run
    passes; the manifest's leakage result is that run plus the audit and the decision id.
Activation: the decision record docs/final-test/k23/decisions/K23-PRACTICE-LEAK.json (this module's --write); installed in-process by
k23_practice (live) only.

    python sim/final_test/matrix/k23_leakage.py --check
    python sim/final_test/matrix/k23_leakage.py --write --by Arash --date 2026-10-20 --said "<his words>"
    python sim/final_test/matrix/k23_leakage.py --diagnose <snapshot dir> --namespace practice-k3-v1   (offline: both gate results)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                       # noqa: E402

DECISION_ID = "K23-PRACTICE-LEAK"
SCHEMA = "k23.decision/1"
CHOICE = "own-fragments"
RULE = ("practice-snapshot gate: a test / reserved instruction that is a fragment of a validated practice row's own pinned instruction is "
        "exempt inside the free-text fields of A30-valid rows when nothing else leaks there; the registered gate runs first and again")


#: the recovery disposition Arash approves together with the rule (Astra K23-PRACTICE-LEAK-r1 point 3); recorded in the decision record
RECOVERY = ("practice-k3-v1: journal rows 1-9 and the verified row-9 checkpoint are kept; attempt 10 is repeated on resume; its four earlier "
            "executions (the first run and three launcher retries, each from the row-9 checkpoint, each stopped by the gate at the K = 10 "
            "snapshot) are discarded and disclosed with their logs; the failed K = 10 snapshot and the logs are archived before the overwrite; "
            "no practice is re-run to change wording; a snapshot that already passed the registered gate is not regenerated.")


def decision_path(root: str | None = None) -> str:
    return os.path.join(C.paths(root)["docs"], "k23", "decisions", f"{DECISION_ID}.json")


def decision_problems(root: str | None = None) -> list:
    p = decision_path(root)
    if not os.path.exists(p):
        return [f"no decision record ({os.path.basename(p)})"]
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        return [f"decision record unreadable ({type(e).__name__})"]
    out = []
    if d.get("schema") != SCHEMA or d.get("id") != DECISION_ID or d.get("sha256") != C.self_sha(d):
        out.append("decision record schema / id / self-sha mismatch")
    if d.get("choice") != CHOICE or d.get("rule") != RULE or d.get("recovery") != RECOVERY:
        out.append(f"decision record choice is not {CHOICE!r} with this module's rule")
    if d.get("by") != "Arash" or not d.get("date") or not str(d.get("said") or "").strip():
        out.append("decision record is not Arash's attributed record (by / date / said)")
    return out


def decision_sha(root: str | None = None):
    p = decision_path(root)
    return C.sha256_file(p) if os.path.exists(p) else None


def _strings(obj, path=""):
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k in sorted(obj):
            yield from _strings(obj[k], f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _strings(v, f"{path}[{i}]")


def _set_path(obj, path, value):
    parts = re.findall(r"[^.\[\]]+|\[\d+\]", path)
    cur = obj
    for p in parts[:-1]:
        cur = cur[int(p[1:-1])] if p.startswith("[") else cur[p]
    last = parts[-1]
    if last.startswith("["):
        cur[int(last[1:-1])] = value
    else:
        cur[last] = value


def mask_rows(data: bytes, expected, refused: set, needles: dict) -> tuple:
    """-> (masked bytes, audit). Only A30 (a)-valid rows are touched; see the module docstring for the rule."""
    import practice_k as PK
    lines = [ln for ln in data.decode("utf-8").split("\n") if ln.strip()]
    rows, valid = [], []
    for n, ln in enumerate(lines):
        try:
            r = json.loads(ln)
        except ValueError:
            rows.append(None)
            continue
        rows.append(r)
        if isinstance(r, dict) and n < len(expected):
            item, lang, text = expected[n][:3]
            if r.get("item") == item and r.get("lang") == lang and item not in refused and r.get("instruction") == text:
                valid.append(n)
    texts = [(i, lg, nt, raw) for i, lg, nt, raw in needles["texts"] if len(nt) >= PK.MIN_NEEDLE]
    own_norm = [PK.norm_text(rows[n]["instruction"]) for n in valid]
    own = [(i, lg, nt, raw) for i, lg, nt, raw in texts if any(nt in on for on in own_norm)]
    id_pats = [re.compile(r"(?<![a-z0-9-])" + re.escape(i.lower()) + r"(?![a-z0-9-])") for i in needles["ids"]]
    audit = []
    for n in valid:
        r = rows[n]
        for path, v in list(_strings(r)):
            if path == "instruction":
                continue
            nv = PK.norm_text(v)
            hit_own = [x for x in own if x[2] in nv]
            if not hit_own:
                continue
            # Astra K23-PRACTICE-LEAK-r1: every NON-own needle is checked against the ORIGINAL normalised field (before any removal),
            # so a longer test sentence that contains an own fragment ("... slice the bread, then turn on the toaster") still leaks
            other = [x for x in texts if x not in own and x[2] in nv]
            raw_hit = [x for x in texts if x not in own and x[3] in v]
            ids = [p.pattern for p in id_pats if p.search(v.lower())]
            if other or raw_hit or ids:
                continue                                       # something else leaks in this field: leave it, the gate will hit
            _set_path(r, path, "")
            audit.append({"row": n + 1, "field": path, "item": r.get("item"), "lang": r.get("lang"),
                          "own_fragments": sorted({x[0] for x in hit_own}), "value_sha": C.sha256_text(v)})
    out = [json.dumps(r, ensure_ascii=False, sort_keys=True) if r is not None else ln for r, ln in zip(rows, lines)]
    return ("\n".join(out) + "\n").encode("utf-8"), audit


def gate(snap_dir: str, needles: dict, allow=None, *, orig=None) -> dict:
    """The decision-gated gate: the registered result unless every hit is an instruction-text hit that the own-fragment rule removes."""
    import practice_k as PK
    run = orig or PK.leakage_gate
    first = run(snap_dir, needles, allow=allow)
    if first.get("ok") or allow is None or any(h.get("kind") != "instruction" for h in first.get("hits") or []):
        return first
    rows_file = os.path.join(snap_dir, allow["rows_file"])
    if not os.path.exists(rows_file):
        return first
    masked, audit = mask_rows(open(rows_file, "rb").read(), allow["expected"], set(allow.get("refused") or ()), needles)
    if not audit:
        return first
    tmp = tempfile.mkdtemp(prefix="k23leak_")
    try:
        for name in os.listdir(snap_dir):
            p = os.path.join(snap_dir, name)
            if os.path.isfile(p):
                shutil.copy2(p, os.path.join(tmp, name))
        with open(os.path.join(tmp, allow["rows_file"]), "wb") as f:
            f.write(masked)
        second = run(tmp, needles, allow=allow)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    second = dict(second, registered_first=first, own_fragment_exempt=audit, rule=DECISION_ID, decision_sha256=decision_sha())
    return second


def snapshot_rule_problems(leak: dict, root: str | None = None) -> list:
    """For the agent freeze: a snapshot whose leakage result used the own-fragment rule must name THIS decision record (valid, same sha)."""
    if not isinstance(leak, dict) or leak.get("rule") != DECISION_ID:
        return []
    why = list(decision_problems(root))
    if leak.get("decision_sha256") != decision_sha(root):
        why.append("the snapshot's leakage result names another K23-PRACTICE-LEAK decision record than the one bound now")
    if not leak.get("ok") or not leak.get("own_fragment_exempt"):
        why.append("an own-fragment leakage result that did not pass or has no audit")
    return why


_STATE = {"installed": False}


def install(root: str | None = None) -> bool:
    """practice_k.leakage_gate -> the decision-gated gate, in THIS process only. False (nothing patched) without a valid decision."""
    if decision_problems(root):
        return False
    if _STATE["installed"]:
        return True
    import practice_k as PK
    orig = PK.leakage_gate

    def leakage_gate(snap_dir, needles, allow=None):
        return gate(snap_dir, needles, allow, orig=orig)
    leakage_gate.k23_leakage = True
    leakage_gate.k23_orig = orig
    PK.leakage_gate = leakage_gate
    _STATE["installed"] = True
    return True


def diagnose(snap_dir: str, ns: str) -> dict:
    """Offline: the registered result and the own-fragment result for an existing snapshot dir of `ns` (no write)."""
    import practice_k as PK
    import treatment as T
    m = json.load(open(os.path.join(snap_dir, PK.MANIFEST_NAME), encoding="utf-8"))
    k, o = int(m["k"]), int(m["order"])
    if ns not in PK.PRACTICE_NAMESPACES:                            # a K23 namespace is admitted only through k23_practice (as live)
        import k23_practice as KP
        KP.admit(ns)
    pr = PK.Practice(namespace=ns, orders=[o], ks=[k], executor=None, channels=(), kitchen=C.ns_scene(ns))
    sched = pr.orders_plan[o][:k]
    refused = set(pr.manifest["test"]) | set(pr.manifest.get("test_uncapped_not_scored") or []) | set(pr.reserved["ids"])
    allow = {"rows_file": T.ROWS_FILE, "refused": refused,
             "expected": [(p["item"], p["lang"], pr.items[p["item"]][p["lang"]], PK.request_line(pr.items[p["item"]], p["lang"])) for p in sched]}
    first = PK.leakage_gate(snap_dir, pr.needles, allow=allow)
    second = gate(snap_dir, pr.needles, allow, orig=PK.leakage_gate)
    return {"registered": {"ok": first["ok"], "hits": first["hits"]}, "own_fragments": {"ok": second["ok"], "hits": second["hits"],
            "exempt": second.get("own_fragment_exempt")}}


def main(argv=None) -> int:
    import time
    ap = argparse.ArgumentParser(description="K23 practice-snapshot leakage: own-fragment rule (decision record)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    g.add_argument("--diagnose", metavar="SNAPSHOT_DIR")
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--by"), ap.add_argument("--date"), ap.add_argument("--said")
    a = ap.parse_args(argv)
    if a.check:
        why = decision_problems()
        print(json.dumps({"decision": DECISION_ID, "active": not why, "problems": why}))
        return 0 if not why else 1
    if a.diagnose:
        if not a.namespace:
            print("REFUSED: --diagnose needs --namespace <practice namespace>", file=sys.stderr)
            return 2
        print(json.dumps(diagnose(a.diagnose, a.namespace), indent=1, ensure_ascii=False))
        return 0
    if a.by != "Arash" or not a.date or not (a.said or "").strip():
        print("REFUSED: --write needs --by Arash --date YYYY-MM-DD --said \"<his words>\"", file=sys.stderr)
        return 2
    p = decision_path()
    if os.path.exists(p):
        print(f"REFUSED: {p} exists (a decision is recorded once)", file=sys.stderr)
        return 2
    d = {"schema": SCHEMA, "id": DECISION_ID, "choice": CHOICE, "rule": RULE, "recovery": RECOVERY, "by": a.by, "date": a.date, "said": a.said,
         "recorded": time.strftime("%Y-%m-%dT%H:%M:%S"), "module": "sim/final_test/matrix/k23_leakage.py"}
    d["sha256"] = C.self_sha(d)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
