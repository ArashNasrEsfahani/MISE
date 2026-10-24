"""K23: practice examples practised in Persian, rendered in English in the package (prereg A24.2 item 2) -- ONLY under a recorded decision.

The defect (found 2026-10-20 in practice-k3-v1, attempt 9 g-l3-breakfast/fa): ltm_examples.ExampleIndex.render shows a practice example
"in its own practice language", so a SUCCESSFUL attempt practised in Persian (practice-k3-v1 attempt 3, g-l3-serve-hot-potato/fa) puts its
Persian instruction into the English package line 'Example from my practice (succeeded): "<instruction>" -> ...'. treatment.Packager
.render enforces A24.2 item 2 ("All stored knowledge records, rendered packages and welcome briefings are English for both EN and FA
presentations") and refuses the package ("Persian character in the package text"); the frozen capture exemption (capture.exempt_spans,
A31 D3) expects a Persian row as '«...» -> ', which an English package line never has. Kitchen 1 never met the case: no Kitchen 1
package (practice-v2 or the 270 test-v1 captures) rendered a Persian-practised example (its matching Persian attempts failed or shared
no object and operation with a later request), so the contradiction stayed latent. Kitchens 2 / 3 can meet it in practice, rehearsal
and test capture whenever such an example is retrieved.

The resolution implemented here (choice "english", the A24.2 reading): a Persian-practised example is RENDERED with the same item's
registered English instruction (the bilingual dataset pair, sim/dataset/general.json), its plan / outcome / wording otherwise unchanged;
retrieval and selection are unchanged (lex, keys, ranking and the precision filter still see the stored row); the frozen capture
exemption sees that row in its English form (lang "en", instruction = the dataset's English text), so it validates the English quote
byte for byte against the pinned dataset text. Nothing changes for an English-practised example, for the FS rendering (raw) or for any
package that shows no Persian-practised example -- in particular no Kitchen 1 package and none of practice-k3-v1 attempts 1-8.

Activation needs Arash's decision record docs/final-test/k23/decisions/K23-FA-EXAMPLES.json (written by this module's CLI). Without it
nothing is patched and the treatment refusal stands. The K23 entry points call install() in their own process only (k23_practice live,
k23_run capture); treatment.py, ltm_examples.py, ltm_package.py and capture.py are not edited.

    python sim/final_test/matrix/k23_fa_examples.py --check
    python sim/final_test/matrix/k23_fa_examples.py --write --by Arash --date 2026-10-20 --said "<his words>"
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                       # noqa: E402

DECISION_ID = "K23-FA-EXAMPLES"
SCHEMA = "k23.decision/1"
CHOICE = "english"
RULE = ("A24.2 item 2: a Persian-practised practice example is rendered in the English package with the same item's registered English "
        "instruction; retrieval unchanged; the A31 capture exemption validates it as that English text")


def decision_path(root: str | None = None) -> str:
    return os.path.join(C.paths(root)["docs"], "k23", "decisions", f"{DECISION_ID}.json")


def decision_problems(root: str | None = None) -> list:
    p = decision_path(root)
    if not os.path.exists(p):
        return [f"no decision record ({os.path.relpath(p, C.paths(root)['root'])})"]
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        return [f"decision record unreadable ({type(e).__name__})"]
    out = []
    if d.get("schema") != SCHEMA or d.get("id") != DECISION_ID or d.get("sha256") != C.self_sha(d):
        out.append("decision record schema / id / self-sha mismatch")
    if d.get("choice") != CHOICE or d.get("rule") != RULE:
        out.append(f"decision record choice is not {CHOICE!r} with this module's rule")
    if d.get("by") != "Arash" or not d.get("date") or not str(d.get("said") or "").strip():
        out.append("decision record is not Arash's attributed record (by / date / said)")
    return out


def english_texts(root: str | None = None) -> dict:
    with open(C.paths(root)["dataset"], encoding="utf-8") as f:
        return {it["id"]: it.get("en") for it in json.load(f)["items"]}


def persian_texts(root: str | None = None) -> dict:
    with open(C.paths(root)["dataset"], encoding="utf-8") as f:
        return {it["id"]: it.get("fa") for it in json.load(f)["items"]}


def english_row(r: dict, en: dict, fa: dict | None = None) -> dict:
    """A Persian practice row in its English form (lang en, the item's registered English instruction) -- ONLY when its stored Persian
    instruction is byte for byte the item's registered Persian text (Astra K23-FA-EXAMPLES-r1: validate the source before mapping);
    any other row is returned unchanged, so the frozen exemption judges it exactly as before."""
    if r.get("lang") != "fa":
        return r
    if fa is not None and (not r.get("instruction") or fa.get(r.get("item")) != r.get("instruction")):
        return r
    t = en.get(r.get("item"))
    if not t:
        return r
    return dict(r, lang="en", instruction=t, lang_practised="fa")


EVENT = "k23_fa_examples"


def decision_sha(root: str | None = None):
    p = decision_path(root)
    return C.sha256_file(p) if os.path.exists(p) else None


def record_event(ns: str, phase: str, root: str | None = None) -> None:
    """The activation boundary (Astra K23-FA-EXAMPLES-r1 4): every K23 process that installed the rendering records it, with the
    decision's sha, in the namespace's events journal before its step; the agent freeze checks that only the bound decision was used."""
    os.makedirs(C.ns_docs(ns, root), exist_ok=True)
    C.Journal(C.journal_path(ns, "events", root)).append(C.make_record("events", event=EVENT, phase=phase, namespace=ns, active=True,
                                                                       decision_sha256=decision_sha(root), rule=RULE))


_STATE = {"installed": False}


def install(root: str | None = None) -> bool:
    """Patch, in THIS process only, (1) ltm_examples.ExampleIndex.render: the English line of a Persian-practised example quotes the item's
    registered English instruction; (2) capture.exempt_spans (when capture is used): rows of Persian attempts are validated in their
    English form. Returns True when active; False (nothing patched) without a valid decision record."""
    if decision_problems(root):
        return False
    if _STATE["installed"]:
        return True
    en, fa_txt = english_texts(root), persian_texts(root)
    import ltm_examples as LE
    orig_render = LE.ExampleIndex.render

    def render(self, ex, lang, raw=False, query=None):
        if not raw and lang == "en" and ex.get("lang") == "fa":
            t = en.get(ex.get("item"))
            if not t:
                raise RuntimeError(f"no registered English instruction for practice item {ex.get('item')!r}")
            ex = dict(ex, instruction=t)
        return orig_render(self, ex, lang, raw=raw, query=query)
    render.k23_fa_examples = True
    LE.ExampleIndex.render = render
    try:
        import capture as CP
        orig_ex = CP.exempt_spans

        def exempt_spans(text, *, rows=(), **kw):
            if kw.get("form", "package") != "package":            # Astra r1: the FS form is not touched (K23 runs no FS arm)
                return orig_ex(text, rows=rows, **kw)
            return orig_ex(text, rows=[english_row(r, en, fa_txt) for r in (rows or ())], **kw)
        exempt_spans.k23_fa_examples = True
        CP.exempt_spans = exempt_spans
    except ImportError:                                     # pragma: no cover - practice processes never import capture
        pass
    _STATE["installed"] = True
    return True


def main(argv=None) -> int:
    import time
    ap = argparse.ArgumentParser(description="K23 FA-practised examples in English (decision record)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    ap.add_argument("--by"), ap.add_argument("--date"), ap.add_argument("--said")
    a = ap.parse_args(argv)
    if a.check:
        why = decision_problems()
        print(json.dumps({"decision": DECISION_ID, "active": not why, "problems": why}))
        return 0 if not why else 1
    if a.by != "Arash" or not a.date or not (a.said or "").strip():
        print("REFUSED: --write needs --by Arash --date YYYY-MM-DD --said \"<his words>\"", file=sys.stderr)
        return 2
    p = decision_path()
    if os.path.exists(p):
        print(f"REFUSED: {p} exists (a decision is recorded once)", file=sys.stderr)
        return 2
    d = {"schema": SCHEMA, "id": DECISION_ID, "choice": CHOICE, "rule": RULE, "by": a.by, "date": a.date, "said": a.said,
         "recorded": time.strftime("%Y-%m-%dT%H:%M:%S"), "module": "sim/final_test/matrix/k23_fa_examples.py"}
    d["sha256"] = C.self_sha(d)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
