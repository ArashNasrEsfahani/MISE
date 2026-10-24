"""M1 - practice with nested snapshots K = 0 / 1 / 3 / 5 / 10 x orders 0 / 1 / 2 (SPEC s12 M1, prereg 6.2 / 6.3, A11).

Gemma practises on practice-pool items only, in the canonical FloorPlan1 world (restored per episode with the item's setup), on G+M
settings: the planner reads the LIVE practice memory online (C2-f) through the one renderer (`PlannerContract`), AFTER the fixed start
sweep (A11: the prompt carries the post-sweep observation summary and the block is rendered from the post-sweep working memory).

  * K = STARTED attempts overall: an attempt counts as soon as it is drawn and begun, whatever happens (executed, rejected plan,
    inapplicable item, crash). A planner TRANSPORT failure is infrastructure, not experience: it aborts before anything is written, and
    the resume repeats that attempt (prereg s13). The order is `C.rng("practice", order)`: practice-pool items WITHOUT replacement,
    round-robin by level (L1, L2, L3, ...), each item's language drawn by the same RNG.
  * One practice memory per order, nested snapshots: the snapshot after attempt K goes to `C.snapshot_dir("o<order>k<K>")`
    (K = 0 is an empty memory, taken before attempt 1). Each carries `practice_manifest.json` = {order, k, state_hash, params_sha,
    channels, attempts: [item ids], leakage: {ok, hits}, sha256}. The LEAKAGE GATE runs on every snapshot: no test item id and no
    normalised test instruction (EN / FA) in the snapshot dump or its raw bytes (locations of shared objects are allowed: that is the
    intervention). A hit aborts.
  * Save after every episode, resumable: `practice.jsonl` (append-only, fsync'd) is the truth. Per order the practice memory lives
    in the work dir with a pre-attempt checkpoint; if a kill landed between consolidation and the journal line, the resume restores
    the checkpoint (state hash must equal the last journaled row's `state_hash_after`) and repeats the attempt.
  * Cost is recorded per attempt (primitives, wall seconds, Gemma calls, tokens) and summed in `practice_cost.json` + a
    `phase_complete` event when every requested order reached its largest K.
  * Test items are refused: every drawn item goes through `C.guard_items` (+ the executor allow-list / forbid-list), and the
    namespace for practice is `practice-v1` (`rehearsal-v1` for dev rehearsals).

    python sim/final_test/matrix/practice_k.py --orders 0,1,2 --k 0,1,3,5,10 --namespace practice-v1      (WSL, Gemma; run from the repo root)
    python sim/final_test/matrix/practice_k.py --dry-run                    (any python: prints the three orders, no simulator)
    python sim/final_test/matrix/practice_k.py --status                     (any python: verifies journal + every snapshot)

Python 3.10 compatible (WSL). Simulator and Ollama code is imported lazily.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                       # noqa: E402  (also puts sim/ on sys.path)
import treatment as T                                    # noqa: E402  (C4: practice-v2 on the package; inert in ft-v1 namespaces)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                        # pragma: no cover
    pass

KS = (0, 1, 3, 5, 10)
ORDERS = (0, 1, 2)
PHASE = "practice"
MANIFEST_NAME = "practice_manifest.json"
SCENE = "FloorPlan1"
PRACTICE_NAMESPACES = ("practice-v1", "rehearsal-v1", "practice-v2", "rehearsal-v3")     # the last two are ft-v2 (C4, treatment.py)
MAX_CONSECUTIVE_CRASHES = 2
MIN_NEEDLE = 8                                           # shorter normalised needles are skipped (they match anything)


class PracticeError(RuntimeError):
    pass


class LeakageError(PracticeError):
    pass


class ReservedItemRefused(PracticeError):
    """PRACTICE-V2 (Astra PRACTICE-V2 1-6): a reserved acceptance item reached an ft-v2 practice draw."""


# ================================================================================================ reserved acceptance items (ft-v2 practice only)
#: PRACTICE-V2: the A25 5 reservation (19 items) UNION the ACC-V2b acceptance slate (16 reserved L1/L2 + 8 new L3 + 2 spares) are
#: excluded from the ft-v2 practice pool BEFORE the seeded draw (prereg A25 5.3 "none planned by any model"; slate barrier "never ...
#: practice items"). Both sources are pinned by full sha256; a changed source refuses (fail closed). ft-v1 practice is untouched.
RESERVED_SOURCES = {
    "reservation": ("docs/final-test/kitchen_memory/signal_set_r2_fresh.json",
                    "42e612b31fbed50f91ec883d85114a7e5a21a770af9fe8eee72030649ef5da91"),
    "slate": ("sim/dataset/acceptance_v2_final_proposals.json",
              "174537f6ffd803b7fddbdcf6aa70f3e5071c7d0a727b22f062ba2e3eb68df161"),
}
RESERVED_N = 29                                          # 19 reserved + 10 new slate L3 (the 16 slate L1/L2 are among the 19)


def reserved_acceptance(root=None, items=None) -> dict:
    """The reserved acceptance set: {ids, texts [(id, lang, norm, raw)], n, sha, sources}. Schema, membership and pins checked; any
    difference raises ReservedItemRefused. `items` (dataset by id) adds the dataset EN/FA texts of the reserved ids."""
    base = C.paths(root)["root"]
    src = {}
    for key, (rel, want) in RESERVED_SOURCES.items():
        path = os.path.join(base, *rel.split("/"))
        try:
            got = C.sha256_file(path)
        except OSError as e:
            raise ReservedItemRefused(f"reserved-acceptance source missing: {rel}") from e
        if got != want:
            raise ReservedItemRefused(f"reserved-acceptance source changed: {rel} ({got[:16]} != pinned {want[:16]})")
        with open(path, encoding="utf-8") as f:
            src[key] = json.load(f)
    res = src["reservation"].get("acceptance_reservation") or {}
    r19 = list(res.get("items") or [])
    if len(r19) != 19 or len(set(r19)) != 19 or res.get("by_level") != {"L1": 8, "L2": 8, "L3": 3}:
        raise ReservedItemRefused("reservation schema: expected 19 unique ids, by_level L1 8 / L2 8 / L3 3")
    sl = src["slate"]
    parts = {k: list(sl.get(k) or []) for k in ("l1_l2_reserved", "l3_new", "l3_spares")}
    if [len(parts[k]) for k in ("l1_l2_reserved", "l3_new", "l3_spares")] != [16, 8, 2]:
        raise ReservedItemRefused("slate schema: expected 16 reserved L1/L2 + 8 new L3 + 2 spares")
    slate_ids = [x.get("id") for k in parts for x in parts[k]]
    if not set(x["id"] for x in parts["l1_l2_reserved"]) <= set(r19):
        raise ReservedItemRefused("slate: a reserved L1/L2 id is not in the A25 reservation")
    ids = sorted(set(r19) | set(slate_ids))
    if len(ids) != RESERVED_N:
        raise ReservedItemRefused(f"reserved set has {len(ids)} ids, expected {RESERVED_N}")
    texts, have = [], set()
    for k in parts:
        for x in parts[k]:
            for lang in ("en", "fa"):
                if not x.get(lang):
                    raise ReservedItemRefused(f"slate item {x.get('id')} lacks its {lang} text")
                texts.append((x["id"], lang, norm_text(x[lang]), str(x[lang])))
                have.add((x["id"], lang))
    for i in r19:
        it = (items or {}).get(i)
        for lang in ("en", "fa"):
            if it and it.get(lang):
                texts.append((i, lang, norm_text(it[lang]), str(it[lang])))
                have.add((i, lang))
    if items is not None:
        missing = [(i, lang) for i in ids for lang in ("en", "fa") if (i, lang) not in have]
        if missing:
            raise ReservedItemRefused(f"reserved needles lack {len(missing)} EN/FA texts: {missing[:3]}")
    return {"ids": ids, "texts": texts, "n": len(ids), "sha": C.sha256_text(json.dumps(ids)),
            "sources": {k: v[1] for k, v in RESERVED_SOURCES.items()}}


def guard_reserved(item_ids, reserved: dict) -> None:
    hit = sorted(set(item_ids) & set(reserved["ids"]))
    if hit:
        raise ReservedItemRefused(f"{len(hit)} reserved acceptance item(s) refused in ft-v2 practice: {hit[:5]}")


# ================================================================================================ the practice order
def level_of_items(items: dict) -> dict:
    return {i: it["level"] for i, it in items.items()}


def practice_order(pool, level_of: dict, order: int) -> list:
    """The practice order of one seed: pool items without replacement, round-robin by level, language drawn by the same RNG.
    Pure and stable across machines (`C.rng`). Returns [{n (1-based attempt), item, lang, level}]."""
    r = C.rng("practice", int(order))
    by = {}
    for i in sorted(pool):
        by.setdefault(level_of[i], []).append(i)
    for lv in sorted(by):
        r.shuffle(by[lv])
    levels = sorted(by)
    out, k = [], 0
    while any(by[lv] for lv in levels):
        for lv in levels:
            if by[lv]:
                k += 1
                out.append({"n": k, "item": by[lv].pop(0), "lang": r.choice(("en", "fa")), "level": lv})
    return out


def attempt_key(namespace: str, order: int, n: int, item: str, lang: str) -> str:
    return f"{namespace}|practice|o{order}|{n}|{item}|{lang}"


# ================================================================================================ leakage gate
_PERSIAN_DIGITS = {ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")}
_ARABIC_DIGITS = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}


def norm_text(s) -> str:
    """Instruction normalisation for the leakage gate: NFKC, lower case, Persian / Arabic letter variants unified, ZWNJ / tatweel /
    diacritics dropped, digits to ASCII, punctuation to a space, whitespace collapsed."""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = s.replace("ي", "ی").replace("ك", "ک").replace("ە", "ه").replace("ة", "ه")
    s = s.translate(_PERSIAN_DIGITS).translate(_ARABIC_DIGITS)
    out = []
    for ch in s:
        if ch in "‌‍‎‏ـ" or unicodedata.category(ch) == "Mn":
            continue
        out.append(ch if (ch.isalnum() or ch.isspace()) else " ")
    return " ".join("".join(out).split())


def test_needles(manifest: dict, items: dict) -> dict:
    """Everything that must not appear in a practice snapshot: test + capped-out item ids and their normalised EN / FA texts."""
    ids = sorted(set(manifest["test"]) | set(manifest.get("test_uncapped_not_scored") or []))
    texts = []
    for i in ids:
        it = items.get(i)
        if it:
            for lang in ("en", "fa"):
                if it.get(lang):
                    texts.append((i, lang, norm_text(it[lang]), str(it[lang])))
    return {"ids": ids, "texts": texts}


def _dump_text(db_path: str) -> str:
    import ltm_store
    s = ltm_store.Store(db_path, readonly=True)
    try:
        return C.canon(s.canonical_dump())
    finally:
        s.close()


ROWS_EXEMPT_FIELD = "instruction"                     # PRACTICE-V2 (Astra PRACTICE-V2-needles 1-5): the ONE exempt field
#: PRACTICE-V2 option (A) (ARASH DECISION 2026-10-13 "Go for A"; prereg 6.3 / A24 7(b) amendment DRAFT): the G1 request line inside
#: the stored package text of a practice row is exempt ONLY when it is byte-identical to the line recomputed here from that practice
#: item's own pinned dataset text (+ the hand state its pinned setup leaves). Field = row["block"]["text"], one whole line.
REQLINE_FIELD = ("block", "text")
REQLINE_PREFIX = "The request in English: "


def request_line(item: dict, lang: str):
    """The G1 line the ft-v2 package renders for this practice presentation, recomputed from PINNED inputs only (the dataset text in
    `lang`, and the holding its pinned setup leaves when every setup step succeeds), or None when G1 renders nothing. Same code path as
    ltm_package (render_request(_parse_request(q))), same format string."""
    import ltm_package as LP
    import step4_refplans as RP
    setup = list(item.get("setup") or [])
    holding = RP.setup_holding(setup, [{"ok": True}] * len(setup))
    rendered = LP.render_request(LP._parse_request({"instruction": item[lang], "lang": lang, "holding": holding}))
    return ("The request in English: %s." % rendered) if rendered else None


def _exempt_reqline(r: dict, n: int, item: str, lang: str, want) -> tuple:
    """-> (row with the validated request line blanked, audit list, problems). Fail closed: the field must be a string when a block
    exists; a line carrying the G1 prefix must equal `want` byte for byte; at most ONE such line; a G1 line where the recompute gives
    None is a problem."""
    blk = r.get(REQLINE_FIELD[0])
    if blk is None:
        return r, [], []
    if not isinstance(blk, dict) or not isinstance(blk.get(REQLINE_FIELD[1], ""), str):
        return r, [], [f"row {n + 1} block.text is not a string"]
    text = blk.get(REQLINE_FIELD[1]) or ""
    lines = text.split("\n")
    g1 = [k for k, ln in enumerate(lines) if ln.startswith(REQLINE_PREFIX) or ln.strip().startswith(REQLINE_PREFIX.strip())]
    if not g1:
        return r, [], []
    if want is None:
        return r, [], [f"row {n + 1} has a request line but none is recomputed from its pinned text"]
    if len(g1) > 1:
        return r, [], [f"row {n + 1} has {len(g1)} request lines (expected at most 1)"]
    if lines[g1[0]] != want:
        return r, [], [f"row {n + 1} request line differs from the line recomputed from its pinned text"]
    lines[g1[0]] = ""
    r = dict(r)
    r[REQLINE_FIELD[0]] = dict(blk, **{REQLINE_FIELD[1]: "\n".join(lines)})
    return r, [{"item": item, "lang": lang, "field": f"{n + 1}.block.text#request_line", "text_sha": C.sha256_text(want), "count": 1}], []


def _exempt_rows(data: bytes, expected, refused: set) -> tuple:
    """ft-v2 rows file -> (bytes with the validated `instruction` field blanked, audit, problems). A row is exempt only if it is a JSON
    object at its schedule position whose item / lang equal the PINNED schedule, whose item is no test / reserved id, and whose
    instruction equals the pinned dataset text EXACTLY. Anything else is a problem (fail closed) and nothing is blanked.
    Option (A): an `expected` entry may carry a 4th element = the request line recomputed from the pinned text (request_line); then
    that ONE line of the same validated row's block.text is blanked too (_exempt_reqline), audited separately under "reqline"."""
    lines = [ln for ln in data.decode("utf-8").split("\n") if ln.strip()]
    probs, audit, out, rq_audit = [], [], [], []
    if len(lines) != len(expected):
        probs.append(f"rows file has {len(lines)} rows, the schedule prefix has {len(expected)}")
    for n, ln in enumerate(lines):
        try:
            r = json.loads(ln)
        except ValueError:
            probs.append(f"row {n + 1} is not JSON")
            out.append(ln)
            continue
        if not isinstance(r, dict) or n >= len(expected):
            probs.append(f"row {n + 1} is malformed or beyond the schedule")
            out.append(ln)
            continue
        item, lang, text = expected[n][:3]
        if r.get("item") != item or r.get("lang") != lang:
            probs.append(f"row {n + 1} is not schedule position {n + 1}")
        elif item in refused:
            probs.append(f"row {n + 1} draws a test / reserved id")
        elif r.get(ROWS_EXEMPT_FIELD) != text:
            probs.append(f"row {n + 1} instruction differs from the pinned dataset text")
        else:
            r = dict(r)
            r[ROWS_EXEMPT_FIELD] = ""
            audit.append({"item": item, "lang": lang, "field": f"{n + 1}.{ROWS_EXEMPT_FIELD}", "text_sha": C.sha256_text(text)})
            if len(expected[n]) > 3:                          # option (A): only for a row already validated above
                r, rq, pb = _exempt_reqline(r, n, item, lang, expected[n][3])
                rq_audit += rq
                probs += pb
        out.append(json.dumps(r, ensure_ascii=False, sort_keys=True))
    if probs:
        return data, audit, probs, rq_audit
    return ("\n".join(out) + "\n").encode("utf-8"), audit, probs, rq_audit


def leakage_gate(snap_dir: str, needles: dict, allow=None) -> dict:
    """Scan the snapshot's canonical dump (normalised) and the raw bytes of its files (exact + normalised) for test ids and test
    instructions. {ok, hits: [{kind, item, lang?}], needles, scanned}. Hits name the item id, never the text.
    PRACTICE-V2 (ft-v2 only): `allow` = {"rows_file", "expected": [(item, lang, pinned text)] from the seeded schedule, "refused": ids}.
    Only the validated `instruction` field of the snapshot's own rows is exempt from TEXT matching (a practice instruction may embed a
    shorter test / reserved sentence); ids are matched on the UNMASKED bytes and dump, the dump and every other field / file get no
    exemption, and an invalid rows file is a hit. allow=None is the unchanged gate.
    Option (A) (prereg 6.3 / A24 7(b) amendment DRAFT): an `expected` entry (item, lang, text, request_line) also exempts that ONE
    byte-identical G1 line of the same validated row's block.text; a differing / duplicate / unexpected G1 line is exemption_invalid."""
    hits, scanned = [], []
    dump_norm, dump_raw = None, None
    db = os.path.join(snap_dir, "ltm.db")
    if os.path.exists(db):
        try:
            dump_raw = _dump_text(db)
            dump_norm = norm_text(dump_raw)
            scanned.append("canonical_dump")
        except Exception as e:                           # a corrupt store must not pass the gate silently
            hits.append({"kind": "dump_unreadable", "item": None, "error": type(e).__name__})
    raws, text_raws, exempt = [], [], None
    for name in sorted(os.listdir(snap_dir)):
        p = os.path.join(snap_dir, name)
        if os.path.isfile(p) and name != MANIFEST_NAME and not name.endswith("-journal"):
            with open(p, "rb") as f:
                data = f.read()
            raws.append(data)
            if allow is not None and name == allow["rows_file"]:
                data, audit, probs, rq = _exempt_rows(data, allow["expected"], set(allow.get("refused") or ()))
                exempt = {"rows_file": name, "exempt": audit, "n": len(audit), "problems": probs}
                if any(len(e) > 3 for e in allow["expected"]):        # option (A) active: the request-line audit is always recorded
                    exempt.update(reqline=rq, reqline_n=len(rq))
                for pb in probs:
                    hits.append({"kind": "exemption_invalid", "item": None, "error": pb})
            text_raws.append(data)
            scanned.append(name)
    if allow is not None:
        if exempt is None:
            exempt = {"rows_file": allow["rows_file"], "exempt": [], "n": 0, "problems": []}
            if allow["expected"]:
                hits.append({"kind": "exemption_invalid", "item": None, "error": "rows file missing"})
        elif not exempt["problems"] and exempt["n"] != len(allow["expected"]):
            hits.append({"kind": "exemption_invalid", "item": None, "error": "exempt count differs from the schedule prefix"})
    raw_all = b"\n".join(raws)
    traw_all = b"\n".join(text_raws)
    raw_norm = norm_text(traw_all.decode("utf-8", errors="ignore")) if traw_all else ""
    low_ids = [(i, i.lower()) for i in needles["ids"]]
    seen = set()
    if allow is None:
        for i, li in low_ids:
            if (li.encode("utf-8") in raw_all.lower()) or (dump_norm is not None and norm_text(li) in dump_norm and len(norm_text(li)) >= MIN_NEEDLE):
                hits.append({"kind": "item_id", "item": i})
                seen.add(("id", i))
    else:                                                # PRACTICE-V2: ids as whole tokens (a test id can be a PREFIX of a longer
        import re                                        # practice id), on the unmasked bytes + the raw dump
        low_raw, low_dump = raw_all.lower(), (dump_raw or "").lower()
        for i, li in low_ids:
            pat = r"(?<![a-z0-9-])" + re.escape(li) + r"(?![a-z0-9-])"
            if re.search(pat.encode("utf-8"), low_raw) or re.search(pat, low_dump):
                hits.append({"kind": "item_id", "item": i})
                seen.add(("id", i))
    n_checked = 0
    for i, lang, nt, raw in needles["texts"]:
        if len(nt) < MIN_NEEDLE:
            continue
        n_checked += 1
        if (dump_norm is not None and nt in dump_norm) or nt in raw_norm or raw.encode("utf-8") in traw_all:
            hits.append({"kind": "instruction", "item": i, "lang": lang})
    out = {"ok": not hits, "hits": hits, "needles": {"ids": len(low_ids), "texts": n_checked}, "scanned": scanned}
    if exempt is not None:
        out["exempt"] = exempt
    return out


# ================================================================================================ small helpers
def _sha_dir_manifest(snap_dir: str) -> str:
    return C.sha256_file(os.path.join(snap_dir, "manifest.json"))


def snapshot_manifest_path(snap_dir: str) -> str:
    return os.path.join(snap_dir, MANIFEST_NAME)


def read_snapshot_manifest(snap_dir: str):
    try:
        with open(snapshot_manifest_path(snap_dir), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def verify_snapshot_dir(snap_dir: str, order: int, k: int) -> list:
    """[] = the snapshot is complete and consistent (store verified, manifest self-sha, order / k, leakage ok)."""
    import ltm_store
    bad = []
    m = read_snapshot_manifest(snap_dir)
    if m is None:
        return ["practice manifest missing"]
    if m.get("sha256") != C.self_sha(m):
        bad.append("practice manifest self-sha mismatch")
    if m.get("order") != order or m.get("k") != k:
        bad.append("order / k differ from the path")
    if len(m.get("attempts") or []) != k:
        bad.append("attempt list length differs from K")
    if not (m.get("leakage") or {}).get("ok"):
        bad.append("leakage gate not ok")
    bad += ltm_store.Store.verify_snapshot(snap_dir)
    try:
        if ltm_store.Store.read_manifest(snap_dir).get("state_hash") != m.get("state_hash"):
            bad.append("state hash differs from the store manifest")
    except Exception:
        bad.append("store manifest unreadable")
    return bad


def verify_nesting(snap_root: str, order: int, ks=KS) -> list:
    """Snapshots of one order are nested: the attempts of a smaller K are a prefix of the attempts of a larger K."""
    bad, prev = [], None
    for k in sorted(ks):
        m = read_snapshot_manifest(os.path.join(snap_root, f"o{order}", f"k{k}"))
        if m is None:
            bad.append(f"o{order}k{k}: missing")
            continue
        a = m.get("attempts") or []
        if prev is not None and a[:len(prev)] != prev:
            bad.append(f"o{order}k{k}: attempts are not an extension of the previous snapshot")
        prev = a
    return bad


def cost_summary(rows: list) -> dict:
    """Practice cost per order and in total: episodes, primitives, wall time, Gemma calls, tokens (prereg 6.2)."""
    def acc(rs):
        return {"episodes": len(rs), "primitives": sum(int(r.get("prims") or 0) for r in rs),
                "wall_s": round(sum(float(r.get("wall_s") or 0.0) for r in rs), 1),
                "gemma_calls": sum(int(r.get("calls") or 0) for r in rs),
                "tokens_in": sum(int(r.get("tokens_in") or 0) for r in rs), "tokens_out": sum(int(r.get("tokens_out") or 0) for r in rs),
                "plan_secs": round(sum(float(r.get("plan_secs") or 0.0) for r in rs), 1)}
    by = {}
    for r in rows:
        by.setdefault(r["order"], []).append(r)
    return {"total": acc(rows), "by_order": {str(o): acc(rs) for o, rs in sorted(by.items())}}


def _jsonable(x):
    return json.loads(json.dumps(x, default=str, ensure_ascii=False))


# ================================================================================================ the planner hook (A11 only)
class PracticeHook:
    """The only place where memory reaches the planner: `ltm.memory_block(post["wm"], ...)` -> `PlannerRunner.plan(memory_block=)`;
    `PlannerContract` renders it (gate 12.11). Runs AFTER the start sweep (A11): the prompt carries the post-sweep summary."""

    def __init__(self, runner, contract, counter, channels, plan_secs_clock=time.time):
        self.runner, self.contract, self.counter, self.channels = runner, contract, counter, tuple(channels)
        self.clock = plan_secs_clock

    def __call__(self, req, it, holding, ltm, post=None):
        """A bug in the hook is NOT a simulator crash: any error here is fatal (LiveInvariant), never a 'crash' attempt."""
        import step5_live as sl
        try:
            return self._plan(req, it, holding, ltm, post)
        except (sl.LiveInvariant, PracticeError):
            raise
        except Exception as e:
            raise sl.LiveInvariant(f"plan hook error ({type(e).__name__}: {str(e)[:160]}): fix the code; this is not a practice outcome") from e

    def _plan(self, req, it, holding, ltm, post):
        import step5_live as sl
        from planners.contract import MEMORY_HEADER, prompt_sha256
        if post is None:
            raise PracticeError("practice runs under A11 only: the planner needs the post-sweep dict")
        blk = None
        if ltm is not None and self.channels:
            blk = ltm.memory_block(post["wm"], it["level"], req.lang, channels=self.channels, holding=holding)
        summary = list(post["summary"])
        mtext = blk["text"] if blk and blk["text"] else None
        msgs = self.contract.messages(it[req.lang], summary, holding, memory_block=mtext)
        t0 = self.clock()
        r = self.runner.plan(it[req.lang], summary, holding, memory_block=mtext)
        steps = [[s.get("action"), s.get("object"), s.get("to"), s.get("arg")] for s in (r["plan"] or [])] if r["ok"] else []
        info = {"arm": "G+M", "ok": r["ok"], "first_ok": r["first_ok"], "repaired": r["repaired"], "rejected": r["rejected"],
                "infra_failure": r["infra_failure"], "calls": r["calls"], "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"],
                "plan_secs": round(self.clock() - t0, 2), "plan": steps,
                "first_plan": [[s.get("action"), s.get("object"), s.get("to"), s.get("arg")] for s in (r["first_plan"] or [])],
                "say": r["say"], "normalised": r["normalised"], "holding": holding, "request_sha": prompt_sha256(msgs),
                "plan_order": "after_sweep", "summary_sha": post["summary_sha"], "summary_n": len(summary),
                "block_sha": C.sha256_text(mtext) if mtext else None,
                "block": None if blk is None else {"empty": blk["empty"], "ids": [x["id"] for x in blk["records"]], "tokens": blk["tokens"],
                                                   "n_records": len(blk["records"]), "text": blk["text"] or ""}}
        if mtext and self.counter is not None:
            info["block"]["real_tokens"] = self.counter.counts(mtext)
            info["block"]["real_tokens_with_header"] = self.counter.counts(MEMORY_HEADER + "\n" + mtext)
            info["block"]["budget"] = ltm.retriever().budgets[it["level"]]
        if r["infra_failure"]:
            raise sl.LiveInvariant("planner transport failure (infrastructure, not experience): re-run resumes this attempt")
        if not r["ok"]:
            info["not_executed"] = "rejected"
        return {"steps": r["plan"] if r["ok"] else None, "info": info}


class PackagePracticeHook:
    """C4 / ft-v2 practice planner hook: the PACKAGE (treatment.Packager, the pinned profile, universal mask, D2) rendered from the
    PRE-ATTEMPT practice store (the engine's checkpoint) + the rows journaled earlier in this order (leave-out of the current item), on
    the ft-v2 contract with the prompt summary of the agent's full post-sweep list and observed_types. The engine sets `ctx` before
    every attempt ({"key", "db", "rows"}) and `executor` once."""

    def __init__(self, runner, contract, packager, plan_secs_clock=time.time):
        self.runner, self.contract, self.packager, self.clock = runner, contract, packager, plan_secs_clock
        self.executor = None
        self.ctx = None
        self.channels = ()

    def __call__(self, req, it, holding, ltm, post=None):
        import step5_live as sl
        try:
            return self._plan(req, it, holding, ltm, post)
        except (sl.LiveInvariant, PracticeError):
            raise
        except T.TreatmentError as e:
            raise PracticeError(f"ft-v2 practice: {e}") from e
        except Exception as e:
            raise sl.LiveInvariant(f"plan hook error ({type(e).__name__}: {str(e)[:160]}): fix the code; this is not a practice outcome") from e

    def _plan(self, req, it, holding, ltm, post):
        import step5_live as sl
        from planners.contract import prompt_sha256
        if post is None:
            raise PracticeError("practice runs under A11 only: the planner needs the post-sweep dict")
        if not self.ctx:
            raise PracticeError("the ft-v2 practice hook has no attempt context (the engine sets it before every attempt)")
        full = T.full_summary(self.executor, post)
        psum = T.prompt_summary(full)
        obs = T.observed_types(psum)
        ctx, self.ctx = self.ctx, None
        try:
            pk = self.packager.render(ctx["key"], ctx["db"], ctx["rows"], item=it["id"], instruction=it[req.lang], lang=req.lang,
                                      holding=holding, summary_lines=psum)
        finally:
            self.packager.drop(ctx["key"])
        mtext = pk["text"] or None
        msgs = self.contract.messages(it[req.lang], psum, holding, memory_block=mtext)
        t0 = self.clock()
        r = self.runner.plan(it[req.lang], psum, holding, memory_block=mtext, observed_types=set(obs))
        steps = [[s.get("action"), s.get("object"), s.get("to"), s.get("arg")] for s in (r["plan"] or [])] if r["ok"] else []
        info = {"arm": "G+M", "ok": r["ok"], "first_ok": r["first_ok"], "repaired": r["repaired"], "rejected": r["rejected"],
                "infra_failure": r["infra_failure"], "calls": r["calls"], "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"],
                "plan_secs": round(self.clock() - t0, 2), "plan": steps,
                "first_plan": [[s.get("action"), s.get("object"), s.get("to"), s.get("arg")] for s in (r["first_plan"] or [])],
                "say": r["say"], "normalised": r["normalised"], "holding": holding, "request_sha": prompt_sha256(msgs),
                "plan_order": "after_sweep", "summary_sha": post["summary_sha"], "summary_n": len(post["summary"]),
                "prompt_summary_sha": C.sha256_text("\n".join(psum)), "prompt_summary_n": len(psum), "observed_types": obs,
                "codes_by_attempt": r.get("codes_by_attempt"),
                "block_sha": C.sha256_text(mtext) if mtext else None,
                "block": {"empty": not mtext, "ids": list(pk.get("kept") or []), "tokens": pk["tokens"], "n_records": len(pk.get("kept") or []),
                          "text": pk["text"] or "", "package": {k: pk.get(k) for k in ("profile", "mask", "off", "params_sha", "query_sha",
                                                                                         "pkg_sha", "d2_dropped", "leave_out", "need",
                                                                                         "empty_reason", "parser_ok")},
                          "source": {"rows_sha": T.rows_sha(ctx["rows"]), "rows_n": len(ctx["rows"]), "db_sha": C.sha256_file(ctx["db"])}}}
        if r["infra_failure"]:
            raise sl.LiveInvariant("planner transport failure (infrastructure, not experience): re-run resumes this attempt")
        if not r["ok"]:
            info["not_executed"] = "rejected"
        return {"steps": r["plan"] if r["ok"] else None, "info": info}


# ================================================================================================ the practice engine
class Practice:
    """One engine for the live run (WSL) and the offline selftest (a fake executor). `executor.execute(req)` returns an
    `EpisodeResult`-like object; the engine owns the practice LTM, the journal, the snapshots and the resume logic."""

    def __init__(self, *, namespace, orders, ks, executor, hook=None, counter=None, channels=("places",), kitchen=SCENE, root=None,
                 snap_root=None, journal=None, work=None, events=None, limit=None, fault=None, log=print, allow_test=False,
                 ident=None, hints=None):
        if namespace not in PRACTICE_NAMESPACES:
            raise PracticeError(f"practice runs in {PRACTICE_NAMESPACES}, not {namespace!r}")
        # C4: practice-v2 / rehearsal-v3 practise on the PACKAGE (treatment.py); ft-v1 practice is unchanged
        self.ft2 = T.is_ft2(namespace)
        if self.ft2:
            if tuple(channels or ()) not in ((), ("places",)):
                raise PracticeError("ft-v2 practice renders the package: memory channels are an ft-v1 option")
            channels = ()
            bad = [int(o) for o in orders if int(o) not in T.FT2["practice_orders"]]
            if bad:
                raise PracticeError(f"ft-v2 practice runs order(s) {T.FT2['practice_orders']} only (A27); refused {bad}")
            if hook is not None and not isinstance(hook, PackagePracticeHook) and not getattr(hook, "ft2_hook", False):
                raise PracticeError("ft-v2 practice needs the package hook (PackagePracticeHook), not the places hook")
            snap_root = snap_root or T.snapshot_root(root, scene=C.ns_scene(namespace))       # K23: per-kitchen subtree
            self.ident = ident or T.identity(namespace, root)
            self.hints = hints
        else:
            self.ident, self.hints = None, None
        self.ns, self.orders, self.ks = namespace, [int(o) for o in orders], sorted({int(k) for k in ks})
        if any(k not in KS for k in self.ks) or any(o not in ORDERS for o in self.orders):
            raise PracticeError("orders must be in 0..2 and K in 0 / 1 / 3 / 5 / 10")
        self.ex, self.hook, self.counter, self.channels, self.kitchen = executor, hook, counter, tuple(channels), kitchen
        self.root = root
        P = C.paths(root)
        self.snap_root = snap_root or P["snapshots"]
        self.journal = C.Journal(journal or C.journal_path(namespace, "practice", root))
        self.events = C.Journal(events or C.journal_path(namespace, "events", root))
        self.work = work or C.ns_work(namespace, root)
        self.limit, self.fault, self.log, self.allow_test = limit, fault, log, allow_test
        self.manifest = C.load_manifest(root)
        with open(P["dataset"], encoding="utf-8") as f:
            self.items = {i["id"]: i for i in json.load(f)["items"]}
        self.pool = sorted(self.manifest[C.NAMESPACES[namespace]["items"]])
        self.needles = test_needles(self.manifest, self.items)
        self.reserved = None
        if self.ft2:                                            # PRACTICE-V2: reserved acceptance items excluded BEFORE the seeded draw
            self.reserved = reserved_acceptance(root, self.items)
            full = self.pool
            self.pool = sorted(set(full) - set(self.reserved["ids"]))
            self.reserved["pool_before"], self.reserved["pool_after"] = len(full), len(self.pool)
            self.reserved["in_pool"] = sorted(set(full) & set(self.reserved["ids"]))
            self.needles = {"ids": sorted(set(self.needles["ids"]) | set(self.reserved["ids"])),
                            "texts": list(self.needles["texts"]) + list(self.reserved["texts"])}
        self.max_k = max(self.ks)
        self.orders_plan = {o: practice_order(self.pool, level_of_items(self.items), o) for o in self.orders}
        for o in self.orders:                                   # the guard BEFORE anything runs, and again on every resume
            C.guard_items([p["item"] for p in self.orders_plan[o][:self.max_k]], namespace, allow_test=allow_test, root=root)
            if self.ft2:
                guard_reserved([p["item"] for p in self.orders_plan[o][:self.max_k]], self.reserved)
        self.mem = None
        self._crashes = 0
        self._prev = []                                         # ft-v2: the journaled records of the current order (package rows)

    # ---- paths
    def snap_dir(self, o, k):
        return os.path.join(self.snap_root, f"o{o}", f"k{k}")

    def _wdir(self, o):
        return os.path.join(self.work, f"o{o}")

    def _params(self):
        import ltm as L
        return L.load_params()

    # ---- memory management
    def _open_mem(self, o, fresh=False):
        import ltm as L
        import ltm_fragments as lf
        wd = self._wdir(o)
        db = os.path.join(wd, "ltm.practice.db")
        if fresh:
            shutil.rmtree(wd, ignore_errors=True)
        os.makedirs(wd, exist_ok=True)
        mem = L.LongTermMemory(db, kitchen=self.kitchen, params=self._params(), count_tokens=self.counter)
        mem.fragment_ledger = lf.FragmentLedger.load(os.path.join(wd, "frag"))
        return mem

    def _close_mem(self):
        if self.mem is not None:
            self.mem.close()
            self.mem = None

    def _restore_ckpt(self, o):
        """Replace the practice DB by the pre-attempt checkpoint (state after the last journaled row)."""
        import ltm_fragments as lf
        wd = self._wdir(o)
        ck = os.path.join(wd, "ckpt")
        self._close_mem()
        db = os.path.join(wd, "ltm.practice.db")
        for suf in ("", "-journal", "-wal", "-shm"):
            if os.path.exists(db + suf):
                os.remove(db + suf)
        shutil.copyfile(os.path.join(ck, "ltm.db"), db)
        shutil.rmtree(os.path.join(wd, "frag"), ignore_errors=True)
        os.makedirs(os.path.join(wd, "frag"), exist_ok=True)
        lf.FragmentLedger.load(ck).save(os.path.join(wd, "frag"))
        self.mem = self._open_mem(o)

    def _prepare_order(self, o, rows):
        """Open (or repair) the practice memory of order `o` so that it equals the state after the last journaled row."""
        if not rows:
            self._close_mem()
            self.mem = self._open_mem(o, fresh=True)
            return
        last = rows[-1]
        self._close_mem()
        self.mem = self._open_mem(o)
        if self.mem.state_hash() == last["state_hash_after"]:
            return
        ck = os.path.join(self._wdir(o), "ckpt")
        import ltm_store
        if os.path.exists(os.path.join(ck, "manifest.json")) and not ltm_store.Store.verify_snapshot(ck) \
                and ltm_store.Store.read_manifest(ck)["state_hash"] == last["state_hash_after"]:
            self.log(f"[practice o{o}] a kill landed after consolidation but before the journal line: restored the pre-attempt checkpoint")
            self._restore_ckpt(o)
            if self.mem.state_hash() != last["state_hash_after"]:
                raise PracticeError(f"o{o}: restored checkpoint does not reproduce the journaled state")
            return
        raise PracticeError(f"o{o}: practice memory (state {self.mem.state_hash()[:12]}) matches neither the last journal row nor the "
                            f"checkpoint; refusing to continue (fail fast)")

    # ---- snapshots
    def take_snapshot(self, o, k, attempts_done, state_hash_expected=None, rows=None):
        d = self.snap_dir(o, k)
        res = self.mem.snapshot(d, overwrite=True)
        rsha = None
        if self.ft2:                                           # C4 / Astra Q5: the example-row prefix travels WITH the snapshot (content-bound)
            prefix = list(rows or [])[:k]
            if len(prefix) != k:
                raise PracticeError(f"o{o}k{k}: the snapshot needs exactly {k} practice rows, got {len(prefix)}")
            rsha = T.write_rows(d, prefix)
        if self.ft2:                                           # PRACTICE-V2: mask only this snapshot's own (guarded) practice instructions
            sched = self.orders_plan[o][:k]                    # the PINNED seeded schedule, not the snapshot's own manifest
            if [(a["item"], a["lang"]) for a in attempts_done[:k]] != [(p["item"], p["lang"]) for p in sched]:
                raise PracticeError(f"o{o}k{k}: the attempts differ from the seeded schedule")
            guard_reserved([p["item"] for p in sched], self.reserved)
            refused = set(self.manifest["test"]) | set(self.manifest.get("test_uncapped_not_scored") or []) | set(self.reserved["ids"])
            allow = {"rows_file": T.ROWS_FILE, "refused": refused,
                     "expected": [(p["item"], p["lang"], self.items[p["item"]][p["lang"]], request_line(self.items[p["item"]], p["lang"]))
                                  for p in sched]}           # option (A): + the G1 line recomputed from the pinned text
            leak = leakage_gate(d, self.needles, allow=allow)
        else:
            leak = leakage_gate(d, self.needles)
        man = {"order": o, "k": k, "state_hash": res["state_hash"], "params_sha": self.mem.params.get("_sha256"),
               "channels": list(self.channels), "attempts": [a["item"] for a in attempts_done[:k]], "attempt_langs": [a["lang"] for a in attempts_done[:k]],
               "namespace": self.ns, "kitchen": self.kitchen, "fragments_sha256": res.get("fragments_sha256"),
               "store_manifest_sha256": _sha_dir_manifest(d), "leakage": leak, "spec": C.SPEC_VERSION}
        if self.ft2:
            man.update(treatment=T.FT2_VERSION, treatment_id=self.ident["sha"], package_profile=T.FT2["package_profile"],
                       rows_file=T.ROWS_FILE, rows_sha=rsha, rows_n=k)
            man["reserved_acceptance"] = {"sha": self.reserved["sha"], "n": self.reserved["n"], "sources": self.reserved["sources"],
                                          "pool_after": self.reserved["pool_after"]}
        man["sha256"] = C.self_sha(man)
        with open(snapshot_manifest_path(d), "w", encoding="utf-8") as f:
            json.dump(man, f, indent=1, ensure_ascii=False, sort_keys=True)
        if not leak["ok"]:
            raise LeakageError(f"o{o}k{k}: leakage gate FAILED: {leak['hits'][:3]}")
        bad = verify_snapshot_dir(d, o, k)
        if bad:
            raise PracticeError(f"o{o}k{k}: snapshot verification failed: {bad}")
        if state_hash_expected is not None and man["state_hash"] != state_hash_expected:
            raise PracticeError(f"o{o}k{k}: snapshot state differs from the practice memory")
        return man

    # ---- one attempt
    def _request(self, key, it, lang):
        import hashlib
        import step5_eval as se
        if self.ft2:                                           # C4: prior chooser + hook B; the store still learns from the event stream
            if self.hook is not None:
                self.hook.ctx = {"key": key, "db": os.path.join(self._wdir(self._cur_o), "ckpt", "ltm.db"), "rows": list(self._prev)}
                if hasattr(self.hook, "executor"):
                    self.hook.executor = self.ex
            T.install_policy(self.ex, "briefing", self.hints)
        return se.EpisodeRequest(cell=key, episode_id="ep:" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24], run_id=f"{self.ns}-practice",
                                 item=it, lang=lang, level=it["level"], mode="practice", world="reset", world_kind=se.CANONICAL,
                                 world_seed=0, policy_seed=0, arm="G+M", features=frozenset({"places"}), memory=self.mem,
                                 plan_source="gemma", kitchen=self.kitchen)

    def _sightings(self):
        last = getattr(self.ex, "last", None) or {}
        ad = last.get("adapter")
        out, seen = [], set()
        for ev in (getattr(ad, "stream", None) or []):
            if ev.get("kind") == "sighting":
                k = (ev.get("otype"), ev.get("in_rtype"))
                if k not in seen:
                    seen.add(k)
                    out.append({"otype": ev.get("otype"), "in_rtype": ev.get("in_rtype"), "state": ev.get("state")})
        return out

    def attempt(self, o, p, rows_so_far):
        import ltm_fragments as lf
        import ltm_lessons as ll
        import step5_live as sl
        n, item, lang = p["n"], p["item"], p["lang"]
        key = attempt_key(self.ns, o, n, item, lang)
        if self.ft2:
            guard_reserved([item], self.reserved)
        self._cur_o = o
        it = self.items[item]
        req = self._request(key, it, lang)
        mem = self.mem
        mem.reset_world(req.episode_id, "task_reset", frame=getattr(self.ex, "frame", "f0"), map_rev=getattr(self.ex, "map_rev", "m0"))
        h0 = mem.state_hash()
        idx_before = lf.build_index(mem.store, self.kitchen, mem.fragment_ledger)["fragments"]
        t0 = time.time()
        crash = None
        try:
            res = self.ex.execute(req)
        except (sl.LiveInvariant, PracticeError, C.TestItemRefused, KeyboardInterrupt):
            raise
        except Exception as e:                                   # a crash is a STARTED attempt (prereg 6.2); no consolidation
            res, crash = None, f"{type(e).__name__}: {str(e)[:160]}"
        wall = round(time.time() - t0, 1)
        if mem.state_hash() != h0:
            raise sl.LiveInvariant(f"the executor wrote to the practice memory during {key}")
        ex_ = (getattr(res, "extra", None) or {}) if res is not None else {}
        core = ex_.get("core") or {}
        a = core.get("audit") or {}
        if int(a.get("violations") or 0) or int((a.get("counts") or {}).get("oracle") or 0):
            raise sl.LiveInvariant(f"honesty audit failed in {key}")
        if ex_.get("status") == "infra_blocked":
            raise sl.LiveInvariant(f"{key}: execution blocked by the state check (infra); re-run resumes this attempt")
        plan = ex_.get("plan") or {}
        not_exec = bool(ex_.get("plan_not_executed"))
        inapp = getattr(res, "inapplicable", None) if res is not None else None
        wm = getattr(res, "wm", None) if res is not None else None
        if crash:
            status = "crash"
        elif inapp:
            status = "inapplicable"
        elif not_exec:
            status = "rejected"
        else:
            status = "executed"
        row = {"key": key, "namespace": self.ns, "order": o, "attempt": n, "item": item, "lang": lang, "level": it["level"],
               "instruction": it[lang], "status": status, "inapplicable": inapp, "crash": crash, "wall_s": wall,
               "goal_ok": getattr(res, "goal_ok", None) if res is not None else None,
               "admission": getattr(res, "admission", None) if res is not None else None,
               "plan_ok": plan.get("ok"), "first_ok": plan.get("first_ok"), "repaired": plan.get("repaired"), "plan": plan.get("plan"),
               "first_plan": plan.get("first_plan"), "say": plan.get("say"), "calls": plan.get("calls", 0), "tokens_in": plan.get("tokens_in", 0),
               "tokens_out": plan.get("tokens_out", 0), "plan_secs": plan.get("plan_secs", 0.0), "request_sha": plan.get("request_sha"),
               "summary_sha": plan.get("summary_sha"), "plan_order": plan.get("plan_order"), "block": plan.get("block"),
               "block_sha": plan.get("block_sha"), "exec_steps": ex_.get("steps"), "sightings": self._sightings() if status != "crash" else [],
               "prims": int(core.get("prims") or 0), "audit": a or None, "plan_not_executed_reason": ex_.get("plan_not_executed_reason")}
        if self.ft2:                                           # C4: ft-v2 rows only
            sk = (getattr(self.ex, "last", None) or {}).get("skills")
            row.update(treatment=T.FT2_VERSION, treatment_id=self.ident["sha"], exec_policy="briefing",
                       briefing_log=list(getattr(sk, "briefing_log", []) or []) if sk is not None else [],
                       prompt_summary_sha=plan.get("prompt_summary_sha"), observed_types=plan.get("observed_types"))
        # consolidation: executed, or a plan rejected AFTER the sweep (A11: a started attempt, admission False)
        consolidated = status in ("executed", "rejected") and wm is not None
        if consolidated:
            adm = False if status == "rejected" else res.admission
            cons = mem.end_episode(wm, admission=adm, extensions=[ll.LessonsExtension()])
            row["consolidation"] = {k: cons.get(k) for k in ("status", "n_ops", "rejected", "reasons", "stats") if k in cons} or str(cons)[:300]
            unit = ll.unit_key(wm)
            row["unit"] = unit
            if status == "executed":
                steps = plan.get("plan") or []
                row["fragment_reuse_rows"] = mem.fragment_ledger.observe(
                    unit, req.episode_id, [{"action": s[0], "object": s[1], "to": s[2]} for s in steps],
                    [bool(s["ok"]) for s in (ex_.get("steps") or [])], list(idx_before.values()))
        row["consolidated"] = bool(consolidated)
        row["n_procedures"] = len(mem.store.select("procedure"))
        row["state_hash_after"] = mem.state_hash()
        row["fragments_sha256"] = mem.fragment_ledger.save(os.path.join(self._wdir(o), "frag"))
        if self.fault:
            self.fault("after_consolidate", o, n)
        snaps = []
        done = rows_so_far + [{"item": item, "lang": lang}]
        if self.ft2:                                           # C4: the record exists BEFORE its snapshot, which stores the row prefix incl. it
            row["snapshot_after"] = [k for k in self.ks if k == n]
            rec = C.make_record("practice", **_jsonable(row))
            if C.key_leak(json.dumps(rec, ensure_ascii=False)):
                raise PracticeError("a key-like string reached a practice record; refusing to write it")
            for k in row["snapshot_after"]:
                self.take_snapshot(o, k, done, state_hash_expected=row["state_hash_after"], rows=list(self._prev) + [rec])
        else:
            for k in self.ks:
                if k == n:
                    self.take_snapshot(o, k, done, state_hash_expected=row["state_hash_after"])
                    snaps.append(k)
            row["snapshot_after"] = snaps
            rec = C.make_record("practice", **_jsonable(row))
        if C.key_leak(json.dumps(rec, ensure_ascii=False)):
            raise PracticeError("a key-like string reached a practice record; refusing to write it")
        if self.fault:
            self.fault("before_journal", o, n)
        self.journal.append(rec)
        if crash:
            self._crashes += 1
            if self._crashes > MAX_CONSECUTIVE_CRASHES:
                raise PracticeError(f"{self._crashes} consecutive crashed attempts: stopping (fail fast)")
        else:
            self._crashes = 0
        return rec

    # ---- one order
    def run_order(self, o):
        rows = [r for r in self.journal.rows() if r.get("namespace") == self.ns and r.get("order") == o]
        rows.sort(key=lambda r: r["attempt"])
        for i, r in enumerate(rows):                                   # the journal must be a prefix of this order
            want = self.orders_plan[o][i]
            if (r["attempt"], r["item"], r["lang"]) != (want["n"], want["item"], want["lang"]):
                raise PracticeError(f"o{o}: journal row {i + 1} is not the seeded order (the order changed since it was written)")
        if self.ft2:                                                   # C4 / Astra Q10: no resume across a treatment change
            bad = T.journal_problems(rows, self.ns, self.ident["sha"])
            if bad:
                raise PracticeError("ft-v2 practice resume refused: " + "; ".join(bad[:3]))
        self._prepare_order(o, rows)
        for k in self.ks:                                              # snapshots of attempts already journaled must be complete
            if 1 <= k <= len(rows):
                bad = verify_snapshot_dir(self.snap_dir(o, k), o, k)
                if bad:
                    raise PracticeError(f"o{o}k{k}: journaled but the snapshot is invalid: {bad}")
        if not rows and 0 in self.ks:
            self.take_snapshot(o, 0, [], state_hash_expected=self.mem.state_hash(), rows=[])
        elif 0 in self.ks and verify_snapshot_dir(self.snap_dir(o, 0), o, 0):
            raise PracticeError(f"o{o}k0: the empty snapshot is invalid; it cannot be rebuilt after attempt 1")
        started = 0
        for p in self.orders_plan[o][:self.max_k]:
            if p["n"] <= len(rows):
                continue
            if self.limit is not None and started >= self.limit:
                return {"order": o, "done": len(rows), "complete": False}
            ck = os.path.join(self._wdir(o), "ckpt")
            self.mem.snapshot(ck, overwrite=True)                      # pre-attempt checkpoint = state after the last journaled row
            self._prev = list(rows)                                    # ft-v2: the package's example rows (earlier attempts only)
            rec = self.attempt(o, p, [{"item": r["item"], "lang": r["lang"]} for r in rows])
            rows.append(rec)
            started += 1
            self.log(json.dumps({"o": o, "n": p["n"], "item": p["item"], "lang": p["lang"], "status": rec["status"], "goal_ok": rec["goal_ok"],
                                 "plan_ok": rec["plan_ok"], "block_ids": (rec.get("block") or {}).get("n_records"), "s": rec["wall_s"],
                                 "snap": rec["snapshot_after"]}, default=str))
            if self.counter is not None and hasattr(self.counter, "save"):
                self.counter.save()
        return {"order": o, "done": len(rows), "complete": len(rows) >= self.max_k}

    def run(self):
        out = [self.run_order(o) for o in self.orders]
        self._close_mem()
        if all(r["complete"] for r in out):
            rows = [r for r in self.journal.rows() if r.get("namespace") == self.ns and r.get("order") in self.orders]
            cost = cost_summary(rows)
            cp = os.path.join(os.path.dirname(self.journal.path), "practice_cost.json")
            with open(cp, "w", encoding="utf-8") as f:
                json.dump(cost, f, indent=1)
            sha = C.sha256_file(self.journal.path)
            if not any(e.get("event") == "phase_complete" and e.get("phase") == PHASE and e.get("journal_sha") == sha for e in self.events.rows()):
                self.events.append(C.make_record("events", event="phase_complete", phase=PHASE, n=len(rows), journal_sha=sha,
                                                 orders=self.orders, ks=self.ks, cost=cost["total"]))
        return out


# ================================================================================================ status (any python)
def status(namespace, orders, ks, root=None, snap_root=None, journal=None) -> dict:
    P = C.paths(root)
    snap_root = snap_root or P["snapshots"]
    rows = C.Journal(journal or C.journal_path(namespace, "practice", root)).rows()
    out = {"namespace": namespace, "orders": {}}
    for o in orders:
        rs = [r for r in rows if r.get("order") == o]
        bad = []
        for k in ks:
            if k <= len(rs) or k == 0:
                bad += [f"k{k}: {b}" for b in verify_snapshot_dir(os.path.join(snap_root, f"o{o}", f"k{k}"), o, k)]
        out["orders"][str(o)] = {"attempts": len(rs), "snapshot_problems": bad, "nesting_problems": verify_nesting(snap_root, o, [k for k in ks if k <= len(rs)])}
    out["ok"] = all(not v["snapshot_problems"] and not v["nesting_problems"] for v in out["orders"].values())
    return out


# ================================================================================================ the live build (WSL only)
def build_live(args):
    import honest_skills as hs
    import ltm as L
    import planners as PL
    import step5_eval as se
    import step5_live as sl
    from planners.cache import ResponseCache
    from planners.contract import MEMORY_HEADER, PlannerContract
    from planners.ollama_provider import OllamaPlanner
    sys.path.insert(0, os.path.join(C.SIM, "final_test", "c3"))
    from real_counter import RealMaxCounter
    ns = args.namespace
    man = C.load_manifest()
    allowed = frozenset(man[C.NAMESPACES[ns]["items"]])
    forbidden = frozenset(man["test"]) | frozenset(man.get("test_uncapped_not_scored") or [])
    se.assert_items_allowed(sorted(allowed), allowed, forbidden, "practice allow-list")
    ft2 = T.is_ft2(ns)
    ex = sl.LiveExecutor(memory="places", policy="console", hints=False, allow=allowed, forbid=forbidden, hint_goals=False,
                         plan_after_sweep=True, scene=C.ns_scene(ns), **({"ltm_chooser": False} if ft2 else {}))   # A11; C4: prior chooser in ft-v2
    ex._boot()
    types = hs.detector_types(ex._det)
    contract = PlannerContract(T.planner_types(types), tools=hs.TOOLS, version=T.FT2_VERSION) if ft2 else PlannerContract(types, tools=hs.TOOLS)
    work = C.ns_work(ns)
    os.makedirs(work, exist_ok=True)
    cache = ResponseCache(os.path.join(work, "cache"), namespace=ns)
    prov = OllamaPlanner(endpoint=args.ollama, strict_schema=True, cache=cache, ledger=C.open_ledger(ns))
    runner = PL.PlannerRunner(prov, contract)
    if ft2:
        from planners.contract import PACKAGE_HEADER
        counter = RealMaxCounter(os.path.join(work, "tokcache.json"), PACKAGE_HEADER, ollama_endpoint=args.ollama,
                                 allow_missing_anthropic=args.gemma_only_counter)
        hook = PackagePracticeHook(runner, contract, T.Packager(ns, counter))
        hook.executor = ex
        hints = T.hints_fn(T.briefing_hint_table(allow_unapproved=T.allow_draft_briefing(ns)))   # registered practice: approved only
        ex.plan_hook = hook
        return ex, hook, counter, L.load_params(), hints
    counter = RealMaxCounter(os.path.join(work, "tokcache.json"), MEMORY_HEADER, ollama_endpoint=args.ollama,
                             allow_missing_anthropic=args.gemma_only_counter)
    hook = PracticeHook(runner, contract, counter, args.channels)
    ex.plan_hook = hook
    return ex, hook, counter, L.load_params(), None


def parse_list(s, cast=int):
    return [cast(x) for x in str(s).split(",") if str(x).strip() != ""]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--orders", default=None, help="default 0,1,2 (ft-v1); 0 in an ft-v2 namespace (A27)")
    ap.add_argument("--k", default="0,1,3,5,10")
    ap.add_argument("--namespace", default="practice-v1")
    ap.add_argument("--channels", default=None, help="active memory channels of the prompt block (places-only unless C2-h passed); ft-v1 only")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--kitchen", default=None, help="must equal the namespace's scene (common.ns_scene); default = that scene")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many NEW attempts (smoke / time-box)")
    ap.add_argument("--allow-test", action="store_true", help="never needed for practice; passed to the guard unchanged")
    ap.add_argument("--steal-lock", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the seeded orders and run the guard; no simulator")
    ap.add_argument("--status", action="store_true", help="verify the journal and every snapshot; no simulator")
    ap.add_argument("--gemma-only-counter", action="store_true")
    a = ap.parse_args(argv)
    ft2 = a.namespace in C.NAMESPACES and T.is_ft2(a.namespace)
    if ft2 and a.channels is not None:
        raise SystemExit("--channels is refused in an ft-v2 namespace: practice renders the package (treatment.FT2)")
    a.channels = () if ft2 else tuple(parse_list(a.channels if a.channels is not None else "places", str))
    orders = parse_list(a.orders if a.orders is not None else ("0" if ft2 else "0,1,2"))
    ks = parse_list(a.k)
    # K23 agent-side (Astra K23-AGENT-r1 1): the kitchen IS the namespace's scene; a conflicting --kitchen is refused
    if a.kitchen is not None and a.kitchen != C.ns_scene(a.namespace):
        raise SystemExit(f"--kitchen {a.kitchen} conflicts with namespace {a.namespace}'s scene {C.ns_scene(a.namespace)}")
    a.kitchen = C.ns_scene(a.namespace)
    if a.status:
        st = status(a.namespace, orders, ks)
        print(json.dumps(st, indent=1))
        return 0 if st["ok"] else 1
    if a.dry_run:
        pr = Practice(namespace=a.namespace, orders=orders, ks=ks, executor=None, channels=a.channels, kitchen=a.kitchen, allow_test=a.allow_test)
        for o in orders:
            print(f"order {o}: " + ", ".join(f"{p['n']}:{p['item']}[{p['lang']}]" for p in pr.orders_plan[o][:max(ks)]))
        out = {"guard": "ok", "pool": len(pr.pool), "orders": orders, "ks": ks, "channels": a.channels}
        if pr.reserved is not None:
            out["reserved_acceptance"] = {k: pr.reserved[k] for k in ("n", "sha", "pool_before", "pool_after", "in_pool")}
        print(json.dumps(out))
        return 0
    with C.PhaseLock(a.namespace, PHASE, steal=a.steal_lock):
        ex, hook, counter, _params, hints = build_live(a)
        try:
            pr = Practice(namespace=a.namespace, orders=orders, ks=ks, executor=ex, hook=hook, counter=counter, channels=a.channels,
                          kitchen=a.kitchen, limit=a.limit, allow_test=a.allow_test, hints=hints)
            res = pr.run()
            print(json.dumps(res))
        finally:
            try:
                counter.save()
            finally:
                ex.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
