"""M4 - Phase A observation capture (SPEC s12 M4; prereg s7 Phase A, A11, A12, A14). WSL, simulator, NO LLM.

For every presentation (item x language x world) it restores the world, applies the item setup, runs the FIXED start sweep, and only
then (A11) renders what the planners will be shown. It records, per presentation:
  * the post-sweep observation summary (+ sha), the told holding, the strict world hash (diagnostic only);
  * the memory / experience block of EVERY prompt source, rendered from frozen snapshots + the post-sweep working memory, and the
    sha256 of each full request (`PlannerContract` is the only renderer): sources `none`, `S*`, `FS`, and (REPEAT presentations,
    canonical world) the learning-curve snapshots; the relocated world has `none` and `S*` only;
  * the post-sweep world STATE for gate 12.4 (`states/<pres>.c<run>.json`, scoring side: it carries the evaluator labels and so
    is never read by a planner process, never copied into `capture.jsonl`).
Nothing from gold (goal, setup, reference plan) and nothing outcome-like enters `capture.jsonl` (the `C.make_record` firewall).

How it runs: `step5_live.LiveExecutor(memory="none", plan_after_sweep=True, state_capture=True)` with a capture hook that returns
`steps=None`. The executor then closes the swept episode unexecuted (no step, no scoring call). The executor never receives a
memory: frozen snapshots are opened HERE, only to render text (G-iso: the `none` arm is rendered with no store open for it).

Gates (all fail fast, none is decided by an outcome):
  masking      every request: `C.mask_problems` (aborts on a hit; nothing is written for that presentation);
  G-iso        the `none` request has no block header, every other request equals `none` plus exactly its block, the system
               message is identical; the executor ran with no memory and no chooser;
  leakage      no block holds a normalised test instruction (EN / FA) or a test item id; snapshot manifests that carry a
               leakage verdict must say ok;
  snapshot     the sha of every snapshot directory is identical before and after the run (and equals freeze.json when frozen);
  A12          S* coverage per presentation (`ltm.a12_coverage`, token counts only) summarised per level;
  key leak     `C.key_leak` over every file this phase wrote;
  reset gate   with `--runs 1,2` (dev): the 24 pairs of post-sweep states go to `reset_pairs.json` (input of
               `reset_gate.py validate --pairs`, which records the ONE-SHOT validation) and, if the tolerances are calibrated, a
               non-recording preview verdict is logged.

    # WSL, repo root, simulator env (see the ledger); one simulator at a time
    python sim/final_test/matrix/capture.py --namespace rehearsal-v1 --items dev --runs 1,2 --worlds canon,reloc --ollama http://HOST:11434
    python sim/final_test/matrix/capture.py --namespace test-v1 --items test --runs 1 --worlds canon,reloc --allow-test   # W5 only

Exit codes: 0 phase complete; 2 a gate failed (no `phase_complete`); 4 aborted on an integrity violation; 5 interrupted
(`--inject-interrupt N`, the rehearsal resume test). Python 3.10 syntax only (WSL runs it).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import time
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                    # noqa: E402  (also puts sim/ on sys.path)
import treatment as T                                 # noqa: E402  (C4: the ft-v2 switch; inert in every ft-v1 namespace)
import obs_health as OH                               # noqa: E402  (A34-A2 2(b): observation-pipeline health evidence)

LEVELS = ("L1", "L2", "L3")
SCENE = "FloorPlan1"
#: Learning-curve snapshot sources (prereg 6.3 E4; same list as final_test_callplan.CURVE_SNAPSHOTS). (0, 10) is S*, never re-listed.
#: A27 (draft, PENDING ARASH SIGN-OFF AT THE FREEZE): order 0 only, K = 1, 3, 5 (was 11 sources over orders 0-2); derived from
#: common.CURVE_SNAPSHOTS so capture, plan and the call plan cannot drift apart.
CURVE_SOURCES = ["o%dk%d" % ok for ok in C.CURVE_SNAPSHOTS]
#: Rehearsal only. In test-v1 the relocation seed comes from freeze.json (`reloc_seed`) and nothing else.
REHEARSAL_RELOC_SEED = 20261011
NS_FOR_ITEMS = {"dev": "rehearsal-v1", "test": "test-v1"}
DEV_NAMESPACES = ("rehearsal-v1", "rehearsal-v2", "rehearsal-v3")       # rehearsal-v2 = the plan-first safety test; rehearsal-v3 = the ft-v2 runner rehearsal (C4)


class CaptureAbort(RuntimeError):
    """An integrity violation: the phase stops, nothing is written for the presentation being captured."""


class Interrupted(RuntimeError):
    """--inject-interrupt N: stop after N new cells (the rehearsal resume test)."""


# ================================================================================================ pure helpers
def state_filename(pres: str, run: int) -> str:
    """`states/<pres>.c<run>.json`. The presentation id carries ':' which is not a legal Windows file name character (and the work
    tree is on DrvFs), so ':' becomes '__'. Phase C (M6) and the tests use THIS function."""
    return "%s.c%d.json" % (pres.replace(":", "__"), int(run))


def restored_state(world: str, reloc_seed, setup) -> str:
    """The restore the world id + setup produce (SPEC s4 `restored_state`): 'canon' or 'reloc:<seed>', then '+' sha256(setup)[:16]."""
    base = "canon" if world == "canon" else "reloc:%s" % reloc_seed
    return base + "+" + C.sha256_json(setup or [])[:16]


def dir_sha(path: str) -> str:
    """sha256 over every file in a snapshot directory (relative posix path + content sha, sorted). Lock / journal side files of
    sqlite are not part of a snapshot."""
    sha = C.dir_sha(path)                          # the ONE shared function (common.py), also used by execute / score / freeze checks
    if sha == "absent":
        raise CaptureAbort("snapshot directory %s is empty or missing" % path)
    return sha


snapshot_sha = dir_sha


_PUNCT = re.compile(r"[\s\.,;:!?،؛؟\"'()\[\]{}\-]+")


def norm_text(s: str) -> str:
    """Normalise an instruction for the leakage scan: NFKC, Persian/Arabic letter variants, no diacritics / ZWNJ / punctuation."""
    s = unicodedata.normalize("NFKC", str(s or "")).lower()
    s = s.replace("ي", "ی").replace("ك", "ک").replace("‌", " ").replace("‍", "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch) and not ("ً" <= ch <= "ْ"))
    return _PUNCT.sub(" ", s).strip()


#: A31 (DRAFT, approved by Arash 2026-10-13, sign at the freeze; Astra A31-design Q2): the phrase rule works on WHOLE delimiter-defined
#: normalised SEGMENTS. Delimiters (block text and forbidden text alike): CRLF / CR / LF, . ! ? ; : | the straight double quote, the
#: curly double quotes U+201C / U+201D, the guillemets U+00AB / U+00BB, the Arabic semicolon U+061B, the Arabic question mark U+061F,
#: and the arrow "->". Commas (Latin and Arabic) are NOT delimiters. These are operational boundaries, not all linguistic sentence ends.
SEG_DELIMS = re.compile("\r\n|->|[\r\n.!?;:|\"“”«»؛؟]")
#: an exempted span is a non-matching barrier: it is never equal to a forbidden segment and it separates its neighbours
BARRIER = None
G1_PREFIX = "The request in English: "                 # the package's G1 format string ltm_package: "The request in English: %s."


def segments(text: str, spans=()) -> list:
    """A31: the normalised non-empty segments of `text`, with every (start, end) span in `spans` replaced by ONE BARRIER element (the
    span's characters never reach a segment and the text on its two sides never becomes one segment or two consecutive ones)."""
    text = str(text or "")
    out, pos = [], 0
    for a, b in sorted(spans):
        if a < pos:
            raise ValueError("overlapping exempt spans")
        out += [s for s in (norm_text(p) for p in SEG_DELIMS.split(text[pos:a])) if s]
        out.append(BARRIER)
        pos = b
    out += [s for s in (norm_text(p) for p in SEG_DELIMS.split(text[pos:])) if s]
    return out


def _seg_hit(block: list, want: tuple) -> bool:
    n = len(want)
    if not n:
        return False
    return any(tuple(block[i:i + n]) == want for i in range(len(block) - n + 1))


def forbidden_texts(manifest: dict, items: dict, reserved: dict | None = None) -> dict:
    """{"id:<id>": id} for every test / test-uncapped id (+ the reserved acceptance ids, A31 D4), {("seg", segments): id} for their EN /
    FA texts (the A31 matching key), and {normalised text: id} (kept for the audit; no longer a matching key since A31)."""
    out = {}

    def add(iid, raw):
        t = norm_text(raw)
        if t:
            out[t] = iid
        sg = tuple(segments(raw))
        if sg:
            out[("seg", sg)] = iid
    for iid in list(manifest.get("test") or []) + list(manifest.get("test_uncapped_not_scored") or []):
        out["id:" + iid] = iid
        it = items.get(iid)
        if it:
            for lg in ("en", "fa"):
                if it.get(lg):
                    add(iid, it[lg])
    if reserved:
        for iid in reserved.get("ids") or []:
            out["id:" + iid] = iid
        for iid, _lg, _norm, raw in reserved.get("texts") or []:
            add(iid, raw)
    return out


def leakage_problems(text: str, forbidden: dict, spans=()) -> list:
    """Test / acceptance ids or instructions found inside a block text (a block is memory or experience, never a test instruction).
    A31: ids = WHOLE ids (treatment.id_in) on the UNMASKED text, no exemption; instructions = a forbidden text's whole segment sequence
    as a contiguous run of whole block segments, outside the exempt `spans` (validated by exempt_spans)."""
    if not text:
        return []
    seg = None
    hits = []
    for k, iid in forbidden.items():
        if isinstance(k, str) and k.startswith("id:"):
            if T.id_in(iid, text):
                hits.append(iid)
        elif isinstance(k, tuple) and k and k[0] == "seg":
            if seg is None:
                seg = segments(text, spans)
            if _seg_hit(seg, k[1]):
                hits.append(iid)
    return sorted(set(hits))


def leakage_problems_legacy(text: str, forbidden: dict) -> list:
    """The pre-A31 scan, kept byte-for-byte in behaviour for the ft-v1 namespaces (rehearsal-v1 / -v2): raw lower-cased id substring,
    normalised test instruction (>= 8 chars) as a substring of the normalised block."""
    if not text:
        return []
    low, n = text.lower(), norm_text(text)
    hits = []
    for k, iid in forbidden.items():
        if not isinstance(k, str):
            continue
        if k.startswith("id:"):
            if iid.lower() in low:
                hits.append(iid)
        elif len(k) >= 8 and k in n:
            hits.append(iid)
    return sorted(set(hits))


_EX_ID = re.compile(r"^example/[^/]+/(?P<ns>[^:/]+):o(?P<o>\d+):a(?P<a>\d+)$")


def exempt_spans(text: str, *, g1: str | None, examples=(), rows=(), items=None, refused=(), pools=None, form: str = "package") -> tuple:
    """A31 D3 -> (spans, audit). Raises CaptureAbort on any invalid exemption (fail closed).
    g1: the G1 line recomputed from the presentation's OWN instruction / language / told holding by the package's own code path
        ("The request in English: <R>." or None). The block may carry the G1 prefix at most once, and then exactly `g1`; only <R> is exempt.
    examples: the example record ids the block's own render log says it shows (package `kept`, FS `retained`); each is mapped through the
        SAME source's practice rows; valid only if the row's item is no test / test-uncapped / reserved id, is in its namespace's pinned
        pool, the row's instruction equals the pinned dataset text byte for byte, and its full rendered wrapper occurs exactly once; only
        the instruction's character range is exempt. form "package": '"<instr>" -> ' (en) / '«<instr>» -> ' (fa); form "fs":
        '<attempt>. <lang> | <instr> | '."""
    text = str(text or "")
    spans, audit = [], []
    n_pre = text.count(G1_PREFIX.strip())
    if n_pre > 1:
        raise CaptureAbort("A31: the block carries %d G1 request lines (at most 1)" % n_pre)
    if n_pre == 1:
        if not g1:
            raise CaptureAbort("A31: the block carries a G1 request line but none is recomputed from the presentation's instruction")
        at = text.find(g1)
        if at < 0 or text.find(G1_PREFIX) != at:
            raise CaptureAbort("A31: the block's G1 request line differs from the line recomputed from the presentation's instruction")
        a, b = at + len(G1_PREFIX), at + len(g1) - 1                     # <R> only (prefix and the final '.' stay)
        spans.append((a, b))
        audit.append({"kind": "g1", "start": a, "end": b, "sha": C.sha256_text(text[a:b])})
    by = {}
    for r in rows or ():
        by[(r.get("namespace"), int(r.get("order", -1)), int(r.get("attempt", -1)))] = r
    exs = [e for e in (examples or ()) if str(e).startswith("example/")]
    for eid in exs:
        m = _EX_ID.match(str(eid))
        if not m:
            raise CaptureAbort("A31: example id %s has no practice-row form" % eid)
        r = by.get((m.group("ns"), int(m.group("o")), int(m.group("a"))))
        if r is None:
            raise CaptureAbort("A31: example %s is not a row of this source's practice rows" % eid)
        iid, lg, ins = r.get("item"), r.get("lang"), r.get("instruction")
        if iid in set(refused):
            raise CaptureAbort("A31: example %s is a test / reserved item (%s)" % (eid, iid))
        pool = (pools or {}).get(r.get("namespace"))
        if pool is None or iid not in pool:
            raise CaptureAbort("A31: example %s item %s is not in the pinned pool of %s" % (eid, iid, r.get("namespace")))
        it = (items or {}).get(iid) or {}
        if not ins or lg not in ("en", "fa") or it.get(lg) != ins:
            raise CaptureAbort("A31: example %s instruction differs from the pinned dataset text of %s/%s" % (eid, iid, lg))
        if form == "fs":
            pre, post = "%d. %s | " % (int(m.group("a")), lg), " | "
        elif lg == "en":
            pre, post = '"', '" -> '
        else:
            pre, post = "«", "» -> "
        wrap = pre + ins + post
        if text.count(wrap) != 1:
            raise CaptureAbort("A31: the rendered field of example %s occurs %d times (expected exactly 1)" % (eid, text.count(wrap)))
        a = text.find(wrap) + len(pre)
        spans.append((a, a + len(ins)))
        audit.append({"kind": "example", "id": eid, "item": iid, "lang": lg, "start": a, "end": a + len(ins), "sha": C.sha256_text(ins)})
    if form == "package":
        n_lines = text.count("Example from my practice") + text.count("نمونه‌ای از تمرین من")
        if n_lines != len(exs):
            raise CaptureAbort("A31: the block shows %d practice examples but its render log lists %d" % (n_lines, len(exs)))
    elif form == "fs":
        n_lines = len([ln for ln in text.split("\n") if ln.strip()])
        if n_lines != len(exs):
            raise CaptureAbort("A31: the FS block has %d lines but its render log lists %d examples" % (n_lines, len(exs)))
    spans.sort()
    for (a1, b1), (a2, _b2) in zip(spans, spans[1:]):
        if a2 < b1:
            raise CaptureAbort("A31: exempt spans overlap")
    return spans, audit


def g_iso_problems(msgs_by_src: dict, blocks: dict, memory_header: str | None = None) -> list:
    """G-iso on the rendered requests: `none` has no block header; every other source is `none` plus exactly its own block.
    memory_header: the header of the memory slot (ft-v1: MEMORY_HEADER, the default; ft-v2: PACKAGE_HEADER)."""
    from planners.contract import EXPERIENCE_HEADER, MEMORY_HEADER, PACKAGE_HEADER
    mem_hdr = memory_header or MEMORY_HEADER
    probs = []
    base = msgs_by_src.get("none")
    if base is None:
        return ["no `none` request rendered"]
    bu = base[1]["content"]
    if MEMORY_HEADER in bu or EXPERIENCE_HEADER in bu or (memory_header is not None and PACKAGE_HEADER in bu):
        probs.append("the none request carries a block header")
    for src, msgs in sorted(msgs_by_src.items()):
        if src == "none":
            continue
        if msgs[0] != base[0]:
            probs.append("%s: system message differs from none" % src)
        b = blocks[src]
        if not b.get("text"):
            if msgs != base:
                probs.append("%s: empty block but the request differs from none" % src)
            continue
        header = EXPERIENCE_HEADER if b.get("slot") == "fewshot" else mem_hdr
        extra = "\n\n" + header + "\n" + b["text"]
        u = msgs[1]["content"]
        if u.count(extra) != 1 or u.replace(extra, "", 1) != bu:
            probs.append("%s: request is not none + its block" % src)
    return probs


def sources_for(world: str, is_repeat: bool, *, fs: bool = True, curve: bool = True, only=None) -> list:
    """The prompt sources of one presentation. Canonical world: none, S*, FS and (REPEAT items) the curve snapshots. Relocated world:
    none and S* only (the probe arms G0, G+M, O0, O+M)."""
    out = ["none", "S*"]
    if world == "canon":
        if fs:
            out.append("FS")
        if is_repeat and curve:
            out += CURVE_SOURCES
    if only:
        out = [s for s in out if s in set(only)]
        if "none" not in out:
            out.insert(0, "none")
    return out


def build_presentations(item_ids, worlds, reloc_items, langs=("en", "fa")) -> list:
    """[(pres, item, lang, world)] - canonical for all items, relocated for the relocation items only, unseeded (the caller orders)."""
    out = []
    for w in worlds:
        for iid in item_ids:
            if w == "reloc" and iid not in reloc_items:
                continue
            for lg in langs:
                out.append((C.pres_id(iid, lg, w), iid, lg, w))
    return out


def file_has_key(path: str) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            return C.key_leak(f.read())
    except OSError:
        return False


def aggregate_coverage(rows: list) -> dict:
    """Per level summary of the S* A12 coverage verdicts stored in capture records (outcome-blind: token counts only)."""
    by, fails = {}, []
    for r in rows:
        b = (r.get("blocks") or {}).get("S*")
        if not b:
            continue
        cv = b.get("coverage")
        d = by.setdefault(r["level"], {"n": 0, "ok": 0, "exempt": 0, "empty": 0, "unchecked": 0})
        d["n"] += 1
        d["empty"] += 1 if b.get("empty") else 0
        if cv is None:
            d["unchecked"] += 1
            continue
        d["ok"] += 1 if cv.get("ok") else 0
        d["exempt"] += 1 if cv.get("exempt") else 0
        if not cv.get("ok"):
            fails.append({"pres": r["pres"], "level": r["level"], "held": cv.get("held"), "N": cv.get("N"),
                          "complete_fits": cv.get("complete_fits"), "ids_identical": cv.get("ids_identical")})
    n = sum(d["n"] for d in by.values())
    return {"n": n, "by_level": by, "failures": fails, "unchecked": sum(d["unchecked"] for d in by.values()),
            "ok": n > 0 and not fails}


def a12_gate(cov: dict, required: bool) -> bool:
    """A12 says the coverage check runs on EVERY S* prompt before execution. Required (test-v1): no failure AND nothing unchecked (a
    params.json without retrieval.a12 leaves every presentation unchecked: that must not pass vacuously). Not required (dev): no
    failure, or nothing was checked at all."""
    if required:
        return bool(cov["ok"]) and cov["unchecked"] == 0
    return bool(cov["ok"]) or cov["unchecked"] == cov["n"]


# ================================================================================================ relocation world
@contextlib.contextmanager
def relocated_reset(seed):
    """Relocated world (prereg E6): the executor's `eval._reset_world(ag, scene)` is wrapped so that, right after the clean reset and
    BEFORE the item setup, `InitialRandomSpawn` is applied with the one fixed seed. Phase C (M6) must use this same context manager
    so Phase A and Phase C restore the same relocated world. Patches only the module attribute the executor looks up at call time."""
    import eval as ev                                  # sim/eval.py (never run from sim/: trace.py shadows the stdlib)
    orig = ev._reset_world

    def _reloc(ag, scene):
        orig(ag, scene)
        r = ag.random_spawn(int(seed))
        ok = getattr(r, "metadata", {}).get("lastActionSuccess", True) if r is not None else True
        if not ok:
            raise CaptureAbort("InitialRandomSpawn(seed=%s) failed: %s" % (seed, str(getattr(r, "metadata", {}).get("errorMessage"))[:160]))

    ev._reset_world = _reloc
    try:
        yield
    finally:
        ev._reset_world = orig


@contextlib.contextmanager
def canonical_reset():
    yield


# ================================================================================================ the capture hook
class CaptureHook:
    """The plan hook of Phase A: it never plans. It is called by the executor right after the fixed sweep (A11) with `post`, renders
    everything the planners will see, and returns `steps=None` so the executor closes the swept episode without executing it."""

    def __init__(self, cap):
        self.cap = cap
        self.pending = None

    def __call__(self, req, it, holding, ltm, post=None):
        if post is None:
            raise CaptureAbort("capture needs the A11 order (planner after the sweep): the executor passed no post-sweep dict")
        if ltm is not None:
            raise CaptureAbort("capture runs with memory='none': the executor was handed a memory")
        self.pending = self.cap.render_presentation(req, it, holding, post)
        return {"steps": None, "info": {"not_executed": "reuse", "capture": True}}


# ================================================================================================ the capturer
class Capturer:
    """All the logic of Phase A, with every outside dependency injected (the selftest drives it with fakes):
    executor          an object like step5_live.LiveExecutor (`plan_hook`, `execute(req)`, `frame`, `map_rev`, `last`);
    contract          planners.contract.PlannerContract;
    counter           RealMaxCounter-like (`counts`, `header`, `tokenizers`);
    open_sources      callable(names) -> {name: frozen LongTermMemory-like} (read-only copies; opened ONCE, test epoch per use);
    items             {id: item dict} from general.json (setup and text; no gold is ever written);
    fs_doc            the parsed fs_blocks.json, or None."""

    def __init__(self, *, namespace, executor, contract, counter, open_sources, items, manifest, fs_doc=None, channels=("places",),
                 allow_test=False, root=None, worlds=("canon",), runs=(1,), reloc_seed=None, reloc_all=False, only_sources=None,
                 use_curve=True, use_fs=True, limit=0, interrupt=0, reset_patch=None, snapshot_sha_fn=None, scene=SCENE,
                 log=print, params=None, freeze=None, tol_loader=None, packager=None, fs_renderer=None, reserved=None):
        self.ns, self.ex, self.contract, self.counter = namespace, executor, contract, counter
        # C4: an ft-v2 namespace renders the +M blocks as the PACKAGE (treatment.Packager) from the ft-v2 snapshots; ft-v1 is unchanged
        self.ft2 = T.is_ft2(namespace)
        self.packager = packager
        # C4-open: the ft-v2 FS block (fs-v2) comes from the SAME packager's stores and counter (A24 item 4)
        self.fs_renderer = fs_renderer or (T.FSRenderer(packager, test_ids=set((manifest or {}).get("test") or [])
                                                        | set((manifest or {}).get("test_uncapped_not_scored") or []))
                                           if self.ft2 and packager is not None else None)
        self.tcfg = T.config(namespace) if self.ft2 else None
        self.ident = None
        self._shas0 = {}
        self.open_sources, self.items, self.manifest, self.fs_doc = open_sources, items, manifest, fs_doc
        self.channels, self.allow_test, self.root = tuple(channels), bool(allow_test), root
        self.worlds, self.runs = tuple(worlds), tuple(runs)
        self.reloc_seed, self.reloc_all = reloc_seed, bool(reloc_all)
        self.only_sources, self.use_curve, self.use_fs = only_sources, use_curve, use_fs
        self.limit, self.interrupt = int(limit or 0), int(interrupt or 0)
        self.reset_patch = reset_patch or relocated_reset
        self.snapshot_sha_fn = snapshot_sha_fn or dir_sha
        self.scene, self.log = scene, log
        self.params = params or {}
        self.freeze = freeze or {}
        self.tol_loader = tol_loader
        self.hook = CaptureHook(self)
        self.ltms = {}
        self.cur = None
        self.reserved = None
        if self.ft2:                                       # A31 D4: the 29 reserved acceptance ids + texts (pinned sources, real root)
            import practice_k as PK
            try:
                self.reserved = reserved if reserved is not None else PK.reserved_acceptance(None, load_items(None))
            except PK.ReservedItemRefused as e:
                raise CaptureAbort("A31: reserved acceptance set: %s" % e)
        self.forbidden = forbidden_texts(manifest, items, self.reserved)
        self.repeat = set(C.repeat_items(namespace, manifest))                 # test-v1: the split's REPEAT set; rehearsal: the declared dev subset
        self.reloc_items = set(C.reloc_items(namespace, manifest))
        self.docs = C.ns_docs(namespace, root)
        self.jr = C.Journal(C.journal_path(namespace, "capture", root))
        self.ev = C.Journal(C.journal_path(namespace, "events", root))
        self.new_cells = 0
        self.dataset_sha = None
        self.stats = {"masking_checked": 0, "giso_checked": 0, "leak_checked": 0}

    # -- bookkeeping ----------------------------------------------------------------------------------------------
    def event(self, name, **kw):
        self.ev.append(C.make_record("events", event=name, phase="capture", **kw))

    def item_ids(self) -> list:
        return sorted(C.namespace_items(self.ns, self.manifest, self.root))            # K23: split group / roster / A29 roster

    def reloc_set(self) -> set:
        if "reloc" not in self.worlds:
            return set()
        base = set(self.item_ids())
        if self.reloc_all:
            if self.ns == "test-v1":
                raise CaptureAbort("--reloc-all is for rehearsal only")
            return base
        return base & self.reloc_items

    def presentations(self) -> list:
        ids = self.item_ids()
        rs = self.reloc_set()
        pl = build_presentations(ids, self.worlds, rs)
        order = C.seeded_order([p[0] for p in pl], "capture", self.ns)
        by = {p[0]: p for p in pl}
        return [by[k] for k in order]

    def needed_sources(self) -> list:
        need = set()
        for pres, iid, lg, w in self.presentations():
            for s in sources_for(w, iid in self.repeat, fs=self._fs_on(), curve=self.use_curve,
                                 only=self.only_sources):
                if s not in ("none", "FS"):
                    need.add(s)
                elif s == "FS" and self.ft2:             # C4-open: ft-v2 FS reads the S* example store (opened, hashed, checked)
                    need.add("S*")
        return sorted(need)

    def _fs_on(self) -> bool:
        """ft-v1: FS needs fs_blocks.json (fs-v1, per level). ft-v2: FS is rendered per presentation (fs-v2) and never reads fs_doc."""
        if self.ft2:
            return bool(self.use_fs)
        return bool(self.use_fs and self.fs_doc is not None)

    # -- one presentation: what the planners will see ------------------------------------------------------------
    def _fresh_wm(self, src_ltm, post):
        wm = src_ltm.begin_episode(post["episode_id"], "test")
        for e in post["events"]:
            wm.ingest(dict(e))
        return wm

    def _count(self, text, header):
        if not text:
            return 0, {}
        cts = {k: int(v) for k, v in self.counter.counts(header + "\n" + text).items()}
        return max(cts.values()), cts

    def render_presentation(self, req, it, holding, post) -> dict:
        if self.ft2:
            return self._render_ft2(req, it, holding, post)
        import step5_live as sl
        from planners.contract import EXPERIENCE_HEADER, MEMORY_HEADER, prompt_sha256
        cur = self.cur
        level, lang, world = cur["level"], cur["lang"], cur["world"]
        summary = list(post["summary"])
        if C.sha256_text("\n".join(summary)) != post["summary_sha"]:
            raise CaptureAbort("summary sha differs from the executor's post-sweep sha")
        instr = it[lang]
        srcs = sources_for(world, it["id"] in self.repeat, fs=self.use_fs and self.fs_doc is not None, curve=self.use_curve,
                           only=self.only_sources)
        blocks = {"none": {"slot": None, "text": "", "ids": [], "real_tokens": 0, "counts": {}, "budget": None, "empty": True,
                           "coverage": None}}
        msgs = {"none": self.contract.messages(instr, summary, holding)}
        snap = [s for s in srcs if s not in ("none", "FS")]
        for s in snap:                                                  # memory sources: frozen snapshots + the post-sweep WM
            self.ltms[s].begin_test_epoch(frame=self.ex.frame, map_rev=self.ex.map_rev)
        rendered = sl.render_blocks(post, {s: self.ltms[s] for s in snap}, level, lang, self.channels, holding) if snap else {}
        for s in snap:
            b = rendered[s]
            text = b["text"] or ""
            n, cts = self._count(text, MEMORY_HEADER)
            budget = self.ltms[s].retriever().budgets[level]
            cov = None
            if s == "S*":
                cov = self._coverage(post, level, lang, b, holding, sl)
            blocks[s] = {"slot": "memory", "text": text, "ids": [r["id"] for r in b["records"]], "real_tokens": n, "counts": cts,
                         "budget": int(budget), "empty": not text, "coverage": cov, "sha256": C.sha256_text(text) if text else None}
            msgs[s] = self.contract.messages(instr, summary, holding, memory_block=text or None)
        if "FS" in srcs:
            fb = (self.fs_doc or {}).get(level) or {}
            text = fb.get("text") or ""
            if not text:
                raise CaptureAbort("fs_blocks.json has no block for %s" % level)
            blocks["FS"] = {"slot": "fewshot", "text": text, "ids": [], "real_tokens": int(fb.get("real_tokens") or 0),
                            "counts": dict(fb.get("counts") or {}), "budget": fb.get("budget"), "empty": False, "coverage": None,
                            "sha256": C.sha256_text(text)}
            msgs["FS"] = self.contract.messages(instr, summary, holding, fewshot_block=text)
        # ---- gates on the rendered text -------------------------------------------------------------------------
        requests = {}
        for s in srcs:
            joined = "\n".join(m["content"] for m in msgs[s])
            hit = C.mask_problems(joined)
            self.stats["masking_checked"] += 1
            if hit:
                raise CaptureAbort("masking hit in %s / %s: %s" % (cur["pres"], s, hit[:5]))
            if C.key_leak(joined):
                raise CaptureAbort("key-like text in the %s request of %s" % (s, cur["pres"]))
            if s != "none":
                self.stats["leak_checked"] += 1
                lk = leakage_problems_legacy(blocks[s]["text"], self.forbidden)        # ft-v1: the pre-A31 scan, unchanged
                if lk:
                    raise CaptureAbort("leakage: the %s block of %s holds test item text / ids %s" % (s, cur["pres"], lk[:3]))
            requests[s] = prompt_sha256(msgs[s])
        gi = g_iso_problems(msgs, blocks)
        self.stats["giso_checked"] += 1
        if gi:
            raise CaptureAbort("G-iso: %s: %s" % (cur["pres"], "; ".join(gi[:3])))
        return {"summary": summary, "summary_sha": post["summary_sha"], "holding": holding, "blocks": blocks, "requests": requests,
                "instruction": instr, "episode_id": post["episode_id"], "sweep_prims": post.get("sweep_prims"),
                "system_sha": C.sha256_text(msgs["none"][0]["content"]), "sources": srcs, "sha_order": sorted(requests)}

    def _g1_recompute(self, instr, lang, holding, psum):
        """A31 D3(a): the G1 line of THIS presentation, recomputed with the package's own query builder (the pinned profile), parser and
        renderer, from the presentation's pinned instruction, language, told holding and prompt summary. None when G1 renders nothing."""
        import ltm_package as LP
        q = LP.make_query_profile(self.packager.profile if self.packager is not None else self.tcfg["package_profile"], instr, lang,
                                  holding, list(psum or []))
        r = LP.render_request(LP._parse_request(q))
        return (G1_PREFIX + "%s." % r) if r else None

    def _pools(self) -> dict:
        if getattr(self, "_pool_cache", None) is None:
            res = set((self.reserved or {}).get("ids") or [])
            self._pool_cache = {ns: set(self.manifest.get(C.NAMESPACES[ns]["items"]) or []) - res
                                for ns in C.NAMESPACES if C.NAMESPACES[ns].get("items") and ns.startswith("practice")
                                and C.ns_scene(ns) == C.ns_scene(self.ns)}         # K23: the namespace's own kitchen only
        return self._pool_cache

    def _exempt(self, s, block, src_rows, instr, lang, holding, psum):
        """A31 D3: the validated exempt spans of one rendered block + their audit (fail closed: CaptureAbort)."""
        text = block.get("text") or ""
        if not text:
            return [], []
        refused = {v for k, v in self.forbidden.items() if isinstance(k, str) and k.startswith("id:")}
        form = "fs" if s == "FS" else "package"
        g1 = self._g1_recompute(instr, lang, holding, psum)
        spans, audit = exempt_spans(text, g1=g1, examples=list(block.get("ids") or []), rows=src_rows.get(s) or [], items=self.items,
                                    refused=refused, pools=self._pools(), form=form)
        for a in audit:
            a.update(source=s, block_sha=C.sha256_text(text), rows_sha=T.rows_sha(src_rows.get(s) or []), rule="A31-v1")
        return spans, audit

    def _render_ft2(self, req, it, holding, post) -> dict:
        """C4 / ft-v2: the prompt summary is the agent's FULL post-sweep list through A24's rule (treatment.prompt_summary); every +M
        source is the package rendered from that source's ft-v2 snapshot db + its own practice-row prefix; G-iso uses PACKAGE_HEADER.
        The executor's 40-line summary sha stays the integrity / unit-key sha; the prompt summary has its own sha."""
        from planners.contract import PACKAGE_HEADER, prompt_sha256
        cur = self.cur
        level, lang, world = cur["level"], cur["lang"], cur["world"]
        summary = list(post["summary"])
        if C.sha256_text("\n".join(summary)) != post["summary_sha"]:
            raise CaptureAbort("summary sha differs from the executor's post-sweep sha")
        try:
            full = T.full_summary(self.ex, post)
            psum = T.prompt_summary(full, self.tcfg)
        except T.TreatmentError as e:
            raise CaptureAbort("ft-v2 summary: %s" % e)
        obs = T.observed_types(psum)
        instr = it[lang]
        srcs = sources_for(world, it["id"] in self.repeat, fs=self._fs_on(), curve=self.use_curve,
                           only=self.only_sources)
        blocks = {"none": {"slot": None, "text": "", "ids": [], "real_tokens": 0, "counts": {}, "budget": None, "empty": True,
                           "coverage": None}}
        msgs = {"none": self.contract.messages(instr, psum, holding)}
        src_rows = {}
        if self.packager is None and any(s not in ("none", "FS") for s in srcs):
            raise CaptureAbort("ft-v2 capture needs a packager for the +M sources")
        for s in [s for s in srcs if s not in ("none", "FS")]:
            d = T.source_dir(self.ns, s, self.root)
            rows = T.read_rows(d)
            src_rows[s] = rows
            try:
                pk = self.packager.render(s, os.path.join(d, "ltm.db"), rows, item=it["id"], instruction=instr, lang=lang,
                                          holding=holding, summary_lines=psum)
            except T.TreatmentError as e:
                raise CaptureAbort("%s / %s: %s" % (cur["pres"], s, e))
            text = pk["text"] or ""
            blocks[s] = {"slot": "memory", "text": text, "ids": list(pk.get("kept") or []), "real_tokens": pk["tokens"]["real_max"],
                         "counts": pk["tokens"]["counts"], "budget": pk["tokens"]["budget"], "empty": not text, "coverage": None,
                         "sha256": C.sha256_text(text) if text else None,
                         "package": {k: pk.get(k) for k in ("profile", "mask", "off", "params_sha", "query_sha", "pkg_sha", "tokens",
                                                             "empty_reason", "dropped", "d2_dropped", "leave_out", "need", "store_shas",
                                                             "parser_ok", "provenance")},
                         "source": {"snapshot_sha": self._shas0.get(s), "rows_sha": T.rows_sha(rows), "rows_n": len(rows)}}
            msgs[s] = self.contract.messages(instr, psum, holding, memory_block=text or None)
        if "FS" in srcs:                                 # C4-open: fs-v2 per presentation (A24 item 4); fs_blocks.json is never read here
            if self.fs_renderer is None:
                raise CaptureAbort("ft-v2 capture needs the FS-v2 renderer (treatment.FSRenderer) for the FS source")
            d = T.source_dir(self.ns, "S*", self.root)
            rows = T.read_rows(d)
            src_rows["FS"] = rows
            try:
                fb = self.fs_renderer.render("S*", os.path.join(d, "ltm.db"), rows, item=it["id"], instruction=instr, lang=lang,
                                             holding=holding, summary_lines=psum)
            except (T.TreatmentError, ValueError) as e:
                raise CaptureAbort("%s / FS: %s" % (cur["pres"], e))
            text = fb["text"] or ""
            blocks["FS"] = {"slot": "fewshot", "text": text, "ids": list(fb.get("retained") or []), "real_tokens": int(fb["real_tokens"]),
                            "counts": dict(fb.get("counts") or {}), "budget": fb.get("budget"), "empty": not text, "coverage": None,
                            "sha256": C.sha256_text(text) if text else None,
                            # A31 step 5: fs-v2's retained_kinds is a {practice verdict: n} dict whose KEYS ("success") trip the
                            # record firewall (C.OUTCOME_KEYS); it is recorded as [[verdict, n], ...] (practice-row verdicts, never a
                            # test outcome). First reached here: FS had never rendered in an ft-v2 capture before A31 (B1).
                            "fs": dict({k: fb.get(k) for k in ("version", "retained", "dropped", "leave_out", "d2_dropped",
                                                                "empty_reason", "query_sha", "corpus_sha", "params_sha", "parser", "ranker",
                                                                "max_tokens", "candidates", "ranked", "has_persian")},
                                       retained_verdicts=sorted([str(k), int(v)] for k, v in (fb.get("retained_kinds") or {}).items())),
                            "source": {"snapshot_sha": self._shas0.get("S*"), "rows_sha": T.rows_sha(rows), "rows_n": len(rows)}}
            msgs["FS"] = self.contract.messages(instr, psum, holding, fewshot_block=text or None)
        requests = {}
        for s in srcs:
            joined = "\n".join(m["content"] for m in msgs[s])
            hit = C.mask_problems(joined)
            self.stats["masking_checked"] += 1
            if hit:
                raise CaptureAbort("masking hit in %s / %s: %s" % (cur["pres"], s, hit[:5]))
            if C.key_leak(joined):
                raise CaptureAbort("key-like text in the %s request of %s" % (s, cur["pres"]))
            if s != "none":
                self.stats["leak_checked"] += 1
                try:
                    spans, ex_audit = self._exempt(s, blocks[s], src_rows, instr, lang, holding, psum)
                except ValueError as e:
                    raise CaptureAbort("A31 exemption: %s / %s: %s" % (cur["pres"], s, e))
                blocks[s]["leak_exempt"] = ex_audit
                lk = leakage_problems(blocks[s]["text"], self.forbidden, spans)
                if lk:
                    raise CaptureAbort("leakage: the %s block of %s holds test item text / ids %s" % (s, cur["pres"], lk[:3]))
            requests[s] = prompt_sha256(msgs[s])
        gi = g_iso_problems(msgs, blocks, memory_header=PACKAGE_HEADER)
        self.stats["giso_checked"] += 1
        if gi:
            raise CaptureAbort("G-iso: %s: %s" % (cur["pres"], "; ".join(gi[:3])))
        return {"summary": summary, "summary_sha": post["summary_sha"], "holding": holding, "blocks": blocks, "requests": requests,
                "instruction": instr, "episode_id": post["episode_id"], "sweep_prims": post.get("sweep_prims"),
                "system_sha": C.sha256_text(msgs["none"][0]["content"]), "sources": srcs, "sha_order": sorted(requests),
                "prompt_summary": psum, "prompt_summary_sha": C.sha256_text("\n".join(psum)), "observed_types": obs,
                "full_summary_n": len(full)}

    def test_v1_checks(self):
        """test-v1 only: what the freeze fixed is what Phase A uses. (a) the memory channels equal the frozen channel set (a CLI
        --channels cannot switch procedures on after C2-h failed, A13); (b) params.json carries the A12 allocation; (c) the counter
        covers EVERY tokenizer (no Gemma-only coverage check). ft-v2 (after the A24.1 flip): channels and A12 are superseded by the
        package; the treatment identity must equal freeze.json's (treatment.entry_problems) and the counter must cover every tokenizer."""
        if self.ft2:
            bad = T.entry_problems(self.ns, self.freeze, (self.ident or {}).get("sha"))
            if bad:
                raise CaptureAbort("; ".join(bad))
            tk = self._tokenizers()
            if getattr(self.counter, "tokenizers", None) is not None and len(tk) < 2:
                raise CaptureAbort("test-v1: the token counter covers %s; the package budget needs every tokenizer" % tk)
            return
        fch = self.freeze.get("channels")
        if not fch:
            raise CaptureAbort("test-v1: freeze.json records no memory channels")
        if sorted(fch) != sorted(self.channels):
            raise CaptureAbort("test-v1: channels %s differ from the frozen channel set %s" % (sorted(self.channels), sorted(fch)))
        if not ((self.params.get("retrieval") or {}).get("a12")):
            raise CaptureAbort("test-v1: params.json has no retrieval.a12 allocation (the A12 coverage gate cannot run)")
        tk = self._tokenizers()
        if getattr(self.counter, "tokenizers", None) is not None and len(tk) < 2:
            raise CaptureAbort("test-v1: the token counter covers %s; A12 needs every tokenizer (Gemma and Anthropic)" % tk)

    def _coverage(self, post, level, lang, blk, holding, sl):
        """A12 coverage of the S* block of this presentation (token counts only). The twin language is rendered from the same evidence."""
        alloc = ((self.params.get("retrieval") or {}).get("a12")) if self.params else None
        if not alloc:
            return None
        import ltm as L
        other = "fa" if lang == "en" else "en"
        ob = sl.render_blocks(post, {"S*": self.ltms["S*"]}, level, other, self.channels, holding)["S*"]
        wm = self._fresh_wm(self.ltms["S*"], post)
        ncand = int(self.ltms["S*"].place_candidate_count(wm, channels=tuple(c for c in self.channels if c != "procedures") or ("places",)))
        cv = L.a12_coverage({lang: blk, other: ob}, ncand, level, self.counter, alloc)
        return {k: cv[k] for k in ("ok", "N", "B", "held", "exempt", "ids_identical", "whole_records", "complete_fits", "n_candidates")}

    # -- one cell (presentation x run) ------------------------------------------------------------------------------
    def make_request(self, key, it, lang, world):
        import step5_eval as se
        return se.EpisodeRequest(cell=key, episode_id="cap:" + C.sha256_text(key)[:24], run_id="ftm-capture", item=it, lang=lang,
                                 level=it["level"], mode="test", world="reset", world_kind=se.CANONICAL, world_seed=0, policy_seed=0,
                                 arm="none", features=frozenset(), memory=None, plan_source="capture", kitchen=self.scene)

    INAPPLICABLE_RERUNS = 2                              # a setup that fails at restore is first treated as a crash: re-run from a fresh restore

    def capture_cell(self, pres, iid, lang, world, run) -> dict:
        """One cell. A setup failure (`inapplicable`) on a presentation that passed the split's applicability gate is a crash, not a
        verdict: it is re-run from a fresh restore up to INAPPLICABLE_RERUNS times; only then is it journaled as `inapplicable`
        (and `finish` names it as a gate failure, except in the relocated world, where the spawn may legitimately hide an object)."""
        for tries in range(1, self.INAPPLICABLE_RERUNS + 2):
            rec = self._capture_cell_once(pres, iid, lang, world, run)
            if rec["status"] != "inapplicable":
                break
            self.event("inapplicable_rerun", pres=pres, run=run, attempt=tries, why=str(rec.get("inapplicable"))[:120])
        rec["tries"] = tries
        return rec

    def _capture_cell_once(self, pres, iid, lang, world, run) -> dict:
        key = C.cell_id(self.ns, "capture", pres, "c%d" % run)
        it = self.items[iid]
        self.cur = {"pres": pres, "level": it["level"], "lang": lang, "world": world}
        self.hook.pending = None
        req = self.make_request(key, it, lang, world)
        ctx = self.reset_patch(self.reloc_seed) if world == "reloc" else canonical_reset()
        t0 = time.time()
        OH.install()                                       # A34-A2 2(b): behaviour-preserving observation-pipeline health counters
        OH.reset()
        with ctx:
            res = self.ex.execute(req)
        obs_pipeline = OH.record(self.ident["sha"] if self.ft2 else None, (self.hook.pending or {}).get("sweep_prims"))
        rs = restored_state(world, self.reloc_seed, it.get("setup"))
        base = dict(key=key, namespace=self.ns, pres=pres, item=iid, lang=lang, level=it["level"], world=world, capture_run=run,
                    restored_state=rs, reloc_seed=self.reloc_seed if world == "reloc" else None, wall_s=round(time.time() - t0, 2))
        if res.inapplicable:
            if self.ft2:
                base.update(treatment=T.FT2_VERSION, treatment_id=self.ident["sha"])
            return C.make_record("capture", status="inapplicable", inapplicable=str(res.inapplicable)[:200], instruction=it[lang],
                                 holding=None, summary=[], summary_sha=None, blocks={}, requests={}, state_ref=None,
                                 world_hash=res.initial_state_hash, audit={"inapplicable": True}, **base)
        p = self.hook.pending
        if p is None:
            raise CaptureAbort("%s: the executor ran without calling the capture hook" % key)
        ex = res.extra or {}
        aud = ((ex.get("core") or {}).get("audit")) or {}
        counts = aud.get("counts") or {}
        if int(aud.get("violations") or 0) or int(counts.get("oracle") or 0) or int(counts.get("map-build") or 0):
            raise CaptureAbort("%s: honesty audit failed during the sweep: %s" % (key, json.dumps(aud, default=str)[:200]))
        if ex.get("summary_sha") != p["summary_sha"]:
            raise CaptureAbort("%s: summary sha differs between the executor and the hook" % key)
        state = ex.get("post_sweep_state")
        if not state:
            raise CaptureAbort("%s: no post-sweep state captured (state_capture off?)" % key)
        chooser_none = (getattr(self.ex, "last", None) or {}).get("chooser") is None
        if not chooser_none:
            raise CaptureAbort("%s: the capture executor built a chooser (memory must be off)" % key)
        sref = "states/" + state_filename(pres, run)
        if self.ft2 and C.sha256_text("\n".join(p["summary"])) != p["summary_sha"]:
            raise CaptureAbort("%s: the executor summary sha changed under the ft-v2 renderer" % key)
        sdoc = {"schema": "ftm.capture-state/1", "pres": pres, "run": run, "key": key, "world_hash": res.initial_state_hash,
                "summary_sha": p["summary_sha"], "state": state, "obs_pipeline": obs_pipeline}
        text = json.dumps(sdoc, ensure_ascii=False, sort_keys=True, default=str)
        if C.key_leak(text):
            raise CaptureAbort("%s: key-like text in the state file" % key)
        os.makedirs(os.path.join(self.docs, "states"), exist_ok=True)
        with open(os.path.join(self.docs, sref), "w", encoding="utf-8") as f:
            f.write(text)
        audit = {"honesty": {"violations": int(aud.get("violations") or 0), "oracle": int(counts.get("oracle") or 0),
                             "map_build": int(counts.get("map-build") or 0)},
                 "g_iso": {"ok": True, "chooser_none": True, "memory_none": True}, "masking": {"ok": True, "n": len(p["requests"])},
                 "leakage": {"ok": True}, "plan_order": "after_sweep", "system_sha": p["system_sha"],
                 "channels": list(self.channels), "sweep_prims": p.get("sweep_prims"), "sources": p["sources"]}
        base["repeat"] = iid in self.repeat
        if self.ft2:                                       # C4: the treatment fields exist ONLY in ft-v2 rows (ft-v1 rows keep their bytes)
            audit["a12"] = "not applicable: superseded by the A24 package (no coverage pass is claimed)"
            audit["channels"] = []
            base.update(treatment=T.FT2_VERSION, treatment_id=self.ident["sha"], prompt_summary=p["prompt_summary"],
                        prompt_summary_sha=p["prompt_summary_sha"], observed_types=p["observed_types"], full_summary_n=p["full_summary_n"])
        return C.make_record("capture", status="ok", instruction=p["instruction"], holding=p["holding"], summary=p["summary"],
                             summary_sha=p["summary_sha"], blocks=p["blocks"], requests=p["requests"], state_ref=sref,
                             world_hash=res.initial_state_hash, audit=audit, obs_pipeline=obs_pipeline, **base)

    # -- the phase ------------------------------------------------------------------------------------------------------
    def snapshot_shas(self, names) -> dict:
        return {s: self.snapshot_sha_fn(T.source_dir(self.ns, s, self.root)) for s in names}

    def check_snapshot_manifest(self, name) -> dict:
        d = T.source_dir(self.ns, name, self.root)
        mp = os.path.join(d, "manifest.json")
        try:
            with open(mp, encoding="utf-8") as f:
                man = json.load(f)
        except (OSError, ValueError):
            raise CaptureAbort("snapshot %s: manifest.json missing or unreadable" % name)
        lk = (man.get("extra") or {}).get("leakage") or man.get("leakage")
        if lk is not None and not (isinstance(lk, dict) and lk.get("ok") is True):
            raise CaptureAbort("snapshot %s: its manifest records a failed leakage gate: %s" % (name, str(lk)[:120]))
        return {"leakage": lk, "state_hash": man.get("state_hash")}

    def run(self) -> int:
        ns = self.ns
        guard = C.guard_items(self.item_ids(), ns, allow_test=self.allow_test, root=self.root)
        pres_list = self.presentations()
        if self.limit:
            pres_list = pres_list[: self.limit]
        cells = [(p, i, lg, w, r) for (p, i, lg, w) in pres_list for r in self.runs]
        try:
            need = self.needed_sources()
            if "reloc" in self.worlds and self.reloc_set() and self.reloc_seed is None:
                raise CaptureAbort("the relocated world needs a seed (--reloc-seed, or reloc_seed in freeze.json)")
            if (not self.ft2 and self.fs_doc is None and self.use_fs and (not self.only_sources or "FS" in self.only_sources)
                    and ns == "test-v1"):
                raise CaptureAbort("test-v1 needs docs/final-test/matrix/fs_blocks.json (FS arms)")
            try:
                self.dataset_sha = C.sha256_file(C.paths(self.root)["dataset"])
            except OSError:
                self.dataset_sha = None                # only a synthetic root (selftest) lacks it; dataset_problems refuses in test-v1
            dp = C.dataset_problems(ns, self.root)
            if dp:
                raise CaptureAbort("; ".join(dp))
            if self.ft2:                                 # C4: no resume across a treatment change, no mixed journal
                if self.ident is None:
                    self.ident = T.identity(ns, self.root)
                bad = T.journal_problems(self.jr.rows(), ns, self.ident["sha"])
                if bad:
                    raise CaptureAbort("ft-v2 resume refused: " + "; ".join(bad[:4]))
            elif ns == "test-v1":
                bad = T.journal_problems(self.jr.rows(), ns) + T.entry_problems(ns, self.freeze, None)
                if bad:
                    raise CaptureAbort("; ".join(bad[:4]))
            if ns == "test-v1":
                self.test_v1_checks()
            shas0 = self.snapshot_shas(need)
            self._shas0 = dict(shas0)
            for s, sh in shas0.items():
                self.check_snapshot_manifest(s)
            if self.freeze and ns != "test-v1" and self.freeze.get("freeze_status") is not None:
                # FREEZE-GATES step 3: a freeze.json marked "treatment-only-unvalidated" (TREAT-FREEZE) records the treatment only; its other
                # fields are a stale earlier configuration, so they are not the reference of a DEV rehearsal (recorded, never silent).
                # test-v1 stays strict (below), and the test needs a validated freeze anyway.
                self.event("freeze_artefacts_not_checked", status=self.freeze.get("freeze_status"),
                           reason="not a validated freeze; dev rehearsal checks snapshots against their own manifests")
            elif self.freeze:                       # per-file shas against freeze.json (the keys freeze.py really writes)
                bad = C.snapshot_freeze_problems(need, self.root, self.freeze) + C.freeze_artifact_problems(self.freeze, self.root)
                if bad:
                    raise CaptureAbort("frozen artefact(s) changed: " + "; ".join(bad[:6]))
            elif ns == "test-v1":
                raise CaptureAbort("test-v1 capture needs freeze.json (snapshot / params / FS-block shas are checked against it)")
            self.event("capture_start", namespace=ns, n_cells=len(cells), runs=list(self.runs), worlds=list(self.worlds),
                       sources=need, snapshot_shas=shas0, reloc_seed=self.reloc_seed, guard=guard, channels=list(self.channels),
                       partial=bool(self.limit), tokenizers=self._tokenizers(), dataset_sha256=self.dataset_sha)
            self.ltms = (self.open_sources(need) if need else {}) if not self.ft2 else {}
            done = self.jr.keys()
            for (pres, iid, lg, w, run) in cells:
                key = C.cell_id(ns, "capture", pres, "c%d" % run)
                if key in done:
                    continue
                if self.interrupt and self.new_cells >= self.interrupt:
                    raise Interrupted()
                rec = self.capture_cell(pres, iid, lg, w, run)
                self.jr.append(rec)
                self.new_cells += 1
                self.log("[capture] %s run %d %s (%d/%d)" % (pres, run, rec["status"], len(self.jr.keys()), len(cells)))
        except Interrupted:
            self.event("interrupt", n_new=self.new_cells)
            return 5
        except CaptureAbort as e:
            self.event("abort_capture", reason=str(e)[:300])
            self.log("ABORT: %s" % e)
            return 4
        return self.finish(cells, need, shas0)

    def _tokenizers(self) -> list:
        return [str(t) for t in (getattr(self.counter, "tokenizers", None) or [])]

    def finish(self, cells, need, shas0) -> int:
        ns = self.ns
        gates = {}
        rows = self.jr.rows()
        want = {C.cell_id(ns, "capture", p, "c%d" % r) for (p, _i, _l, _w, r) in cells}
        have = {r["key"] for r in rows}
        gates["all_cells"] = {"pass": want <= have, "missing": len(want - have)}
        shas1 = self.snapshot_shas(need)
        gates["snapshot_hash"] = {"pass": shas1 == shas0, "changed": sorted(s for s in shas0 if shas1.get(s) != shas0[s])}
        run1 = [r for r in rows if r["key"] in want and r["capture_run"] == min(self.runs)]
        ok_rows = [r for r in run1 if r.get("status") == "ok"]
        cov = aggregate_coverage(ok_rows)
        required = ns == "test-v1"
        not_applicable = {}
        if self.ft2:                                 # C4 / Astra Q6: A12 is superseded under A24 (no manufactured coverage pass)
            not_applicable["a12_coverage"] = "superseded by the A24 package (places coverage does not apply)"
        else:
            # A12 says the check runs on EVERY S* test prompt before execution: in test-v1 an unchecked presentation is a failure, never a pass
            gates["a12_coverage"] = {"pass": a12_gate(cov, required), "required": required, "summary": cov}
        bad_status = sorted(r["pres"] for r in run1 if r.get("status") != "ok")
        inapp_canon = [p for p in bad_status if p.startswith("canon:")]
        gates["applicable"] = {"pass": not inapp_canon, "canon_inapplicable": inapp_canon[:20],
                               "reloc_inapplicable": [p for p in bad_status if p.startswith("reloc:")]}
        gates["masking"] = {"pass": True, "n": self.stats["masking_checked"]}
        gates["g_iso"] = {"pass": True, "n": self.stats["giso_checked"]}
        gates["leakage"] = {"pass": True, "n": self.stats["leak_checked"]}
        paths = [C.journal_path(ns, "capture", self.root)]
        sd = os.path.join(self.docs, "states")
        if os.path.isdir(sd):
            paths += [os.path.join(sd, f) for f in sorted(os.listdir(sd))]
        gates["key_leak"] = {"pass": not any(file_has_key(p) for p in paths), "files": len(paths)}
        if len(self.runs) > 1:
            gates["reset_pairs"] = self.reset_pairs(rows, want)
        failed = sorted(g for g, v in gates.items() if not v.get("pass"))
        if self.ft2:
            self.event("capture_gates", gates=gates, failed=failed, partial=bool(self.limit), tokenizers=self._tokenizers(),
                       not_applicable=not_applicable, treatment=T.FT2_VERSION, treatment_id=self.ident["sha"])
        else:
            self.event("capture_gates", gates=gates, failed=failed, partial=bool(self.limit), tokenizers=self._tokenizers())
        if failed:
            self.log("GATES FAILED: %s" % failed)
            return 2
        if self.limit:                                               # a smoke subset is never a finished phase
            self.log("partial run (--limit): no phase_complete")
            return 0
        extra = {"treatment": T.FT2_VERSION, "treatment_id": self.ident["sha"]} if self.ft2 else {}
        self.event("phase_complete", n=len(want), journal_sha=C.sha256_file(C.journal_path(ns, "capture", self.root)),
                   runs=list(self.runs), namespace=ns, channels=list(self.channels), tokenizers=self._tokenizers(),
                   reloc_inapplicable=gates["applicable"]["reloc_inapplicable"], dataset_sha256=self.dataset_sha, **extra)
        return 0

    def reset_pairs(self, rows, want) -> dict:
        """Run 1 vs run 2 post-sweep states of the same presentation -> `reset_pairs.json` (input of `reset_gate.py validate --pairs`,
        which records the one-shot A14 c validation). A non-recording preview verdict is added when the tolerances are calibrated."""
        by = {}
        for r in rows:
            if r["key"] in want and r.get("status") == "ok":
                by.setdefault(r["pres"], {})[r["capture_run"]] = r
        pairs, names = [], []
        for pres in sorted(by):
            d = by[pres]
            if 1 in d and 2 in d:
                a = json.load(open(os.path.join(self.docs, d[1]["state_ref"]), encoding="utf-8"))["state"]
                b = json.load(open(os.path.join(self.docs, d[2]["state_ref"]), encoding="utf-8"))["state"]
                pairs.append([a, b])
                names.append(pres)
        out_path = os.path.join(self.docs, "reset_pairs.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, sort_keys=True)
        res = {"pass": bool(pairs), "n_pairs": len(pairs), "file": "reset_pairs.json", "file_sha256": C.sha256_file(out_path),
               "presentations": names, "preview": None}
        try:
            from final_test import reset_gate as rg
            tol = self.tol_loader() if self.tol_loader else rg.load_tolerances(None, "calibrated")
            v = rg.validate([tuple(p) for p in pairs], tol, expected=len(pairs))
            res["preview"] = {"ok": v["ok"], "pairs": v["pairs"], "pass": v["pass"], "failures": v["failures"][:5]}
            res["pass"] = bool(pairs) and bool(v["ok"])
        except Exception as e:                                       # tolerances not calibrated yet: nothing to preview
            res["preview"] = {"skipped": "%s: %s" % (type(e).__name__, str(e)[:120])}
        return res


# ================================================================================================ live wiring (WSL)
def load_items(root=None) -> dict:
    with open(C.paths(root)["dataset"], encoding="utf-8") as f:
        return {i["id"]: i for i in json.load(f)["items"]}


def load_fs(root=None):
    p = C.paths(root)["fs_blocks"]
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("sha256") != C.self_sha(d):
        raise CaptureAbort("fs_blocks.json self-sha mismatch (edited after writing)")
    for lv in LEVELS:
        b = d.get(lv) or {}
        if C.sha256_text(b.get("text", "")) != b.get("sha256"):
            raise CaptureAbort("fs_blocks.json: %s text sha mismatch" % lv)
    return d


def load_freeze(root=None) -> dict:
    try:
        with open(C.paths(root)["freeze"], encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def make_opener(ns, root, counter, params, work, ex_getter):
    def _open(names):
        import ltm as L
        import step5_live as sl
        out = {}
        for s in names:
            d = T.source_dir(ns, s, root) if C.is_replication(ns) else C.snapshot_dir(s, root)      # K23: the kitchen's own snapshots
            tl = os.path.join(work, "capture_testlog.%s.db" % s.replace("*", "star"))
            if os.path.exists(tl):
                os.remove(tl)
            f = L.LongTermMemory.open_frozen(d, tl, kitchen=C.ns_scene(ns), params=params)
            f.count_tokens = counter
            sl.assert_b_memory(f)
            ex = ex_getter()
            f.begin_test_epoch(frame=ex.frame, map_rev=ex.map_rev)
            out[s] = f
        return out
    return _open


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase A capture (WSL, simulator, no LLM)")
    ap.add_argument("--namespace", required=True, choices=sorted(C.NAMESPACES))
    ap.add_argument("--items", required=True, choices=sorted(NS_FOR_ITEMS))
    ap.add_argument("--runs", default="1")
    ap.add_argument("--worlds", default="canon")
    ap.add_argument("--sources", default=None, help="restrict the prompt sources, e.g. none,S*")
    ap.add_argument("--no-curve", action="store_true")
    ap.add_argument("--no-fs", action="store_true")
    ap.add_argument("--channels", default=None, help="memory channels of the S* block, e.g. places or places,procedures")
    ap.add_argument("--reloc-seed", type=int, default=None)
    ap.add_argument("--reloc-all", action="store_true", help="rehearsal only: relocate every item, not just the probe items")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--allow-missing-anthropic", action="store_true")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--steal-lock", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke: only the first N presentations (never a phase_complete)")
    ap.add_argument("--inject-interrupt", type=int, default=0)
    ap.add_argument("--root", default=None)
    a = ap.parse_args(argv)
    if (a.namespace not in DEV_NAMESPACES) if a.items == "dev" else (NS_FOR_ITEMS[a.items] != a.namespace):
        raise SystemExit("--items %s belongs to namespace %s" % (a.items, NS_FOR_ITEMS[a.items]))
    runs = tuple(int(x) for x in a.runs.split(",") if x)
    worlds = tuple(w for w in a.worlds.split(",") if w)
    if not set(runs) <= {1, 2} or 1 not in runs or (2 in runs and a.namespace == "test-v1"):
        raise SystemExit("--runs: 1 (matrix) or 1,2 (dev reset-gate re-capture only)")
    if not set(worlds) <= set(C.WORLDS) or "canon" not in worlds:
        raise SystemExit("--worlds: canon[,reloc]")
    root = a.root
    if a.namespace == "test-v1" and a.allow_missing_anthropic:
        raise SystemExit("--allow-missing-anthropic is refused in test-v1: A12 coverage needs every tokenizer")
    manifest = C.load_manifest(root)
    items = load_items(root)
    freeze = load_freeze(root)
    P = C.paths(root)
    ft2 = T.is_ft2(a.namespace)
    fs_doc = None if (a.no_fs or ft2) else load_fs(root)         # C4-open: ft-v2 never reads the fs-v1 fs_blocks.json
    if ft2 and a.channels:
        raise SystemExit("--channels is an ft-v1 option: an ft-v2 namespace renders the package (treatment.FT2)")
    channels = () if ft2 else tuple(c for c in (a.channels or ",".join(freeze.get("channels") or ["places"])).split(",") if c)
    if a.namespace == "test-v1" and a.channels and sorted(channels) != sorted(freeze.get("channels") or []):
        raise SystemExit("--channels %s differs from the frozen channel set %s (test-v1)" % (",".join(channels), freeze.get("channels")))
    seed = a.reloc_seed if a.reloc_seed is not None else freeze.get("reloc_seed")
    if seed is None and a.namespace != "test-v1":
        seed = REHEARSAL_RELOC_SEED
    only = [s for s in a.sources.split(",") if s] if a.sources else None
    if a.dry_run:
        cap = Capturer(namespace=a.namespace, executor=None, contract=None, counter=None, open_sources=None, items=items,
                       manifest=manifest, fs_doc=fs_doc, channels=channels, allow_test=a.allow_test, root=root, worlds=worlds,
                       runs=runs, reloc_seed=seed, reloc_all=a.reloc_all, only_sources=only, use_curve=not a.no_curve,
                       use_fs=not a.no_fs, freeze=freeze)
        g = C.guard_items(cap.item_ids(), a.namespace, allow_test=a.allow_test, root=root)
        need = cap.needed_sources()
        missing = [s for s in need if not os.path.isdir(T.source_dir(a.namespace, s, root))]
        print(json.dumps({"guard": g, "presentations": len(cap.presentations()), "cells": len(cap.presentations()) * len(runs),
                          "sources": need, "missing_snapshots": missing, "fs_blocks": fs_doc is not None, "channels": list(channels),
                          "reloc_seed": seed, "reloc_items": len(cap.reloc_set()), "treatment": T.treatment_of(a.namespace)}, indent=1))
        return 0 if not missing else 2
    # ---- live: WSL only
    os.environ.setdefault("TRACE_ENABLED", "0")
    import ltm as L
    import step5_live as sl
    import honest_skills as hs
    from planners.contract import MEMORY_HEADER, PACKAGE_HEADER, PlannerContract
    sys.path.insert(0, os.path.join(C.SIM, "final_test", "c3"))
    from real_counter import RealMaxCounter
    work = C.ns_work(a.namespace, root)
    os.makedirs(work, exist_ok=True)
    with C.PhaseLock(a.namespace, "capture", root, steal=a.steal_lock):
        counter = RealMaxCounter(os.path.join(work, "tokcache.json"), PACKAGE_HEADER if ft2 else MEMORY_HEADER, ollama_endpoint=a.ollama,
                                 allow_missing_anthropic=a.allow_missing_anthropic)
        ex = sl.LiveExecutor(memory="none", policy="console", hints=False, plan_after_sweep=True, state_capture=True,
                             scene=C.ns_scene(a.namespace))
        try:
            ex._boot()
            if ft2:                                           # C4: the D2-filtered type list, ft-v2 contract, the package
                contract = PlannerContract(T.planner_types(hs.detector_types(ex._det)), tools=hs.TOOLS, version=T.FT2_VERSION)
            else:
                contract = PlannerContract(hs.detector_types(ex._det), tools=hs.TOOLS)
            params = L.load_params()
            packager = T.Packager(a.namespace, counter, root=root) if ft2 else None
            cap = Capturer(namespace=a.namespace, executor=ex, contract=contract, counter=counter,
                           open_sources=make_opener(a.namespace, root, counter, params, work, lambda: ex), items=items,
                           manifest=manifest, fs_doc=fs_doc, channels=channels, allow_test=a.allow_test, root=root, worlds=worlds,
                           runs=runs, reloc_seed=seed, reloc_all=a.reloc_all, only_sources=only, use_curve=not a.no_curve,
                           use_fs=not a.no_fs, limit=a.limit, interrupt=a.inject_interrupt, params=params, freeze=freeze,
                           packager=packager, scene=C.ns_scene(a.namespace))
            ex.plan_hook = cap.hook
            if a.steal_lock:
                cap.event("lock_stolen")
            rc = cap.run()
        finally:
            try:
                counter.save()
            except Exception:
                pass
            ex.close()
    if a.limit and rc == 0:
        print("smoke run (--limit): no phase_complete is meaningful")
    return rc


if __name__ == "__main__":
    sys.exit(main())
