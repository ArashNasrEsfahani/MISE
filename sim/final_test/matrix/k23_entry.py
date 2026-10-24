"""K23 (Kitchens 2 / 3) entry checks shared by the K23 entry points (k23_practice.py, k23_run.py, k23_size_decision.py,
k23_freeze_agent.py). NEW module (2026-10-20): it adds behaviour for the replication namespaces WITHOUT changing a byte of the files
bound by the A37 alignment record (common.py, plan.py, k23_ns.py, freeze.py, treatment.py, ...), the treatment pins, the K2 perception
freeze or the Kitchen 1 manifest. Kitchen 1 namespaces are refused here (they keep their own entry points).

  * kind(ns)                practice / rehearsal / test for a K23 namespace (from common.NAMESPACES: scene + items), else ValueError
  * sample_problems(scene)  the FROZEN A29 sample manifest of one kitchen (kitchen, self-hash, split + inclusion hashes, unique test ids,
                            nested candidates) -- the checks of k23_ns.a29_roster without the size decision
  * samples_problems()      A29 item 2 / A29.1 item 3: BOTH kitchens' manifests exist and verify ("generated and hashed before practice
                            in either kitchen"); the practice entry refuses a live run until this is []
  * package_problems(scene) the kitchen's planner package inputs are final: uses table + homeowner inventory carry version 'k23-v1'
                            (not a '-draft'), name the kitchen where they have the field, and the approved briefing loads
  * k23_manifest(ns)        the effective split manifest + the A29 roster under C.A29_ROSTER for a K23 TEST namespace (the
                            acc_common.acceptance_manifest pattern: load_manifest itself never carries it; only K23 runner objects do)
  * coco_problems()         A38 item 4 + Arash 2026-10-20 ("keep the current order"): the FloorPlan15 COCO-start sensitivity checkpoint
                            is appended to the K2 perception freeze (flags cleared, the record lists it) AND the successor metadata of
                            k23_coco_meta.py is written (record_sha256 / n_files describe the record as it stands, coco_append entry)
  * order_problems(ns)      prereg A39.2 (Arash 2026-10-20 ~16:15): Kitchen 2 practice and rehearsal start only after the COCO run,
                            --add-coco and k23_coco_meta (= coco_problems() == []); Kitchen 3 namespaces have no such condition
  * test_entry_problems(ns) every K23 TEST entry / resume: the kitchen's agent freeze binds the RUNNING treatment identity, the dataset
                            the split was cut from (A26 inclusion) and (FloorPlan15) the COCO append -- the test-v1 checks of
                            treatment.entry_problems / common.dataset_problems, which only fire for test-v1
  * briefing_hints(ns)      hook B's hint function from THE NAMESPACE'S kitchen briefing (practice_k.build_live reads the FloorPlan1
                            table: treatment.briefing_hint_table's default scene)

Pure python + file reads; no simulator, no provider. Python 3.10 syntax.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                       # noqa: E402  (also puts sim/ on sys.path)
import k23_ns as K                                       # noqa: E402

KINDS = {"practice_pool": "practice", "dev_pilot": "rehearsal", C.A29_ROSTER: "test"}
FINAL_PACKAGE_VERSION = "k23-v1"


def k23_namespaces(kind_: str | None = None) -> tuple:
    out = []
    for ns in sorted(C.NAMESPACES):
        if C.ns_scene(ns) in C.REPLICATION_SCENES and C.NAMESPACES[ns].get("items") in KINDS:
            if kind_ is None or KINDS[C.NAMESPACES[ns]["items"]] == kind_:
                out.append(ns)
    return tuple(out)


def kind(ns: str) -> str:
    """'practice' | 'rehearsal' | 'test' for a K23 namespace; ValueError for anything else (Kitchen 1 namespaces included)."""
    cfg = C.ns_check(ns)
    if C.ns_scene(ns) not in C.REPLICATION_SCENES or cfg.get("items") not in KINDS:
        raise ValueError(f"{ns!r} is not a Kitchen 2 / 3 namespace (K23 entry points serve {', '.join(k23_namespaces())} only)")
    k = KINDS[cfg["items"]]
    if k == "test" and not cfg.get("replication_test"):
        raise ValueError(f"{ns!r}: an A29-roster namespace must be marked replication_test")
    return k


def scene_of(ns: str) -> str:
    kind(ns)
    return C.ns_scene(ns)


def rel(path: str, root: str | None = None) -> str:
    """Repo-relative '/' path when under the repo, else the absolute path (another drive, a temp dir)."""
    try:
        r = os.path.relpath(path, C.paths(root)["root"])
    except ValueError:
        return path
    return path if r.startswith("..") else r.replace(os.sep, "/")


def sample_problems(scene: str, root: str | None = None) -> list:
    """The frozen A29 sample manifest of `scene` (written once by a29_sample.py). [] = present and intact."""
    mp, _dp = K.a29_files(scene, root)
    if not os.path.exists(mp):
        return [f"{scene}: no frozen A29 sample ({rel(mp, root)})"]
    try:
        with open(mp, encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError) as e:
        return [f"{scene}: A29 sample unreadable ({type(e).__name__})"]
    out = []
    if m.get("kitchen") != scene:
        out.append(f"{scene}: A29 sample names kitchen {m.get('kitchen')!r}")
    if C.self_sha(m) != m.get("sha256"):
        out.append(f"{scene}: A29 sample self-hash mismatch (edited after writing)")
    man = C.load_manifest(root)
    if m.get("split_manifest_sha256") != man.get("sha256") or m.get("inclusion") != (man.get("inclusion") or {}).get("sha256"):
        out.append(f"{scene}: A29 sample written for another split / inclusion manifest")
    order = list(m.get("order") or [])
    if not order or len(set(order)) != len(order) or set(order) - set(man["test"]):
        out.append(f"{scene}: A29 sample order empty, duplicated or not test items")
    cands = m.get("candidates") or {}
    if not cands or any(list(ids) != order[:len(ids)] for ids in cands.values()) or list(cands.get("all") or []) != order:
        out.append(f"{scene}: A29 sample candidates are not nested prefixes of the order")
    want = {"first_50": min(50, len(order)), "first_60": min(60, len(order)), "all": len(order)}      # A29 item 2 (Astra K23-TOOLING-G1 3)
    if sorted(cands) != sorted(want) or any(len(cands.get(k) or []) != n for k, n in want.items()) or m.get("n_pool") != len(order):
        out.append(f"{scene}: A29 sample must hold exactly the candidates first_50 / first_60 / all at their registered lengths")
    return out


def samples_problems(root: str | None = None) -> list:
    """A29 item 2 (and A29.1 item 3): both replication kitchens' pools and candidate manifests exist, hashed, before practice in EITHER."""
    out = []
    for sc in C.REPLICATION_SCENES:
        out += sample_problems(sc, root)
    return out


def _version_problems(path: str, scene: str, label: str) -> list:
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        return [f"{scene}: {label} unreadable ({type(e).__name__}: {os.path.basename(path)})"]
    out = []
    v = str(d.get("version") or "")
    if v != FINAL_PACKAGE_VERSION:
        out.append(f"{scene}: {label} version {v!r} is not the final {FINAL_PACKAGE_VERSION!r} (finalize it at the kitchen's freeze, "
                   f"BEFORE practice: the treatment identity hashes it)")
    if "kitchen" in d and d.get("kitchen") != scene:
        out.append(f"{scene}: {label} names kitchen {d.get('kitchen')!r}")
    return out


def package_problems(scene: str, root: str | None = None) -> list:
    """The planner package inputs of one replication kitchen are final before its practice (their bytes enter the identity)."""
    if scene not in C.REPLICATION_SCENES:
        return [f"{scene} is not a replication kitchen"]
    import ltm_package as LP
    out = _version_problems(LP.uses_path(scene), scene, "uses table")
    out += _version_problems(LP.inventory_path(scene), scene, "homeowner inventory")
    try:
        import kitchen_briefing as KB
        if not KB.load_records(scene, allow_unapproved=False):
            out.append(f"{scene}: the approved briefing has no records")
    except Exception as e:                                # noqa: BLE001  (BriefingNotApproved, missing natural tour, ...)
        out.append(f"{scene}: approved briefing not loadable ({type(e).__name__}: {str(e)[:120]})")
    return out


def k23_manifest(ns: str, root: str | None = None, manifest: dict | None = None) -> dict:
    """The effective split manifest; for a K23 TEST namespace plus the A29 roster under C.A29_ROSTER (a29_roster validates the frozen
    sample + the recorded item-6 size decision and raises RuntimeError otherwise; never a default)."""
    m = dict(manifest if manifest is not None else C.load_manifest(root))
    if kind(ns) == "test":
        m[C.A29_ROSTER] = K.a29_roster(ns, root)
    return m


COCO_SCENE = "FloorPlan15"
COCO_WEIGHTS = "sim/weights/v3/FloorPlan15.yolo26s_coco.pt"


def coco_problems(root: str | None = None) -> list:
    base = C.paths(root)["root"]
    d = os.path.join(base, "sim", "kitchen_v3", COCO_SCENE)
    try:
        with open(os.path.join(d, "frozen_weights.meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError) as e:
        return [f"{COCO_SCENE}: perception freeze metadata unreadable ({type(e).__name__})"]
    out = []
    if meta.get("coco_pending") is not False or meta.get("test_evaluation_blocked") is not False:
        out.append(f"{COCO_SCENE}: the COCO-start sensitivity is not appended to the perception freeze (coco_pending / TEST block still set): "
                   "finish the COCO run, then k23_freeze.py --scene FloorPlan15 --add-coco (A38 item 4; Arash 2026-10-20 order)")
    try:
        listed = [ln.split(None, 1)[1].strip().lstrip("*") for ln in open(os.path.join(d, "frozen_weights.sha256"), encoding="utf-8") if ln.strip()]
    except OSError:
        listed = []
    if COCO_WEIGHTS not in listed:
        out.append(f"{COCO_SCENE}: the perception freeze record does not list {COCO_WEIGHTS}")
    try:
        raw = open(os.path.join(d, "frozen_weights.sha256"), "rb").read()
    except OSError:
        raw = None
    ca = meta.get("coco_append")
    if (raw is None or not isinstance(ca, dict) or ca.get("coco") != COCO_WEIGHTS or meta.get("record_sha256") != __import__("hashlib").sha256(raw).hexdigest()
            or meta.get("n_files") != sum(1 for ln in raw.split(b"\n") if ln.strip())):
        out.append(f"{COCO_SCENE}: the successor perception-freeze metadata is not written for the record as it stands: run "
                   "sim/kitchen_v3/k23/k23_coco_meta.py --write after --add-coco")
    return out


ORDER_EVENT = "k23_order_ok"


def order_evidence(root: str | None = None) -> dict:
    """The COCO state an order check passed on: sha256 of the FloorPlan15 perception record and its metadata."""
    import hashlib
    d = os.path.join(C.paths(root)["root"], "sim", "kitchen_v3", COCO_SCENE)
    out = {}
    for k, f in (("record_sha256", "frozen_weights.sha256"), ("meta_sha256", "frozen_weights.meta.json")):
        try:
            out[k] = hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest()
        except OSError:
            out[k] = None
    return out


def record_order_ok(ns: str, phase: str, root: str | None = None) -> None:
    """prereg A39.2 evidence (Astra K23-TOOLING-T4-r1): the wrapper appends an event to the namespace's events journal right after
    order_problems(ns) == [] and BEFORE dispatching a live FloorPlan15 practice / rehearsal step. execute.py / common.py are frozen
    (Kitchen 1 freeze, A37 record), so a direct execute.py run cannot be refused at that boundary; it is DETECTED at the agent freeze,
    which requires this event before the first practice row / the first capture_start of the namespace (k23_freeze_agent)."""
    if scene_of(ns) != COCO_SCENE or kind(ns) not in ("practice", "rehearsal"):
        return
    os.makedirs(C.ns_docs(ns, root), exist_ok=True)
    C.Journal(C.journal_path(ns, "events", root)).append(C.make_record("events", event=ORDER_EVENT, phase=phase, namespace=ns,
                                                                       rule="prereg A39.2", **order_evidence(root)))


def order_evidence_problems(ns: str, first_kind: str, root: str | None = None) -> list:
    """For the freeze: a FloorPlan15 practice / rehearsal namespace must show an order event no later than its first practice row
    (first_kind='practice') or its first capture_start event (first_kind='capture_start')."""
    if scene_of(ns) != COCO_SCENE or kind(ns) not in ("practice", "rehearsal"):
        return []
    ev = C.Journal(C.journal_path(ns, "events", root)).rows() if os.path.exists(C.journal_path(ns, "events", root)) else []
    oks = [e.get("ts") for e in ev if e.get("event") == ORDER_EVENT]
    if first_kind == "practice":
        jp = C.journal_path(ns, "practice", root)
        firsts = [r.get("ts") for r in (C.Journal(jp).rows() if os.path.exists(jp) else [])]
    else:
        firsts = [e.get("ts") for e in ev if e.get("event") == "capture_start"]
    if not firsts:
        return []
    if not oks or min(oks) > min(firsts):
        return [f"A39.2 order: no {ORDER_EVENT} event in {ns} before its first {first_kind} (a step ran outside the checked wrapper, or "
                "before the COCO run + --add-coco + k23_coco_meta)"]
    return []


def order_problems(ns: str, root: str | None = None) -> list:
    """prereg A39.2: a FloorPlan15 practice / rehearsal namespace waits for the COCO run + --add-coco + k23_coco_meta."""
    if scene_of(ns) != COCO_SCENE or kind(ns) not in ("practice", "rehearsal"):
        return []
    return [f"A39.2 order: Kitchen 2 {kind(ns)} starts only after the COCO run, --add-coco and k23_coco_meta.py --write -- " + p
            for p in coco_problems(root)]


def load_agent_freeze(scene: str, root: str | None = None) -> dict:
    try:
        with open(K.freeze_path(scene, root), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def test_entry_problems(ns: str, root: str | None = None, *, ident_sha: str | None = None) -> list:
    """K23 TEST entry checks beyond k23_ns.replication_freeze_status (which guard_items runs): [] = the test may enter / resume."""
    if kind(ns) != "test":
        return [f"{ns} is not a K23 test namespace"]
    scene = scene_of(ns)
    fz = load_agent_freeze(scene, root)
    if not fz:
        return [f"{scene}: no agent freeze ({rel(K.freeze_path(scene, root), root)})"]
    out = []
    if ident_sha is None:
        import treatment as T
        try:
            ident_sha = T.identity(ns, root)["sha"]
        except Exception as e:                           # noqa: BLE001  (pins moved: fails closed)
            return [f"treatment identity not computable ({type(e).__name__}: {str(e)[:160]})"]
    if fz.get("treatment_id") != ident_sha:
        out.append(f"{scene}: the running treatment identity {str(ident_sha)[:12]} differs from the agent freeze's {str(fz.get('treatment_id'))[:12]}")
    try:
        got = C.sha256_file(C.paths(root)["dataset"])
    except OSError:
        got = None
    man = C.load_manifest(root)
    want = (man.get("inclusion") or {}).get("dataset_sha256") or man.get("dataset_sha256")
    if got is None or got != want or got != fz.get("dataset_sha256"):
        out.append(f"{scene}: the gold dataset differs from the split's / the agent freeze's")
    if scene == COCO_SCENE:
        out += coco_problems(root)
    dec = "docs/final-test/k23/decisions/K23-FA-EXAMPLES.json"         # Astra K23-FA-EXAMPLES-r1: the decision is bound by the freeze
    if dec not in (fz.get("code") or {}):
        out.append(f"{scene}: the agent freeze does not bind {dec} (the rendering decision must be fixed at the freeze)")
    return out


def wsl_path(win_path: str) -> str:
    """The WSL (/mnt/<drive>/...) form of a Windows path, so a launcher's WSL side runs in the SAME checkout its Windows preflight checked
    (Astra K23-TOOLING-G1 3). A path that is already POSIX is returned unchanged (on either platform)."""
    s = str(win_path)
    if s.startswith("/") and not s.startswith("//"):
        return s                                                  # already POSIX (the WSL side)
    drive = len(s) >= 3 and s[0].isalpha() and s[1] == ":" and s[2] in ("/", "\\")
    p = s if drive else os.path.abspath(s)
    if len(p) >= 2 and p[1] == ":":
        return "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")
    return p


#: The perception freeze records list ~/thor-data files by their WSL paths (/home/arash/...). On Windows the same files are read through
#: the WSL share, so the K23 entry points running on Windows (k23_run plan / score, the freeze writer) verify the SAME bytes instead of
#: failing on a path Windows cannot open (k23_ns.perception_problems joins it to the repo root there).
WSL_SHARE = "//wsl.localhost/Ubuntu-22.04"


def host_path(rel: str, base: str) -> str:
    if rel.startswith("/") and os.name == "nt":
        return WSL_SHARE + rel
    return rel if os.path.isabs(rel) else os.path.join(base, rel)


def perception_problems(scene: str, root: str | None = None) -> list:
    """k23_ns.perception_problems with the WSL-share mapping on Windows (identical on Linux): every listed file re-hashed."""
    base = C.paths(root)["root"]
    p = os.path.join(base, "sim", "kitchen_v3", scene, "frozen_weights.sha256")
    if not os.path.exists(p):
        return [f"{scene}: perception freeze record missing ({p})"]
    bad = []
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        sha, r = line.split(None, 1)
        r = r.strip()
        try:
            if C.sha256_file(host_path(r, base)) != sha:
                bad.append(r)
        except OSError:
            bad.append(r)
    return [f"{scene}: frozen perception changed / missing: {bad[:5]}"] if bad else []


def install_perception_reader() -> None:
    """In THIS process only: k23_ns.replication_freeze_status (called by common.guard_items) verifies the perception freeze through
    perception_problems above (the WSL-share mapping on Windows; the same check on Linux)."""
    if K.perception_problems is not perception_problems:
        K.perception_problems = perception_problems


def briefing_table(ns: str) -> dict:
    import treatment as T
    return T.briefing_hint_table(scene_of(ns), allow_unapproved=T.allow_draft_briefing(ns))


def briefing_hints(ns: str):
    import treatment as T
    return T.hints_fn(briefing_table(ns))
