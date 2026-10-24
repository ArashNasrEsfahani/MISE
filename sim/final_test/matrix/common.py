"""Final-test matrix: the small shared core every matrix module imports (stage D-spec; contract in SPEC.md, same folder).

Pure stdlib + `planners.ledger` / `planners.keys`. No simulator, no network, no API. Runs on Windows python (3.14) and in WSL
(~/thor-env, python 3.10), so nothing here uses syntax newer than 3.10.

What lives here (and nowhere else):
  * paths and namespaces (SPEC s2), the arm table and memory sources (SPEC s3);
  * ids: presentation id, cell ids, plan normalisation, the EXECUTION-UNIT key (prereg s7 C);
  * seeded orders: interleaved planning blocks (Phase B) and execution blocks (Phase C), from the master seed;
  * the append-only, fsync'd, resumable JSONL journal and the record envelope (`make_record`) with the gold / outcome firewall;
  * the test-item refusal guard (needs --allow-test AND a valid freeze.json AND families_approved.json);
  * `ScopedLedger`: the ceilings (core 30 USD until A24.1; whole-study cap 100 USD since A24.10 with an 80 USD cost gate; were A22 30 / 65, A18 25 / 55, A15 25 / 45) with a worst-case BLOCK hold, and the cost guard;
  * masking (prereg s11) and key-leak (prereg s15) checks; a per-namespace phase lock.

Selftest: ``python sim/_ftm_common_selftest.py`` (offline).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import re
import socket
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(SIM)
if SIM not in sys.path:
    sys.path.insert(0, SIM)

from planners.base import CostCeilingExceeded          # noqa: E402
from planners.keys import redact                       # noqa: E402
from planners.ledger import CostLedger                 # noqa: E402

SPEC_VERSION = "ftm-1"
MASTER_SEED = 20261010
SPLIT_SHA256 = "09c7da232244c4fbec20ff98b47a1a459efff1308f29481aa09961a621e2794f"


# ================================================================================================ paths (SPEC s2)
def paths(root: str | None = None) -> dict:
    r = root or ROOT
    d = os.path.join(r, "docs", "final-test")
    return {"root": r, "docs": d, "split": os.path.join(d, "split_manifest.json"), "freeze": os.path.join(d, "freeze.json"),
            "families": os.path.join(d, "families_approved.json"), "tolerances": os.path.join(d, "reset_tolerances.json"),
            "prereg": os.path.join(d, "preregistration.md"), "dataset": os.path.join(r, "sim", "dataset", "general.json"),
            "matrix_docs": os.path.join(d, "matrix"), "matrix_work": os.path.join(r, "sim", "final_test", "matrix", "work"),
            "snapshots": os.path.join(r, "sim", "final_test", "snapshots"), "fs_blocks": os.path.join(d, "matrix", "fs_blocks.json")}


#: A24.10: the Anthropic TOTAL CAP for the whole study (replaces A22's 65 USD) and the COST GATE that must hold before the first test call
#: (spend so far + the A21.G S2 remaining-cost projection <= the gate, which leaves CONTINGENCY inside the cap). A24.1-C (approved by Arash
#: 2026-10-14 ~11:40; docs/final-test/a24_1_record.json) re-derived the test-v1 ceilings (core 30, total 50) and the tripwire (0.040) below.
#: A37 (Arash 2026-10-16: normal API for the whole Kitchen 1 plan): the remaining Opus cells plan synchronously at the normal (2 x batch) price,
#: so the study cap 100 -> 180 USD (Arash's total budget for all kitchens), the test-v1 ceilings core 30 -> 60 / total 50 -> 100 and the
#: tripwire 0.040 -> 0.080 (reference 0.022 -> 0.044 = the same S2 projection at the sync price). COST_GATE_USD (the pre-first-call gate) unchanged.
ANTHROPIC_CAP_USD = 180.0                 # A37 (was 100.0, A24.10)
COST_GATE_USD = 80.0
CONTINGENCY_USD = ANTHROPIC_CAP_USD - COST_GATE_USD

#: namespace -> which split set it may touch, and its Anthropic ceilings (A15 for test-v1; this workflow's 1 USD for rehearsal).
NAMESPACES = {
    # A23: "archived_arms" = the ARCHIVED_ARMS (Sonnet / Haiku) this namespace has EXPLICITLY opted into. test-v1 never does (guarded, see
    # `archived_guard`); the two dev rehearsal namespaces keep their historical Haiku cells (their journals and the PF-safety evidence
    # were produced with them); a future namespace (for example "archived-sh-v1") opts in by listing arms here.
    "rehearsal-v1": {"items": "dev_pilot", "ceilings": {"core": 0.5, "total": 1.0}, "archived_arms": ("H0", "H+M")},      # W2: total stays 1.0; core 0.5 leaves room for a few non-core (Haiku) cells under the core reserve rule
    "rehearsal-v2": {"items": "dev_pilot", "ceilings": {"core": 0.0, "total": 0.5}, "archived_arms": ("H0", "H+M")},      # PF-safety-test (plan first, then simulate): dev items, Haiku only, 0.5 USD HARD cap for ALL its passes together
    "practice-v1": {"items": "practice_pool", "ceilings": {"core": 0.0, "total": 0.0}, "archived_arms": ()},     # Gemma only
    "test-v1": {"items": "test", "ceilings": {"core": 60.0, "total": 100.0}, "archived_arms": ()},  # A37 (Arash 2026-10-16: normal API for the whole Kitchen 1 plan): core 60 / total 100 = 2 x the A24.1-C 30 / 50 (ceil5 of the S3 batch projection 27.11 / 45.12) at the sync price; A23: no archived arm, ever
    # C4 (ft-v2, treatment.py): fresh practice under the package (A24 11, A25 11.3; Gemma only, approved briefing required) and a dev
    # rehearsal of the ft-v2 runner (the ONLY namespace where the draft briefing and the package-only STRICT mask are allowed).
    # test-v1 has no "treatment" key: treatment.TEST_V1_TREATMENT decides it (ft-v1 until A24.1 / the freeze).
    "practice-v2": {"items": "practice_pool", "ceilings": {"core": 0.0, "total": 0.0}, "archived_arms": (), "treatment": "ft-v2"},
    "rehearsal-v3": {"items": "dev_pilot", "ceilings": {"core": 0.5, "total": 1.0}, "archived_arms": (), "treatment": "ft-v2"},
    # ACC-RUNNER (2026-10-14): the acceptance gate of A24 item 7 / A25 / A32 (docs/final-test/acceptance_registration.json). Its items are the
    # REGISTERED ROSTER, not a split-manifest group (acc_common.acceptance_roster); guard_items refuses them in every other namespace (they are in no
    # split group). Ceilings = the registration's spend section: 16 USD for the namespace, 8 USD per attempt (acc_common.register_ledger_guard), plus the
    # A29 study-wide pre-submission check (acc_common.study_wide_problems) on every paid reservation and block hold.
    "acceptance-v2": {"items": "acceptance_roster", "ceilings": {"total": 16.0}, "attempt_ceiling": 8.0, "archived_arms": (),   # no A15 core reserve
                      "treatment": "ft-v2", "study_guard": True},
    # ACC-ATTEMPT2: attempt 2 = the ONE correction cycle (own journals / cache / ledger, same roster + order; 8 USD, 16 across both): acc_attempt2.py
    "acceptance-v2-a2": {"items": "acceptance_roster", "ceilings": {"total": 8.0}, "attempt_ceiling": 8.0, "archived_arms": (),
                         "treatment": "ft-v2", "study_guard": True, "attempt_of": "acceptance-v2", "attempt": 2},
    # FREEZE-GATES-FT2 (2026-10-15): the two owed freeze gates G-iso + 12.5 under the FINAL identity (not draft: approved briefing). Dev items, Gemma only, 0 USD.
    "freeze-gates-ft2": {"items": "dev_pilot", "ceilings": {"core": 0.0, "total": 0.0}, "archived_arms": (), "treatment": "ft-v2"},
}
import k23_ns as _k23                            # noqa: E402  K23: Kitchen 2/3 namespaces + A29 roster (Astra K23-NAMESPACES-r1)
NAMESPACES.update(_k23.K23_NAMESPACES)
REPLICATION_SCENES, REPLICATION_USD, A29_ROSTER = _k23.REPLICATION_SCENES, _k23.REPLICATION_USD, _k23.A29_ROSTER
ACCEPTANCE_NS, ACCEPTANCE_ROSTER = "acceptance-v2", "acceptance_roster"     # ACC-RUNNER: the roster resolver + spend guard live in acc_common.py
ACCEPTANCE_NS_A2 = "acceptance-v2-a2"; ACCEPTANCE_NAMESPACES = (ACCEPTANCE_NS, ACCEPTANCE_NS_A2)      # noqa: E702 -- ACC-ATTEMPT2


def _acc():
    import acc_common                                   # lazy: acc_common imports this module
    return acc_common
JOURNALS = ("practice", "capture", "plan", "exec", "outcome", "score", "events", "monitor")


#: K23 agent-side (Astra K23-AGENT-r1 1): the ONE scene of a namespace, from its NAMESPACES entry ('scene'), FloorPlan1 when
#: absent (every existing namespace is Kitchen 1). Callers pass it explicitly; no module global is mutated at runtime.
DEFAULT_SCENE = "FloorPlan1"


def ns_scene(namespace: str) -> str:
    return (NAMESPACES.get(namespace) or {}).get("scene", DEFAULT_SCENE)


def ns_check(namespace: str) -> dict:
    if namespace not in NAMESPACES:
        raise ValueError(f"unknown namespace {namespace!r}; use one of {', '.join(NAMESPACES)}")
    return NAMESPACES[namespace]


#: READY (Arash ruling 1b, 2026-10-14): attempt 1 restarts in FRESH storage. The logical namespace (NAMESPACES key, registration, seeded order
#: labels '<master>|acceptance-v2|...', events' namespace) stays "acceptance-v2"; the voided start's files stay in VOIDED_STORAGE (no journal write).
NS_STORAGE, VOIDED_STORAGE = {"acceptance-v2": "acceptance-v2-restart1"}, {"acceptance-v2": ("acceptance-v2",)}


def ns_storage(namespace: str) -> str:
    return NS_STORAGE.get(namespace, namespace) if ns_check(namespace) is not None else namespace


def namespace_of_dir(dirname: str):
    """Storage (or voided) dir basename -> its namespace, else None: every path -> namespace inference goes through here."""
    return next((n for n in NAMESPACES if NS_STORAGE.get(n, n) == dirname or dirname in VOIDED_STORAGE.get(n, ())), None)


def voided_storage_dir(path_dir: str) -> bool:
    d = os.path.abspath(path_dir)                       # THIS repo's voided dirs only (temporary test trees are not affected)
    return (os.path.normcase(os.path.dirname(d)) in {os.path.normcase(os.path.abspath(paths(None)[k])) for k in ("matrix_docs", "matrix_work")}
            and any(os.path.basename(d) in v and NS_STORAGE.get(n) != os.path.basename(d) for n, v in VOIDED_STORAGE.items()))


def ns_docs(namespace: str, root: str | None = None) -> str:
    return os.path.join(paths(root)["matrix_docs"], ns_storage(namespace))


def ns_work(namespace: str, root: str | None = None) -> str:
    return os.path.join(paths(root)["matrix_work"], ns_storage(namespace))


def journal_path(namespace: str, kind: str, root: str | None = None) -> str:
    if kind not in JOURNALS:
        raise ValueError(f"unknown journal {kind!r}")
    name = "outcome.sealed.jsonl" if kind == "outcome" else f"{kind}.jsonl"
    return os.path.join(ns_docs(namespace, root), name)


def snapshot_dir(source: str, root: str | None = None) -> str:
    """Memory source -> its frozen snapshot directory. 'S*' is o0k10 (prereg 6.3)."""
    src = "o0k10" if source == "S*" else source
    m = re.fullmatch(r"o([0-2])k(0|1|3|5|10)", src)
    if not m:
        raise ValueError(f"not a snapshot source: {source!r}")
    return os.path.join(paths(root)["snapshots"], f"o{m.group(1)}", f"k{m.group(2)}")


# ================================================================================================ hashing
SNAP_FILES = ("manifest.json", "ltm.db", "fragments.json")             # the files freeze.py hashes per snapshot


def canon(x) -> str:
    """The repo's canonical JSON (= step5_eval.canon): sorted keys, no ASCII escaping, compact."""
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_json(x) -> str:
    return sha256_text(canon(x))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dir_sha(path: str) -> str:
    """THE one sha of a snapshot directory tree (capture, execute, score and the freeze checks all use this function): sorted
    [relative posix path, file sha] rows, sqlite side files (-journal / -wal / -shm / .lock) excluded. 'absent' = no such directory
    or no file in it (callers decide whether that is fatal)."""
    if not os.path.isdir(path):
        return "absent"
    rows = []
    for dp, dn, fn in os.walk(path):
        dn.sort()
        for f in sorted(fn):
            if f.endswith(("-journal", "-wal", "-shm", ".lock")):
                continue
            full = os.path.join(dp, f)
            rows.append([os.path.relpath(full, path).replace(os.sep, "/"), sha256_file(full)])
    return sha256_json(sorted(rows)) if rows else "absent"


def self_sha(obj: dict) -> str:
    """sha256 of a manifest without its own 'sha256' field (split-manifest convention; freeze.json uses it too)."""
    return sha256_json({k: v for k, v in obj.items() if k != "sha256"})


#: A26 item 4 (K-B2b 2026-10-13): the EFFECTIVE inclusion manifest of the successor dataset. The frozen split manifest stays the archived
#: original (its sha is still verified); every consumer gets the split with the archived ids removed (no reshuffle, no regrouping).
INCLUSION_NAME = "inclusion_manifest_a26.json"


def apply_inclusion(m: dict, root: str | None = None) -> dict:
    """The split manifest with the A26 archived ids removed from every id list and family, when the inclusion manifest exists and was
    written for THIS split (its base self-sha equals the manifest's). Records ``inclusion`` {path, sha256, dataset_sha256, archived}.
    A mismatching inclusion manifest is an error (never silently ignored). A root without its own copy (a temporary test tree that
    copied the frozen split) uses the study's copy, because the inclusion manifest is keyed to the split, not to a directory."""
    p = os.path.join(paths(root)["docs"], INCLUSION_NAME)
    if not os.path.exists(p):
        p = os.path.join(paths(None)["docs"], INCLUSION_NAME)
    if not os.path.exists(p):
        return m
    with open(p, encoding="utf-8") as f:
        inc = json.load(f)
    if (inc.get("base_split_manifest") or {}).get("self_sha256") != m.get("sha256"):
        raise RuntimeError("the A26 inclusion manifest was not written for this split manifest")
    arch = set(inc.get("archived") or [])
    out = {}
    for k, v in m.items():
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            out[k] = [x for x in v if x not in arch]
        elif k == "families":
            out[k] = [dict(fam, items=[x for x in fam["items"] if x not in arch]) for fam in v]
        else:
            out[k] = v
    out["inclusion"] = {"path": "docs/final-test/" + INCLUSION_NAME, "sha256": sha256_file(p),
                        "dataset_sha256": (inc.get("dataset") or {}).get("sha256"), "archived": sorted(arch)}
    return out


def load_manifest(root: str | None = None, verify: bool = True, effective: bool = True) -> dict:
    with open(paths(root)["split"], encoding="utf-8") as f:
        m = json.load(f)
    if verify and (m.get("sha256") != SPLIT_SHA256 or self_sha(m) != SPLIT_SHA256):
        raise RuntimeError("split manifest sha mismatch: the frozen split changed (an amendment is needed)")
    return apply_inclusion(m, root) if effective else m


# ================================================================================================ REPEAT / relocation sets (per namespace)
DEV_SUBSET_N = 4        # rehearsal-v1: a DECLARED dev REPEAT set and a dev relocation set, so W2 can exercise the REPEAT, curve, Codex,
                        # three-run and relocation paths (resume included) on dev items only


is_replication, a29_files, a29_roster = _k23.is_replication, _k23.a29_files, _k23.a29_roster


def namespace_items(namespace: str, manifest: dict, root: str | None = None) -> list:
    """The items of a namespace: a split-manifest group, the acceptance roster, or (K23) the A29 roster."""
    key = ns_check(namespace)["items"]
    if key == ACCEPTANCE_ROSTER and key not in manifest:
        return _acc().acceptance_roster(root)
    if key == A29_ROSTER:
        return a29_roster(namespace, root)
    return list(manifest[key])


def _ns_item_list(manifest: dict, namespace: str) -> list:
    return namespace_items(namespace, manifest, None)


def repeat_items(namespace: str, manifest: dict) -> list:
    """The REPEAT items of a namespace. test-v1: the split manifest's `repeat_subset`. Any other namespace: the manifest's subset
    restricted to the namespace's own items if it names any (tests), else the declared deterministic dev subset (DEV_SUBSET_N items,
    seeded). practice-v1 has none. acceptance-v2 has none (capture 1 / exec 1 / r 0, registration world section)."""
    if namespace in ACCEPTANCE_NAMESPACES:                    # ACC-ATTEMPT2: attempt 2 runs the same canon-only design
        return []
    if is_replication(namespace):                             # K23: canonical world, r=0, one execution (no REPEAT)
        return []
    items = _ns_item_list(manifest, namespace)
    inside = [i for i in (manifest.get("repeat_subset") or ()) if i in set(items)]
    if namespace == "test-v1" or inside:
        return sorted(inside)
    if namespace == "practice-v1":
        return []
    return sorted(seeded_order(items, "dev-repeat", namespace)[:DEV_SUBSET_N])


def reloc_items(namespace: str, manifest: dict) -> list:
    """The relocation-probe items (same rule as repeat_items; the dev subset uses its own seed label). acceptance-v2: none (canon only)."""
    if namespace in ACCEPTANCE_NAMESPACES:                    # ACC-ATTEMPT2: attempt 2 runs the same canon-only design
        return []
    if is_replication(namespace):                             # K23: no relocation probe
        return []
    items = _ns_item_list(manifest, namespace)
    inside = [i for i in (manifest.get("relocation_probe") or ()) if i in set(items)]
    if namespace == "test-v1" or inside:
        return sorted(inside)
    if namespace == "practice-v1":
        return []
    return sorted(seeded_order(items, "dev-reloc", namespace)[:DEV_SUBSET_N])


# ================================================================================================ arms (SPEC s3)
PLANNER_SPEC = {"G": "gemma", "S": "sonnet", "O": "opus", "H": "haiku"}       # -> planners.make_planner spec
PAID = ("S", "O", "H")
#: A23: the core four (Gemma and Opus, with and without S* memory). Sonnet (S0 / S+M) and Haiku (H0 / H+M) are ARCHIVED, not deleted.
CORE_ARMS = ("G0", "G+M", "O0", "O+M")
CODEX_MODELS = ("gpt-6-sol", "gpt-6-astra", "gpt-6-luna", "gpt-5.6-terra")
#: A18: gpt-6-sol is a MAIN executed comparator (arms X0 / X+M, in the main and REPEAT blocks, never cut); the other three stay exploratory (X-<m>0 / X-<m>+M).
CODEX_MAIN = "gpt-6-sol"
CODEX_EXPLORATORY = tuple(m for m in CODEX_MODELS if m != CODEX_MAIN)
X_MAIN_ARMS = ("X0", "X+M")
#: every arm of the main matrix (r = 0); A18 added X0 / X+M. A27 (draft): only the core four (REPEAT_ARMS) also get the REPEAT block (r = 1, 2).
#: A23: 4 core + 2 FS control + 2 gpt-6-sol = 8 main arms.
MAIN_ARMS = ("G0", "G+M", "O0", "O+M", "G+FS", "O+FS") + X_MAIN_ARMS
BLOCKS = ("main", "repeat", "curve", "reloc", "codex")

# ================================================================================================ A27 run-time plan (DRAFT, PENDING ARASH SIGN-OFF AT THE FREEZE)
#: Arash 2026-10-13: "Accept astra's vision. it's logical." (docs/final-test/_astra/runtime-cuts.txt; prereg A27). The single source of these
#: numbers for plan.py, execute.py, capture.py and final_test_callplan.py:
#:  * REPEAT planner calls r = 1, 2 only for the core four (G0, G+M, O0, O+M); G+FS, O+FS, X0, X+M get r = 0 (the main matrix) only;
#:  * on REPEAT presentations every distinct execution unit that carries a core-four cell (or a /p replay of a core +M cell) of block main or
#:    repeat runs EXEC_RUNS_REPEAT = 2 times (was 3); every other unit runs once;
#:  * the learning curve keeps practice order 0 only: K = 1, 3, 5 (+ the main K = 10 = S*, shared); orders 1 and 2 are dropped;
#:  * the exploratory Codex models (X-<m>0 / X-<m>+M) are PLAN-ONLY (A25 item 3): planned (r = 0, 1, 2), scored as planning diagnostics
#:    (validity, PC, TDS), never executed;
#:  * unchanged: the 8 main arms x every test presentation at r = 0, the relocation block, the P0 / P+M reference controls, the exact unit dedup.
RUNTIME_PLAN = "A27-draft"
REPEAT_ARMS = CORE_ARMS
EXEC_RUNS_REPEAT = 2
CURVE_SNAPSHOTS = ((0, 1), (0, 3), (0, 5))
PLAN_ONLY_PREFIX = "X-"


def plan_only(arm: str) -> bool:
    """A27 / A25 item 3: True for an arm that is planned but never executed (the exploratory Codex models X-<m>0 / X-<m>+M)."""
    return str(arm or "").startswith(PLAN_ONLY_PREFIX)


def repeat_exec_arm(arm: str) -> bool:
    """A27: does a main / repeat cell of `arm` on a REPEAT presentation make its unit run EXEC_RUNS_REPEAT times? The core four, and the
    /p replay controls of the core +M arms (their replayed arm is in REPEAT_ARMS: "align the replay controls with the repeat policy")."""
    if arm in REPEAT_ARMS:
        return True
    info = ARMS.get(arm) or {}
    p = str(info.get("planner") or "")
    return p.startswith("replay:") and p.split(":", 1)[1] in REPEAT_ARMS


def _arm(planner, prompt, chooser):
    return {"planner": planner, "prompt": prompt, "chooser": chooser}


#: A23 ARCHIVED_ARMS: Sonnet and Haiku, kept (definitions, providers PLANNER_SPEC / PAID, prices in the callplan / plan, code paths) so a
#: later study can test them. They are NOT in ARMS, MAIN_ARMS or CORE_ARMS; `arm_info` / `scope_of` refuse them unless the namespace lists
#: them in its "archived_arms" (explicit opt-in; test-v1 can never, see `archived_guard`).
ARCHIVED_ARMS = {"S0": _arm("S", "none", "prior"), "S+M": _arm("S", "S*", "S*"),
                 "H0": _arm("H", "none", "prior"), "H+M": _arm("H", "S*", "S*")}
ARCHIVED_PLANNERS = ("S", "H")
ARCHIVED_ARM_NAMES = tuple(ARCHIVED_ARMS)
#: Hypotheses withdrawn by A23 (IDs kept for traceability; never computed, never in the Holm family).
WITHDRAWN_HYPOTHESES = ("H2b", "H3b", "H3c")

ARMS = {"G0": _arm("G", "none", "prior"), "G+M": _arm("G", "S*", "S*"),
        "O0": _arm("O", "none", "prior"), "O+M": _arm("O", "S*", "S*"),
        "G+FS": _arm("G", "FS", "S*"), "O+FS": _arm("O", "FS", "S*"),
        "P0": _arm("ref", "none", "prior"), "P+M": _arm("ref", "none", "S*"),              # diagnostics: reference plans, no call
        "G+M/p": _arm("replay:G+M", "S*", "prior"), "O+M/p": _arm("replay:O+M", "S*", "prior"),   # cached plans, no call
        "E0": _arm("empty", "none", "prior")}      # FREEZE-GATES-FT2: gate 12.5 (b) empty-plan control (execute.py --p-arms E0), no call
ARMS["X0"] = _arm(f"X:{CODEX_MAIN}", "none", "prior")
ARMS["X+M"] = _arm(f"X:{CODEX_MAIN}", "S*", "S*")
for _m in CODEX_EXPLORATORY:
    ARMS[f"X-{_m}0"] = _arm(f"X:{_m}", "none", "prior")
    ARMS[f"X-{_m}+M"] = _arm(f"X:{_m}", "S*", "S*")


def archived_arms_of(namespace: str | None) -> tuple:
    """The archived arms a namespace has explicitly opted into (A23); () for an unknown / absent namespace and for test-v1."""
    if namespace is None or namespace not in NAMESPACES:
        return ()
    return tuple(NAMESPACES[namespace].get("archived_arms") or ())


def archived_guard(namespace: str, arms=None) -> None:
    """A23 guard: raises ValueError if test-v1 opts into ANY archived arm, if a namespace opts into an unknown one, or if `arms`
    (arm names a plan / callplan / roster is about to use in `namespace`) include an archived arm the namespace did not opt into."""
    ns_check(namespace)
    opted = tuple(NAMESPACES[namespace].get("archived_arms") or ())
    bad = [a for a in opted if a not in ARCHIVED_ARMS]
    if bad:
        raise ValueError(f"namespace {namespace!r} opts into unknown archived arms {bad}")
    if namespace == "test-v1" and opted:
        raise ValueError(f"A23: archived arms {list(opted)} cannot enter test-v1 (Sonnet / Haiku are archived)")
    out = [a for a in (arms or ()) if a in ARCHIVED_ARMS and a not in opted]
    if out:
        raise ValueError(f"A23: archived arms {out} are not enabled in namespace {namespace!r}")


def main_arms_for(namespace: str) -> tuple:
    """The main-matrix roster of a namespace: MAIN_ARMS plus the archived arms it opted into (test-v1: exactly MAIN_ARMS)."""
    archived_guard(namespace)
    if is_replication(namespace):                             # K23 (A29 item 1): the four core arms only; P0 / P+M at execution
        return tuple(CORE_ARMS)
    return tuple(MAIN_ARMS) + archived_arms_of(namespace)


def arm_info(arm: str, namespace: str | None = None) -> dict:
    """Arm name -> {planner, prompt (memory source of the prompt block), chooser ('prior' or a snapshot source)}.
    Learning-curve arms are 'G+M@o1k3' / 'O+M@o0k5': prompt AND chooser use that snapshot (prereg 6.3 E4).
    An ARCHIVED arm (A23) is returned only when `namespace` opted into it; otherwise ValueError (it can never enter test-v1)."""
    if namespace is not None and is_replication(namespace) and arm not in tuple(CORE_ARMS) + ("P0", "P+M"):
        raise ValueError(f"arm {arm!r} is not in the K2/K3 design (G0, G+M, O0, O+M; P0, P+M at execution)")
    m = re.fullmatch(r"([GO])\+M@(o[0-2]k(?:1|3|5|10))", arm)
    if m:
        return _arm(m.group(1), m.group(2), m.group(2))
    if arm in ARCHIVED_ARMS:
        if arm not in archived_arms_of(namespace):
            raise ValueError(f"arm {arm!r} is ARCHIVED (A23: Sonnet / Haiku); namespace {namespace!r} has not opted into it")
        return dict(ARCHIVED_ARMS[arm])
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    return dict(ARMS[arm])


def scope_of(arm: str, block: str, namespace: str | None = None) -> str:
    """Ledger scope (A15): 'core' = the four core arms (A23) in the main matrix and their REPEAT repeats; 'other' = every other
    paid (Anthropic) call; 'free' = no Anthropic spend (Gemma, Codex, reference / replay)."""
    if block not in BLOCKS:
        raise ValueError(f"unknown block {block!r}")
    if arm_info(arm, namespace)["planner"] not in PAID:
        return "free"
    return "core" if (arm in CORE_ARMS and block in ("main", "repeat")) else "other"


#: Design groups that Arash may CUT (prereg 3.1 / A15: "exploratory arms are cut first"). A cut is a recorded `exploratory_cut` event
#: (never automatic); cut cells are not run, not executed and not scored (they are NOT scored 0). Core, Gemma and the FS arms
#: (H4 is in the Holm family) cannot be cut: a shortfall there is an incomplete study (`stopping_rule`, score --partial).
#: The CUT ORDER is the registered one (prereg 3.1: R, X-*, H, learning-curve seeds 1-2, the relocation probe; R = OpenRouter was dropped,
#: A6). `curve12` = the learning-curve arms of practice orders 1 and 2 only: the order-0 curve (K = 1, 3, 5 beside S*) is never cut.
#: A23: the "haiku" group left the order with Haiku (archived arms are not cuttable groups).
CUT_ORDER = ("codex", "curve12", "reloc")
CUTTABLE_GROUPS = CUT_ORDER


def cut_group(arm: str, pres: str | None = None) -> str | None:
    """Which cuttable group a cell belongs to (None = never cuttable): Codex arms, learning-curve arms, and every cell
    of the relocated world. (A23: archived Haiku arms are not a cuttable group.)"""
    if arm.startswith("X-"):
        return "codex"
    if "@" in arm:
        return "curve12" if re.search(r"@o[12]k", arm) else None          # the order-0 curve arms (@o0k*) are never cut
    if pres is not None and pres.startswith("reloc:"):
        return "reloc"
    return None


def cut_problem(current, new) -> str | None:
    """Why recording the cut of `new` (groups) on top of `current` would break the registered order (None = fine). Core, Gemma and
    FS arms are not groups at all, so they can never be named; the cut must grow as a PREFIX of CUT_ORDER."""
    new = list(new or ())
    bad = [g for g in new if g not in CUT_ORDER]
    if bad:
        return f"not a cuttable group: {bad} (cuttable, in cut order: {list(CUT_ORDER)}; core, gemma and fs arms are never cut)"
    union = set(current or ()) | set(new)
    k = sum(1 for _ in __import__("itertools").takewhile(lambda g: g in union, CUT_ORDER))
    if len(union) != k:
        return f"cuts must follow the registered order {list(CUT_ORDER)}; {sorted(union)} skips a group"
    return None


def cut_groups(event_rows) -> set:
    """Union of the groups named by `exploratory_cut` events (events.jsonl rows); unknown names are ignored."""
    out = set()
    for r in event_rows or ():
        if r.get("event") == "exploratory_cut":
            out |= {g for g in (r.get("groups") or ()) if g in CUTTABLE_GROUPS}
    return out


# ================================================================================================ ids and units (SPEC s4)
WORLDS = ("canon", "reloc")


def pres_id(item: str, lang: str, world: str = "canon") -> str:
    if lang not in ("en", "fa") or world not in WORLDS or not item or ":" in item:
        raise ValueError(f"bad presentation ({item!r}, {lang!r}, {world!r})")
    return f"{world}:{item}:{lang}"


def parse_pres(pid: str) -> dict:
    world, item, lang = pid.split(":")
    pres_id(item, lang, world)
    return {"world": world, "item": item, "lang": lang}


def cell_id(namespace: str, phase: str, *parts) -> str:
    ns_check(namespace)
    return "|".join([namespace, phase] + [str(p) for p in parts])


def norm_plan(steps) -> list:
    """Order kept, every execution-relevant argument kept (prereg s7 C): [[action, object, to, arg], ...]."""
    out = []
    for s in steps or ():
        if isinstance(s, (list, tuple)):
            a, o, t, g = (list(s) + [None] * 4)[:4]
        else:
            a, o, t, g = s.get("action"), s.get("object"), s.get("to"), s.get("arg")
        out.append([a, o, t, g])
    return out


#: Frozen action caps (values re-recorded in freeze.json; step5_live EP_PRIM_CAP / LiveExecutor timeout).
CAPS = {"max_steps": 12, "retries_per_step": 3, "ep_prim_cap": 3000, "episode_wall_s": 900}
#: The honest executor takes NO simulator seed (step5_live always builds its requests with world_seed = 0, and the AI2-THOR reset is
#: deterministic), so the unit key says so instead of naming a per-presentation seed that nothing consumed (M-fix2). `world_seed_of`
#: in execute.py returns 0 for the same reason; capture and execute record the same value.
SEED_POLICY = "none (the honest executor takes no simulator seed; world_seed is a recorded label, always 0)"


def execution_unit_key(*, plan, pres: str, chooser: str, snapshot_sha: str | None, restored_state: str, summary_sha: str,
                       caps: dict | None = None, seed_policy: str = SEED_POLICY, exec_policy: str | None = None) -> str:
    """Arms whose units are equal share ONE execution (prereg s7 C). chooser = 'prior' or a snapshot source; a prior chooser
    carries no snapshot. restored_state = the world id + setup sha the restore produces (not the post-run hash)."""
    parse_pres(pres)
    if chooser == "prior":
        snapshot_sha = None
    elif not snapshot_sha:
        raise ValueError("an LTM chooser needs the snapshot sha")
    doc = {"plan": norm_plan(plan), "pres": pres, "chooser": "prior" if chooser == "prior" else "ltm",
           "snapshot": snapshot_sha, "restored": restored_state, "summary": summary_sha,
           "caps": caps or CAPS, "seed_policy": seed_policy}
    if exec_policy is not None:                    # C4 / ft-v2 only: hook B (briefing-guided search); absent = the frozen ftm-unit-1 bytes
        doc["exec_policy"] = str(exec_policy)
    return sha256_json(doc)[:32]


#: A21 D -- the deduplication key, frozen. Two plan cells share ONE execution only if every field below is equal. The list is hashed into
#: freeze.json, and `unit_key_selfcheck()` recomputes a fixed sample key, so any change to the key function trips the selftest and the freeze.
UNIT_KEY_SPEC = {
    "version": "ftm-unit-1",
    "fields": ["plan (normalised: action, object, to, arg, order kept)", "pres (item, language, world)", "chooser (prior | ltm)",
               "snapshot sha (None for a prior chooser)", "restored state (world id + setup sha)", "post-sweep summary sha of the Phase A capture",
               "caps", "seed policy"],
    "sharing": "arms with equal keys share one simulator run; each plan cell keeps its own plan row and is linked to the unit by the exec "
               "record's `cells` list and by the score row's `unit`",
    "dependence": "shared outcomes are copies of one realisation, never independent replications: analysis clusters on the 42 families "
                  "(a unit sits inside one presentation, a presentation inside one item, an item inside one family) and reports the shared share",
    "costs": "planner USD is charged to the arm whose call made the plan; the wall time of a shared execution is split equally among the scored "
             "cells that share it; simulator time is otherwise reported as a total",
}
#: C4 / ft-v2: the same key with ONE extra field when an executor policy is set (treatment.exec_policy: "briefing:<hint-table sha16>"
#: for +M, curve and P+M units). Hook-on and hook-off units never share; with no policy the key is byte-identical to ftm-unit-1.
UNIT_KEY_SPEC_FT2 = dict(UNIT_KEY_SPEC, version="ftm-unit-1+exec_policy",
                         fields=UNIT_KEY_SPEC["fields"] + ["exec_policy (ft-v2 only, present only when set: briefing:<hint table sha16>)"])
UNIT_KEY_GOLDEN_INPUT = {"plan": [{"action": "open", "object": "Fridge", "to": None, "arg": None}], "pres": "canon:g-l1-sample:en",
                         "chooser": "o0k10", "snapshot_sha": "0" * 64, "restored_state": "canon+0000000000000000", "summary_sha": "1" * 64}


def unit_key_selfcheck() -> dict:
    """-> {'key', 'golden', 'ok'}: the key of a fixed sample input against the value frozen with A21 (UNIT_KEY_GOLDEN)."""
    k = execution_unit_key(**UNIT_KEY_GOLDEN_INPUT)
    return {"key": k, "golden": UNIT_KEY_GOLDEN, "ok": k == UNIT_KEY_GOLDEN}


UNIT_KEY_GOLDEN = "d4f6dcbd3511fed50e4d63992f3dc62a"

#: A21 A -- the blocked-run rule: which statuses are infrastructure (not caused by the agent or the plan), and how each is scored.
RUN_CLASSES = {
    "plan_infra": "infra_planner", "exec_crash": "infra_exec", "exec_blocked_state": "infra_exec",
    "plan_rejected": "agent", "not_executed": "agent", "ok": "ok", "plan_ceiling": "budget", "missing_exec": "missing", "missing_outcome": "missing",
}
BLOCKED_RULE = {
    "version": "ftm-blocked-1",
    "infra": "a run that never reached the agent's plan through no fault of the plan: restored world differs from Phase A (blocked_state), simulator crash or "
             "hang (the hang guard kills it), planner transport failure after 3 transport retries",
    "agent": "everything after the first plan step started: wrong plan, refusal, invalid plan, step failures, the 900 s episode wall cap, the 3000 primitive cap",
    "retries": "execution: attempts 0, 1, 2, each from a fresh restore; planner: transport retries 3, then re-queued in up to 2 later sessions",
    "scoring": "an infra run that is still missing after the retries counts as failure (SR 0) in the primary analysis and stays in every denominator (section 13)",
    "sensitivity": "reported next to every contrast: the presentations where EITHER arm of the contrast has an infra_planner or infra_exec scored cell are dropped from BOTH arms",
    "flag": "an arm with more than 5 % infra cells in the scored run, or a contrast whose primary and sensitivity wordings differ, is marked 'infrastructure-sensitive' in the report",
    "g_iso": "a blocked run is never a G-iso pass (a pass needs every presentation measured)",
    "precedence": "one cell gets one class, first match wins: (1) INVALID episode (honesty counter) -> section 13, not a failure class; (2) planner transport failure after the "
                  "retries -> infra_planner; (3) a plan the validator rejected (no JSON, schema, unknown action or type, refusal, truncation) -> agent, nothing executed; "
                  "(4) restored world not equal to Phase A, simulator crash or hang at ANY time, including after the first step (the run is discarded and re-run from a fresh "
                  "restore) -> infra_exec once the 3 attempts are used; (5) step failures, the 900 s wall cap, the 3000 primitive cap -> agent (the episode ends with its own result)",
    "hang": "the 60 s detector hang guard (exit 75) kills the process; the interrupted attempt is repeated without using one of the 3 attempts; a unit that hangs on 3 successive launches "
            "is declared a terminal crash by Arash with a ledger note (it then counts as an infra_exec failure)",
}
INFRA_FLAG_RATE = 0.05


def run_class(failure_class: str) -> str:
    return RUN_CLASSES.get(failure_class, "missing")


# ================================================================================================ seeded orders (SPEC s5)
def rng(*labels) -> random.Random:
    """A Random seeded from the master seed + labels via sha256 (stable across python versions and machines)."""
    return random.Random(int(sha256_text("|".join([str(MASTER_SEED)] + [str(x) for x in labels]))[:16], 16))


def seeded_order(seq, *labels) -> list:
    out = sorted(seq)
    rng(*labels).shuffle(out)
    return out


def plan_blocks(pres_ids, cells_of, *labels) -> list:
    """Phase B: [(pres, [cell, ...]), ...] - presentations in a seeded order, the cells (arm, r) of one presentation together
    in a seeded order (arms interleaved over time; a stop between blocks leaves every arm balanced)."""
    out = []
    for p in seeded_order(pres_ids, "plan", *labels):
        cs = sorted(cells_of(p), key=canon)
        rng("plan-in", p, *labels).shuffle(cs)
        out.append((p, cs))
    return out


def exec_blocks(units_by_pres: dict, *labels) -> list:
    """Phase C: a seeded permutation of presentation blocks, a seeded unit order inside each block."""
    out = []
    for p in seeded_order(units_by_pres, "exec", *labels):
        us = sorted(set(units_by_pres[p]))
        rng("exec-in", p, *labels).shuffle(us)
        out.append((p, us))
    return out


# ================================================================================================ journal + records (SPEC s6)
class JournalCorrupt(RuntimeError):
    pass


class JournalRefused(RuntimeError):
    """ACC-ATTEMPT2: a write to a CLOSED acceptance attempt, or one its armed session's lifecycle refuses; nothing was written."""


class Journal:
    """Append-only JSONL, one fsync'd line per record. A torn LAST line (crash mid-write) is moved to <path>.torn on the next
    append and ignored on read; a torn line anywhere else is corruption (fail fast)."""

    def __init__(self, path: str):
        self.path = path

    def rows(self) -> list:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "rb") as f:
            data = f.read()
        lines = data.split(b"\n")
        out = []
        for i, ln in enumerate(lines):
            if not ln.strip():
                continue
            try:
                out.append(json.loads(ln.decode("utf-8")))
            except ValueError:
                if i == len(lines) - 1:              # torn tail: no newline after it
                    continue
                raise JournalCorrupt(f"{self.path}: unparsable line {i + 1}")
        return out

    def keys(self, field: str = "key", where=None) -> set:
        return {r.get(field) for r in self.rows() if where is None or where(r)}

    def latest(self, field: str = "key") -> dict:
        d = {}
        for r in self.rows():
            d[r.get(field)] = r
        return d

    def _repair_tail(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, "rb+") as f:
            data = f.read()
            if not data or data.endswith(b"\n"):
                return
            cut = data.rfind(b"\n") + 1
            try:                                       # a COMPLETE json line that only lacks its newline is kept (rows() already read it)
                json.loads(data[cut:].decode("utf-8"))
                f.seek(0, 2)
                f.write(b"\n")
                return
            except ValueError:
                pass
            with open(self.path + ".torn", "ab") as t:
                t.write(data[cut:] + b"\n")
            f.truncate(cut)

    def append(self, rec: dict) -> dict:
        if voided_storage_dir(d_ := os.path.dirname(os.path.abspath(self.path))):         # READY: never written to (Arash ruling 1b)
            raise JournalRefused(f"{self.path}: voided storage (attempt 1's infrastructure-stopped start): no journal write")
        if namespace_of_dir(os.path.basename(d_)) in ACCEPTANCE_NAMESPACES and (bad := _acc().journal_write_problems(self.path)):
            raise JournalRefused(bad[0])                 # ACC-ATTEMPT2: a closed attempt / a refusing session lifecycle writes nothing
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._repair_tail()
        line = json.dumps(rec, ensure_ascii=False, default=str)
        if "\n" in line:
            raise ValueError("record serialises with a newline")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        return rec


GOLD_KEYS = frozenset({"goal", "gold", "setup", "reference_plan", "ref_plan", "goal_predicates"})
OUTCOME_KEYS = frozenset({"goal_ok", "exec_ok", "sr", "strict_sr", "success", "step_ok", "steps_ok", "admission"})

#: kind -> required fields. 'forbid' = keys that may appear NOWHERE inside the record (recursive).
SCHEMAS = {
    "practice": {"req": ("key", "namespace", "order", "attempt", "item", "lang", "level", "status", "snapshot_after"),
                 "forbid": frozenset()},                    # practice = training on non-test items: outcomes allowed
    "capture": {"req": ("key", "namespace", "pres", "item", "lang", "level", "world", "capture_run", "instruction", "holding",
                        "summary", "summary_sha", "blocks", "requests", "state_ref", "world_hash", "audit"),
                "forbid": GOLD_KEYS | OUTCOME_KEYS},
    "plan": {"req": ("key", "namespace", "arm", "block", "pres", "r", "session", "request_sha", "status", "plan", "first_plan",
                     "first_ok", "repaired", "calls", "cached_calls", "tokens_in", "tokens_out", "cost_usd", "model", "scope"),
             "forbid": GOLD_KEYS | OUTCOME_KEYS},
    "exec": {"req": ("key", "namespace", "unit", "run", "attempt", "pres", "status", "state_check", "audit", "prims", "wall_s",
                     "snapshot_sha_before", "snapshot_sha_after"),
             "forbid": GOLD_KEYS | OUTCOME_KEYS | {"steps"}},
    "outcome": {"req": ("key", "namespace", "unit", "run", "goal_ok", "exec_ok", "steps"), "forbid": frozenset()},
    "score": {"req": ("key", "namespace", "arm", "pres", "r", "run", "unit", "sr", "strict_sr", "valid", "infra"),
              "forbid": frozenset()},
    "events": {"req": ("event", "phase"), "forbid": GOLD_KEYS | OUTCOME_KEYS},
    "monitor": {"req": ("tick", "counters"), "forbid": GOLD_KEYS | OUTCOME_KEYS},
}
PLAN_STATUS = ("ok", "rejected", "infra", "ceiling")
EXEC_STATUS = ("done", "not_executed", "blocked_state", "crash", "invalid")


def forbidden_keys(obj, forbid, path="$") -> list:
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in forbid:
                out.append(f"{path}.{k}")
            out += forbidden_keys(v, forbid, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out += forbidden_keys(v, forbid, f"{path}[{i}]")
    return out


def make_record(kind: str, **fields) -> dict:
    """Envelope + validation: required fields present, status values legal, and no gold / outcome key anywhere in a
    planner-side or integrity record (the firewall that keeps outcomes sealed and gold out of planning)."""
    if kind not in SCHEMAS:
        raise ValueError(f"unknown record kind {kind!r}")
    sc = SCHEMAS[kind]
    miss = [k for k in sc["req"] if k not in fields]
    if miss:
        raise ValueError(f"{kind} record misses {miss}")
    if kind == "plan" and fields["status"] not in PLAN_STATUS:
        raise ValueError(f"plan status {fields['status']!r}")
    if kind == "exec" and fields["status"] not in EXEC_STATUS:
        raise ValueError(f"exec status {fields['status']!r}")
    bad = forbidden_keys(fields, sc["forbid"])
    if bad:
        raise ValueError(f"{kind} record carries gold / outcome keys: {bad[:5]}")
    return {"schema": f"ftm.{kind}/1", "spec": SPEC_VERSION, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}


# ================================================================================================ test-item guard (SPEC s7)
class TestItemRefused(RuntimeError):
    pass


def families_status(root: str | None = None) -> dict:
    p = paths(root)["families"]
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        return {"ok": False, "reasons": [f"families_approved.json unreadable: {type(e).__name__}"]}
    ok = d.get("approved_by") == "Arash" and bool(d.get("date"))
    return {"ok": ok, "reasons": [] if ok else ["families_approved.json does not record Arash's approval"]}


def freeze_status(root: str | None = None, check_code: bool = True) -> dict:
    """freeze.json is valid iff: schema, self-sha, every gate passed, Arash's freeze check recorded, the split and
    preregistration shas equal the files on disk, and (check_code) every hashed code file is unchanged."""
    P = paths(root)
    reasons = []
    try:
        with open(P["freeze"], encoding="utf-8") as f:
            fz = json.load(f)
    except (OSError, ValueError) as e:
        return {"ok": False, "reasons": [f"freeze.json missing or unreadable: {type(e).__name__}"]}
    if fz.get("schema") != "final-test.freeze/1":
        reasons.append("freeze.json schema")
    if fz.get("sha256") != self_sha(fz):
        reasons.append("freeze.json self-sha mismatch (edited after writing)")
    if fz.get("freeze_status") is not None:            # TREAT-FREEZE: freeze.py --record-treatment marks a stale file "treatment-only"
        reasons.append(f"freeze.json status {fz.get('freeze_status')!r}: not a validated freeze")
    gates = fz.get("gates") or {}
    failed = sorted(g for g, v in gates.items() if not (isinstance(v, dict) and v.get("pass") is True))
    if not gates or failed:
        reasons.append(f"gates not all passed: {failed or 'none recorded'}")
    if not ((fz.get("arash_check") or {}).get("by") == "Arash" and (fz.get("arash_check") or {}).get("date")):
        reasons.append("Arash's freeze check not recorded")
    if fz.get("split_sha256") != SPLIT_SHA256:
        reasons.append("split sha differs")
    try:
        if fz.get("prereg_sha256") != sha256_file(P["prereg"]):
            reasons.append("preregistration changed after the freeze")
    except OSError:
        reasons.append("preregistration unreadable")
    if check_code:
        for rel, sha in sorted((fz.get("code") or {}).items()):
            try:
                if sha256_file(os.path.join(P["root"], rel)) != sha:
                    reasons.append(f"code changed after the freeze: {rel}")
            except OSError:
                reasons.append(f"frozen code file missing: {rel}")
        if not fz.get("code"):
            reasons.append("no code hashes in freeze.json")
        reasons += freeze_artifact_problems(fz, root)
    return {"ok": not reasons, "reasons": reasons}


def invalid_pres(event_rows) -> set:
    """Presentations an amendment declared unrecoverably INVALID (prereg s13): events `invalid_unrecoverable` with `pres` (str or list).
    They are dropped from EVERY arm (the pairing is kept); execute does not wait for them, score flags them `invalid_episode`."""
    out = set()
    for r in event_rows or ():
        if r.get("event") == "invalid_unrecoverable":
            p = r.get("pres")
            out |= set([p] if isinstance(p, str) else (p or ()))
    return out


#: model- or simulator-generated free text: the registered key PATTERNS (sk-..., Bearer ...) match ordinary hyphenated words in prose
#: ("a task-specific plan"), so on these fields only the exact loaded secret is searched.
LEAK_FREE_FIELDS = ("raw", "say", "parse_note", "error", "stderr_tail", "events", "errors", "notes", "cause", "why", "breaches",
                    "detail", "msg", "reason", "note")


def leak_paths(rec, secrets=()) -> list:
    """Key-leak check of one record: patterns on structural fields, the exact secret everywhere."""
    hits = []

    def strip(o):
        if isinstance(o, dict):
            return {k: ("<free>" if k in LEAK_FREE_FIELDS else strip(v)) for k, v in o.items()}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o

    full = json.dumps(rec, ensure_ascii=False, default=str)
    if key_leak(json.dumps(strip(rec), ensure_ascii=False, default=str), secrets):
        hits.append("structural")
    if any(s.reveal() in full for s in secrets if hasattr(s, "reveal") and len(s.reveal()) >= 8):
        hits.append("exact-secret")
    return hits


def scan_jsonl_leaks(path: str, secrets=()) -> list:
    """Key-leak scan of a journal file, line by line with `leak_paths` (not the raw text: a hyphenated word in a free-text field must not
    withhold a phase). -> [line numbers with a hit]."""
    bad = []
    try:
        with open(path, encoding="utf-8") as f:
            for n, ln in enumerate(f):
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except ValueError:
                    if key_leak(ln, secrets):
                        bad.append(n)
                    continue
                if leak_paths(rec, secrets):
                    bad.append(n)
    except OSError:
        pass
    return bad


class DatasetChanged(RuntimeError):
    pass


def dataset_problems(namespace: str, root: str | None = None, pinned: str | None = None) -> list:
    """The gold dataset file (editable from the console) must be the file the run was cut from. test-v1: equal to
    the split manifest's `dataset_sha256` (and to freeze.json's, once frozen). Any namespace: equal to `pinned` (the sha a phase
    recorded when it started, so Phase C can tell that the dataset moved after Phase A)."""
    P = paths(root)
    try:
        got = sha256_file(P["dataset"])
    except OSError:
        return ["the gold dataset is unreadable"] if namespace == "test-v1" or pinned else []
    out = []
    if namespace == "test-v1":
        man = load_manifest(root)
        want = (man.get("inclusion") or {}).get("dataset_sha256") or man.get("dataset_sha256")   # A26: the successor dataset
        if want and got != want:
            out.append("the gold dataset differs from the one the split was cut from")
        try:
            with open(P["freeze"], encoding="utf-8") as f:
                fw = json.load(f).get("dataset_sha256")
            if fw and got != fw:
                out.append("the gold dataset changed after the freeze")
        except (OSError, ValueError):
            pass
    if pinned and got != pinned:
        out.append("the gold dataset changed since the phase that pinned it (capture) started")
    return out


def thesis_key():
    """The Anthropic key of THIS study: THESIS_ANTHROPIC_API_KEY only (environment, else the per-user Windows registry). The generic
    ANTHROPIC_API_KEY is never read here: Claude Code itself picks that one up (A15). -> planners.keys.Secret, else PlannerSkip."""
    from planners.base import PlannerSkip
    from planners.keys import Secret, _windows_user_env
    v = (os.environ.get("THESIS_ANTHROPIC_API_KEY") or "").strip() or _windows_user_env("THESIS_ANTHROPIC_API_KEY")
    if not v:
        raise PlannerSkip("THESIS_ANTHROPIC_API_KEY is not set: live Anthropic planning skipped")
    return Secret(v)


KEY_ENV = "THESIS_ANTHROPIC_API_KEY"


def wsl_key_env(env: dict) -> dict:
    """A COPY of `env` for a WSL child that carries the study's key: the value is looked up (environment, else the per-user Windows registry)
    and forwarded through WSLENV; it is never printed, logged or returned separately. Raises PlannerSkip when there is no key."""
    out = dict(env)
    out[KEY_ENV] = thesis_key().reveal()
    parts = [p for p in out.get("WSLENV", "").split(":") if p and not p.startswith(KEY_ENV)]
    out["WSLENV"] = ":".join(parts + [KEY_ENV + "/u"])
    return out


def freeze_artifact_problems(fz: dict, root: str | None = None) -> list:
    """Non-code frozen artefacts versus the disk (M-fix): every snapshot's per-file shas, params.json (the A12 allocation),
    fs_blocks.json, reset_tolerances.json, the detector weights and the walked map. Only artefacts the freeze recorded are checked."""
    P = paths(root)
    r = P["root"]
    out = []

    def chk(label, path, want):
        if not want:
            return
        try:
            if sha256_file(path) != want:
                out.append(f"{label} changed after the freeze")
        except OSError:
            out.append(f"{label} missing after the freeze")

    chk("the gold dataset", P["dataset"], fz.get("dataset_sha256"))
    chk("params.json (A12 allocation)", os.path.join(r, "sim", "step5", "params.json"), fz.get("params_sha256"))
    chk("fs_blocks.json", P["fs_blocks"], fz.get("fs_blocks_sha256"))
    chk("reset_tolerances.json", P["tolerances"], fz.get("reset_tolerances_sha256"))
    chk("families_approved.json", P["families"], fz.get("families_approved_sha256"))
    for nm, w in sorted((fz.get("detector_weights") or {}).items()):
        if isinstance(w, dict) and w.get("path"):
            chk(f"detector weights {nm}", os.path.join(r, w["path"]), w.get("sha256"))
    mp = fz.get("map")
    if isinstance(mp, dict) and mp.get("path"):
        chk("walked map", os.path.join(r, mp["path"]), mp.get("sha256"))
    for key, ent in sorted((fz.get("snapshots") or {}).items()):
        if not isinstance(ent, dict) or not ent.get("files"):
            continue
        src = ent.get("source") or key
        if ent.get("dir"):                                   # ACC-RUNNER / F1: an ft-v2 freeze records each snapshot's own directory
            d = os.path.join(r, *str(ent["dir"]).split("/"))
        else:
            try:
                d = snapshot_dir(src, root)
            except ValueError:
                continue
        for fn, want in sorted(ent["files"].items()):
            chk(f"snapshot {key}/{fn}", os.path.join(d, fn), want)
    for label, ent in sorted((fz.get("artefacts") or {}).items()):   # F9: further frozen artefacts {label: {path, sha256}}
        if isinstance(ent, dict) and ent.get("path"):
            chk(f"artefact {label}", os.path.join(r, *str(ent["path"]).split("/")), ent.get("sha256"))
    return out


def snapshot_freeze_problems(sources, root: str | None = None, freeze: dict | None = None) -> list:
    """Per-file shas of the snapshots in `sources` versus freeze.json (empty list when nothing is frozen yet)."""
    if freeze is None:
        try:
            with open(paths(root)["freeze"], encoding="utf-8") as f:
                freeze = json.load(f)
        except (OSError, ValueError):
            return []
    snaps = freeze.get("snapshots") or {}
    out = []
    for s in sources:
        ent = snaps.get(s) or snaps.get("o0k10" if s == "S*" else s)
        if not isinstance(ent, dict) or not ent.get("files"):
            continue
        d = os.path.join(paths(root)["root"], *str(ent["dir"]).split("/")) if ent.get("dir") else snapshot_dir(s, root)
        for fn, want in sorted(ent["files"].items()):
            try:
                got = sha256_file(os.path.join(d, fn))
            except OSError:
                got = None
            if got != want:
                out.append(f"snapshot {s}/{fn} differs from freeze.json")
    return out


def guard_items(item_ids, namespace: str, *, allow_test: bool = False, root: str | None = None) -> dict:
    """Refuse (TestItemRefused) any item outside the namespace's split set. A TEST item additionally needs namespace test-v1,
    allow_test=True (the --allow-test flag), a valid freeze.json and families_approved.json. Call it BEFORE any capture,
    planning or execution of a batch; every phase calls it again on resume."""
    ns = ns_check(namespace)
    m = load_manifest(root)
    test_like = set(m["test"]) | set(m["test_uncapped_not_scored"])
    try:                                                 # ACC-RUNNER: acceptance-v2 = the registered roster (broken registration -> refuse)
        allowed = set(namespace_items(namespace, m, root))
    except RuntimeError as e:
        raise TestItemRefused(f"{namespace}: {e}") from None
    if ns["items"] == ACCEPTANCE_ROSTER and _acc().session_problems(namespace):  # only through acc_runner (armed for THIS namespace)
        raise TestItemRefused(_acc().session_problems(namespace)[0])
    ids = list(item_ids)
    outside = sorted(i for i in ids if i not in allowed)
    if outside:
        raise TestItemRefused(f"{len(outside)} item(s) outside {namespace}'s set {ns['items']}: {outside[:5]}")
    tests = [i for i in ids if i in test_like]
    if tests:
        why = []
        if namespace != "test-v1" and not ns.get("replication_test"):
            why.append("test items only in namespace test-v1 (Kitchen 1) or a replication test namespace")
        if not allow_test:
            why.append("--allow-test not given")
        why += (_k23.replication_freeze_status(namespace, root)["reasons"] if ns.get("replication_test")   # K23: THIS kitchen's freeze
                else freeze_status(root)["reasons"])
        why += families_status(root)["reasons"]
        if why:
            raise TestItemRefused(f"{len(tests)} test item(s) refused: " + "; ".join(why))
    return {"n": len(ids), "n_test": len(tests), "namespace": namespace}


# ================================================================================================ masking + keys (SPEC s8)
#: prereg s11 pattern, with '+' added to the objectId class: AI2-THOR ids carry signed coordinates ("Apple|+01.20|..."), which
#: the registered `[A-Za-z]+\|[-0-9.]+\|` misses (the pilot's MASK had that gap). Strictly tighter; disclosed in SPEC s8.
MASK = re.compile(r"FloorPlan|AI2-THOR|ai2thor|iTHOR|Unity|_physics|[A-Za-z]+\|[-+0-9.]+\|")


def mask_problems(text: str) -> list:
    return sorted(set(m.group(0) for m in MASK.finditer(text or "")))


def key_leak(text: str, secrets=()) -> bool:
    """True if `text` holds anything key-like (planners.keys.redact would change it) or one of `secrets` verbatim."""
    return redact(text, secrets) != str(text)


# ================================================================================================ ledger (SPEC s9, A15)
class BlockRefused(CostCeilingExceeded):
    pass


def _cell_of(entry):
    """The cell tag a ledger row carries in its `entry` JSON (plan first: attempt id = row id, cell = the plan cell it belongs to), or None."""
    try:
        return (json.loads(entry) or {}).get("cell") if entry else None
    except (TypeError, ValueError, AttributeError):
        return None


class ScopedLedger(CostLedger):
    """CostLedger with SCOPES. Every row's provider column is '<provider>@<scope>'. Ceilings: {'core': c, 'total': t}
    ('other' has no own ceiling; it is bounded by 'total'). A paid (anthropic) reservation without a scope is refused.

    `block(scope, worst_usd)` holds the block's worst case BEFORE the block starts (refused -> BlockRefused, nothing sent);
    inside the block per-call reservations are checked against the ceilings without double-counting the hold; the hold is
    released when the block ends. Holds left by a dead process are released when the ledger is opened (a PhaseLock makes
    the namespace single-writer)."""

    def __init__(self, path: str, ceilings: dict, audit: bool = True, core_done=None, release_stale: bool = True):
        super().__init__(path, global_ceiling_usd=float(ceilings["total"]), provider_ceilings=None, audit=audit)
        self.ceilings = {k: float(v) for k, v in ceilings.items()}
        self.core_done = core_done                          # callable -> True once every core cell is terminal (A15 reserve ends)
        self.scope = None
        self._hold = None                                   # (rid, usd, scope)
        self.cell_tag = None                                # plan-first: the cell this object is working on; stored on every reservation (attempt id = row id)
        if release_stale:                                   # a WORKER ledger (parallel planning) must not release the live holds of its siblings
            with self._conn() as c:
                n = c.execute("UPDATE ledger SET state='released' WHERE state='open' AND provider LIKE 'hold@%'").rowcount
            if n:
                self._audit({"event": "stale_holds_released", "n": n})

    @contextlib.contextmanager
    def use(self, scope: str):
        old, self.scope = self.scope, scope
        try:
            yield self
        finally:
            self.scope = old

    def _sum(self, c, like: str, exclude_rid=None) -> float:
        q = ("SELECT COALESCE(SUM(CASE WHEN state='open' THEN reserved WHEN state='settled' THEN actual ELSE 0 END),0) "
             "FROM ledger WHERE provider LIKE ?")
        args = [like]
        if exclude_rid is not None:
            q += " AND id != ?"
            args.append(exclude_rid)
        return c.execute(q, args).fetchone()[0]

    def committed(self, scope: str | None = None) -> float:
        with self._lock, self._conn() as c:
            return round(self._sum(c, f"%@{scope}" if scope else "%"), 6)

    def billed_scope(self, scope: str | None = None) -> float:
        with self._lock, self._conn() as c:
            r = c.execute("SELECT COALESCE(SUM(actual),0) FROM ledger WHERE state='settled' AND provider LIKE ?",
                          (f"%@{scope}" if scope else "%",)).fetchone()
            return round(r[0], 6)

    def _check(self, c, scope, usd, exclude_rid=None):
        tot = self._sum(c, "%", exclude_rid)
        if tot + usd > self.ceilings["total"] + 1e-12:
            raise CostCeilingExceeded(f"total ceiling {self.ceilings['total']:.2f} USD: committed {tot:.4f} + {usd:.4f}")
        if scope in self.ceilings and scope != "total":
            st = self._sum(c, f"%@{scope}", exclude_rid)
            if st + usd > self.ceilings[scope] + 1e-12:
                raise CostCeilingExceeded(f"{scope} ceiling {self.ceilings[scope]:.2f} USD: committed {st:.4f} + {usd:.4f}")
        if scope == "other" and "core" in self.ceilings and self.core_done is not None and not self._core_finished():
            # A15 / s10: "the primary budget is reserved first": exploratory and control spend may use only what the core six leave
            # unreserved inside the total, until every core cell is terminal.
            head = max(0.0, self.ceilings["core"] - self._sum(c, "%@core"))
            if tot + usd + head > self.ceilings["total"] + 1e-12:
                raise CostCeilingExceeded(f"core reserve: {head:.2f} USD of the core ceiling is still unspent; non-core spend would "
                                          f"take the total to {tot + usd + head:.2f} > {self.ceilings['total']:.2f}")
        gp = _acc().guard_problems(self, c, usd, exclude_rid)   # ACC-RUNNER: attempt ceiling + A29 study-wide check (guarded ledgers)
        if gp:
            raise CostCeilingExceeded(gp[0])

    def _core_finished(self) -> bool:
        try:
            return bool(self.core_done()) if self.core_done else False
        except Exception:
            return False

    def open_reservations(self) -> dict:
        """Per-call reservations that never resolved (an ambiguous transport failure, or a killed process): they stay `open` at their
        full worst case and count against the ceilings. Block holds are not listed (they are released on every open)."""
        with self._lock, self._conn() as c:
            rows = c.execute("SELECT id, ts, provider, reserved, entry FROM ledger WHERE state='open' AND provider NOT LIKE 'hold@%'").fetchall()
        return {"n": len(rows), "usd": round(sum(r[3] for r in rows), 6),
                "rows": [{"id": r[0], "ts": r[1], "provider": r[2], "reserved": r[3], "cell": _cell_of(r[4])} for r in rows]}

    def reconcile_open(self, note: str, ids=None, settle: dict | None = None) -> dict:
        """AUDITED reconciliation, run by Arash through `plan.py --reconcile-open "<note>"`: every listed (default: all) stale per-call
        reservation is RELEASED, or settled at the actual USD given in `settle` {id: usd} (from the provider's own billing record).
        Nothing is hidden: the audit log carries the note, the ids and the USD."""
        if not str(note or "").strip():
            raise ValueError("reconciliation needs a note")
        cur = self.open_reservations()["rows"]
        want = {int(i) for i in ids} if ids else {r["id"] for r in cur}
        settle = {int(k): float(v) for k, v in (settle or {}).items()}
        done = {"released": [], "settled": {}}
        with self._lock, self._conn() as c:
            for r in cur:
                if r["id"] not in want:
                    continue
                if r["id"] in settle:
                    c.execute("UPDATE ledger SET state='settled', actual=? WHERE id=?", (settle[r["id"]], r["id"]))
                    done["settled"][r["id"]] = settle[r["id"]]
                else:
                    c.execute("UPDATE ledger SET state='released' WHERE id=?", (r["id"],))
                    done["released"].append(r["id"])
        usd = round(sum(r["reserved"] for r in cur if r["id"] in want), 6)
        self._audit({"event": "reconcile_open", "note": str(note)[:300], "released": done["released"], "settled": done["settled"], "reserved_usd": usd})
        return {**done, "reserved_usd": usd}

    def reserve(self, provider: str, usd: float) -> int:
        scope = self.scope
        if scope is None:
            if provider == "anthropic" or usd > 0:
                raise CostCeilingExceeded(f"paid reservation for {provider} without a ledger scope; nothing sent")
            scope = "free"
        hold = self._hold if (self._hold and self._hold[2] == scope) else None
        with _acc().study_boundary(self), self._lock, self._conn() as c:    # ACC-RUNNER c3: check + insert inside the study-wide boundary
            c.execute("BEGIN IMMEDIATE")
            try:
                self._check(c, scope, usd, exclude_rid=hold[0] if hold else None)
                rid = c.execute("INSERT INTO ledger (ts, provider, state, reserved, entry) VALUES (?,?,?,?,?)",
                                (time.strftime("%Y-%m-%dT%H:%M:%S"), f"{provider}@{scope}", "open", float(usd),
                                 json.dumps({"cell": self.cell_tag}) if self.cell_tag else None)).lastrowid
                c.execute("COMMIT")
            except BaseException:
                c.execute("ROLLBACK")
                raise
        self._audit({"event": "reserve", "id": rid, "provider": provider, "scope": scope, "usd": round(usd, 6),
                     **({"cell": self.cell_tag} if self.cell_tag else {})})
        return rid

    def settle(self, rid: int, actual_usd: float, entry: dict):
        super().settle(rid, actual_usd, {**entry, "cell": self.cell_tag} if self.cell_tag else entry)

    def book(self, entry: dict):
        super().book({**entry, "provider": f"{entry['provider']}@{self.scope or 'free'}", **({"cell": self.cell_tag} if self.cell_tag else {})})

    def spent(self, provider: str | None = None) -> float:
        with self._lock, self._conn() as c:
            return round(self._sum(c, f"{provider}@%" if provider else "%"), 6)

    @contextlib.contextmanager
    def block(self, scope: str, worst_usd: float):
        if self._hold is not None:
            raise RuntimeError("blocks do not nest")
        _acc().refresh_on_block(self.path)                  # ACC-RUNNER: every submission re-reads the other ledgers first
        try:
            bnd = _acc().study_boundary(self)               # ACC-RUNNER c3: check + hold inside the study-wide boundary
            bnd.__enter__()
        except CostCeilingExceeded as e:
            raise BlockRefused(f"block not started ({scope}, worst case {worst_usd:.4f} USD): {e}") from None
        try:
            with self._lock, self._conn() as c:
                c.execute("BEGIN IMMEDIATE")
                try:
                    try:
                        self._check(c, scope, worst_usd)
                    except CostCeilingExceeded as e:
                        raise BlockRefused(f"block not started ({scope}, worst case {worst_usd:.4f} USD): {e}") from None
                    rid = c.execute("INSERT INTO ledger (ts, provider, state, reserved) VALUES (?,?,?,?)",
                                    (time.strftime("%Y-%m-%dT%H:%M:%S"), f"hold@{scope}", "open", float(worst_usd))).lastrowid
                    c.execute("COMMIT")
                except BaseException:
                    c.execute("ROLLBACK")
                    raise
        finally:
            bnd.__exit__(None, None, None)
        self._hold = (rid, float(worst_usd), scope)
        self._audit({"event": "block_hold", "id": rid, "scope": scope, "usd": round(worst_usd, 6)})
        try:
            with self.use(scope):
                yield rid
        finally:
            self._hold = None
            with self._lock, self._conn() as c:
                c.execute("UPDATE ledger SET state='released' WHERE id=?", (rid,))
            self._audit({"event": "block_release", "id": rid})


def block_worst_case(n_calls: int, reserve_per_call_usd: float, repairs: bool = True) -> float:
    """Worst case of one planning block: every first transmission plus its one repair, each at its own worst-case reserve."""
    return round(n_calls * (2 if repairs else 1) * float(reserve_per_call_usd), 6)


OPUS_REF_USD_PER_CALL = 0.044           # A37 (Arash 2026-10-16: normal API for the whole Kitchen 1 plan): 2 x the A24.1-C batch reference 0.022 (S2 projection 26.606 / 1,210 cells) = the sync price
OPUS_GUARD_USD_PER_CALL = 0.080         # A37: the A24.1-C rule at sync prices, floor_0.001(min(2 x 0.044, 60 / 740, 100 / 1,210)) = 0.081, set at 0.080 (one step stricter); NOT 2 x the reference (A24.1-C was 0.040)
OPUS_GUARD_EVERY = 60
RECONCILE_ABS_USD = 0.05                # A21 B: ledger-vs-journal tolerance, absolute ...
RECONCILE_REL = 0.02                    # ... or relative to the larger of the two totals, whichever is larger
PAUSE_RESPONSES = ("continue", "cut", "stop")

#: A21 B -- the cost guard, written out (hashed into freeze.json). Every word here is enforced by `cost_guard` / plan.py and tested.
COST_GUARD_SPEC = {
    "version": "ftm-guard-4",                 # A37 (was ftm-guard-3 = A24.1-C, 0.040 at the batch price): explicit 0.080 threshold at the sync price, ft-v2 unit (two repairs), A29 batch in-flight
    "unit": "one billed Opus cell = the first transmission + up to two repair turns + every billed transport attempt; cache replays are not cells",
    "numerator": "sum of cost_usd over every attempt of the cell, including repaired, rejected and infra-failed cells that were billed",
    "denominator": "number of billed Opus cells (cost_usd > 0); a transport failure that was never billed is not counted",
    "reference_usd": OPUS_REF_USD_PER_CALL, "pause_above_usd": OPUS_GUARD_USD_PER_CALL, "units": "USD per billed Opus cell",
    "threshold": "explicit (A24.1-C rule, A37 sync prices): pause above 0.080 USD per billed Opus cell; NOT 2 x the 0.044 reference; a heuristic tripwire, not a completion test",
    "minimum_sample": OPUS_GUARD_EVERY,
    "checks": ["cumulative mean over all billed Opus cells", "recent window: mean of the last 60 billed Opus cells"],
    "cadence": "at block boundaries only, each time 60 more billed Opus cells exist; the cumulative mean AND the recent window must be <= pause_above_usd",
    "projection": "spent (settled + open reservations at their worst case) + remaining cells x running mean per model <= each ceiling (core, total)",
    "in_flight": "covers A29 batch execution (chunk holds, request reservations, unresolved results): a transmission or a batch chunk in flight is a ledger reservation / hold at its worst case and counts against both ceilings until settled or released; an absent batch result stays unresolved, never free",
    "reconciliation": {"abs_usd": RECONCILE_ABS_USD, "rel": RECONCILE_REL,
                       "rule": "at every boundary the ledger's settled paid total must equal the sum of cost_usd over all plan-journal rows within max(abs, rel x larger total); otherwise pause"},
    "billing_statement": "after W5, and at every pause, Arash compares the ledger total with the provider's own usage page; a gap above 5 % is reported with the results",
    "responses_to_a_pause": {"continue": "--ack-pause NOTE --pause-response continue (nothing else changes)",
                             "cut": "--ack-pause NOTE --pause-response cut AFTER recording the next group of the registered cut order (--cut), blind to outcomes",
                             "stop": "declare the study incomplete (--declare-incomplete); section 13 applies"},
    "decision_rule": "every pause decision is outcome-blind: only spend, token, error and integrity counters may be used; in test-v1 the note names that evidence (>= 20 characters) and --outcome-blind attests it",
    "not_allowed": "changing a prompt, a model, a budget, the arm roster outside the cut order, or the ceilings after the first call; that is a protocol change = amendment + restart of the phase",
}


def reconcile_spend(ledger_usd: float, journal_usd: float) -> dict:
    """A21 B: the settled paid total of the ledger against the plan journal's own cost column (pure)."""
    tol = max(RECONCILE_ABS_USD, RECONCILE_REL * max(ledger_usd, journal_usd))
    gap = abs(ledger_usd - journal_usd)
    return {"ok": gap <= tol + 1e-12, "gap_usd": round(gap, 6), "tolerance_usd": round(tol, 6),
            "ledger_usd": round(ledger_usd, 6), "journal_usd": round(journal_usd, 6)}


def cost_guard(*, opus_calls: int, opus_usd: float, checked_at: int, spent: float, remaining: dict, mean_cost: dict,
               ceiling: float, opus_recent=None, ledger_usd=None, journal_usd=None) -> dict:
    """A15 guard as written out in A21 B (`COST_GUARD_SPEC`); evaluated at block boundaries only (pure; inputs are spend counters, never outcomes).
    -> {'pause': bool, 'reasons': [...], 'checked_at': n, 'projected': usd, 'reconcile': {...}|None}.
    Every 60 billed Opus cells the CUMULATIVE mean and (when `opus_recent` is given) the mean of the LAST 60 billed Opus cells (USD per billed
    Opus cell) are compared with the EXPLICIT threshold 0.080 (A37; reference 0.044; not 2 x the reference): a cumulative mean can hide a
    recent price or token surge. No mean check below 60 cells. The projected
    completion (spent + remaining cells x running mean per model) must stay within the ceiling. When both totals are given, the ledger's settled
    paid total must agree with the plan journal's cost column (`reconcile_spend`)."""
    reasons = []
    new_checked = checked_at
    if opus_calls - checked_at >= OPUS_GUARD_EVERY:
        new_checked = opus_calls - (opus_calls - checked_at) % OPUS_GUARD_EVERY
        mean = opus_usd / max(1, opus_calls)
        if mean > OPUS_GUARD_USD_PER_CALL + 1e-12:
            reasons.append(f"Opus mean {mean:.4f} USD per billed Opus cell > the A24.1-C threshold {OPUS_GUARD_USD_PER_CALL:.3f} (reference {OPUS_REF_USD_PER_CALL:.3f})")
        if opus_recent is not None and len(opus_recent) >= OPUS_GUARD_EVERY:
            w = list(opus_recent)[-OPUS_GUARD_EVERY:]
            wm = sum(w) / len(w)
            if wm > OPUS_GUARD_USD_PER_CALL + 1e-12:
                reasons.append(f"recent-window Opus mean {wm:.4f} USD per billed Opus cell (last {len(w)} billed cells) > the A24.1-C threshold {OPUS_GUARD_USD_PER_CALL:.3f}")
    rec = None
    if ledger_usd is not None and journal_usd is not None:
        rec = reconcile_spend(float(ledger_usd), float(journal_usd))
        if not rec["ok"]:
            reasons.append(f"ledger {rec['ledger_usd']:.4f} USD differs from the plan journal {rec['journal_usd']:.4f} USD by "
                           f"{rec['gap_usd']:.4f} (tolerance {rec['tolerance_usd']:.4f})")
    projected = spent + sum(int(n) * float(mean_cost.get(m, 0.0)) for m, n in remaining.items())
    if projected > ceiling + 1e-12:
        reasons.append(f"projected completion {projected:.2f} USD > ceiling {ceiling:.2f}")
    return {"pause": bool(reasons), "reasons": reasons, "checked_at": new_checked, "projected": round(projected, 4), "reconcile": rec}


# ================================================================================================ phase lock (SPEC s10)
class PhaseLocked(RuntimeError):
    pass


class PhaseLock:
    """One writer per (namespace, phase): a lock file with host / pid / start. A stale lock (crash) is removed only by an
    explicit `steal=True` (CLI --steal-lock), which is logged by the caller."""

    def __init__(self, namespace: str, phase: str, root: str | None = None, steal: bool = False):
        self.path = os.path.join(ns_work(namespace, root), f"{phase}.lock")
        self.steal = steal

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if self.steal and os.path.exists(self.path):
            os.remove(self.path)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                held = open(self.path, encoding="utf-8").read()
            except OSError:
                held = "?"
            raise PhaseLocked(f"{self.path} is held: {held}") from None
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(canon({"host": socket.gethostname(), "pid": os.getpid(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}))
        return self

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except OSError:
            pass
        return False


def open_ledger(namespace: str, root: str | None = None) -> ScopedLedger:
    return ScopedLedger(os.path.join(ns_work(namespace, root), "ledger.db"), NAMESPACES[namespace]["ceilings"])


def sqlite_ok() -> bool:                 # used by the selftest; the ledger needs sqlite3 in both pythons
    return bool(sqlite3.sqlite_version)
