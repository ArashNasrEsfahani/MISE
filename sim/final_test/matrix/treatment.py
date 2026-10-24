"""C4 - the ft-v2 treatment switch of the locked final-test runner (FREEZE-READINESS item C4; prereg A24 items 1-4 and 11, A25 items 1-2).

ONE pinned dict, `FT2`, says how an ft-v2 namespace runs; its canonical sha is the "switch sha". Every capture / plan / unit / practice row
of an ft-v2 namespace records the TREATMENT IDENTITY (`identity()`): the switch sha + the effective package params + the sha of every
source file the package, the prompt summary and the briefing hook read. A resume across a different identity is refused (Astra C4 Q10).

What ft-v2 changes (and ONLY in an ft-v2 namespace; every older namespace keeps its bytes, gate `sim/_ftm_treatment_selftest.py`):
  * contract  `PlannerContract(version="ft-v2")` with the D2-filtered type list (`planner_types`); <= 2 repairs; `observed_types`;
  * summary   the agent's FULL object-memory list after the sweep, D2-filtered, then `summary_v2.compact` + A24's overflow rule
              (`prompt_summary`: more than 40 types -> the first 39 types A-Z and one omitted-count line);
  * package   for the +M arms the S* / curve block is the package (`Packager`): `ltm_package.PackageBuilder` with the ONE pinned profile
              `FT2["package_profile"]` (pkg-1.r4 / pkg-1.r6 / pkg-1.r6b; Arash decides), the D2 filter forced on for r4 (re-hashed), the
              UNIVERSAL mask (+ `mechanisms_off_all`), from the ft-v2 snapshot db + that snapshot's own practice-row prefix, with the item
              leave-out (`ItemLeaveOut`; on a TEST presentation an attributable record is an integrity failure, never a silent drop);
  * executor  prior chooser for every arm; briefing-guided search (hook B) for the +M arms, the curve arms and P+M only (`exec_policy`);
              hook-on and hook-off units never share (the unit key gets ONE extra field `exec_policy` only when a policy is set);
  * practice  practice-v2: the planner gets the package rendered from the pre-attempt practice store + the earlier journaled rows; the
              executor runs prior + hook B while the practice store still learns from the honest event stream.

test-v1 stays on ft-v1 until `TEST_V1_TREATMENT` is flipped at A24.1 / the freeze (Astra Q1 (b)); after the flip every test entry path
checks that freeze.json records this identity (`entry_problems`). TREAT-FREEZE (2026-10-13): profile pkg-1.r6b, and FT2["pins"] pins the
post-fix executor, its switches and the contract pair; identity() fails closed on any difference (`pin_problems`). STRICT masks are a development diagnostic ("package-only STRICT": the
executor-kind mechanism E1 is not wired here) and are refused in every registered namespace. Draft briefings only in DEV_DRAFT_NAMESPACES.

Pure python + file reads; the simulator is never imported here. Python 3.10 syntax.
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                    # noqa: E402  (also puts sim/ on sys.path)

LEGACY = "ft-v1"
FT2_VERSION = "ft-v2"
VERSIONS = (LEGACY, FT2_VERSION)
SCENE = "FloorPlan1"

#: THE switch. Arash's choices: package_profile (pkg-1.r4 = A25 1.5 registered; pkg-1.r6 / pkg-1.r6b after D5; PINNED pkg-1.r6b by
#: TREAT-FREEZE), mechanisms_off_all (["request_ops"] if he rejects G1, A25 1.2 / 11.1), type_scope ("none" if A26 is not approved: A26
#: item 9). Any change after TREAT-FREEZE is a re-pin: a new identity, and practice under the old one no longer qualifies.
FT2 = {
    "version": FT2_VERSION,
    "package_profile": "pkg-1.r6b",   # TREAT-FREEZE 2026-10-13: ARASH DECISION R6b AUTHORIZED + overnight GO "freeze the treatment (pkg-1.r6b)"
    "mask": "universal",
    "mechanisms_off_all": [],
    "type_scope": "D2",
    "repairs": 2,
    "summary": {"source": "skills.om.summary(limit=None) after the sweep", "rule": "D2 filter, summary_v2.compact, A24 overflow",
                "limit": 40, "agent_xz": None},
    "package_max_tokens": 2000,
    "snapshot_root": "sim/final_test/snapshots_ft2",
    "practice_namespace": "practice-v2",
    "practice_orders": [0],
    "exec_hook": "briefing-B",
    "chooser": "prior",
    "fs_version": "fs-v2",            # C4-open: A24 item 4 (examples only, the package's store, the package_max_tokens maximum)
    # TREAT-FREEZE 2026-10-13 (Astra docs/final-test/_astra/TREAT-FREEZE.txt): the post-fix executor (SPEC-INHAND-apply), its switches as
    # they are now, and the contract pair rebuilt by BK-LINE. Full sha256 of the raw bytes; `self` = the file's own "sha256" field.
    # identity() FAILS CLOSED (TreatmentError) when the running code differs; a change here is a deliberate re-pin (new identity).
    "pins": {
        "executor": {"sim/honest_skills.py": "21543604e419439e485e8fcc482c41cd2e9c94438fa1edeb8411848ee131b8a8",
                     "sim/honest.py": "a3b76525d80509b8e7388cdd7a5213dd4e9d11f5116ca60f5da079b8c65db8a2",
                     "sim/objmem.py": "a74e2492d1bfaa15fe68c93ca432c6911bfacb52e2c21ce3e005ef97c581ee2f"},
        "switches": {"honest_skills": {"AIM_PROTRUDE_ON": True, "CUT_RULE_ON": True, "CUTTING_TOOLS": ["ButterKnife"], "IDENT_ON": True,
                                       "INSPECT_RELOOK_ON": True, "LOOKALIKE_ON": True, "PICK_MULTI_ON": True, "PUT_DRAWER_PIX_ON": True,
                                       "PUT_IN_VERIFY_ON": True, "PUT_SURFACE_ON": True, "TYPE_SCOPE_ON": True},
                     "objmem": {"DEDUP_FRAME": True},                       # F-dup ON
                     "type_scope": {"EXCLUDED_TYPES": ["Knife"]}},          # the D2 Knife filter (A26 item 1)
        "contract": {"sim/planners/exec_contract.json": {"file": "f6900b3e66bc2db0fac63d1c505d119ccfe01162369b18bb9a58d8795c7fd1e6",
                                                         "self": "667aa3d225ae76175b0d3625d2a5251cd8ae16c44278414b61ee75991b1634c2"},
                     "sim/planners/ft_v2.json": {"file": "70bb32eb3327be3526a9a5d65a60c7edd6a54468e050ae9d2038e2d15f0d17e3",
                                                 "self": "42524a6b5ffef1ce561bda1921e13d8fdadb5d153b125e58afea4b572b3f2060",
                                                 "texts": "9ab6477a98805606d998ba166130624a66eb42f9c17b8e1ab71c5c13b0892f5d"}},
    },
}
#: Astra C4 Q1 (b): test-v1 keeps ft-v1 by default; flipped to "ft-v2" in ONE place at A24.1 / the freeze.
TEST_V1_TREATMENT = FT2_VERSION
#: namespaces in which the UNAPPROVED (draft) briefing may be used, explicitly (development only). Registered practice, acceptance and
#: test need the approved briefing (A25 11.1).
DEV_DRAFT_NAMESPACES = ("rehearsal-v3",)
#: namespaces where a STRICT (package-only) mask is allowed: development diagnostics only (A25 2.5; Astra C4 Q9).
STRICT_NAMESPACES = ("rehearsal-v3",)
MASKS = ("universal", "strict")
PROFILE_DIRS = {"pkg-1.r4": "r4", "pkg-1.r6": "r6", "pkg-1.r6b": "r6b"}
#: A24 item 2 overflow rule (text pinned in A24.1): line 40 when more than 40 types remain after the compaction.
OMITTED_LINE = "(+{n} more object types seen; not listed)"
#: the arms whose executor gets hook B under ft-v2 (A25 item 2 table + Astra C4 Q2): +M main arms, the curve arms, the P+M reference control.
PLUS_M_MAIN = ("G+M", "O+M", "X+M")
#: components of default_sources() the item leave-out wraps (the signal harness list)
LEAVE_OUT_COMPONENTS = ("template", "procedure", "profile", "places", "example")
#: source files whose bytes define what the package / summary / hook produce (identity); paths relative to the repo root
IDENTITY_FILES = ("sim/ltm_package.py", "sim/goal_schema.py", "sim/type_scope.py", "sim/kitchen_briefing.py", "sim/ltm_procedural.py",
                  "sim/ltm_templates.py", "sim/ltm_examples.py", "sim/ltm_profile.py", "sim/step5/package_params.json",
                  "sim/step5/procedures_kb.json", "sim/step5/l3_templates.json", "sim/step5/briefing_conventions.json",
                  "sim/step5/uses_table.json", "sim/step5/homeowner_inventory.json", "sim/step5/lexicon/need_cues_r6b.json",
                  "sim/planners/summary_v2.py", "sim/planners/contract.py", "sim/planners/validator_v2.py", "sim/planners/ft_v2.json",
                  "sim/final_test/matrix/treatment.py", "sim/final_test/matrix/fs_block.py",
                  "sim/step5/lexicon/concepts.json",
                  # TREAT-FREEZE (Astra 1 / 6): the behaviour-determining executor, perception, contract and provider files. The rest of
                  # the code closure (runner, scoring, analysis) stays pinned by the freeze's gate 12.1 only.
                  "sim/honest_skills.py", "sim/honest.py", "sim/objmem.py", "sim/honest_map.py", "sim/navmap.py", "sim/search.py",
                  "sim/step5_live.py", "sim/perception.py", "sim/step4_gemma.py", "sim/planners/exec_contract.json",
                  "sim/planners/base.py", "sim/planners/ollama_provider.py", "sim/planners/anthropic_provider.py",
                  "sim/planners/anthropic_batch.py", "sim/planners/codex_provider.py",
                  ) + tuple(f"sim/weights/v3/FloorPlan1.{w}" for w in (
                      "yolo26s.pt", "amb26m.pt", "state_open.pt", "state_on.pt", "state_sliced.pt", "state_cooked.pt",
                      "state_broken.pt", "state_dirty.pt", "state_filled.pt")) + ("sim/kitchen_v3/FloorPlan1/state_calib.json",)
#: K23 agent-side (Astra K23-AGENT-r1 2): the perception files of the namespace's scene; FloorPlan1 gives EXACTLY the list above
#: (same paths, same order), so every Kitchen 1 identity is unchanged.
SCENE_FILE_TEMPLATES = tuple(f"sim/weights/v3/{{scene}}.{w}" for w in (
    "yolo26s.pt", "amb26m.pt", "state_open.pt", "state_on.pt", "state_sliced.pt", "state_cooked.pt", "state_broken.pt",
    "state_dirty.pt", "state_filled.pt")) + ("sim/kitchen_v3/{scene}/state_calib.json",)


def identity_files(scene: str = SCENE) -> tuple:
    n = len(SCENE_FILE_TEMPLATES)
    return IDENTITY_FILES[:-n] + tuple(t.format(scene=scene) for t in SCENE_FILE_TEMPLATES)
#: TREAT-FREEZE: freeze_status written by freeze.py --record-treatment; such a freeze.json is NEVER a validated freeze (Astra 4).
TREATMENT_ONLY = "treatment-only-unvalidated"


class TreatmentError(RuntimeError):
    """An ft-v2 integrity violation (mixed treatment, wrong mask for the namespace, a package over budget, a test-derived record ...)."""


# ================================================================================================ which namespace runs what
def treatment_of(ns: str) -> str:
    C.ns_check(ns)
    if ns == "test-v1":
        return TEST_V1_TREATMENT
    return C.NAMESPACES[ns].get("treatment") or LEGACY


def is_ft2(ns: str) -> bool:
    return treatment_of(ns) == FT2_VERSION


def config(ns: str | None = None) -> dict:
    """The switch, deep-copied (callers never mutate the pinned dict)."""
    return json.loads(json.dumps(FT2))


def switch_sha(cfg: dict | None = None) -> str:
    return C.sha256_json(cfg or FT2)


def allow_draft_briefing(ns: str) -> bool:
    return ns in DEV_DRAFT_NAMESPACES


def mask_check(ns: str, mask: str) -> None:
    if mask not in MASKS:
        raise TreatmentError(f"unknown mask {mask!r}")
    if mask == "strict" and ns not in STRICT_NAMESPACES:
        raise TreatmentError(f"STRICT is a development diagnostic (package-only STRICT); it is refused in {ns!r} (A25 2.5: one frozen UNIVERSAL "
                             f"configuration for practice, acceptance and test)")


def snapshot_root(root: str | None = None, scene: str = C.DEFAULT_SCENE) -> str:
    """ft-v2 snapshot root. K23 (Astra K23-NAMESPACES-r1 item 3): a replication kitchen has its own subtree <root>/<scene>/ (Kitchen 1's
    paths unchanged); a missing kitchen snapshot never falls back to Kitchen 1's."""
    base = os.path.join(C.paths(root)["root"], *FT2["snapshot_root"].split("/"))
    return base if scene == C.DEFAULT_SCENE else os.path.join(base, scene)


def snapshot_dir(source: str, root: str | None = None, scene: str = C.DEFAULT_SCENE) -> str:
    """ft-v2 snapshot directory of a memory source ('S*' = o0k10); the practice-v1 tree (C.snapshot_dir) is history and never touched."""
    leg = C.snapshot_dir(source, root)
    tail = os.path.relpath(leg, C.paths(root)["snapshots"])
    return os.path.join(snapshot_root(root, scene), tail)


def source_dir(ns: str, source: str, root: str | None = None) -> str:
    """The namespace-facing snapshot authority (K23: a replication namespace resolves into its kitchen's ft-v2 subtree only)."""
    if C.is_replication(ns):
        if not is_ft2(ns):
            raise TreatmentError(f"{ns}: a replication namespace must be ft-v2")
        return snapshot_dir(source, root, scene=C.ns_scene(ns))
    return snapshot_dir(source, root) if is_ft2(ns) else C.snapshot_dir(source, root)


# ================================================================================================ D2, types and the prompt summary
def d2_on(cfg: dict | None = None) -> bool:
    return (cfg or FT2)["type_scope"] == "D2"


def _ts():
    import type_scope as TS
    return TS


def planner_types(types, cfg: dict | None = None) -> list:
    """The planner's type list: the frozen detector list (checked against freeze.json unfiltered by the caller), then the D2 filter."""
    ts = sorted(set(types))
    return _ts().filter_types(ts) if d2_on(cfg) else ts


def _line_type(line: str) -> str:
    return (str(line).strip().split(" ", 1) or [""])[0]


def prompt_summary(full_lines, cfg: dict | None = None) -> list:
    """A24 item 2 on the agent's FULL list: D2 filter (before compaction), summary_v2.compact (limit 40, agent_xz None); if more than
    40 types remain, the first 39 types A-Z keep one line each and line 40 states how many types were omitted."""
    from planners import summary_v2 as SV
    cfg = cfg or FT2
    lim = int(cfg["summary"]["limit"])
    lines = [str(l) for l in (full_lines or [])]
    if d2_on(cfg):
        lines = [l for l in lines if _ts().in_scope(_line_type(l))]
    types = sorted({_line_type(l) for l in lines if _line_type(l) and _line_type(l) != "Floor"})
    if len(types) <= lim:
        out = SV.compact(lines, lim, cfg["summary"]["agent_xz"])
    else:
        allrows = SV.compact(lines, 10 ** 9, cfg["summary"]["agent_xz"])
        keep, first = types[: lim - 1], {}
        for l in allrows:
            first.setdefault(_line_type(l), l)
        out = [first[t] for t in keep] + [OMITTED_LINE.format(n=len(types) - len(keep))]
    if d2_on(cfg) and any(_ts().mentions_excluded(l) for l in out):
        raise TreatmentError("an excluded type survived the D2 filter in the prompt summary")
    return out


def full_summary(executor, post) -> list:
    """The agent's complete belief list right after the sweep (Astra C4 Q4): read from the executor's live skills object, and proven
    complete: its first lines ARE the executor's (40-line) post-sweep summary."""
    sk = ((getattr(executor, "last", None) or {}).get("skills")) if executor is not None else None
    om = getattr(sk, "om", None)
    if om is None:
        raise TreatmentError("ft-v2 capture needs the executor's skills object (ex.last['skills'].om) after the sweep")
    full = list(om.summary(limit=None))
    cut = list(post["summary"])
    if full[: len(cut)] != cut:
        raise TreatmentError("the full object-memory list does not start with the executor's post-sweep summary (not the same belief state)")
    if len(cut) < len(full) and len(cut) < int(FT2["summary"]["limit"]):
        raise TreatmentError("the executor's summary is shorter than the full list but below its cut: the list is not demonstrably complete")
    return full


def observed_types(prompt_lines) -> list:
    import ltm_package as LP
    return sorted({o["type"] for o in LP.observed_from_summary(prompt_lines)})


# ================================================================================================ package params and masks
def mechanisms_of(profile: str) -> tuple:
    import ltm_package as LP
    return {LP.PROFILE_R4: LP.R4_MECHANISMS, LP.PROFILE_R6: LP.R6_MECHANISMS, LP.PROFILE_R6B: LP.R6B_MECHANISMS}.get(profile, ())


def _mech_file(profile: str, root: str | None = None) -> dict:
    d = PROFILE_DIRS.get(profile)
    if not d:
        return {}
    p = os.path.join(C.paths(root)["root"], "sim", "final_test", "signal", d, "mechanisms.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def strict_off(profile: str, item: str, root: str | None = None) -> tuple:
    """Package-only STRICT: every mechanism of THIS profile's own list switched off for the items its development provenance names
    (tuned_on, by name: r4/mechanisms.json has no `kind` field, so no kind filter is used; executor-kind entries are not in the list)."""
    mf = _mech_file(profile, root)
    return tuple(m for m in mechanisms_of(profile) if item in ((mf.get(m) or {}).get("tuned_on") or []))


def off_for(profile: str, item: str, mask: str, cfg: dict | None = None, root: str | None = None) -> tuple:
    cfg = cfg or FT2
    off = list(cfg.get("mechanisms_off_all") or [])
    bad = [m for m in off if m not in mechanisms_of(profile)]
    if bad:
        raise TreatmentError(f"mechanisms_off_all {bad} are not mechanisms of {profile}")
    if mask == "strict":
        off += [m for m in strict_off(profile, item, root) if m not in off]
    return tuple(m for m in mechanisms_of(profile) if m in off)


def effective_params(profile: str, off=(), cfg: dict | None = None, kitchen: str = C.DEFAULT_SCENE) -> dict:
    """profile_params(profile, off) with the D2 filter set by the switch (forced ON for pkg-1.r4 exactly as the r6b gate builds
    "r4 + D2"; OFF everywhere when type_scope = "none"), re-hashed."""
    import ltm_package as LP
    p = LP.profile_params(profile, off=tuple(off), kitchen=kitchen)        # K23: the kitchen's own uses table / inventory hashes
    want = d2_on(cfg)
    pr = p.get("precision")
    if pr is not None and bool(pr.get("type_scope")) != want:
        pr["type_scope"] = want
        p["sha256"] = LP.params_sha(p)
    return p


# ================================================================================================ briefing hook B
def briefing_hint_table(scene: str = SCENE, *, allow_unapproved: bool, cfg: dict | None = None, out_dir: str | None = None) -> dict:
    """otype -> [storage container types] from the briefing's convention records (the round-5 hook), D2-filtered on both sides.
    load_records refuses an unapproved briefing unless allow_unapproved (dev only)."""
    import kitchen_briefing as KB
    keep = _ts().in_scope if d2_on(cfg) else (lambda t: True)
    tab = collections.OrderedDict()
    for r in KB.load_records(scene, allow_unapproved=allow_unapproved, out_dir=out_dir):
        if r.get("kind") != "conv":
            continue
        st = [x.split(":", 1)[1] for x in (r["keys"].get("rels") or []) if x.startswith("stores:")]
        st = [s for s in st if keep(s)]
        for t in r["keys"].get("types") or []:
            if t in st or not keep(t):
                continue
            for s in st:
                tab.setdefault(t, [])
                if s not in tab[t]:
                    tab[t].append(s)
    return dict(tab)


def hint_table_sha(tab: dict) -> str:
    """Canonical sha: keys sorted, each container list in its own order (Astra C4 Q8)."""
    return C.sha256_json([[k, list(tab[k])] for k in sorted(tab)])


def hints_fn(tab: dict):
    def briefing_hints(otype):
        return list(tab.get(otype) or [])
    briefing_hints.table_sha = hint_table_sha(tab)
    return briefing_hints


def gets_hook(arm: str) -> bool:
    """+M main arms, the learning-curve arms (G+M@o0kK / O+M@o0kK) and the P+M reference control; NOT 0, FS, P0 or the /p replays."""
    if arm in PLUS_M_MAIN or arm == "P+M":
        return True
    return bool(re.fullmatch(r"[GO]\+M@o[0-2]k(?:1|3|5|10)", arm or ""))


def exec_policy(arm: str, ns: str, table_sha: str | None) -> str | None:
    """The unit-key field: None in every ft-v1 namespace (legacy key bytes) and for hook-off arms; "briefing:<sha16>" otherwise."""
    if not is_ft2(ns) or not gets_hook(arm):
        return None
    if not table_sha:
        raise TreatmentError("a +M unit needs the briefing hint table sha")
    return "briefing:" + table_sha[:16]


def effective_chooser(arm: str, ns: str) -> str:
    """ft-v2: the prior chooser for EVERY arm (no long-term memory is opened in execute); ft-v1: the arm table's chooser."""
    if is_ft2(ns):
        return "prior"
    ch = C.arm_info(arm, ns)["chooser"]
    return "prior" if ch == "prior" else ("o0k10" if ch == "S*" else ch)


def install_policy(executor, policy: str | None, hints=None) -> None:
    """Per-episode executor policy: hook B or nothing. The executor consumes it once (step5_live.LiveExecutor.set_episode_policy)."""
    fn = getattr(executor, "set_episode_policy", None)
    if policy is None:
        if fn is not None:
            fn(None)
        return
    if fn is None:
        raise TreatmentError("this executor cannot take a per-episode policy (hook B needs step5_live.LiveExecutor.set_episode_policy)")
    if hints is None:
        raise TreatmentError("policy %s without a hint function" % policy)
    fn({"briefing_hints": hints})


# ================================================================================================ identity
_SHA_CACHE = {}


def _sha_cached(p: str) -> str:
    """sha256 of a file, cached per process on (path, size, mtime_ns) (the detector weights are ~160 MB together)."""
    st = os.stat(p)
    k = (os.path.realpath(p), st.st_size, st.st_mtime_ns)
    if k not in _SHA_CACHE:
        _SHA_CACHE[k] = C.sha256_file(p)
    return _SHA_CACHE[k]


def file_shas(root: str | None = None, scene: str = SCENE) -> dict:
    r = C.paths(root)["root"]
    out = {}
    for rel in identity_files(scene):
        p = os.path.join(r, *rel.split("/"))
        out[rel] = _sha_cached(p) if os.path.exists(p) else "absent"
    return out


def _norm(v):
    """Comparable form of a pinned value: frozenset / set / tuple -> sorted / plain list."""
    if isinstance(v, (set, frozenset)):
        return sorted(v)
    if isinstance(v, tuple):
        return list(v)
    return v


def _source_values(path: str, names) -> tuple:
    """(values, problems): the top-level assignments of `names` in the module source, read with ast (no import). A name assigned more
    than once at top level, or not as a literal / frozenset({...}) literal, is a problem (a later reassignment would hide the pin)."""
    import ast
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    seen, probs = {}, []
    for node in tree.body:
        tgts = node.targets if isinstance(node, ast.Assign) else ([node.target] if isinstance(node, ast.AnnAssign) else [])
        for t in tgts:
            if isinstance(t, ast.Name) and t.id in names:
                if t.id in seen:
                    probs.append(f"{os.path.basename(path)}: {t.id} is assigned more than once at top level")
                val = node.value
                if isinstance(val, ast.Call) and isinstance(val.func, ast.Name) and val.func.id in ("frozenset", "set", "tuple") and len(val.args) == 1:
                    val = val.args[0]
                try:
                    seen[t.id] = _norm(ast.literal_eval(val))
                except ValueError:
                    probs.append(f"{os.path.basename(path)}: {t.id} is not a literal")
    return seen, probs


def _lf_sha16(path: str) -> str:
    """validator_v2._src_sha16: the LF-normalised sha256 prefix that exec_contract.json's derived_from records."""
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()[:16]


def pin_problems(cfg: dict | None = None, *, loaded: bool = True) -> list:
    """TREAT-FREEZE: every difference between FT2["pins"] and the RUNNING code tree (C.SIM; never a data root). (1) full sha256 of the
    executor files; (2) each pinned switch in the module source (ast) and, with loaded=True, in the module object this process
    actually imports (its file must be the pinned path; Astra 3: effective values, bound to the loaded code); (3) the contract pair:
    file shas, self shas, ft_v2's texts sha and its exec_contract_sha256, and exec_contract's derived_from = the live executor."""
    import importlib
    pins = (cfg or FT2).get("pins") or {}
    if not pins:
        return ["FT2 has no pins (the treatment is not frozen)"]
    root = C.ROOT
    probs = []

    def ap(rel):
        return os.path.join(root, *rel.split("/"))
    for rel, want in sorted((pins.get("executor") or {}).items()):
        got = _sha_cached(ap(rel)) if os.path.exists(ap(rel)) else "absent"
        if got != want:
            probs.append(f"{rel}: sha256 {got[:16]} != pinned {want[:16]}")
    for mod, want in sorted((pins.get("switches") or {}).items()):
        path = os.path.join(C.SIM, mod + ".py")
        vals, pp = _source_values(path, set(want))
        probs += pp
        for k, v in sorted(want.items()):
            if k not in vals:
                probs.append(f"{mod}.{k}: not assigned at top level")
            elif vals[k] != v:
                probs.append(f"{mod}.{k} = {vals[k]!r} in the source, pinned {v!r}")
        if loaded:
            m = importlib.import_module(mod)
            mf = os.path.realpath(getattr(m, "__file__", "") or "")
            if os.path.normcase(mf) != os.path.normcase(os.path.realpath(path)):
                probs.append(f"{mod}: the loaded module is {mf or '<no file>'}, not the pinned {path}")
            for k, v in sorted(want.items()):
                if not hasattr(m, k):
                    probs.append(f"{mod}.{k}: missing on the loaded module")
                elif _norm(getattr(m, k)) != v:
                    probs.append(f"{mod}.{k} = {_norm(getattr(m, k))!r} at run time, pinned {v!r} (overridden after import)")
    docs = {}
    for rel, want in sorted((pins.get("contract") or {}).items()):
        if not os.path.exists(ap(rel)):
            probs.append(f"{rel}: absent")
            continue
        got = _sha_cached(ap(rel))
        with open(ap(rel), encoding="utf-8") as f:
            docs[rel] = json.load(f)
        if got != want.get("file"):
            probs.append(f"{rel}: file sha256 {got[:16]} != pinned {str(want.get('file'))[:16]}")
        if docs[rel].get("sha256") != want.get("self"):
            probs.append(f"{rel}: self sha {str(docs[rel].get('sha256'))[:16]} != pinned {str(want.get('self'))[:16]}")
        if "texts" in want and docs[rel].get("texts_sha256") != want["texts"]:
            probs.append(f"{rel}: texts_sha256 {str(docs[rel].get('texts_sha256'))[:16]} != pinned {want['texts'][:16]}")
    ec, ft = docs.get("sim/planners/exec_contract.json"), docs.get("sim/planners/ft_v2.json")
    if ec is not None and ft is not None and ft.get("exec_contract_sha256") != ec.get("sha256"):
        probs.append("ft_v2.json exec_contract_sha256 != exec_contract.json self sha (the pair is not one build)")
    if ec is not None:
        for name, rel in (("honest_skills", "sim/honest_skills.py"), ("step4_gemma", "sim/step4_gemma.py")):
            want16 = (ec.get("derived_from") or {}).get(name)
            if want16 != _lf_sha16(ap(rel)):
                probs.append(f"exec_contract.json derived_from.{name} {want16} != the live {rel} (re-run validator_v2.py --write)")
    return probs


def briefing_identity(scene: str = SCENE, *, allow_unapproved: bool, out_dir: str | None = None) -> dict:
    import kitchen_briefing as KB
    recs = KB.load_records(scene, allow_unapproved=allow_unapproved, out_dir=out_dir)
    tab = briefing_hint_table(scene, allow_unapproved=allow_unapproved, out_dir=out_dir)
    return {"records_sha": C.sha256_json(recs), "n_records": len(recs), "allow_unapproved": bool(allow_unapproved),
            "hint_table_sha": hint_table_sha(tab)}


def identity(ns: str, root: str | None = None, *, cfg: dict | None = None, mask: str = "universal", briefing: dict | None = None) -> dict:
    """The treatment identity of an ft-v2 namespace: switch + effective universal params + source files + briefing. `briefing` may be
    injected (selftest); otherwise it is read with the namespace's approval policy. Fails closed (TreatmentError) when the running code
    differs from FT2["pins"] (TREAT-FREEZE; Astra 2)."""
    cfg = cfg or FT2
    pp = pin_problems(cfg)
    if pp:
        raise TreatmentError("the running code is not the pinned treatment: " + "; ".join(pp[:6]))
    prof = cfg["package_profile"]
    scene = C.ns_scene(ns)                                    # K23 agent-side: the namespace's ONE scene (FloorPlan1 for K1)
    p = effective_params(prof, off=tuple(cfg.get("mechanisms_off_all") or ()), cfg=cfg, kitchen=scene)
    br = briefing if briefing is not None else briefing_identity(scene, allow_unapproved=allow_draft_briefing(ns))
    doc = {"version": cfg["version"], "switch_sha": switch_sha(cfg), "profile": prof, "mask": mask, "params_sha": p["sha256"],
           "files": file_shas(root, scene), "briefing": br, "test_v1_treatment": TEST_V1_TREATMENT}
    doc["sha"] = C.sha256_json(doc)
    return doc


def row_treatment(rec: dict) -> str:
    return rec.get("treatment") or LEGACY


def journal_problems(rows, ns: str, ident_sha: str | None = None) -> list:
    """Mixed-version or stale rows: every row must carry this namespace's treatment; in ft-v2 also this identity sha."""
    want = treatment_of(ns)
    bad = []
    for r in rows:
        t = row_treatment(r)
        if t != want:
            bad.append(f"{r.get('key') or r.get('pres')}: treatment {t} in a {want} namespace")
        elif want == FT2_VERSION and ident_sha and r.get("treatment_id") != ident_sha:
            bad.append(f"{r.get('key') or r.get('pres')}: treatment identity {str(r.get('treatment_id'))[:12]} != {ident_sha[:12]} (resume across a change)")
    return bad


def entry_problems(ns: str, freeze: dict | None, ident_sha: str | None) -> list:
    """Every test entry / resume path: after the flip, freeze.json must record THIS identity; before it, an ft-v2 record is refused."""
    if ns != "test-v1":
        return []
    if (freeze or {}).get("freeze_status") is not None:            # TREAT-FREEZE / Astra 4: a treatment-only record never authorizes
        return [f"freeze.json status {freeze.get('freeze_status')!r}: not a validated freeze (test entry refused)"]
    fz = (freeze or {}).get("treatment") or {}
    if TEST_V1_TREATMENT == FT2_VERSION:
        if not fz:
            return ["test-v1 runs ft-v2 but freeze.json records no treatment identity"]
        if fz.get("sha") != ident_sha:
            return [f"test-v1: the treatment identity {str(ident_sha)[:12]} differs from freeze.json {str(fz.get('sha'))[:12]}"]
        return []
    if fz and fz.get("version") == FT2_VERSION:
        return ["freeze.json records an ft-v2 treatment but TEST_V1_TREATMENT is still ft-v1 (flip the constant, A24.1)"]
    return []


# ================================================================================================ leave-out (defence in depth, Astra C4 Q7)
#: A31 (DRAFT, approved by Arash 2026-10-13; Astra A31-design Q1): an item id inside an evidence string counts only as a WHOLE id, i.e.
#: not preceded or followed by a character of the ASCII id alphabet [A-Za-z0-9_-] ("|", ":", "/", space, quotes, start / end are
#: boundaries), case-insensitive. Before A31 this was a plain substring test, so test id g-l2-slice-bread "occurred" inside practice id
#: g-l2-slice-bread-tomato (the B1 false alarm). support / tuned_on stay exact set membership.
_ID_EDGE = "A-Za-z0-9_-"


def id_in(iid: str, text) -> bool:
    """True iff `iid` occurs in `text` as a whole id (A31). The one shared matcher of _cites and corpus_test_citations."""
    if not iid:
        return False
    return re.search(r"(?<![%s])%s(?![%s])" % (_ID_EDGE, re.escape(str(iid)), _ID_EDGE), str(text), re.ASCII | re.I) is not None


def _cites(prov: dict, item: str) -> str | None:
    if item in (prov.get("support") or []):
        return "support"
    if item in (prov.get("tuned_on") or []):
        return "tuned_on"
    if any(id_in(item, e) for e in (prov.get("evidence") or [])):
        return "evidence"
    return None


class ItemLeaveOut:
    """Wraps one package store. A candidate / affordance record that cites the presentation's item (support, tuned_on, evidence) is
    dropped and LOGGED; with fail=True (a test presentation) it is an integrity failure instead (a test-derived record must not exist)."""

    def __init__(self, src, item: str, log: list, comp: str, fail: bool = False):
        self.src, self.item, self.log, self.comp, self.fail = src, item, log, comp, bool(fail)

    def __getattr__(self, k):
        return getattr(self.src, k)

    def _filter(self, recs, get):
        out = []
        for c in recs:
            rec = get(c)
            pv = (rec or {}).get("provenance") or {}
            why = _cites(pv, self.item)
            if why is None:
                out.append(c)
                continue
            rid = (rec or {}).get("id")
            if self.fail:
                raise TreatmentError(f"provenance: the {self.comp} record {rid} cites test item {self.item} ({why})")
            self.log.append({"component": self.comp, "id": rid, "why": why})
        return out

    def candidates(self, *a, **k):
        return self._filter(self.src.candidates(*a, **k), lambda c: c.get("rec"))

    def afford_records(self, *a, **k):
        return self._filter(self.src.afford_records(*a, **k), lambda r: r)


# ================================================================================================ the package renderer
def rows_sha(rows) -> str:
    return C.sha256_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


ROWS_FILE = "practice_rows.jsonl"


def read_rows(snap_dir: str) -> list:
    p = os.path.join(snap_dir, ROWS_FILE)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_rows(snap_dir: str, rows) -> str:
    p = os.path.join(snap_dir, ROWS_FILE)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return rows_sha(rows)


class Packager:
    """Renders the +M block (the package) for one presentation from one source (a snapshot db + its rows prefix).
    stores_factory(db_path, rows) -> {component: store, ...} (default: ltm_package.default_sources with the namespace's briefing policy).
    The same source + query gives identical bytes for G+M / O+M / X+M (they read the same capture block)."""

    def __init__(self, ns: str, counter, *, cfg: dict | None = None, mask: str = "universal", stores_factory=None, categories=None,
                 test_side: bool | None = None, root: str | None = None, scene: str | None = None):
        # K23 (Astra K23-NAMESPACES-r1 item 4): the package kitchen is the namespace's scene; a conflicting explicit scene is refused
        if scene is not None and scene != C.ns_scene(ns):
            raise TreatmentError(f"Packager: scene {scene} is not {ns}'s kitchen {C.ns_scene(ns)}")
        scene = C.ns_scene(ns)
        self.ns, self.counter, self.cfg, self.mask, self.root, self.scene = ns, counter, cfg or config(ns), mask, root, scene
        mask_check(ns, mask)
        self.profile = self.cfg["package_profile"]
        self.test_side = ((C.ns_check(ns)["items"] == "test" or bool(C.ns_check(ns).get("replication_test")))
                          if test_side is None else bool(test_side))
        self._factory = stores_factory
        self._cats = categories
        self._stores = {}

    def categories(self):
        if self._cats is None:
            import ltm_package as LP
            self._cats = LP.load_categories()
        return self._cats

    def stores(self, key: str, db_path: str, rows):
        if key not in self._stores:
            if self._factory is not None:
                st = self._factory(db_path, rows)
            else:
                import ltm_package as LP
                st = LP.default_sources(self.scene, db_path, list(rows), allow_unapproved=allow_draft_briefing(self.ns),
                                        categories=self.categories())
            self._stores[key] = {k: v for k, v in st.items() if not str(k).startswith("_")}
        return self._stores[key]

    def drop(self, key: str):
        self._stores.pop(key, None)

    def render(self, key: str, db_path: str, rows, *, item: str, instruction: str, lang: str, holding, summary_lines) -> dict:
        import ltm_package as LP
        from planners.contract import PACKAGE_HEADER
        off = off_for(self.profile, item, self.mask, self.cfg, self.root)
        prm = effective_params(self.profile, off=off, cfg=self.cfg, kitchen=self.scene)
        q = LP.make_query_profile(self.profile, instruction, lang, holding, list(summary_lines or []))
        lo_log = []
        st = dict(self.stores(key, db_path, rows))
        for comp in LEAVE_OUT_COMPONENTS:
            if comp in st:
                st[comp] = ItemLeaveOut(st[comp], item, lo_log, comp, fail=self.test_side)
        pkg = LP.PackageBuilder(self.scene, st, self.counter, prm, categories=self.categories()).build(q)
        text = pkg["text"] or ""
        tok = pkg.get("tokens") or {}
        cts = {k: int(v) for k, v in self.counter.counts(PACKAGE_HEADER + "\n" + text).items()} if text else {}
        real = max(cts.values()) if cts else 0
        budget = int(prm.get("budget") or 0)
        retrieved = int(tok.get("retrieved_max", tok.get("max", 0)) or 0)
        probs = []
        if real > int(self.cfg["package_max_tokens"]):
            probs.append(f"package {real} tokens > {self.cfg['package_max_tokens']} (header included, max over tokenizers)")
        if budget and retrieved > budget:
            probs.append(f"retrieved part {retrieved} tokens > the profile cap {budget}")
        if any("؀" <= ch <= "ۿ" for ch in text):
            probs.append("Persian character in the package text")
        if d2_on(self.cfg) and _ts().mentions_excluded(text):
            probs.append(f"excluded type in the package text: {_ts().mentions_excluded(text)}")
        nr = pkg.get("need_rule") or {}
        if nr.get("active") and pkg.get("need_audit"):
            probs.append(f"leftover target-type hint on an active NEED presentation: {pkg['need_audit'][:3]}")
        if probs:
            raise TreatmentError(f"package of {item}/{lang}: " + "; ".join(probs))
        return {"text": text, "sha256": C.sha256_text(text) if text else None, "pkg_sha": pkg.get("sha256"), "profile": self.profile,
                "mask": self.mask, "off": list(off), "params_sha": prm["sha256"], "query_sha": pkg.get("query_sha"),
                "tokens": {"real_max": real, "counts": cts, "builder_max": tok.get("max"), "retrieved_max": retrieved, "budget": budget},
                "empty": not text, "empty_reason": pkg.get("empty_reason"), "kept": pkg.get("kept"),
                "dropped": pkg.get("dropped"), "d2_dropped": sorted({d.get("id") for d in (pkg.get("dropped") or [])
                                                                      if d.get("reason") == getattr(_ts(), "REASON", "")}),
                "leave_out": lo_log, "need": {"active": bool(nr.get("active")), "instruction": nr.get("instruction")},
                "store_shas": pkg.get("store_shas"), "parser_ok": pkg.get("parser_ok"), "provenance": pkg.get("provenance")}


# ================================================================================================ the FS-v2 renderer (C4-open; A24 item 4)
#: the frozen A17 concept lexicon: the alias source of the D2 text check (12.12(d): "covers the excluded-type names and the aliases")
LEXICON_PATH = ("step5", "lexicon", "concepts.json")             # under sim/ of THIS code tree (a frozen code input, not run data)
_ALIAS = {}


def _alias_rx(root: str | None = None):
    """(mask_rx, hit_rx): the aliases of the EXCLUDED types (type name, en, en_plural, fa of concepts.json) and, masked first, every alias
    of another type that CONTAINS one of them ("butter knife", "ButterKnife", "چاقوی کره", "کارد کره"), so ButterKnife is never a hit.
    Latin: case-insensitive, no Latin letter on either side. Persian: no Persian letter before (suffixes after are allowed: "چاقوی تیز")."""
    key = "lexicon"
    if key in _ALIAS:
        return _ALIAS[key]
    with open(os.path.join(C.SIM, *LEXICON_PATH), encoding="utf-8") as f:
        doc = json.load(f)
    lex = dict(doc.get("types") or {})
    for t, e in (doc.get("excluded_aliases") or {}).items():       # A26 (K-B2b): the excluded types left the parser's types table
        lex.setdefault(t, e)
    if not lex:
        raise TreatmentError("the A17 lexicon has no types table")
    TS = _ts()

    def names(t, e):
        return [t] + list(e.get("en") or []) + list(e.get("en_plural") or []) + list(e.get("fa") or [])

    bad = sorted({a for t in TS.EXCLUDED_TYPES for a in names(t, lex.get(t) or {}) if a}, key=lambda a: (-len(a), a))
    low = [b.lower() for b in bad]
    mask = sorted({a for t, e in lex.items() if t not in TS.EXCLUDED_TYPES and isinstance(e, dict) for a in names(t, e)
                   if a and any(b in a.lower() for b in low)}, key=lambda a: (-len(a), a))

    def alt(xs):
        out = []
        for a in xs:
            if any("؀" <= ch <= "ۿ" for ch in a):
                out.append(r"(?<![؀-ۿ])" + re.escape(a))
            else:
                out.append(r"(?<![A-Za-z])" + re.escape(a) + r"(?![A-Za-z])")
        return "|".join(out)
    _ALIAS[key] = (re.compile(alt(mask), re.I) if mask else None, re.compile(alt(bad), re.I))
    return _ALIAS[key]


def excluded_alias_hits(text: str, root: str | None = None) -> list:
    """Alias-aware D2 text check (sorted unique hits, lower-cased). Superset of type_scope.mentions_excluded on the inputs it is used on."""
    mrx, hrx = _alias_rx(root)
    t = str(text or "")
    if mrx is not None:
        t = mrx.sub(" ", t)
    return sorted({m.group(0).lower() for m in hrx.finditer(t)})


class _FSIndex:
    """The FS view of the package's OWN example store (A24 item 4: the same corpus, before ranking). `src` is that store wrapped by the
    same ItemLeaveOut as the package; `ex_index` is the unwrapped ExampleIndex (its corpus maps a record id back to the practice example,
    whose raw line ExampleIndex.render(ex, lang, raw=True) writes). D2: a line naming an excluded type or one of its aliases is dropped
    BEFORE ranking and logged. A missing or duplicate corpus id is an error (Astra a4)."""

    def __init__(self, ex_index, src, *, d2: bool, log: list, root: str | None = None):
        self.ex_index, self.src, self.d2, self.log, self.root = ex_index, src, bool(d2), log, root
        self.by_id = {}
        for ex in getattr(ex_index, "corpus", None) or []:
            if ex["id"] in self.by_id:
                raise TreatmentError("FS: duplicate example id %s in the corpus" % ex["id"])
            self.by_id[ex["id"]] = ex

    def _ex(self, rec):
        ex = self.by_id.get((rec or {}).get("id"))
        if ex is None:
            raise TreatmentError("FS: example record %s is not in the store's corpus" % (rec or {}).get("id"))
        return ex

    def candidates(self, query, include_failures=True):
        out = []
        for c in self.src.candidates(query, include_failures=include_failures):
            rec = dict(c["rec"])
            ex = self._ex(rec)
            rec["verdict"] = ex.get("verdict")                     # render_fs_v2 checks that a raw line ends with its own verdict
            rec["item"] = ex.get("item")                           # and refuses a line that carries the item id (plumbing leak)
            if self.d2:
                line = self.ex_index.render(ex, "en", raw=True)
                hit = sorted(set(_ts().mentions_excluded(line)) | set(excluded_alias_hits(line, self.root)))
                if hit:
                    self.log.append({"id": rec.get("id"), "reason": getattr(_ts(), "REASON", "excluded_type"), "hits": hit})
                    continue
            out.append(dict(c, rec=rec))
        return out

    def render(self, rec, lang, raw=True):
        if not raw:
            raise TreatmentError("FS renders raw practice lines only")
        return self.ex_index.render(self._ex(rec), lang, raw=True)


def fs_query(instruction: str, lang: str, holding, summary_lines) -> dict:
    """The FS retrieval query: the A17 concept parser only (A25 item 2 table: the parser dictionary of goal_schema is a +M component,
    never an FS retrieval key). Same inputs as the package query: instruction, language, holding, the prompt summary. No global is
    toggled (ltm_package.USE_GOAL_SCHEMA is never touched)."""
    import ltm_package as LP
    return LP._make_query_legacy(instruction, lang, holding, list(summary_lines or []))


def corpus_test_citations(ex_index, test_ids) -> list:
    """Astra a1: the WHOLE example corpus, not just the retrieved candidates: every record whose item, support, tuned_on or evidence
    names a test item. Must be empty (a test-derived example must not exist; leave-out cannot see what dedup or ranking hides)."""
    bad = []
    tid = set(test_ids)
    for ex in getattr(ex_index, "corpus", None) or []:
        pv = ex.get("provenance") or {}
        hits = ({ex.get("item")} | set(pv.get("support") or []) | set(pv.get("tuned_on") or [])) & tid
        hits |= {t for t in tid for e in (pv.get("evidence") or []) if id_in(t, e)}          # A31: whole ids only
        if hits:
            bad.append({"id": ex.get("id"), "items": sorted(hits)})
    return bad


class FSRenderer:
    """The ft-v2 FS block of ONE presentation (A24 item 4): the package's example store for the S* source (same rows, same ExampleIndex
    object, same ItemLeaveOut), the A17 query, the PACKAGE's own scorer (PackageBuilder._rank, the "example" section, the effective
    profile params: weights, need discount, irrelevance rule, tie-break; Astra a2) and fs_block.render_fs_v2's whole-raw-line packing
    under the SAME counter object and the same maximum as the package (FT2["package_max_tokens"]). Successes AND failures are eligible
    (PACKAGE-SPEC D6 / s10). Identical bytes for G+FS and O+FS (they read one capture block). Nothing kept -> "" (no block; the FS request
    equals the none request), with `empty_reason` eligibility / filtered / budget. Raw lines keep their practice language (may be
    Persian): an open A24.2 question, flagged per block (`has_persian`), not resolved here."""

    def __init__(self, packager: "Packager", test_ids=None):
        if packager is None:
            raise TreatmentError("FS-v2 needs the namespace's Packager (the same stores and counter as the package)")
        self.pk = packager
        self.cfg = packager.cfg
        if self.cfg.get("fs_version", "fs-v2") != "fs-v2":
            raise TreatmentError("ft-v2 renders FS as fs-v2 only (A24 item 4)")
        self._test_ids = test_ids
        self._audited = {}

    def test_ids(self) -> set:
        if self._test_ids is None:
            m = C.load_manifest(self.pk.root)
            self._test_ids = set(m.get("test") or []) | set(m.get("test_uncapped_not_scored") or [])
        return set(self._test_ids)

    def _audit(self, key: str, ex):
        if key not in self._audited:
            bad = corpus_test_citations(ex, self.test_ids())
            if bad:
                raise TreatmentError("FS corpus of %s cites test items: %s" % (key, bad[:3]))
            self._audited[key] = ex.sha()
        return self._audited[key]

    def ranker(self, stores: dict, prm: dict):
        import ltm_package as LP
        b = LP.PackageBuilder(self.pk.scene, stores, self.pk.counter, prm, categories=self.pk.categories())

        def rank(cands, query):
            ranked, dropped, _rows = b._rank("example", cands, query)
            return ranked, dropped
        rank.name = "ltm_package.PackageBuilder._rank(example)"
        return rank

    def render(self, key: str, db_path: str, rows, *, item: str, instruction: str, lang: str, holding, summary_lines) -> dict:
        import fs_block as FB
        st = self.pk.stores(key, db_path, rows)
        ex = st.get("example")
        if ex is None:
            raise TreatmentError("FS-v2: the %s source has no example store (no practice rows)" % key)
        corpus_sha = self._audit(key, ex)
        lo_log, d2_log = [], []
        src = ItemLeaveOut(ex, item, lo_log, "example", fail=self.pk.test_side)
        idx = _FSIndex(ex, src, d2=d2_on(self.cfg), log=d2_log, root=self.pk.root)
        q = fs_query(instruction, lang, holding, summary_lines)
        prm = effective_params(self.pk.profile, off=off_for(self.pk.profile, item, self.pk.mask, self.cfg, self.pk.root), cfg=self.cfg,
                               kitchen=self.pk.scene)
        cap = int(self.cfg["package_max_tokens"])
        out = FB.render_fs_v2(q, idx, self.pk.counter, cap, ranker=self.ranker(st, prm))
        text = out["text"] or ""
        probs = []
        if out["real_tokens"] > cap:
            probs.append("FS block %d tokens > %d" % (out["real_tokens"], cap))
        if d2_on(self.cfg) and (_ts().mentions_excluded(text) or excluded_alias_hits(text, self.pk.root)):
            probs.append("excluded type in the FS text")
        if probs:
            raise TreatmentError("FS of %s/%s: %s" % (item, lang, "; ".join(probs)))
        reasons = {d.get("reason") for d in out["dropped"]}
        if text:
            why = None
        elif out["candidates"] == 0 and (lo_log or d2_log):
            why = "filtered"
        elif "budget" in reasons:
            why = "budget"
        else:
            why = "eligibility"
        return dict(out, leave_out=lo_log, d2_dropped=d2_log, query_sha=C.sha256_json(q), corpus_sha=corpus_sha, source=key,
                    max_tokens=cap, parser="A17 (ltm_package._make_query_legacy)", params_sha=prm["sha256"], empty_reason=why,
                    has_persian=any("؀" <= ch <= "ۿ" for ch in text))
