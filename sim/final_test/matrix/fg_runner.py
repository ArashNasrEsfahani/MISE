"""FREEZE-GATES-FT2 (2026-10-15): the runner of the two owed freeze gates, G-iso and 12.5, under the FINAL treatment identity (7a2ee211...)
in the dev namespace `freeze-gates-ft2` (common.NAMESPACES; dev_pilot items, Gemma only, 0 USD). A thin layer over the FROZEN phases:
capture.py and execute.py stay byte-identical at their signed pins (A34-A2 + A35); nothing here changes the treatment.

  capture   capture.main itself. Its `--items dev` check admits only capture.DEV_NAMESPACES (a CLI consistency check, not a rule), so this
            process adds `freeze-gates-ft2` to that tuple (only here, only for this namespace) and calls capture.main unchanged.
  execute   execute.ExecutePhase, subclassed for four things only:
            (a) E0, the gate-12.5 (b) empty-plan control: one unit per captured presentation whose plan is EMPTY, through the same restore,
                state check, replay hook and scoring() goal read as every other unit; its exec row carries plan [] so score.py (unchanged)
                scores it as an empty-plan row (plan_len 0);
            (b) --only-arms: execute only the cells of these arms (P0,E0 for 12.5); an arm subset never writes phase_complete
                (event arm_subset_complete instead);
            (c) the A26 12.5 registration (docs/final-test/a26_12_5_registration.json) must exist and be intact BEFORE an E0 run executes
                anything; its file sha is recorded as event a26_12_5_registration first;
            (d) --g-iso: the report gets the treatment identity it ran under (treatment_id), re-sealed (self sha) by the same process.

    python sim/final_test/matrix/fg_runner.py capture --namespace freeze-gates-ft2 --items dev --runs 1 --worlds canon      (WSL)
    python sim/final_test/matrix/fg_runner.py execute --namespace freeze-gates-ft2 --items dev --partial --worlds canon --g-iso --g-iso-n 12   (WSL)
    python sim/final_test/matrix/fg_runner.py execute --namespace freeze-gates-ft2 --items dev --partial --worlds canon --p-arms P0,E0 --only-arms P0,E0

Selftest: sim/_ftm_freeze_gates_selftest.py (offline). Python 3.10 syntax (WSL).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                              # noqa: E402
import execute as X                                             # noqa: E402

NS = "freeze-gates-ft2"
EMPTY_ARM = "E0"
A26_125_REGISTRATION = ("docs", "final-test", "a26_12_5_registration.json")
REG_SCHEMA = "ftm.a26-12.5-registration/1"


def a26_registration(root: str | None = None) -> dict:
    """The A26 12.5 registration must exist and be intact before an empty-plan-control run starts. -> {path, sha256, self_sha256,
    written_utc}; X.ExecAborted otherwise."""
    p = os.path.join(C.paths(root)["root"], *A26_125_REGISTRATION)
    try:
        with open(p, encoding="utf-8") as f:
            reg = json.load(f)
    except (OSError, ValueError) as e:
        raise X.ExecAborted(f"the A26 12.5 registration is missing or unreadable ({type(e).__name__}): write it before the 12.5 run")
    if reg.get("schema") != REG_SCHEMA or reg.get("sha256") != C.self_sha(reg):
        raise X.ExecAborted("the A26 12.5 registration is not intact (schema / self sha)")
    return {"path": "/".join(A26_125_REGISTRATION), "sha256": C.sha256_file(p), "self_sha256": reg["sha256"], "written_utc": reg.get("written_utc")}


def empty_only(u: dict) -> bool:
    return bool(u.get("cells")) and all(c["arm"] == EMPTY_ARM for c in u["cells"])


class FGExecutePhase(X.ExecutePhase):
    def __init__(self, namespace: str, *, only_arms=(), **kw):
        super().__init__(namespace, **kw)
        self.only_arms = tuple(only_arms or ())

    def _man(self) -> dict:
        man = dict(C.load_manifest(self.root))
        if self.repeat_override is not None:
            man["repeat_subset"] = list(self.repeat_override)
        return man

    def build(self) -> dict:
        p_all, e0 = self.p_arms, EMPTY_ARM in self.p_arms
        self.p_arms = tuple(a for a in p_all if a != EMPTY_ARM)           # the frozen build sees only the registered P arms
        try:
            b = super().build()
        finally:
            self.p_arms = p_all
        man = self._man()
        if e0:                                                             # (a) the same unit builder, with the EMPTY plan as the "reference"
            empty = {k: dict(v, plan=[]) for k, v in self.items().items()}
            eb = X.build_units(ns=self.ns, captures=self.captures, plans={}, items=empty, snap_sha=self._snap_sha, manifest=man,
                               p_arms=(EMPTY_ARM,), replay=False, worlds=self.worlds, reloc_seed=self.reloc_seed, cuts=self.cuts, hint_sha=None)
            for k, u in eb["units"].items():
                if k in b["units"]:                                        # an identical unit (an empty cached plan) is shared, as prereg s7 C
                    b["units"][k]["cells"] = sorted(b["units"][k]["cells"] + u["cells"], key=C.canon)
                else:
                    b["units"][k] = u
        if self.only_arms:                                                 # (b) the arm subset
            rep, kept = set(C.repeat_items(self.ns, man)), {}
            for k, u in b["units"].items():
                cells = [c for c in u["cells"] if c["arm"] in self.only_arms]
                if cells:
                    rr = any(u["item"] in rep and c["block"] in ("main", "repeat") and C.repeat_exec_arm(c["arm"]) for c in cells)
                    kept[k] = dict(u, cells=cells, repeat=rr, runs=X.REPEAT_RUNS if rr else 1)
            b["units"] = kept
        if e0 or self.only_arms:
            with open(self.units_path, encoding="utf-8") as f:
                doc = json.load(f)
            doc.update(units=b["units"], p_arms=list(self.p_arms), only_arms=list(self.only_arms), runner="fg_runner.py (FREEZE-GATES-FT2)")
            X._atomic_json(self.units_path, doc)
        self.b = b
        return b

    def run_one(self, u: dict, run: int, attempt: int, *, tag=None) -> tuple:
        rec, outc, dig = super().run_one(u, run, attempt, tag=tag)
        if empty_only(u):
            rec["plan"] = []                                               # score.py reads the executed plan of a P-style cell here
        return rec, outc, dig

    def run(self, stop_after: int | None = None, dry_run: bool = False) -> dict:
        if EMPTY_ARM in self.p_arms and not dry_run:                       # (c) registered BEFORE anything of the 12.5 run executes
            self.event("a26_12_5_registration", **a26_registration(self.root))
        return super().run(stop_after=stop_after, dry_run=dry_run)

    def _finish(self, info: dict) -> None:
        if not self.only_arms:
            return super()._finish(info)
        for kind in ("exec", "outcome", "events"):
            p = self.J[kind].path
            if os.path.exists(p) and C.scan_jsonl_leaks(p, self._secrets()):
                raise X.ExecInvalid(f"key-like text found in {p}: arm_subset_complete withheld")
        sha = C.sha256_file(self.J["exec"].path) if os.path.exists(self.J["exec"].path) else ""
        self.event("arm_subset_complete", arms=list(self.only_arms), n=info["cells"], journal_sha=sha, units=info["units"])
        info["complete"] = False                                           # never the phase: no phase_complete for an arm subset

    def g_iso(self, n: int = 12, conds=("absent", "present", "absent")) -> dict:
        rep = super().g_iso(n=n, conds=conds)
        if self.ft2:                                                       # (d) bind the report to the identity it ran under
            rep = {k: v for k, v in rep.items() if k != "sha256"}
            rep["treatment_id"] = (self.ident or {}).get("sha")
            rep["sha256"] = C.self_sha(rep)
            X._atomic_json(os.path.join(self.docs, "g_iso.json"), rep)
            self.event("g_iso_identity", treatment_id=rep["treatment_id"], report_sha256=rep["sha256"])
        return rep


def main_execute(argv) -> int:
    ap = argparse.ArgumentParser(description="FREEZE-GATES-FT2 execute (WSL only; no LLM): G-iso, or the P0 + E0 run of gate 12.5")
    ap.add_argument("--namespace", required=True, choices=(NS,))
    ap.add_argument("--items", choices=("dev",), default="dev")
    ap.add_argument("--worlds", default="canon", choices=("canon",))
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--p-arms", default=",".join(X.P_ARMS))
    ap.add_argument("--only-arms", default="")
    ap.add_argument("--g-iso", action="store_true")
    ap.add_argument("--g-iso-n", type=int, default=12)
    ap.add_argument("--steal-lock", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    ph = FGExecutePhase(a.namespace, worlds=("canon",), p_arms=tuple(x for x in a.p_arms.split(",") if x), partial=a.partial,
                        executor_factory=None if a.dry_run else X.live_executor_factory,
                        memory_opener=None if a.dry_run else X.live_memory_opener, only_arms=tuple(x for x in a.only_arms.split(",") if x))
    ph.steal = a.steal_lock
    try:
        if a.g_iso:
            rep = ph.g_iso(n=a.g_iso_n)
            print(json.dumps({k: rep.get(k) for k in ("status", "n", "diff", "control_noise", "blocked", "treatment_id")}))
            return X.EXIT["ok"] if rep["status"] == "pass" else X.EXIT["abort_invalid"]
        info = ph.run(dry_run=a.dry_run)
        print(json.dumps({k: v for k, v in info.items() if k != "guard"}))
        return X.EXIT["ok"]
    except C.TestItemRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return X.EXIT["refused"]
    except C.PhaseLocked as e:
        print(f"LOCKED: {e}", file=sys.stderr)
        return X.EXIT["refused"]
    except X.ExecAborted as e:
        print(f"ABORTED: {e}", file=sys.stderr)
        return X.EXIT["abort_invalid"]
    except X.ExecInvalid as e:
        with contextlib.suppress(Exception):
            ph.event("abort_key_leak" if "key-like" in str(e) else "abort_invalid", msg=str(e)[:300])
        print(f"ABORTED (integrity): {e}", file=sys.stderr)
        return X.EXIT["abort_invalid"]


def main_capture(argv) -> int:
    if "--namespace" not in argv or argv[argv.index("--namespace") + 1:argv.index("--namespace") + 2] != [NS]:
        print(f"fg_runner capture runs namespace {NS} only", file=sys.stderr)
        return 2
    if C.NAMESPACES[NS]["items"] != "dev_pilot":
        print(f"{NS} is not a dev namespace", file=sys.stderr)
        return 2
    import capture as CP
    CP.DEV_NAMESPACES = tuple(CP.DEV_NAMESPACES) + (NS,)                 # this process only: admit the registered dev namespace
    return CP.main(argv)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("capture", "execute"):
        print(__doc__)
        return 2
    return (main_capture if argv[0] == "capture" else main_execute)(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
