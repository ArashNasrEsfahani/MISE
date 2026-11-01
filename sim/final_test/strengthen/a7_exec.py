"""A6: the execution passes (WSL simulator; Windows for `plan` = units + targets without a simulator). Chunked and stoppable.

    (WSL)      bash sim/final_test/strengthen/a6_exec.sh <ns> <kind> <max_seconds>      (under the Windows gpu_lock; hang-guarded)
    (Windows)  python sim/final_test/strengthen/a6_exec.py plan --ns <ns> --kind <kind>  (counts; writes the pass's units.json)

kind  new    the A6 plans' units that no existing execution covers (sealed main [+ GPT6-Sol addendum], A4, task L, earlier A6 chunks);
             executor policy of the full-memory arm (prior chooser + briefing hook B), so an arm differs from G+M only in the planner block
      rerun  no-exclusion replacements: every unit an ablation cell (A3 + A6) adopts whose final run-0 history is three reset-gate stops
             and that task R has not re-run; fresh attempts 0-2 in this pass's own journal (the R way)
      noise  Kitchen 1 execution-noise floor: 60 sealed G+M units (seeded sample of the units whose sealed run 0 finished), executed again

Every pass: execute.ExecutePhase (same restored worlds, reset gate, attempts 0-2, live executor), journals in A/a6/<ns>/<kind>/, the exec
row is the commit (a stop repeats only the unit in flight). Per attempt (task R's rules, reused from rerun_reset.py): 0 gateway violations,
0 oracle reads, 0 map-build reads, frozen hashes unchanged (full sha256 check before and after the chunk, size / mtime per attempt), the
simulator on the GPU renderer when it starts; plus the namespace's treatment identity unchanged and the sealed main journals unchanged.
Any breach: the attempt is INVALID and the pass stops (exit 4). Kitchen 1 runs on Arash's same-executor ruling exactly as task R applies it
(guard / identity bypass and the A34-A2 settings equivalence, in this process only, recorded); Kitchens 2 / 3 run with no bypass at all.
A chunk stops at a unit boundary after --max-seconds (exit 10: more remains; 0: pass complete) or when the global STOP appears (exit 77).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a7_common as A6         # noqa: E402  (A7 paths and sources under the A6 names)
import ablate_common as A          # noqa: E402

import common as C                 # noqa: E402
import treatment as T              # noqa: E402

EXIT_MORE, EXIT_STOP, EXIT_ABORT = 10, 77, 4
KINDS = ("new", "rerun", "noise")
NOISE_N, NOISE_SEED = 60, 20261028
FULL_ARM = {"G": "G+M", "O": "O+M"}
PASS_ID = {"new": "strengthen-A7-new", "rerun": "strengthen-A7-rerun", "noise": "strengthen-A7-noise"}


def stop_requested() -> bool:
    """The global STOP (stop_all.py) or task A's own STOP_A6 (a6_common.A6_STOP)."""
    return os.path.exists(os.path.join(A6.STRENGTHEN, "STOP")) or os.path.exists(A6.A6_STOP)


def plan_rows(ns: str, ident_sha: str, phases=("A7",)) -> dict:
    """key -> plan-like row for build_units (arm = the full-memory arm: its executor policy; block 'ablation': one run, no REPEAT)."""
    out = {}
    for r in A6.all_plans(ns):
        if r.get("a_phase") not in phases or r.get("status") != "ok":
            continue
        out[r["key"]] = {"key": r["key"], "arm": FULL_ARM[r["planner"]], "pres": r["pres"], "r": 0, "block": "ablation", "status": "ok",
                         "plan": r["plan"], "treatment": T.FT2_VERSION, "treatment_id": ident_sha}
    return out


def _execute_cls():
    import execute as XE
    import rerun_reset as RR

    class A6Execute(XE.ExecutePhase):
        def __init__(self, ns: str, kind: str, **kw):
            super().__init__(ns, allow_test=True, worlds=("canon",), p_arms=(), replay=False, **kw)
            self.kind = kind
            self.xp = A6.pass_paths(ns, kind)
            os.makedirs(self.xp["work"], exist_ok=True)
            os.makedirs(self.xp["dir"], exist_ok=True)
            self.work = self.xp["work"]
            self.main = {k: self.J[k] for k in ("plan", "exec", "outcome", "events")}
            self.J = {"capture": self.J["capture"], "plan": self.J["plan"], "exec": C.Journal(self.xp["exec"]),
                      "outcome": C.Journal(self.xp["outcome"]), "events": C.Journal(self.xp["events"])}
            self.units_path = self.xp["units"]
            self.fs = self.fs_stat = None
            self.shas0 = None
            self.k1_bypass = None

        # ---- inputs / guards -------------------------------------------------------------------------------------------
        def _epochs(self):
            self.epoch = {}
            rows = self.main["events"].rows() + self.J["events"].rows()
            for r in rows:
                if r.get("event") == "invalid_resolved":
                    for p in r.get("pres") or []:
                        self.epoch[p] = self.epoch.get(p, 0) + 1
            self.unrecoverable = C.invalid_pres(rows)

        def preflight(self) -> dict:
            mev = self.main["events"].rows()
            for ph in ("capture", "plan", "execute", "score"):
                if not [e for e in mev if e.get("event") == "phase_complete" and e.get("phase") == ph]:
                    raise XE.ExecAborted(f"the main test of {self.ns} has no {ph} phase_complete")
            self.captures = {r["pres"]: r for r in self.J["capture"].rows() if r.get("capture_run") == 1}
            self.ident = T.identity(self.ns, self.root, briefing=None)
            ids = sorted({C.parse_pres(p)["item"] for p in self.captures})
            try:
                guard = C.guard_items(ids, self.ns, allow_test=True, root=self.root)
            except C.TestItemRefused as e:
                if self.ns != "test-v1":
                    raise                                             # a held-out kitchen never runs with a bypass
                # task R's rule, verbatim in substance: only the freeze-drift reasons of the authorized K23 re-pin era are bypassable
                allowed = set(C.namespace_items(self.ns, C.load_manifest(self.root), self.root))
                outside = sorted(i for i in ids if i not in allowed)
                reasons = C.freeze_status(self.root)["reasons"]
                fam = C.families_status(self.root)["reasons"]
                bad = [r for r in reasons if not (r == "preregistration changed after the freeze" or r.startswith("code changed after the freeze: ")
                                                  or (r.startswith("artefact ") and r.endswith(" changed after the freeze")))]
                if outside or fam or bad or not reasons:
                    raise C.TestItemRefused(f"{e} (not bypassable: outside {outside[:3]}, families {fam[:2]}, other {bad[:3]})") from None
                self.k1_bypass = {"guard_reasons": reasons}
                guard = {"n": len(ids), "guard_bypassed": True, "ruling": RR.K1_RULING["id"]}
            self._epochs()
            cs = [e for e in mev if e.get("event") == "capture_start"]
            dp = C.dataset_problems(self.ns, self.root, (cs[-1].get("dataset_sha256") if cs else None))
            if dp:
                raise XE.ExecAborted("; ".join(dp))
            if self._invalid_open():
                raise XE.ExecAborted("an earlier invalid attempt of this pass is unresolved")
            self._treatment_check()
            self.fs = RR.frozen_set(self.ns, self.root)
            probs, self.fs_stat = RR.frozen_problems(self.fs, self.root, full=True)
            if probs:
                raise XE.ExecAborted("frozen hashes differ before the chunk: " + "; ".join(probs[:4]))
            self.shas0 = self._main_shas()
            return guard

        def _treatment_check(self):
            cap_ids = {r.get("treatment_id") for r in self.captures.values()}
            if cap_ids != {self.ident["sha"]}:
                if self.ns != "test-v1":
                    raise XE.ExecAborted(f"{self.ns}: treatment identity {self.ident['sha'][:12]} differs from its capture "
                                         f"{sorted(str(x)[:12] for x in cap_ids)}: no bypass for a held-out kitchen")
                if cap_ids != {RR.K1_RULING["test_identity"]} or self.ident["sha"] != RR.K1_RULING["running_identity"]:
                    raise XE.ExecAborted("test-v1: the identities are not exactly the ruling's (test 7a2ee211 -> today 9799088)")
                self.k1_bypass = dict(self.k1_bypass or {}, identity={"test": RR.K1_RULING["test_identity"], "running": self.ident["sha"]})
                self._install_a34_equivalence()
            bad = T.journal_problems(list(self.exec_latest().values()), self.ns, self.ident["sha"])
            if bad:
                raise XE.ExecAborted("treatment mismatch inside this pass's journal: " + "; ".join(bad[:4]))

        def _install_a34_equivalence(self):
            """Task R's rule under the ruling: A34-A2 2(b) compares the pipelines' settings with today's identity mapped to the test's."""
            from final_test import reset_gate as rg
            if getattr(rg.pipeline_settings, "_k1_ruling", False):
                return
            orig = rg.pipeline_settings

            def settings(obs):
                if isinstance(obs, dict) and obs.get("treatment_id") == RR.K1_RULING["running_identity"]:
                    obs = dict(obs, treatment_id=RR.K1_RULING["test_identity"])
                return orig(obs)
            settings._k1_ruling = True
            rg.pipeline_settings = settings

        def _main_shas(self) -> dict:
            d = A.main_dir(self.ns)
            return {f: A.sha256_file(os.path.join(d, f)) for f in ("capture.jsonl", "plan.jsonl", "exec.jsonl", "outcome.sealed.jsonl",
                                                                    "events.jsonl", "score.jsonl", "units.json")
                    if os.path.exists(os.path.join(d, f))}

        # ---- units / targets ---------------------------------------------------------------------------------------------
        def _build_from(self, plans: dict) -> dict:
            man = dict(C.load_manifest(self.root))
            hsha = T.hint_table_sha(self.briefing_table())
            b = XE.build_units(ns=self.ns, captures=self.captures, plans=plans, items=self.items(), snap_sha=self._snap_sha, manifest=man,
                               p_arms=(), replay=False, worlds=("canon",), reloc_seed=None, cuts=frozenset(), hint_sha=hsha)
            for u in b["units"].values():
                u["runs"] = 1
            return b

        def build(self) -> dict:
            if self.kind == "new":
                b = self._build_from(plan_rows(self.ns, self.ident["sha"], ("A7",)))
                srcs = A6.sources(self.ns, include_a6_new=False)
                for uk, u in b["units"].items():
                    s, _h = A6.adopt(self.ns, uk, srcs)
                    u["target"], u["covered_by"] = s is None, (s.name if s else None)
            elif self.kind == "rerun":
                b = self._build_from(plan_rows(self.ns, self.ident["sha"], ("A7",)))
                srcs = A6.sources(self.ns, include_a6_new=True)
                rr = A6.rerun_sources(self.ns)[0]                        # task R's pass
                for uk, u in b["units"].items():
                    s, h = A6.adopt(self.ns, uk, srcs)
                    blocked = bool(h) and A6.three_blocked(h)
                    u["adopted"] = s.name if s else None
                    u["target"] = blocked and rr.get(A6.exec_key(self.ns, uk)) is None
                    u["covered_by"] = ("R" if blocked and not u["target"] else None)
            else:                                                        # noise
                if self.ns != "test-v1":
                    raise XE.ExecAborted("the noise floor is a Kitchen 1 pass")
                b = {"units": self._noise_units()}
            for u in b["units"].values():
                u.setdefault("target", False)
            doc = {"namespace": self.ns, "kind": self.kind, "pass_id": PASS_ID[self.kind], "units": b["units"],
                   "n_units": len(b["units"]), "n_targets": sum(1 for u in b["units"].values() if u["target"]),
                   "treatment_id": self.ident["sha"], "k1_bypass": self.k1_bypass,
                   "noise_seed": NOISE_SEED if self.kind == "noise" else None}
            A.atomic_write_json(self.units_path, doc)
            self.b = b
            return b

        def _noise_units(self) -> dict:
            mu = json.load(open(os.path.join(A.main_dir(self.ns), "units.json"), encoding="utf-8"))["units"]
            main = A6.sources(self.ns)[0]
            pool = []
            for uk, u in sorted(mu.items()):
                if not any(c["arm"] == "G+M" and c["r"] == 0 and c["block"] == "main" for c in u["cells"]):
                    continue
                h = main.get(A6.exec_key(self.ns, uk))
                if h and h[-1].get("status") == "done":
                    pool.append(uk)
            pick = sorted(random.Random(NOISE_SEED).sample(pool, NOISE_N))
            out = {}
            for uk in pick:
                u = dict(mu[uk])
                u["runs"], u["target"], u["noise_pool"] = 1, True, len(pool)
                out[uk] = u
            return out

        def cell_state(self, unit: dict, run: int) -> tuple:
            if not unit.get("target"):
                return "done", 0
            return super().cell_state(unit, run)

        def _append(self, kind: str, rec: dict) -> None:
            if kind in ("exec", "outcome"):
                rec = dict(rec, pass_id=PASS_ID[self.kind])
                if self.k1_bypass:
                    rec["k1_ruling"] = RR.K1_RULING["id"]
            super()._append(kind, rec)

        # ---- execution (task R's per-attempt invariants) -----------------------------------------------------------------
        def _executor(self):
            new = self.ex is None
            ex = super()._executor()
            if new:
                boot = getattr(ex, "_boot", None)
                if callable(boot):
                    boot()
                st = RR.renderer_status()
                self.event("renderer_check", status=st, pass_id=PASS_ID[self.kind])
                if st is not None and not st["gpu"]:
                    self._drop_executor()
                    raise XE.ExecAborted(f"the simulator did not start on the GPU renderer ({json.dumps(st)[:200]}): keep the display on")
            return ex

        def run_one(self, u: dict, run: int, attempt: int, *, tag=None) -> tuple:
            rec, outc, dig = super().run_one(u, run, attempt, tag=tag)
            why = []
            aud = rec.get("audit") or {}
            cnt = aud.get("counts") if isinstance(aud.get("counts"), dict) else None
            if rec.get("status") in ("done", "blocked_state", "not_executed"):
                if cnt is None or aud.get("violations") is None:
                    why.append("audit counts missing")
                else:
                    for k, v in (("violations", aud.get("violations")), ("oracle", cnt.get("oracle")), ("map-build", cnt.get("map-build"))):
                        if int(v or 0) != 0:
                            why.append(f"{k} = {v}")
                    if aud.get("invalid"):
                        why.append("audit invalid")
            probs, _ = RR.frozen_problems(self.fs, self.root, full=False, stat0=self.fs_stat)
            why += probs
            now = T.identity(self.ns, self.root, briefing=None)["sha"]
            if now != self.ident["sha"]:
                why.append(f"treatment identity changed ({self.ident['sha'][:12]} -> {now[:12]})")
            if self._main_shas() != self.shas0:
                why.append("a sealed main journal changed")
            sc = rec.get("state_check") or {}
            if self.ns == "test-v1" and sc.get("override") is True:
                rec["identity_equivalence_override"] = ((sc.get("obs_pipeline") or {}).get("treatment_id") != RR.K1_RULING["test_identity"])
            if why and rec.get("status") != "invalid":
                rec["status"], rec["error"] = "invalid", ("strengthen-A7 invariant: " + "; ".join(why))[:300]
                outc = None
            return rec, outc, dig

        def do_cell(self, u: dict, run: int) -> dict:
            if self.factory is not None:
                self._executor()
            return super().do_cell(u, run)

        def remaining(self) -> int:
            return sum(len(us) for blocks in self.todo().values() for _, us in blocks)

        def chunk(self, max_seconds: float, dry_run: bool = False) -> tuple:
            guard = self.preflight()
            self.build()
            info = {"ns": self.ns, "kind": self.kind, "units": len(self.b["units"]),
                    "targets": sum(1 for u in self.b["units"].values() if u["target"]), "todo": self.remaining(), "executed": 0}
            if dry_run:
                return 0, info
            lk = C.PhaseLock(self.ns, XE.PHASE, self.root, steal=getattr(self, "steal", False))
            lk.path = self.xp["lock"]
            t0 = time.time()
            code = 0
            with lk:
                self.event("a7_chunk_start", pass_kind=self.kind, todo=info["todo"], pass_id=PASS_ID[self.kind], treatment_id=self.ident["sha"],
                           guard=guard, k1_bypass=self.k1_bypass,
                           ruling=(RR.K1_RULING if self.k1_bypass else None))
                try:
                    for pres, unit_keys in self.todo()[0]:
                        for uk in unit_keys:
                            if stop_requested():
                                code = EXIT_STOP
                                raise StopIteration
                            if time.time() - t0 > max_seconds:
                                code = EXIT_MORE
                                raise StopIteration
                            rec = self.do_cell(self.b["units"][uk], 0)
                            info["executed"] += 1
                            self.out(json.dumps({"t": time.strftime("%H:%M:%S"), "kind": self.kind, "pres": pres, "unit": uk[:10],
                                                 "status": rec["status"], "attempt": rec["attempt"], "s": rec["wall_s"]}))
                except StopIteration:
                    pass
                finally:
                    self._drop_executor()
                probs, self.fs_stat = RR.frozen_problems(self.fs, self.root, full=True)
                if probs:
                    raise XE.ExecInvalid("frozen hashes differ after the chunk: " + "; ".join(probs[:4]))
                left = self.remaining()
                info["left"] = left
                if left == 0 and code == 0:
                    sha = C.sha256_file(self.J["exec"].path) if os.path.exists(self.J["exec"].path) else ""
                    self.event("phase_complete", pass_kind=self.kind, n=info["targets"], journal_sha=sha, units_sha=C.sha256_file(self.units_path),
                               pass_id=PASS_ID[self.kind], frozen="full sha256 check passed before and after the chunk")
                elif code == 0:
                    code = EXIT_MORE
            return code, info

    return A6Execute, XE


def run(ns: str, kind: str, max_seconds: float, dry_run: bool = False, steal_lock: bool = False) -> int:
    if stop_requested() and not dry_run:
        return EXIT_STOP
    Ex, XE = _execute_cls()
    if ns != "test-v1":
        import k23_run as R
        import plan as PL
        PL.load_types = R.k23_load_types(ns)
    ph = Ex(ns, kind, executor_factory=None if dry_run else XE.live_executor_factory,
            memory_opener=None if dry_run else XE.live_memory_opener)
    ph.steal = bool(steal_lock)
    try:
        code, info = ph.chunk(max_seconds, dry_run=dry_run)
    except C.TestItemRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    except C.PhaseLocked as e:
        print(f"LOCKED: {e}", file=sys.stderr)
        return 2
    except (XE.ExecAborted, XE.ExecInvalid) as e:
        print(f"ABORTED: {e}", file=sys.stderr)
        return EXIT_ABORT
    print(json.dumps(info))
    return code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("plan", "execute"))
    ap.add_argument("--ns", required=True, choices=sorted(A6.KITCHEN_OF))
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--max-seconds", type=float, default=540)
    ap.add_argument("--steal-lock", action="store_true")
    a = ap.parse_args(argv)
    return run(a.ns, a.kind, a.max_seconds, dry_run=a.cmd == "plan", steal_lock=a.steal_lock)


if __name__ == "__main__":
    sys.exit(main())
