"""A6: plan-only screen of the A6 cells (a6_common.A6_CELLS), in chunks (Windows; no simulator).

    python sim/final_test/strengthen/gpu_lock.py run A -- python sim/final_test/strengthen/a6_plan.py --planner gemma --ns <ns> --max-seconds 540
    python sim/final_test/strengthen/a6_plan.py --planner opus --ns test-v1 --max-seconds 540                    (no GPU; 25 USD cap)

The A3 screen (ablate_plan.Screen) with: A6 cell keys ("A6|ns|planner|arm|pres"), rows appended to docs/final-test/strengthen/A/a6/plans.jsonl,
blocks from A2 + A6, and plan once across A3 and A6 (a request a planner already answered in either journal is never sent again; the
response caches are A3's own). A chunk ends at a presentation boundary once --max-seconds have passed (exit 10: more remains; 0:
complete), or when the global STOP appears (exit 77). Identity: Kitchen 1 runs only with the recorded decision (A/decision.json);
Kitchens 2 / 3 must match their capture's treatment identity exactly (no bypass: exit 3 otherwise).
"""
import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a7_common as A6         # noqa: E402  (A7 paths and cells under the A6 names)
import ablate_common as A          # noqa: E402
import ablate_plan as AP           # noqa: E402
import gpu_lock as GL              # noqa: E402

from planners.base import CostCeilingExceeded, PlannerSkip  # noqa: E402

AP.PLANS = A6.PLANS                                                       # Screen.cell appends here (this process only)
AP.cell_key = lambda ns, planner, arm, pres: f"A7|{ns}|{planner}|{arm}|{pres}"   # noqa: E731
EXIT_MORE, EXIT_STOP = 10, 77
DECISION = os.path.join(A.DOCS, "decision.json")


class Screen6(AP.Screen):
    def __init__(self, ns: str, planner: str, ollama: str):
        self.ns, self.pl, self.out = ns, planner, print
        self.cap = A.load_capture(ns)
        self.main = A.main_plans(ns)
        self.types, _ = A.load_types(ns)
        self.contract = AP.PL.make_contract(self.types, ns)
        self.arms = list(A6.A6_CELLS[(ns, planner)])
        self.blocks = A6.all_blocks(ns)
        miss = [(a, p) for a in self.arms for p in self.cap if (a, p) not in self.blocks]
        if miss:
            raise SystemExit(f"blocks lack {len(miss)} rows for {ns} (first {miss[0]}): run a6_blocks first")
        bad = [k for k in self.blocks if k[0] in self.arms and not self.blocks[k]["checks"]["ok"]]
        if bad:
            raise SystemExit(f"{len(bad)} blocks failed their checks (first {bad[0]})")
        dec = None
        if ns == "test-v1":
            with open(DECISION, encoding="utf-8") as f:
                dec = json.load(f)["decision"]
        self.ident = self._identity(dec)
        if ns != "test-v1" and not self.ident["match"]:
            raise SystemExit(f"IDENTITY: {ns} today {self.ident['identity_now'][:12]} differs from its capture: no bypass for a held-out kitchen")
        self.ollama = ollama
        self.secrets = ()
        self.prov = self.runner = None
        self.rows = A.read_jsonl(A6.PLANS)
        self.done = {r["key"] for r in self.rows}
        self.calls = {}
        for r in A6.all_plans(ns):                                        # plan once across A3 and A6
            if r.get("planner") == planner and r.get("source") == "call":
                self.calls.setdefault(r["request_sha"], r)

    def run_chunk(self, max_seconds: float) -> int:
        order = sorted(self.cap, key=lambda p: A.sha256_text(f"{A.SEED}|A6|{self.ns}|{p}"))
        t0, n = time.time(), 0
        for p in order:
            if all(AP.cell_key(self.ns, self.pl, a, p) in self.done for a in self.arms):
                continue
            if GL.stop_requested() or os.path.exists(A6.A6_STOP):
                self._resume_line()
                return EXIT_STOP
            if time.time() - t0 > max_seconds:
                self._resume_line()
                return EXIT_MORE
            for arm in self.arms:
                r = self.cell(arm, p)
                if r is not None:
                    n += 1
                    if r["source"] == "call":
                        self.out(f"  {arm:<16s} {p:<44s} {r['status']:<8s} usd={r['cost_usd']:.4f} differs={r['differs_from_full']}")
        self._resume_line()
        return 0

    def remaining(self) -> int:
        return sum(1 for p in self.cap for a in self.arms if AP.cell_key(self.ns, self.pl, a, p) not in self.done)

    def _resume_line(self):
        mine = [r for r in self.rows if r.get("ns") == self.ns and r.get("planner") == self.pl]
        A6.resume(f"A7 planning {AP.SPEC[self.pl]} {self.ns}: {len(mine)} cells journaled, {self.remaining()} left",
                  [f"A6 plans so far: {len(A.read_jsonl(A6.PLANS))} rows in A/a6/plans.jsonl"],
                  "python sim/final_test/strengthen/a7_drive.py")


def new_calls(s) -> int:
    """Distinct requests of the screen's remaining cells that no journal row answers yet (each costs one first call)."""
    seen = set(s.calls)
    n = 0
    for p in s.cap:
        for arm in s.arms:
            if AP.cell_key(s.ns, s.pl, arm, p) in s.done:
                continue
            b = s.blocks[(arm, p)]
            if arm != "rep" and b["same_request_as"] in ("S*", "none"):
                continue
            if b["request_sha"] not in seen:
                seen.add(b["request_sha"])
                n += 1
    return n


def project(namespaces) -> int:
    """Opus projection before a paid step: committed spend (cumulative ledger) + new calls x the mean cost of the Opus calls made so far
    (x 1.15 for repairs and spread). Exit 5 when it would cross the cap: the driver stops and the coordinator is told."""
    calls = [r for r in A6.all_plans() if r.get("planner") == "O" and r.get("source") == "call"]
    mean = sum(float(r.get("cost_usd") or 0) for r in calls) / max(1, len(calls))
    need = {ns: new_calls(Screen6(ns, "O", "")) for ns in namespaces}
    spent = A.opus_spent()
    proj = spent + 1.15 * mean * sum(need.values())
    out = {"committed_usd": round(spent, 4), "mean_usd_per_call": round(mean, 5), "new_calls": need, "projected_total_usd": round(proj, 2),
           "cap_usd": A6.OPUS_CAP_USD, "ok": proj <= A6.OPUS_CAP_USD}
    A.atomic_write_json(os.path.join(A6.WORK, f"opus_projection_{'_'.join(namespaces)}.json"), dict(out, written=time.strftime("%Y-%m-%d %H:%M:%S")))
    print(json.dumps(out))
    return 0 if out["ok"] else 5


def main(argv=None) -> int:
    if argv is None and len(sys.argv) > 1 and sys.argv[1] == "project":
        return project(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--planner", required=True, choices=("gemma", "opus"))
    ap.add_argument("--ns", required=True, choices=sorted(A6.KITCHEN_OF))
    ap.add_argument("--max-seconds", type=float, default=540)
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    pl = "G" if a.planner == "gemma" else "O"
    if (a.ns, pl) not in A6.A6_CELLS:
        print(f"no A6 cells for {a.ns} / {a.planner}")
        return 0
    if GL.stop_requested() or os.path.exists(A6.A6_STOP):
        return EXIT_STOP
    if pl == "G" and not a.dry_run:
        import ablate_blocks as AB
        if not AB.lock_held_by_a():
            print("refused: Gemma planning runs only under `gpu_lock.py run A -- ...`", file=sys.stderr)
            return 2
    try:
        s = Screen6(a.ns, pl, a.ollama)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 3 if "IDENTITY" in str(e) else 1
    if a.dry_run:
        print(json.dumps({"ns": a.ns, "planner": a.planner, "remaining": s.remaining(), "identity": s.ident}, indent=1))
        return 0
    lk = os.path.join(A6.WORK, "a7_plan.lock")
    os.makedirs(A6.WORK, exist_ok=True)
    try:
        fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"LOCKED: another a6_plan run holds {lk}", file=sys.stderr)
        return 2
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps({"pid": os.getpid(), "planner": a.planner, "ns": a.ns}))
    try:
        try:
            s.open()
        except PlannerSkip as e:
            print(f"SKIP: {e}", file=sys.stderr)
            return 2
        try:
            return s.run_chunk(a.max_seconds)
        except CostCeilingExceeded as e:
            print(f"STOP at the Opus cap: {e}", file=sys.stderr)
            return 4
        finally:
            s.close()
    finally:
        os.remove(lk)


if __name__ == "__main__":
    sys.exit(main())
