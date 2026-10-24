"""Selftest of the C5 SIMPLE energy method (sim/final_test/matrix/energy.py, method c5-simple-1) and its analyze join (stage C5-SIMPLE,
2026-10-14). Offline: a fake nvidia-smi reading list on a simulated clock and a fake Plan class with the real _call_cell signature.
No hardware, no Ollama, no simulator.  python sim/_ftm_energy_selftest.py
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(HERE, "final_test", "matrix")
sys.path.insert(0, HERE)
sys.path.insert(0, MATRIX)

import common as C                      # noqa: E402
import analyze as A                     # noqa: E402
import energy as EN                     # noqa: E402
import matrix_run as MR                 # noqa: E402

N = {"pass": 0, "fail": 0}


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -- {detail}"))


def raises(f):
    try:
        f()
    except Exception:                                                     # noqa: BLE001
        return True
    return False


def close(a, b, tol=1e-6):
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(b))


class Sim:
    """Simulated time; GPU power is piecewise constant (set_w); nvidia-smi 'prints' a reading every 100 ms."""

    def __init__(self, w=40.0):
        self.t, self.segs = 1000.0, [(-1e18, w)]

    def set_w(self, w):
        self.segs.append((self.t, w))

    def w_at(self, t):
        return [x for ts, x in self.segs if ts <= t][-1]

    def clock(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class FakeSmi:
    """energy.NvidiaSmiPower's interface (open / close / window) on the simulated clock; dead = (t0, t1) with no readings."""

    def __init__(self, sim, fail_open=False, dead=None):
        self.sim, self.fail_open, self.dead, self.closed = sim, fail_open, dead, 0

    def open(self):
        if self.fail_open:
            raise EN.MeterUnavailable("nvidia-smi not found (fake)")
        return {"name": "FakeGPU"}

    def close(self):
        self.closed += 1

    def window(self, p0, p1, pad=EN.NEAR_S):
        out, k = [], int((p0 - pad) * 10) - 1
        while k / 10.0 <= min(p1 + pad, self.sim.t):
            t = k / 10.0
            if t >= p0 - pad and not (self.dead and self.dead[0] <= t <= self.dead[1]):
                out.append((t, self.sim.w_at(t)))
            k += 1
        return out


REAL_METER = EN.Meter


def meter(sim, **kw):
    return REAL_METER(kw.pop("gpu", None) or FakeSmi(sim), clock=sim.clock, perf=sim.clock, sleep=sim.sleep)


# ---------------------------------------------------------------------------------------------------- 1. the method, as registered
em = A.ENERGY_METHOD
check("energy.METHOD_ID = analyze ENERGY_METHOD id = c5-simple-1; A24_2_COST carries it", EN.METHOD_ID == em["id"] == "c5-simple-1"
      and A.A24_2_COST["energy_method"] is em and A.A24_2_COST["usd_per_kwh"] == 0.20)
check("nvidia-smi command: power.draw.instant of GPU 0 every 100 ms", EN.NVSMI_CMD[:2] == ["nvidia-smi", "--query-gpu=power.draw.instant"]
      and "-lms" in EN.NVSMI_CMD and EN.NVSMI_CMD[EN.NVSMI_CMD.index("-lms") + 1] == "100")
dis = em["disclosure"]
check("disclosure says plainly: simple, approximate, nvidia-smi average power x planning time, 0.20 USD/kWh, GPU only (understates), "
      "idle included, 10 practice rows estimated and flagged", all(w in dis for w in ("simple, approximate", "nvidia-smi", "multiplied by the "
      "planning time", "0.20 USD/kWh", "Only the GPU", "understates", "idle power is included", "10 practice", "estimated", "flagged")))
check("the decision is Arash's, dated, cited from the ledger; the old checks are not claimed", "Arash, 2026-10-14" in em["decision"]
      and "checks" not in em and "calibration" not in em and "no calibration, no coverage or gap checks" in em["accuracy"])
check("the practice-row rule is ACTIVE under Arash's decision with the approved terms (10 practice-v2 rows, headline, measured rate)",
      em["unmeasured_rule_basis"] == "arash-decision:2026-10-14" and em["unmeasured_rule_terms"] == A.UNMEASURED_TERMS_APPROVED
      and A.approved_terms(A.A24_2_COST) == A.UNMEASURED_TERMS_APPROVED and len(A.UNMEASURED_TERMS_APPROVED["rows"]["row_ids"]) == 10)

# ---------------------------------------------------------------------------------------------------- 2. mean power x time
check("mean_power_j: 40 W readings over 2 s = 80 J (3 readings)", EN.mean_power_j([(0.0, 40.0), (1.0, 40.0), (2.0, 40.0)], 0.0, 2.0) == (40.0, 80.0, 3))
check("mean_power_j: plain mean of the readings IN the interval (10 W, 30 W -> 20 W x 1 s); readings outside ignored",
      EN.mean_power_j([(-0.5, 99.0), (0.2, 10.0), (0.8, 30.0), (1.5, 99.0)], 0.0, 1.0) == (20.0, 20.0, 2))
check("mean_power_j: no reading inside -> the readings within 1 s; none at all or t1 <= t0 -> None",
      EN.mean_power_j([(1.4, 50.0)], 0.0, 0.5) == (50.0, 25.0, 1) and EN.mean_power_j([(9.0, 50.0)], 0.0, 0.5) == (None, None, 0)
      and EN.mean_power_j([(0.1, 50.0)], 1.0, 1.0) == (None, None, 0))
sim = Sim(40.0)
m = meter(sim)
info = m.open()
check("Meter.open with nvidia-smi -> ok, method id", info["ok"] and info["method"] == "c5-simple-1" and info["problems"] == [])
a = m.edge()
sim.sleep(1.2)
sim.set_w(80.0)
sim.sleep(0.8)
b = m.edge()
iv = m.interval(a, b)
n_lo, n_hi = sum(1 for t, _ in FakeSmi(sim).window(a["p"], b["p"], 0) if t < 1001.2), sum(1 for t, _ in FakeSmi(sim).window(a["p"], b["p"], 0) if t >= 1001.2)
check("interval: planning_s = edge difference (2.0 s); gpu_w_mean = mean of the readings in it; energy_j_gpu = mean x 2.0; CPU None",
      close(iv["planning_s"], 2.0) and close(iv["gpu_w_mean"], (40.0 * n_lo + 80.0 * n_hi) / (n_lo + n_hi), 1e-3)
      and close(iv["energy_j_gpu"], iv["gpu_w_mean"] * 2.0, 1e-3) and iv["energy_j_cpu"] is None and iv["gpu_samples"] == n_lo + n_hi
      and iv["problems"] == [], iv)
sd = Sim(30.0)
md_ = meter(sd, gpu=FakeSmi(sd, dead=(0, 1e9)))
md_.open()
a2 = md_.edge()
sd.sleep(1.0)
iv2 = md_.interval(a2, md_.edge())
check("no nvidia-smi reading at all -> energy None, problem named (the row is then UNKNOWN, never guessed)", iv2["energy_j_gpu"] is None
      and iv2["problems"] == ["no nvidia-smi reading for this cell"])
bl = m.baseline(1.0)
check("baseline: the idle mean in W, reported", close(bl["gpu_w"], 80.0) and bl["secs"] >= 1.0)
mf = meter(sim, gpu=FakeSmi(sim, fail_open=True))
check("open without nvidia-smi -> not ok, the problem named", mf.open()["ok"] is False and "nvidia-smi" in mf.problems[0])


# ---------------------------------------------------------------------------------------------------- 3. the Plan hook
class FakeJ:
    def __init__(self, path):
        self.path = path

    def append(self, row):
        with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row) + "\n")


class FakePlan:
    """The real _call_cell contract: plan one cell (time passes), write ONE plan journal row, return it."""

    def __init__(self, root, sim, ns="rehearsal-v1", pass_id="", dry_run=False):
        self.ns, self.root, self.pass_id, self.dry_run, self.sim = ns, root, pass_id, dry_run, sim
        os.makedirs(C.ns_docs(ns, root), exist_ok=True)
        self.J = FakeJ(os.path.join(C.ns_docs(ns, root), f"plan.{pass_id}.jsonl" if pass_id else "plan.jsonl"))
        self.n = 0

    def _call_cell(self, c, rec):
        t0 = self.sim.t
        self.sim.sleep(c.get("dur", 1.0))
        self.n += 1
        row = {"schema": "ftm.plan/1", "ts": f"2026-10-14T20:00:{self.n:02d}", "key": c["key"], "arm": c["arm"], "pres": c["pres"],
               "r": 0, "block": None, "request_sha": "sha-" + c["key"], "status": "ok", "wall_s": round(self.sim.t - t0, 3), "calls": 1,
               "cached_calls": 0, "cost_usd": 0.0, "planner": c["planner"]}
        self.J.append(row)
        if c.get("raise"):
            raise RuntimeError("ceiling (fake)")
        return row


tmp = tempfile.mkdtemp(prefix="c5s_")
try:
    simh = Sim(50.0)
    EN.uninstall(FakePlan)
    infoh = EN.install(FakePlan, meter=meter(simh), require=True, baseline_s=1.0)
    check("install: wraps the class method, idle floor measured; idempotent", getattr(FakePlan._call_cell, "_c5_energy", False)
          and close(infoh["idle_floor"]["gpu_w"], 50.0) and EN.install(FakePlan, require=True) is infoh)
    pl = FakePlan(tmp, simh)
    cells = [{"key": "k1", "arm": "G+M", "planner": "G", "pres": "g-l1-a|en|canon|1", "dur": 1.5},
             {"key": "k2", "arm": "O0", "planner": "opus", "pres": "g-l1-a|en|canon|1", "dur": 0.7},
             {"key": "k3", "arm": "G0", "planner": "G", "pres": "g-l1-a|fa|canon|1", "dur": 2.0}]
    rows = [pl._call_cell(c, {}) for c in cells]
    check("an exception inside the cell is re-raised", raises(lambda: pl._call_cell({"key": "k4", "arm": "G+FS", "planner": "G", "pres": "x",
                                                                                      "dur": 0.4, "raise": True}, {})))
    er = [json.loads(x) for x in open(os.path.join(C.ns_docs("rehearsal-v1", tmp), "energy.jsonl"), encoding="utf-8")]
    cr = [e for e in er if e["kind"] == "cell"]
    check("energy journal: one 'open' row, then one 'cell' row per GEMMA cell only (k1, k3, k4)", er[0]["kind"] == "open"
          and [e["key"] for e in cr] == ["k1", "k3", "k4"] and cr[2]["raised"] == "RuntimeError")
    lines = open(pl.J.path, "rb").read().split(b"\n")
    check("cell row: 50 W x planning_s, planning_s = the cell's time, bound to its plan line by sha256",
          close(cr[0]["energy_j_gpu"], 50.0 * cr[0]["planning_s"], 1e-3) and close(cr[0]["planning_s"], rows[0]["wall_s"], 1e-3)
          and cr[0]["bound"]["plan_line_sha256"] == hashlib.sha256(lines[0]).hexdigest() and cr[0]["energy_j_cpu"] is None)
    pl3 = FakePlan(tmp, simh, ns="rehearsal-v2", dry_run=True)
    pl3._call_cell({"key": "kd", "arm": "G0", "planner": "G", "pres": "x", "dur": 0.2}, {})
    check("a dry-run plan writes no energy row", not os.path.exists(os.path.join(C.ns_docs("rehearsal-v2", tmp), "energy.jsonl")))
    EN.uninstall(FakePlan)
    check("uninstall restores the original method", not getattr(FakePlan._call_cell, "_c5_energy", False))
    check("install(require=True) without nvidia-smi raises MeterUnavailable and does not wrap",
          raises(lambda: EN.install(FakePlan, meter=meter(simh, gpu=FakeSmi(simh, fail_open=True)), require=True))
          and not getattr(FakePlan._call_cell, "_c5_energy", False))

    # ------------------------------------------------------------------------------------------------ 4. join + price
    prow = [json.loads(x) for x in open(pl.J.path, encoding="utf-8")]
    att, rep = A.attach_energy(prow, A.load_energy_rows(C.ns_docs("rehearsal-v1", tmp)), A.plan_line_shas(pl.J.path), strict=True)
    check("join: the 3 Gemma rows measured, the Opus row untouched, complete", rep["gemma_plan_rows"] == 3 and rep["measured"] == 3
          and att[1] is prow[1] and rep["complete"] is True and att[0]["energy_j_cpu"] is None, rep)
    pc = A.price_row(att[0], "gemma", A.A24_2_COST)
    check("price: energy USD = energy_j_gpu / 3.6e6 x 0.20; hardware = 399 / 6000 h x planning_s", pc["known"]
          and close(pc["energy"], att[0]["energy_j_gpu"] / 3.6e6 * 0.20) and close(pc["hardware"], 399.0 / 6000.0 * att[0]["planning_s"] / 3600.0))
    _, rep_n = A.attach_energy(prow, [], A.plan_line_shas(pl.J.path))
    check("no energy rows -> no energy figure -> UNKNOWN (no silent number)", rep_n["no_energy_row"] == 3 and rep_n["complete"] is False
          and A.price_row(prow[0], "gemma", A.A24_2_COST)["known"] is False)
    rr = A.measured_rate([dict(r, energy_state="measured", planning_s=10.0, energy_j_gpu=500.0, arm="G+M") for r in prow[:3]] * 7, A.A24_2_COST)
    check("measured rate = sum measured J / sum planning s (21 rows x 500 J / 10 s = 50 W)", rr["cells"] == 21 and close(rr["rate_w"], 50.0))
    pv2 = A.read_jsonl(os.path.join(C.paths()["root"], *A.UNMEASURED_TERMS_APPROVED["rows"]["journal"].split("/")))
    pids = A.approved_practice_ids([dict(r, cost_kind="gemma", cost_source="practice") for r in pv2], A.UNMEASURED_TERMS_APPROVED)
    pe = A.price_row(dict(pv2[0], cost_kind="gemma", cost_source="practice"), "gemma", A.A24_2_COST, prep=True, rate_w=50.0, practice_ids=pids)
    sec0 = A._plan_seconds(pv2[0], prep=True)
    check("a real practice-v2 row: ESTIMATED = 50 W x its own planning seconds, flagged, priced at 0.20 USD/kWh", len(pids) == 10
          and pe.get("estimated") is True and close(pe["joules"], 50.0 * sec0) and close(pe["energy"], 50.0 * sec0 / 3.6e6 * 0.20), (pe, sec0))
    po = A.price_row({"plan_secs": 5.0, "cost_kind": "gemma", "cost_source": "practice"}, "gemma", A.A24_2_COST, prep=True, rate_w=50.0,
                     practice_ids=pids)
    check("any other unmeasured Gemma row stays UNKNOWN (not one of the 10)", po["known"] is False and "estimated" not in po)
    fz = {"usd_per_kwh": 0.2, "hardware_price_usd": 399, "operating_hours": 6000, "hardware_time_factor": 3, "energy_method": dict(em),
          "hardware_price_basis": "amendment:A33"}
    check("freeze record: must carry the active rule's terms (missing -> not recorded; present -> recorded)",
          A.check_cost_freeze({"a24_2_cost": fz})["recorded"] is False
          and A.check_cost_freeze({"a24_2_cost": dict(fz, unmeasured_rule={"terms": A.UNMEASURED_TERMS_APPROVED})})["recorded"] is True)
    check("freeze record with another energy method id -> refused", raises(lambda: A.check_cost_freeze(
        {"a24_2_cost": dict(fz, energy_method=dict(em, id="c5-energy-3"), unmeasured_rule={"terms": A.UNMEASURED_TERMS_APPROVED})})))
    md = "\n".join(A.render_cost_md({"assumptions": A.A24_2_COST, "available": False, "reason": "x"}))
    check("markdown prints the plain disclosure", "Energy (C5, c5-simple-1)" in md and "simple, approximate" in md)

    # ------------------------------------------------------------------------------------------------ 5. launcher wiring
    import plan as PL
    import planfirst as PF
    calls, real_main = [], PL.main

    def fake_main(argv=None):
        calls.append(bool(getattr(PL.Plan._call_cell, "_c5_energy", False)))
        return 0
    PL.main = fake_main
    root6 = os.path.join(tmp, "root6")
    try:
        simr = Sim()
        rc = MR.dispatch("plan", ["--namespace", "test-v1"], root6, _meter=meter(simr, gpu=FakeSmi(simr, fail_open=True)))
        check("matrix_run.dispatch: test-v1 plan without nvidia-smi -> REFUSED, plan.main never called", rc == MR.EXIT["refused"] and not calls)
        rc2 = MR.dispatch("plan", ["--namespace", "test-v1"], root6, _meter=meter(simr))
        check("matrix_run.dispatch: wrapped during plan.main only, uninstalled after", rc2 == 0 and calls == [True]
              and not getattr(PL.Plan._call_cell, "_c5_energy", False))
        d6 = C.ns_docs("test-v1", root6)
        os.makedirs(d6, exist_ok=True)
        with open(os.path.join(d6, "plan.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(prow[0]) + "\n")
        check("after_planning: a Gemma row without an energy figure -> exit 9; a non-zero code passes through",
              EN.after_planning(0, "test-v1", root6, out=lambda *_: None) == EN.ENERGY_INCOMPLETE
              and EN.after_planning(4, "test-v1", root6, out=lambda *_: None) == 4)
        real_run, real_meter, seen_pf = PF.run, EN.Meter, []
        PF.run = lambda ns, **kw: seen_pf.append((ns, bool(getattr(PL.Plan._call_cell, "_c5_energy", False)))) or 0
        try:
            EN.Meter = lambda: meter(simr)
            r_off = PF.main(["--namespace", "test-v1", "--energy", "off", "--root", root6])
            r_dev = PF.main(["--namespace", "rehearsal-v3", "--root", root6])
            r_t = PF.main(["--namespace", "test-v1", "--root", root6])
        finally:
            PF.run, EN.Meter = real_run, real_meter
        check("planfirst: --energy off refused in test-v1; dev default unmeasured; test-v1 measured (exit 9: the fake journal row has no energy)",
              r_off == PF.REFUSED and r_dev == 0 and r_t == EN.ENERGY_INCOMPLETE and seen_pf == [("rehearsal-v3", False), ("test-v1", True)],
              (r_off, r_dev, r_t, seen_pf))
    finally:
        EN.uninstall(PL.Plan)
        PL.main = real_main
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{N['pass']}/{N['pass'] + N['fail']} passed")
sys.exit(0 if N["fail"] == 0 else 1)
