"""C5 energy, SIMPLE method (stage C5-SIMPLE, 2026-10-14, Arash's decision "simplify the energy method"): GPU energy of every Gemma planning cell.

WHY THIS FILE EXISTS. A24.2 prices local Gemma planning as energy x 0.20 USD/kWh (+ the A33 hardware share). plan.py and practice_k.py are
pinned by the acceptance registration, so they are NOT edited: this module wraps ``plan.Plan._call_cell`` at run time (``install``), only
for Gemma cells (planner "G"), and writes its own journal next to the plan journal. analyze.py joins the two (``analyze.attach_energy``) by
the sha256 of the plan journal line.

METHOD (id ``c5-simple-1``; the same text is ANALYSIS_PLAN.a24_2_cost.energy_method). A simple, approximate method:
  * ``nvidia-smi --query-gpu=power.draw.instant -i 0 -lms 100`` runs in the background for the whole plan phase and prints the GPU board power
    (W) about every 100 ms. Each line is time-stamped on arrival.
  * For each Gemma cell: planning_s = the time from just before ``Plan._call_cell`` starts to just after it returns; gpu_w_mean = the plain
    mean of the readings that arrived in that time; energy_j_gpu = gpu_w_mean x planning_s. A cell with no reading in its time uses the
    readings of the 1 s before / after it; a cell with none at all has no energy (energy_j_gpu None).
  * GPU only. The CPU, memory, display, fans and power-supply losses are NOT measured (energy_j_cpu is None). The reading is the whole GPU
    board, every process on it; the idle floor is included, never subtracted. The idle floor is measured when the meter opens and reported.

CLI:
    python sim/final_test/matrix/energy.py probe          # start nvidia-smi, print the GPU name + a 3 s idle mean (no file written)

Python 3.10 syntax only.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time

METHOD_ID = "c5-simple-1"
SCHEMA = "ftm.energy/1"
SAMPLE_MS = 100                 # nvidia-smi -lms
NEAR_S = 1.0                    # a cell with no reading in its time uses the readings this close to it
NVSMI_CMD = ["nvidia-smi", "--query-gpu=power.draw.instant", "--format=csv,noheader,nounits", "-i", "0", "-lms", str(SAMPLE_MS)]


class MeterUnavailable(RuntimeError):
    pass


def mean_power_j(samples, t0: float, t1: float, near: float = NEAR_S):
    """(t, W) readings -> (gpu_w_mean, joules, readings used). Mean of the readings in [t0, t1] x (t1 - t0); with none there, the readings
    within `near` s of the interval; with none at all (or t1 <= t0) -> (None, None, 0)."""
    if t1 <= t0:
        return None, None, 0
    xs = [w for t, w in samples if t0 <= t <= t1]
    if not xs:
        xs = [w for t, w in samples if t0 - near <= t <= t1 + near]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    return m, m * (t1 - t0), len(xs)


class NvidiaSmiPower:
    """nvidia-smi in the background; readings (perf time on arrival, W)."""

    def __init__(self, cmd=None, perf=time.perf_counter):
        self.cmd, self.perf = list(cmd or NVSMI_CMD), perf
        self.proc, self.th, self.buf, self.lock, self.info = None, None, [], threading.Lock(), {}

    def open(self, wait_s: float = 5.0) -> dict:
        try:
            name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"], capture_output=True, text=True,
                                  timeout=20).stdout.strip()
            self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except (OSError, subprocess.SubprocessError) as e:
            raise MeterUnavailable(f"nvidia-smi: {e}") from None

        def loop():
            for ln in self.proc.stdout:
                try:
                    w = float(ln.strip())
                except ValueError:
                    continue
                with self.lock:
                    self.buf.append((self.perf(), w))
        self.th = threading.Thread(target=loop, name="c5-nvidia-smi", daemon=True)
        self.th.start()
        t = time.time()
        while not self.buf and time.time() - t < wait_s:
            time.sleep(0.05)
        if not self.buf:
            self.close()
            raise MeterUnavailable("nvidia-smi gave no power.draw.instant reading")
        self.info = {"name": name, "source": " ".join(self.cmd)}
        return self.info

    def close(self):
        if self.proc is not None:
            with contextlib.suppress(Exception):
                self.proc.terminate()
                self.proc.wait(timeout=5)
        self.proc = None

    def window(self, p0: float, p1: float, pad: float = NEAR_S) -> list:
        with self.lock:
            return [s for s in self.buf if p0 - pad <= s[0] <= p1 + pad]


class Meter:
    """``edge()`` = one time stamp; ``interval(a, b)`` = planning_s, gpu_w_mean, energy_j_gpu (GPU only; CPU not measured)."""

    def __init__(self, gpu=None, *, clock=time.time, perf=time.perf_counter, sleep=time.sleep):
        self.gpu = gpu if gpu is not None else NvidiaSmiPower(perf=perf)
        self.clock, self.perf, self.sleep = clock, perf, sleep
        self.ok, self.info, self.problems = False, {}, []

    def open(self) -> dict:
        self.problems, devs = [], {}
        try:
            devs["gpu"] = self.gpu.open()
        except Exception as e:                                              # noqa: BLE001
            self.problems.append(f"gpu: {e}")
        self.ok = not self.problems
        self.info = {"method": METHOD_ID, "ok": self.ok, "problems": list(self.problems), "devices": devs,
                     "host": platform.node(), "platform": platform.platform(), "python": sys.version.split()[0]}
        return self.info

    def close(self):
        with contextlib.suppress(Exception):
            self.gpu.close()
        self.ok = False

    def edge(self) -> dict:
        return {"p": self.perf(), "t": self.clock()}

    read = edge

    def interval(self, a: dict, b: dict) -> dict:
        secs = b["p"] - a["p"]
        self.sleep(SAMPLE_MS / 1000.0)                                      # let the reading just after the end arrive
        w, j, n = mean_power_j(self.gpu.window(a["p"], b["p"]), a["p"], b["p"])
        out = {"planning_s": round(secs, 6), "gpu_w_mean": None if w is None else round(w, 3), "gpu_samples": n,
               "energy_j_gpu": None if j is None else round(j, 4), "energy_j_cpu": None, "problems": []}
        if j is None:
            out["problems"].append("no nvidia-smi reading for this cell")
        return out

    def baseline(self, secs: float = 3.0) -> dict:
        """The idle mean (W) over `secs` -- reported, never subtracted."""
        a = self.edge()
        self.sleep(secs)
        iv = self.interval(a, self.edge())
        return {"secs": iv["planning_s"], "gpu_w": iv["gpu_w_mean"], "problems": iv["problems"]}


# ================================================================================================ the Plan hook
_STATE = {"installed": False, "meter": None, "info": None, "lock": threading.Lock(), "opened": set()}


def _iso(t: float) -> str:
    return _dt.datetime.fromtimestamp(t).isoformat(timespec="milliseconds")


def energy_path(plan_obj) -> str:
    import common as C
    d = C.ns_docs(plan_obj.ns, getattr(plan_obj, "root", None))
    pid = getattr(plan_obj, "pass_id", "") or ""
    return os.path.join(d, f"energy.{pid}.jsonl" if pid else "energy.jsonl")


def _append(path: str, row: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _bound_line(jpath: str, size0: int, key: str):
    """The plan journal line(s) appended since size0 for `key` -> the last one's identity (row fields + line sha256), or None."""
    try:
        with open(jpath, "rb") as fh:
            fh.seek(size0)
            blob = fh.read()
    except OSError:
        return None
    hit = None
    for ln in blob.split(b"\n"):
        s = ln.rstrip(b"\r")
        if not s.strip():
            continue
        try:
            r = json.loads(s.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if r.get("key") == key:
            hit = {"plan_line_sha256": hashlib.sha256(s).hexdigest(), "ts": r.get("ts"), "status": r.get("status"),
                   "request_sha": r.get("request_sha"), "wall_s": r.get("wall_s"), "calls": r.get("calls"), "cached_calls": r.get("cached_calls")}
    return hit


def install(plan_cls, *, meter: Meter | None = None, require: bool = False, baseline_s: float = 3.0, out=print) -> dict:
    """Wrap plan_cls._call_cell (class attribute, at run time; the file is not touched). Idempotent. require=True raises MeterUnavailable
    when nvidia-smi cannot be read (the caller refuses to plan)."""
    st = _STATE
    if getattr(plan_cls._call_cell, "_c5_energy", False):
        if require and not (st.get("meter") is not None and st["meter"].ok):
            raise MeterUnavailable("the installed meter is not available")
        return st["info"]
    m = meter or Meter()
    try:
        info = m.open()
        if not info["ok"]:
            if require:
                raise MeterUnavailable("; ".join(info["problems"]))
            out(f"[energy] meter unavailable ({'; '.join(info['problems'])}): Gemma cells get NO energy row -> their A24.2 cost is UNKNOWN")
        else:
            info["idle_floor"] = m.baseline(baseline_s) if baseline_s > 0 else None
    except BaseException:
        m.close()
        raise
    orig = plan_cls._call_cell

    def _call_cell(self, c, rec):
        if c.get("planner") != "G" or getattr(self, "dry_run", False) or not m.ok:
            return orig(self, c, rec)
        path = energy_path(self)
        with st["lock"]:
            if path not in st["opened"]:
                st["opened"].add(path)
                _append(path, {"schema": SCHEMA, "kind": "open", "method": METHOD_ID, "ts": _iso(time.time()), "namespace": self.ns,
                               "pass_id": getattr(self, "pass_id", "") or "", "pid": os.getpid(), "meter": info})
        jpath = self.J.path
        try:
            size0 = os.path.getsize(jpath)
        except OSError:
            size0 = 0
        a = m.edge()
        raised = None
        try:
            return orig(self, c, rec)
        except BaseException as e:                                          # noqa: BLE001  (a ceiling / abort still spent energy; re-raised)
            raised = type(e).__name__
            raise
        finally:
            b = m.edge()
            row = {"schema": SCHEMA, "kind": "cell", "method": METHOD_ID, "ts": _iso(b["t"]), "namespace": self.ns,
                   "pass_id": getattr(self, "pass_id", "") or "", "key": c.get("key"), "arm": c.get("arm"), "planner": c.get("planner"),
                   "pres": c.get("pres"), "r": c.get("r"), "block": c.get("block"), "t0": round(a["t"], 6), "t1": round(b["t"], 6),
                   "raised": raised, "bound": _bound_line(jpath, size0, c.get("key")), **m.interval(a, b)}
            try:
                _append(path, row)
            except OSError as e:
                out(f"[energy] could not write the energy row of {c.get('key')}: {e}")
    _call_cell._c5_energy = True
    _call_cell._c5_orig = orig
    plan_cls._call_cell = _call_cell
    st.update(installed=True, meter=m, info=info, plan_cls=plan_cls)
    return info


def uninstall(plan_cls):
    f = plan_cls._call_cell
    if getattr(f, "_c5_energy", False):
        plan_cls._call_cell = f._c5_orig
    if _STATE.get("meter") is not None:
        _STATE["meter"].close()
    _STATE.update(installed=False, meter=None, info=None, opened=set())


@contextlib.contextmanager
def measuring(plan_cls, *, require: bool = True, meter: Meter | None = None, baseline_s: float = 3.0, out=print):
    """install() for the lifetime of one launch (matrix_run / planfirst), always uninstalled and closed afterwards."""
    if getattr(plan_cls._call_cell, "_c5_energy", False):
        uninstall(plan_cls)
    try:
        info = install(plan_cls, meter=meter, require=require, baseline_s=baseline_s, out=out)
        yield info
    finally:
        uninstall(plan_cls)
        if meter is not None:
            meter.close()


def coverage(ns: str, root=None, plan_journal: str | None = None) -> dict:
    """How many Gemma plan rows of a namespace's plan journal have an energy figure (the analyze join). Read-only."""
    import common as C
    import analyze as A
    d = C.ns_docs(ns, root)
    pj = plan_journal or os.path.join(d, "plan.jsonl")
    if not os.path.exists(pj):
        return {"plan_journal": pj, "complete": None, "note": "no plan journal"}
    rows = A.read_jsonl(pj)
    _, rep_ = A.attach_energy(rows, A.load_energy_rows(d), A.plan_line_shas(pj), strict=True)
    return dict(rep_, plan_journal=pj)


ENERGY_INCOMPLETE = 9            # launcher exit code: planning finished but a Gemma row has no energy figure


def plan_journal_of(ns: str, root=None, pass_id: str = "") -> str:
    import common as C
    return os.path.join(C.ns_docs(ns, root), f"plan.{pass_id}.jsonl") if pass_id else C.journal_path(ns, "plan", root)


def after_planning(code: int, ns: str, root=None, pass_id: str = "", out=print) -> int:
    """Launcher epilogue: when planning ended with 0 and a Gemma row has no energy figure, say so loudly and exit ENERGY_INCOMPLETE."""
    if code != 0:
        return code
    cov = coverage(ns, root, plan_journal_of(ns, root, pass_id))
    if cov.get("complete") is False:
        out(f"ENERGY INCOMPLETE (C5): {cov.get('gemma_plan_rows')} Gemma plan rows, {cov.get('measured')} with an energy figure: the rest "
            f"have an UNKNOWN A24.2 cost. Stop and ask Arash (exit {ENERGY_INCOMPLETE}).")
        return ENERGY_INCOMPLETE
    return code


def require_flag(ns: str, energy: str, dry_run: bool = False):
    """test-v1: the meter is mandatory (energy 'off' is refused). -> (install?, refusal message or None)."""
    if dry_run:
        return False, None
    if ns == "test-v1":
        if energy == "off":
            return False, "--energy off in test-v1: A24.2 needs the Gemma energy of every plan cell (C5)"
        return True, None
    return energy == "on", None


# ================================================================================================ CLI
def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="C5 energy meter, simple method (A24.2 / A33)")
    ap.add_argument("what", choices=("probe",))
    ap.add_argument("--baseline", type=float, default=3.0)
    a = ap.parse_args(argv)
    m = Meter()
    info = m.open()
    if info["ok"]:
        info["idle_floor"] = m.baseline(a.baseline)
    m.close()
    print(json.dumps(info, indent=1))
    return 0 if info["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
