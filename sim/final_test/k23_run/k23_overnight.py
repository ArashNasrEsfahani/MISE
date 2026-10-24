"""K23 OVERNIGHT runner (Windows, repo root): runs Arash's Kitchen 2 / 3 test sequence unattended, STRICT -- it stops at the FIRST
failure and leaves everything as it is for the morning. It never forces anything, never skips or waives a gate (and never writes a
waiver), never re-runs a G-iso, never edits preregistration.md, never kills or steals from another job, and never runs git checkout /
merge / reset (only `git add` + `git commit -- <paths>` of evidence files). Infrastructure only: every rule, gate and decision lives in the
tools it calls. Chains run through the STRICT wrapper copies (k23_strict_wrappers.py: chain_k23_strict.sh / launch_k23_strict.py /
run_k23_strict.sh; practice through practice/launch_strict.py / run_strict.sh / inner_strict.sh): bounded sim-lock waits, never stolen,
only the job's own processes cleaned up, no restart loops, a failed execute session never masked, no G-iso re-run.

Before anything: git HEAD is master, no merge / rebase / cherry-pick in progress, the strict wrappers equal their rendered originals,
preregistration.md sha256 = 247ef23a... and treatment.pin_problems() == [] (the integrity pair is re-checked before every step and after
every commit).
Steps (each is skipped when its own durable done-check holds -- its evidence is still committed -- so a morning re-run resumes):
  W   wait until rehearsal-k3-v1 is complete (or complete up to the context fit with a waived G-iso, see R3c) AND the Kitchen 2 COCO run ended
      with `END rc=0` (sim/kitchen_v3/k23/FloorPlan15/train/train_yolo26s_coco.log). A G-iso result that is not a pass (inconclusive, any
      blocked presentation, fail) without Arash's waiver record stops the runner for Arash (A20 / A20.1 / A21.A.6). A COCO run stopped by a
      pause (END rc 143 / 137) or a WSL loss (START + a stale log) after the K3 work is over is resumed with train_k23.sh (at most 3 times;
      only with the WSL process list readable, no training process, the GPU idle; it counts only when a NEW START marker and a training
      process appear; --no-coco-resume stops instead). A HEAT PAUSE / TELEMETRY LOSS (it takes precedence over any later END) or any other
      END rc stops the runner. K3 rehearsal not complete with nothing running it -> stop.
  C   WSL: k23_freeze.py --scene FloorPlan15 --add-coco, then k23_coco_meta.py --write, then k23_coco_meta.py (check rc 0);
      k23_entry.coco_problems() == [] afterwards (prereg A39.2).
  O   GPU / Ollama: no training process in WSL; GPU < 2,000 MiB before Ollama is started; Ollama serves gemma4:e2b-it-q4_K_M.
  R3c rehearsal-k3-v1 resumed at its context-fit stage ONLY (k23_run --phase context, as the chain's context stage) when capture, plan,
      execute and the sealed score are complete, the G-iso result is not a pass but Arash's waiver record covers exactly that result, and
      no chain of it is alive; earlier stages are never re-run.
  P2  python sim/final_test/k23_run/practice/launch_strict.py practice-k2-v1 (background, supervised: it must start within 15 min and end
      within lock wait + 4 h + 1 h; one try, no restart); then k23_practice.py --status (ok and 10 attempts).
  R2  ENERGY-free strict chain rehearsal-k2-v1 (background, supervised); the same G-iso rule and context-fit resume as for Kitchen 3.
  S2  k23_size_decision.py --checkpoint after-k1 (dry run), --write, --check  (spend = the ledger: no --provider-usd; Arash 16:15).
  F2  k23_freeze_agent.py --scene FloorPlan15 --dry-run, --write; an Astra review (astra_call.py) whose prompt carries the freeze's
      self-sha and demands a fixed four-line acceptance (RULING: ACCEPTED / KITCHEN / FREEZE-SHA16 / CONDITIONS: none, nothing else);
      only then --presign --astra-review <file>; --check. A freeze found already signed is accepted only with Arash's own check or a
      presignature whose review file still has its recorded sha and passes the same four-line test.
  T2  ENERGY=on strict chain test-k2-v1 (background, supervised). While it runs, once test-k2-v1 has its full plan_first_complete event
  S3    (Kitchen 2 paid planning complete) and the chain has not failed: k23_size_decision.py --checkpoint after-fp15 (dry run), --write,
  F3    --check; then the FloorPlan17 agent freeze, the same as F2. The Kitchen 2 chain is polled before every K3 command, commit, WSL call
        and Ollama start; a Kitchen 2 failure stops the K3 work at once. A K3-side failure (or an unexpected error) never kills the
        Kitchen 2 chain: the runner waits for it (with status and lock checks), commits its evidence, then stops.
  T3  only after the Kitchen 2 chain ended rc 0 with durable evidence (CHAIN DONE after the last CHAIN START, no masked execute failure in
      any chain run, every phase complete and unchanged, the score seal matching score.jsonl, no chain alive): ENERGY=on strict chain
      test-k3-v1 (background, supervised).
Every chain start is preceded by: the K23 capture artefact gate (Astra K23-CAPTURE-FREEZE-r1 ACCEPTED for exactly the reviewed
k23_capture_freeze.py, and its --check rc 0), Ollama up, no training process, the sim lock free (waited <= 30 min, never stolen).
A supervised chain whose WSL job sits "queued" for > 30 min stops the runner (the strict job itself gives up at 30 min and does not run).
WSL connection failures are retried as k23_k3_practice_now.ps1 does: WSL commands up to 10 x 60 s; practice / chain once after 90 s.
G-iso waiver: the ONLY supported one is Arash's K23-GISO-WAIVER-FloorPlan17 (k23_giso_waiver.py; A20.1, narrowed claim), checked with
exactly the agent freeze's conditions (FloorPlan17, 'inconclusive', diff == [], the archived result); any other non-pass G-iso (a 'fail',
or any Kitchen 2 one) stops for Arash. The runner never writes a waiver.
Rehearsal completion needs durable success evidence: "CHAIN DONE" after the last "CHAIN START" (the chain's own exit), or -- only after a
waived G-iso -- the runner's own record of a context stage that exited 0 (overnight/context_resume_<ns>.json, bound to the context_fit,
g_iso and waiver shas). A context fit present without either (e.g. run by hand) stops for the morning; it is never re-run.
Practice launch authorization: each strict practice launch gets a one-time token file; run_strict.sh claims it (atomic rename) after it
holds the sim lock and before anything runs, and refuses (exit 74) when it is gone. On a start stall the runner revokes it (atomic
rename), so a late launcher can never start practice after the runner stopped.

Logs: sim/final_test/k23_run/overnight/{overnight.log, steps.jsonl, status.json, STOPPED.json | DONE.json, <ns>.console.txt}.

    python sim/final_test/k23_run/k23_overnight.py --dry-run     (checks every precondition, prints the plan; runs nothing)
    python sim/final_test/k23_run/k23_overnight.py --run         (normally via k23_overnight.ps1, launched hidden)
Python 3.10 (Windows).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MATRIX = os.path.join(ROOT, "sim", "final_test", "matrix")
if MATRIX not in sys.path:                                # common / k23_giso_waiver (this checkout's K23 modules; read-only use)
    sys.path.insert(0, MATRIX)

PREREG = "docs/final-test/preregistration.md"
PREREG_SHA = "247ef23ac91fec3ca6483d48883b3ce4583b09197e6a5bb0135fafa1d6803f69"
GEMMA = "gemma4:e2b-it-q4_K_M"
OLLAMA_URL = "http://127.0.0.1:11434"
DISTRO = "Ubuntu-22.04"
WSL_PY = "/home/arash/thor-env/bin/python"
SIM_LOCK = "/home/arash/sim.lock"
GIT_BASH = r"C:\Program Files\Git\bin\bash.exe"
OLLAMA_EXE = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")
COCO_LOG = "sim/kitchen_v3/k23/FloorPlan15/train/train_yolo26s_coco.log"
COCO_WEIGHTS = "sim/weights/v3/FloorPlan15.yolo26s_coco.pt"
COCO_TRAIN = ["bash", "sim/kitchen_v3/k23/train_k23.sh", "FloorPlan15", "yolo26s_coco"]
TRAIN_PROCS = r"k23_train\.py|train_kitchen\.py|k23_amb_train\.py|k23_state_train\.py|train_amb\.py|train_state\.py"
WSL_SIGNATURE = re.compile(r"Wsl/Service|0x8007274c")
WSL_RCS = (-1, 255, 4294967295)
OUT = "sim/final_test/k23_run/overnight"
LOGS = "sim/final_test/k23_run/logs"
ASTRA_CALL = "docs/final-test/k23/merge-drafts/sc2/astra_call.py"
ASTRA_DIR = "docs/final-test/k23/_astra"
ANSWER_MARK = "=== ANSWER (verbatim) ==="
#: the K23 capture artefact fix and its review (Astra K23-CAPTURE-FREEZE-r1, ACCEPTED for 92a701b): the module must be exactly that version
CAPTURE_FREEZE = "sim/final_test/matrix/k23_capture_freeze.py"
CAPTURE_FREEZE_LF_SHA = "7894ea4ee02fcfd2b7d99e778c8c1ea4d18c9e0c3a41d73b12a2bbf20206ed35"
CAPTURE_REVIEW = "docs/final-test/k23/_astra/K23-CAPTURE-FREEZE-r1.txt"
CAPTURE_CASE = "K23-CAPTURE-FREEZE-r1"
PRACTICE_STRICT = "sim/final_test/k23_run/practice/launch_strict.py"
CHAIN = "sim/final_test/k23_run/chain_k23_strict.sh"
GISO_WAIVER = "docs/final-test/k23/decisions/K23-GISO-WAIVER-{scene}.json"
PRACTICE_START_S = 900                                    # the strict practice must log its queued line within 15 min
PRACTICE_MAX_S = 1800 + 14400 + 3600                      # lock wait + inner timeout + margin
AUTH_DIR = "sim/final_test/k23_run/overnight/auth"
KIT = {"FloorPlan15": {"k": 2, "practice": "practice-k2-v1", "rehearsal": "rehearsal-k2-v1", "test": "test-k2-v1"},
       "FloorPlan17": {"k": 3, "practice": "practice-k3-v1", "rehearsal": "rehearsal-k3-v1", "test": "test-k3-v1"}}
K3_NS = ("practice-k3-v1", "rehearsal-k3-v1")
SCENE_OF = {ns: sc for sc, k in KIT.items() for ns in (k["practice"], k["rehearsal"], k["test"])}
TOOLS = (CAPTURE_FREEZE, CAPTURE_REVIEW, PRACTICE_STRICT, "sim/final_test/k23_run/practice/run_strict.sh", CHAIN,
         "sim/final_test/k23_run/practice/inner_strict.sh", "sim/final_test/k23_run/launch_k23_strict.py",
         "sim/final_test/k23_run/run_k23_strict.sh", "sim/final_test/k23_run/k23_strict_wrappers.py",
         "sim/kitchen_v3/k23/k23_freeze.py", "sim/kitchen_v3/k23/k23_coco_meta.py", "sim/kitchen_v3/k23/train_k23.sh",
         "sim/final_test/k23_run/chain_k23.sh", "sim/final_test/matrix/k23_practice.py", "sim/final_test/matrix/k23_run.py",
         "sim/final_test/matrix/k23_size_decision.py", "sim/final_test/matrix/k23_freeze_agent.py", ASTRA_CALL, PREREG)
MAX_COCO_RESUMES = 3
POLL_S = 300
LOCK_WAIT_S = 1800


class Stop(Exception):
    def __init__(self, step, why, rc=1):
        super().__init__(f"{step}: {why}")
        self.step, self.why, self.rc = step, why, rc


class K2Ended(Exception):
    """Raised inside the K3 work when the Kitchen 2 chain has exited non-zero: no further K3 command runs."""


# ================================================================================================ pure helpers (selftested)
MARK = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (START|END rc=(-?\d+)|HEAT PAUSE|TELEMETRY LOSS)")


def parse_coco_log(text: str) -> dict:
    """The train_k23.sh markers of a training log: the last START / END, its rc, and a safety stop (heat / telemetry) after the last START."""
    marks = [(m.group(1), m.group(2).split(" ")[0], int(m.group(3)) if m.group(3) is not None else None) for m in MARK.finditer(text)]
    last = next(((ts, k, rc) for ts, k, rc in reversed(marks) if k in ("START", "END")), None)
    i_start = max((i for i, m in enumerate(marks) if m[1] == "START"), default=-1)
    safety = [m[1] for m in marks[i_start + 1:] if m[1] in ("HEAT", "TELEMETRY")]
    out = {"n_start": sum(1 for m in marks if m[1] == "START"), "last": last[1] if last else None, "last_ts": last[0] if last else None,
           "rc": last[2] if last else None, "safety": safety}
    out["ended_ok"] = out["last"] == "END" and out["rc"] == 0 and not safety
    return out


def coco_verdict(lg: dict, age_s: float | None, stale_s: float = 1800) -> str:
    """safety (heat / telemetry: left for the morning; precedes everything) | ended_ok | running | resumable (pause / WSL loss) |
    failed (another END rc) | none."""
    if lg.get("last") is None:
        return "none"
    if lg["safety"]:
        return "safety"
    if lg["ended_ok"]:
        return "ended_ok"
    if lg["last"] == "START":
        return "running" if age_s is not None and age_s < stale_s else "resumable"   # START without END + a stale log = WSL loss
    return "resumable" if lg["rc"] in (143, 137) else "failed"


def _tokens(cmd: str) -> list:
    import shlex
    try:
        return [t.strip('"').strip("'") for t in shlex.split(cmd, posix=False)]
    except ValueError:
        return []


def runs_ns(cmd: str, namespaces) -> bool:
    """True when the process's PROGRAM runs a K23 chain / launcher / WSL job for one of `namespaces`: bash .../chain_k23[_strict].sh <ns>;
    python .../practice/launch[_strict].py <ns>, .../launch_k23[_strict].py ... <ns>, .../k23_run.py --phase <not status> --namespace <ns>;
    wsl ... run_k23[_strict].sh / practice/run[_strict].sh ... <ns>. A shell whose -c text (or a prompt) only mentions these names does not count."""
    t = _tokens(cmd)
    if not t:
        return False
    exe = os.path.basename(t[0].replace("\\", "/")).lower()
    args = [a.replace("\\", "/") for a in t[1:]]
    if exe in ("bash.exe", "bash", "sh.exe", "sh"):
        return bool(args) and bool(re.search(r"chain_k23(_strict)?[.]sh$", args[0])) and args[1:2] != [] and args[1] in namespaces
    if exe.startswith("python"):
        a = [x for x in args if not x.startswith("-")]
        if a and re.search(r"practice/launch(_strict)?\.py$", a[0]):
            return a[1:2] != [] and a[1] in namespaces
        return bool(a) and bool(re.search(r"(launch_k23(_strict)?|k23_run)[.]py$", a[0])) and any(n in a for n in namespaces) \
            and "status" not in a
    if exe in ("wsl.exe", "wsl"):
        if "--" not in args:
            return False
        rest = args[args.index("--") + 1:]
        if rest[:1] == ["bash"]:
            rest = rest[1:]
        return bool(rest) and bool(re.search(r"(run_k23(_strict)?[.]sh|practice/run(_strict)?[.]sh)$", rest[0])) and any(n in rest for n in namespaces)
    return False


def busy_match(cmd: str) -> bool:
    """The K3 work still running: powershell -File .../k23_k3_*.ps1, or a chain / launcher / WSL job of practice-k3-v1 / rehearsal-k3-v1."""
    t = _tokens(cmd)
    if not t:
        return False
    exe = os.path.basename(t[0].replace("\\", "/")).lower()
    args = [a.replace("\\", "/") for a in t[1:]]
    if exe in ("powershell.exe", "powershell", "pwsh.exe", "pwsh"):
        low = [a.lower() for a in args]
        if "-file" in low:
            i = low.index("-file") + 1
            return i < len(args) and os.path.basename(args[i]).startswith("k23_k3_") and args[i].endswith(".ps1")
        return False
    return runs_ns(cmd, K3_NS)


def review_answer(full: str) -> str:
    """The answer of an astra_call.py review file (the same rule as k23_ns.a37_review_answer: exactly one marker line)."""
    lines = full.splitlines()
    at = [i for i, ln in enumerate(lines) if ln.strip() == ANSWER_MARK]
    if full.count(ANSWER_MARK) != 1 or len(at) != 1:
        return ""
    return "\n".join(lines[at[0] + 1:])


def acceptance_lines(scene: str, sha: str) -> list:
    return ["RULING: ACCEPTED", f"KITCHEN: {scene}", f"FREEZE-SHA16: {sha[:16]}", "CONDITIONS: none"]


def astra_verdict(full: str, scene: str, sha: str) -> tuple:
    """(ok, why): ok ONLY when the whole answer is exactly the four acceptance lines (blank lines and surrounding spaces ignored): no text
    before or after, so no condition can hide anywhere (Astra K23-OVERNIGHT-r1 item 3)."""
    ans = review_answer(full)
    if not ans.strip():
        return False, "no single Astra answer in the review file"
    got = [ln.strip() for ln in ans.splitlines() if ln.strip()]
    want = acceptance_lines(scene, sha)
    if got == want:
        return True, "accepted (four-line unconditional form)"
    first = got[0] if got else ""
    return False, (f"not the four-line unconditional acceptance (first line {first[:60]!r}, {len(got)} line(s))")


def capture_review_problems(root: str) -> list:
    """The K23 capture artefact gate: the module is byte-for-byte (LF-normalised) the reviewed version and its review is an ACCEPTED answer
    of case K23-CAPTURE-FREEZE-r1 with no CHANGES REQUIRED (Astra K23-OVERNIGHT-r1 item 6)."""
    why = []
    try:
        b = open(os.path.join(root, *CAPTURE_FREEZE.split("/")), "rb").read().replace(b"\r\n", b"\n")
        if hashlib.sha256(b).hexdigest() != CAPTURE_FREEZE_LF_SHA:
            why.append(f"{CAPTURE_FREEZE} is not the version Astra reviewed ({CAPTURE_FREEZE_LF_SHA[:12]})")
    except OSError:
        why.append(f"{CAPTURE_FREEZE} missing")
    try:
        full = open(os.path.join(root, *CAPTURE_REVIEW.split("/")), encoding="utf-8").read()
    except OSError:
        return why + [f"{CAPTURE_REVIEW} missing"]
    ans = review_answer(full)
    first = next((ln.strip() for ln in ans.splitlines() if ln.strip()), "")
    if f"case {CAPTURE_CASE}" not in full.split("\n", 2)[0] or first != "RULING: ACCEPTED" or "CHANGES REQUIRED" in ans.upper():
        why.append(f"{CAPTURE_REVIEW} is not an ACCEPTED answer of case {CAPTURE_CASE}")
    return why


def astra_case(scene: str, sha: str) -> str:
    return f"K23-FREEZE-{scene}-{sha[:16]}"


def astra_prompt(scene: str, sha: str, root: str) -> str:
    k = KIT[scene]
    extra = (" incl. its COCO-start sensitivity line (A38 item 4) and the k23_coco_meta successor metadata" if scene == "FloorPlan15" else "")
    fp17 = ("\n 6. the after-fp15 checkpoint of docs/final-test/k23/a29_size_decision.json was recorded from aggregate cost counters only, "
            "after test-k2-v1's full plan_first_complete event and before any test-k3-v1 plan row." if scene == "FloorPlan17" else "")
    acc = "\n".join(acceptance_lines(scene, sha))
    return (f"Case {astra_case(scene, sha)} (read-only review). Repo {root}, branch master. The K23 overnight runner asks whether Arash's "
            f"standing pre-signature (approve-all-6 item 1) may be recorded for the Kitchen {k['k']} agent freeze.\n"
            f"The freeze was written by `python sim/final_test/matrix/k23_freeze_agent.py --scene {scene} --write` after its dry run passed:\n"
            f"  record   docs/final-test/k23/freeze_{scene}.json   self-sha {sha}\n"
            f"  summary  docs/final-test/k23/FREEZE-SUMMARY-{scene}.md\n"
            "Please verify by reading files only (no simulator, Ollama, provider or write; do not open Kitchen 2/3 images or the item rows "
            "of docs/final-test/k23/applicability/**/runs.jsonl):\n"
            " 1. every gate of the record passes with concrete evidence (no placeholder; a missing field is never a pass), for every gate "
            "of GATE_ORDER in k23_freeze_agent.py;\n"
            f" 2. the bound shas match the files on disk now: preregistration.md ({PREREG_SHA[:8]}), the code map, the practice snapshot "
            f"of {k['practice']}, the treatment identity, the A29 roster through docs/final-test/k23/a29_size_decision.json, and the "
            f"perception freeze record{extra};\n"
            " 3. the decision records K23-FA-EXAMPLES and K23-PRACTICE-LEAK are bound, and an own-fragment leakage pass of the practice "
            "snapshot (if any) names exactly that decision record;\n"
            f" 4. the test namespace {k['test']} has no capture / plan / exec / score row;\n"
            f" 5. FREEZE-SUMMARY-{scene}.md is the summary generated for this record, and every rehearsal capture_start of "
            f"{k['rehearsal']} was preceded by an applied k23_capture_freeze_view event.{fp17}\n"
            "ANSWER FORMAT (machine-checked). If and only if you accept with NO condition, your ENTIRE answer must be exactly these four "
            "lines and nothing else (no explanation, no notes, no markdown):\n"
            f"{acc}\n"
            "Any other answer is treated as NOT accepted. Otherwise answer with a first line \"RULING: CHANGES REQUIRED\", followed by the "
            "problems.\n")


def evidence(step: str) -> list:
    """Repo-relative evidence paths committed after `step` (filtered to existing, non-ignored paths at commit time). Kitchen 1's pattern:
    practice journal + snapshot, the rehearsal journals, the records, the test seal (test journals and outcomes are not committed)."""
    if step == "P2":
        return ["docs/final-test/matrix/practice-k2-v1", "sim/final_test/snapshots_ft2/FloorPlan15",
                "sim/final_test/k23_run/practice/practice-k2-v1"]
    if step == "R2":
        return ["docs/final-test/matrix/rehearsal-k2-v1", "sim/final_test/k23_run/rehearsal-k2-v1-chain.log"]
    if step in ("S2", "S3"):
        return ["docs/final-test/k23/a29_size_decision.json"]
    if step in ("F2", "F3"):
        sc = "FloorPlan15" if step == "F2" else "FloorPlan17"
        return [f"docs/final-test/k23/freeze_{sc}.json", f"docs/final-test/k23/FREEZE-SUMMARY-{sc}.md", "docs/final-test/k23/freeze_history"]
    if step in ("T2", "T3"):
        ns = "test-k2-v1" if step == "T2" else "test-k3-v1"
        return [f"docs/final-test/matrix/{ns}/score.seal.json", f"sim/final_test/k23_run/{ns}-chain.log"]
    return []


def safe_path(p: str) -> bool:
    """Never commit the quarantined applicability rows, the preregistration, or anything outside the repo."""
    q = p.replace("\\", "/")
    return not (q.startswith("/") or ".." in q.split("/") or "/applicability" in "/" + q or q == PREREG)


def chain_log_done(text: str, ns: str) -> bool:
    """chain_k23.sh's own record: a 'CHAIN DONE <ns>' line after the LAST 'CHAIN START <ns>' line."""
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if f"CHAIN START {ns} " in ln + " "]
    return bool(starts) and any(f"CHAIN DONE {ns} " in ln + " " for ln in lines[starts[-1] + 1:])


def chain_log_problems(text: str, ns: str, require_done: bool) -> list:
    """Per chain run of `ns` (each 'CHAIN START <ns>' block): an execute session that ended with an rc other than 0 / 5 (5 = the authorized
    300-execution interruption) in a block whose execute stage then passed is a MASKED failure (Astra K23-OVERNIGHT-r2 item 2). With
    require_done: 'CHAIN DONE <ns>' after the last 'CHAIN START <ns>'."""
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if f"CHAIN START {ns} " in ln + " "]
    why = []
    for k, i in enumerate(starts):
        block = lines[i:starts[k + 1] if k + 1 < len(starts) else len(lines)]
        bad = [int(m.group(1)) for ln in block for m in [re.search(r"execute session end rc=(-?\d+)", ln)] if m and int(m.group(1)) not in (0, 5)]
        if bad and any(re.search(r"STAGE execute rc=0(\s|$)", ln) for ln in block):
            why.append(f"chain run #{k + 1} of {ns}: an execute session ended rc {bad[0]} but the execute stage passed (masked failure)")
    if require_done and not chain_log_done(text, ns):
        why.append(f"no 'CHAIN DONE {ns}' after the last 'CHAIN START {ns}' in its chain log")
    return why


def chain_job(ns: str) -> str:
    """chain_k23.sh's job prefix: J=${NS//-v1/}; J=${J//rehearsal-/r}; J=${J//test-/t}."""
    return ns.replace("-v1", "").replace("rehearsal-", "r").replace("test-", "t")


def _sha_or_none(p):
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    except OSError:
        return None


def strict_wrapper_problems(root: str) -> list:
    """The strict wrappers equal their rendered originals (k23_strict_wrappers.check) in THIS checkout."""
    import importlib.util
    p = os.path.join(root, "sim", "final_test", "k23_run", "k23_strict_wrappers.py")
    if not os.path.exists(p):
        return ["k23_strict_wrappers.py missing"]
    spec = importlib.util.spec_from_file_location("k23_strict_wrappers_chk", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.check(os.path.dirname(p))


# ================================================================================================ the machine (real side effects)
def _env(extra=None) -> dict:
    """The process environment + utf-8 / no-bytecode + `extra` (a None value removes the variable, e.g. ENERGY_WAIVER)."""
    e = dict(os.environ)
    e.update(PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    for k, v in (extra or {}).items():
        if v is None:
            e.pop(k, None)
        else:
            e[k] = str(v)
    return e


class Sys:
    def __init__(self, root: str, logf):
        self.root, self.logf = root, logf

    def now(self) -> float:
        return time.time()

    def sleep(self, s: float):
        time.sleep(s)

    def sh(self, argv, *, env=None, timeout=None, stream=False) -> tuple:
        """Run argv in the repo root; output is appended to the overnight log (stream) or captured and returned."""
        e = _env(env)
        try:
            if stream:
                with open(self.logf, "a", encoding="utf-8") as f:
                    r = subprocess.run(argv, cwd=self.root, env=e, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
                return r.returncode, ""
            r = subprocess.run(argv, cwd=self.root, env=e, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
            return r.returncode, (r.stdout or "") + (r.stderr or "")
        except subprocess.TimeoutExpired:
            return 124, f"timeout after {timeout}s"
        except OSError as ex:
            return 127, f"{type(ex).__name__}: {ex}"

    def spawn(self, argv, *, env=None, out=None, detached=False):
        e = _env(env)
        flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if (detached and os.name == "nt") else 0
        fh = open(out, "a", encoding="utf-8") if out else subprocess.DEVNULL
        return subprocess.Popen(argv, cwd=self.root, env=e, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags)

    def http_tags(self):
        import urllib.request
        try:
            with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=10) as r:
                return [m.get("name") for m in json.load(r).get("models") or []]
        except Exception:                                    # noqa: BLE001
            return None

    def gpu(self) -> dict:
        rc, out = self.sh(["nvidia-smi", "--query-gpu=memory.used,temperature.gpu", "--format=csv,noheader,nounits"], timeout=60)
        try:
            mib, temp = [int(x.strip()) for x in out.strip().splitlines()[0].split(",")[:2]]
            return {"mib": mib, "temp": temp}
        except (ValueError, IndexError):
            return {"mib": None, "temp": None, "error": out.strip()[:200]}

    def win_cmdlines(self) -> list:
        rc, out = self.sh(["powershell", "-NoProfile", "-Command",
                           "Get-CimInstance Win32_Process | ForEach-Object { [string]$_.ProcessId + ' ' + $_.CommandLine }"], timeout=120)
        me = str(os.getpid())
        return [ln.strip() for ln in out.splitlines() if ln.strip() and not ln.startswith(me + " ")] if rc == 0 else None

    def which(self, exe):
        return shutil.which(exe)

    def exists(self, path):
        return os.path.exists(path)

    def rename(self, src, dst):
        os.replace(src, dst)

    def read(self, path):
        try:
            return open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            return None

    def mtime(self, path):
        try:
            return os.path.getmtime(path)
        except OSError:
            return None

    def listdir(self, path):
        try:
            return os.listdir(path)
        except OSError:
            return []

    def pid_alive(self, pid: int) -> bool:
        if os.name != "nt" or not pid:
            return False
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))                # PROCESS_QUERY_LIMITED_INFORMATION; never a signal
        if not h:
            return False
        code = ctypes.c_ulong()
        k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return code.value == 259                                  # STILL_ACTIVE


# ================================================================================================ repo state (reads only)
class State:
    def __init__(self, root: str, sysx: Sys):
        self.root, self.sys = root, sysx
        if MATRIX not in sys.path:
            sys.path.insert(0, MATRIX)

    def p(self, rel):
        return os.path.join(self.root, *rel.split("/"))

    def _json_cli(self, argv, timeout=900):
        rc, out = self.sys.sh([sys.executable] + argv, timeout=timeout)
        i = out.find("{")
        try:
            return rc, json.loads(out[i:out.rfind("}") + 1]) if i >= 0 else None
        except ValueError:
            return rc, None

    def integrity(self) -> list:
        why = []
        try:
            got = hashlib.sha256(open(self.p(PREREG), "rb").read()).hexdigest()
        except OSError as e:
            got = f"unreadable ({e})"
        if got != PREREG_SHA:
            why.append(f"preregistration.md sha256 {got[:16]} != {PREREG_SHA[:16]}")
        rc, out = self.sys.sh([sys.executable, "-c", "import json, sys; sys.path.insert(0, 'sim/final_test/matrix'); import treatment as T; "
                               "print('PINS ' + json.dumps(T.pin_problems()))"], timeout=600)
        line = next((ln for ln in out.splitlines() if ln.startswith("PINS ")), None)
        pins = json.loads(line[5:]) if (rc == 0 and line) else [f"pin check failed (rc {rc}): {out.strip()[-200:]}"]
        if pins:
            why.append(f"treatment.pin_problems() = {pins[:3]}")
        return why

    def coco(self) -> dict:
        p = self.p(COCO_LOG)
        try:
            text = open(p, "rb").read().decode("utf-8", "replace")
            age = self.sys.now() - os.path.getmtime(p)
        except OSError:
            text, age = "", None
        lg = parse_coco_log(text)
        lg.update(age_s=None if age is None else round(age), verdict=coco_verdict(lg, age), weights=os.path.exists(self.p(COCO_WEIGHTS)))
        return lg

    def coco_freeze(self) -> dict:
        import k23_entry as E
        d = self.p("sim/kitchen_v3/FloorPlan15")
        try:
            meta = json.load(open(os.path.join(d, "frozen_weights.meta.json"), encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        try:
            listed = any(ln.split(None, 1)[1].strip().lstrip("*") == COCO_WEIGHTS
                         for ln in open(os.path.join(d, "frozen_weights.sha256"), encoding="utf-8") if len(ln.split(None, 1)) == 2)
        except OSError:
            listed = False
        return {"problems": E.coco_problems(self.root), "pending": meta.get("coco_pending"), "listed": listed,
                "append_meta": isinstance(meta.get("coco_append"), dict)}

    def practice(self, ns) -> dict:
        rc, st = self._json_cli(["sim/final_test/matrix/k23_practice.py", "--namespace", ns, "--status"])
        att = ((st or {}).get("orders") or {}).get("0", {}).get("attempts", 0)
        return {"rc": rc, "ok": bool(st and st.get("ok")), "attempts": att, "done": rc == 0 and bool(st and st.get("ok")) and att >= 10}

    def status(self, ns) -> dict:
        rc, st = self._json_cli(["sim/final_test/matrix/k23_run.py", "--phase", "status", "--namespace", ns])
        return st if (rc == 0 and st) else {"phases": {}, "error": f"status rc {rc}"}

    def chain_alive(self, ns):
        lines = self.sys.win_cmdlines()
        if lines is None:
            return None
        return [ln[:160] for ln in lines if runs_ns(ln.split(" ", 1)[1] if " " in ln else "", (ns,))]

    def _upstream(self, ns) -> list:
        """capture / plan / execute complete and unchanged, the score sealed, the seal matching score.jsonl."""
        import common as C
        ph = self.status(ns).get("phases") or {}
        why = [f"{k} not complete / changed" for k in ("capture", "plan", "execute")
               if not ((ph.get(k) or {}).get("complete") and (ph.get(k) or {}).get("unchanged") is not False)]
        if not (ph.get("score") or {}).get("complete"):
            why.append("score not sealed complete")
        seal, sp = os.path.join(C.ns_docs(ns, self.root), "score.seal.json"), C.journal_path(ns, "score", self.root)
        try:
            if json.load(open(seal, encoding="utf-8")).get("sha256") != C.sha256_file(sp):
                why.append("score.jsonl no longer matches its seal")
        except (OSError, ValueError):
            why.append("score seal / score.jsonl unreadable")
        return why

    def _chain_log(self, ns) -> str:
        return self.sys.read(self.p(f"sim/final_test/k23_run/{ns}-chain.log")) or ""

    def chain_evidence(self, ns) -> list:
        """Durable evidence that the TEST chain of `ns` ended successfully (Astra K23-OVERNIGHT-r1 item 5, r2 item 2)."""
        why = chain_log_problems(self._chain_log(ns), ns, True) + self._upstream(ns)
        alive = self.chain_alive(ns)
        if alive is None or alive:
            why.append(f"a chain process of {ns} may still be alive ({alive if alive else 'process list unreadable'})")
        return why

    def giso(self, ns) -> dict:
        """The rehearsal's G-iso result and whether the ONLY supported waiver (Arash's K23-GISO-WAIVER-FloorPlan17, with exactly the agent
        freeze's conditions) covers it. Any other non-pass G-iso -- a 'fail', any Kitchen 2 one -- is not waivable here (Astra r3 A2)."""
        import common as C
        p = os.path.join(C.ns_docs(ns, self.root), "g_iso.json")
        try:
            raw = open(p, "rb").read()
            gi = json.loads(raw.decode("utf-8"))
        except OSError:
            return {"status": None, "waived": False, "waiver_why": ["no G-iso result"], "waiver": None}
        except ValueError:
            return {"status": "unreadable", "waived": False, "waiver_why": ["g_iso.json unreadable"], "waiver": None}
        st = str(gi.get("status") or "missing-status")
        scene = SCENE_OF.get(ns)
        rel = GISO_WAIVER.format(scene=scene)
        if st == "pass":
            return {"status": st, "waived": False, "waiver_why": [], "waiver": rel}
        if scene != "FloorPlan17":
            return {"status": st, "waived": False, "waiver": rel,
                    "waiver_why": [f"no supported waiver for {scene} (only Arash's K23-GISO-WAIVER-FloorPlan17 exists)"]}
        import k23_giso_waiver as GW
        why = GW.waiver_problems(self.root)
        return {"status": st, "waived": not why, "waiver_why": why, "waiver": rel,
                "waiver_decision_sha256": GW.decision_sha(self.root) if not why else None}

    def context_record(self, ns) -> dict:
        """The runner's own record of a context stage that exited 0 after a waived G-iso (overnight/context_resume_<ns>.json)."""
        import common as C
        try:
            rec = json.load(open(self.p(f"{OUT}/context_resume_{ns}.json"), encoding="utf-8"))
        except (OSError, ValueError):
            return {"ok": False, "why": "no runner record of a successful context stage"}
        cf, gp = os.path.join(C.ns_docs(ns, self.root), "context_fit.json"), os.path.join(C.ns_docs(ns, self.root), "g_iso.json")
        g = self.giso(ns)
        ok = (rec.get("namespace") == ns and rec.get("rc") == 0 and rec.get("context_fit_sha256") == _sha_or_none(cf)
              and rec.get("g_iso_sha256") == _sha_or_none(gp) and g["waived"] and rec.get("waiver_decision_sha256") == g.get("waiver_decision_sha256"))
        return {"ok": ok, "why": None if ok else "the runner's context record does not match the current context fit / G-iso / waiver"}

    def rehearsal(self, ns) -> dict:
        """The chain's own done-checks (capture / plan / execute / score / G-iso / context fit), no masked execute failure, no chain alive.
        context_only: everything but the context fit holds and the non-pass G-iso is waived (resume at the context stage only).
        giso_block: a non-pass G-iso without a valid waiver (a STOP for Arash; never re-run)."""
        import common as C
        up = self._upstream(ns) + chain_log_problems(self._chain_log(ns), ns, False)
        g = self.giso(ns)
        cfp = os.path.join(C.ns_docs(ns, self.root), "context_fit.json")
        if not os.path.exists(cfp):
            ctx_state, ctx_ok = "absent", False                     # the context stage was never attempted
        else:
            try:
                cf = json.load(open(cfp, encoding="utf-8"))
                cap = C.journal_path(ns, "capture", self.root)
                ctx_ok = cf.get("freeze_grade") is True and os.path.exists(cap) and cf.get("capture_sha") == C.sha256_file(cap)
            except (OSError, ValueError):
                ctx_ok = False
            ctx_state = "ok" if ctx_ok else "bad"                   # failed / stale / unreadable: never re-run (Astra r4 1)
        alive = self.chain_alive(ns)
        why = list(up)
        if g["status"] != "pass" and not g["waived"]:
            why.append(f"G-iso {g['status'] or 'not run'}" + (f" without a valid waiver ({'; '.join(g['waiver_why'][:2])})" if g["status"] else ""))
        if ctx_state == "absent":
            why.append("context fit not measured")
        elif ctx_state == "bad":
            why.append("a context-fit report exists but is not freeze-grade on this capture (failed / stale / unreadable): it is never re-run")
        if alive is None or alive:
            why.append(f"a chain process of {ns} may still be alive ({alive if alive else 'process list unreadable'})")
        if not chain_log_done(self._chain_log(ns), ns):              # durable SUCCESS evidence (Astra r3 A1)
            cr = self.context_record(ns) if g["waived"] else {"ok": False, "why": None}
            if not cr["ok"]:
                why.append(f"no successful-completion evidence: no 'CHAIN DONE {ns}' after its last 'CHAIN START'"
                           + (f" and {cr['why']}" if g["waived"] else ""))
        return {"done": not why, "why": why, "giso": g, "alive": alive, "ctx_state": ctx_state,
                "context_only": (not up) and g["status"] not in (None, "pass") and g["waived"] and ctx_state == "absent" and alive == [],
                "giso_block": g["status"] not in (None, "pass") and not g["waived"]}

    def test_done(self, ns) -> dict:
        why = self.chain_evidence(ns)
        return {"done": not why, "why": why}

    def k2_plan_complete(self) -> bool:
        import common as C
        p = C.journal_path("test-k2-v1", "events", self.root)
        evs = C.Journal(p).rows() if os.path.exists(p) else []
        return any(e.get("event") == "plan_first_complete" and e.get("scope") == "full" for e in evs)

    def size_checkpoints(self) -> list:
        try:
            rec = json.load(open(self.p("docs/final-test/k23/a29_size_decision.json"), encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [c.get("checkpoint") for c in rec.get("checkpoints") or []]

    def freeze(self, scene) -> dict:
        try:
            fz = json.load(open(self.p(f"docs/final-test/k23/freeze_{scene}.json"), encoding="utf-8"))
        except (OSError, ValueError):
            return {"exists": False}
        return {"exists": True, "sha": fz.get("sha256"), "all_gates_pass": fz.get("all_gates_pass") is True,
                "presigned": bool(fz.get("standing_presignature") or fz.get("arash_check")),
                "presignature": fz.get("standing_presignature"), "arash_check": fz.get("arash_check")}

    def k3_busy(self):
        lines = self.sys.win_cmdlines()
        if lines is None:
            return None
        return [ln[:160] for ln in lines if busy_match(ln.split(" ", 1)[1] if " " in ln else "")]

    def training(self, run_wsl) -> list | None:
        # wsl.exe hands "--" commands to bash -c, so the "|" alternation must be single-quoted (unquoted, bash ran it as a pipeline:
        # rc 127 -> "cannot list WSL processes", the 2026-10-20 23:18 and 23:30 stops)
        rc, out = run_wsl(["pgrep", "-af", "'" + TRAIN_PROCS + "'"], cd=False)
        if rc == 1:
            return []
        return [ln for ln in out.splitlines() if ln.strip()] if rc == 0 else None


# ================================================================================================ the runner
class Runner:
    def __init__(self, root: str, sysx, state, *, coco_resume=True, max_wait_h=36.0, branch="master", poll_s=POLL_S):
        self.root, self.sys, self.st = root, sysx, state
        self.coco_resume, self.max_wait_h, self.branch, self.poll = coco_resume, max_wait_h, branch, poll_s
        self.out = os.path.join(root, *OUT.split("/"))
        self.logf = os.path.join(self.out, "overnight.log")
        self.resumes = 0
        self.current = None
        self.abort_check = None                                   # set during the K3 work: polls the Kitchen 2 chain before each command
        self.chain_t0 = {}                                        # ns -> when this runner started its chain (stall check scope)

    # ---- logging
    def log(self, msg):
        os.makedirs(self.out, exist_ok=True)
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        with open(self.logf, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def record(self, step, what, rc, t0, note=""):
        os.makedirs(self.out, exist_ok=True)
        with open(os.path.join(self.out, "steps.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "what": what, "rc": rc, "start": t0, "end": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "note": note[:300]}, ensure_ascii=False) + "\n")

    def status_file(self, **kw):
        os.makedirs(self.out, exist_ok=True)
        with open(os.path.join(self.out, "status.json"), "w", encoding="utf-8") as f:
            json.dump(dict(kw, step=self.current, ts=time.strftime("%Y-%m-%dT%H:%M:%S"), pid=os.getpid()), f, indent=1, default=str)

    def _abort(self):
        """During the K3 work: the Kitchen 2 chain is polled before every command, commit, WSL call and Ollama start (Astra r2 item 4)."""
        if self.abort_check is not None:
            self.abort_check()

    def cmd(self, step, what, argv, *, env=None, timeout=None, stream=True, ok=(0,)):
        """One command: logged with its rc; a rc outside `ok` stops the runner."""
        self._abort()
        t0 = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.log(f"[{step}] RUN {what}: {' '.join(argv)}")
        rc, out = self.sys.sh(argv, env=env, timeout=timeout, stream=stream)
        if out:
            with open(self.logf, "a", encoding="utf-8") as f:
                f.write(out if out.endswith("\n") else out + "\n")
        self.log(f"[{step}] {what} rc={rc}")
        self.record(step, what, rc, t0, out.strip()[-300:] if out else "")
        if ok is not None and rc not in ok:
            raise Stop(step, f"{what} failed (rc {rc})", rc if isinstance(rc, int) and 0 < rc < 256 else 1)
        return rc, out

    def wsl(self, args, *, cd=True, step="wsl", tries=10, wait=60):
        """A WSL command; WSL connection refusals (signature or rc -1 / 255) retried like k23_k3_practice_now.ps1 (10 x 60 s)."""
        self._abort()
        argv = ["wsl", "-d", DISTRO] + (["--cd", self.root] if cd else []) + ["--"] + list(args)
        for t in range(1, tries + 1):
            rc, out = self.sys.sh(argv, timeout=3600)
            if not (WSL_SIGNATURE.search(out or "") or rc in WSL_RCS):
                return rc, out
            self.log(f"[{step}] WSL refused the connection (try {t}, rc {rc}); retrying in {wait} s")
            self.sys.sleep(wait)
        return 255, "WSL unreachable"

    def guard(self, step):
        why = self.st.integrity()
        if why:
            raise Stop(step, "integrity: " + "; ".join(why), 9)

    # ---- git (add + commit only)
    def git(self, *args, timeout=300):
        return self.sys.sh(["git"] + list(args), timeout=timeout)

    def git_state(self, step):
        """HEAD is the expected branch and no merge / rebase / cherry-pick is in progress (before the first step and before every commit)."""
        rc, head = self.git("rev-parse", "--abbrev-ref", "HEAD")
        if rc != 0 or head.strip() != self.branch:
            raise Stop(step, f"git: HEAD is {head.strip()!r}, not {self.branch!r}", 10)
        rc, gd = self.git("rev-parse", "--git-dir")
        gdir = gd.strip() if os.path.isabs(gd.strip()) else os.path.join(self.root, gd.strip())
        if any(self.sys.exists(os.path.join(gdir, x)) for x in ("MERGE_HEAD", "rebase-merge", "rebase-apply", "CHERRY_PICK_HEAD")):
            raise Stop(step, "git: a merge / rebase / cherry-pick is in progress", 10)

    def commit(self, step, paths, msg):
        self._abort()                                               # before git add: never a half-staged commit
        self.git_state(step)
        keep = []
        for p in list(paths) + [f"{OUT}/overnight.log", f"{OUT}/steps.jsonl"]:
            if not safe_path(p) or not self.sys.exists(os.path.join(self.root, *p.split("/"))):
                continue
            if self.git("check-ignore", "-q", p)[0] == 0:
                self.log(f"[{step}] not committed (gitignored): {p}")
                continue
            keep.append(p)
        if keep:
            rc, out = self.git("add", "--", *keep)
            if rc != 0:
                raise Stop(step, f"git add failed: {out.strip()[-200:]}", 10)
            staged = self.git("diff", "--cached", "--name-only", "--", *keep)[1].split()
            if staged:
                body = f"{msg}\n\nAutomated evidence commit by sim/final_test/k23_run/k23_overnight.py (git add + commit only).\n\n" \
                       "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
                for t in range(3):
                    rc, out = self.git("commit", "-q", "-m", body, "--", *keep)
                    if rc == 0:
                        break
                    self.log(f"[{step}] git commit rc {rc} ({out.strip()[-120:]}); retry in 30 s")
                    self.sys.sleep(30)
                if rc != 0:
                    raise Stop(step, "git commit failed 3 times", 10)
                self.log(f"[{step}] committed {len(staged)} file(s): {self.git('log', '--oneline', '-1')[1].strip()}")
        why = self.st.integrity()                                   # after every commit: prereg 247ef23a + pins
        if why:
            raise Stop(step, "integrity after commit: " + "; ".join(why), 9)

    # ---- GPU / Ollama / lock
    def no_training(self, step):
        tr = self.st.training(lambda a, cd=False: self.wsl(a, cd=cd, step=step))
        if tr is None:
            raise Stop(step, "cannot list WSL processes (training check)", 11)
        if tr:
            raise Stop(step, f"a training process is running on the GPU: {tr[:2]}", 11)

    def ensure_ollama(self, step):
        self._abort()
        tags = self.sys.http_tags()
        if tags is not None and GEMMA in tags:
            return
        self.no_training(step)
        g = self.sys.gpu()
        if g.get("mib") is None or g["mib"] >= 2000:
            raise Stop(step, f"GPU not idle before starting Ollama ({g})", 11)
        if not self.sys.exists(OLLAMA_EXE):
            raise Stop(step, f"Ollama not found at {OLLAMA_EXE}", 11)
        env = {} if os.environ.get("OLLAMA_HOST") else {"OLLAMA_HOST": "0.0.0.0"}     # WSL reaches it through the Windows host IP
        self.log(f"[{step}] starting Ollama (serve, hidden){' OLLAMA_HOST=0.0.0.0' if env else ''}")
        self.sys.spawn([OLLAMA_EXE, "serve"], env=env, detached=True)
        for _ in range(30):
            self.sys.sleep(2)
            tags = self.sys.http_tags()
            if tags is not None and GEMMA in tags:
                self.log(f"[{step}] Ollama up ({GEMMA})")
                return
        raise Stop(step, f"Ollama did not come up with {GEMMA}", 11)

    def wait_lock(self, step):
        """The ONE sim lock is free (flock -n in WSL); waited for at most LOCK_WAIT_S, never stolen, its holder never touched."""
        t_end = self.sys.now() + LOCK_WAIT_S
        while True:
            rc, out = self.wsl(["flock", "-n", SIM_LOCK, "true"], cd=False, step=step)
            if rc == 0:
                return
            if rc != 1:
                raise Stop(step, f"cannot test the sim lock (rc {rc}: {out.strip()[-120:]})", 11)
            if self.sys.now() >= t_end:
                raise Stop(step, f"the sim lock is held by another job for more than {LOCK_WAIT_S // 60} min (not stolen; left as it is)", 11)
            self.log(f"[{step}] the sim lock is held by another job; waiting (never stolen)")
            self.sys.sleep(60)

    def lock_stall(self, step, ns):
        """A supervised chain whose WSL job has sat 'queued' (waiting for the sim lock) for more than LOCK_WAIT_S -> stop (no kill)."""
        base = os.path.join(self.root, *LOGS.split("/"))
        for d in self.sys.listdir(base):
            if not d.startswith(chain_job(ns) + "-"):
                continue
            pf = os.path.join(base, d, "progress.txt")
            txt, mt = self.sys.read(pf), self.sys.mtime(pf)
            last = next((ln for ln in reversed((txt or "").splitlines()) if ln.strip()), "")
            if mt is None or mt < self.chain_t0.get(ns, 0):
                continue                                           # a job file of an earlier run, not of this chain
            if "queued" in last and self.sys.now() - mt > LOCK_WAIT_S:
                raise Stop(step, f"{ns}: job {d} has waited for the sim lock for more than {LOCK_WAIT_S // 60} min (the chain is left "
                                 "waiting; nothing killed or stolen)", 11)

    # ---- steps
    def preflight(self):
        self.current = "start"
        self.git_state("start")
        sw = strict_wrapper_problems(self.root)
        if sw:
            raise Stop("start", "strict wrappers: " + "; ".join(sw), 12)
        self.guard("start")

    def step_wait(self):
        s = self.current = "W"
        self.guard(s)
        t_end = self.sys.now() + self.max_wait_h * 3600
        grace, last_note = 0, None
        while True:
            reh = self.st.rehearsal("rehearsal-k3-v1")
            co = self.st.coco()
            busy = self.st.k3_busy()
            k3_ready = reh["done"] or reh["context_only"]
            if k3_ready and co["verdict"] == "ended_ok":
                self.log(f"[W] rehearsal-k3-v1 {'complete' if reh['done'] else 'complete up to its context fit (G-iso waived)'}; "
                         f"COCO ended rc=0 at {co['last_ts']}")
                self.record(s, "wait", 0, "", "")
                return
            if busy is None:
                raise Stop(s, "cannot list Windows processes", 11)
            if reh["giso_block"] and not busy:
                raise Stop(s, self.giso_stop_text("rehearsal-k3-v1", reh), 4)
            if not k3_ready and not busy:
                grace += 1
                if grace >= 3:
                    raise Stop(s, "rehearsal-k3-v1 is not complete and nothing is running it: " + "; ".join(reh["why"][:4]), 4)
            else:
                grace = 0
            v = co["verdict"]
            if v == "safety":
                raise Stop(s, f"the COCO run stopped on a safety event {co['safety']} (left for the morning)", 5)
            if v == "failed":
                raise Stop(s, f"the COCO run ended with rc {co['rc']} (not a pause): left for the morning", 5)
            if v in ("resumable", "none") and k3_ready and not busy:
                if v == "none":
                    raise Stop(s, f"no START / END marker in {COCO_LOG}", 5)
                if not self.coco_resume or self.resumes >= MAX_COCO_RESUMES:
                    raise Stop(s, f"the COCO run is stopped ({co['last']} rc {co['rc']}) and may not be resumed "
                                  f"({'--no-coco-resume' if not self.coco_resume else f'{self.resumes} resumes used'})", 5)
                self.resume_coco(co)
            if self.sys.now() > t_end:
                raise Stop(s, f"waited more than {self.max_wait_h} h", 6)
            if last_note is None or self.sys.now() - last_note >= 1800:
                last_note = self.sys.now()
                self.log(f"[W] waiting: rehearsal-k3-v1 {'complete' if reh['done'] else 'not complete (' + '; '.join(reh['why'][:2]) + ')'}; "
                         f"K3 processes {len(busy)}; COCO {v} (last {co['last']} rc {co['rc']} at {co['last_ts']}, log age {co['age_s']} s)")
                self.status_file(rehearsal_k3=reh, coco=co)
            self.sys.sleep(self.poll)

    def resume_coco(self, co):
        """Resume a paused / WSL-killed COCO run (train_k23.sh resumes from its checkpoint). Only with the WSL process list readable, no
        training process, the GPU idle; it counts as resumed only when a NEW START marker and a training process appear."""
        tr = self.st.training(lambda a, cd=False: self.wsl(a, cd=cd, step="W"))
        if tr is None:
            raise Stop("W", "COCO resume: the WSL process list is unreadable (state unknown; nothing launched)", 5)
        if tr:
            raise Stop("W", f"COCO resume: a training process is running although the log says stopped ({tr[:1]}); left for the morning", 5)
        g = self.sys.gpu()
        if g.get("mib") is None or g["mib"] >= 2000:
            raise Stop("W", f"COCO resume: the GPU is not idle ({g}); nothing launched", 5)
        self.resumes += 1
        n0 = co.get("n_start", 0)
        self.log(f"[W] resuming the COCO run (train_k23.sh; resume {self.resumes} of {MAX_COCO_RESUMES}; START markers so far {n0})")
        self.record("W", f"COCO resume {self.resumes}", 0, time.strftime("%Y-%m-%dT%H:%M:%S"), "")
        self.sys.spawn(["wsl", "-d", DISTRO, "--cd", self.root, "--"] + COCO_TRAIN, detached=True)
        for _ in range(10):
            self.sys.sleep(60)
            c2 = self.st.coco()
            tr = self.st.training(lambda a, cd=False: self.wsl(a, cd=cd, step="W"))
            if c2.get("n_start", 0) > n0 and c2.get("last") == "START" and tr:
                self.log(f"[W] COCO resumed (new START at {c2['last_ts']}, training process {tr[0][:80]})")
                return
        raise Stop("W", "the COCO resume did not show a new START marker and a training process within 10 min", 5)

    def step_coco(self):
        s = self.current = "C"
        self.guard(s)
        cf = self.st.coco_freeze()
        if not cf["problems"]:
            self.log("[C] COCO already appended to the FloorPlan15 perception freeze with current metadata: skipped")
            return
        if not self.st.coco()["weights"]:
            raise Stop(s, f"{COCO_WEIGHTS} missing after END rc=0", 5)
        if cf["pending"] is True and not cf["listed"]:
            self.wsl_cmd(s, "k23_freeze --add-coco", [WSL_PY, "sim/kitchen_v3/k23/k23_freeze.py", "--scene", "FloorPlan15", "--add-coco"])
            self.wsl_cmd(s, "k23_coco_meta --write", [WSL_PY, "sim/kitchen_v3/k23/k23_coco_meta.py", "--write"])
        elif cf["pending"] is False and cf["listed"] and not cf["append_meta"]:
            self.wsl_cmd(s, "k23_coco_meta --write", [WSL_PY, "sim/kitchen_v3/k23/k23_coco_meta.py", "--write"])
        else:
            raise Stop(s, f"unexpected FloorPlan15 COCO freeze state {cf} (left for the morning)", 5)
        self.wsl_cmd(s, "k23_coco_meta check", [WSL_PY, "sim/kitchen_v3/k23/k23_coco_meta.py"])
        left = self.st.coco_freeze()["problems"]
        if left:
            raise Stop(s, "coco_problems after the append: " + "; ".join(left), 5)
        self.commit(s, [], "K23 overnight C: FloorPlan15 COCO-start sensitivity appended to the perception freeze + successor metadata")

    def wsl_cmd(self, step, what, args):
        t0 = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.log(f"[{step}] RUN (WSL) {what}: {' '.join(args)}")
        rc, out = self.wsl(args, step=step)
        with open(self.logf, "a", encoding="utf-8") as f:
            f.write((out or "") + "\n")
        self.log(f"[{step}] {what} rc={rc}")
        self.record(step, what, rc, t0, (out or "").strip()[-300:])
        if rc != 0:
            raise Stop(step, f"{what} failed (rc {rc})", 5)

    def step_ollama(self):
        s = self.current = "O"
        self.guard(s)
        self.ensure_ollama(s)

    def giso_stop_text(self, ns, reh) -> str:
        g = reh["giso"]
        return (f"G-iso of {ns} is {g['status']!r} (not a pass): under A20 / A20.1 / A21.A.6 this needs Arash's written waiver; the runner "
                f"never re-runs a G-iso. Waiver record expected: {g.get('waiver')} ({'; '.join(g['waiver_why'][:2])})")

    def run_context(self, step, ns):
        """The chain's context stage only (k23_run --phase context on the Gemma server); earlier stages are never re-run. On exit 0 the runner
        records it (overnight/context_resume_<ns>.json, bound to the context_fit / g_iso / waiver shas): the rehearsal's completion evidence
        after a waived G-iso (Astra r3 A1)."""
        g = self.st.giso(ns)
        if not g.get("waived"):
            raise Stop(step, f"context-stage resume of {ns} without a valid G-iso waiver", 4)
        self.ensure_ollama(step)
        self.no_training(step)                                      # the context fit is measured on the Gemma server: GPU free of training
        self.cmd(step, f"context fit {ns} (resume at the context stage)", [sys.executable, "sim/final_test/matrix/k23_run.py", "--phase", "context",
                                                                           "--namespace", ns, "--ollama", OLLAMA_URL], timeout=7200)
        import common as C
        d = C.ns_docs(ns, self.root)
        rec = {"namespace": ns, "rc": 0, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "step": step,
               "context_fit_sha256": _sha_or_none(os.path.join(d, "context_fit.json")), "g_iso_sha256": _sha_or_none(os.path.join(d, "g_iso.json")),
               "waiver_decision_sha256": g.get("waiver_decision_sha256"), "runner": "sim/final_test/k23_run/k23_overnight.py"}
        with open(os.path.join(self.out, f"context_resume_{ns}.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1)

    def step_context_resume(self, step, ns):
        """R3c: a rehearsal complete up to its context fit with a waived G-iso is resumed at the context stage only."""
        self.current = step
        self.guard(step)
        r = self.st.rehearsal(ns)
        if r["done"]:
            return
        if not r["context_only"]:
            raise Stop(step, f"{ns} is neither complete nor resumable at its context stage: " + "; ".join(r["why"][:4]), 4)
        self.run_context(step, ns)
        r = self.st.rehearsal(ns)
        if not r["done"]:
            raise Stop(step, f"{ns} after the context stage: " + "; ".join(r["why"][:4]), 4)
        self.commit(step, [f"docs/final-test/matrix/{ns}", f"{OUT}/context_resume_{ns}.json"],
                    f"K23 overnight {step}: {ns} context fit measured (G-iso waived by Arash's record)")

    def _revoke(self, tok) -> str:
        """Atomically revoke an UNCLAIMED launch authorization (rename to .revoked) and CONFIRM the outcome (Astra r5 1):
        'revoked' (the token is gone and <token>.revoked exists) | 'claimed' (<token>.claimed exists) | 'unknown' (anything else -- a failed
        rename, a vanished token: never treated as safe)."""
        if self.sys.exists(tok + ".claimed"):
            return "claimed"
        try:
            self.sys.rename(tok, tok + ".revoked")
        except FileNotFoundError:
            pass
        except OSError as e:
            self.log(f"(revoking {os.path.basename(tok)} failed: {type(e).__name__}: {e})")
            return "claimed" if self.sys.exists(tok + ".claimed") else "unknown"
        if self.sys.exists(tok + ".claimed"):
            return "claimed"
        return "revoked" if (self.sys.exists(tok + ".revoked") and not self.sys.exists(tok)) else "unknown"

    @staticmethod
    def _auth_text(state: str) -> str:
        return {"revoked": "its unclaimed launch authorization is revoked, so it can never start later",
                "claimed": "it claimed its authorization (it started): left as it is, nothing killed",
                "unknown": "the revocation of its authorization could NOT be confirmed: left as it is; check sim/final_test/k23_run/overnight/auth "
                           "by hand before anything else"}.get(state, state)

    def run_practice(self, s, ns):
        """The strict practice in the background, supervised: it must log its queued line within PRACTICE_START_S and end within
        PRACTICE_MAX_S (else stop; nothing killed). Its one-time launch authorization is created INSIDE the cleanup scope and revoked on
        EVERY exit while unclaimed (token creation, logging, spawn, stall, timeout, any return, exceptions), so no late launcher can ever start
        practice after this call (Astra r3 A3, r4 3, r5 2); a WSL-signature exit is retried once ONLY when the revocation is CONFIRMED (nothing
        can have run and nothing can start later), with a new authorization; a claimed or uncertain launch is never retried (r5 1)."""
        pf = os.path.join(self.root, "sim", "final_test", "k23_run", "practice", ns, "progress.txt")
        for attempt in (1, 2):
            self.wait_lock(s)
            m0, t0, last_note = self.sys.mtime(pf), self.sys.now(), None
            tok_rel = f"{AUTH_DIR}/practice-{ns}-{int(time.time() * 1000)}-{os.getpid()}-{attempt}.token"
            tok = os.path.join(self.root, *tok_rel.split("/"))
            created, rc = False, None
            try:
                os.makedirs(os.path.dirname(tok), exist_ok=True)
                with open(tok, "w", encoding="utf-8") as f:
                    created = True
                    f.write(f"one-time launch authorization for {ns} (k23_overnight.py pid {os.getpid()})\n")
                argv = [sys.executable, PRACTICE_STRICT, ns, str(LOCK_WAIT_S), tok_rel]
                self.log(f"[{s}] START (background) practice {ns} (strict: one try, no restart): {' '.join(argv)}")
                proc = self.sys.spawn(argv, out=os.path.join(self.out, f"{ns}.console.txt"))
                while True:
                    rc = proc.poll()
                    if rc is not None:
                        break
                    m, el = self.sys.mtime(pf), self.sys.now() - t0
                    if el > PRACTICE_START_S and (m is None or (m0 is not None and m <= m0)) and not self.sys.exists(tok + ".claimed"):
                        raise Stop(s, f"practice {ns} did not start within {PRACTICE_START_S // 60} min (launcher / WSL stalled); "
                                      + self._auth_text(self._revoke(tok)), 4)
                    if el > PRACTICE_MAX_S:
                        raise Stop(s, f"practice {ns} exceeded {PRACTICE_MAX_S // 3600} h; " + self._auth_text(self._revoke(tok)), 4)
                    if last_note is None or self.sys.now() - last_note >= 1800:
                        last_note = self.sys.now()
                        self.log(f"[{s}] practice {ns} running")
                        self.status_file(practice=ns)
                    self.sys.sleep(120)
            finally:
                claim = self._revoke(tok) if created else "not created"     # every exit: an unclaimed authorization never survives
            self.log(f"[{s}] practice {ns} ended rc={rc} (launch authorization {claim})")
            self.record(s, f"practice {ns}", rc, "", f"authorization {claim}")
            if claim == "unknown":
                raise Stop(s, f"practice {ns} ended rc {rc}; " + self._auth_text("unknown"), 4)
            if rc in WSL_RCS and attempt == 1:
                if claim != "revoked":
                    self.log(f"[{s}] practice rc={rc} with the authorization {claim} (practice may have run): no retry")
                    return rc
                self.log(f"[{s}] practice rc={rc} (WSL connection signature, revocation confirmed): one retry in 90 s with a new authorization")
                self.sys.sleep(90)
                continue
            return rc

    def step_practice(self, ns):
        s = self.current = "P2"
        self.guard(s)
        if self.st.practice(ns)["done"]:
            self.log(f"[{s}] {ns} already complete: skipped (its evidence is still committed)")
            self.commit(s, evidence(s), f"K23 overnight P2: {ns} complete (evidence)")
            return
        self.ensure_ollama(s)
        self.no_training(s)
        rc = self.run_practice(s, ns)
        self.cmd(s, f"practice {ns} --status", [sys.executable, "sim/final_test/matrix/k23_practice.py", "--namespace", ns, "--status"], ok=None)
        st = self.st.practice(ns)
        if rc != 0 or not st["done"]:
            raise Stop(s, f"{ns}: strict practice rc {rc}{' (sim lock not acquired)' if rc == 73 else ''}, status {st}", 4)
        self.commit(s, evidence(s), f"K23 overnight P2: {ns} complete (10 attempts, status ok)")

    def chain_env(self, kind):
        env = {"HANG_RELAUNCH": None}                               # the strict chain never relaunches after a hang
        if kind == "test":
            env.update(ENERGY="on", ENERGY_WAIVER=None)             # energy metered (Arash); a waiver is never passed
        return env

    def capture_gate(self, step, ns):
        """The K23 capture artefact gate before every chain start (incl. retries): the reviewed module + its accepted review, then --check."""
        why = capture_review_problems(self.root)
        if why:
            raise Stop(step, "capture artefact gate: " + "; ".join(why), 8)
        self.cmd(step, f"capture artefact check {ns}", [sys.executable, CAPTURE_FREEZE, "--check", "--namespace", ns], stream=False, timeout=900)

    def start_chain(self, step, ns, kind):
        self.capture_gate(step, ns)
        self.ensure_ollama(step)
        self.no_training(step)
        self.wait_lock(step)
        argv = [GIT_BASH, CHAIN, ns]
        env = self.chain_env(kind)
        self.log(f"[{step}] START (background) chain {ns}{' ENERGY=on' if kind == 'test' else ''}: {' '.join(argv)}")
        self.record(step, f"chain {ns} started", 0, time.strftime("%Y-%m-%dT%H:%M:%S"), "")
        self.chain_t0[ns] = self.sys.now()
        return self.sys.spawn(argv, env=env, out=os.path.join(self.out, f"{ns}.console.txt"))

    def supervise(self, step, ns, kind):
        """Start the chain of `ns`, supervise it (status every 30 min, sim-lock stall check), one relaunch after a WSL connection failure;
        returns the final rc."""
        proc, retried, last_note = self.start_chain(step, ns, kind), False, None
        while True:
            rc = proc.poll()
            if rc is not None:
                self.log(f"[{step}] chain {ns} ended rc={rc}")
                self.record(step, f"chain {ns}", rc, "", "")
                if rc in WSL_RCS and not retried:
                    retried = True
                    self.log(f"[{step}] WSL connection signature: one relaunch in 90 s (the chain resumes from its journals)")
                    self.sys.sleep(90)
                    proc = self.start_chain(step, ns, kind)
                    continue
                return rc
            self.lock_stall(step, ns)
            if last_note is None or self.sys.now() - last_note >= 1800:
                last_note = self.sys.now()
                self.log(f"[{step}] chain {ns} running")
                self.status_file(chain=ns)
            self.sys.sleep(120)

    def step_rehearsal(self, ns):
        s = self.current = "R2"
        self.guard(s)
        r = self.st.rehearsal(ns)
        if r["done"]:
            self.log(f"[{s}] {ns} already complete: skipped (its evidence is still committed)")
            self.commit(s, evidence(s), f"K23 overnight R2: {ns} complete (evidence)")
            return
        if r["giso_block"]:
            raise Stop(s, self.giso_stop_text(ns, r), 4)
        if r.get("ctx_state", "absent") != "absent":
            raise Stop(s, f"{ns}: a context-fit report exists without durable success evidence ({'; '.join(r['why'][:3])}); the chain is not "
                          "launched (it would skip the context stage and print CHAIN DONE) and the measurement is never re-run", 4)
        if not r["context_only"]:
            rc = self.supervise(s, ns, "rehearsal")
            r = self.st.rehearsal(ns)
            if rc == 0 and r["done"]:
                self.commit(s, evidence(s), f"K23 overnight R2: {ns} complete (capture, plan, execute, score, G-iso, context fit)")
                return
            if r["giso_block"]:
                raise Stop(s, self.giso_stop_text(ns, r), 4)
            if not r["context_only"]:
                raise Stop(s, f"{ns}: chain rc {rc}; " + ("; ".join(r["why"][:4]) or "done-checks hold"), 4)
        self.run_context(s, ns)
        r = self.st.rehearsal(ns)
        if not r["done"]:
            raise Stop(s, f"{ns} after the context stage: " + "; ".join(r["why"][:4]), 4)
        self.commit(s, evidence(s) + [f"{OUT}/context_resume_{ns}.json"], f"K23 overnight R2: {ns} complete (G-iso waived by Arash's record; context fit measured)")

    def step_size(self, step, checkpoint):
        self.current = step
        self.guard(step)
        sd = "sim/final_test/matrix/k23_size_decision.py"
        if checkpoint in self.st.size_checkpoints():
            self.log(f"[{step}] size decision {checkpoint} already recorded: check only")
        else:
            self.cmd(step, f"size decision {checkpoint} (dry run)", [sys.executable, sd, "--checkpoint", checkpoint], stream=False)
            self.cmd(step, f"size decision {checkpoint} --write", [sys.executable, sd, "--checkpoint", checkpoint, "--write"], stream=False)
        self.cmd(step, "size decision --check", [sys.executable, sd, "--check"], stream=False)
        if checkpoint not in self.st.size_checkpoints():
            raise Stop(step, f"the {checkpoint} checkpoint is not in the record after --write", 7)
        self.commit(step, evidence(step), f"K23 overnight {step}: A29 item 6 size decision {checkpoint} (ledger spend)")

    def signed_ok(self, step, scene, fz):
        """An already-signed freeze: Arash's own check, or a presignature whose review file still has its recorded sha and passes the
        four-line acceptance test for the presigned sha (Astra K23-OVERNIGHT-r1 item 3)."""
        if fz.get("arash_check"):
            return []
        ps = fz.get("presignature") or {}
        rel = str(ps.get("astra_review") or "")
        p = os.path.join(self.root, *rel.replace("\\", "/").split("/")) if rel else ""
        txt = self.sys.read(p) if p else None
        if txt is None:
            return [f"the presignature's review file {rel!r} is missing"]
        why = []
        try:
            if hashlib.sha256(open(p, "rb").read()).hexdigest() != ps.get("astra_review_sha256"):
                why.append("the review file differs from the sha recorded in the presignature")
        except OSError:
            why.append("the review file is unreadable")
        okv, w = astra_verdict(txt, scene, str(ps.get("presigned_sha") or ""))
        if not okv:
            why.append(f"the presignature's review is not the four-line acceptance of {scene} {str(ps.get('presigned_sha'))[:16]}: {w}")
        return why

    def step_freeze(self, step, scene):
        self.current = step
        self.guard(step)
        fa = "sim/final_test/matrix/k23_freeze_agent.py"
        fz = self.st.freeze(scene)
        if fz.get("exists") and fz.get("presigned"):
            why = self.signed_ok(step, scene, fz)
            if why:
                raise Stop(step, f"freeze_{scene}.json is signed, but: " + "; ".join(why), 8)
            self.log(f"[{step}] freeze_{scene}.json already signed (verified): check only (its evidence is still committed)")
            self.cmd(step, f"freeze {scene} --check", [sys.executable, fa, "--scene", scene, "--check"], stream=False, timeout=3600)
            rel = str((fz.get("presignature") or {}).get("astra_review") or "").replace("\\", "/")
            self.commit(step, evidence(step) + ([rel, rel + ".raw.txt", rel + ".prompt.txt"] if rel else []),
                        f"K23 overnight {step}: {scene} agent freeze signed (evidence)")
            return
        self.ensure_ollama(step)                                   # gate 12.9 probes the served Gemma
        self.wsl(["true"], cd=False, step=step)                     # the perception gate reads ~/thor-data through the WSL share
        if not fz.get("exists"):
            self.cmd(step, f"freeze {scene} --dry-run", [sys.executable, fa, "--scene", scene, "--dry-run"], stream=False, timeout=5400)
            self.cmd(step, f"freeze {scene} --write", [sys.executable, fa, "--scene", scene, "--write"], stream=False, timeout=5400)
            fz = self.st.freeze(scene)
        if not (fz.get("exists") and fz.get("all_gates_pass") and fz.get("sha")):
            raise Stop(step, f"freeze_{scene}.json not written with every gate passing ({fz})", 8)
        sha = fz["sha"]
        rev_rel = f"{ASTRA_DIR}/{astra_case(scene, sha)}.txt"
        rev = os.path.join(self.root, *rev_rel.split("/"))
        if not self.sys.exists(rev):
            prompt = rev + ".prompt.txt"
            with open(prompt, "w", encoding="utf-8", newline="\n") as f:
                f.write(astra_prompt(scene, sha, self.root))
            for t in range(1, 4):
                self.cmd(step, f"Astra review {astra_case(scene, sha)} (try {t})",
                         [sys.executable, ASTRA_CALL, "--case", astra_case(scene, sha), "--prompt-file", prompt, "--out", rev,
                          "--cwd", self.root], stream=False, timeout=2400, ok=None)
                if self.sys.exists(rev):
                    break
                self.sys.sleep(600)
            if not self.sys.exists(rev):
                raise Stop(step, "no Astra answer after 3 tries", 8)
        ok, why = astra_verdict(self.sys.read(rev) or "", scene, sha)
        self.log(f"[{step}] Astra {astra_case(scene, sha)}: {why}")
        if not ok:
            self.commit(step, [rev_rel, rev_rel + ".raw.txt", rev_rel + ".prompt.txt"], f"K23 overnight {step}: Astra review of {scene} {sha[:16]} (not accepted)")
            raise Stop(step, f"Astra did not give the unconditional four-line acceptance of freeze_{scene} {sha[:16]}: {why} (see {rev_rel})", 8)
        self.cmd(step, f"freeze {scene} --presign", [sys.executable, fa, "--scene", scene, "--presign", "--astra-review", rev_rel], stream=False)
        self.cmd(step, f"freeze {scene} --check", [sys.executable, fa, "--scene", scene, "--check"], stream=False, timeout=3600)
        self.commit(step, evidence(step) + [rev_rel, rev_rel + ".raw.txt", rev_rel + ".prompt.txt"],
                    f"K23 overnight {step}: {scene} agent freeze {sha[:16]} written, Astra ACCEPTED, standing pre-signature recorded")

    def step_tests(self):
        """T2 in the background; S3 + F3 once Kitchen 2 paid planning is complete (K2 polled before every K3 command); T3 only after the
        Kitchen 2 chain ended rc 0 with durable evidence."""
        s = self.current = "T2"
        self.guard(s)
        k2 = {"proc": None, "exit": None, "rc": None, "retried": False}
        k3 = {"state": "pending", "stop": None}
        if self.st.test_done("test-k2-v1")["done"]:
            self.log("[T2] test-k2-v1 already complete (durable chain evidence): chain not started")
            k2["rc"] = 0
        else:
            k2["proc"] = self.start_chain(s, "test-k2-v1", "test")

        def poll():
            if k2["rc"] is None and k2["exit"] is None and k2["proc"] is not None:
                k2["exit"] = k2["proc"].poll()

        def abort_check():                                         # before every K3 command
            poll()
            if k2["exit"] not in (None, 0) or k2["rc"] not in (None, 0):
                raise K2Ended(k2["exit"] if k2["exit"] is not None else k2["rc"])

        last_note = None
        try:
            while True:
                poll()
                if k2["exit"] is not None and k2["rc"] is None:
                    ex = k2["exit"]
                    self.log(f"[T2] chain test-k2-v1 ended rc={ex}")
                    self.record("T2", "chain test-k2-v1", ex, "", "")
                    if ex in WSL_RCS and not k2["retried"]:
                        k2["retried"] = True
                        self.log("[T2] WSL connection signature: one relaunch in 90 s (the chain resumes from its journals)")
                        self.sys.sleep(90)
                        k2["exit"], k2["proc"] = None, self.start_chain(s, "test-k2-v1", "test")
                        continue
                    k2["rc"] = ex
                if k2["rc"] not in (None, 0):
                    break                                          # a terminal Kitchen 2 failure first: no (further) K3 work
                if k2["rc"] is None:
                    self.lock_stall(s, "test-k2-v1")
                if k3["state"] == "pending" and self.st.k2_plan_complete():
                    self.abort_check = abort_check
                    try:
                        self.step_size("S3", "after-fp15")
                        self.step_freeze("F3", "FloorPlan17")
                        k3["state"] = "done"
                    except K2Ended:
                        self.log("[T2] the Kitchen 2 chain exited non-zero during the K3 work: K3 work stopped before its next command")
                    except Stop as e:
                        k3.update(state="stopped", stop=e)
                        self.log(f"[{e.step}] K3 SIDE STOPPED: {e.why} -- the Kitchen 2 chain keeps running; the runner stops after it ends")
                    except Exception as e:                         # noqa: BLE001  (an internal error never abandons the K2 chain)
                        k3.update(state="stopped", stop=Stop(self.current or "F3", f"internal error {type(e).__name__}: {e}", 1))
                        self.log(f"[{self.current}] K3 SIDE INTERNAL ERROR {type(e).__name__}: {e} -- waiting for the Kitchen 2 chain")
                    finally:
                        self.abort_check = None
                        self.current = "T2"
                    continue
                if k2["rc"] == 0 and (k3["state"] != "pending" or not self.st.k2_plan_complete()):
                    break
                if last_note is None or self.sys.now() - last_note >= 1800:
                    last_note = self.sys.now()
                    self.log(f"[T2] chain test-k2-v1 {'running' if k2['rc'] is None else 'ended rc ' + str(k2['rc'])}; Kitchen 2 paid "
                             f"planning {'complete' if self.st.k2_plan_complete() else 'in progress'}; K3 work {k3['state']}")
                    self.status_file(k3=k3["state"], k3_stop=str(k3["stop"]) if k3["stop"] else None)
                self.sys.sleep(120 if k2["rc"] is None else 1)
        except Stop:
            raise                                                  # a lock stall: the chain is left as it is
        except Exception as e:                                     # noqa: BLE001  an internal error in the loop: drain K2, then stop
            self.log(f"[T2] INTERNAL ERROR {type(e).__name__}: {e} -- waiting for the Kitchen 2 chain to end (never killed)")
            last_note = None
            while k2["proc"] is not None and k2["rc"] is None and k2["proc"].poll() is None:
                self.lock_stall("T2", "test-k2-v1")                # a queued job past the lock limit: stop (chain left as it is)
                if last_note is None or self.sys.now() - last_note >= 1800:
                    last_note = self.sys.now()
                    self.log("[T2] draining: chain test-k2-v1 still running after an internal error")
                    self.status_file(draining=True, error=f"{type(e).__name__}: {e}")
                self.sys.sleep(300)
            k3.update(state="stopped", stop=Stop("T2", f"internal error {type(e).__name__}: {e}", 1))
            k2["rc"] = k2["proc"].poll() if k2["proc"] is not None and k2["rc"] is None else k2["rc"]
        if k2["rc"] != 0:
            extra = f"; the K3 work also stopped: {k3['stop'].why}" if k3["stop"] else ""
            raise Stop("T2", f"chain test-k2-v1 stopped (rc {k2['rc']}); see sim/final_test/k23_run/test-k2-v1-chain.log{extra}", 4)
        td = self.st.test_done("test-k2-v1")
        if not td["done"]:
            raise Stop("T2", "chain test-k2-v1 rc 0 but no durable completion evidence: " + "; ".join(td["why"][:4]), 4)
        self.commit("T2", evidence("T2"), "K23 overnight T2: test-k2-v1 chain complete (CHAIN DONE, phases complete, score sealed)")
        if k3["stop"] is not None:
            raise k3["stop"]
        if k3["state"] != "done":
            raise Stop("T2", "test-k2-v1 complete but its full plan_first_complete event is missing (K3 size decision not taken)", 7)
        s = self.current = "T3"
        self.guard(s)
        if self.st.test_done("test-k3-v1")["done"]:
            self.log("[T3] test-k3-v1 already complete (durable chain evidence): skipped (its evidence is still committed)")
            self.commit(s, evidence(s), "K23 overnight T3: test-k3-v1 complete (evidence)")
            return
        rc = self.supervise(s, "test-k3-v1", "test")
        td = self.st.test_done("test-k3-v1")
        if rc != 0 or not td["done"]:
            raise Stop(s, f"chain test-k3-v1 stopped (rc {rc}); {'; '.join(td['why'][:3])}; see sim/final_test/k23_run/test-k3-v1-chain.log", 4)
        self.commit(s, evidence(s), "K23 overnight T3: test-k3-v1 chain complete (CHAIN DONE, phases complete, score sealed)")

    def run(self) -> int:
        os.makedirs(self.out, exist_ok=True)
        for f in ("STOPPED.json", "DONE.json"):
            p = os.path.join(self.out, f)
            if self.sys.exists(p):
                os.replace(p, p + f".{time.strftime('%Y%m%dT%H%M%S')}.prev")
        self.log(f"OVERNIGHT START (pid {os.getpid()}, root {self.root}, coco_resume {self.coco_resume})")
        try:
            try:
                self.preflight()
                self.step_wait()
                self.step_coco()
                self.step_ollama()
                self.step_context_resume("R3c", "rehearsal-k3-v1")
                self.step_practice("practice-k2-v1")
                self.step_rehearsal("rehearsal-k2-v1")
                self.step_size("S2", "after-k1")
                self.step_freeze("F2", "FloorPlan15")
                self.step_tests()
            except Stop:
                raise
            except Exception as e:                                 # noqa: BLE001  an internal error is a stop, never a silent exit
                raise Stop(self.current or "?", f"internal error {type(e).__name__}: {e}", 1)
        except Stop as e:
            self.log(f"STOPPED at {e.step}: {e.why} (rc {e.rc}) -- everything is left as it is for the morning")
            with open(os.path.join(self.out, "STOPPED.json"), "w", encoding="utf-8") as f:
                json.dump({"step": e.step, "why": e.why, "rc": e.rc, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=1)
            self.status_file(stopped=True, why=e.why)
            try:
                self.commit(e.step, [], f"K23 overnight: STOPPED at {e.step} ({e.why[:120]})")
            except Stop as e2:
                self.log(f"(the stop log was not committed: {e2.why})")
            return e.rc
        self.log("OVERNIGHT DONE: K2 practice, rehearsal, size decision, freeze, test; K3 size decision, freeze, test")
        with open(os.path.join(self.out, "DONE.json"), "w", encoding="utf-8") as f:
            json.dump({"ts": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
        self.status_file(done=True)
        self.commit("END", [], "K23 overnight: DONE (Kitchen 2 + Kitchen 3 test chains complete)")
        return 0


# ================================================================================================ dry run
def dry_run(root: str, sysx, st, *, branch="master") -> int:
    """Every precondition now, then the plan with each step's current state. Nothing is started, written or committed."""
    bad, lines = [], []

    def chk(name, ok, info="", blocking=True):
        lines.append(f"  [{'ok' if ok else ('FAIL' if blocking else 'note')}] {name}{(': ' + info) if info else ''}")
        if not ok and blocking:
            bad.append(name)

    print(f"K23 OVERNIGHT dry run ({time.strftime('%Y-%m-%d %H:%M:%S')}), root {root}\nPreconditions:")
    chk("python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0])
    rc, head = sysx.sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout=60)
    chk(f"git HEAD is {branch}", rc == 0 and head.strip() == branch, head.strip())
    rc, gd = sysx.sh(["git", "rev-parse", "--git-dir"], timeout=60)
    gdir = gd.strip() if os.path.isabs(gd.strip()) else os.path.join(root, gd.strip())
    chk("no merge / rebase / cherry-pick in progress",
        not any(os.path.exists(os.path.join(gdir, x)) for x in ("MERGE_HEAD", "rebase-merge", "rebase-apply", "CHERRY_PICK_HEAD")))
    for w in st.integrity() or ["(none)"]:
        chk("integrity (preregistration 247ef23a, treatment pins)", w == "(none)", "" if w == "(none)" else w)
    miss = [t for t in TOOLS if not os.path.exists(os.path.join(root, *t.split("/")))]
    chk("tools present", not miss, ", ".join(miss))
    chk("Git Bash", os.path.exists(GIT_BASH), GIT_BASH)
    chk("Ollama executable", os.path.exists(OLLAMA_EXE), OLLAMA_EXE)
    chk("codex on PATH (Astra reviews)", bool(sysx.which("codex")), str(sysx.which("codex")))
    rc, out = sysx.sh(["wsl", "-d", DISTRO, "--", "true"], timeout=120)
    chk(f"WSL {DISTRO} answers", rc == 0, out.strip()[:120])
    g = sysx.gpu()
    chk("nvidia-smi readable", g.get("mib") is not None, json.dumps(g))
    tags = sysx.http_tags()
    chk("Ollama serving Gemma now", tags is not None and GEMMA in tags, "started by the runner at step O" if not tags else "", blocking=False)
    other = None
    try:
        sj = json.load(open(os.path.join(root, *OUT.split("/"), "status.json"), encoding="utf-8"))
        other = sj.get("pid") if sysx.pid_alive(sj.get("pid") or 0) and not (sj.get("done") or sj.get("stopped")) else None
    except (OSError, ValueError):
        pass
    chk("no other overnight runner alive", other is None, f"pid {other}" if other else "")
    sw = strict_wrapper_problems(root)
    chk("strict wrappers equal their rendered originals (chain / launcher / run / inner)", not sw, "; ".join(sw))
    cw = capture_review_problems(root)
    chk("K23 capture artefact gate: reviewed k23_capture_freeze.py + Astra K23-CAPTURE-FREEZE-r1 ACCEPTED", not cw, "; ".join(cw))
    rc, out = sysx.sh([sys.executable, CAPTURE_FREEZE, "--check", "--namespace", "rehearsal-k2-v1"], timeout=900)
    chk("K23 capture artefact step passes on the re-pin view (rehearsal-k2-v1)", rc == 0, "" if rc == 0 else out.strip()[-200:])
    print("\n".join(lines))
    print("\nSteps (current state; each is skipped when already done, its evidence still committed):")
    reh3, co, busy = st.rehearsal("rehearsal-k3-v1"), st.coco(), st.k3_busy()
    g3 = reh3["giso"]
    print(f"  W   wait: rehearsal-k3-v1 {'COMPLETE' if reh3['done'] else ('READY for the context stage (G-iso waived)' if reh3['context_only'] else 'not complete: ' + '; '.join(reh3['why'][:4]))}; "
          f"G-iso {g3['status']}{' WAIVED' if g3['waived'] else (' -> STOP for Arash (no valid waiver)' if reh3['giso_block'] else '')}; K3 processes alive "
          f"{len(busy) if busy is not None else '?'}; COCO {co['verdict']} (last {co['last']} rc {co['rc']} at {co['last_ts']}, log age "
          f"{co['age_s']} s, weights {'present' if co['weights'] else 'not yet'})")
    cf = st.coco_freeze()
    print(f"  C   COCO append: {'DONE' if not cf['problems'] else 'pending (coco_pending ' + str(cf['pending']) + ', listed ' + str(cf['listed']) + ')'}"
          f"\n        wsl --cd <repo> -- {WSL_PY} sim/kitchen_v3/k23/k23_freeze.py --scene FloorPlan15 --add-coco; k23_coco_meta.py --write; "
          "k23_coco_meta.py")
    print("  O   no WSL training process; GPU < 2,000 MiB; Ollama serve (OLLAMA_HOST=0.0.0.0 unless set) -> " + GEMMA)
    print("  R3c only if rehearsal-k3-v1 is complete up to its context fit with a waived G-iso: k23_run --phase context --namespace rehearsal-k3-v1")
    for sc in ("FloorPlan15", "FloorPlan17"):
        k = KIT[sc]
        pr, rh, td = st.practice(k["practice"]), st.rehearsal(k["rehearsal"]), st.test_done(k["test"])
        fz = st.freeze(sc)
        print(f"  K{k['k']} state: {k['practice']} {'DONE' if pr['done'] else 'attempts ' + str(pr['attempts'])}; {k['rehearsal']} "
              f"{'DONE' if rh['done'] else 'not complete'}; freeze_{sc} {'signed' if fz.get('presigned') else ('written ' + str(fz.get('sha'))[:16] if fz.get('exists') else 'none')}; "
              f"{k['test']} {'COMPLETE' if td['done'] else 'not run / not complete'}")
    print(f"  P2  python {PRACTICE_STRICT} practice-k2-v1 {LOCK_WAIT_S} (one try, no restart); k23_practice.py --namespace practice-k2-v1 --status")
    print(f"  R2  [capture gate, Ollama, no training, sim lock free] bash {CHAIN} rehearsal-k2-v1 (supervised; G-iso not a pass -> STOP "
          "unless waived; waived -> context stage only)")
    print(f"  S2  k23_size_decision.py --checkpoint after-k1 [dry run] -> --write -> --check   (recorded: {st.size_checkpoints() or 'none'})")
    print("  F2  k23_freeze_agent.py --scene FloorPlan15 --dry-run -> --write -> Astra (" + ASTRA_CALL + ", prompt with the self-sha) -> only "
          "on the four-line answer RULING: ACCEPTED / KITCHEN: FloorPlan15 / FREEZE-SHA16: <sha16> / CONDITIONS: none: --presign --astra-review "
          "<file> -> --check")
    print("  T2  [capture gate, Ollama, no training, sim lock free] ENERGY=on bash chain_k23_strict.sh test-k2-v1 (background, supervised); when its full "
          "plan_first_complete event exists and the chain has not failed (polled before every K3 command):")
    print("  S3    k23_size_decision.py --checkpoint after-fp15 [dry run] -> --write -> --check")
    print("  F3    k23_freeze_agent.py --scene FloorPlan17 (as F2)")
    print("  T3  after test-k2-v1 ended rc 0 with durable evidence: ENERGY=on bash chain_k23_strict.sh test-k3-v1 (supervised)")
    print("  git: add + commit -- <evidence> after each step; then preregistration 247ef23a + pin_problems() == [] re-checked")
    print(f"\nDRY RUN {'OK: nothing blocks the start' if not bad else 'BLOCKED: ' + ', '.join(bad)} (nothing was run)")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="K23 overnight runner (strict; stops at the first failure)")
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--dry-run", action="store_true")
    m.add_argument("--run", action="store_true")
    ap.add_argument("--no-coco-resume", action="store_true", help="stop instead of resuming a paused / WSL-killed COCO run")
    ap.add_argument("--max-wait-hours", type=float, default=36.0)
    ap.add_argument("--root", default=None, help="dry run only: inspect another checkout (read-only); --run always uses this checkout")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.root and a.run:
        print("REFUSED: --root is for --dry-run only (the runner runs in its own checkout)")
        return 2
    root = os.path.abspath(a.root) if a.root else ROOT
    sysx = Sys(root, os.path.join(root, *OUT.split("/"), "overnight.log"))
    st = State(root, sysx)
    if a.dry_run:
        return dry_run(root, sysx, st)
    os.makedirs(os.path.join(root, *OUT.split("/")), exist_ok=True)
    try:
        sj = json.load(open(os.path.join(root, *OUT.split("/"), "status.json"), encoding="utf-8"))
        if sysx.pid_alive(sj.get("pid") or 0) and sj.get("pid") != os.getpid() and not (sj.get("done") or sj.get("stopped")):
            print(f"REFUSED: another overnight runner is alive (pid {sj.get('pid')})")
            return 2
    except (OSError, ValueError):
        pass
    r = Runner(root, sysx, st, coco_resume=not a.no_coco_resume, max_wait_h=a.max_wait_hours)
    r.status_file(started=True)
    return r.run()


if __name__ == "__main__":
    sys.exit(main())
