"""K23 Kitchen 1 priority check (A28 15.9 condition 2; Astra K23-PREP r5 6). Read-only. Windows python.

Called by capture_k23.ps1 before EVERY batch (pinned in the authorization record): may a FloorPlan15 batch start now?
Prints ONE json line {"free": bool, "why": [...]}; exit 0 = free, 1 = busy (yield: stop at this batch boundary, poll again
in 5 minutes), 2 = could not decide. FAIL CLOSED: any query that fails, times out, exits non-zero or does not print its
completion marker makes the answer busy.

Busy when ANY of:
  1. the ledger: the newest heading of a Kitchen 1 simulator stage (ACCEPTANCE-V2*, FREEZE / FREEZE-* except
     *PLUMBING*, TEST-V1*, LOCKED*, K1-TEST*, FINAL-TEST-RUN*, VALIDATE-*) is active (STARTED / resumed / step /
     running / about to start) and not DONE / STOPPED;
  2. the ledger: an ARASH / MAIN SESSION heading timed within the last 60 minutes that says acceptance / the freeze /
     the (locked / Kitchen 1) test starts, runs or goes;
  3. WSL: /home/arash/sim.lock is held, any process waits on it (`flock ... sim.lock`), or any simulator / thor-env python
     job runs (a K23 batch of this stage never runs between batches);
  4. Windows: a python process RUNS acc_runner.py / freeze_ft2.py / freeze.py / run_matrix* / execute.py / webapp.py /
     train_*.py, or a Kitchen 1 LOCAL planner runs (a python command line naming Gemma / `--stage gemma` / Ollama or
     its port), or a PowerShell supervisor capture_all / amb_capture / train_* runs, or a llama-server that is not
     Ollama's own runner runs (another project's model server = an unknown GPU user);
  5. GPU (Windows nvidia-smi): a compute application that is python / thor / Unity, or is not a known desktop app,
     holds the GPU (unknown GPU apps = busy); the GPU is working (median utilization of 3 samples > UTIL_BUSY_PCT,
     e.g. Gemma generating); or free VRAM is below what the capture needs (VRAM rule below);
  6. Ollama (K2-CAPTURE, 2026-10-14): Ollama RUNNING, even with a model LOADED, is NOT busy by itself (Kitchen 1 keeps
     Gemma loaded all night). It is busy only through rule 4 (a Kitchen 1 planner process), rule 5 (the GPU is
     working = a generation) or the VRAM rule; its /api/ps must answer (a running Ollama that does not = busy).
VRAM rule: free VRAM >= capture_mib (k23_vram.json, measured at the FloorPlan15 probe; CAPTURE_MIB_DEFAULT before)
     + MARGIN_MIB, + GEMMA_RESERVE_MIB while Kitchen 1's Gemma is NOT loaded (Kitchen 1 must still be able to load it
     while a capture batch runs).
Remote planners (Batch API / Codex calls: `--stage opus|codex`) use neither the GPU nor the simulator: not busy.
The batch itself also yields after taking the lock when another process waits on it (capture_inner_k23.sh, rc 24).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LEDGER = os.path.join(REPO, "docs", "final-test", "progress.md")
K1_STAGE = re.compile(r"STAGE (ACCEPTANCE-V2[\w-]*|FREEZE(?:-[\w-]+)?|TEST-V1[\w-]*|LOCKED[\w-]*|K1-TEST[\w-]*|"
                      r"FINAL-TEST-RUN[\w-]*|VALIDATE-[\w-]+)")
ACTIVE = re.compile(r"STARTED|resumed|\bstep\b|running|about to start", re.I)
ENDED = re.compile(r"\bDONE\b|STOPPED", re.I)
DECISION = re.compile(r"ARASH|MAIN SESSION")
STARTS = re.compile(r"(acceptance|freeze|locked test|kitchen 1 test|the test)\b.*\b(start|starts|run|runs|go|now)\b", re.I)
HEAD_TIME = re.compile(r"^## (\d{4}-\d{2}-\d{2}) ~?(\d{1,2}):(\d{2})")
WIN_PY = re.compile(r"^(acc_runner|freeze_ft2|freeze|run_matrix\w*|execute|webapp|train_\w+)\.py$")
WIN_PS = re.compile(r"capture_all\.ps1|amb_capture\.ps1|balance_train\.ps1|train_only\.ps1|amb_train\.ps1", re.I)  # r8b (Astra r8 6)
# r8b (K2-CAPTURE 2026-10-14): Ollama >= 0.12 runs its models as lib/ollama/llama-server.exe under its OWN install
# folder; that runner is Ollama (judged by the Ollama + VRAM + utilization rules), any other llama-server stays busy
OLLAMA_RUNNER = re.compile(r"^\s*(?:\"[^\"]*[\\/]Ollama[\\/]lib[\\/]ollama[\\/]llama-server(?:\.exe)?\"|[^\"\s]*[\\/]Ollama[\\/]lib[\\/]ollama[\\/]llama-server(?:\.exe)?)(?:\s|$)", re.I)  # the EXECUTABLE (first token), never an argument (Astra r8b B)
WIN_PS_PATH = re.compile(r"(?:^|[\\/\s\"'&])(capture_all|amb_capture|balance_train|train_only|amb_train)\.ps1\b", re.I)
GPU_APPS = re.compile(r"python|thor|unity", re.I)                         # always busy on the GPU
OLLAMA_APPS = re.compile(r"^(ollama|ollama_llama_server)(\.exe)?$", re.I)  # judged by the Ollama + VRAM rules
DESKTOP_APPS = {n.lower() for n in (
    "dwm.exe", "explorer.exe", "chrome.exe", "msedge.exe", "msedgewebview2.exe", "firefox.exe", "Code.exe",
    "GameBar.exe", "SystemSettings.exe", "TextInputHost.exe", "StartMenuExperienceHost.exe", "ShellExperienceHost.exe",
    "SearchHost.exe", "SearchApp.exe", "Widgets.exe", "WidgetBoard.exe", "PhoneExperienceHost.exe",
    "ApplicationFrameHost.exe", "LockApp.exe", "LogonUI.exe", "WindowsTerminal.exe", "OpenConsole.exe", "Taskmgr.exe", "ctfmon.exe",
    "Claude.exe", "ScreenClippingHost.exe", "SnippingTool.exe", "CrossDeviceResume.exe", "Photos.exe",
    "NVIDIA Overlay.exe", "NVIDIA Share.exe", "nvcontainer.exe", "msrdc.exe", "Teams.exe", "ms-teams.exe",
    "Discord.exe", "Telegram.exe", "WhatsApp.exe", "Notepad.exe", "AcroCEF.exe", "Acrobat.exe")}
K1_PLANNER = re.compile(r"--stage\s+gemma\b|gemma|:11434\b|\bollama\b", re.I)   # a Kitchen 1 LOCAL (GPU) planner
UTIL_BUSY_PCT = 40.0         # idle desktop on this laptop: 2-16 %; Gemma generating: 53-98 % (dress-fa-v1 gpu_gemma)
MARGIN_MIB = 1024            # kept free on top of the capture's measured need
GEMMA_RESERVE_MIB = 3584     # Kitchen 1 Gemma (gemma4:e2b-it-q4_K_M) resident: ~3.1 GB (3421 MiB GPU total, dress-fa-v1)
K1_GEMMA = "gemma4:e2b"
CAPTURE_MIB_DEFAULT = 2560   # before the probe measures it (k23_vram.json)
VRAM_FILE = os.path.join(HERE, "k23_vram.json")
OLLAMA_PS = "http://127.0.0.1:11434/api/ps"


def ledger_busy(text: str, now: dt.datetime) -> list:
    heads = [l.rstrip("\n") for l in text.splitlines() if l.startswith("## ")]
    last, out = {}, []
    for h in heads:
        m = K1_STAGE.search(h)
        if m and "PLUMBING" not in m.group(1):
            last[m.group(1)] = h
        t = HEAD_TIME.match(h)
        if t and DECISION.search(h) and STARTS.search(h):
            when = dt.datetime.strptime(f"{t.group(1)} {int(t.group(2)):02d}:{t.group(3)}", "%Y-%m-%d %H:%M")
            if 0 <= (now - when).total_seconds() <= 3600:
                out.append(f"ledger: a decision of the last hour says a Kitchen 1 stage starts: {h[:150]}")
    for st, h in last.items():
        if ACTIVE.search(h) and not ENDED.search(h):
            out.append(f"ledger: Kitchen 1 stage {st} is active: {h[:150]}")
    return out


def run(cmd: list, timeout: int = 60, stdin: str | None = None):
    """(stdout, problem) -- problem is a string when the query failed (= busy)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, input=stdin)
    except Exception as exc:
        return "", f"query {cmd[0]} failed: {exc!r}"
    if r.returncode != 0:
        return r.stdout, f"query {cmd[0]} exited {r.returncode}: {r.stderr.strip()[:120]}"
    return r.stdout, None


def wsl_busy() -> list:
    # r6 6: every query's own status is checked (pgrep: 0 = found, 1 = none, anything else = failure)
    cmd = ("ok=1; if flock -n /home/arash/sim.lock true; then echo LOCK_FREE; else echo LOCK_HELD; fi; "
           "w=$(pgrep -af 'flock.*sim\\.lock'); r=$?; [ $r -gt 1 ] && ok=0; "
           "echo \"$w\" | grep -v -e pgrep -e 'flock -n /home/arash/sim.lock true' | grep . | sed 's/^/WAITER /'; "
           "j=$(pgrep -af 'thor-Linux64|thor-env/bin/python'); r=$?; [ $r -gt 1 ] && ok=0; "
           "echo \"$j\" | grep -v -e pgrep | grep . | sed 's/^/JOB /'; "
           "[ $ok = 1 ] && echo CHECK_DONE || echo CHECK_FAILED")
    # the script goes through stdin: wsl.exe's own argument handling mangles quotes, `|` and `$?`
    out, prob = run(["wsl", "-d", "Ubuntu-22.04", "--", "bash", "-s"], stdin=cmd + chr(10))
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    res = [prob] if prob else []
    if "CHECK_DONE" not in lines or not ({"LOCK_FREE", "LOCK_HELD"} & set(lines)):
        res.append("wsl: the check did not complete (fail closed)")
    res += [f"wsl: {l[:160]}" for l in lines if l == "LOCK_HELD" or l.startswith(("WAITER ", "JOB "))]
    return res


def windows_busy() -> list:
    # r6 6: a CIM / process error stops the script before CHECK_DONE (fail closed)
    ps = ("$ErrorActionPreference = 'Stop'; try { $all = @(Get-CimInstance Win32_Process); "
          "$all | Where-Object { $_.Name -match '^(python|pythonw|py|powershell|pwsh|ollama|"
          "ollama_llama_server|llama-server)\\.exe$' } | ForEach-Object { $_.Name + '|' + ($_.CommandLine -replace "
          "'[\\r\\n]+', ' ') }; $all | ForEach-Object { 'PROC|' + $_.ProcessId + '|' + $_.Name }; 'CHECK_DONE' } "
          "catch { 'CHECK_FAILED ' + $_.Exception.Message }")
    out, prob = run(["powershell", "-NoProfile", "-Command", ps])
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    res = [prob] if prob else []
    if "CHECK_DONE" not in lines:
        res.append("windows: the process query did not complete (fail closed)")
    me = os.path.basename(__file__)
    del K1_TEST_PLANNERS[:]
    PROCS_SEEN.clear()
    del OLLAMA_SEEN[:]
    for l in lines:
        if l.startswith("PROC|"):
            parts = l.split("|", 2)
            if len(parts) == 3 and parts[1].isdigit():
                PROCS_SEEN[int(parts[1])] = parts[2]
            continue
        if "|" not in l:
            continue
        name, cl = l.split("|", 1)
        n = name.lower()
        if n.startswith("ollama"):
            OLLAMA_SEEN.append(name)                              # judged by ollama_busy + the VRAM rule (K2-CAPTURE)
        elif n.startswith("llama") and OLLAMA_RUNNER.match(cl):
            OLLAMA_SEEN.append(name)                              # Ollama's own runner (r8b)
        elif n.startswith("llama"):
            res.append(f"windows: {name} runs (a model server that is not Ollama's: unknown GPU user)")
        elif n.startswith("py"):
            if me in cl:
                continue
            m = re.search(r"[\s\"']([^\s\"']+\.py)\b", " " + cl)        # quoted paths too (r6 6)
            if m and WIN_PY.match(os.path.basename(m.group(1))):
                res.append(f"windows: {cl[:160]}")
            elif K1_TEST_PLANFIRST.search(" " + cl) and k1_test_free_done():
                K1_TEST_PLANNERS.append(cl[:160])                 # K2-K1TEST: remote stages only (rule 7), not busy
            elif K1_PLANNER.search(cl):
                res.append(f"windows: a local (GPU) planner runs: {cl[:160]}")
        elif WIN_PS.search(cl) and "Get-CimInstance" not in cl:
            # r7 (Astra r7 6): any supervisor script NAME as a path component (quoted, with spaces, -File or
            # -Command) = busy (fail closed); only this check's own query is exempt
            if WIN_PS_PATH.search(cl):
                res.append(f"windows: {cl[:160]}")
    return res


PROCS_SEEN: dict = {}
OLLAMA_SEEN: list = []
K1_TEST_PLANNERS: list = []

# ======================================================================================================================
# K2-K1TEST (2026-10-16, Arash: "at least collect the data for kitchen 2 ... in the meantime"): the Kitchen 1 LOCKED
# TEST (test-v1, sim/final_test/test_run/chain_test.sh) is running. Rule 7: until its execute stage is complete, a
# FloorPlan15 batch may start ONLY while a test-v1 planfirst.py process is alive AND its registered "free" stage (the
# only local-GPU planning: Gemma) is complete since that process started (events.jsonl: plan_first_stage free code 0
# after the newest plan_first_start). Such a planner runs only remote Batch-API / Codex stages, so it is not busy under
# rule 4. No planner alive (chain stopped, between stages, before execute, a restart pending) = busy: planfirst refuses
# to (re)start while an AI2-THOR process lives, and execute needs the sim lock. Fail closed on any unreadable file.
# Read-only; nothing of Kitchen 1 is written or changed.
# ======================================================================================================================
K1_TEST_PLANFIRST = re.compile(r"[\s\"'/\\]planfirst\.py[\"']?\s.*--namespace[\s=]+test-v1(?:\s|$)")
K1_TEST_EVENTS = os.path.join(REPO, "docs", "final-test", "matrix", "test-v1", "events.jsonl")
K1_TEST_CHAINLOG = os.path.join(REPO, "sim", "final_test", "test_run", "test-chain.log")
K1_TEST_EXEC_DONE = re.compile(r"STAGE execute rc=0|execute already complete|CHAIN DONE")


def k1_test_free_done(path: str | None = None) -> bool:
    """True when the newest plan_first_start in test-v1 is followed by plan_first_stage free code 0 (fail closed)."""
    try:
        done = None
        for l in open(path or K1_TEST_EVENTS, encoding="utf-8"):
            if '"plan_first_' not in l:
                continue
            d = json.loads(l)
            if d.get("event") == "plan_first_start":
                done = False
            elif d.get("event") == "plan_first_stage" and d.get("stage") == "free" and done is False:
                done = d.get("code") == 0
        return done is True
    except Exception:
        return False


def k1_test_busy(chainlog: str | None = None) -> list:
    """Rule 7. Run after windows_busy (which fills K1_TEST_PLANNERS)."""
    p = chainlog or K1_TEST_CHAINLOG
    if not os.path.exists(p):
        return []                                         # no locked test launched on this machine
    try:
        text = open(p, encoding="utf-8", errors="replace").read()
    except Exception as exc:
        return [f"k1 test: chain log unreadable ({type(exc).__name__}; fail closed)"]
    if K1_TEST_EXEC_DONE.search(text):
        return []                                         # execute complete: Kitchen 1 no longer needs the simulator
    if K1_TEST_PLANNERS:
        return []
    return ["k1 test: the Kitchen 1 locked test (test-v1) has not finished execute and no test-v1 planfirst.py in its "
            "remote stages is running (stopped / restarting / about to execute = Kitchen 1 may need the simulator)"]
LOADED: dict = {"models": None}
VRAM_SEEN: dict = {}


def capture_need() -> tuple:
    """(MiB the capture needs, where the number comes from). k23_vram.json = the FloorPlan15 probe measurement."""
    try:
        d = json.load(open(VRAM_FILE, encoding="utf-8"))
        v = d["capture_mib"]
        if isinstance(v, int) and not isinstance(v, bool) and 256 <= v <= 8192:
            return v, f"k23_vram.json ({d.get('source', '?')})"
        return CAPTURE_MIB_DEFAULT, "default (k23_vram.json value out of range)"
    except FileNotFoundError:
        return CAPTURE_MIB_DEFAULT, "default (not measured yet)"
    except Exception as exc:
        return CAPTURE_MIB_DEFAULT, f"default (k23_vram.json unreadable: {type(exc).__name__})"


def ollama_busy(fetch=None) -> list:
    """K2-CAPTURE: a running Ollama is not busy by itself; its /api/ps must answer (else busy). Records the loaded
    models for the VRAM rule. Run after windows_busy (which lists the Ollama processes)."""
    LOADED["models"] = None
    if not OLLAMA_SEEN:
        LOADED["models"] = []
        return []
    try:
        if fetch is None:
            import urllib.request
            with urllib.request.urlopen(OLLAMA_PS, timeout=10) as r:
                d = json.loads(r.read().decode("utf-8"))
        else:
            d = fetch()
        ms = d.get("models")
        if not isinstance(ms, list):
            return ["ollama: /api/ps answer unreadable (fail closed)"]
        LOADED["models"] = [str(m.get("name") or m.get("model") or "") for m in ms if isinstance(m, dict)]
        return []
    except Exception as exc:
        return [f"ollama: runs but /api/ps did not answer ({type(exc).__name__}; fail closed)"]


def vram_util_busy(samples: int = 3, pause: float = 1.0) -> list:
    """GPU working (median utilization) or not enough free VRAM for the capture (+ margin, + the Gemma reserve while
    Kitchen 1's Gemma is not loaded). Run after ollama_busy."""
    import time
    utils, frees = [], []
    for i in range(samples):
        out, prob = run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.free", "--format=csv,noheader,nounits"])
        if prob:
            return [f"gpu: {prob}"]
        try:
            u, f = (float(x) for x in out.strip().splitlines()[0].split(","))
        except Exception:
            return [f"gpu: utilization / free memory unreadable ({out.strip()[:60]!r}; fail closed)"]
        utils.append(u)
        frees.append(f)
        if i + 1 < samples:
            time.sleep(pause)
    res = []
    med = sorted(utils)[len(utils) // 2]
    if med > UTIL_BUSY_PCT:
        res.append(f"gpu: working (median utilization {med:.0f} % > {UTIL_BUSY_PCT:.0f} %: e.g. Gemma generating)")
    need, src = capture_need()
    loaded = LOADED["models"]
    if loaded is None:
        res.append("gpu: Ollama's loaded models unknown (fail closed)")
        loaded = []
    reserve = 0 if any(m.startswith(K1_GEMMA) for m in loaded) else GEMMA_RESERVE_MIB
    want = need + MARGIN_MIB + reserve
    if min(frees) < want:
        res.append(f"gpu: free VRAM {min(frees):.0f} MiB < {want} MiB (capture {need} [{src}] + margin {MARGIN_MIB}"
                   f" + Gemma reserve {reserve})")
    VRAM_SEEN.clear()
    VRAM_SEEN.update(free_mib=min(frees), need_mib=want, capture_mib=need, capture_src=src, util_median=med,
                     loaded=loaded)
    return res


def gpu_busy() -> list:
    """GPU compute apps: named python / ollama / llama / thor / unity = busy. An app whose name nvidia-smi cannot show
    ("[Insufficient Permissions]") is identified by its pid from the Windows process list (windows_busy runs first);
    a pid that cannot be identified = busy (fail closed, r6 6)."""
    out, prob = run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"])
    if prob:
        return [f"gpu: {prob}"]
    res = []
    for l in out.splitlines():
        if not l.strip():
            continue
        pid_s, _, name = l.partition(",")
        name = name.strip()
        if name.startswith("[") and pid_s.strip().isdigit():
            name = PROCS_SEEN.get(int(pid_s.strip()), "")
            if not name:
                res.append(f"gpu: unidentified compute app pid {pid_s.strip()} (fail closed)")
                continue
        base = os.path.basename(name.replace("\\", "/"))
        if OLLAMA_APPS.match(base) or OLLAMA_RUNNER.match(name):
            continue                                   # Ollama: the Ollama + VRAM + utilization rules decide
        if GPU_APPS.search(base) or base.lower() not in DESKTOP_APPS:
            res.append(f"gpu: {l.strip()[:160]}" + ("" if GPU_APPS.search(base) else " (unknown GPU app)"))
    return res


# ======================================================================================================================
# K23 CO-RUN (Arash 2026-10-18 ~18:50, "Yes, run both at once"; Astra K23-CORUN-r1): while the record k23_corun.json is
# present, valid and in force, a Kitchen 2/3 TRAINING job (k23_train.py / k23_amb_train.py / k23_state_train.py `train`
# under the thor-env python) does not by itself make a K23 capture batch wait. Exempted ONLY: that job's WSL JOB line(s),
# and -- while such a job runs and NO other WSL job runs -- the GPU lines it causes (the WSL compute app nvidia-smi cannot
# name, the utilization rule) and the VRAM rule re-evaluated with the record's co-run need (measured capture peak + margin,
# no Gemma reserve: no Kitchen 1 stage runs; every Kitchen 1 ledger / Windows / Ollama / test rule is unchanged). The sim
# lock, lock waiters, any other WSL job and every other line stay busy. Record missing / invalid / expired = no exemption.
# ======================================================================================================================
CORUN_FILE = os.path.join(HERE, "k23_corun.json")
K23_TRAIN_JOB = re.compile(r"^wsl: JOB \d+ /home/arash/thor-env/bin/python(?: -u)? sim/kitchen_v3/k23/"
                           r"(?:k23_train|k23_amb_train|k23_state_train)\.py train\b")


CORUN_CAPTURE_MIB, CORUN_MARGIN_MIB = 1024, 512          # the only accepted co-run budget (Astra K23-CORUN-r1 item 1)
CORUN_LAST = dt.datetime(2026, 10, 25, 23, 59)            # standing approvals end; a later expiry is invalid
WSL_GPU_PID = 4                                           # WSL's paravirtualised GPU work = Windows pid 4 'System'


def corun_record(now: dt.datetime) -> dict | None:
    try:
        r = json.load(open(CORUN_FILE, encoding="utf-8"))
    except Exception:
        return None
    try:
        until = dt.datetime.strptime(r["until"], "%Y-%m-%d %H:%M")
        cm, mm = r.get("capture_mib"), r.get("margin_mib")
        ok = (r.get("schema") == "k23-corun/1" and r.get("enabled") is True and until <= CORUN_LAST and now <= until
              and type(cm) is int and type(mm) is int and cm == CORUN_CAPTURE_MIB and mm == CORUN_MARGIN_MIB)
    except Exception:
        return None
    return r if ok else None


def _wsl_gpu_line(w: str) -> bool:
    """Positive WSL attribution only: nvidia-smi's '<pid>, [Insufficient Permissions]' line for pid 4 resolved by the
    Windows process list to 'System' (WSL's GPU work). Any other unnamed / unresolved app stays busy."""
    m = re.match(r"^gpu: (\d+), \[Insufficient Permissions\]", w)
    return bool(m) and int(m.group(1)) == WSL_GPU_PID and PROCS_SEEN.get(WSL_GPU_PID, "").lower() == "system"


def corun_filter(why: list, now: dt.datetime) -> tuple:
    """(kept, exempted) under the K23 co-run record; without a valid record nothing is exempted."""
    rec = corun_record(now)
    if rec is None:
        return why, []
    jobs = [w for w in why if w.startswith("wsl: JOB ")]
    train = [w for w in jobs if K23_TRAIN_JOB.match(w)]
    others = [w for w in jobs if not K23_TRAIN_JOB.match(w)]
    kept, ex = [], []
    for w in why:
        if w in train:
            ex.append(w)
        elif train and not others and _wsl_gpu_line(w):
            ex.append(w)
        elif train and not others and w.startswith("gpu: working (median utilization"):
            ex.append(w)
        elif train and not others and w.startswith("gpu: free VRAM") and VRAM_SEEN.get("free_mib", 0) >= rec["capture_mib"] + rec["margin_mib"]:
            ex.append(w + f" -> co-run need {rec['capture_mib']} + {rec['margin_mib']} MiB met")
        else:
            kept.append(w)
    return kept, ex


def main() -> int:
    try:
        text = open(LEDGER, encoding="utf-8").read()
        why = ledger_busy(text, dt.datetime.now()) + wsl_busy() + windows_busy() + k1_test_busy() + gpu_busy() + ollama_busy()
        why += vram_util_busy()
        why, corun_ex = corun_filter(why, dt.datetime.now())
    except Exception as exc:
        print(json.dumps({"free": False, "why": [f"could not decide: {exc!r}"]}))
        return 2
    print(json.dumps({"free": not why, "t": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "why": why,
                      "gpu": VRAM_SEEN, "corun_exempt": corun_ex}))
    return 0 if not why else 1


if __name__ == "__main__":
    raise SystemExit(main())
