"""Offline gate for the STRICT wrapper copies (k23_strict_wrappers.py; Astra K23-OVERNIGHT-r2 items 1-3, 6): every strict file equals its
rendered original (drift is caught), the strict properties hold in the text, and the REAL practice/run_strict.sh, practice/inner_strict.sh
and run_k23_strict.sh are exercised in WSL in a throw-away fake repo with a temporary lock file and a fake hang_guard (no simulator, no GPU,
no Ollama, never the real sim lock): one try, no --steal-lock, a held lock -> exit 73 and the job never runs, the cleanup kills only the
job's own process group, a missing / revoked launch authorization -> exit 74 and nothing runs (also when revoked while waiting). The
behavioural part is MANDATORY (Astra K23-OVERNIGHT-r3 A4): WSL unreachable = failure; --static-only exits 3 and never satisfies the gate."""
import os, re, shutil, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import k23_strict_wrappers as W
ok = fail = 0


def check(n, c):
    global ok, fail
    if c: ok += 1
    else: fail += 1; print("FAIL", n)


def rd(rel):
    return open(os.path.join(HERE, *rel.split("/")), encoding="utf-8").read()


check("every strict file equals its rendered original (k23_strict_wrappers --check)", W.check() == [])
t = tempfile.mkdtemp(prefix="k23strict_")
for rel in ["k23_strict_wrappers.py"] + list(W.EDITS) + [src for src, _ in W.EDITS.values()]:
    os.makedirs(os.path.dirname(os.path.join(t, *rel.split("/"))), exist_ok=True)
    shutil.copyfile(os.path.join(HERE, *rel.split("/")), os.path.join(t, *rel.split("/")))
open(os.path.join(t, "chain_k23.sh"), "a", encoding="utf-8", newline="\n").write("# a later change of the original\n")
check("a changed original is caught (the strict copy must be regenerated + reviewed)", any("chain_k23_strict.sh" in w for w in W.check(t)))
shutil.copyfile(os.path.join(HERE, "chain_k23.sh"), os.path.join(t, "chain_k23.sh"))
open(os.path.join(t, "run_k23_strict.sh"), "a", encoding="utf-8", newline="\n").write("# hand edit\n")
check("a hand-edited strict file is caught", any("run_k23_strict.sh" in w for w in W.check(t)))
shutil.rmtree(t, ignore_errors=True)

ch = rd("chain_k23_strict.sh")
ex = ch[ch.index("execute() {"):ch.index("score() {")]
check("strict chain: an execute session's rc is checked BEFORE its completion event; no-progress sessions stop",
      ex.index("|| return $rc") < ex.index("done_phase execute && return 0") and "without new exec rows" in ex)
gi = ch[ch.index("giso() {"):ch.index("context() {")]
check("strict chain: an existing non-pass G-iso is never re-run", "no re-run" in gi and gi.index("return 4") < gi.index("wsl_job $J-giso"))
check("strict chain: the strict launcher, no hang-guard relaunch, START marked strict", "launch_k23_strict.py" in ch and "launch_k23.py" not in ch
      and "${HANG_RELAUNCH" not in ch and 'CHAIN START $NS ($KIND, strict)' in ch)
rk = rd("run_k23_strict.sh")
code = [ln for ln in rk.splitlines() if not ln.lstrip().startswith("#")]
check("run_k23_strict.sh: bounded flock -w, exit 73 on a timeout (the job does not run), no global pkill, own-group cleanup",
      any('flock -w "$LW"' in ln for ln in code) and "exit 73" in rk and not any('pkill -9 -f "[t]hor-Linux64"' in ln for ln in code)
      and any('pkill -9 -g "$PG"' in ln for ln in code) and any("setsid --wait" in ln for ln in code))
ins = rd("practice/inner_strict.sh")
check("inner_strict.sh: no --steal-lock, own process group, own-group cleanup only", not any("--steal-lock" in ln for ln in ins.splitlines()
      if not ln.lstrip().startswith("#")) and "setsid --wait" in ins and 'pkill -9 -g "$PG"' in ins and "pkill -9 -f '[t]hor-Linux64'" not in ins)
check("launch_k23_strict.py runs run_k23_strict.sh (never run_k23.sh)", '"sim/final_test/k23_run/run_k23_strict.sh"' in rd("launch_k23_strict.py")
      and '"sim/final_test/k23_run/run_k23.sh"' not in rd("launch_k23_strict.py"))

# ---------------------------------------------------------------------------------------------- behaviour of the real wrappers (WSL)
FAKE_HG = r'''import os, subprocess, sys, time
with open(os.environ["FAKE_CALLS"], "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\n")
if os.environ.get("FAKE_SPAWN"):
    subprocess.Popen(["bash", "-c", "exec -a thor-Linux64-fake-own sleep 300"])
    time.sleep(0.5)
sys.exit(int(os.environ.get("FAKE_RC", "0")))
'''
DRIVER = r'''#!/bin/bash
set -u
export PY=/home/arash/thor-env/bin/python FAKE_CALLS=$PWD/calls.txt
[ -x "$PY" ] || export PY=$(command -v python3)
L=/tmp/k23wt_$$
AU=sim/final_test/k23_run/overnight/auth
mkdir -p $AU
tok() { echo x > $AU/$1.token; echo $AU/$1.token; }
: > calls.txt
T=$(tok a)
SIM_LOCK=${L}_a.lock FAKE_RC=3 bash sim/final_test/k23_run/practice/run_strict.sh practice-k2-v1 5 $T >/dev/null 2>&1; echo "A_RC=$?"
echo "A_CALLS=$(wc -l < calls.txt)"; echo "A_STEAL=$(grep -c -- --steal-lock calls.txt)"; echo "A_NS=$(grep -c -- '--namespace practice-k2-v1' calls.txt)"
echo "A_CLAIMED=$( [ -f $T.claimed ] && [ ! -f $T ] && echo 1 || echo 0)"
: > calls.txt
T=$(tok b)
flock ${L}_b.lock sleep 4 & H=$!; sleep 0.5
SIM_LOCK=${L}_b.lock FAKE_RC=0 bash sim/final_test/k23_run/practice/run_strict.sh practice-k2-v1 1 $T >/dev/null 2>&1; echo "B_RC=$?"
echo "B_CALLS=$(wc -l < calls.txt)"; wait $H 2>/dev/null
: > calls.txt
T=$(tok c)
setsid bash -c 'exec -a thor-Linux64-fake-other sleep 300' & sleep 0.3
SIM_LOCK=${L}_c.lock FAKE_SPAWN=1 FAKE_RC=0 bash sim/final_test/k23_run/practice/run_strict.sh practice-k2-v1 5 $T >/dev/null 2>&1; echo "C_RC=$?"
sleep 0.5; echo "C_OWN=$(pgrep -fc thor-Linux64-fake-own)"; echo "C_OTHER=$(pgrep -fc thor-Linux64-fake-other)"; pkill -f thor-Linux64-fake-other
: > calls.txt
flock ${L}_d.lock sleep 4 & H=$!; sleep 0.5
SIM_LOCK=${L}_d.lock LOCK_WAIT=1 FAKE_RC=0 bash sim/final_test/k23_run/run_k23_strict.sh tjob 60 --phase execute --namespace rehearsal-k2-v1 >/dev/null 2>&1; echo "D_RC=$?"
echo "D_CALLS=$(wc -l < calls.txt)"; echo "D_LAST=$(tail -1 sim/final_test/k23_run/logs/tjob/progress.txt)"; wait $H 2>/dev/null
: > calls.txt
setsid bash -c 'exec -a thor-Linux64-fake-other sleep 300' & sleep 0.3
SIM_LOCK=${L}_e.lock LOCK_WAIT=5 FAKE_SPAWN=1 FAKE_RC=4 bash sim/final_test/k23_run/run_k23_strict.sh tjob2 60 --phase execute --namespace rehearsal-k2-v1 >/dev/null 2>&1; echo "E_RC=$?"
sleep 0.5; echo "E_CALLS=$(wc -l < calls.txt)"; echo "E_ARGS=$(head -1 calls.txt)"
echo "E_OWN=$(pgrep -fc thor-Linux64-fake-own)"; echo "E_OTHER=$(pgrep -fc thor-Linux64-fake-other)"; pkill -f thor-Linux64-fake-other
: > calls.txt
SIM_LOCK=${L}_f.lock FAKE_RC=0 bash sim/final_test/k23_run/practice/run_strict.sh practice-k2-v1 5 $AU/never-created.token >/dev/null 2>&1; echo "F_RC=$?"
echo "F_CALLS=$(wc -l < calls.txt)"
: > calls.txt
T=$(tok g)
flock ${L}_g.lock sleep 3 & H=$!; sleep 0.3
( SIM_LOCK=${L}_g.lock FAKE_RC=0 bash sim/final_test/k23_run/practice/run_strict.sh practice-k2-v1 8 $T >/dev/null 2>&1; echo "G_RC=$?" > g_rc.txt ) &
W=$!; sleep 1; mv $T $T.revoked; wait $H 2>/dev/null; wait $W 2>/dev/null
cat g_rc.txt; echo "G_CALLS=$(wc -l < calls.txt)"
echo "DRIVER_DONE=1"
rm -f ${L}_*.lock
'''
STATIC_ONLY = "--static-only" in sys.argv[1:]
rs_ = rd("practice/run_strict.sh")
check("run_strict.sh: the token is checked before queuing and CLAIMED (atomic rename) only inside the lock, right before the try",
      "mv -- \"$0\" \"$0.claimed\"" in rs_ and rs_.index("launch authorization $TOKEN is gone") < rs_.index("flock -w") and "exit 74" in rs_)
if STATIC_ONLY:
    print(f"{ok}/{ok + fail} (static only: the behavioural tests did NOT run -- this mode never satisfies the freeze gate)")
    sys.exit(3)
fake = tempfile.mkdtemp(prefix="k23wrapfake_")
try:
    for rel in ("sim/final_test/matrix/k23_practice.py", "sim/final_test/matrix/k23_run.py"):
        os.makedirs(os.path.dirname(os.path.join(fake, *rel.split("/"))), exist_ok=True)
        open(os.path.join(fake, *rel.split("/")), "w").close()
    open(os.path.join(fake, "sim", "final_test", "matrix", "hang_guard.py"), "w", newline="\n").write(FAKE_HG)
    for rel in ("practice/run_strict.sh", "practice/inner_strict.sh", "run_k23_strict.sh"):
        dst = os.path.join(fake, "sim", "final_test", "k23_run", *rel.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(HERE, *rel.split("/")), dst)
    open(os.path.join(fake, "driver.sh"), "w", newline="\n").write(DRIVER)
    try:
        r = subprocess.run(["wsl", "-d", "Ubuntu-22.04", "--cd", fake, "--", "bash", "driver.sh"], capture_output=True, text=True, timeout=240)
        out = dict(ln.split("=", 1) for ln in r.stdout.splitlines() if "=" in ln and ln.split("=", 1)[0].isupper())
        err = (r.stderr or "")[-200:]
    except (OSError, subprocess.TimeoutExpired) as e:
        out, err = {}, f"{type(e).__name__}: {e}"
    # Astra K23-OVERNIGHT-r3 A4: the behavioural tests are MANDATORY -- an unreachable WSL or an incomplete driver is a failure, never a skip
    check("the behavioural wrapper tests ran to completion in WSL (mandatory)", out.get("DRIVER_DONE") == "1" or print("  driver:", err) or False)
    check("run_strict.sh (real): one try, the try's rc returned (3), no --steal-lock passed, the namespace passed, the token claimed",
          out.get("A_RC") == "3" and out.get("A_CALLS") == "1" and out.get("A_STEAL") == "0" and out.get("A_NS") == "1" and out.get("A_CLAIMED") == "1")
    check("run_strict.sh (real): a held lock -> exit 73 and k23_practice never runs", out.get("B_RC") == "73" and out.get("B_CALLS") == "0")
    check("inner_strict.sh (real): the try's own leftover simulator is killed, another group's is untouched",
          out.get("C_RC") == "0" and out.get("C_OWN") == "0" and out.get("C_OTHER") == "1")
    check("run_k23_strict.sh (real): a held lock -> exit 73, the queued job never runs, RUN END rc=73 logged",
          out.get("D_RC") == "73" and out.get("D_CALLS") == "0" and "RUN END rc=73" in out.get("D_LAST", ""))
    check("run_k23_strict.sh (real): the job's rc returned (4), one run of k23_run, own-group cleanup only",
          out.get("E_RC") == "4" and out.get("E_CALLS") == "1" and out.get("E_ARGS", "").startswith("sim/final_test/matrix/k23_run.py --phase execute")
          and out.get("E_OWN") == "0" and out.get("E_OTHER") == "1")
    check("run_strict.sh (real): no authorization token -> exit 74, nothing runs", out.get("F_RC") == "74" and out.get("F_CALLS") == "0")
    check("run_strict.sh (real): DELAYED START -- the token revoked while the launch waits for the lock -> exit 74 after the lock is free, "
          "nothing runs (Astra r3 A3)", out.get("G_RC") == "74" and out.get("G_CALLS") == "0")
finally:
    shutil.rmtree(fake, ignore_errors=True)
print(f"{ok}/{ok + fail}"); sys.exit(1 if fail else 0)
