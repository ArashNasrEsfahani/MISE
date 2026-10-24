"""Offline gate for k23_overnight.py (the strict K23 overnight runner) with a fake machine and a fake repo state: the step order, stop at the
first failure, never forcing (four-line Astra acceptance for the pre-signature, no waiver, no --provider-usd), git add + commit only with the
integrity re-check, preflight before side effects, the wait / COCO rules, WSL retries, the strict practice wrapper, the sim-lock limits,
resume-by-skip with evidence, durable chain evidence, the Kitchen 2 chain / Kitchen 3 freeze overlap (Astra K23-OVERNIGHT-r1 items 1-9),
the strict wrappers, the supervised practice, K2 polling at every K3 boundary, the G-iso rule (stop for Arash unless waived) and the
context-stage resume (r2 + coordinator 21:10), plus behavioural tests of the REAL State checks (chain evidence, masking, G-iso waiver)
on a temporary root. Nothing real is started: every command goes to the fake."""
import hashlib, json, os, re, shutil, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import k23_overnight as O
ok = fail = 0
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def check(n, c):
    global ok, fail
    if c: ok += 1
    else: fail += 1; print("FAIL", n)


SHA = {"FloorPlan15": "a1" * 32, "FloorPlan17": "b2" * 32}


def four(scene, sha=None):
    return "\n".join(O.acceptance_lines(scene, sha or SHA[scene])) + "\n"


def review_file(ans, case="K23-FREEZE-x"):
    return f"Astra (codex exec -m gpt-6-astra, read-only, stdin closed), case {case}\n\n=== PROMPT ===\n(prompt)\n\n{O.ANSWER_MARK}\n{ans}"


class Proc:
    def __init__(self, fs, ns, polls, rc):
        self.fs, self.ns, self.n, self.polls, self.rc = fs, ns, 0, polls, rc

    def poll(self):
        self.n += 1
        if self.ns == "test-k2-v1" and self.n >= self.fs.plan_after:
            self.fs.st.d["k2_plan"] = self.fs.plan_ok
        if self.n < self.polls:
            return None
        if self.rc == 0 and self.fs.k2_evidence:
            self.fs.st.d["done"][self.ns] = True
        return self.rc


class FakeSys:
    def __init__(self, st, root):
        self.st, self.root, self.calls, self.spawned, self.t, self.sleeps = st, root, [], [], 1000.0, []
        self.rc, self.answers = {}, {}
        self.tags, self.gpu_mib, self.branch = None, 500, "master"
        self.wsl_fail, self.lock_busy = 0, False
        self.k2, self.k3_chain = (3, 0), (2, 0)
        self.plan_after, self.plan_ok, self.k2_evidence = 2, True, True
        self.integrity_fail_after_commits, self.commits = None, 0
        self.coco_start_on_resume = True
        self.claims, self.tokens = [], []                          # practice launch tokens: claimed by the fake run_strict or not
        self.rename_fail, self.spawn_fail = False, False           # fault injection: revocation rename / practice spawn
        self.files = {}                                            # path -> (text, mtime) overrides for read / mtime

    def _rc(self, key, default=0):
        v = self.rc.get(key, default)
        return (v.pop(0) if v else default) if isinstance(v, list) else v

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s

    def exists(self, p):
        return os.path.exists(p)

    def rename(self, a, b):
        if self.rename_fail:
            raise PermissionError(13, "sharing violation (injected)", a)
        os.replace(a, b)

    def read(self, p):
        if p in self.files:
            return self.files[p][0]
        try:
            return open(p, encoding="utf-8").read()
        except OSError:
            return None

    def mtime(self, p):
        if p in self.files:
            v = self.files[p][1]
            return v() if callable(v) else v
        return os.path.getmtime(p) if os.path.exists(p) else None

    def listdir(self, p):
        names = {os.path.relpath(f, p).split(os.sep)[0] for f in self.files if f.startswith(p + os.sep)}
        return sorted(names | set(os.listdir(p) if os.path.isdir(p) else []))

    def which(self, exe):
        return "codex.exe"

    def pid_alive(self, pid):
        return False

    def http_tags(self):
        return self.tags

    def gpu(self):
        return {"mib": self.gpu_mib, "temp": 60}

    def win_cmdlines(self):
        return []

    def spawn(self, argv, *, env=None, out=None, detached=False):
        self.spawned.append((list(argv), dict(env or {})))
        if argv[0] == O.OLLAMA_EXE:
            self.tags = [O.GEMMA]
            return None
        if argv[0] == "wsl" and "train_k23.sh" in " ".join(argv):
            if self.coco_start_on_resume:
                c = self.st.d["coco"]
                c.insert(0, {"verdict": "running", "last": "START", "rc": None, "last_ts": "t2", "safety": [], "age_s": 1, "weights": False,
                             "n_start": 9})
                self.st.d["training_after_resume"] = 1                 # the training process is seen once (the resume check)
            return None
        if argv[0] == sys.executable and argv[1].endswith("launch_strict.py"):
            tok = os.path.join(self.root, *argv[4].split("/"))
            self.tokens.append(tok)
            if self.spawn_fail:
                raise OSError("spawn failed (injected)")
            rc = self._rc("practice")
            claim = self.claims.pop(0) if self.claims else rc != "never"   # run_strict claims its token right before the try
            if claim:
                os.replace(tok, tok + ".claimed")
            if rc == "never":
                return Proc(self, argv[2], 10 ** 9, 0)
            if rc == 0:
                self.st.d["practice"][argv[2]] = True
            return Proc(self, argv[2], 1, rc)
        if argv[0] == O.GIT_BASH:
            assert argv[1] == O.CHAIN, argv
            ns = argv[-1]
            if ns == "test-k2-v1":
                return Proc(self, ns, *self.k2)
            polls, rc = self.rc.get("chain " + ns, (2, 0)) if isinstance(self.rc.get("chain " + ns), tuple) else (2, self._rc("chain " + ns))
            p = Proc(self, ns, polls, rc)
            if ns.startswith("rehearsal") and rc == 0:
                self.st.d["rehearsal"][ns] = True
            return p
        raise AssertionError(f"unexpected spawn {argv}")

    def sh(self, argv, *, env=None, timeout=None, stream=False):
        self.calls.append((list(argv), dict(env or {})))
        a0, d = argv[0], self.st.d
        if a0 == "git":
            sub = argv[1]
            if sub == "rev-parse":
                return (0, self.branch + "\n") if "--abbrev-ref" in argv else (0, ".git\n")
            if sub == "check-ignore":
                return (0, "") if argv[-1].startswith("sim/kitchen_v3") else (1, "")
            if sub == "diff":
                return 0, "\n".join(argv[argv.index("--") + 1:])
            if sub == "commit":
                self.commits += 1
                if self.integrity_fail_after_commits is not None and self.commits >= self.integrity_fail_after_commits:
                    d["integrity"] = ["preregistration.md sha256 deadbeef != 247ef23a"]
                return 0, ""
            if sub in ("add", "log"):
                return 0, "abc1234 msg\n"
            raise AssertionError(f"git {sub} must never run")
        if a0 == "wsl":
            if self.wsl_fail > 0:
                self.wsl_fail -= 1
                return 255, "Error code: Wsl/Service/0x8007274c"
            rest = argv[argv.index("--") + 1:]
            if rest[0] == "pgrep":
                if not (rest[-1].startswith("'") and rest[-1].endswith("'")):
                    return 127, "/bin/bash: line 1: train_kitchen.py: command not found"   # real wsl.exe: bash -c runs an unquoted | as a pipe
                if d.get("training") is None:
                    return 2, "pgrep: error"
                if d.get("training_after_resume"):
                    d["training_after_resume"] -= 1
                    return 0, "1234 /home/arash/thor-env/bin/python -u sim/kitchen_v3/k23/k23_train.py train"
                if d.get("training"):
                    return 0, "1234 /home/arash/thor-env/bin/python -u sim/kitchen_v3/k23/k23_train.py train"
                return 1, ""
            if rest[0] == "true":
                return 0, ""
            if rest[0] == "flock":
                return (1, "") if self.lock_busy else (0, "")
            if "k23_freeze.py" in rest[1] and "--add-coco" in rest:
                rc = self._rc("add-coco")
                if rc == 0:
                    d["coco_freeze"].update(pending=False, listed=True)
                return rc, "COCO sensitivity checkpoint appended"
            if "k23_coco_meta.py" in rest[1]:
                if "--write" in rest:
                    rc = self._rc("coco-meta-write")
                    if rc == 0:
                        d["coco_freeze"].update(append_meta=True, problems=[])
                    return rc, ""
                return self._rc("coco-meta-check"), ""
            raise AssertionError(f"unexpected wsl {rest}")
        if a0 == sys.executable:
            script = os.path.basename(argv[1])
            if script == "k23_run.py" and "--phase" in argv and argv[argv.index("--phase") + 1] == "context":
                ns = argv[argv.index("--namespace") + 1]
                d["context_runs"].append(ns)
                rc = self._rc("context")
                if rc == 0:
                    d["rehearsal"][ns] = True
                return rc, ""
            if script in ("launch.py", "launch_strict.py"):
                raise AssertionError("the restarting practice launcher must never be used overnight")
            if script == "k23_practice.py":
                return 0, "{}"
            if script == "k23_capture_freeze.py":
                return self._rc("capture-check"), "{}"
            if script == "k23_size_decision.py":
                if "--write" in argv:
                    rc = self._rc("size-write")
                    if rc == 0:
                        d["size"].append(argv[argv.index("--checkpoint") + 1])
                        p = os.path.join(self.root, "docs", "final-test", "k23", "a29_size_decision.json")
                        os.makedirs(os.path.dirname(p), exist_ok=True)
                        open(p, "w").write("{}")
                    return rc, ""
                return self._rc("size-check" if "--check" in argv else "size-dry"), ""
            if script == "k23_freeze_agent.py":
                sc = argv[argv.index("--scene") + 1]
                if "--write" in argv:
                    rc = self._rc("freeze-write " + sc)
                    if rc == 0:
                        d["freeze"][sc] = {"exists": True, "sha": SHA[sc], "all_gates_pass": True, "presigned": False}
                    return rc, ""
                if "--presign" in argv:
                    d["freeze"][sc]["presigned"] = True
                    d["presign_reviews"].append(argv[argv.index("--astra-review") + 1])
                    return self._rc("presign " + sc), ""
                if "--check" in argv:
                    return self._rc("freeze-check " + sc), ""
                return self._rc("freeze-dry " + sc), ""
            if script == "astra_call.py":
                out = argv[argv.index("--out") + 1]
                case = argv[argv.index("--case") + 1]
                sc = "FloorPlan15" if "FloorPlan15" in case else "FloorPlan17"
                if self._rc("astra " + sc) == 0:
                    open(out, "w", encoding="utf-8").write(review_file(self.answers.get(sc, four(sc)), case))
                    return 0, ""
                return 1, "STOP: no Astra answer"
            raise AssertionError(f"unexpected python {argv}")
        raise AssertionError(f"unexpected command {argv}")


class FakeState:
    training = O.State.training

    def __init__(self):
        self.d = {"integrity": [], "rehearsal": {"rehearsal-k3-v1": True}, "practice": {"practice-k3-v1": True}, "done": {},
                  "size": [], "freeze": {}, "k2_plan": False, "presign_reviews": [], "busy": [], "training": False,
                  "coco": [{"verdict": "ended_ok", "last": "END", "rc": 0, "last_ts": "t", "safety": [], "age_s": 5, "weights": True, "n_start": 4}],
                  "coco_freeze": {"problems": ["coco pending"], "pending": True, "listed": False, "append_meta": False}, "freeze_raise": None,
                  "context_runs": []}

    def integrity(self):
        return list(self.d["integrity"])

    def coco(self):
        c = self.d["coco"]
        return dict(c.pop(0) if len(c) > 1 else c[0])

    def rehearsal(self, ns):
        r = self.d["rehearsal"].get(ns, False)
        if callable(r):
            r = r()
        if isinstance(r, dict):
            base = {"done": False, "why": ["G-iso inconclusive"], "context_only": False, "giso_block": False, "alive": [], "ctx_state": "absent",
                    "giso": {"status": "inconclusive", "waived": False, "waiver_why": ["no waiver record"], "waiver": "docs/x.json"}}
            base.update(r)
            return base
        return {"done": bool(r), "why": [] if r else ["capture not complete"], "context_only": False, "giso_block": False, "alive": [],
                "ctx_state": "ok" if r else "absent",
                "giso": {"status": "pass" if r else None, "waived": False, "waiver_why": [], "waiver": "docs/x.json"}}

    def test_done(self, ns):
        r = self.d["done"].get(ns, False)
        return {"done": bool(r), "why": [] if r else ["no CHAIN DONE"]}

    def k3_busy(self):
        b = self.d["busy"]
        return b.pop(0) if isinstance(b, list) and b and isinstance(b[0], list) else b

    def coco_freeze(self):
        return dict(self.d["coco_freeze"])

    def practice(self, ns):
        return {"done": self.d["practice"].get(ns, False), "attempts": 10 if self.d["practice"].get(ns) else 0, "ok": True, "rc": 0}

    def giso(self, ns):
        g = dict(self.rehearsal(ns)["giso"])
        g.setdefault("waiver_decision_sha256", "d" * 64 if g.get("waived") else None)
        return g

    def k2_plan_complete(self):
        return self.d["k2_plan"]

    def size_checkpoints(self):
        return list(self.d["size"])

    def freeze(self, scene):
        if self.d["freeze_raise"] == scene:
            raise RuntimeError("boom (fake internal error)")
        return dict(self.d["freeze"].get(scene) or {"exists": False})


STRICT_FILES = ["sim/final_test/k23_run/" + f for f in ("k23_strict_wrappers.py", "chain_k23.sh", "chain_k23_strict.sh", "launch_k23.py",
                                                          "launch_k23_strict.py", "run_k23.sh", "run_k23_strict.sh", "practice/inner.sh",
                                                          "practice/inner_strict.sh")]


def setup():
    root = tempfile.mkdtemp(prefix="k23ovn_")
    for rel in [O.CAPTURE_FREEZE, O.CAPTURE_REVIEW] + STRICT_FILES:     # the capture gate / strict-wrapper check read the real files
        dst = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(REPO, *rel.split("/")), dst)
    st = FakeState()
    fs = FakeSys(st, root)
    return root, st, fs, O.Runner(root, fs, st, poll_s=300)


def ran(fs, needle):
    return any(needle in " ".join(c) for c, _ in fs.calls)


def stopped(root):
    p = os.path.join(root, *O.OUT.split("/"), "STOPPED.json")
    return json.load(open(p)) if os.path.exists(p) else {}


O_OLLAMA = O.OLLAMA_EXE
O.OLLAMA_EXE = os.path.abspath(__file__)                       # an existing file stands in for ollama.exe (never executed: spawn is fake)

# ---------------------------------------------------------------------------------------------- 1. the whole night, every step pending
root, st, fs, r = setup()
rc = r.run()
seq = [c for c, _ in fs.calls if c[0] != "git"] + [s[0] for s in fs.spawned]
calls_only = [c for c, _ in fs.calls if c[0] != "git"]
order = [("add-coco", lambda c: "--add-coco" in c), ("practice status", lambda c: c[1].endswith("k23_practice.py")),
         ("S2", lambda c: "after-k1" in c and "--write" in c), ("F2", lambda c: "--presign" in c and "FloorPlan15" in c),
         ("S3", lambda c: "after-fp15" in c and "--write" in c), ("F3", lambda c: "--presign" in c and "FloorPlan17" in c)]
idx = [next((i for i, c in enumerate(calls_only) if f(c)), None) for _, f in order]
check("the full night runs to DONE (rc 0) with DONE.json", rc == 0 and os.path.exists(os.path.join(root, *O.OUT.split("/"), "DONE.json")))
check("steps in the registered order: COCO append, K2 practice, after-k1, F2 presign, after-fp15, F3 presign", None not in idx and idx == sorted(idx))
chains = [s for s in fs.spawned if s[0][0] == O.GIT_BASH]
check("chains started in order rehearsal-k2, test-k2, test-k3 (each once), all through the STRICT chain",
      [c[0][-1] for c in chains] == ["rehearsal-k2-v1", "test-k2-v1", "test-k3-v1"] and all(c[0][1] == O.CHAIN for c in chains))
sp = [s[0] for s in fs.spawned]
check("the strict practice is spawned (supervised) before the rehearsal chain", any(a[0] == sys.executable and a[1].endswith("launch_strict.py") for a in sp)
      and next(i for i, a in enumerate(sp) if a[0] == sys.executable) < next(i for i, a in enumerate(sp) if a[0] == O.GIT_BASH))
check("the test chains run with ENERGY=on and the waiver removed; the rehearsal without ENERGY; HANG_RELAUNCH removed for all",
      all(c[1].get("ENERGY") == "on" and "ENERGY_WAIVER" in c[1] and c[1]["ENERGY_WAIVER"] is None for c in chains[1:]) and "ENERGY" not in chains[0][1]
      and all(c[1].get("HANG_RELAUNCH", "x") is None for c in chains))
check("no --provider-usd, --supersede, --no-selftests or --no-probe anywhere", not any(ran(fs, x) for x in ("--provider-usd", "--supersede", "--no-selftests", "--no-probe")))
gits = {c[1] for c, _ in fs.calls if c[0] == "git"}
check("git: only rev-parse / check-ignore / add / diff / commit / log", gits <= {"rev-parse", "check-ignore", "add", "diff", "commit", "log"} and "commit" in gits)
check("the presign uses the review file of THIS freeze sha", st.d["presign_reviews"] == [f"{O.ASTRA_DIR}/{O.astra_case(sc, SHA[sc])}.txt" for sc in ("FloorPlan15", "FloorPlan17")])
pr = open(os.path.join(root, *O.ASTRA_DIR.split("/"), O.astra_case("FloorPlan15", SHA["FloorPlan15"]) + ".txt.prompt.txt"), encoding="utf-8").read()
check("the Astra prompt carries the kitchen, the full self-sha and the exact four acceptance lines, never the answer marker",
      SHA["FloorPlan15"] in pr and four("FloorPlan15") in pr and O.ANSWER_MARK not in pr)
check("the capture artefact check runs before each chain start", sum(1 for c, _ in fs.calls if c[0] == sys.executable and c[1].endswith("k23_capture_freeze.py")) == 3)
check("the sim lock is tested (flock -n, never stolen) before the practice and every chain", sum(1 for c, _ in fs.calls if c[0] == "wsl" and "flock" in c) >= 4)
first_wsl = next(i for i, (c, _) in enumerate(fs.calls) if c[0] == "wsl")
check("preflight: git state checked before any WSL side effect", any(c[:2] == ["git", "rev-parse"] for c, _ in fs.calls[:first_wsl]))
adds = [c for c, _ in fs.calls if c[:2] == ["git", "add"]]
check("evidence commits never include applicability rows, the preregistration or gitignored paths",
      not any("applicability" in x or x == O.PREREG or x.startswith("sim/kitchen_v3") for c in adds for x in c[3:]))
check("the size decision record and the freeze review files are committed", any("docs/final-test/k23/a29_size_decision.json" in c for c in adds)
      and any(any(x.endswith(".txt.prompt.txt") for x in c) for c in adds))
shutil.rmtree(root, ignore_errors=True)

# ---------------------------------------------------------------------------------------------- 2. stop at the first failure (items 1, 3, 9)
root, st, fs, r = setup()
fs.rc["practice"] = 3
rc = r.run()
PR = lambda fs: [s[0] for s in fs.spawned if s[0][0] == sys.executable and s[0][1].endswith("launch_strict.py")]
check("a failed strict practice stops at P2 and nothing after it runs (no chain, no size decision)", rc == 4 and stopped(root).get("step") == "P2"
      and not any(s[0][0] == O.GIT_BASH for s in fs.spawned) and not ran(fs, "k23_size_decision.py"))
check("... the practice ran once (strict: no restart), with the 30-min lock limit and a one-time launch token", len(PR(fs)) == 1
      and PR(fs)[0][3] == str(O.LOCK_WAIT_S) and PR(fs)[0][4].startswith(O.AUTH_DIR + "/") and PR(fs)[0][4].endswith(".token"))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
fs.rc["practice"] = 73
rc = r.run()
check("strict practice rc 73 (sim lock not acquired) stops, named as such, no retry", rc == 4 and "sim lock not acquired" in stopped(root).get("why", "")
      and len(PR(fs)) == 1)
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
fs.rc["practice"] = [255, 0]
fs.claims = [False, True]                                     # the first launch never reached its claim (WSL refused): retry is safe
rc = r.run()
check("a practice rc 255 (WSL connection signature) is retried once after 90 s", rc == 0 and 90 in fs.sleeps and len(PR(fs)) == 2)
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
fs.rc["practice"] = "never"
rc = r.run()
tokd = os.path.join(root, *O.AUTH_DIR.split("/"))
check("a practice that never logs its queued line -> stop after 15 min, its launch token REVOKED (a late launcher can never start), nothing after it",
      rc == 4 and "did not start" in stopped(root).get("why", "") and "revoked" in stopped(root).get("why", "")
      and any(f.endswith(".token.revoked") for f in os.listdir(tokd)) and not any(f.endswith(".token") for f in os.listdir(tokd))
      and not any(s[0][0] == O.GIT_BASH for s in fs.spawned))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
fs.rc["practice"] = "never"
fs.files[os.path.join(root, "sim", "final_test", "k23_run", "practice", "practice-k2-v1", "progress.txt")] = ("t strict try queued\n", lambda: fs.t)
rc = r.run()
check("a practice running past lock wait + 4 h + 1 h -> stop (left as it is), with status heartbeats meanwhile", rc == 4 and "exceeded" in stopped(root).get("why", "")
      and os.path.exists(os.path.join(root, *O.OUT.split("/"), "status.json")))
tk = fs.tokens[0]
late = True
try:
    os.rename(tk, tk + ".claimed")                              # a late launcher (queued, stalled before flock) tries to claim now
except OSError:
    late = False
check("QUEUED then stalled before its claim, supervisor timeout -> the unclaimed authorization is revoked; a late claim fails (Astra r4 3)",
      "revoked" in stopped(root).get("why", "") and os.path.exists(tk + ".revoked") and not late)
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
fs.rc["practice"] = [255]
fs.claims = [True]
rc = r.run()
check("a practice rc 255 AFTER its launch was claimed (practice may have run) -> no retry, stop", rc == 4 and len(PR(fs)) == 1
      and os.path.exists(fs.tokens[0] + ".claimed"))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
fs.rc["practice"] = [255, 0]
fs.claims = [False, True]
rc = r.run()
check("a never-claimed rc 255 launch is retried with a NEW authorization; the first one is revoked", rc == 0 and len(PR(fs)) == 2
      and os.path.exists(fs.tokens[0] + ".revoked") and os.path.exists(fs.tokens[1] + ".claimed") and fs.tokens[0] != fs.tokens[1])
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
fs.rc["practice"] = [255, 0]
fs.claims = [False, True]
fs.rename_fail = True
rc = r.run()
check("rc 255 but the revocation rename FAILS (unknown) -> no retry, stop, the message says so (Astra r5 1)", rc == 4 and len(PR(fs)) == 1
      and "could NOT be confirmed" in stopped(root).get("why", "") and os.path.exists(fs.tokens[0]))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
fs.rc["practice"] = "never"
fs.files[os.path.join(root, "sim", "final_test", "k23_run", "practice", "practice-k2-v1", "progress.txt")] = ("t strict try queued\n", lambda: fs.t)
fs.rename_fail = True
rc = r.run()
check("total timeout with a FAILED revocation -> stop says the revocation could not be confirmed (never 'claimed')", rc == 4
      and "exceeded" in stopped(root).get("why", "") and "could NOT be confirmed" in stopped(root).get("why", "")
      and "claimed its authorization" not in stopped(root).get("why", ""))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
fs.spawn_fail = True
rc = r.run()
check("an exception at spawn (after the token was created) -> the token is revoked by the cleanup scope, stop (Astra r5 2)",
      rc == 1 and "internal error" in stopped(root).get("why", "") and fs.tokens and os.path.exists(fs.tokens[0] + ".revoked")
      and not os.path.exists(fs.tokens[0]))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
fs.rc["practice"] = [2]
fs.claims = [False]
rc = r.run()
check("any return before the claim (launcher refused, rc 2) -> the authorization is revoked on the way out", rc == 4
      and os.path.exists(fs.tokens[0] + ".revoked") and not os.path.exists(fs.tokens[0]))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
st.d["practice"]["practice-k2-v1"] = True
st.d["rehearsal"]["rehearsal-k2-v1"] = {"ctx_state": "ok", "why": ["no successful-completion evidence"],
                                        "giso": {"status": "pass", "waived": False, "waiver_why": [], "waiver": "x"}}
rc = r.run()
check("K2 rehearsal with a context report but no CHAIN DONE -> stop at R2: no chain launched, no downstream step (Astra r4 2)",
      rc == 4 and stopped(root).get("step") == "R2" and not any(s[0][0] == O.GIT_BASH for s in fs.spawned) and not ran(fs, "k23_size_decision.py"))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
st.d["rehearsal"]["rehearsal-k3-v1"] = {"ctx_state": "bad", "why": ["a context-fit report exists but is not freeze-grade"],
                                        "giso": {"status": "inconclusive", "waived": True, "waiver_why": [], "waiver": "x"}}
rc = r.run()
check("K3 with a failed context report (waived G-iso) -> stop in W, the context stage never re-run (Astra r4 1)",
      rc == 4 and stopped(root).get("step") == "W" and not st.d["context_runs"])
shutil.rmtree(root, ignore_errors=True)

for name, ans in (("CHANGES REQUIRED", "RULING: CHANGES REQUIRED\n1. gate tokens evidence missing"),
                  ("ACCEPTED + a condition line", four("FloorPlan15") + "Proceed only after fixing the remaining gate.\n"),
                  ("text before the ruling", "Looks fine.\n" + four("FloorPlan15")),
                  ("another sha", four("FloorPlan15", "ff" * 32)),
                  ("old free-form ACCEPTED", f"RULING: ACCEPTED\nFloorPlan15 {SHA['FloorPlan15'][:16]} verified.")):
    root, st, fs, r = setup()
    fs.answers["FloorPlan15"] = ans
    rc = r.run()
    check(f"Astra {name} -> no presign, stop at F2, no test chain, the review committed", rc == 8 and not st.d["presign_reviews"]
          and stopped(root).get("step") == "F2" and not any(s[0][0] == O.GIT_BASH and s[0][-1].startswith("test") for s in fs.spawned))
    shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
fs.rc["astra FloorPlan15"] = 1
rc = r.run()
check("no Astra answer after 3 tries (10 min apart) -> stop, no presign", rc == 8 and not st.d["presign_reviews"]
      and sum(1 for c, _ in fs.calls if c[0] == sys.executable and c[1].endswith("astra_call.py")) == 3)
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
fs.integrity_fail_after_commits = 1
check("an integrity failure after a commit stops the night (rc 9)", r.run() == 9)
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
st.d["integrity"] = ["treatment.pin_problems() = ['x']"]
rc = r.run()
check("an integrity failure at the start stops before any WSL / COCO side effect", rc == 9 and not any(c[0] == "wsl" for c, _ in fs.calls) and not fs.spawned)
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
fs.branch = "k23-agent-side"
st.d["coco"] = [{"verdict": "resumable", "last": "END", "rc": 143, "last_ts": "t", "safety": [], "age_s": 5000, "weights": False, "n_start": 4}]
rc = r.run()
check("HEAD not master -> stop in the preflight: no COCO resume, no WSL call, nothing committed (item 9)",
      rc == 10 and not fs.spawned and not any(c[0] == "wsl" for c, _ in fs.calls) and fs.commits == 0)
shutil.rmtree(root, ignore_errors=True)

# ---------------------------------------------------------------------------------------------- 3. the K2 chain / K3 work overlap (items 2, 5)
root, st, fs, r = setup()
fs.k2, fs.plan_after = (6, 4), 99
rc = r.run()
check("K2 chain fails before its paid planning completed -> stop at T2; no K3 size decision / freeze / test", rc == 4 and stopped(root).get("step") == "T2"
      and not ran(fs, "after-fp15") and not ran(fs, "FloorPlan17") and not any(s[0][-1] == "test-k3-v1" for s in fs.spawned))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
fs.k2, fs.plan_after = (6, 4), 2
rc = r.run()
check("K2 chain fails AFTER plan_first_complete while the K3 work runs: S3 done, F3's commands never run (polled before each), stop at T2",
      rc == 4 and stopped(root).get("step") == "T2" and ran(fs, "after-fp15") and not ran(fs, "--scene FloorPlan17 --dry-run")
      and not any(s[0][-1] == "test-k3-v1" for s in fs.spawned))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
fs.k2, fs.plan_after = (6, 4), 2
n_tags = []
fs.http_tags = (lambda f=fs.http_tags: (n_tags.append(1), f())[1])
rc = r.run()
commits = [c[0][4] for c in fs.calls if c[0][:2] == ["git", "commit"]]
check("K2 fails right after S3's last command: S3's commit and every F3 operation (Ollama, WSL, freeze) are skipped (polled at each boundary)",
      rc == 4 and not any("K23 overnight S3" in c for c in commits) and not ran(fs, "FloorPlan17")
      and not any(c[0] == "wsl" and c[-1] == "true" for c, _ in fs.calls[[i for i, (c, _) in enumerate(fs.calls) if "after-fp15" in c][-1]:]))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
fs.k2, fs.plan_after = (3, 4), 2
rc = r.run()
check("K2 exits non-zero at the poll where planning completes -> the K3 work never starts", rc == 4 and not ran(fs, "after-fp15"))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
fs.k2 = (9, 0)
fs.answers["FloorPlan17"] = "RULING: CHANGES REQUIRED\nx"
rc = r.run()
commits = [c[0][4] for c in fs.calls if c[0][:2] == ["git", "commit"]]
check("a K3 freeze failure while K2 runs: K2 waited for (never killed), its evidence committed, then stop at F3; no K3 test",
      rc == 8 and st.d["done"].get("test-k2-v1") and stopped(root).get("step") == "F3" and any("K23 overnight T2" in c for c in commits)
      and not any(s[0][-1] == "test-k3-v1" for s in fs.spawned))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
fs.k2 = (9, 0)
st.d["freeze_raise"] = "FloorPlan17"
rc = r.run()
check("an unexpected error in the K3 work goes through the drain-K2 path: K2 evidence committed, then stop (item 2)",
      rc == 1 and st.d["done"].get("test-k2-v1") and "internal error" in stopped(root).get("why", "")
      and any("K23 overnight T2" in c[0][4] for c in fs.calls if c[0][:2] == ["git", "commit"]))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
fs.k2, fs.k2_evidence = (3, 0), False
rc = r.run()
check("K2 chain rc 0 without durable evidence (CHAIN DONE / phases / seal) -> stop at T2, no K3 test (item 5)",
      rc == 4 and stopped(root).get("step") == "T2" and "durable" in stopped(root).get("why", "") and not any(s[0][-1] == "test-k3-v1" for s in fs.spawned))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
fs.k2 = (5, 255)
r.run()
check("a K2 chain rc 255 is relaunched once (resumes from its journals)", sum(1 for s in fs.spawned if s[0][-1] == "test-k2-v1") == 2)
shutil.rmtree(root, ignore_errors=True)

# ---------------------------------------------------------------------------------------------- 4. sim lock (item 8)
root, st, fs, r = setup()
fs.lock_busy = True
rc = r.run()
check("the sim lock held by another job for > 30 min -> stop before the practice (nothing stolen, nothing launched)",
      rc == 11 and stopped(root).get("step") == "P2" and not ran(fs, "launch_strict.py") and fs.sleeps.count(60) >= 29)
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
st.d["practice"]["practice-k2-v1"] = True
fs.rc["chain rehearsal-k2-v1"] = (999, 0)
pf = os.path.join(root, *O.LOGS.split("/"), "rk2-capture", "progress.txt")
fs.files[pf] = ("2026-10-21 01:00:00 queued k23_run --phase capture --namespace rehearsal-k2-v1\n", 1000.0 + 60)
rc = r.run()
check("a supervised chain whose WSL job waits for the sim lock > 30 min -> stop (the chain left as it is, never killed)",
      rc == 11 and stopped(root).get("step") == "R2" and "waited for the sim lock" in stopped(root).get("why", ""))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
st.d["practice"]["practice-k2-v1"] = True
pf = os.path.join(root, *O.LOGS.split("/"), "rk2-capture", "progress.txt")
fs.files[pf] = ("2026-10-19 01:00:00 queued k23_run --phase capture --namespace rehearsal-k2-v1\n", 10.0)
check("a 'queued' job file of an EARLIER run is ignored by the stall check", r.run() == 0)
shutil.rmtree(root, ignore_errors=True)

# ---------------------------------------------------------------------------------------------- 5. the wait / COCO rules (item 4)
root, st, fs, r = setup()
reh = iter([False, False, True])
st.d["rehearsal"]["rehearsal-k3-v1"] = lambda: next(reh, True)
st.d["busy"] = [["9 powershell -File k23_k3_x.ps1"], ["9 powershell -File k23_k3_x.ps1"], []]
st.d["coco"] = [{"verdict": "resumable", "last": "END", "rc": 143, "last_ts": "t", "safety": [], "age_s": 5, "weights": False, "n_start": 4}] * 2 + \
               [{"verdict": "resumable", "last": "END", "rc": 143, "last_ts": "t", "safety": [], "age_s": 5000, "weights": False, "n_start": 4},
                {"verdict": "running", "last": "START", "rc": None, "last_ts": "t", "safety": [], "age_s": 5, "weights": False, "n_start": 5},
                {"verdict": "ended_ok", "last": "END", "rc": 0, "last_ts": "t", "safety": [], "age_s": 5, "weights": True, "n_start": 5}]
fs.tags = [O.GEMMA]
rc = r.run()
res = [s for s in fs.spawned if s[0][0] == "wsl"]
check("a paused COCO is left alone while the K3 work runs, resumed once after it (new START + training process seen), then the night continues",
      rc == 0 and len(res) == 1 and res[0][0][-2:] == ["FloorPlan15", "yolo26s_coco"] and r.resumes == 1)
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
st.d["coco"] = [{"verdict": "resumable", "last": "END", "rc": 143, "last_ts": "t", "safety": [], "age_s": 9000, "weights": False, "n_start": 4}]
st.d["training"] = None
rc = r.run()
check("COCO resume with the WSL process list unreadable -> stop, nothing launched", rc == 5 and not any(s[0][0] == "wsl" for s in fs.spawned))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
st.d["coco"] = [{"verdict": "resumable", "last": "END", "rc": 143, "last_ts": "t", "safety": [], "age_s": 9000, "weights": False, "n_start": 4}]
fs.gpu_mib = 4500
rc = r.run()
check("COCO resume with the GPU not idle -> stop, nothing launched", rc == 5 and not any(s[0][0] == "wsl" for s in fs.spawned))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
st.d["coco"] = [{"verdict": "resumable", "last": "END", "rc": 143, "last_ts": "t", "safety": [], "age_s": 9000, "weights": False, "n_start": 4}]
fs.coco_start_on_resume = False
rc = r.run()
check("a COCO resume that shows no new START marker + training process in 10 min -> stop", rc == 5 and "new START" in stopped(root).get("why", ""))
shutil.rmtree(root, ignore_errors=True)

for verdict, extra, why in (("safety", {"safety": ["HEAT"]}, "heat pause"), ("failed", {"rc": 1}, "END rc 1")):
    root, st, fs, r = setup()
    st.d["coco"] = [dict({"verdict": verdict, "last": "END", "rc": 143, "last_ts": "t", "safety": [], "age_s": 5, "weights": False, "n_start": 4}, **extra)]
    check(f"COCO {why} -> stop in W, no resume", r.run() == 5 and not any(s[0][0] == "wsl" for s in fs.spawned))
    shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
r.coco_resume = False
st.d["coco"] = [{"verdict": "resumable", "last": "START", "rc": None, "last_ts": "t", "safety": [], "age_s": 9999, "weights": False, "n_start": 4}]
check("--no-coco-resume: a WSL-killed COCO run stops the night", r.run() == 5 and not any(s[0][0] == "wsl" for s in fs.spawned))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
st.d["rehearsal"]["rehearsal-k3-v1"] = False
check("K3 rehearsal not complete and nothing running it (3 polls) -> stop in W", r.run() == 4 and stopped(root).get("step") == "W")
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
st.d["busy"] = ["9 bash chain_k23.sh rehearsal-k3-v1"]
st.d["coco"] = [{"verdict": "running", "last": "START", "rc": None, "last_ts": "t", "safety": [], "age_s": 5, "weights": False, "n_start": 4}]
r.max_wait_h = 1
check("the wait has a deadline -> stop rc 6", r.run() == 6)
shutil.rmtree(root, ignore_errors=True)

# ---------------------------------------------------------------------------------------------- 6. machine rules, branches, resume (items 6, 7)
root, st, fs, r = setup()
fs.wsl_fail = 2
check("WSL connection refusals are retried 60 s apart", r.wsl(["true"], step="t")[0] == 0 and fs.sleeps.count(60) == 2)
fs.wsl_fail = 99
check("WSL unreachable after 10 tries -> 255", r.wsl(["true"], step="t")[0] == 255)
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
st.d["training"] = True
check("a training process in WSL stops the night before Ollama / practice", r.run() == 11 and not ran(fs, "launch_strict.py"))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
fs.gpu_mib = 6000
check("GPU not idle with Ollama down -> stop before starting Ollama", r.run() == 11 and not any(s[0][0] == O.OLLAMA_EXE for s in fs.spawned))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
open(os.path.join(root, *O.CAPTURE_FREEZE.split("/")), "a").write("\n# edited\n")
st.d["practice"]["practice-k2-v1"] = True
rc = r.run()
check("a k23_capture_freeze.py that is not the reviewed version -> the capture gate stops before the chain (item 6)",
      rc == 8 and stopped(root).get("step") == "R2" and not any(s[0][0] == O.GIT_BASH for s in fs.spawned))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
st.d["coco_freeze"] = {"problems": ["meta stale"], "pending": False, "listed": True, "append_meta": False}
r.step_coco()
check("COCO appended, metadata stale -> only k23_coco_meta --write (+ check), never a second --add-coco",
      not ran(fs, "--add-coco") and ran(fs, "k23_coco_meta.py --write"))
st.d["coco_freeze"] = {"problems": ["odd"], "pending": True, "listed": True, "append_meta": False}
try:
    r.step_coco(); check("an unexpected COCO freeze state stops", False)
except O.Stop as e:
    check("an unexpected COCO freeze state stops", e.step == "C")
shutil.rmtree(root, ignore_errors=True)

# resume by skip, with the evidence still committed; already-signed freezes verified
root, st, fs, r = setup()
st.d.update(practice={"practice-k2-v1": True, "practice-k3-v1": True}, rehearsal={"rehearsal-k2-v1": True, "rehearsal-k3-v1": True},
            size=["after-k1", "after-fp15"], k2_plan=True, done={"test-k2-v1": True, "test-k3-v1": True})
st.d["coco_freeze"] = {"problems": [], "pending": False, "listed": True, "append_meta": True}
for sc in SHA:
    rel = f"{O.ASTRA_DIR}/{O.astra_case(sc, SHA[sc])}.txt"
    p = os.path.join(root, *rel.split("/"))
    open(p, "w", encoding="utf-8").write(review_file(four(sc)))
    st.d["freeze"][sc] = {"exists": True, "sha": "c3" * 32, "all_gates_pass": True, "presigned": True,
                          "presignature": {"astra_review": rel, "astra_review_sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
                                           "presigned_sha": SHA[sc]}}
ev = os.path.join(root, "docs", "final-test", "matrix", "practice-k2-v1")
os.makedirs(ev); open(os.path.join(ev, "practice.jsonl"), "w").write("{}\n")
fs.tags = [O.GEMMA]
rc = r.run()
adds = [x for c, _ in fs.calls if c[:2] == ["git", "add"] for x in c[3:]]
check("a re-run after everything is done launches nothing (only --check calls) and ends DONE", rc == 0 and not fs.spawned
      and not ran(fs, "--write") and not PR(fs) and ran(fs, "--check"))
check("... and still commits the evidence of skipped steps (practice journal, freeze review files) (item 7)",
      "docs/final-test/matrix/practice-k2-v1" in adds and any(x.endswith(O.astra_case("FloorPlan15", SHA["FloorPlan15"]) + ".txt") for x in adds))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
st.d.update(practice={"practice-k2-v1": True, "practice-k3-v1": True}, rehearsal={"rehearsal-k2-v1": True, "rehearsal-k3-v1": True},
            size=["after-k1"])
st.d["coco_freeze"] = {"problems": [], "pending": False, "listed": True, "append_meta": True}
rel = f"{O.ASTRA_DIR}/{O.astra_case('FloorPlan15', SHA['FloorPlan15'])}.txt"
p = os.path.join(root, *rel.split("/"))
open(p, "w", encoding="utf-8").write(review_file(four("FloorPlan15")))
st.d["freeze"]["FloorPlan15"] = {"exists": True, "sha": "c3" * 32, "all_gates_pass": True, "presigned": True,
                                 "presignature": {"astra_review": rel, "astra_review_sha256": "0" * 64, "presigned_sha": SHA["FloorPlan15"]}}
fs.tags = [O.GEMMA]
rc = r.run()
check("an already-signed freeze whose review file no longer has its recorded sha -> stop at F2, no test chain (item 3)",
      rc == 8 and stopped(root).get("step") == "F2" and not any(s[0][0] == O.GIT_BASH for s in fs.spawned))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
st.d.update(practice={"practice-k2-v1": True, "practice-k3-v1": True}, rehearsal={"rehearsal-k2-v1": True, "rehearsal-k3-v1": True}, size=["after-k1"])
st.d["coco_freeze"] = {"problems": [], "pending": False, "listed": True, "append_meta": True}
open(p := os.path.join(root, *rel.split("/")), "w", encoding="utf-8").write(review_file(f"RULING: ACCEPTED\nFloorPlan15 {SHA['FloorPlan15'][:16]}"))
st.d["freeze"]["FloorPlan15"] = {"exists": True, "sha": "c3" * 32, "all_gates_pass": True, "presigned": True,
                                 "presignature": {"astra_review": rel, "astra_review_sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
                                                  "presigned_sha": SHA["FloorPlan15"]}}
fs.tags = [O.GEMMA]
check("an already-signed freeze whose review is not the four-line acceptance -> stop at F2", r.run() == 8 and stopped(root).get("step") == "F2")
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
st.d.update(practice={"practice-k2-v1": True, "practice-k3-v1": True}, rehearsal={"rehearsal-k2-v1": True, "rehearsal-k3-v1": True}, size=["after-k1"])
st.d["coco_freeze"] = {"problems": [], "pending": False, "listed": True, "append_meta": True}
st.d["freeze"]["FloorPlan15"] = {"exists": True, "sha": "c3" * 32, "all_gates_pass": True, "presigned": True, "arash_check": {"by": "Arash"}}
fs.tags = [O.GEMMA]
fs.k2 = (2, 4)
rc = r.run()
check("a freeze carrying Arash's own check is accepted (the night proceeds to the K2 test)", stopped(root).get("step") == "T2")
shutil.rmtree(root, ignore_errors=True)

# ---------------------------------------------------------------------------------------------- 6b. G-iso rule + context-stage resume
GISO_BLOCK = {"giso_block": True, "why": ["G-iso inconclusive without a valid waiver"]}
root, st, fs, r = setup()
st.d["rehearsal"]["rehearsal-k3-v1"] = GISO_BLOCK
rc = r.run()
check("K3 G-iso inconclusive without a waiver -> stop in W for Arash (no retry, nothing run), the reason names the waiver record",
      rc == 4 and stopped(root).get("step") == "W" and "waiver" in stopped(root).get("why", "") and "A20" in stopped(root).get("why", "")
      and not fs.spawned and not ran(fs, "--add-coco"))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
st.d["rehearsal"]["rehearsal-k3-v1"] = {"context_only": True, "why": ["context fit not measured"],
                                        "giso": {"status": "inconclusive", "waived": True, "waiver_why": [], "waiver": "docs/x.json"}}
rc = r.run()
ctx = [c for c, _ in fs.calls if c[0] == sys.executable and c[1].endswith("k23_run.py")]
check("K3 complete up to the context fit with a waived G-iso: the night goes on; R3c runs ONLY the context stage (after COCO + Ollama)",
      rc == 0 and st.d["context_runs"][:1] == ["rehearsal-k3-v1"] and ctx and ctx[0][2:5] == ["--phase", "context", "--namespace"]
      and not any(s[0][-1] == "rehearsal-k3-v1" for s in fs.spawned))
crp = os.path.join(root, *O.OUT.split("/"), "context_resume_rehearsal-k3-v1.json")
check("... and only after it exited 0 the runner records it (context_resume_rehearsal-k3-v1.json, rc 0, waiver sha) and commits it",
      os.path.exists(crp) and json.load(open(crp))["rc"] == 0 and json.load(open(crp))["waiver_decision_sha256"] == "d" * 64
      and any("context_resume_rehearsal-k3-v1.json" in x for c, _ in fs.calls if c[:2] == ["git", "add"] for x in c))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
st.d["rehearsal"]["rehearsal-k3-v1"] = {"context_only": True, "why": ["context fit not measured"],
                                        "giso": {"status": "inconclusive", "waived": True, "waiver_why": [], "waiver": "docs/x.json"}}
fs.rc["context"] = 3
rc = r.run()
check("a context stage that exits non-zero -> stop, NO success record written", rc == 3 and stopped(root).get("step") == "R3c"
      and not os.path.exists(os.path.join(root, *O.OUT.split("/"), "context_resume_rehearsal-k3-v1.json")))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
st.d["practice"]["practice-k2-v1"] = True
seq2 = iter([False, True])
st.d["rehearsal"]["rehearsal-k2-v1"] = lambda: next(seq2, True)
fs.rc["chain rehearsal-k2-v1"] = (2, 3)
rc = r.run()
check("a rehearsal chain that exits non-zero is never accepted as complete, even when the done-checks hold afterwards (Astra r3 A1)",
      rc == 4 and stopped(root).get("step") == "R2" and "chain rc 3" in stopped(root).get("why", ""))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
st.d["practice"]["practice-k2-v1"] = True
st.d["rehearsal"]["rehearsal-k2-v1"] = GISO_BLOCK
rc = r.run()
check("K2 rehearsal with a non-pass G-iso and no waiver -> stop at R2 before any chain (no re-run)", rc == 4 and stopped(root).get("step") == "R2"
      and not any(s[0][0] == O.GIT_BASH for s in fs.spawned))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
st.d["practice"]["practice-k2-v1"] = True
seq = iter([False, GISO_BLOCK])
st.d["rehearsal"]["rehearsal-k2-v1"] = lambda: next(seq, GISO_BLOCK)
fs.rc["chain rehearsal-k2-v1"] = (2, 4)
rc = r.run()
check("K2 rehearsal chain stops at an inconclusive G-iso -> stop at R2 for Arash (G-iso reason, not a generic chain failure)",
      rc == 4 and "G-iso" in stopped(root).get("why", "") and "waiver" in stopped(root).get("why", ""))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
st.d["practice"]["practice-k2-v1"] = True
st.d["rehearsal"]["rehearsal-k2-v1"] = {"context_only": True, "giso": {"status": "inconclusive", "waived": True, "waiver_why": [], "waiver": "x"}}
rc = r.run()
check("K2 rehearsal resumable at the context stage (waived G-iso): only the context stage runs, no chain, then the night goes on",
      rc == 0 and "rehearsal-k2-v1" in st.d["context_runs"] and not any(s[0][-1] == "rehearsal-k2-v1" for s in fs.spawned))
shutil.rmtree(root, ignore_errors=True)

root, st, fs, r = setup()
open(os.path.join(root, "sim", "final_test", "k23_run", "chain_k23_strict.sh"), "a").write("# drift\n")
rc = r.run()
check("a strict wrapper that drifted from its rendered original -> stop in the preflight (rc 12), nothing run", rc == 12 and not fs.spawned
      and not any(c[0] == "wsl" for c, _ in fs.calls))
shutil.rmtree(root, ignore_errors=True)

# ---------------------------------------------------------------------------------------------- 6c. the REAL State checks on a temporary root
sys.path.insert(0, os.path.join(REPO, "sim", "final_test", "matrix"))
import common as C


class RS(O.State):
    phases = {}

    def status(self, ns):
        return {"phases": self.phases}


class RSys:
    def __init__(self):
        self.alive = []

    def read(self, p):
        try:
            return open(p, encoding="utf-8").read()
        except OSError:
            return None

    def win_cmdlines(self):
        return list(self.alive)


t = tempfile.mkdtemp(prefix="k23ovnstate_")
rsy = RSys()
S = RS(t, rsy)
good = {k: {"complete": True, "unchanged": True} for k in ("capture", "plan", "execute")}
good["score"] = {"complete": True}
RS.phases = good
ns = "test-k2-v1"
os.makedirs(C.ns_docs(ns, t), exist_ok=True)
open(C.journal_path(ns, "score", t), "w").write("{}\n")
json.dump({"sha256": C.sha256_file(C.journal_path(ns, "score", t))}, open(os.path.join(C.ns_docs(ns, t), "score.seal.json"), "w"))
logp = os.path.join(t, "sim", "final_test", "k23_run", f"{ns}-chain.log")
os.makedirs(os.path.dirname(logp), exist_ok=True)
DONE_LOG = (f"t CHAIN START {ns} (test, strict)\nt STAGE execute: execute\nt   execute session end rc=5\nt   execute session end rc=0\n"
            f"t STAGE execute rc=0\nt STAGE score rc=0\nt CHAIN DONE {ns} (next)\n")
open(logp, "w").write(DONE_LOG)
check("real chain_evidence: CHAIN DONE, sessions 5/0, phases + seal ok, nothing alive -> durable", S.chain_evidence(ns) == [])
open(logp, "w").write(DONE_LOG.replace("execute session end rc=0", "execute session end rc=75"))
check("real chain_evidence: a session rc 75 followed by a passed execute stage -> masked failure reported", any("masked" in w for w in S.chain_evidence(ns)))
open(logp, "w").write(DONE_LOG + f"t CHAIN START {ns} (test, strict)\n")
check("real chain_evidence: a later CHAIN START without CHAIN DONE -> not durable", any("CHAIN DONE" in w for w in S.chain_evidence(ns)))
open(logp, "w").write(DONE_LOG)
open(C.journal_path(ns, "score", t), "a").write("{}\n")
check("real chain_evidence: score.jsonl changed after its seal -> not durable", any("seal" in w for w in S.chain_evidence(ns)))
json.dump({"sha256": C.sha256_file(C.journal_path(ns, "score", t))}, open(os.path.join(C.ns_docs(ns, t), "score.seal.json"), "w"))
rsy.alive = ['9 "C:\\Program Files\\Git\\bin\\bash.exe" sim/final_test/k23_run/chain_k23_strict.sh test-k2-v1']
check("real chain_evidence: a strict chain of the namespace still alive -> not durable", any("alive" in w for w in S.chain_evidence(ns)))
rsy.alive = []
RS.phases = dict(good, execute={"complete": True, "unchanged": False})
check("real chain_evidence: a changed exec journal -> not durable", any("execute" in w for w in S.chain_evidence(ns)))
RS.phases = good

import k23_giso_waiver as GW


def mk_ns(ns):
    os.makedirs(C.ns_docs(ns, t), exist_ok=True)
    open(C.journal_path(ns, "score", t), "w").write("{}\n")
    json.dump({"sha256": C.sha256_file(C.journal_path(ns, "score", t))}, open(os.path.join(C.ns_docs(ns, t), "score.seal.json"), "w"))
    open(C.journal_path(ns, "capture", t), "w").write("{}\n")


def sealed(d):
    d = dict(d); d.pop("sha256", None); d["sha256"] = C.self_sha(d); return d


def ctx_fit(ns):
    json.dump({"freeze_grade": True, "capture_sha": C.sha256_file(C.journal_path(ns, "capture", t))}, open(os.path.join(C.ns_docs(ns, t), "context_fit.json"), "w"))


# Kitchen 2: no waiver is supported at all (Astra r3 A2)
ns = "rehearsal-k2-v1"
mk_ns(ns)
gp = os.path.join(C.ns_docs(ns, t), "g_iso.json")
json.dump(sealed({"status": "inconclusive", "n": 12, "n_registered": 12, "namespace": ns, "diff": []}), open(gp, "w"))
wp = os.path.join(t, *O.GISO_WAIVER.format(scene="FloorPlan15").split("/"))
os.makedirs(os.path.dirname(wp), exist_ok=True)
w = {"schema": "k23.decision/1", "id": "K23-GISO-WAIVER-FloorPlan15", "by": "Arash", "date": "2026-10-21", "said": "waive it",
     "g_iso_sha256": hashlib.sha256(open(gp, "rb").read()).hexdigest()}
w["sha256"] = C.self_sha(w)
json.dump(w, open(wp, "w"))
rr = S.rehearsal(ns)
check("real rehearsal (K2): an inconclusive G-iso is never waivable, even with an attributed self-hashed record -> giso_block",
      rr["giso_block"] and not rr["giso"]["waived"] and not rr["context_only"] and not rr["done"])
json.dump(sealed({"status": "pass", "n": 12, "n_registered": 12, "namespace": ns, "diff": []}), open(gp, "w"))
ctx_fit(ns)
rr = S.rehearsal(ns)
check("real rehearsal (K2): G-iso pass + context fit but NO 'CHAIN DONE' -> not done (no durable success evidence; Astra r3 A1)",
      not rr["done"] and any("successful-completion" in x for x in rr["why"]))
lp2 = os.path.join(t, "sim", "final_test", "k23_run", f"{ns}-chain.log")
open(lp2, "w").write(f"t CHAIN START {ns} (rehearsal, strict)\nt   execute session end rc=0\nt STAGE execute rc=0\nt CHAIN DONE {ns} (next)\n")
check("real rehearsal (K2): ... with 'CHAIN DONE' after the last START -> done", S.rehearsal(ns)["done"])
open(lp2, "w").write(f"t CHAIN START {ns} (rehearsal, strict)\nt   execute session end rc=4\nt STAGE execute rc=0\nt CHAIN DONE {ns} (next)\n")
check("real rehearsal (K2): a masked execute failure in the chain log -> not done", not S.rehearsal(ns)["done"])

# Kitchen 3: Arash's K23-GISO-WAIVER-FloorPlan17 with the freeze's conditions; completion only with the runner's context record
ns = "rehearsal-k3-v1"
mk_ns(ns)
gp = os.path.join(C.ns_docs(ns, t), "g_iso.json")
body = json.dumps(sealed({"status": "inconclusive", "n": 12, "n_registered": 12, "namespace": ns, "diff": [], "diff_components": {},
                          "components": {"request": {"absent_vs_present": 0}, "audit": {"absent_vs_present": 0}}}))
open(gp, "w", newline="\n").write(body)
ap = os.path.join(t, *GW.ARCHIVE.split("/"))
os.makedirs(os.path.dirname(ap), exist_ok=True)
open(ap, "w", newline="\n").write(body)
rr = S.rehearsal(ns)
check("real rehearsal (K3): inconclusive, no decision record -> giso_block", rr["giso_block"] and not rr["context_only"])
d = {"schema": GW.SCHEMA, "id": GW.DECISION_ID, "choice": GW.CHOICE, "rule": GW.RULE, "claim": GW.CLAIM, "scene": GW.SCENE, "namespace": GW.NS,
     "archive": GW.ARCHIVE, "g_iso_sha256": hashlib.sha256(body.encode()).hexdigest(), "by": "Arash", "date": "2026-10-20", "said": "Written waiver"}
os.makedirs(os.path.dirname(GW.decision_path(t)), exist_ok=True)
json.dump(sealed(d), open(GW.decision_path(t), "w"))
rr = S.rehearsal(ns)
check("real rehearsal (K3): Arash's valid waiver (archived result, diff []) + no context fit -> context_only", rr["context_only"] and rr["giso"]["waived"]
      and not rr["done"])
json.dump({"freeze_grade": False, "capture_sha": "x"}, open(os.path.join(C.ns_docs(ns, t), "context_fit.json"), "w"))
rr = S.rehearsal(ns)
check("real rehearsal (K3): a FAILED context report (freeze_grade false) -> ctx_state bad: neither done nor context_only (never re-run)",
      rr["ctx_state"] == "bad" and not rr["context_only"] and not rr["done"])
ctx_fit(ns)
rr = S.rehearsal(ns)
check("real rehearsal (K3): a context fit WITHOUT the runner's success record (e.g. run by hand) -> neither done nor re-run (context_only False)",
      not rr["done"] and not rr["context_only"])
rec = {"namespace": ns, "rc": 0, "context_fit_sha256": C.sha256_file(os.path.join(C.ns_docs(ns, t), "context_fit.json")),
       "g_iso_sha256": hashlib.sha256(open(gp, "rb").read()).hexdigest(), "waiver_decision_sha256": GW.decision_sha(t)}
os.makedirs(os.path.join(t, *O.OUT.split("/")), exist_ok=True)
json.dump(rec, open(os.path.join(t, *O.OUT.split("/"), f"context_resume_{ns}.json"), "w"))
check("real rehearsal (K3): ... with the runner's record of a context stage that exited 0 (matching shas) -> done", S.rehearsal(ns)["done"])
json.dump(dict(rec, rc=1), open(os.path.join(t, *O.OUT.split("/"), f"context_resume_{ns}.json"), "w"))
check("real rehearsal (K3): a record with rc 1 -> not done", not S.rehearsal(ns)["done"])
json.dump(rec, open(os.path.join(t, *O.OUT.split("/"), f"context_resume_{ns}.json"), "w"))
open(gp, "w", newline="\n").write(json.dumps(sealed({"status": "fail", "n": 12, "n_registered": 12, "namespace": ns, "diff": ["canon:x:en"]})))
rr = S.rehearsal(ns)
check("real rehearsal (K3): a 'fail' G-iso is never waivable (giso_block, not done)", rr["giso_block"] and not rr["done"])
shutil.rmtree(t, ignore_errors=True)

# ---------------------------------------------------------------------------------------------- 7. pure helpers + wrappers
LOG = ("2026-10-18 19:07:29 START FloorPlan15 yolo26s_coco\n\x1b[K 17/60 4.95G 0.67 640: 0% \r 1/1022\r"
       "2026-10-20 19:43:27 END rc=143\n2026-10-20 19:46:21 START FloorPlan15 yolo26s_coco\n")
g = O.parse_coco_log(LOG)
check("COCO log: last START after a pause END rc=143; running while fresh, resumable when stale", g["last"] == "START" and g["n_start"] == 2
      and O.coco_verdict(g, 10) == "running" and O.coco_verdict(g, 5000) == "resumable")
check("END rc=0 after the last START = ended_ok", O.coco_verdict(O.parse_coco_log(LOG + "2026-10-21 05:56:57 END rc=0\n"), 10) == "ended_ok")
check("a HEAT PAUSE takes precedence over a later END rc=0 (item 4)",
      O.coco_verdict(O.parse_coco_log(LOG + "2026-10-21 01:00:00 HEAT PAUSE: >=87 C\n2026-10-21 01:00:01 END rc=0\n"), 10) == "safety")
check("END rc 143 resumable; END rc 1 failed; no marker none",
      O.coco_verdict(O.parse_coco_log(LOG + "2026-10-21 01:00:01 END rc=143\n"), 10) == "resumable"
      and O.coco_verdict(O.parse_coco_log(LOG + "2026-10-21 01:00:01 END rc=1\n"), 10) == "failed" and O.coco_verdict(O.parse_coco_log("x"), 1) == "none")
s15 = SHA["FloorPlan15"]
check("Astra verdict: the exact four lines (blank lines / spaces ignored) -> ok", O.astra_verdict(review_file("\n  " + four("FloorPlan15").replace("\n", "\n\n")), "FloorPlan15", s15)[0])
for bad, why in ((four("FloorPlan15") + "Note: fine.", "a trailing note"), ("**" + four("FloorPlan15") + "**", "markdown"),
                 (four("FloorPlan15").replace("none", "the tokens gate first"), "a condition"), (four("FloorPlan17", s15), "another kitchen")):
    check(f"Astra verdict refuses {why}", not O.astra_verdict(review_file(bad), "FloorPlan15", s15)[0])
check("Astra verdict refuses no / two answer markers", not O.astra_verdict(four("FloorPlan15"), "FloorPlan15", s15)[0]
      and not O.astra_verdict(review_file(four("FloorPlan15")) + "\n" + O.ANSWER_MARK, "FloorPlan15", s15)[0])
check("the real capture review + module pass the capture gate in this tree", O.capture_review_problems(REPO) == [])
t = tempfile.mkdtemp(prefix="k23ovncap_")
for rel_ in (O.CAPTURE_FREEZE, O.CAPTURE_REVIEW):
    os.makedirs(os.path.dirname(os.path.join(t, *rel_.split("/"))), exist_ok=True)
    shutil.copyfile(os.path.join(REPO, *rel_.split("/")), os.path.join(t, *rel_.split("/")))
txt = open(os.path.join(t, *O.CAPTURE_REVIEW.split("/")), encoding="utf-8").read()
head, ans_ = txt.split(O.ANSWER_MARK)
open(os.path.join(t, *O.CAPTURE_REVIEW.split("/")), "w", encoding="utf-8").write(head + O.ANSWER_MARK + ans_.replace("RULING: ACCEPTED", "RULING: CHANGES REQUIRED"))
check("a capture review that is not ACCEPTED fails the capture gate", O.capture_review_problems(t) != [])
shutil.rmtree(t, ignore_errors=True)
check("chain log evidence: CHAIN DONE after the LAST CHAIN START only",
      O.chain_log_done("t CHAIN START rehearsal-k3-v1 (rehearsal)\nt CHAIN STOPPED at capture (rc 4)\nt CHAIN START rehearsal-k3-v1 (rehearsal)\n"
                       "t CHAIN DONE rehearsal-k3-v1 (next)\n", "rehearsal-k3-v1")
      and not O.chain_log_done("t CHAIN START test-k2-v1 (test)\nt CHAIN DONE test-k2-v1\nt CHAIN START test-k2-v1 (test)\n", "test-k2-v1"))
check("chain_log_problems: masking is per chain run (an earlier stopped run with a failed session is not masking)",
      O.chain_log_problems("t CHAIN START test-k2-v1 (test)\nt   execute session end rc=75\nt STAGE execute rc=75\nt CHAIN STOPPED at execute (rc 75)\n"
                           "t CHAIN START test-k2-v1 (test, strict)\nt   execute session end rc=0\nt STAGE execute rc=0\nt CHAIN DONE test-k2-v1\n",
                           "test-k2-v1", True) == [])
check("chain job prefixes as chain_k23.sh derives them", [O.chain_job(n) for n in ("rehearsal-k2-v1", "test-k3-v1")] == ["rk2", "tk3"])
for c, want in (('"C:\\Program Files\\Git\\bin\\bash.exe" sim/final_test/k23_run/chain_k23.sh rehearsal-k3-v1', True),
                ("powershell -ExecutionPolicy Bypass -File sim\\kitchen_v3\\k23\\k23_k3_practice_now.ps1", True),
                ('"C:\\Program Files\\Git\\bin\\bash.exe" -c "cat > p <<EOF ... chain_k23.sh rehearsal-k3-v1 ... EOF"', False),
                ("python sim/final_test/matrix/k23_run.py --phase plan --namespace rehearsal-k3-v1 --parallel 4", True),
                ("python sim/final_test/matrix/k23_run.py --phase status --namespace rehearsal-k3-v1", False),
                ("wsl -d Ubuntu-22.04 --cd /mnt/d/x -- bash sim/final_test/k23_run/run_k23.sh rk3-execute 172800 --phase execute --namespace rehearsal-k3-v1", True),
                ("python sim/final_test/k23_run/practice/launch.py practice-k2-v1", False),
                ("codex exec -m gpt-6-astra - chain_k23.sh rehearsal-k3-v1", False)):
    check(f"busy_match {'counts' if want else 'ignores'}: {c[:60]}", O.busy_match(c) is want)
check("runs_ns finds a K2 test chain process, strict or not (for 'no chain alive')", O.runs_ns('bash sim/final_test/k23_run/chain_k23.sh test-k2-v1', ("test-k2-v1",))
      and O.runs_ns('bash sim/final_test/k23_run/chain_k23_strict.sh test-k2-v1', ("test-k2-v1",))
      and O.runs_ns("wsl -d Ubuntu-22.04 --cd /mnt/d/x -- bash sim/final_test/k23_run/run_k23_strict.sh tk2-execute 1 --phase execute --namespace test-k2-v1", ("test-k2-v1",))
      and not O.runs_ns('bash sim/final_test/k23_run/chain_k23.sh test-k3-v1', ("test-k2-v1",)))
check("evidence never lists applicability rows; safe_path refuses them and the preregistration",
      not any("applicability" in p for s_ in ("P2", "R2", "S2", "F2", "T2", "S3", "F3", "T3") for p in O.evidence(s_))
      and not O.safe_path("docs/final-test/k23/applicability/FloorPlan15/runs.jsonl") and not O.safe_path(O.PREREG) and not O.safe_path("../x"))
rs = open(os.path.join(HERE, "practice", "run_strict.sh"), encoding="utf-8").read()
check("run_strict.sh: one try (no loop), flock -w (bounded, never stolen), the try's rc returned, 73 on a lock timeout, inner_strict.sh",
      not re.search(r"^\s*(while|until|for)\b", rs, re.M) and "flock -w" in rs and 'exit "$rc"' in rs and "exit 73" in rs
      and "practice/inner_strict.sh" in rs and not any("--steal" in ln for ln in rs.splitlines() if not ln.lstrip().startswith("#")))
ls = open(os.path.join(HERE, "practice", "launch_strict.py"), encoding="utf-8").read()
check("launch_strict.py: launch.py's own preflight (readiness, Ollama) then run_strict.sh; never run.sh", "L.KP.readiness" in ls and "L.ollama_ok()" in ls
      and "run_strict.sh" in ls and "practice/run.sh" not in ls)
import importlib.util
spec = importlib.util.spec_from_file_location("k23_launch_strict", os.path.join(HERE, "practice", "launch_strict.py"))
LS = importlib.util.module_from_spec(spec); spec.loader.exec_module(LS)
check("launch_strict.py refuses a non-K23 namespace / extra arguments before any check (rc 2)",
      LS.main(["practice-v2"]) == 2 and LS.main(["practice-k2-v1", "--limit"]) == 2)

# ---------------------------------------------------------------------------------------------- 8. the dry run runs nothing
root, st, fs, r = setup()
rc_dry = O.dry_run(root, fs, st)
mut = [c for c, _ in fs.calls if (c[0] == "git" and c[1] in ("add", "commit")) or c[0] == O.GIT_BASH
       or (c[0] == sys.executable and (c[1].endswith("launch_strict.py") or "--write" in c or "--presign" in c or c[1].endswith("astra_call.py")))
       or (c[0] == "wsl" and c[-1] != "true")]
check("the dry run starts, writes and commits nothing", not mut and not fs.spawned and fs.commits == 0)
check("the dry run in a fake root reports blocking problems (tools missing) with rc 1", rc_dry == 1)
shutil.rmtree(root, ignore_errors=True)
ps = open(os.path.join(HERE, "k23_overnight.ps1"), encoding="utf-8").read()
check("k23_overnight.ps1: -DryRun -> --dry-run, otherwise --run; hidden launch line documented; no hang relaunch option",
      "--dry-run" in ps and '"--run"' in ps and "-WindowStyle Hidden" in ps and "HangRelaunch" not in ps)
check("--root is refused with --run", O.main(["--run", "--root", "x"]) == 2)
O.OLLAMA_EXE = O_OLLAMA
print(f"{ok}/{ok + fail}"); sys.exit(1 if fail else 0)
