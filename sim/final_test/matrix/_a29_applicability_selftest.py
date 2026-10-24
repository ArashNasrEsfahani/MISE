"""Offline, hermetic gate for a29_applicability.py (fake executor, temp files; never starts a simulator, never calls main() live)."""
import hashlib, json, os, sys, tempfile, types
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import a29_applicability as A
import common as C
ok = fail = 0


def check(n, c):
    global ok, fail
    if c: ok += 1
    else: fail += 1; print("FAIL", n)


def raises(f, exc=Exception):
    try:
        f()
    except exc:
        return True
    return False


R = lambda **k: types.SimpleNamespace(**{"inapplicable": None, "goal_ok": False, "extra": {}, **k})
# -- classify
check("absent", A.classify(R(inapplicable="object does not exist: Toaster"), None, 2)[:2] == ("inapplicable", "object_absent"))
check("setup -> review (never auto-excluded)", A.classify(R(inapplicable="setup failed -- x"), None, 2)[:2] == ("unresolved", "setup_failed_review"))
check("goal true", A.classify(R(inapplicable="goal already true before acting"), None, 2)[:2] == ("inapplicable", "goal_true_at_setup"))
check("applicable", A.classify(R(goal_ok=True), None, 2)[:2] == ("applicable", "goal_reached"))
c = A.classify(R(goal_ok=False, extra={"steps": [{"ok": True}, {"ok": False, "cause": "episode_budget", "under": "episode_wall"}]}), None, 2)
check("ref failure keeps cause/under", c[:2] == ("inapplicable", "reference_execution_failure") and "episode_wall" in c[2])
check("wall overrun after the last step = failure", A.classify(R(goal_ok=True, extra={"_wall_s": 950.0}), None, 2)[:2]
      == ("inapplicable", "reference_execution_failure"))
check("step cap", A.classify(R(goal_ok=True), None, 13)[:2] == ("inapplicable", "step_cap"))
check("infra", A.classify(None, "RuntimeError: x", 2)[0] == "unresolved")
check("invariant -> INVALID", A.classify(None, "LiveInvariant: x", 2, invariant=True)[0] == "INVALID")
check("final: first decisive", A.final_class([{"class": "unresolved"}, {"class": "applicable", "cause": "goal_reached"}])["class"] == "applicable")
check("final: resolution wins", A.final_class([{"class": "unresolved", "cause": "setup_failed_review"}],
                                              {"decision": "inapplicable", "why": "x"})["class"] == "inapplicable")

# -- verify_freeze on a temp root
root = tempfile.mkdtemp()
orig_paths = C.paths
C.paths = lambda r=None: dict(orig_paths(None), root=root) if r == root else orig_paths(r)
os.makedirs(os.path.join(root, "sim", "kitchen_v3", "FloorPlan15"))
w = os.path.join(root, "w.pt"); open(w, "wb").write(b"weights")
rec = os.path.join(root, "sim", "kitchen_v3", "FloorPlan15", "frozen_weights.sha256")
open(rec, "w").write(hashlib.sha256(b"weights").hexdigest() + " w.pt\n")
check("freeze verified", A.verify_freeze("FloorPlan15", root)["files"] == {"w.pt": hashlib.sha256(b"weights").hexdigest()})
open(w, "wb").write(b"changed")
check("freeze mismatch refused", raises(lambda: A.verify_freeze("FloorPlan15", root), RuntimeError))
check("no freeze refused", raises(lambda: A.verify_freeze("FloorPlan17", root), RuntimeError))
C.paths = orig_paths


# -- run(): scripted fake executor
class LiveInvariant(RuntimeError):
    pass


script = {"i1": [R(goal_ok=True)], "i2": [RuntimeError("crash"), R(goal_ok=False)], "i3": [RuntimeError("a")] * 9,
          "i4": [R(inapplicable="goal already true before acting")], "i5": [R(inapplicable="setup failed -- y"), R(goal_ok=True)]}
calls = {}


class Fake:
    def execute(self, req):
        iid = req.item["id"]; k = calls.get(iid, 0); calls[iid] = k + 1
        x = script[iid][k]
        if isinstance(x, Exception):
            raise x
        return x


A.request = lambda kitchen, item, n, rnd, lang="en": types.SimpleNamespace(item=item)
items = {i: {"id": i, "level": "L1", "plan": [{"action": "pickup"}]} for i in script}
d = tempfile.mkdtemp(); rp = os.path.join(d, "runs.jsonl")
bind = {"kind": "binding", "version": A.VERSION, "kitchen": "FloorPlan15", "code": {"x": "1"}}
fake = lambda k: Fake()
by = A.run("FloorPlan15", sorted(script), items, rp, 1, bind, {}, executor_factory=fake)
fc = lambda i: A.final_class(by[i])
check("i1 one run", calls["i1"] == 1 and fc("i1")["class"] == "applicable")
check("i2 infra rerun then valid failure", calls["i2"] == 2 and fc("i2")["cause"] == "reference_execution_failure")
check("i3 3 attempts in round 1", calls["i3"] == 3 and fc("i3")["class"] == "unresolved")
check("i4 goal true at setup", fc("i4")["cause"] == "goal_true_at_setup")
check("i5 setup failure waits for review", calls["i5"] == 1 and fc("i5")["cause"] == "setup_failed_review")
by = A.run("FloorPlan15", sorted(script), items, rp, 1, bind, {}, executor_factory=fake)
check("resume of round 1: nothing re-run (round full / decided / review)", calls == {"i1": 1, "i2": 2, "i3": 3, "i4": 1, "i5": 1})
check("round 3 refused (last round 1)", raises(lambda: A.run("FloorPlan15", ["i3"], items, rp, 3, bind, {}, executor_factory=fake), RuntimeError))
by = A.run("FloorPlan15", sorted(script), items, rp, 2, bind, {"i5": {"decision": "retry", "why": "sim glitch"}}, executor_factory=fake)
check("round 2: i3 gets 3 more, i5 retried by decision", calls["i3"] == 6 and calls["i5"] == 2 and A.final_class(by["i5"])["class"] == "applicable")
check("binding mismatch refused", raises(lambda: A.run("FloorPlan15", ["i1"], items, rp, 2, dict(bind, code={"x": "2"}), {},
                                                      executor_factory=fake), RuntimeError))
script["i6"] = [LiveInvariant("oracle read")]
items["i6"] = {"id": "i6", "level": "L2", "plan": []}
check("INVALID stops the run", raises(lambda: A.run("FloorPlan15", ["i6", "i1"], items, rp, 2, bind, {}, executor_factory=fake), A.Invalid))
A.run("FloorPlan15", ["i6"], items, rp, 2, bind, {}, executor_factory=fake)
check("INVALID never retried", calls["i6"] == 1)
_b, by = A.load_runs(rp)
check("rows carry goal truth + round", all("round" in a and "goal_true_after_plan" in a for att in by.values() for a in att))
doc = A.summarize("FloorPlan15", by, sorted(script), {i: "f" + i for i in script}, {i: "L1" for i in script}, _b, {})
check("summary counts", doc["counts"] == {"applicable": 2, "inapplicable": 2, "unresolved": 2} and doc["binding"] == _b)
s1 = A.equivalence_sample("FloorPlan15", [f"x{i}" for i in range(40)])
check("equivalence sample: 10, deterministic, per kitchen", len(s1) == 10 and s1 == A.equivalence_sample("FloorPlan15", [f"x{i}" for i in range(40)])
      and s1 != A.equivalence_sample("FloorPlan17", [f"x{i}" for i in range(40)]))
print(f"{ok}/{ok + fail}"); sys.exit(1 if fail else 0)
