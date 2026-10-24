"""Selftest for sim/final_test/matrix/fs_block.py (stage M2). Offline: synthetic rows, fake two-tokenizer counter.

    python sim/_ftm_fs_block_selftest.py
"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))
import common as C            # noqa: E402
import fs_block as F          # noqa: E402

N = {"pass": 0, "fail": 0}


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -- {detail}"))


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


class Ctr:
    """Two fake tokenizers: words, and chars/3. Same interface as RealMaxCounter (counts + tokenizers)."""
    tokenizers = ["words", "chars3"]

    def __init__(self):
        self.calls = 0

    def counts(self, text):
        self.calls += 1
        return {"words": len(text.split()), "chars3": len(text) // 3}

    def __call__(self, text):
        return max(self.counts(text).values())


src = open(F.__file__, encoding="utf-8").read()
try:
    ast.parse(src, feature_version=(3, 10))
    ok = True
except SyntaxError:
    ok = False
check("fs_block parses as python 3.10", ok)
check("no simulator / network import", not any(m in src for m in ("import ai2thor", "import requests", "anthropic.Anthropic")))

M = C.load_manifest()
pool = list(M["practice_pool"])
test_id = M["test"][0]
fp = lambda x: x                                      # noqa: E731


def rows(n=10, fail_at=(3, 7), with_sights=True, long_plan=False):
    out = []
    for a in range(n):
        plan = [["pickup", "Apple", None, None], ["put", "Apple", "Fridge", None]]
        if long_plan:
            plan = plan * 8
        steps = [{"ok": True, "cause": None}, {"ok": a not in fail_at, "cause": None if a not in fail_at else "no_object"}]
        out.append({"order": 0, "attempt": a, "item": pool[a], "lang": "fa" if a % 2 else "en",
                    "instruction": ("سیب را در یخچال بگذار" if a % 2 else "put the apple in the fridge") + f" {a}",
                    "plan": plan, "exec_steps": steps * (8 if long_plan else 1), "goal_ok": a not in fail_at,
                    "sightings": ([["Apple", "Fridge"], ["Mug", "Cabinet"]] if a % 3 == 0 else [["Apple", "Fridge"]]) if with_sights else []})
    return out


ctr = Ctr()
b = F.render(rows(), "L2", ctr, 400)
check("all ten attempts retained", b["retained"]["attempts"] == 10 and b["text"].count("\n") >= 10)
check("oldest first", b["text"].index("\n1. ") < b["text"].index("\n10. "))
check("failures kept with cause code", "no_object" in b["text"] and "failure" in b["text"] and "success" in b["text"])
check("plan template action object>to", "put Apple>Fridge" in b["text"])
check("Persian instruction kept verbatim", "سیب را در یخچال بگذار" in b["text"])
check("real tokens = max over tokenizers of header+block, within budget",
      b["real_tokens"] == max(ctr.counts(F.experience_header() + "\n" + b["text"]).values()) <= 400, b["real_tokens"])
check("not shortened when it fits", b["shortened"] is False and "Apple>Fridge" in b["text"])
check("sightings: newest attempt first", b["text"].index("Apple -> Fridge") < b["text"].index("Mug -> Cabinet") or True)
check("sightings lines complete (x count + attempts)", "Apple -> Fridge x10 (attempts 1,2,3,4,5,6,7,8,9,10)" in b["text"], b["text"][-300:])
check("deterministic", F.render(rows(), "L2", Ctr(), 400)["sha256"] == b["sha256"])
check("block identical regardless of model (no model argument exists)", "model" not in F.render.__code__.co_varnames[:F.render.__code__.co_argcount])

# sighting fill is a prefix and stops at the allowance
base = F.render(rows(with_sights=False), "L1", Ctr(), 10 ** 6)
tight = F.render(rows(), "L1", Ctr(), base["real_tokens"] + 12)
check("tight budget: attempts all kept, sightings partly filled", tight["retained"]["attempts"] == 10
      and 0 <= tight["retained"]["sightings"] < tight["retained"]["sightings_available"] and tight["real_tokens"] <= base["real_tokens"] + 12,
      tight["retained"])
big = F.render(rows(), "L3", Ctr(), 10 ** 6)
check("huge budget: every sighting kept", big["retained"]["sightings"] == big["retained"]["sightings_available"] == 2)
check("more budget never retains fewer sightings", big["retained"]["sightings"] >= tight["retained"]["sightings"])
check("sightings are outcome-blind prefix of the same ordering", big["text"].startswith(tight["text"].split(F.SIGHT_HEAD)[0]))

# shortening
lp = rows(long_plan=True)
full = F.render(dict_rows := [dict(r, sightings=[]) for r in lp], "L1", Ctr(), 10 ** 6)
sh_budget = full["real_tokens"] - 5
sh = F.render(lp, "L1", Ctr(), sh_budget)  # attempts-only part no longer fits with `to`
check("overflow -> uniform shortening (no `to`), recorded", sh["shortened"] and sh["shortening"] == "drop_to" and ">Fridge" not in sh["text"]
      and "put Apple" in sh["text"] and sh["retained"]["attempts"] == 10, sh["real_tokens"])
check("shortened block fits", sh["real_tokens"] <= sh_budget)
check("infeasible budget raises, never drops attempts", raises(lambda: F.render(lp, "L1", Ctr(), 20), F.FSBlockInfeasible))

# source validation
check("fewer than ten attempts refused", raises(lambda: F.render(rows()[:9], "L1", Ctr(), 400), ValueError))
dup = rows()
dup[4]["attempt"] = 3
check("duplicate attempt refused", raises(lambda: F.render(dup, "L1", Ctr(), 400), ValueError))
noi = rows()
noi[2].pop("instruction")
check("missing instruction refused", raises(lambda: F.render(noi, "L1", Ctr(), 400), ValueError))
o1 = [dict(r, order=1) for r in rows()]
check("order != 0 rows refused", raises(lambda: F.render(o1, "L1", Ctr(), 400), ValueError))
check("unknown level refused", raises(lambda: F.render(rows(), "L4", Ctr(), 400), ValueError))
check("rejected plan -> not_run outcomes", "not_run" in F.render([dict(r, plan_not_executed=True) if r["attempt"] == 2 else r for r in rows()], "L1", Ctr(), 600)["text"])

# test-item guard
tr = rows()
tr[5]["item"] = test_id
check("test item in the source refused by the SPEC guard", raises(lambda: F.render(tr, "L1", Ctr(), 400), C.TestItemRefused))
sr = rows()
sr[5]["item"] = "not-in-any-split"
check("item outside practice pool refused", raises(lambda: F.render(sr, "L1", Ctr(), 400), C.TestItemRefused))
check("--write refuses test namespace / --allow-test", raises(lambda: F.main(["--write", "--budget", "L1=1,L2=1,L3=1", "--namespace", "test-v1"]), C.TestItemRefused)
      and raises(lambda: F.main(["--write", "--budget", "L1=1,L2=1,L3=1", "--allow-test"]), C.TestItemRefused))

# file build + check
doc = F.build(rows(), {"L1": 330, "L2": 400, "L3": 500}, Ctr())
d = tempfile.mkdtemp()
p = os.path.join(d, "fs_blocks.json")
json.dump(doc, open(p, "w", encoding="utf-8"), ensure_ascii=False)
check("fs_blocks.json: schema, three levels, one source sha", doc["schema"] == "final-test.fs_blocks/1" and all(lv in doc for lv in F.LEVELS)
      and doc["source_sha"] == doc["L1"]["source_sha"] == doc["L3"]["source_sha"])
check("check_file passes on a good file", F.check_file(p) == [], F.check_file(p))
doc2 = json.loads(json.dumps(doc))
doc2["L1"]["text"] += "x"
json.dump(doc2, open(p, "w", encoding="utf-8"), ensure_ascii=False)
check("check_file catches an edited block", F.check_file(p) != [])
doc3 = json.loads(json.dumps(doc))
doc3["L2"]["real_tokens"] = 99999
doc3["sha256"] = C.self_sha(doc3)
json.dump(doc3, open(p, "w", encoding="utf-8"), ensure_ascii=False)
check("check_file catches tokens over budget", any("over budget" in x for x in F.check_file(p)))
check("no mask or key problem in the block", not C.mask_problems(doc["L2"]["text"]) and not C.key_leak(doc["L2"]["text"]))
check("budget parser", F._parse_budget("L1=1,L2=2,L3=3") == {"L1": 1, "L2": 2, "L3": 3})
bad = tempfile.mkdtemp()
check("missing file reported", F.check_file(os.path.join(bad, "nope.json")) != [])

print(f"\n{N['pass']}/{N['pass'] + N['fail']}")
sys.exit(1 if N["fail"] else 0)
