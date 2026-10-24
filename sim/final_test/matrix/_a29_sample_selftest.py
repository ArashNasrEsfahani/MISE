"""Offline gate for a29_sample.py (A29 item 2 sampler)."""
import os, sys, json, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import a29_sample as S
import common as C
ok = fail = 0
def check(name, cond):
    global ok, fail
    if cond: ok += 1
    else: fail += 1; print("FAIL", name)
# synthetic: 6 families, levels
fam = {f"i{n}": f"f{n % 6}" for n in range(30)}
lev = {f"i{n}": ("L1", "L2", "L3")[n % 3] for n in range(30)}
pool = sorted(fam)
o1 = S.ordered_sample("FloorPlan15", pool, fam, lev)
o2 = S.ordered_sample("FloorPlan15", list(reversed(pool)), fam, lev)
o3 = S.ordered_sample("FloorPlan17", pool, fam, lev)
check("deterministic + input-order free", o1 == o2)
check("kitchens differ", o1 != o3)
check("permutation of pool", sorted(o1) == pool and len(set(o1)) == 30)
check("first 6 = one per family", len({fam[i] for i in o1[:6]}) == 6)
# level balance tracks shares at every prefix after the family stage
for t in range(6, 30):
    m = collections.Counter(lev[i] for i in o1[:t + 1])
    check(f"balance t{t}", all(abs(m[l] - (t + 1) / 3) <= 6.5 for l in ("L1", "L2", "L3")))
try:
    S.ordered_sample("FloorPlan1", pool, fam, lev); check("refuse FP1", False)
except ValueError: check("refuse FP1", True)
try:
    S.ordered_sample("FloorPlan15", pool + ["zz"], fam, lev); check("refuse unknown", False)
except ValueError: check("refuse unknown", True)
# family order: biggest family first
fam2 = {"a1": "A", "b1": "B", "b2": "B", "b3": "B", "c1": "C", "c2": "C"}
lev2 = {k: "L1" for k in fam2}
o = S.ordered_sample("FloorPlan15", list(fam2), fam2, lev2)
check("family order B,C,A", [fam2[i] for i in o[:3]] == ["B", "C", "A"])
# real manifest: all-applicable fake
man = C.load_manifest()
st = {i: "applicable" for i in man["test"]}
rep = S.build("FloorPlan15", {"kitchen": "FloorPlan15", "items": st})
check("real pool 125", rep["n_pool"] == 125 and rep["primary_note"] is None)
check("nested candidates", rep["candidates"]["first_50"] == rep["candidates"]["first_60"][:50] == rep["order"][:50])
check("presentations 100/120/250", rep["presentations"] == {"first_50": 100, "first_60": 120, "all": 250})
st2 = dict(st); drop = sorted(st)[:10]
for i in drop: st2[i] = "inapplicable"
rep2 = S.build("FloorPlan15", {"kitchen": "FloorPlan15", "items": st2})
check("shortfall reported", rep2["n_pool"] == 115 and "shortfall 10" in rep2["primary_note"] and not set(drop) & set(rep2["order"]))
st3 = dict(st); st3[sorted(st)[0]] = "unresolved"
try: S.build("FloorPlan15", {"kitchen": "FloorPlan15", "items": st3}); check("refuse unresolved", False)
except ValueError: check("refuse unresolved", True)
st4 = dict(st); del st4[sorted(st)[0]]
try: S.build("FloorPlan15", {"kitchen": "FloorPlan15", "items": st4}); check("refuse missing", False)
except ValueError: check("refuse missing", True)
try: S.build("FloorPlan15", {"kitchen": "FloorPlan17", "items": st}); check("refuse kitchen mismatch", False)
except ValueError: check("refuse kitchen mismatch", True)
# exact tie (Astra r1 item 1): pool counts (1,4,1), selected (1,2,0), position 4 -> L2 and L3 both 2/3 -> L2
check("exact tie -> earlier level", S.next_level(["L1", "L2", "L3"], {"L1": 1, "L2": 4, "L3": 1}, 6, {"L1": 1, "L2": 2, "L3": 0}, 4) == "L2")
check("deficit wins", S.next_level(["L1", "L2", "L3"], {"L1": 1, "L2": 4, "L3": 1}, 6, {"L1": 1, "L2": 1, "L3": 0}, 4) == "L2")
check("empty level skipped", S.next_level(["L1", "L3"], {"L1": 1, "L2": 4, "L3": 1}, 6, {"L1": 1, "L2": 1, "L3": 0}, 4) == "L3")
st5 = dict(st); st5[sorted(st)[0]] = "infra_error"
try: S.build("FloorPlan15", {"kitchen": "FloorPlan15", "items": st5}); check("refuse other status", False)
except ValueError: check("refuse other status", True)
check("no --out flag", 'add_argument("--out"' not in open(S.__file__, encoding="utf-8").read())
check("self sha", rep["sha256"] == C.self_sha(rep))
print(f"{ok}/{ok + fail}")
sys.exit(1 if fail else 0)
