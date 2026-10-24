"""Offline gate for k23_leakage.py (the decision-gated own-fragment rule of the practice-snapshot leakage gate), on synthetic snapshot
directories; the registered practice_k.leakage_gate always runs first and is never edited."""
import json, os, shutil, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C
import practice_k as PK
import k23_leakage as KL
ok = fail = 0


def check(n, c):
    global ok, fail
    if c: ok += 1
    else: fail += 1; print("FAIL", n)


man = C.load_manifest()
items = {i["id"]: i for i in json.load(open(C.paths()["dataset"], encoding="utf-8"))["items"]}
needles = PK.test_needles(man, items)
P8 = "g-l2-slice-bread-tomato"
P1 = "g-l1-place-mug-counter-top"
refused = set(man["test"]) | set(man.get("test_uncapped_not_scored") or [])
expected = [(P1, "en", items[P1]["en"], None), (P8, "en", items[P8]["en"], None)]
allow = {"rows_file": "practice_rows.jsonl", "refused": refused, "expected": expected}
# a test sentence that is NOT a fragment of either practice instruction (a genuine leak if it appears)
own_norm = [PK.norm_text(items[P1]["en"]), PK.norm_text(items[P8]["en"])]
foreign = next((i, items[i]["en"]) for i in sorted(man["test"]) if len(PK.norm_text(items[i]["en"])) >= PK.MIN_NEEDLE
               and not any(PK.norm_text(items[i]["en"]) in on for on in own_norm) and "slice" not in items[i]["en"].lower())


def snap(say8, extra_file=None, instr8=None, say1="Placing the mug."):
    d = tempfile.mkdtemp(prefix="k23leaksnap_")
    rows = [{"namespace": "practice-k3-v1", "order": 0, "attempt": 1, "item": P1, "lang": "en", "instruction": items[P1]["en"], "say": say1},
            {"namespace": "practice-k3-v1", "order": 0, "attempt": 2, "item": P8, "lang": "en", "instruction": instr8 or items[P8]["en"],
             "say": say8, "block": {"text": "This kitchen:\nSteps that work here:"}}]
    with open(os.path.join(d, "practice_rows.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    if extra_file:
        open(os.path.join(d, "fragments.json"), "w", encoding="utf-8").write(extra_file)
    return d


SAY = "I will slice the bread and then the tomato."     # practice-k3-v1 attempt 8's reply, verbatim
d = snap(SAY)
reg = PK.leakage_gate(d, needles, allow=allow)
check("the registered gate fails on the K3 case (slice the bread in the say field)", not reg["ok"]
      and {h["item"] for h in reg["hits"]} >= {"g-l2-slice-bread"} and all(h["kind"] == "instruction" for h in reg["hits"]))
res = KL.gate(d, needles, allow)
check("the own-fragment rule passes it, with exactly the say field of the practice row audited", res["ok"] is True
      and [(a["row"], a["field"], a["item"]) for a in res["own_fragment_exempt"]] == [(2, "say", P8)])
check("the result keeps the registered first run, the rule id and the decision sha field", res["registered_first"]["ok"] is False
      and res["rule"] == KL.DECISION_ID and "decision_sha256" in res)
check("the snapshot files are not modified (the second run used a temporary copy)",
      json.loads(open(os.path.join(d, "practice_rows.jsonl"), encoding="utf-8").read().splitlines()[1])["say"] == SAY)
shutil.rmtree(d)

d = snap(SAY + " " + foreign[1])
check("a genuine test sentence beside the own fragment in the same field still fails", not KL.gate(d, needles, allow)["ok"]); shutil.rmtree(d)
# Astra K23-PRACTICE-LEAK-r1: a LONGER test sentence that contains the own fragment must still leak (case / punctuation / spacing variants)
over = next((i, items[i]["en"]) for i in sorted(man["test"]) if "slice the bread" in PK.norm_text(items[i]["en"])
            and PK.norm_text(items[i]["en"]) != "slice the bread" and not any(PK.norm_text(items[i]["en"]) in on for on in own_norm))
for variant in (over[1].upper(), over[1].replace(",", " ,  ").replace(".", " !"), "Ok:  " + over[1].lower()):
    d = snap(variant)
    check(f"an overlapping longer test sentence ({over[0]}) in a variant form still fails", not KL.gate(d, needles, allow)["ok"])
    shutil.rmtree(d)
d = snap(SAY + " (" + sorted(man["test"])[0] + ")")
check("a whole test id in the same field still fails (registered id rule)", not KL.gate(d, needles, allow)["ok"]); shutil.rmtree(d)
d = snap("ok", say1=SAY)
check("a fragment of ANOTHER valid row's practice instruction is exempt too (practice-own text of the same snapshot)",
      KL.gate(d, needles, allow)["ok"] is True)
shutil.rmtree(d)
d = snap(SAY, extra_file=json.dumps({"note": SAY}))
check("the same fragment in another file of the snapshot still fails (only row fields are exempt)", not KL.gate(d, needles, allow)["ok"])
shutil.rmtree(d)
d = snap(SAY, instr8=items[P8]["en"] + " ")
check("a row that is not A30-valid gets no exemption (the registered gate's exemption_invalid stands)", not KL.gate(d, needles, allow)["ok"])
shutil.rmtree(d)
d = snap("Taking the butter knife and slicing the bread and tomato.")    # Kitchen 1's attempt-8 wording
r1 = KL.gate(d, needles, allow)
check("a snapshot the registered gate passes is returned unchanged (Kitchen 1's wording: no audit)", r1["ok"] is True
      and "own_fragment_exempt" not in r1)
shutil.rmtree(d)

# ---- decision gate + install
fx = tempfile.mkdtemp(prefix="k23leakfx_")
check("no decision -> install() patches nothing", KL.install(fx) is False and not getattr(PK.leakage_gate, "k23_leakage", False))
os.makedirs(os.path.dirname(KL.decision_path(fx)))
dec = {"schema": KL.SCHEMA, "id": KL.DECISION_ID, "choice": KL.CHOICE, "rule": KL.RULE, "recovery": KL.RECOVERY, "by": "Arash", "date": "2026-10-20", "said": "x"}
nore = dict(dec); nore.pop("recovery"); nore["sha256"] = C.self_sha(nore)
json.dump(nore, open(KL.decision_path(fx), "w"))
check("a decision without the recovery disposition is refused", KL.decision_problems(fx) != [])
bad = dict(dec, by="someone"); bad["sha256"] = C.self_sha(bad)
json.dump(bad, open(KL.decision_path(fx), "w"))
check("a decision not by Arash is refused", KL.decision_problems(fx) != [])
dec["sha256"] = C.self_sha(dec)
json.dump(dec, open(KL.decision_path(fx), "w"))
check("a valid decision", KL.decision_problems(fx) == [])
orig = PK.leakage_gate
check("install() with the decision patches practice_k.leakage_gate in this process", KL.install(fx) is True
      and getattr(PK.leakage_gate, "k23_leakage", False) and PK.leakage_gate.k23_orig is orig)
PK.leakage_gate = orig
leak_ok = {"rule": KL.DECISION_ID, "decision_sha256": KL.decision_sha(fx), "ok": True, "own_fragment_exempt": [{"row": 8}]}
check("freeze: an own-fragment pass naming the bound decision -> []", KL.snapshot_rule_problems(leak_ok, fx) == [])
check("freeze: another decision sha -> problem", KL.snapshot_rule_problems(dict(leak_ok, decision_sha256="0" * 64), fx) != [])
check("freeze: a registered (plain) pass needs no decision", KL.snapshot_rule_problems({"ok": True}, fx) == [])
os.remove(KL.decision_path(fx))
check("freeze: an own-fragment pass without a decision record -> problem", KL.snapshot_rule_problems(leak_ok, fx) != [])
shutil.rmtree(fx, ignore_errors=True)
# --diagnose in a FRESH process admits the K23 namespace itself (coordinator 2026-10-20: it raised PracticeError without k23_practice.admit)
import subprocess, treatment as T
dd = tempfile.mkdtemp(prefix="k23leakdiag_")
json.dump({"k": 1, "order": 0}, open(os.path.join(dd, PK.MANIFEST_NAME), "w"))
open(os.path.join(dd, T.ROWS_FILE), "w").close()
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
cp = subprocess.run([sys.executable, os.path.join(HERE, "k23_leakage.py"), "--diagnose", dd, "--namespace", "practice-k3-v1"],
                    capture_output=True, text=True, encoding="utf-8", env=env)
check("--diagnose admits a K23 practice namespace in a fresh process (no PracticeError)", cp.returncode == 0
      and "registered" in json.loads(cp.stdout) and "PracticeError" not in cp.stderr)
cp = subprocess.run([sys.executable, os.path.join(HERE, "k23_leakage.py"), "--diagnose", dd], capture_output=True, text=True, env=env)
check("--diagnose without --namespace is refused (rc 2)", cp.returncode == 2)
shutil.rmtree(dd, ignore_errors=True)
src = open(os.path.join(HERE, "k23_practice.py"), encoding="utf-8").read()
check("k23_practice (live) installs it", "KL.install()" in src)
check("the freeze binds the decision record and checks the snapshot's rule", "K23-PRACTICE-LEAK.json" in open(os.path.join(HERE, "k23_freeze_agent.py"),
      encoding="utf-8").read() and "snapshot_rule_problems" in open(os.path.join(HERE, "k23_freeze_agent.py"), encoding="utf-8").read())
print(f"{ok}/{ok + fail}"); sys.exit(1 if fail else 0)
