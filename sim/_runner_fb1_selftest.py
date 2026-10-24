"""FREEZE-GATES step 2 gate: the ft-v2 repair-feedback contract fb-1 in planners/runner.py (Astra FREEZE-GATES Q1 a-c).
Offline: Windows python, repo root; no simulator, no LLM, no network (a fake provider; the token evidence is read from its saved files).

    python sim/_runner_fb1_selftest.py
"""
from __future__ import annotations

import json
import os
import random
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                     # pragma: no cover
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))

from planners import runner as R                                      # noqa: E402
from planners.contract import PlannerContract, REPAIR, REPAIR_V2, problems_text, parse_reply   # noqa: E402
from planners import validator_v2 as V                                # noqa: E402
import honest_skills as hs                                            # noqa: E402
import treatment as T                                                 # noqa: E402

N = {"pass": 0, "fail": 0}
FOLDED = set(chr(o) for o in range(32, 127)) | {"\n"}
DIAG = os.path.join(HERE, "final_test", "diag", "freeze_gates")


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  -- " + str(detail)[:400]))


print("== 1 constants")
check("1.1 contract fb-1, cap 600 chars incl. marker, 2 repairs under ft-v2", R.FEEDBACK_CONTRACT == "fb-1" and R.PROBLEM_TEXT_CAP_CHARS == 600
      and T.FT2["repairs"] == 2)
check("1.2 token bounds: gemma 600 (1 per char), anthropic / codex / max 1,200 (2 per char)",
      (R.problem_token_bound("gemma"), R.problem_token_bound("anthropic"), R.problem_token_bound("codex"), R.problem_token_bound("max"))
      == (600, 1200, 1200, 1200))

print("== 2 ASCII fold")
check("2.1 printable ASCII + newline unchanged", R.fold_ascii("- step 2: pickup Bowl\n- x ~`{}") == "- step 2: pickup Bowl\n- x ~`{}")
check("2.2 Persian letters -> '?', one per code point", R.fold_ascii("سیب") == "???")
check("2.3 accents decompose (combining marks dropped): é -> e, ñ -> n", R.fold_ascii("é ñ") == "e n")
check("2.4 control characters except newline -> space (tab, CR, BEL, DEL)", R.fold_ascii("a\tb\rc\x07d\x7fe") == "a b c d e")
check("2.5 emoji and astral code points -> '?'", R.fold_ascii("\U0001F600\U00010348") == "??")
check("2.6 compatibility forms fold (fullwidth Ａ -> A, ligature ﬁ -> fi)", R.fold_ascii("Ａﬁ") == "Afi")

print("== 3 cap: invariants (fuzz) and exact accounting")
rng = random.Random(20261013)
alphabets = ["abc -\n", "".join(sorted(FOLDED)), "سیبé\U0001F600\t\x07a\n", "".join(chr(rng.randrange(0x20, 0x2FFFF)) for _ in range(300)) + "\n"]
bad, n_trunc = [], 0
for i in range(3000):
    a = alphabets[i % len(alphabets)]
    t = "".join(rng.choice(a) for _ in range(rng.choice([0, 1, 50, 599, 600, 601, 700, 1500, 5000])))
    s, info = R.cap_feedback(t)
    f = R.fold_ascii(t)
    ok = len(s) <= 600 and set(s) <= FOLDED and info["chars_sent"] == len(s) and info["chars_raw"] == len(t)
    if len(f) <= 600:
        ok = ok and s == f and not info["truncated"]
    else:
        n_trunc += 1
        head, _, marker = s.rpartition("\n")
        ok = ok and info["truncated"] and f.startswith(head) and marker == R.OMIT_MARKER.format(n=len(f) - len(head)) \
            and info["omitted_chars"] == len(f) - len(head)
    if not ok:
        bad.append((i, len(t), len(s)))
check("3.1 3,000 random texts (4 alphabets, 0-5,000 chars): <= 600 chars, folded alphabet only, exact marker count, sent = prefix + marker",
      not bad and n_trunc > 1000, bad[:5])
check("3.2 exactly 600 folded chars: unchanged; 601: cut", R.cap_feedback("x" * 600)[0] == "x" * 600 and R.cap_feedback("x" * 601)[1]["truncated"])
lines = "\n".join("- step %d: unknown type NotAType%d" % (i, i) for i in range(60))
s, info = R.cap_feedback(lines)
head = s.rpartition("\n")[0]
check("3.3 whole lines are kept when at least one fits (no partial last line)", all(ln in lines.split("\n") for ln in head.split("\n")), head[-80:])
s2, i2 = R.cap_feedback("y" * 2000 + "\n- short")
check("3.4 a first line longer than the room is cut inside, and the marker still fits", len(s2) <= 600 and s2.startswith("y" * 100)
      and i2["omitted_chars"] == 2000 + 8 - len(s2.rpartition("\n")[0]), (len(s2), i2))
check("3.5 the marker's digits always fit (worst case: 1 char kept of a huge text)", len(R.cap_feedback("z" * 10 ** 6)[0]) <= 600)
check("3.6 the audit keeps the raw hash and the sent hash", info["sha_raw"] != info["sha_sent"] and len(info["sha_raw"]) == 64)

print("== 4 adversarial validator output (Astra c: exercise truncation)")
TYPES = json.load(open(os.path.join(os.path.dirname(HERE), "docs", "final-test", "matrix", "detector_types.json"), encoding="utf-8"))["types"]
for n in (12, 48, 80):
    plan = [{"action": "pour_%d" % i, "object": "سیب%d\U0001F600" % i, "to": "NotAType%d" % i, "arg": "lava" * 20} for i in range(n)]
    v = V.validate({"plan": plan}, TYPES, None)
    raw = problems_text(v["errors"])
    s, info = R.cap_feedback(raw)
    check("4.%d %d-step adversarial reply: raw problem text %d chars -> sent %d (<= 600, folded alphabet)" % (n, n, len(raw), len(s)),
          len(raw) > 600 and len(s) <= 600 and set(s) <= FOLDED and info["truncated"])

print("== 5 the runner path (fake provider)")


class Fake:
    """Returns the scripted replies in order; records every message list it was sent."""
    def __init__(self, replies):
        self.replies, self.sent = list(replies), []

    def plan(self, messages, schema, budget, replicate, stage="plan"):
        self.sent.append([dict(m) for m in messages])
        raw = self.replies.pop(0)
        obj, note = parse_reply(raw)
        return {"raw": raw, "parsed": obj, "parse_note": note, "error": "", "error_kind": "", "tokens_in": 10, "tokens_out": 5,
                "cost_usd": 0.0, "latency_s": 0.0, "cached": False}


long_bad = json.dumps({"say": "x", "plan": [{"action": "pickup" if i % 2 else "pour_%d" % i, "object": "سیب%d" % i, "to": None, "arg": None}
                                            for i in range(60)]},
                      ensure_ascii=False)
good = json.dumps({"say": "ok", "plan": [{"action": "pickup", "object": "Apple", "to": None, "arg": None}]})
c2 = PlannerContract(T.planner_types(TYPES), tools=hs.TOOLS, version="ft-v2")
fk = Fake([long_bad, long_bad, good])
r2 = R.PlannerRunner(fk, c2).plan("pick up the apple", [], None)
rep_user = [m[-1]["content"] for m in fk.sent[1:]]
probs = [rep.split("Problems:\n", 1)[1].rsplit("\nReply again", 1)[0] for rep in rep_user]
check("5.1 ft-v2: 3 calls (2 repairs), each repair's problem text <= 600 chars and folded", r2["calls"] == 3 and len(probs) == 2
      and all(len(p) <= 600 and set(p) <= FOLDED for p in probs), [len(p) for p in probs])
check("5.2 ft-v2: the repair frame is REPAIR_V2 around exactly the capped text", all(rep == REPAIR_V2.format(problems=p) for rep, p in zip(rep_user, probs)))
fa = r2["attempts"]
check("5.3 ft-v2 audit: attempts[0/1].feedback (fb-1, truncated, folded); the FULL validator errors stay in .validation",
      all(fa[k]["feedback"]["contract"] == "fb-1" and fa[k]["feedback"]["truncated"] and fa[k]["feedback"]["folded"] for k in (0, 1))
      and len(problems_text(fa[0]["validation"]["errors"])) == fa[0]["feedback"]["chars_raw"] > 600 and "feedback" not in fa[2])
check("5.4 ft-v2: the replayed assistant turn is the model's raw reply, unchanged", fk.sent[1][-2]["content"] == long_bad)
c1 = PlannerContract(T.planner_types(TYPES), tools=hs.TOOLS, version="ft-v1")
fk1 = Fake([long_bad, good])
r1 = R.PlannerRunner(fk1, c1).plan("pick up the apple", [], None)
exp = REPAIR.format(problems=problems_text(r1["attempts"][0]["validation"]["errors"]))
check("5.5 ft-v1 byte-identical: ONE repair, the problem text uncapped (> 600 chars, Persian kept), no feedback key",
      r1["calls"] == 2 and fk1.sent[1][-1]["content"] == exp and len(exp) > 700 and "سیب" in exp and "feedback" not in r1["attempts"][0])
check("5.6 ft-v1 result keys unchanged (no contract / codes keys)", "contract" not in r1 and "codes" not in r1["attempts"][0])

print("== 6 practice-v2 (attempt 9, the only repairs so far): fb-1 is the identity on both texts sent")
p9 = "- step 2: pickup Bowl while already holding Potato (put it somewhere or drop it first)\n- step 3: pick_and_place Potato while already holding Bowl (put it somewhere or drop it first)"
check("6.1 the 181-char ASCII feedback (both repairs) is sent byte-identical under fb-1", len(p9) == 181 and R.cap_feedback(p9)[0] == p9
      and not R.cap_feedback(p9)[1]["folded"])

print("== 7 token evidence (saved probe files)")
try:
    gv = json.load(open(os.path.join(DIAG, "gemma_vocab_probe.json"), encoding="utf-8"))
    ab = json.load(open(os.path.join(DIAG, "ascii_bound_probe.json"), encoding="utf-8"))
except (OSError, ValueError) as e:
    gv, ab = {}, {}
    print("   missing evidence:", e)
check("7.1 Gemma vocabulary: all 96 folded characters are single tokens (=> <= 1 token per character)", gv.get("ok") and gv.get("alphabet") == 96
      and gv.get("missing_single_token") == [], gv)
check("7.2 measured 16 adversarial 600-char families: Gemma max <= 600, Anthropic max <= 1,200 (the factor-2 bound)",
      bool(ab) and ab["max"]["gemma"] <= 600 and ab["max"]["anthropic"] <= R.problem_token_bound("anthropic"), ab.get("max"))

print("\n_runner_fb1_selftest: %d/%d passed" % (N["pass"], N["pass"] + N["fail"]))
sys.exit(0 if N["fail"] == 0 else 1)
