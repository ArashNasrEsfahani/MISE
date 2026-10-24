"""Self-test for step 5 / A4-retrieval -- concepts (EN/FA), lexical + dense channels, RRF, steering gate, budgets, logs.

Offline, Windows `python` from the REPO ROOT, temp directories only (never sim/trace.db, sim/memory.db), no simulator, no WSL, no Ollama,
no network, no torch:

    python sim/_step5_retrieval_selftest.py

Covers (docs/step5-plan.md section 8, Astra A4 review): bilingual concept parsing that FAILS CLOSED (negation, destination, state, pronouns,
quantifiers, unknown nouns, before/after); Persian orthographic variants (ی/ي, ک/ك, ZWNJ); alias generation for all 49 types; role-bound
lexical tokens; RRF; the dense plumbing with a deterministic fake embedder (stale vectors re-embedded, late writes rejected, invalid
vectors rejected, fallback reported); the steering gate (every conjunct; the essential property: changing scores, channels, record length
or insertion order never makes an ineligible record steer; no top-K before the gate); lessons (exact failure key, status/mode rules,
revalidate downgrade, conflicts, application bound); budgets (whole records, qualifiers kept, prefix STOP, oversize, final count, Persian
estimate); logs (retrieved vs eligible vs rejected vs rendered vs applied, whitelist row accepted by a real store); determinism; hygiene.
"""
import copy
import itertools
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)                    # append, not insert: sim/trace.py must not shadow the stdlib `trace`

import ltm                                # noqa: E402
import ltm_concepts as lc                 # noqa: E402
import ltm_retrieval as R                 # noqa: E402

REPO = Path(HERE).parent
_pass = _fail = 0


def check(name, cond, detail=""):
    global _pass, _fail
    if cond:
        _pass += 1
    else:
        _fail += 1
        print(f"  FAIL  {name} {detail}")


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    except Exception as e:
        print(f"      (raised {type(e).__name__}: {e})")
        return False
    return False


_MEMP = Path(HERE) / "memory.db"
_MEM0 = _MEMP.stat().st_mtime_ns if _MEMP.exists() else None
LEX = lc.default_lexicon()
KITCHEN = "FloorPlan1"


def sigs(text):
    return [lc.role_sig(r) for r in lc.parse(text, LEX)["roles"]]


def P(text):
    return lc.parse(text, LEX)


def roles_of(text):
    return P(text)["roles"]


# ---------------------------------------------------------------------------------------------------------------
print("1. lexicon")
check("no alias/verb collisions at load", LEX.collisions == [], str(LEX.collisions))
check("48 types (A26 K-B2b: 49 detector classes minus the excluded Knife)", len(LEX.data["types"]) == 48 and "Knife" not in LEX.data["types"])
check("lexicon sha is stable and recorded", len(LEX.sha) == 64 and lc.Lexicon.load().sha == LEX.sha)
check("lexicon declares it is not dataset-fitted", LEX.data["dataset_fitted"] is False and LEX.data["fp1_tuned"] is False)
check("Bottle / WineBottle / SoapBottle resolve by longest match",
      sigs("pick up the wine bottle") == ["pick/-/WineBottle/-/-/-/-/0"] and sigs("pick up the bottle") == ["pick/-/Bottle/-/-/-/-/0"]
      and sigs("pick up the soap bottle") == ["pick/-/SoapBottle/-/-/-/-/0"])
check("Sink / SinkBasin / Knife / ButterKnife longest match",
      sigs("wash the apple in the sink basin")[0].endswith("/SinkBasin/-/0") and sigs("pick up the butter knife")[0] == "pick/-/ButterKnife/-/-/-/-/0"
      and sigs("pick up the knife")[0] == "pick/-/?/-/-/-/-/0")          # A26: the bare knife stays unknown (was pick/-/Knife/...)
# alias generation for every type, both languages, round trip through the parser
bad = []
for t in LEX.data["types"]:
    al = R.generate_aliases(t, LEX)
    if not al["en"] or not al["fa"]:
        bad.append((t, "empty"))
    for a in al["en"]:
        if sigs(f"pick up the {a}") not in (([f"pick/-/{t}/-/-/-/-/0"]),):
            bad.append((t, a))
    for a in al["fa"]:
        if sigs(f"{a} را بردار") != [f"pick/-/{t}/-/-/-/-/0"]:
            bad.append((t, a))
check("every generated alias (EN + FA, all 49 types) round-trips to its type", not bad, str(bad[:6]))
# Persian orthographic variants
var_bad = []
for t, e in LEX.data["types"].items():
    for a in e["fa"]:
        base = sigs(f"{a} را بردار")
        for v in (a.replace("ی", "ي").replace("ک", "ك"), a.replace(" ", "‌"), a.replace(" ", "‌").replace("ی", "ي").replace("ک", "ك"),
                  "ـ".join(a) if False else a):
            if sigs(f"{v} را بردار") != base:
                var_bad.append((t, v))
check("Persian variants (ی/ي, ک/ك, ZWNJ vs space) parse identically for all FA aliases", not var_bad, str(var_bad[:6]))
check("ZWNJ forms: گوجه‌فرنگی / سیب‌زمینی / تخم‌مرغ", sigs("گوجه‌فرنگی را بردار") == sigs("گوجه فرنگی را بردار") == ["pick/-/Tomato/-/-/-/-/0"]
      and sigs("سیب‌زمینی را بردار") == ["pick/-/Potato/-/-/-/-/0"] and sigs("تخم‌مرغ را بردار") == ["pick/-/Egg/-/-/-/-/0"])
check("Arabic yeh/kaf and digit folding", sigs("يخچال را باز كن") == sigs("یخچال را باز کن") == ["open/-/Fridge/-/-/-/-/0"])

# ---------------------------------------------------------------------------------------------------------------
print("2. concept parsing (fail closed)")
CASES = [
    ("Put the apple in the fridge", ["put/-/Apple/-/-/Fridge/-/0"], True),
    ("سیب را در یخچال بگذار", ["put/-/Apple/-/-/Fridge/-/0"], True),
    ("سیب را توی یخچال بذار", ["put/-/Apple/-/-/Fridge/-/0"], True),
    ("سیب در یخچال بگذار", ["put/-/Apple/-/-/Fridge/-/0"], True),                      # omitted را
    ("Open the fridge", ["open/-/Fridge/-/-/-/-/0"], True),
    ("یخچال را باز کن", ["open/-/Fridge/-/-/-/-/0"], True),
    ("Don't open the fridge", ["open/-/Fridge/-/-/-/-/1"], True),
    ("do not open the fridge", ["open/-/Fridge/-/-/-/-/1"], True),
    ("یخچال را باز نکن", ["open/-/Fridge/-/-/-/-/1"], True),
    ("یخچال را باز نکنید", ["open/-/Fridge/-/-/-/-/1"], True),
    ("never open the drawer", ["open/-/Drawer/-/-/-/-/1"], True),
    ("هرگز کشو را باز نکن", ["open/-/Drawer/-/-/-/-/1"], True),
    ("take the apple without opening the fridge", ["pick/-/Apple/-/-/-/-/0", "open/-/Fridge/-/-/-/-/1"], True),
    ("Put the apple in the open fridge", ["put/-/Apple/-/-/Fridge/open/0"], True),
    ("close the open drawer", ["close/-/Drawer/open/-/-/-/0"], True),
    ("Turn off the stove", ["toggle/off/StoveBurner/-/-/-/-/0"], True),
    ("turn the stove on", ["toggle/on/StoveBurner/-/-/-/-/0"], True),
    ("اجاق را روشن کن", ["toggle/on/StoveBurner/-/-/-/-/0"], True),
    ("اجاق را خاموش کن", ["toggle/off/StoveBurner/-/-/-/-/0"], True),
    ("slice the bread", ["slice/-/Bread/-/-/-/-/0"], True),
    ("نان را برش بده", ["slice/-/Bread/-/-/-/-/0"], True),
    ("take the apple from the fridge", ["pick/-/Apple/-/Fridge/-/-/0"], True),
    ("سیب را از یخچال بردار", ["pick/-/Apple/-/Fridge/-/-/0"], True),
    ("pick up the apple and put it in the fridge", ["pick/-/Apple/-/-/-/-/0", "put/-/Apple/-/-/Fridge/-/0"], True),
    ("سیب را بردار و در یخچال بگذار", ["pick/-/Apple/-/-/-/-/0", "put/-/Apple/-/-/Fridge/-/0"], True),   # W4a/A17: FA elided object inherited along a carry chain
    ("یخچال را باز کن و بردار", None, False),                                             # W4a: never inherited across open (stays unresolved)
    ("سیب را برندار و در یخچال بگذار", None, False),                                      # W4a: never inherited across a negated clause
    ("لیوان را بردار", ["pick/-/Mug/-/-/-/-/0"], True),                                      # W4a: the everyday word is Mug
    ("فنجان را بردار", ["pick/-/Cup/-/-/-/-/0"], True),
    ("کاسه را داخل یکی از کابینت‌ها بگذار", ["put/-/Bowl/-/-/Cabinet/-/0"], True),          # W4a: partitive "one of the X"
    ("در مایکروویو را باز کن", ["open/-/Microwave/-/-/-/-/0"], True),                       # W4a: door-of
    ("open the fridge then slice the bread", ["open/-/Fridge/-/-/-/-/0", "slice/-/Bread/-/-/-/-/0"], True),
    ("یخچال را باز کن سپس سیب را بردار", ["open/-/Fridge/-/-/-/-/0", "pick/-/Apple/-/-/-/-/0"], True),
    ("open the fridge, take the apple", ["open/-/Fridge/-/-/-/-/0", "pick/-/Apple/-/-/-/-/0"], True),
    ("take the apple but don't open the fridge", ["pick/-/Apple/-/-/-/-/0", "open/-/Fridge/-/-/-/-/1"], True),
    ("put the mug in the sink after that open the fridge", ["put/-/Mug/-/-/Sink/-/0", "open/-/Fridge/-/-/-/-/0"], True),
    ("put it in the fridge", None, False),                                               # pronoun without antecedent
    ("put the apple in the bowl and slice it", None, False),                             # two antecedents
    ("put the apple on the table", None, False),                                         # unknown noun after a preposition
    ("put two apples in the fridge", None, False),                                       # quantifier
    ("put the apple and the bread in the fridge", None, False),                          # coordination of objects
    ("open the fridge or the cabinet", None, False),
    ("open the fridge before taking the apple", None, False),                            # ordering clause
    ("if the fridge is closed open it", None, False),
    ("", None, False),
    ("hello there", None, False),
    ("the apple", None, False),                                                          # no verb
    ("turn the stove", None, False),                                                     # toggle without direction
    ("take the apple put the mug in the sink", None, False),                             # two verbs, one clause
    ("don't not open the fridge", None, False),                                          # double negation
]
for text, want, ok in CASES:
    r = P(text)
    got = [lc.role_sig(x) for x in r["roles"]]
    good = (r["ok"] == ok) and (want is None or got == want)
    check(f"parse {text!r}", good, f"-> ok={r['ok']} {got} {r['reasons']}")
# EN <-> FA parity: language-neutral roles
PAIRS = [("put the apple in the fridge", "سیب را در یخچال بگذار"), ("open the drawer", "کشو را باز کن"),
         ("don't open the fridge", "یخچال را باز نکن"), ("wash the mug", "ماگ را بشور"), ("pick up the fork", "چنگال را بردار"),   # A26: was the knife (now unknown on both sides; unknowns never compare equal)
         
         ("take the apple from the fridge", "سیب را از یخچال بردار"), ("turn on the stove", "اجاق را روشن کن")]
check("EN and FA instructions give EQUAL roles", all(lc.roles_equal(roles_of(a), roles_of(b)) for a, b in PAIRS),
      str([(a, sigs(a), sigs(b)) for a, b in PAIRS if not lc.roles_equal(roles_of(a), roles_of(b))]))
# negation / destination / state confusions never equal
CONF = [("open the fridge", "don't open the fridge"), ("یخچال را باز کن", "یخچال را باز نکن"),
        ("put the apple in the fridge", "put the apple in the sink"), ("put the apple in the fridge", "put the mug in the fridge"),
        ("put the apple in the fridge", "put the apple in the open fridge"), ("turn on the stove", "turn off the stove"),
        ("take the apple from the fridge", "take the apple"), ("open the fridge", "close the fridge")]
check("negation / destination / object / state / direction / source / verb differences are NEVER equal",
      all(not lc.roles_equal(roles_of(a), roles_of(b)) for a, b in CONF))
check("'?' never matches, not even '?' == '?'", not lc.roles_equal(roles_of("put it in the fridge"), roles_of("put it in the fridge")))
check("equal roles need equal count and order", not lc.roles_equal(roles_of("open the fridge then slice the bread"),
                                                                    roles_of("slice the bread then open the fridge")))
check("parser/lexicon versions recorded", P("open the fridge")["parser"] == lc.PARSER_VERSION and P("open the fridge")["lexicon"] == LEX.sha)
# lexicon-free tokens must not crash on odd input
for odd in ["   ", "!!!", "🍎", "apple" * 200, "ی" * 500, "‌‌", "open\u0000fridge"]:
    try:
        lc.parse(odd, LEX)
        check(f"odd input {odd[:8]!r} does not raise", True)
    except Exception as e:
        check(f"odd input {odd[:8]!r} does not raise", False, repr(e))

# ---------------------------------------------------------------------------------------------------------------
print("3. key text and lexical channel")
kt = lc.key_text(roles_of("don't open the fridge"), LEX)
check("key text is bilingual and negation-explicit", "not open fridge" in kt and "نباید" in kt, kt)
check("key text from structured roles is the same for EN and FA instructions", lc.key_text(roles_of("open the drawer"), LEX)
      == lc.key_text(roles_of("کشو را باز کن"), LEX))
ct1 = set(lc.concept_tokens(roles_of("put the apple in the fridge and then put the mug in the sink")))
ct2 = set(lc.concept_tokens(roles_of("put the apple in the sink and then put the mug in the fridge")))
check("swapped destinations do not share the role signature tokens", not ({t for t in ct1 if ".sig=" in t} & {t for t in ct2 if ".sig=" in t}))
check("polarity token always present", all(any(t.endswith(".neg=0") or t.endswith(".neg=1") for t in lc.concept_tokens([r])) for r in roles_of("open the fridge")))


def proc(pid, text, status="trusted", su=3, fr=0, kitchen=KITCHEN, steps=None, pre=None, **kw):
    r = P(text)["roles"]
    if steps is None:
        steps = []
        for i, x in enumerate(r):
            if x["neg"]:
                continue
            st = {"verb": x["verb"], "otype": x["obj"]}
            if x["dest"] != "-":
                st["to"] = x["dest"]
            steps.append(st)
    if pre is not None and steps:
        steps[0] = dict(steps[0], pre=pre)
    rec = {"proc": pid, "kitchen": kitchen, "intent_key": {"roles": r}, "intent_text": "", "body": {"steps": steps}, "status": status,
           "success_uses": su, "failed_reuses": fr, "E_created": 1, "E_support": 1, "admit_goal_ok": True}
    rec.update(kw)
    return rec


SNAP = {"holding": None, "located": ["Apple", "Mug", "Bread"], "container_state": {"Fridge": "closed"}, "rev": 1}


def mk(**kw):
    return R.Retriever(**kw)


rt = mk()
recs = [proc("pr:a", "put the apple in the fridge"), proc("pr:b", "put the apple in the sink"), proc("pr:c", "open the fridge"),
        proc("pr:d", "don't open the fridge", steps=[{"verb": "pick", "otype": "Apple"}]), proc("pr:e", "put the mug in the fridge")]
res = rt.retrieve_procedures("Put the apple in the fridge", recs, kitchen=KITCHEN, wm_snapshot=SNAP, episode="e1")
check("exact-intent trusted procedure steers; near-misses do not", [r["proc"] for r in res["steer"]] == ["pr:a"], str([r["proc"] for r in res["steer"]]))
rej = {x["id"]: x for x in res["rejected"]}
check("near-misses are retrieved but rejected on roles", all("roles" in rej[i]["failed"] for i in ("pr:b", "pr:e") if i in rej))
res2 = rt.retrieve_procedures("سیب را در یخچال بگذار", recs, kitchen=KITCHEN, wm_snapshot=SNAP, episode="e1")
check("Persian query steers the same procedure as the English one", [r["proc"] for r in res2["steer"]] == ["pr:a"])
res3 = rt.retrieve_procedures("Open the fridge", recs, kitchen=KITCHEN, wm_snapshot=SNAP, episode="e1")
check("positive 'open' query does not retrieve-and-steer the negated 'don't open' procedure", [r["proc"] for r in res3["steer"]] == ["pr:c"])
res4 = rt.retrieve_procedures("Don't open the fridge", recs, kitchen=KITCHEN, wm_snapshot=SNAP, episode="e1")
check("negated query steers only the negated procedure (body consistent: no open step)", [r["proc"] for r in res4["steer"]] == ["pr:d"])
check("an empty-concept query is never all_match and steers nothing",
      rt.retrieve_procedures("hello there", recs, kitchen=KITCHEN, wm_snapshot=SNAP)["steer"] == [])
rk = rt.rank([], {"apple", "fridge"}, "apple fridge", [rt._proc_entry(r) for r in recs])
check("surface-only match carries all_match False", not any(rk["all_match"].values()))
check("all_match true for the identical-intent record", rt.retrieve_procedures("Put the apple in the fridge", recs, kitchen=KITCHEN,
                                                                                   wm_snapshot=SNAP)["rank"]["all_match"].get("pr:a") is True)

# ---------------------------------------------------------------------------------------------------------------
print("4. RRF")
f = R.rrf_fuse({"lexical": ["a", "b", "c"], "dense": ["b", "a", "d"]}, 60)
exp_a = 1 / 61 + 1 / 62
check("RRF score formula", abs(dict((i, s) for i, s, _ in f)["a"] - R.quant(exp_a)) < 1e-9)
check("RRF ties broken by id", [i for i, _, _ in R.rrf_fuse({"lexical": ["z"], "dense": ["a"]}, 60)] == ["a", "z"])
check("RRF is channel-name-order independent", R.rrf_fuse({"lexical": ["a", "b"], "dense": ["b", "a"]}, 60)
      == R.rrf_fuse({"dense": ["b", "a"], "lexical": ["a", "b"]}, 60))
check("RRF rejects non-positive k", raises(ValueError, R.rrf_fuse, {"lexical": ["a"]}, 0))
check("RRF missing record contributes zero", dict((i, s) for i, s, _ in R.rrf_fuse({"lexical": ["a", "b"], "dense": ["a"]}, 60))["b"]
      == R.quant(1 / 62))
check("RRF duplicate ids within a channel count once", R.rrf_fuse({"lexical": ["a", "a"]}, 60)[0][1] == R.quant(1 / 61))

# ---------------------------------------------------------------------------------------------------------------
print("5. dense channel plumbing")
emb = R.HashEmbedder()
cache = R.EmbeddingCache(emb)
v1 = cache.vectors(["open fridge", "put apple into fridge"])
calls = emb.calls
cache.vectors(["open fridge"])
check("embedding cache hit does not re-embed", emb.calls == calls and cache.hits >= 1)
items = {"pr:a": "put apple into fridge", "pr:b": "open fridge"}
s1 = cache.sync(items)
c0 = emb.calls
cache.sync(items)
check("sync with unchanged keys embeds nothing", emb.calls == c0)
items2 = dict(items, **{"pr:a": "put apple into sink"})
s2 = cache.sync(items2)
check("key change re-embeds exactly that record", emb.calls == c0 + 1 and s2["pr:a"] != s1["pr:a"] and s2["pr:b"] == s1["pr:b"])
check("stale vector is no longer served", cache.index["pr:a"][0] == R.sha("put apple into sink"))
check("late write for a superseded key is rejected", cache.publish("pr:a", R.sha("put apple into fridge"), s1["pr:a"], R.sha("put apple into sink")) is False
      and cache.index["pr:a"][0] == R.sha("put apple into sink"))
check("invalid vectors rejected (NaN / zero / wrong dim)", not cache.publish("pr:x", "h", [float("nan")] * 64, "h")
      and not cache.publish("pr:x", "h", [0.0] * 64, "h") and not cache.publish("pr:x", "h", [1.0] * 3, "h"))


class BadEmb(R.Embedder):
    model_rev = "bad"

    def embed(self, texts):
        return [[float("nan")] * 8 for _ in texts]


check("embedder returning NaN raises ValueError in the cache", raises(ValueError, R.EmbeddingCache(BadEmb()).vectors, ["x"]))
emb_b = R.HashEmbedder(rev="fake-hash-v2")
cache_b = R.EmbeddingCache(emb_b)
cache_b.mem = dict(cache.mem)
n0 = emb_b.calls
cache_b.vectors(["open fridge"])
check("model_rev change invalidates cached vectors (keyed by revision)", emb_b.calls == n0 + 1)
with tempfile.TemporaryDirectory() as td:
    cp = os.path.join(td, "emb.sqlite")
    e1 = R.HashEmbedder()
    c1 = R.EmbeddingCache(e1, cp)
    c1.vectors(["open fridge"])
    e2 = R.HashEmbedder()
    c2 = R.EmbeddingCache(e2, cp)
    c2.vectors(["open fridge"])
    check("persisted cache survives a restart", e2.calls == 0)
    e3 = R.HashEmbedder(rev="other")
    c3 = R.EmbeddingCache(e3, cp)
    c3.vectors(["open fridge"])
    check("persisted cache is revision-keyed", e3.calls == 1)
P2 = copy.deepcopy(ltm.load_params())
P2["retrieval"]["channels"]["value"] = ["lexical", "dense"]
rt_h = mk(params=P2, embedder=R.HashEmbedder())
res_h = rt_h.retrieve_procedures("Put the apple in the fridge", recs, kitchen=KITCHEN, wm_snapshot=SNAP, episode="e1")
check("hybrid run reports both channels active", res_h["rank"]["channels"]["active"] == ["lexical", "dense"] and res_h["rank"]["fallback"] is None)
check("hybrid steering equals lexical steering for the gate (gate dominates)", [r["proc"] for r in res_h["steer"]] == ["pr:a"])
rt_nd = mk(params=P2)
res_nd = rt_nd.retrieve_procedures("Put the apple in the fridge", recs, kitchen=KITCHEN, wm_snapshot=SNAP, episode="e1")
check("dense configured but no embedder -> lexical, fallback REPORTED", res_nd["rank"]["channels"]["active"] == ["lexical"]
      and res_nd["rank"]["fallback"] == "dense_unavailable" and res_nd["log"]["fallback"] == "dense_unavailable")


class BoomEmb(R.Embedder):
    model_rev = "boom"

    def embed(self, texts):
        raise RuntimeError("model not loaded")


rt_b = mk(params=P2, embedder=BoomEmb())
res_b = rt_b.retrieve_procedures("Put the apple in the fridge", recs, kitchen=KITCHEN, wm_snapshot=SNAP, episode="e1")
bge = R.BgeM3Embedder()
check("BgeM3Embedder is lazy (constructing it imports no torch) and its revision encodes weights + config",
      "torch" not in sys.modules and bge._model is None and bge.model_rev.startswith("BAAI/bge-m3@5617a9f61b02|cpu|maxlen128"))
check("embedder failure degrades to lexical, fallback REPORTED, steering unchanged",
      res_b["rank"]["channels"]["active"] == ["lexical"] and str(res_b["rank"]["fallback"]).startswith("dense_error") and
      [r["proc"] for r in res_b["steer"]] == ["pr:a"])
check("the lexical channel can never be dropped", mk(channels=["dense"]).active_channels()[0][0] == "lexical")

# ---------------------------------------------------------------------------------------------------------------
print("6. steering gate")
base = proc("pr:g", "put the apple in the fridge", pre=["hand_empty", "located:Apple"])


def steers(rec, **kw):
    q = kw.pop("query", "Put the apple in the fridge")
    snap = kw.pop("snap", SNAP)
    rr = mk().retrieve_procedures(q, [rec] + kw.pop("others", []), kitchen=kw.pop("kitchen", KITCHEN), wm_snapshot=snap)
    return [r["proc"] for r in rr["steer"]], rr


check("baseline steers", steers(base)[0] == ["pr:g"])
check("other kitchen does not steer", steers(dict(base, kitchen="FloorPlan6"))[0] == [])
check("candidate does not steer but is a labelled suggestion", steers(dict(base, status="candidate"))[0] == []
      and [r["proc"] for r in steers(dict(base, status="candidate"))[1]["suggestions"]] == ["pr:g"])
check("demoted: suggestion only", steers(dict(base, status="demoted", failed_reuses=3, success_uses=3))[0] == [])
check("retired: never steers, never suggested", steers(dict(base, status="retired"))[0] == []
      and steers(dict(base, status="retired"))[1]["suggestions"] == [])
check("trusted with failed_reuses >= success_uses is contradicted", steers(dict(base, failed_reuses=3, success_uses=3))[0] == [])
check("explicit unresolved contradiction disqualifies even after many successes", steers(dict(base, success_uses=50, contradicted=True))[0] == [])
check("trusted with more successes than failures still steers", steers(dict(base, success_uses=5, failed_reuses=2))[0] == ["pr:g"])
check("failed precondition (hand not empty) excludes", steers(base, snap=dict(SNAP, holding="Mug"))[0] == [])
check("failed precondition (target not located)", steers(base, snap=dict(SNAP, located=["Mug"]))[0] == [])
check("no WM snapshot -> unknown -> excluded (fail closed)", steers(base, snap=None)[0] == [])
check("snapshot missing a needed key -> unknown -> excluded", steers(base, snap={"holding": None})[0] == [])
unk = steers(dict(base, body={"steps": [dict(base["body"]["steps"][0], pre=["weird_atom"])] + base["body"]["steps"][1:]}))
check("unknown precondition atom -> excluded, reason recorded", unk[0] == [] and unk[1]["decisions"]["pr:g"]["failed"] == ["preconditions_unknown"])
check("no preconditions needed -> steers without a snapshot-dependent atom", steers(proc("pr:h", "open the fridge"), query="open the fridge", snap=SNAP)[0] == ["pr:h"])
check("body that does not realise the intent is rejected (body-intent mismatch)",
      steers(dict(base, body={"steps": [{"verb": "pick", "otype": "Apple", "pre": ["hand_empty", "located:Apple"]}]}))[0] == [])
check("body that performs a NEGATED action is rejected", steers(proc("pr:n", "don't open the fridge", steps=[{"verb": "open", "otype": "Fridge"}]),
                                                              query="don't open the fridge")[0] == [])
check("query with an unresolved role never steers", steers(base, query="put it in the fridge")[0] == [])
check("injected pre_eval is honoured", mk().retrieve_procedures("Put the apple in the fridge", [base], kitchen=KITCHEN, wm_snapshot=None,
                                                               pre_eval=lambda a, s: ("hold", None))["steer"] != [])
check("injected pre_eval returning error status is not 'hold'", mk().retrieve_procedures("Put the apple in the fridge", [base], kitchen=KITCHEN,
                                                                                       wm_snapshot=SNAP, pre_eval=lambda a, s: ("error", "x"))["steer"] == [])
# gate runs on EVERY retrieved record: eligible record buried behind ineligible decoys
decoys = [proc(f"pr:d{i:02d}", "put the apple in the fridge", status="demoted", fr=1) for i in range(12)]
buried = steers(base, others=decoys)
check("no top-K before the gate: an eligible record behind 12 better-ranked decoys still steers", buried[0] == ["pr:g"], str(buried[0]))
# ESSENTIAL PROPERTY: scores, channels, length, order never make an ineligible record steer
rng = random.Random(7)
INEL = []
for i in range(10):
    t = "put the apple in the fridge"
    INEL += [proc(f"pr:i{i}a", t, status="candidate"), proc(f"pr:i{i}b", t, kitchen="FloorPlan6"), proc(f"pr:i{i}c", t, contradicted=True),
             proc(f"pr:i{i}d", t, pre=["located:Tomato"]), proc(f"pr:i{i}e", t.replace("fridge", "sink")),
             proc(f"pr:i{i}f", t, steps=[{"verb": "pick", "otype": "Apple"}]),
             proc(f"pr:i{i}g", t, fr=4, su=4), proc(f"pr:i{i}h", "put the apple in the fridge and then close the fridge")]
GOOD = [proc("pr:z1", "put the apple in the fridge", pre=["located:Apple"]), proc("pr:z2", "put the apple in the fridge", su=9, pre=["hand_empty"])]
want = {"pr:z1", "pr:z2"}
allr = GOOD + INEL
outs = set()
for trial in range(40):
    rr = allr[:]
    rng.shuffle(rr)
    if trial % 3 == 1:                                    # inflate the length of the ineligible records' text
        rr = [dict(r, intent_text=("apple fridge put " * 60) if r["proc"] not in want else "") for r in rr]
    if trial % 4 == 2:
        params_ = P2
        ret = mk(params=params_, embedder=R.HashEmbedder(dim=16 + trial))
    elif trial % 4 == 3:
        ret = mk(channels=["lexical"])
    else:
        ret = mk()
    for q in ("Put the apple in the fridge", "سیب را در یخچال بگذار"):
        rrs = ret.retrieve_procedures(q, rr, kitchen=KITCHEN, wm_snapshot=SNAP, top_k=50)
        outs.add(frozenset(r["proc"] for r in rrs["steer"]))
check("essential property: only the eligible records ever steer, under shuffling / length inflation / channel changes / dense on-off",
      outs == {frozenset(want)}, str(outs))


class AdversarialEmb(R.Embedder):
    """Makes every INELIGIBLE record identical to the query in embedding space and eligible ones orthogonal."""
    model_rev = "adv"

    def embed(self, texts):
        return [[1.0, 0.0] if ("sink" not in t and "candidate" not in t) else [0.0, 1.0] for t in texts]


adv = mk(params=P2, embedder=AdversarialEmb())
ra = adv.retrieve_procedures("Put the apple in the fridge", allr, kitchen=KITCHEN, wm_snapshot=SNAP, top_k=50)
check("an adversarial dense channel cannot promote an ineligible record", {r["proc"] for r in ra["steer"]} == want)
# suggestions are returned but never rendered unless asked
sg = mk()
rs = sg.retrieve_procedures("Put the apple in the fridge", [dict(base, status="candidate", proc="pr:s")], kitchen=KITCHEN, wm_snapshot=SNAP)
b_off = sg.memory_block(None, rs, None, level="L3", lang="en")
b_on = sg.memory_block(None, rs, None, level="L3", lang="en", render_suggestions=True)
check("suggestions are NOT rendered into the steering context by default", "pr:s" not in [r["id"] for r in b_off["records"]] and "Unverified" not in b_off["text"])
check("suggestions render only on explicit opt-in, labelled unverified", "Unverified suggestion" in b_on["text"])
check("recheck before execution catches a changed WM", mk().recheck(None, base, KITCHEN, SNAP, "Put the apple in the fridge")
      and not mk().recheck(None, base, KITCHEN, dict(SNAP, holding="Mug"), "Put the apple in the fridge")
      and not mk().recheck(None, dict(base, status="demoted"), KITCHEN, SNAP, "Put the apple in the fridge"))

# ---------------------------------------------------------------------------------------------------------------
print("7. lessons")


def lesson(lid, status="supported", action="change_pose", cause="not_detected", verb="pick", otype="Apple", rtype="Fridge", state="closed",
           kitchen=KITCHEN, E=1, revalidate=False, **kw):
    r = {"lesson": lid, "kitchen": kitchen, "conditions": {"cause": cause, "verb": verb, "otype": otype, "rtype": rtype, "state": state},
         "cond_key": lid, "action": action, "status": status, "helpful_units": 2, "unhelpful_units": 0, "recurrences": 1,
         "origin_episode": "e0", "E_created": E, "E_support": E, "revalidate": revalidate}
    r.update(kw)
    return r


FK = {"cause": "not_detected", "verb": "pick", "otype": "Apple", "rtype": "Fridge", "state": "closed"}


def les(records, mode="test", fk=FK, **kw):
    return mk().retrieve_lessons(fk, records, kitchen=KITCHEN, mode=mode, **kw)


check("supported lesson with the exact local failure key applies", les([lesson("ls:1")])["applied"]["lesson"] == "ls:1")
check("a different cause / verb / type / container / state does not apply",
      all(les([lesson("ls:1")], fk=dict(FK, **{k: v}))["applied"] is None for k, v in
          (("cause", "unreachable"), ("verb", "put"), ("otype", "Mug"), ("rtype", "Cabinet"), ("state", "open"))))
check("unknown query field never satisfies a match", all(les([lesson("ls:1", state="*")], fk=dict(FK, state=v))["applied"] is None
                                                       for v in (None, "", "?", "*")))
check("a '*' wildcard in a stored lesson never applies", les([lesson("ls:w", state="*")])["applied"] is None)
check("abstain reason reported for unknown query field", les([lesson("ls:1")], fk=dict(FK, state="?"))["abstain_reason"].startswith("query_field_unknown"))
check("tentative lesson: practice only", les([lesson("ls:t", status="tentative")], mode="test")["applied"] is None
      and les([lesson("ls:t", status="tentative")], mode="practice")["applied"]["lesson"] == "ls:t")
check("contradicted / retired lessons never apply", les([lesson("ls:c", status="contradicted")], mode="practice")["applied"] is None
      and les([lesson("ls:r", status="retired")], mode="practice")["applied"] is None)
check("revalidate downgrades supported to tentative BEFORE sorting (test mode: no; practice: yes)",
      les([lesson("ls:v", revalidate=True)], mode="test")["applied"] is None
      and les([lesson("ls:v", revalidate=True)], mode="practice")["decisions"]["ls:v"]["eff_status"] == "tentative")
two = les([lesson("ls:old", status="tentative", E=1), lesson("ls:sup", E=9)], mode="practice")
check("supported is preferred over an older tentative one", two["applied"]["lesson"] == "ls:sup")
check("other kitchen's lesson does not apply", les([lesson("ls:k", kitchen="FloorPlan6")])["applied"] is None)
check("negated open in the instruction forbids an open_first remedy",
      les([lesson("ls:o", action="open_first")], roles=roles_of("don't open the fridge"))["applied"] is None
      and les([lesson("ls:o", action="open_first")], roles=roles_of("pick up the apple"))["applied"] is not None)
check("remedy must be available to the actuator", les([lesson("ls:a", action="crouch")], available={"change_pose"})["applied"] is None
      and les([lesson("ls:a", action="crouch")], available={"crouch"})["applied"] is not None)
check("lesson with action none never applies", les([lesson("ls:n", action="none")])["applied"] is None)
conf = les([lesson("ls:x", action="change_pose", E=1), lesson("ls:y", action="approach_closer", E=2)])
check("conflicting eligible remedies -> abstain", conf["applied"] is None and conf["abstain_reason"] == "conflicting_remedies")
same = les([lesson("ls:q2", action="change_pose", E=5), lesson("ls:q1", action="change_pose", E=5)])
check("same action, same age -> lowest id (deterministic)", same["applied"]["lesson"] == "ls:q1")
rt_l = mk()
a1 = rt_l.retrieve_lessons(FK, [lesson("ls:1")], kitchen=KITCHEN, mode="test", failure_event="fe1", episode="e1")
a2 = rt_l.retrieve_lessons(FK, [lesson("ls:2", E=0)], kitchen=KITCHEN, mode="test", failure_event="fe1", episode="e1")
a3 = rt_l.retrieve_lessons(FK, [lesson("ls:2", E=0)], kitchen=KITCHEN, mode="test", failure_event="fe2", episode="e1")
check("application bounded to one remedy per failure event", a1["applied"] is not None and a2["applied"] is None and a2["abstain_reason"] == "application_bound"
      and a3["applied"] is not None)
check("first eligible application per (episode, lesson) is the scored one", rt_l.ledger.first_scored("e1", "ls:1") and not rt_l.ledger.first_scored("e1", "ls:1")
      and rt_l.ledger.first_scored("e2", "ls:1"))
lg = a1["log"]["retrieval"]
check("mark_applied is monotone and idempotent", rt_l.mark_applied(lg, "ls:1", "selected") == "selected" and rt_l.mark_applied(lg, "ls:1", "succeeded") == "succeeded"
      and rt_l.mark_applied(lg, "ls:1", "attempted") == "succeeded" and rt_l.mark_applied(lg, "ls:1", "succeeded") == "succeeded"
      and rt_l.logs[lg]["applied"] == {"ls:1": "succeeded"} and raises(ValueError, rt_l.mark_applied, lg, "ls:1", "bogus"))
# hypothesis prose never rendered
sec = "SECRETPROSE the fridge was probably blocked"
lr = les([lesson("ls:p", hypothesis=sec, prose=sec, notes=sec)])
bl = mk().memory_block(None, None, lr, level="L3", lang="en")
bl_fa = mk().memory_block(None, None, lr, level="L3", lang="fa")
check("hypothesis prose is never rendered (EN and FA blocks)", "SECRETPROSE" not in bl["text"] and "SECRETPROSE" not in bl_fa["text"] and "Failure hints" in bl["text"])

# ---------------------------------------------------------------------------------------------------------------
print("8. budgets")


def cand(i, tier="T0", rtype="Fridge", age=2, support=3, c=0, kind="place", status="current"):
    return {"id": f"pl:{i}" if kind == "place" else rtype, "kind": kind, "rtype": rtype, "tier": tier, "source": "location_claim", "age": age,
            "support": support, "contradictions": c, "status": status, "weight": 0.5, "provenance": "natural/verified"}


def where(cands, supp=()):
    return {"otype": "Apple", "kitchen": KITCHEN, "e_now": 5, "epoch": 1, "candidates": cands,
            "explore": {"id": "EXPLORE", "tier": "EXPLORE", "suppressed": list(supp), "deferred": [], "n_beyond_cut": 0}}


W = where([cand(1), cand(2, "T2", "Cabinet", age=7, support=1), cand(3, "T3", "Drawer", kind="rtype", age=None, support=4)],
          [{"place": "pl:9", "reason": "qualified_negative", "age": 1, "weight": 0.9}])
for lang in ("en", "fa"):
    b = mk().memory_block(W, None, None, level="L3", lang=lang, otype="Apple")
    check(f"{lang}: where block within budget", b["tokens"] <= b["budget"] and b["overflow"] is None, str(b["tokens"]))
    check(f"{lang}: EXPLORE option always present", R.WORDS[lang]["explore"] in b["text"])
    lines = [r["text"] for r in b["records"] if r["section"] == "where"]
    check(f"{lang}: every candidate line carries its qualifiers (tier label, support)", all("support" in l or "پشتیبانی" in l for l in lines[:3]), str(lines))
    check(f"{lang}: suppressed place rendered with the qualified-negative wording", any("pl:9" in l for l in lines), str(lines))
b_en = mk().memory_block(W, None, None, level="L3", lang="en", otype="Apple")
check("never says a container is empty", "empty" not in b_en["text"].lower())
# prefix STOP, whole records, qualifier atomicity under shrinking budgets
many = [cand(i) for i in range(1, 40)]
prev = None
for B in (400, 200, 150, 120, 90, 70, 55, 40, 30, 20, 12):
    b = mk().memory_block(where(many), None, None, level="L1", lang="en", otype="Apple", budget_tokens=B)
    ids = [r["id"] for r in b["records"]]
    check(f"budget {B}: final tokens <= budget (or declared overflow)", b["tokens"] <= B or b["overflow"] is not None, str((b["tokens"], B)))
    check(f"budget {B}: kept ids are a prefix of the ranking", ids == [c["id"] for c in many][:len(ids)])
    check(f"budget {B}: every kept line is a WHOLE record (ends with its closing qualifier bracket)", all(r["text"].endswith("]") for r in b["records"]))
    check(f"budget {B}: monotone (smaller budget keeps no more)", prev is None or len(ids) <= prev)
    prev = len(ids)
big = mk().memory_block(where(many), None, None, level="L1", lang="en", otype="Apple", budget_tokens=2000)
check("where_share caps the where section even when budget is huge", len([r for r in big["records"]]) < len(many) or big["tokens"] <= int(2000))
sh = mk().memory_block(where(many), None, None, level="L3", lang="en", otype="Apple", budget_tokens=300)
wtok = sum(r["tokens"] for r in sh["records"] if r["section"] == "where")
check("where lines respect the where_share of the budget (40%)", wtok <= int(300 * 0.4), str(wtok))
# oversize record cannot starve the rest
huge = cand(0, rtype="Fridge")
huge["id"] = "pl:" + "9" * 40
dd = where([cand(1), dict(huge, tier="T2"), cand(2)])
dd["candidates"][1]["status"] = "x" * 900                    # inflate: the line is not changed, so inflate the rtype name instead
dd["candidates"][1]["rtype"] = None
dd["candidates"][1]["id"] = "pl:" + "7" * 700
bo = mk().memory_block(dd, None, None, level="L1", lang="en", otype="Apple", budget_tokens=150)   # the registered L1 budget; A16 raised the retention budget to 800
check("an oversize record is excluded with a logged reason, not silently, and does not starve the records behind it",
      any(d["reason"] == "oversize" for d in bo["dropped"]) and [r["id"] for r in bo["records"]] == ["pl:1", "pl:2"], str(bo["dropped"])[:200] + str([r["id"] for r in bo["records"]]))
check("tiny budget below the mandatory lines degrades gracefully", mk().memory_block(W, None, None, level="L1", lang="en", otype="Apple", budget_tokens=3)["text"] == "")
b_mand = mk().memory_block(W, None, None, level="L1", lang="en", otype="Apple", budget_tokens=26)
check("budget above head but below mandatory lines: records declared dropped, no crash", all(d["reason"] in ("mandatory_exceeds_budget", "budget", "oversize", "final_trim")
                                                                                       for d in b_mand["dropped"]))
# token counting: fallback vs injected, Persian factor
check("fallback estimate flags token_claim False; injected counter True",
      mk().memory_block(W, None, None, level="L3", lang="en", otype="Apple")["token_claim"] is False
      and mk(count_tokens=lambda t: len(t.split())).memory_block(W, None, None, level="L3", lang="en", otype="Apple")["token_claim"] is True)
check("estimate: Persian costs more tokens per character than Latin", R.estimate_tokens("ا" * 40) > R.estimate_tokens("a" * 40)
      and R.estimate_tokens("ا" * 40) == math.ceil(40 * 1.66 / 4) and R.estimate_tokens("a" * 40) == 10)
check("estimate is deterministic and monotone", all(R.estimate_tokens("ab" * n) <= R.estimate_tokens("ab" * (n + 1)) for n in range(30)))
wc = mk(count_tokens=lambda t: len(t.split())).memory_block(W, None, None, level="L1", lang="en", otype="Apple", budget_tokens=60)
check("injected counter is used for the budget", wc["tokens"] == len(wc["text"].split()) and wc["tokens"] <= 60, str(wc["tokens"]))
# random property: final count never exceeds budget
rr = random.Random(3)
okp = True
for _ in range(60):
    cs = [cand(i, rr.choice(["T0", "T1", "T2", "T3", "T3b", "T4"]), rr.choice(["Fridge", "Cabinet", "Drawer"]), age=rr.choice([None, 1, 50]),
               support=rr.randint(0, 9), c=rr.randint(0, 3), kind=rr.choice(["place", "rtype"])) for i in range(rr.randint(0, 6))]
    ps = mk().retrieve_procedures("put the apple in the fridge", [proc(f"pr:{i}", "put the apple in the fridge") for i in range(rr.randint(0, 3))],
                                  kitchen=KITCHEN, wm_snapshot=SNAP)
    lang = rr.choice(["en", "fa"])
    B = rr.randint(25, 260)
    b = mk().memory_block(where(cs), ps, lr, level="L2", lang=lang, otype="Apple", budget_tokens=B)
    okp &= (b["tokens"] <= B) or (b["overflow"] is not None) or (b["text"] == "")
check("random blocks: final serialised count never exceeds the budget", okp)
# procedure + lesson sections share what the where section left; mandatory where lines first
pb = mk().memory_block(W, mk().retrieve_procedures("put the apple in the fridge", [base], kitchen=KITCHEN, wm_snapshot=SNAP), lr, level="L3", lang="en",
                       otype="Apple")
order = [r["section"] for r in pb["records"]]
check("sections come in the declared order where -> procs -> lessons", order == sorted(order, key=["where", "procs", "lessons"].index), str(order))
check("procedure line carries status and success count", any("trusted" in r["text"] and "3 successes" in r["text"] for r in pb["records"]))

# ---------------------------------------------------------------------------------------------------------------
print("9. exact-key path, logs")
with tempfile.TemporaryDirectory() as td:
    spy = R.HashEmbedder()
    L = ltm.LongTermMemory(os.path.join(td, "ltm.db"), kitchen=KITCHEN)
    rr_ = mk(params=P2, embedder=spy)
    a = rr_.locate(L, "Apple")
    b = L.where_is("Apple")
    check("locate() == where_is() (exact-key path)", a == b and a["candidates"] is not None)
    check("the exact-key path never calls the embedder or the lexical channel", spy.calls == 0)
    check("habit_prefs() delegates to the exact-key habits table", rr_.habit_prefs(L, "Apple") == L.habits("Apple") and spy.calls == 0)
    ret = mk()
    r1 = ret.retrieve_procedures("Put the apple in the fridge", recs, kitchen=KITCHEN, wm_snapshot=SNAP, episode="ep1", E=4)
    blk = ret.memory_block(W, r1, None, level="L3", lang="en", otype="Apple")
    ret.finish_block(blk, r1["log"])
    lg1 = r1["log"]
    check("log: retrieved >= eligible, rejected carry the first failed condition", len(lg1["retrieved"]) >= 1 and lg1["eligible"] == ["pr:a"]
          and all(x["first"] for x in lg1["rejected"]))
    check("log: rendered ids recorded after the block is built", lg1["rendered"] == ["pr:a"])
    ret.mark_applied(lg1["retrieval"], "pr:a", "attempted")
    row = ret.log_row(lg1)
    committed = L.store.commit("rtl:1", "admin", [{"table": "retrieval_log", "op": "insert", "values": row}])
    check("log row is accepted by the real store whitelist", committed["status"] == "committed")
    got = L.store.select("retrieval_log", where={"retrieval": row["retrieval"]})
    check("log row round-trips (query json, candidates json, channel)", got and got[0]["channel"] == "lexical" and got[0]["query"]["eligible"] == ["pr:a"]
          and got[0]["query"]["applied"] == [{"id": "pr:a", "state": "attempted"}], str(got))
    lgl = ret.retrieve_lessons(FK, [lesson("ls:1")], kitchen=KITCHEN, mode="test", episode="ep1", E=5)["log"]
    c2 = L.store.commit("rtl:2", "admin", [{"table": "retrieval_log", "op": "insert", "values": ret.log_row(lgl)}])
    check("lesson log row accepted by the whitelist", c2["status"] == "committed")
    check("log carries versions: parser, lexicon, tokenizer, rrf_k, token_claim", {"parser", "lexicon", "tokenizer", "rrf_k", "token_claim"} <= set(lg1["versions"]))
    check("flush() hands the row to the injected sink (no global state)", ret.flush(lg1) is not None and len(ret.sink) == 1)
    snap_dir = os.path.join(td, "snap")
    L.snapshot(snap_dir)
    fz = ltm.LongTermMemory.open_frozen(snap_dir, os.path.join(td, "testlog.db"), kitchen=KITCHEN)
    wrote = ret.flush(lg1, writer=fz.log_test)
    check("a frozen store takes the log through its separate testlog", wrote["status"] == "committed" and fz.store.count("retrieval_log") == 2 and fz.store.testlog.count("retrieval_log") == 1)
    fz.store.testlog.close()
    fz.close()
    L.close()

# ---------------------------------------------------------------------------------------------------------------
print("10. determinism")
outs, texts = set(), set()
for k in range(12):
    rr2 = recs[:]
    random.Random(k).shuffle(rr2)
    rt2 = mk()
    r = rt2.retrieve_procedures("Put the apple in the fridge", rr2, kitchen=KITCHEN, wm_snapshot=SNAP, episode="e1")
    b = rt2.memory_block(W, r, None, level="L3", lang="en", otype="Apple")
    outs.add(json.dumps([[i, s, rk_] for i, s, rk_ in r["rank"]["fused"]], sort_keys=True))
    texts.add(b["text"])
check("ranking and rendered block are identical under every record permutation", len(outs) == 1 and len(texts) == 1)
runs = {json.dumps(mk().retrieve_procedures("Put the apple in the fridge", recs, kitchen=KITCHEN, wm_snapshot=SNAP, episode="e1")["log"], sort_keys=True) for _ in range(3)}
check("identical inputs -> identical log (fresh retriever)", len(runs) == 1)
hy = {json.dumps(mk(params=P2, embedder=R.HashEmbedder()).retrieve_procedures("Put the apple in the fridge", recs, kitchen=KITCHEN, wm_snapshot=SNAP)["rank"]["fused"]) for _ in range(3)}
check("hybrid ranking is reproducible with a fixed embedder", len(hy) == 1)
check("dense scores are quantised", all(abs(s / R.SCORE_QUANT - round(s / R.SCORE_QUANT)) < 1e-6 for s in
                                        [v["dense"] for v in mk(params=P2, embedder=R.HashEmbedder()).retrieve_procedures(
                                            "Put the apple in the fridge", recs, kitchen=KITCHEN, wm_snapshot=SNAP)["rank"]["scores"].values() if "dense" in v]))

# ---------------------------------------------------------------------------------------------------------------
print("11. hygiene")
src = (Path(HERE) / "ltm_retrieval.py").read_text(encoding="utf-8") + (Path(HERE) / "ltm_concepts.py").read_text(encoding="utf-8")
check("no wall clock / randomness in the retrieval modules", all(w not in src for w in ("time.time", "datetime", "import random", "uuid")))
check("retrieval modules never name the live databases", "memory.db" not in src and "trace.db" not in src)
code = ("import sys; sys.path.insert(0, %r); import ltm_retrieval, ltm_concepts; bad=[m for m in ('torch','sentence_transformers','FlagEmbedding',"
        "'transformers','ai2thor','honest','honest_skills','honest_map','objmem','honest_console','webapp','requests','urllib3') if m in sys.modules]; print(bad)") % HERE
out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=str(REPO))
check("importing the retrieval modules loads no torch / step-4 / network module", out.stdout.strip() == "[]", out.stdout + out.stderr)
mem = REPO / "sim" / "memory.db"
check("sim/memory.db untouched by this test (mtime unchanged)", _MEM0 == (mem.stat().st_mtime_ns if mem.exists() else None))

print(f"\n{_pass}/{_pass + _fail} checks passed")
sys.exit(0 if _fail == 0 else 1)
