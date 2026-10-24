"""Selftest for the Anthropic Message Batches path (stage BATCH-A29): sim/planners/anthropic_batch.py + sim/final_test/matrix/plan_batch.py.

    python sim/_planners_batch_selftest.py

Offline: no network, no key, no simulator. A FAKE batch client stands in for the API; the real AnthropicPlanner (with a fake transport) is
the synchronous reference, so "same body / same repair frames / same cache / same journal row" is checked against the real sync code.
Synthetic dev-item captures only (made-up instruction text).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "final_test", "matrix"))

import common as C                                                  # noqa: E402
import plan as PL                                                   # noqa: E402
import plan_batch as PB                                             # noqa: E402
import planfirst as PF                                              # noqa: E402
import honest_skills as hs                                          # noqa: E402
from planners import CostLedger, PlannerContract, PlannerRunner, PriceTable, ResponseCache, make_planner, sha256_json   # noqa: E402
from planners import anthropic_batch as AB                          # noqa: E402
from planners.anthropic_provider import AnthropicPlanner            # noqa: E402
from planners.base import CostCeilingExceeded                       # noqa: E402
from planners.keys import Secret                                    # noqa: E402

N = {"pass": 0, "fail": 0}


def check(name, cond, detail=""):
    N["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -- {detail}"))


FAKEKEY = "sk-ant-FAKEKEY0123456789abcdefBATCHTEST"
TYPES = ["Apple", "Bread", "Fridge", "Knife", "Mug", "Plate", "Sink"]
PRICES = PriceTable.load()


# ================================================================================================ replies (shared by sync + batch fakes)
def idx_of_body(body):
    txt = body["messages"][0]["content"].split("Command:")[-1]
    for tok in txt.replace(".", " ").replace("،", " ").split():
        if tok.isdigit():
            return int(tok)
    return -1


class World:
    """What the fake model answers: idx in bad_first -> the first reply is invalid, a repair is valid; idx in reject -> always invalid;
    bad_twice -> the first two replies invalid (ft-v2 second repair)."""

    def __init__(self, bad_first=(), reject=(), bad_twice=(), out=40, cache_read=0):
        self.bad_first, self.reject, self.bad_twice = set(bad_first), set(reject), set(bad_twice)
        self.out, self.cache_read = out, cache_read

    def text(self, body):
        i, k = idx_of_body(body), (len(body["messages"]) - 1) // 2          # k = repair number
        if i in self.reject or (i in self.bad_first and k == 0) or (i in self.bad_twice and k < 2):
            return "this is not json"
        return json.dumps({"say": "ok", "plan": [{"action": "pickup", "object": "Apple", "to": None, "arg": None}]})

    def message(self, body, mid="msg_x"):
        u = {"input_tokens": 1000 + 7 * len(body["messages"]), "output_tokens": self.out}
        if self.cache_read:
            u["cache_read_input_tokens"] = self.cache_read
        return {"id": mid, "type": "message", "model": body["model"], "content": [{"type": "thinking", "thinking": ""},
                {"type": "text", "text": self.text(body)}], "stop_reason": "end_turn", "usage": u}


class SyncFake:
    def __init__(self, world):
        self.w, self.bodies = world, []

    def __call__(self, url, headers, body, timeout):
        assert headers["x-api-key"] == FAKEKEY
        self.bodies.append(json.loads(json.dumps(body)))
        return 200, {}, self.w.message(body, f"msg_s{len(self.bodies)}")


class FakeBatch:
    """The batch API: create -> in_progress for `polls` retrieves -> ended; results by custom_id (any order: reversed)."""

    def __init__(self, world, polls=1, plan=None):
        self.w, self.polls = world, polls
        self.batches, self.creates, self.retrieves, self.cancels = {}, [], 0, []
        self.plan = plan or (lambda cid, body, nth: "succeeded")    # -> result type for the nth submission of this custom_id
        self.seen = {}
        self.crash_on_retrieve = False

    def create(self, requests):
        bid = f"msgbatch_{len(self.creates) + 1:03d}"
        self.creates.append(json.loads(json.dumps(requests)))
        self.batches[bid] = {"requests": requests, "left": self.polls}
        return {"id": bid, "type": "message_batch", "processing_status": "in_progress"}

    def retrieve(self, bid):
        self.retrieves += 1
        if self.crash_on_retrieve:
            self.crash_on_retrieve = False
            raise Crash("synthetic kill while waiting")
        b = self.batches[bid]
        if b["left"] > 0:
            b["left"] -= 1
            return {"id": bid, "processing_status": "in_progress"}
        return {"id": bid, "processing_status": "ended", "results_url": f"https://x/{bid}/results",
                "request_counts": {"succeeded": len(b["requests"])}}

    def cancel(self, bid):
        self.cancels.append(bid)
        return {"id": bid, "processing_status": "canceling"}

    def results(self, b):
        out = []
        for r in reversed(self.batches[b["id"]]["requests"]):
            cid = r["custom_id"]
            self.seen[cid] = self.seen.get(cid, 0) + 1
            t = self.plan(cid, r["params"], self.seen[cid])
            if t == "succeeded":
                out.append({"custom_id": cid, "result": {"type": "succeeded", "message": self.w.message(r["params"], f"msg_b{cid[:6]}")}})
            elif t == "invalid":
                out.append({"custom_id": cid, "result": {"type": "errored", "error": {"type": "error", "error": {"type": "invalid_request_error"}}}})
            elif t == "api_error":
                out.append({"custom_id": cid, "result": {"type": "errored", "error": {"type": "error", "error": {"type": "api_error"}}}})
            else:
                out.append({"custom_id": cid, "result": {"type": t}})
        return out


class Crash(BaseException):
    pass


def ct1():
    return PlannerContract(TYPES, tools=hs.TOOLS)


def ct2():
    return PlannerContract(TYPES, tools=hs.TOOLS, version="ft-v2")


def cmd(i, lang="en"):
    return f"Pick up the apple number {i}." if lang == "en" else f"سیب شماره {i} را بردار."


def tmp():
    return tempfile.mkdtemp(prefix="ftm_batch_")


def sync_plan(world, contract, ids, root, model="claude-opus-5-5", led=None):
    fake = SyncFake(world)
    prov = AnthropicPlanner(model, cache=ResponseCache(os.path.join(root, "cache_sync")), ledger=led or CostLedger(os.path.join(root, "s.db")),
                            prices=PRICES, transport=fake, env={"THESIS_ANTHROPIC_API_KEY": FAKEKEY}, sleep=lambda s: None)
    run = PlannerRunner(prov, contract)
    return {i: run.plan(cmd(i), ["Fridge [closed]"], None) for i in ids}, fake


def batch_plan(world, contract, ids, root, model="claude-opus-5-5", fb=None, led=None, cache_dir="cache_batch", max_retries=3):
    led = led or CostLedger(os.path.join(root, "b.db"))
    prov = AB.BatchPlanner(model, cache=ResponseCache(os.path.join(root, cache_dir)), ledger=led, prices=PRICES)
    run = PlannerRunner(prov, contract)
    fb = fb or FakeBatch(world)
    journal = []
    drv = AB.BatchDriver(fb, led, journal=lambda e, **kw: journal.append({"event": e, **kw}), poll_s=0, sleep=lambda s: None,
                         max_retries=max_retries)
    units = [{"id": f"cell-{i}", "provider": prov, "run": (lambda i=i: run.plan(cmd(i), ["Fridge [closed]"], None))} for i in ids]
    out = drv.run(units)
    return {int(k.split("-")[1]): v for k, v in out.items()}, fb, drv, led, journal, prov


SAME = ("ok", "plan", "first_ok", "repaired", "rejected", "infra_failure", "calls", "tokens_in", "tokens_out", "say", "first_plan")

# ================================================================================================ 1. static
for path in (AB.__file__, PB.__file__):
    src = open(path, encoding="utf-8").read()
    try:
        ast.parse(src, feature_version=(3, 10))
        ok = True
    except SyntaxError:
        ok = False
    check(f"{os.path.basename(path)} parses as python 3.10", ok)
    check(f"{os.path.basename(path)}: no key literal, no environment read of a key",
          not re.search(r"sk-[A-Za-z0-9_-]{8,}", src) and "environ" not in src)
check("the protected files are not imported for editing: anthropic_batch only subclasses AnthropicPlanner",
      issubclass(AB.BatchPlanner, AnthropicPlanner))

# ================================================================================================ 2. custom_id
cid = AB.custom_id_for("rehearsal-v3|plan|O+M|canon:g-l1-x:en|r0")
check("custom_id: a cell key with | + : is mapped to a valid, deterministic id", AB.CUSTOM_ID_RE.fullmatch(cid) and
      cid == AB.custom_id_for("rehearsal-v3|plan|O+M|canon:g-l1-x:en|r0") and len(cid) <= 64, cid)
check("custom_id: a key the API accepts is used verbatim", AB.custom_id_for("cell-1_a") == "cell-1_a")
check("custom_id: two different cells never share an id", AB.custom_id_for("a|b") != AB.custom_id_for("a|c"))

# ================================================================================================ 3. same body, same parse
root = tmp()
for model in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5-20251001"):
    ct = ct2()
    msgs = ct.messages(cmd(3), ["Fridge [closed]"], None)
    sp = AnthropicPlanner(model, cache=ResponseCache(root), ledger=CostLedger(os.path.join(root, "x.db")), prices=PRICES)
    bp = AB.BatchPlanner(model, cache=ResponseCache(root), ledger=CostLedger(os.path.join(root, "x.db")), prices=PRICES)
    check(f"{model}: batch body == sync body (model, system, messages, output_config, thinking, max_tokens)",
          sha256_json(sp.build_body(msgs, ct.schema)) == sha256_json(bp.build_body(msgs, ct.schema)))
    check(f"{model}: decoding identical (cache keys and decoding hashes match the sync path)", sp.decoding() == bp.decoding())
    check(f"{model}: batch reserve = {AB.BATCH_DISCOUNT} x sync reserve",
          abs(bp.reserve_usd(msgs, None) - AB.BATCH_DISCOUNT * sp.reserve_usd(msgs, None)) < 1e-12)
w = World(cache_read=500)
for stop, extra in (("end_turn", {}), ("max_tokens", {}), ("refusal", {"stop_details": {"category": "cyber"}}), ("tool_use", {})):
    body = AB.BatchPlanner("claude-opus-5-5", cache=ResponseCache(root), ledger=CostLedger(os.path.join(root, "y.db")),
                           prices=PRICES).build_body(ct2().messages(cmd(1), [], None), ct2().schema)
    msg = {**w.message(body), "stop_reason": stop, **extra}
    sp = AnthropicPlanner("claude-opus-5-5", cache=ResponseCache(root), ledger=CostLedger(os.path.join(root, "y.db")), prices=PRICES,
                          transport=lambda u, h, b, t, m=msg: (200, {}, m), env={"THESIS_ANTHROPIC_API_KEY": FAKEKEY})
    sp._reserve_amt = 0.0
    sync = sp._call(ct2().messages(cmd(1), [], None), ct2().schema, None)
    bat = AB.parse_message(msg, body, PRICES, "claude-opus-5-5")
    same = all(sync[k] == bat[k] for k in ("raw", "tokens_in", "tokens_out", "stop_reason", "error", "error_kind"))
    check(f"parse_message == AnthropicPlanner._call on the same message (stop_reason {stop})", same,
          {k: (sync[k], bat[k]) for k in ("raw", "tokens_in", "tokens_out", "stop_reason", "error", "error_kind") if sync[k] != bat[k]})
    check(f"batch cost = 0.5 x list (cache reads included), list cost = the sync cost (stop {stop})",
          abs(bat["cost_usd"] - 0.5 * sync["cost_usd"]) < 1e-9 and abs(bat["list_cost_usd"] - sync["cost_usd"]) < 1e-9)
shutil.rmtree(root, ignore_errors=True)

# ================================================================================================ 4. rounds = sync repairs (ft-v1 and ft-v2)
for label, ctf, world in (("ft-v1", ct1, World(bad_first={2}, reject={3})), ("ft-v2", ct2, World(bad_first={2}, reject={3}, bad_twice={4}))):
    root = tmp()
    ids = [1, 2, 3, 4]
    s_res, sfake = sync_plan(world, ctf(), ids, root)
    b_res, fb, drv, led, journal, prov = batch_plan(world, ctf(), ids, root)
    check(f"{label}: every result field equals the sync run (ok, plan, first_ok, repaired, rejected, calls, tokens)",
          all(s_res[i][k] == b_res[i][k] for i in ids for k in SAME),
          [(i, k, s_res[i][k], b_res[i][k]) for i in ids for k in SAME if s_res[i][k] != b_res[i][k]][:4])
    sb = sorted(sha256_json(b) for b in sfake.bodies)
    bb = sorted(sha256_json(r["params"]) for c in fb.creates for r in c)
    check(f"{label}: the multiset of request bodies (first calls AND repair frames) is byte-identical to the sync path", sb == bb,
          (len(sb), len(bb)))
    nrounds = 1 + (2 if label == "ft-v2" else 1)
    check(f"{label}: one batch per round ({nrounds} rounds: first calls, then repairs)", len(fb.creates) == nrounds, len(fb.creates))
    check(f"{label}: batch cost = 0.5 x sync cost per cell", all(abs(b_res[i]["cost_usd"] - 0.5 * s_res[i]["cost_usd"]) < 1e-6 for i in ids),
          [(b_res[i]["cost_usd"], s_res[i]["cost_usd"]) for i in ids])
    rows = led.rows()
    check(f"{label}: ledger: no open reservation, settled = sum of cell costs",
          not [r for r in rows if r["state"] == "open"] and
          abs(sum(r["actual"] or 0 for r in rows if r["state"] == "settled") - sum(b_res[i]["cost_usd"] for i in ids)) < 1e-6)
    ents = [json.loads(r["entry"]) for r in rows if r["state"] == "settled"]
    check(f"{label}: every settled ledger entry is the sync format + billing_tier batch + list_cost_usd",
          all({"provider", "model", "prompt_sha256", "stage", "tokens_in", "tokens_out", "cost_usd", "cached"} <= set(e) and
              e["billing_tier"] == "batch" and abs(e["list_cost_usd"] * 0.5 - e["cost_usd"]) < 1e-6 for e in ents))
    check(f"{label}: results arriving in reverse order are matched by custom_id", all(b_res[i]["ok"] == (i not in (3,)) for i in ids))
    # the same cache: a synchronous provider on the batch cache replays without transmitting
    replay_fake = SyncFake(world)
    sp = AnthropicPlanner("claude-opus-5-5", cache=ResponseCache(os.path.join(root, "cache_batch")), ledger=CostLedger(os.path.join(root, "r.db")),
                          prices=PRICES, transport=replay_fake, env={"THESIS_ANTHROPIC_API_KEY": FAKEKEY})
    rr = PlannerRunner(sp, ctf()).plan(cmd(2), ["Fridge [closed]"], None)
    check(f"{label}: the sync path replays batch answers from the same response cache (0 transmissions, cached)",
          not replay_fake.bodies and rr["cached_calls"] == rr["calls"] and rr["plan"] == b_res[2]["plan"])
    check(f"{label}: a batch answer carries batch_id, custom_id, list cost and the wire sha",
          all(a["result"]["extra"].get("batch_id") and a["result"]["extra"].get("custom_id") and
              a["result"]["extra"].get("request_sha256") for i in ids for a in b_res[i]["attempts"]))
    check(f"{label}: submission records journal every custom_id -> cell mapping",
          all(it["cell"].startswith("cell-") and it["custom_id"] for e in journal if e["event"] == "batch_submitted" for it in e["items"]))
    shutil.rmtree(root, ignore_errors=True)

# ================================================================================================ 5. unbilled results and retries
root = tmp()
plan_exp = lambda cid, body, nth: "expired" if (cid == "cell-1" and nth == 1) else ("canceled" if cid == "cell-2" and nth <= 2 else "succeeded")
b_res, fb, drv, led, journal, prov = batch_plan(World(), ct2(), [1, 2, 3], root, fb=FakeBatch(World(), plan=plan_exp))
check("expired / canceled results are resubmitted and then succeed", all(b_res[i]["ok"] for i in (1, 2, 3)) and drv.stats["resubmitted"] == 3,
      drv.stats)
check("unbilled results release their reservation (no open rows; released rows exist)",
      not [r for r in led.rows() if r["state"] == "open"] and len([r for r in led.rows() if r["state"] == "released"]) == 3)
shutil.rmtree(root, ignore_errors=True)

root = tmp()
b_res, fb, drv, led, journal, prov = batch_plan(World(), ct2(), [1, 2], root,
                                                fb=FakeBatch(World(), plan=lambda cid, b, n: "expired" if cid == "cell-1" else "succeeded"))
check("expired more than max_retries (3) times -> a transport failure -> infra (sync rule for unbilled statuses)",
      b_res[1]["infra_failure"] and not b_res[1]["ok"] and len(fb.creates) == 4 and b_res[2]["ok"], (len(fb.creates), b_res[1]["infra_failure"]))
check("an infra answer is never cached", not any(os.path.exists(os.path.join(root, "cache_batch", a["result"]["cache_key"][:2],
                                                                             a["result"]["cache_key"] + ".json")) for a in b_res[1]["attempts"]))
shutil.rmtree(root, ignore_errors=True)

root = tmp()
b_res, fb, drv, led, journal, prov = batch_plan(World(), ct2(), [1], root, fb=FakeBatch(World(), plan=lambda cid, b, n: "invalid"))
check("errored invalid_request_error -> not retried (= a sync 4xx), infra", b_res[1]["infra_failure"] and len(fb.creates) == 1)
shutil.rmtree(root, ignore_errors=True)
root = tmp()
b_res, fb, drv, led, journal, prov = batch_plan(World(), ct2(), [1], root,
                                                fb=FakeBatch(World(), plan=lambda cid, b, n: "api_error" if n == 1 else "succeeded"))
check("errored api_error -> resubmitted once, then ok", b_res[1]["ok"] and len(fb.creates) == 2)
shutil.rmtree(root, ignore_errors=True)

# ================================================================================================ 6. cost guard at submission
root = tmp()
led = CostLedger(os.path.join(root, "c.db"), global_ceiling_usd=0.05)
fb = FakeBatch(World())
try:
    batch_plan(World(), ct2(), [1, 2, 3], root, fb=fb, led=led)
    refused = False
except CostCeilingExceeded:
    refused = True
check("submission reservation over the ceiling -> CostCeilingExceeded, NOTHING created, no open reservation",
      refused and not fb.creates and not [r for r in led.rows() if r["state"] == "open"])
shutil.rmtree(root, ignore_errors=True)


class AmbClient(FakeBatch):
    def create(self, requests):
        raise AB.BatchSubmitAmbiguous("synthetic timeout after send")


root = tmp()
led = CostLedger(os.path.join(root, "a.db"))
try:
    batch_plan(World(), ct2(), [1, 2], root, fb=AmbClient(World()), led=led)
    amb = False
except AB.BatchSubmitAmbiguous:
    amb = True
check("an ambiguous create keeps the reservations OPEN (may have been billed)", amb and len([r for r in led.rows() if r["state"] == "open"]) == 2)
shutil.rmtree(root, ignore_errors=True)

# ================================================================================================ 7. BatchClient HTTP + redaction
calls = []


def tr(method, url, headers, body, timeout):
    calls.append((method, url, headers.get("x-api-key") == FAKEKEY, body))
    if url.endswith("/batches") and method == "POST":
        return 200, {}, json.dumps({"id": "msgbatch_1", "processing_status": "in_progress"})
    if url.endswith("/results"):
        return 200, {}, '{"custom_id":"a","result":{"type":"expired"}}\n\n{"custom_id":"b","result":{"type":"canceled"}}\n'
    if "/bad" in url:
        return 401, {}, json.dumps({"error": {"message": f"bad key {FAKEKEY}"}})
    return 200, {}, json.dumps({"id": "msgbatch_1", "processing_status": "ended", "results_url": "https://h/v1/messages/batches/msgbatch_1/results"})


bc = AB.BatchClient(lambda: Secret(FAKEKEY), transport=tr, sleep=lambda s: None)
b = bc.create([{"custom_id": "a", "params": {}}])
r = bc.retrieve(b["id"])
res = bc.results(r)
check("BatchClient: POST create, GET retrieve, GET results_url JSONL (blank lines skipped); the key only in the header",
      calls[0][0] == "POST" and calls[0][1] == AB.BATCH_URL and calls[1][0] == "GET" and len(res) == 2 and all(c[2] for c in calls))
bc2 = AB.BatchClient(lambda: Secret(FAKEKEY), transport=tr, url="https://h/bad", sleep=lambda s: None)
try:
    bc2.retrieve("x")
    msg = ""
except Exception as e:                                               # noqa: BLE001
    msg = str(e)
check("BatchClient: a non-retryable HTTP error is raised with the key redacted", msg and FAKEKEY not in msg, msg)
n = {"k": 0}


def tr500(method, url, headers, body, timeout):
    n["k"] += 1
    return 500, {}, "{}"


try:
    AB.BatchClient(lambda: Secret(FAKEKEY), transport=tr500, sleep=lambda s: None).create([{"custom_id": "a", "params": {}}])
    amb = False
except AB.BatchSubmitAmbiguous:
    amb = True
check("BatchClient: a 500 on create is AMBIGUOUS (never blindly re-posted)", amb and n["k"] == 1)
n["k"] = 0
try:
    AB.BatchClient(lambda: Secret(FAKEKEY), transport=tr500, sleep=lambda s: None, retries=2).retrieve("x")
except Exception:                                                    # noqa: BLE001
    pass
check("BatchClient: a 500 on an idempotent GET is retried (retries + 1 tries)", n["k"] == 3, n["k"])

# timeout -> cancel once
fb = FakeBatch(World(), polls=5)
clock = {"t": 0.0}
drv = AB.BatchDriver(fb, CostLedger(os.path.join(tempfile.mkdtemp(), "t.db")), poll_s=10, wait_timeout_s=25, sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
                     now=lambda: clock["t"])
fb.batches["msgbatch_x"] = {"requests": [], "left": 5}
drv.wait("msgbatch_x", 0.0)
check("wait: past the timeout the batch is cancelled exactly once, then polled until ended", fb.cancels == ["msgbatch_x"])

# ================================================================================================ 8. BatchPlan on a synthetic namespace
SYN = "synthetic-batch-v1"
MANIFEST = C.load_manifest()
DEV = list(MANIFEST["dev_pilot"])
C.NAMESPACES[SYN] = {"items": "dev_pilot", "ceilings": {"core": 50.0, "total": 100.0}, "archived_arms": ()}


def build_capture(root, ns=SYN):
    ct = ct1()
    J = C.Journal(C.journal_path(ns, "capture", root))
    for it in DEV:
        for lg in ("en", "fa"):
            i = DEV.index(it)
            rec = {"instruction": cmd(i, lg), "summary": ["Fridge [closed] at (canon)", "Apple on CounterTop"]}
            blocks, reqs = {}, {}
            for s in ("none", "S*", "FS"):
                text = "" if s == "none" else f"memory line for {s} ({lg})"
                blocks[s] = {"text": text, "ids": [], "real_tokens": len(text), "budget": 150, "empty": not text, "coverage": {}}
                m = ct.messages(rec["instruction"], rec["summary"], None, memory_block=(text or None) if s == "S*" else None,
                                fewshot_block=(text or None) if s == "FS" else None)
                reqs[s] = hashlib.sha256(json.dumps([[x["role"], x["content"]] for x in m], ensure_ascii=False,
                                                    separators=(",", ":")).encode("utf-8")).hexdigest()
            pres = C.pres_id(it, lg)
            J.append(C.make_record("capture", key=C.cell_id(ns, "capture", pres, "c1"), namespace=ns, pres=pres, item=it, lang=lg,
                                   level=f"L{1 + i % 3}", world="canon", capture_run=1, instruction=rec["instruction"], holding=None,
                                   summary=rec["summary"], summary_sha=C.sha256_text("\n".join(rec["summary"])), blocks=blocks,
                                   requests=reqs, state_ref=f"states/{pres}.c1.json", world_hash="h", audit={}))
    C.Journal(C.journal_path(ns, "events", root)).append(
        C.make_record("events", event="phase_complete", phase="capture", n=len(J.rows()), journal_sha=C.sha256_file(J.path)))


def new_root():
    root = tempfile.mkdtemp(prefix="ftm_batchplan_")
    d = os.path.join(root, "docs", "final-test")
    os.makedirs(d)
    shutil.copy(C.paths()["split"], d)
    shutil.copy(C.paths()["families"], d)
    inc = os.path.join(C.paths()["docs"], getattr(C, "INCLUSION_NAME", "") or "-")
    if os.path.isfile(inc):                                          # A26 (K-B2b): the effective split = the manifest minus archived ids
        shutil.copy(inc, d)
    json.dump(TYPES, open(os.path.join(root, "types.json"), "w"))
    build_capture(root)
    return root


def sync_factory(world, fake=None):
    fake = fake or SyncFake(world)

    def f(spec, **kw):
        kw.setdefault("sleep", lambda s: None)
        return make_planner(spec, transport=fake, env={"THESIS_ANTHROPIC_API_KEY": FAKEKEY}, **kw)
    return f


def mkplan(root, cls=PL.Plan, **kw):
    kw.setdefault("types_path", os.path.join(root, "types.json"))
    return cls(SYN, passes=["core"], rs=[0], root=root, gpu_sample=False, key_getter=lambda: Secret(FAKEKEY), out=lambda m: None, **kw)


WORLD = World(bad_first={1}, reject={2})
r_sync, r_batch = new_root(), new_root()
code_s = mkplan(r_sync, provider_factory=sync_factory(WORLD)).run()
fbp = FakeBatch(WORLD)
code_b = mkplan(r_batch, PB.BatchPlan, batch_client=fbp, batch_poll_s=0, batch_sleep=lambda s: None).run()
check("BatchPlan core r0 completes (exit 0) like the sync Plan", code_s == 0 and code_b == 0, (code_s, code_b))
rs = {r["key"]: r for r in C.Journal(C.journal_path(SYN, "plan", r_sync)).rows()}
rb = {r["key"]: r for r in C.Journal(C.journal_path(SYN, "plan", r_batch)).rows()}
FIELDS = ("status", "plan", "first_plan", "first_ok", "repaired", "calls", "cached_calls", "tokens_in", "tokens_out", "request_sha", "decoding",
          "arm", "pres", "r", "prompt_version", "types_sha", "steps")
check("BatchPlan writes one plan row per O-cell, the same keys as the sync Plan (O0 / O+M; G cells are not in the core O selection)",
      set(rs) == set(rb) and len(rb) > 0, (len(rs), len(rb)))
diff = [(k, f) for k in rs for f in FIELDS if rs[k].get(f) != rb.get(k, {}).get(f)]
check("every plan-row field except cost / latency equals the sync row (status, plans, repairs, calls, tokens, request sha, decoding)",
      not diff, diff[:5])
check("plan-row cost = 0.5 x the sync row cost", all(abs(rb[k]["cost_usd"] - 0.5 * rs[k]["cost_usd"]) < 1e-6 for k in rs))
check("statuses exercise ok, repaired and rejected", {r["status"] for r in rb.values()} == {"ok", "rejected"} and any(r["repaired"] for r in rb.values()))
rep = PF.attempts_report(SYN, r_batch)
check("ledger vs journal reconciliation per cell (attempts_report): no diff, no open reservation", not rep["diffs"] and rep["ledger_open_usd"] == 0
      and abs(rep["journal_usd"] - rep["ledger_settled_usd"]) < 1e-6, rep["diffs"][:3])
evs = C.Journal(C.journal_path(SYN, "events", r_batch)).rows()
names = [e["event"] for e in evs]
check("events: batch_mode, batch_submitted, batch_collected, batch_report, pass_complete(core)",
      all(n in names for n in ("batch_mode", "batch_submitted", "batch_collected", "batch_report")) and
      any(e["event"] == "pass_complete" and e.get("pass") == "core" for e in evs), names)
check("2 rounds for ft-v1 (first calls + one repair round) in one chunk", len(fbp.creates) == 2, len(fbp.creates))
check("custom ids are the mapped cell keys (cell key -> ck_ digest), one per cell per batch",
      all(r["custom_id"] == AB.custom_id_for(it["cell"]) for e in evs if e["event"] == "batch_submitted"
          for r, it in zip(fbp.creates[0], e["items"]) if e["round"] == 0))
check("a sync resume over the batch namespace finds every cell terminal and transmits nothing",
      mkplan(r_batch, provider_factory=sync_factory(WORLD, SyncFake(WORLD))).run() == 0)

# ---- chunking + the guard at every chunk boundary
r3 = new_root()
fb3 = FakeBatch(WORLD)
code = mkplan(r3, PB.BatchPlan, batch_client=fb3, batch_poll_s=0, batch_sleep=lambda s: None, batch_max_cells=4).run()
n_cells = len(C.Journal(C.journal_path(SYN, "plan", r3)).rows())
check("batch_max_cells 4 splits the work into several chunks of whole presentations (2 O-cells each)", code == 0 and len(fb3.creates) >= n_cells // 4,
      (len(fb3.creates), n_cells))
subs = [e for e in C.Journal(C.journal_path(SYN, "events", r3)).rows() if e["event"] == "batch_submitted" and e["round"] == 0]
check("each chunk holds complete presentations (both arms of a presentation in the same batch)",
      all(len({it["cell"].split("|")[3] for it in s["items"]}) * 2 == len(s["items"]) for s in subs))

# ---- block hold refused at submission
C.NAMESPACES[SYN] = {"items": "dev_pilot", "ceilings": {"core": 1.0, "total": 2.0}, "archived_arms": ()}   # projection (prior) fits, the hold does not
r4 = new_root()
fb4 = FakeBatch(WORLD)
code = mkplan(r4, PB.BatchPlan, batch_client=fb4, batch_poll_s=0, batch_sleep=lambda s: None).run()
check("a chunk whose worst-case hold passes the ceiling is refused BEFORE any batch (exit 4, 0 creates, block_refused event)",
      code == PL.EXIT["ceiling"] and not fb4.creates and any(e["event"] == "block_refused" for e in C.Journal(C.journal_path(SYN, "events", r4)).rows()),
      (code, len(fb4.creates)))
C.NAMESPACES[SYN] = {"items": "dev_pilot", "ceilings": {"core": 50.0, "total": 100.0}, "archived_arms": ()}

# ---- the guard on ACTUAL spend at the chunk boundary (projected completion over the ceiling -> pause)
r5 = new_root()
C.NAMESPACES[SYN] = {"items": "dev_pilot", "ceilings": {"core": 50.0, "total": 100.0}, "archived_arms": ()}
fb5 = FakeBatch(World(out=4000))
pl5 = mkplan(r5, PB.BatchPlan, batch_client=fb5, batch_poll_s=0, batch_sleep=lambda s: None, batch_max_cells=2)
orig = PL.Plan._guard_eval
seen = []


def spy(self):
    g = orig(self)
    seen.append(self.ledger.billed_scope())
    return g


PL.Plan._guard_eval = spy
try:
    pl5.run()
finally:
    PL.Plan._guard_eval = orig
check("the cost guard runs at every chunk boundary on settled (actual) spend, which grows chunk by chunk",
      len(seen) >= 3 and seen[0] == 0 and seen[-1] > seen[1] > 0, seen[:5])

# ---- refusal of non-Anthropic cells
r6 = new_root()
try:
    PB.BatchPlan(SYN, passes=["gemma"], root=r6, gpu_sample=False, key_getter=lambda: Secret(FAKEKEY), out=lambda m: None,
                 types_path=os.path.join(r6, "types.json"), batch_client=FakeBatch(WORLD))._prepare()
    refused = False
except ValueError:
    refused = True
check("--batch refuses a selection with Gemma cells (plan those without --batch)", refused)

# ---- crash while waiting -> resume re-attaches (no second create of that batch) and the ledger reconciles
r7 = new_root()
fb7 = FakeBatch(WORLD)
fb7.crash_on_retrieve = True
try:
    mkplan(r7, PB.BatchPlan, batch_client=fb7, batch_poll_s=0, batch_sleep=lambda s: None).run()
    crashed = False
except Crash:
    crashed = True
open_after_crash = [r for r in C.ScopedLedger(os.path.join(C.ns_work(SYN, r7), "ledger.db"), C.NAMESPACES[SYN]["ceilings"],
                                               release_stale=False).rows() if r["state"] == "open" and not r["provider"].startswith("hold@")]
check("a kill while a batch is in flight leaves its reservations open (1 batch created)", crashed and len(fb7.creates) == 1 and open_after_crash,
      (crashed, len(fb7.creates), len(open_after_crash)))
code = mkplan(r7, PB.BatchPlan, batch_client=fb7, batch_poll_s=0, batch_sleep=lambda s: None).run()
ev7 = C.Journal(C.journal_path(SYN, "events", r7)).rows()
check("the next start re-attaches to the in-flight batch (batch_reattach) instead of re-submitting its round 0",
      code == 0 and any(e["event"] == "batch_reattach" for e in ev7) and
      sum(1 for e in ev7 if e["event"] == "batch_submitted" and any(it["stage"] == "plan" for it in e["items"])) == 1 and
      len(fb7.creates) == 2, [e["event"] for e in ev7][-12:])
rep7 = PF.attempts_report(SYN, r7)
check("after the resume: journal == ledger per cell, no open reservation", not rep7["diffs"] and rep7["ledger_open_usd"] == 0, rep7["diffs"][:3])
rb7 = {r["key"]: r for r in C.Journal(C.journal_path(SYN, "plan", r7)).rows()}
check("resumed rows equal the uninterrupted batch rows", all(rb7[k][f] == rb[k][f] for k in rb for f in FIELDS + ("cost_usd",)),
      [(k, f) for k in rb for f in FIELDS + ("cost_usd",) if rb7.get(k, {}).get(f) != rb[k][f]][:4])

# ---- crash AFTER collection but before the rows -> answers served as billed (not replays)
r8 = new_root()
fb8 = FakeBatch(WORLD)
orig_record = PL.Plan._record
state = {"n": 0}


def crash_record(self, *a, **k):
    state["n"] += 1
    if state["n"] == 1:
        raise Crash("synthetic kill before the first row")
    return orig_record(self, *a, **k)


PL.Plan._record = crash_record
try:
    try:
        mkplan(r8, PB.BatchPlan, batch_client=fb8, batch_poll_s=0, batch_sleep=lambda s: None).run()
    except Crash:
        pass
finally:
    PL.Plan._record = orig_record
code = mkplan(r8, PB.BatchPlan, batch_client=fb8, batch_poll_s=0, batch_sleep=lambda s: None).run()
rep8 = PF.attempts_report(SYN, r8)
ev8 = [e["event"] for e in C.Journal(C.journal_path(SYN, "events", r8)).rows()]
check("a kill after collection: the resume serves collected answers as BILLED (batch_answers_restored), journal == ledger",
      code == 0 and "batch_answers_restored" in ev8 and not rep8["diffs"] and rep8["ledger_open_usd"] == 0, rep8["diffs"][:3])

# ---- Astra A29 6: the unbilled-retry allowance survives a restart
r9 = new_root()
target = {}


def plan9(cid, body, nth):
    target.setdefault("cid", cid)
    return "expired" if cid == target["cid"] else "succeeded"


fb9 = FakeBatch(WORLD, plan=plan9)
orig_retrieve = fb9.retrieve
cnt = {"n": 0}


def retrieve9(bid):
    cnt["n"] += 1
    if cnt["n"] == 5:                                                 # batch 1: retrieves 1-2, batch 2: 3-4, batch 3: crash
        raise Crash("synthetic kill on the third batch")
    return orig_retrieve(bid)


fb9.retrieve = retrieve9
try:
    mkplan(r9, PB.BatchPlan, batch_client=fb9, batch_poll_s=0, batch_sleep=lambda s: None).run()
except Crash:
    pass
code = mkplan(r9, PB.BatchPlan, batch_client=fb9, batch_poll_s=0, batch_sleep=lambda s: None).run()
subs9 = sum(1 for c in fb9.creates for r in c if r["custom_id"] == target["cid"])
ev9 = [e["event"] for e in C.Journal(C.journal_path(SYN, "events", r9)).rows()]
row9 = [r for r in C.Journal(C.journal_path(SYN, "plan", r9)).rows() if AB.custom_id_for(r["key"]) == target["cid"]]
check("restart: earlier unbilled results still count -> exactly 4 submissions (1 + 3 retries), then infra",
      code == PL.EXIT["incomplete"] and subs9 == 4 and row9 and row9[-1]["status"] == "infra" and "batch_retry_counts_restored" in ev9,
      (code, subs9, row9[-1]["status"] if row9 else None))

# ---- Astra A29 6: an absent result line is unresolved (reservation kept, no retry, run stops)
class DropBatch(FakeBatch):
    dropped = False

    def results(self, b):
        out = FakeBatch.results(self, b)
        if not self.dropped:                                          # the first batch loses one result line; later batches are complete
            self.dropped = True
            return out[1:]
        return out


r10 = new_root()
fb10 = DropBatch(WORLD)
code = mkplan(r10, PB.BatchPlan, batch_client=fb10, batch_poll_s=0, batch_sleep=lambda s: None).run()
lg10 = C.ScopedLedger(os.path.join(C.ns_work(SYN, r10), "ledger.db"), C.NAMESPACES[SYN]["ceilings"], release_stale=False)
open10 = [r for r in lg10.rows() if r["state"] == "open" and not r["provider"].startswith("hold@")]
ev10 = [e["event"] for e in C.Journal(C.journal_path(SYN, "events", r10)).rows()]
check("a missing result line: abort (exit 1), its reservation stays OPEN, no resubmission, batch_result_missing recorded",
      code == PL.EXIT["abort"] and len(open10) == 1 and len(fb10.creates) == 1 and "batch_result_missing" in ev10, (code, len(open10), len(fb10.creates)))
code2 = mkplan(r10, PB.BatchPlan, batch_client=fb10, batch_poll_s=0, batch_sleep=lambda s: None).run()
check("... and the next start quarantines that cell until it is reconciled (exit 7), still no resubmission",
      code2 == PL.EXIT["quarantine"] and len([c for c in fb10.creates for r in c if r["custom_id"] in
                                              {AB.custom_id_for(x["cell"] or "") for x in lg10.open_reservations()["rows"]}]) == 1, code2)
for r_ in (r9, r10):
    shutil.rmtree(r_, ignore_errors=True)

# ================================================================================================ 9. CLI wiring
import inspect                                                        # noqa: E402
check("plan.py --batch selects plan_batch.BatchPlan (the default path is unchanged)",
      "--batch" in inspect.getsource(PL.main) and "plan_batch.BatchPlan" in inspect.getsource(PL.main))
mkf = PF.batch_selecting_factory({"batch_hold_usd": 3.0}, base_factory=lambda n, **kw: ("plain", kw.get("passes")),
                                 batch_cls=lambda n, **kw: ("batch", kw.get("passes"), kw.get("batch_hold_usd")))
check("planfirst --batch: core / fs / curve / reloc -> BatchPlan, gemma / xmain / codex -> the ordinary Plan",
      mkf(SYN, passes=["core"]) == ("batch", ["core"], 3.0) and mkf(SYN, passes=["fs"])[0] == "batch" and
      mkf(SYN, passes=["gemma", "xmain"])[0] == "plain" and mkf(SYN, passes=["codex"])[0] == "plain" and mkf(SYN, passes=["gemma"])[0] == "plain")
for r_ in (r_sync, r_batch, r3, r4, r5, r6, r7, r8):
    shutil.rmtree(r_, ignore_errors=True)
C.NAMESPACES.pop(SYN, None)

# ================================================================================================ 10. the A29 study projection
import final_test_callplan as CP                                      # noqa: E402
kc = CP.k23_cells(250, "FloorPlan15")
check("A29 Kitchen 2/3 call plan: the core four only, r = 0, 250 presentations each; only O0 / O+M are paid",
      [c["arm"] for c in kc] == ["G0", "G+M", "O0", "O+M"] and all(c["calls"] == 250 and c["replicates"] == 1 for c in kc) and
      sorted(c["arm"] for c in kc if c["scope"] != "free") == ["O+M", "O0"])
pa = CP.study_projection_a29(0.0, 250)
sc = pa["scenarios"][CP.GATE_SCENARIO]
check("A29 projection: list-price sensitivity = 2 x the batch total (no spend), Kitchens 2 and 3 equal",
      abs(sc["list_price"]["study_total_usd"] - 2 * sc["batch"]["study_total_usd"]) < 0.05 and sc["batch"]["kitchen2"] == sc["batch"]["kitchen3"])
pb = CP.study_projection_a29(10.0, 250)
check("A29 projection: spend so far adds 1:1 to the study total; the gate reads the S2 batch total",
      abs(pb["gate"]["study_total_usd"] - pa["gate"]["study_total_usd"] - 10.0) < 0.011 and
      pb["gate"]["study_total_usd"] == pb["scenarios"][CP.GATE_SCENARIO]["batch"]["study_total_usd"])
p100 = CP.study_projection_a29(0.0, 100)
check("A29 projection: Kitchen 1 is independent of the replication size; Kitchens 2/3 scale with it",
      p100["scenarios"][CP.GATE_SCENARIO]["batch"]["kitchen1"] == sc["batch"]["kitchen1"] and
      abs(p100["scenarios"][CP.GATE_SCENARIO]["batch"]["kitchen2"] * 2.5 - sc["batch"]["kitchen2"]) < 0.05)

print(f"\n{N['pass']}/{N['pass'] + N['fail']} {'OK' if not N['fail'] else 'FAILED'}")
sys.exit(1 if N["fail"] else 0)
