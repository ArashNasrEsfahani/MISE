"""W2: real-token re-measure of the re-captured (A11, post-sweep summary) planner prompts, and the cost re-projection.

For every ok run-1 canonical capture row of a namespace and every prompt source it holds (none, S*, ...) it renders the messages with the one renderer
(`PlannerContract`, exactly as plan.py does), counts them with Gemma (Ollama `prompt_eval_count`, num_predict 1) and with the Anthropic
`count_tokens` endpoint (FREE; key only through `planners.keys`, never printed), and writes ``docs/final-test/matrix/<ns>/token_remeasure.json``.
Cost: tokens_in x list price + the pilot's mean output tokens x list price (docs/final-test/pilot.md section 3), scaled to the registered call plan.
Counts are of the messages only (no JSON-schema constraint overhead). Windows python; Ollama must be running for the Gemma column.

    python sim/final_test/matrix/w2_tokens.py --namespace rehearsal-v1 [--ollama http://127.0.0.1:11434]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                                       # noqa: E402

import honest_skills as hs                                               # noqa: E402
from planners import PlannerContract                                     # noqa: E402
from planners import keys as K                                           # noqa: E402

GEMMA = "gemma4:e2b-it-q4_K_M"
ANTH_MODEL = "claude-sonnet-5-5"
#: pilot (docs/final-test/pilot.md s3): mean tokens in / out per call (pre-A11 prompts, no observation summary) and list prices (sim/planners/prices.json)
PILOT = {"gemma_in_0": 876, "gemma_in_M": 918, "anth_in_0": 1699, "anth_in_M": 1739,
         "out_opus": 105, "out_sonnet": 117, "out_haiku": 107, "in_haiku_0": 1341}
PRICE = {"opus": (4.0, 20.0), "sonnet": (2.0, 10.0), "haiku": (1.0, 5.0)}
#: first-pass calls of the registered plan (docs/final-test/pilot.md s4)
CALLS = {"core_opus": 740, "core_sonnet": 740, "whole_opus": 1810, "whole_sonnet": 740, "whole_haiku": 740}


def gemma_count(endpoint, msgs):
    body = {"model": GEMMA, "messages": msgs, "stream": False, "think": False,
            "options": {"temperature": 0, "seed": 7, "num_ctx": 8192, "num_predict": 1, "top_k": 1}}
    req = urllib.request.Request(endpoint.rstrip("/") + "/api/chat", json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"})
    return int(json.loads(urllib.request.urlopen(req, timeout=300).read())["prompt_eval_count"])


def anth_count(key, msgs):
    sys_txt = "\n".join(m["content"] for m in msgs if m["role"] == "system")
    rest = [m for m in msgs if m["role"] != "system"]
    body = {"model": ANTH_MODEL, "messages": rest}
    if sys_txt:
        body["system"] = sys_txt
    req = urllib.request.Request("https://api.anthropic.com/v1/messages/count_tokens", json.dumps(body).encode("utf-8"),
                                 {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    return int(json.loads(urllib.request.urlopen(req, timeout=60).read())["input_tokens"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    a = ap.parse_args(argv)
    if C.ns_check(a.namespace)["items"] == "test":
        raise SystemExit("this tool is for rehearsal / practice namespaces only")
    types = json.load(open(os.path.join(C.paths()["docs"], "pilot", "types.json"), encoding="utf-8"))
    types = sorted(set(types["types"] if isinstance(types, dict) else types))
    contract = PlannerContract(types, tools=hs.TOOLS)
    key = K.get_anthropic_key(None).reveal()
    rows = [r for r in C.Journal(C.journal_path(a.namespace, "capture")).rows()
            if r.get("status") == "ok" and r.get("capture_run") == 1 and r.get("world") == "canon"]
    out, per = [], {}
    for r in rows:
        for src in ("none", "S*"):
            blk = (r.get("blocks") or {}).get(src) or {}
            mem = (blk.get("text") or None) if src != "none" else None
            msgs = contract.messages(r["instruction"], r.get("summary") or [], r.get("holding"), memory_block=mem, fewshot_block=None)
            g, an = gemma_count(a.ollama, msgs), anth_count(key, msgs)
            out.append({"pres": r["pres"], "level": r["level"], "lang": r["lang"], "source": src, "gemma_in": g, "anthropic_in": an,
                        "summary_lines": len(r.get("summary") or []), "block_empty": not bool(mem)})
            per.setdefault(src, []).append((g, an))
    res = {"schema": "ftm.w2-token-remeasure/1", "namespace": a.namespace, "n_presentations": len(rows), "rows": out, "summary": {}}
    for src, v in per.items():
        res["summary"][src] = {"n": len(v), "gemma_mean": round(statistics.mean(x[0] for x in v), 1), "gemma_max": max(x[0] for x in v),
                               "anthropic_mean": round(statistics.mean(x[1] for x in v), 1), "anthropic_max": max(x[1] for x in v)}
    a0, aM = res["summary"]["none"]["anthropic_mean"], res["summary"]["S*"]["anthropic_mean"]
    mean_in = (a0 + aM) / 2.0
    def usd(model, tin, tout):
        pi, po = PRICE[model]
        return (tin * pi + tout * po) / 1e6
    # Haiku's tokenizer differs; scale its pilot input by the same relative growth as the API tokenizer's
    grow = mean_in / ((PILOT["anth_in_0"] + PILOT["anth_in_M"]) / 2.0)
    per_call = {"opus": usd("opus", mean_in, PILOT["out_opus"]), "sonnet": usd("sonnet", mean_in, PILOT["out_sonnet"]),
                "haiku": usd("haiku", PILOT["in_haiku_0"] * grow, PILOT["out_haiku"])}
    pilot_call = {"opus": usd("opus", 1719, PILOT["out_opus"]), "sonnet": usd("sonnet", 1719, PILOT["out_sonnet"]),
                  "haiku": usd("haiku", PILOT["in_haiku_0"], PILOT["out_haiku"])}
    core = CALLS["core_opus"] * per_call["opus"] + CALLS["core_sonnet"] * per_call["sonnet"]
    whole = CALLS["whole_opus"] * per_call["opus"] + CALLS["whole_sonnet"] * per_call["sonnet"] + CALLS["whole_haiku"] * per_call["haiku"]
    core0 = CALLS["core_opus"] * pilot_call["opus"] + CALLS["core_sonnet"] * pilot_call["sonnet"]
    whole0 = CALLS["whole_opus"] * pilot_call["opus"] + CALLS["whole_sonnet"] * pilot_call["sonnet"] + CALLS["whole_haiku"] * pilot_call["haiku"]
    res["projection"] = {"basis": "anthropic count_tokens of the A11 prompts (mean over none and S* sources) + pilot mean output tokens; first-pass calls only, no repairs, no thinking tail",
                         "mean_in_tokens": round(mean_in, 1), "pilot_mean_in_tokens": 1719.0, "growth_vs_pilot": round(grow, 3),
                         "usd_per_call": {k: round(v, 5) for k, v in per_call.items()}, "pilot_usd_per_call": {k: round(v, 5) for k, v in pilot_call.items()},
                         "core_usd": round(core, 2), "whole_usd": round(whole, 2), "pilot_core_usd_same_method": round(core0, 2),
                         "pilot_whole_usd_same_method": round(whole0, 2), "ceilings": dict(C.NAMESPACES["test-v1"]["ceilings"]),   # A22: 30 / 65 (the W2 file written on 2026-10-11 recorded the then-current 25 / 45)
                        
                         "thinking_tail_core_at_500": 11.1, "thinking_tail_whole_at_500": 22.2}
    res["pilot_gemma_in"] = {"0": PILOT["gemma_in_0"], "M": PILOT["gemma_in_M"]}
    p = os.path.join(C.ns_docs(a.namespace), "token_remeasure.json")
    text = json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True)
    if C.key_leak(text):
        raise SystemExit("key-like text in the output: not written")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    print(json.dumps({"summary": res["summary"], "projection": res["projection"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
