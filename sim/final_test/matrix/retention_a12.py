"""W2 / A12 + A12a: measure H, head, r, s from token counts ONLY and (with --write) put the allocation into the step-5 registry.

No outcome field is read. Place candidates come from a frozen practice snapshot (non-test), rendered one at a time for the block types of every
non-test instruction in EN and FA at every level (`ltm.a12_measure(candidates=)`); counts are the max over Gemma (Ollama) and the API tokenizer
(free count_tokens, key only from THESIS_ANTHROPIC_API_KEY). The allocation is `a12_allocation(H_eff, r, s=s)` (A12a).

    python sim/final_test/matrix/retention_a12.py --measure [--snapshot DIR] [--ollama URL] [--write]

--measure   writes docs/final-test/matrix/a12_measure.json (always) and compares it with the W3 overlay docs/final-test/w3/a12.json (same method).
--write     also sets retrieval.retention = "a12" and retrieval.a12 = {...} in sim/step5/params.json (old file copied to
            sim/final_test/_checkpoints/W2-rehearsal/params.json.pre-a12 first; refused if the registry already holds an a12 block with other numbers).
The coverage criterion itself (A12: >= N_L whole records in every tokenizer x language on the re-captured dev presentations) is checked by capture.py
(gate `a12_coverage`) once the registry carries the block.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                    # noqa: E402  (puts sim/ on sys.path)

ROOT = C.ROOT
sys.path.insert(0, os.path.join(C.SIM, "final_test", "c3"))
OUT = os.path.join(C.paths()["matrix_docs"], "a12_measure.json")
W3 = os.path.join(C.paths()["docs"], "w3", "a12.json")
PARAMS = os.path.join(C.SIM, "step5", "params.json")
CKPT = os.path.join(C.SIM, "final_test", "_checkpoints", "W2-rehearsal")


def measure(snapshot: str, tokcache: str, ollama: str, budget: dict | None = None, kitchen: str = C.DEFAULT_SCENE) -> dict:
    import ltm as L
    import ltm_adapter as la
    import ltm_retrieval as lr
    from planners.contract import MEMORY_HEADER
    from real_counter import RealMaxCounter
    man = C.load_manifest()
    gen = {i["id"]: i for i in json.load(open(C.paths()["dataset"], encoding="utf-8"))["items"]}
    ids = sorted(man["nontest"])
    assert not (set(ids) & (set(man["test"]) | set(man["test_uncapped_not_scored"]))), "test item in the measurement set"
    counter = RealMaxCounter(tokcache, MEMORY_HEADER, ollama_endpoint=ollama)
    assert len(counter.tokenizers) >= 2, counter.tokenizers
    P = L.load_params()
    tmp = tempfile.mkdtemp(prefix="w2a12-")
    ltm = L.LongTermMemory.open_frozen(snapshot, os.path.join(tmp, "t.db"), kitchen=kitchen, params=P)
    ltm.count_tokens = counter
    ltm.begin_test_epoch(frame="f0", map_rev="m0")
    cands, n_pres = [], 0
    for iid in ids:
        it = gen[iid]
        for lang in ("en", "fa"):
            wm = ltm.begin_episode(f"w2a12-{iid}-{lang}", "test")
            ad = la.HonestEventAdapter(ltm, wm=wm, episode=f"w2a12-{iid}-{lang}", mode="test", world="reset", frame="f0", map_rev="m0",
                                       kitchen=kitchen)
            ad.on_task(it[lang], it["level"], lang)
            cands.append((ltm, wm))
            n_pres += 1
    m = L.a12_measure([], counter, candidates=cands)
    registered = {k: int(v) for k, v in P["retrieval"]["budget_tokens"]["value"].items()}
    if budget:                                     # A16: one fixed block budget at every level (replaces 150/300/450 as the A12 B_L)
        registered = {k: int(budget[k]) for k in registered}
    alloc = lr.a12_allocation(m["H_eff"], m["r"], B=registered, where_share=float(P["retrieval"]["where_share"]["value"]), s=m["s"])
    res = {"schema": "final-test.a12-measure/1", "rule": "prereg s16 A12 + A12a: H_eff = H + head; need_L = H_eff + N_L x (r + s)",
           "measured_on": {"snapshot": os.path.relpath(snapshot, ROOT).replace("\\", "/"), "snapshot_state_hash": ltm.state_hash(),
                           "instructions": len(ids), "presentations": n_pres, "dev_pilot_included": True, "fp1_measured": kitchen == C.DEFAULT_SCENE,
                           **({} if kitchen == C.DEFAULT_SCENE else {"kitchen": kitchen})},
           "tokenizers": counter.tokenizers, "measure": m, "A": alloc["A"], "B": alloc["B"], "N": alloc["N"], "need": alloc["need"],
           "raised": alloc["raised"], "incapable": alloc["incapable"], "not_applied": alloc["not_applied"],
           "reason": "A12, token-derived (outcome-blind)" + ("; B fixed by A16" if budget else ""), "params_sha256_base": P.get("_sha256")}
    counter.save()
    ltm.close()
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A12 / A12a token measurement")
    ap.add_argument("--measure", action="store_true", required=True)
    ap.add_argument("--snapshot", default=os.path.join(C.SIM, "final_test", "c3", "snapshot"))
    ap.add_argument("--tokcache", default=os.path.join(C.ns_work("rehearsal-v1"), "tokcache.json"))
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--budget", type=int, default=None, help="A16: one block budget B (real tokens) for every level, e.g. 800")
    ap.add_argument("--replace", action="store_true", help="A16: replace the existing a12 block (restarts from the pre-a12 checkpoint copy)")
    ap.add_argument("--kitchen", default=C.DEFAULT_SCENE, help="K23: measure on a replication kitchen's snapshot (measure only; "
                    "the registered allocation is FloorPlan1's and is never rewritten from another kitchen)")
    a = ap.parse_args(argv)
    if a.kitchen != C.DEFAULT_SCENE and (a.write or a.replace):
        print("REFUSED: --write / --replace are FloorPlan1-only (the registered A12 allocation is not rewritten from a replication kitchen)")
        return 2
    os.makedirs(os.path.dirname(a.tokcache), exist_ok=True)
    bud = {lv: a.budget for lv in ("L1", "L2", "L3")} if a.budget else None
    res = measure(a.snapshot, a.tokcache, a.ollama, bud, kitchen=a.kitchen)
    raw = json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True)
    res["sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    cmp_ = None
    out = OUT if a.kitchen == C.DEFAULT_SCENE else os.path.join(os.path.dirname(OUT), f"a12_measure_{a.kitchen}.json")
    if a.kitchen == C.DEFAULT_SCENE and os.path.exists(W3):
        w3 = json.load(open(W3, encoding="utf-8"))
        cmp_ = {k: (w3.get(k) == res.get(k)) for k in ("A", "B", "N", "raised", "incapable")}
        cmp_["measure_equal"] = all(w3["measure"].get(k) == res["measure"].get(k) for k in ("H", "head", "H_eff", "r", "s", "n_candidates_measured"))
        res["w3_overlay_compare"] = cmp_
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(json.dumps({k: res[k] for k in ("measure", "A", "B", "raised", "incapable", "w3_overlay_compare") if k in res}, ensure_ascii=False, indent=1))
    if a.write:
        os.makedirs(CKPT, exist_ok=True)
        pre0 = os.path.join(CKPT, "params.json.pre-a12")
        if a.replace:
            assert os.path.exists(pre0), "no pre-a12 checkpoint copy to restart from"
            shutil.copyfile(PARAMS, os.path.join(CKPT, "params.json.pre-a16"))
            shutil.copyfile(pre0, PARAMS)
        old_raw = open(PARAMS, "rb").read()
        reg = json.loads(old_raw.decode("utf-8"))
        block = {"H": res["measure"]["H"], "head": res["measure"]["head"], "H_eff": res["measure"]["H_eff"], "r": res["measure"]["r"],
                 "s": res["measure"]["s"], "N": res["N"], "A": res["A"], "B": res["B"], "raised": res["raised"], "incapable": res["incapable"],
                 "reason": "A12, token-derived" + ("; B fixed by A16" if bud else ""), "fp1_measured": True, "measure_sha256": res["sha256"]}
        r = reg["retrieval"]
        if r.get("retention") == "a12" and r.get("a12") and {k: r["a12"].get(k) for k in ("A", "B")} != {"A": block["A"], "B": block["B"]}:
            raise SystemExit("the registry already holds an a12 block with other numbers: refused (a new amendment is needed)")
        if r.get("retention") == "a12":
            print("params.json already carries this a12 block: nothing to write")
            return 0
        pre = os.path.join(CKPT, "params.json.pre-a12")
        if not os.path.exists(pre):
            shutil.copyfile(PARAMS, pre)
        # text-level insertion at the end of the "retrieval" object, so that every other byte of the frozen registry stays as it was
        marker = '    }\r\n  },\r\n  "eval"'
        raw_txt = old_raw.decode("utf-8")
        if raw_txt.count(marker) != 1:
            raise SystemExit("params.json layout not recognised (retrieval / eval marker): not written")
        body = json.dumps(block, ensure_ascii=False, indent=2).replace("\n", "\n    ").replace("\n", "\r\n")
        ins = '    },\r\n    "retention": "a12",\r\n    "a12": ' + body + '\r\n  },\r\n  "eval"'
        text = raw_txt.replace(marker, ins)
        json.loads(text)                                           # still valid JSON
        with open(PARAMS, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        print("params.json written; sha256 %s (was %s)" % (hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                                                           hashlib.sha256(old_raw).hexdigest()[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
