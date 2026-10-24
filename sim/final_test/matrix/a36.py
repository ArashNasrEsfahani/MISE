"""A36 (SIGNED by Arash 2026-10-15, signatures.json key "A36", text docs/final-test/amendments/A36-DRAFT.md r2; in force for the locked test
once the freeze binds it) -- the `excluded_reset` flag and the common-subset analysis. Pure: no simulator, no network, no journal writes.
Astra review of the first implementation: docs/final-test/_astra/A36-IMPL-analysis-r1.txt (all six blocking items + note 8 applied).

Flag (A36 rule 2). A core cell (canon world, r0, run0, block "main", an executed arm) is `excluded_reset` iff ALL hold:
  (a) its final registered class (A21 precedence) is `infra_exec`;
  (b) every one of its 3 registered attempts (0, 1, 2; the latest epoch) ended at the pre-execution reset gate, i.e. the registered comparison
      failed and was not overridden under A34-A2, and no plan step was dispatched;
  (c) each attempt's evidence is present, well-formed and complete: one exec row AND one `exec_attempt_start` event per attempt number.
Anything else (a mixed history, a crash at any stage, a hold / setup block, an attempt that started but left no row -- a hang --, a missing,
malformed or uncertain field) -> NOT flagged; registered scoring. Never raises.

What an exec row proves (execute.run_one + step5_live.LiveExecutor.execute). The only path that writes status `blocked_state` together with
state_check.ok False AND registered_ok False is the post-sweep `state_check` failure, which returns through `_unexecuted(reason=
"blocked_state")` BEFORE the plan hook is called: no plan is handed over, so no plan step can run. The other `blocked_state` paths are told
apart and never flagged: the hook's holding re-check (SummaryMismatch: state_check.ok True, prims None) and `inapplicable` at restore
(registered_ok None, prims None). Corroboration required as well: prims == the sweep's primitive count, 0 `action-resolve` audit events,
no outcome row. A hang (hang_guard exit 75) writes NO exec row and the resume repeats the attempt number; execute.do_cell therefore journals
`exec_attempt_start` (events.jsonl) before every attempt, and a start without its row (or two starts for one attempt number) blocks the flag.
"""
from __future__ import annotations

from collections import Counter

A36_VERSION = "a36-r2"
A36_STATUS = "SIGNED (Arash, 2026-10-15; signatures.json key A36; A36-DRAFT.md r2); binds the locked test once the freeze records it"
ATTEMPTS = (0, 1, 2)
RESET_LIMITED_SHARE = 0.05             # rule 6: MORE than 5 % of an arm's core cells flagged -> every contrast with that arm is "reset-limited"
STAGE_RESET_GATE = "reset_gate"
START_EVENT = "exec_attempt_start"
NO_RETAINED = "inconclusive: fewer than 2 families retained in the common subset (no claim)"


def _int(x):
    return x if isinstance(x, int) and not isinstance(x, bool) else None


# ================================================================================================ the flag (scoring side)
def attempt_stage(row) -> tuple:
    """-> (stage, problems). stage "reset_gate" only when the row PROVES a stop at the pre-execution reset gate with no plan step dispatched;
    otherwise "done" / "crash" / "blocked_other" / "evidence_incomplete" with the reasons."""
    if not isinstance(row, dict):
        return "evidence_incomplete", ["no exec row"]
    st = row.get("status")
    if st == "crash":
        return "crash", [f"crash: {str(row.get('error'))[:80]}"]
    if st in ("done", "not_executed", "invalid"):
        return str(st), [f"status {st}"]
    if st != "blocked_state":
        return "evidence_incomplete", [f"unknown status {st!r}"]
    why = []
    if row.get("error") is not None:
        why.append("error recorded")
    sc = row.get("state_check")
    if not isinstance(sc, dict):
        return "evidence_incomplete", why + ["no state_check"]
    if sc.get("ok") is not False:
        why.append(f"state_check.ok is {sc.get('ok')!r} (not the registered comparison failing)")
    if sc.get("registered_ok") is not False:
        why.append(f"registered_ok is {sc.get('registered_ok')!r}")
    if sc.get("override") is not False:
        why.append(f"override is {sc.get('override')!r}")
    br = sc.get("breaches")
    if not isinstance(br, list) or not br:
        why.append("no breach recorded")
    obs = sc.get("obs_pipeline")
    sweep = _int(obs.get("sweep_prims")) if isinstance(obs, dict) else None
    prims = _int(row.get("prims"))
    if sweep is None:
        why.append("sweep primitive count missing")
    if prims is None:
        why.append("primitive count missing")
    elif sweep is not None and prims != sweep:
        why.append(f"prims {prims} != sweep {sweep} (something ran after the sweep)")
    aud = row.get("audit")
    counts = aud.get("counts") if isinstance(aud, dict) else None
    if not isinstance(counts, dict) or "action-resolve" not in counts:
        why.append("audit counts missing")
    elif counts.get("action-resolve") != 0:
        why.append(f"audit action-resolve = {counts.get('action-resolve')!r} (a plan step resolved an object)")
    if why:
        kind = "blocked_other" if (sc.get("ok") is not False or sc.get("registered_ok") is None) else "evidence_incomplete"
        return kind, why
    return STAGE_RESET_GATE, []


def breach_kinds(row) -> list:
    sc = (row or {}).get("state_check") if isinstance(row, dict) else None
    br = sc.get("breaches") if isinstance(sc, dict) else None
    if not isinstance(br, list):
        return []
    return [str(b.get("kind", "?")) if isinstance(b, dict) else str(b) for b in br]


def history_of(rows) -> list:
    """All exec rows of one cell key -> the rows of its LATEST epoch (execute.cell_state ignores older epochs), journal order kept.
    Raises ValueError on a malformed epoch (the caller turns that into 'not flagged')."""
    rows = list(rows or ())
    if not rows:
        return []
    if any(not isinstance(r, dict) or _int(r.get("epoch", 0)) is None for r in rows):
        raise ValueError("an exec row is not a record or has a malformed epoch")
    ep = max(r.get("epoch", 0) for r in rows)
    return [r for r in rows if r.get("epoch", 0) == ep]


def current_epoch(history, starts, epoch=None) -> int:
    """The epoch the evidence must come from: the LATEST of the executor's epoch state for the presentation (`epoch`, = the number of
    invalid_resolved events naming it, execute._epochs), the rows' epochs and the starts' epochs (Astra A36-IMPL-r2: a start in a newer
    epoch with no row must not leave the older epoch's complete history in charge). Raises ValueError on a malformed value."""
    eps = [] if epoch is None else [epoch]
    for r in list(history or ()):
        if not isinstance(r, dict) or _int(r.get("epoch", 0)) is None:
            raise ValueError("an exec row is not a record or has a malformed epoch")
        eps.append(r.get("epoch", 0))
    for e in list(starts or ()):
        if not isinstance(e, dict) or _int(e.get("epoch", 0)) is None:
            raise ValueError("an attempt-start event is not a record or has a malformed epoch")
        eps.append(e.get("epoch", 0))
    if epoch is not None and _int(epoch) is None:
        raise ValueError("malformed executor epoch")
    return max(eps) if eps else 0


def _flag(run_class, history, starts, outcome_present, core, epoch=None) -> dict:
    out = {"flag": False, "rule": A36_VERSION, "reason": None, "attempts": []}
    if not core:
        out["reason"] = "not a core cell (main block, canon world, r0, run0)"
        return out
    if run_class != "infra_exec":
        out["reason"] = f"class {run_class!r} is not infra_exec"
        return out
    ep = current_epoch(history, starts, epoch)
    hist = [r for r in list(history or ()) if r.get("epoch", 0) == ep]
    out["epoch"] = ep
    if not hist:
        out["reason"] = f"no exec rows in the current epoch {ep}"
        return out
    att = [r.get("attempt") for r in hist]
    for r in hist:
        st, why = attempt_stage(r)
        out["attempts"].append({"attempt": r.get("attempt"), "status": r.get("status"), "stage": st, "breaches": breach_kinds(r),
                                "problems": why})
    if any(_int(a) is None for a in att):
        out["reason"] = f"malformed attempt id(s) {att!r}"
        return out
    if sorted(att) != list(ATTEMPTS):
        out["reason"] = f"retry history incomplete or duplicated (attempts {sorted(att)})"
        return out
    if starts is None:
        out["reason"] = f"no {START_EVENT} evidence (executor predates it): a lost (hung) attempt cannot be ruled out"
        return out
    st_ep = []
    for e in starts:
        if not isinstance(e, dict) or _int(e.get("attempt")) is None or _int(e.get("epoch", 0)) is None:
            out["reason"] = "malformed attempt-start event"
            return out
        if e.get("epoch", 0) == ep:
            st_ep.append(e["attempt"])
    out["starts"] = sorted(st_ep)
    if Counter(st_ep) != Counter(att):
        out["reason"] = (f"attempt starts {sorted(st_ep)} do not match the journaled attempts {sorted(att)} "
                         "(an attempt began and left no row: hang / interruption)")
        return out
    if outcome_present:
        out["reason"] = "an outcome row exists for this cell"
        return out
    bad = [a for a in out["attempts"] if a["stage"] != STAGE_RESET_GATE]
    if bad:
        out["reason"] = "mixed history: attempt " + ", ".join(f"{a['attempt']} = {a['stage']}" for a in bad)
        return out
    out["flag"], out["reason"] = True, "all 3 attempts started once and stopped at the pre-execution reset gate"
    return out


def excluded_reset(run_class, history, *, starts=None, outcome_present=False, core=True, epoch=None) -> dict:
    """-> {"flag", "reason", "attempts": [{attempt, status, stage, breaches, problems}], "starts"}. `history` = every exec row of the cell's
    unit/run; `starts` = its `exec_attempt_start` events (None: not recorded); `epoch` = the executor's epoch state for the presentation.
    Only the CURRENT epoch (current_epoch: latest of rows, starts and `epoch`) counts. Conservative and total: any
    doubt or malformed input -> flag False with the reason, never an exception."""
    try:
        return _flag(run_class, history, starts, outcome_present, core, epoch)
    except Exception as e:                                                   # noqa: BLE001 - malformed evidence keeps registered scoring
        return {"flag": False, "rule": A36_VERSION, "reason": f"malformed evidence ({type(e).__name__}: {str(e)[:120]})", "attempts": []}


# ================================================================================================ the analysis (rules 3-6)
def flagged(r) -> bool:
    return bool((r or {}).get("excluded_reset") is True)


def _arm_sr(AN, S, fam_of, lvl_of, arm, drop_invalid, drop):
    """Registered weighting (analyze.arm_rates): per item the mean of its retained EN / FA cells, then equal item weights."""
    Sx = {k: r for k, r in S.items() if k[0] == arm and r.get("sr") is not None and not (drop and flagged(r))}
    n_pres = sum(1 for (a, i, l) in Sx if (i, l) not in drop_invalid)
    ar = AN.arm_rates(Sx, fam_of, lvl_of, arm, drop_invalid) if n_pres else None
    return {"sr": (ar or {}).get("sr"), "n_items": (ar or {}).get("n_items", 0), "n_families": (ar or {}).get("n_families", 0),
            "n_presentations": n_pres}


def arm_flags(S, arms, drop_invalid, fam_of=None, lvl_of=None) -> dict:
    import analyze as AN
    fam_of = fam_of or {}
    lvl_of = lvl_of or {}
    out = {}
    for arm in arms:
        rows = {(i, l): r for (a, i, l), r in S.items() if a == arm and r.get("sr") is not None}
        fl = sorted(f"{i}:{l}" for (i, l), r in rows.items() if flagged(r))
        kinds = {}
        for (i, l), r in rows.items():
            if flagged(r):
                for at in ((r.get("excluded_reset_evidence") or {}).get("attempts") or ()):
                    for k in at.get("breaches") or ():
                        kinds[k] = kinds.get(k, 0) + 1
        n = len(rows)
        share = (len(fl) / n) if n else 0.0
        out[arm] = {"core_cells": n, "flagged": len(fl), "share": share, "reset_limited": share > RESET_LIMITED_SHARE,
                    "flagged_presentations": fl, "breach_kinds": dict(sorted(kinds.items()))}
        if fam_of:
            out[arm]["sr_primary"] = _arm_sr(AN, S, fam_of, lvl_of, arm, drop_invalid, True)
            out[arm]["sr_registered"] = _arm_sr(AN, S, fam_of, lvl_of, arm, drop_invalid, False)
    return out


def retained(S, fam_of, a, b, drop_invalid, drop=True) -> dict:
    """Explicit counts of the paired subset a contrast uses (zeros, never None)."""
    import analyze as AN
    diffs, info = AN.item_diffs(S, a, b, drop_invalid, drop_flag=flagged if drop else None)
    return {"presentations": int(info["n_pres"]), "items": len(diffs), "families": len({fam_of[i] for i in diffs}),
            "K": len({fam_of[i] for i in info["_disc_items"] if i in diffs}), "dropped_flagged": int(info["dropped_flagged"]),
            "missing": int(info["missing_pres"])}


def a36_block(S, fam_of, drop_invalid, B, complete, *, lvl_of=None, registered_h1=None, registered_holm=None, registered_exploratory=None,
              registered_equivalence=None) -> dict:
    """Rules 3-6 for EVERY contrast (H1, the m = 3 Holm family, the exploratory H4a-b / H5a-c, the exploratory equivalence): paired common
    subset (a presentation flagged in EITHER compared arm leaves BOTH), the registered aggregation via analyze (item_diffs / h1_rule_c7 /
    Holm), the registered result alongside, explicit retained counts, the reset-limited label. Exploratory stays exploratory."""
    import analyze as AN
    arms = sorted({a for (a, i, l) in S})
    flags = arm_flags(S, arms, drop_invalid, fam_of, lvl_of)
    out = {"rule": A36_VERSION, "status": A36_STATUS, "any_flagged": any(v["flagged"] for v in flags.values()), "by_arm": flags,
           "reset_limited_threshold": RESET_LIMITED_SHARE,
           "note": ("Primary = paired common subset (a flagged presentation is unavailable in that arm and leaves both arms of every contrast "
                    "with it); registered = A21 (infra_exec = SR 0, in every denominator). Agreement between the two does not show the "
                    "exclusions were unbiased. h1-rule-1's validation did not cover this selection step.")}
    lim = lambda a, b: [x for x in (a, b) if (flags.get(x) or {}).get("reset_limited")]       # noqa: E731
    cnt = lambda a, b: {"amended": retained(S, fam_of, a, b, drop_invalid), "registered": retained(S, fam_of, a, b, drop_invalid, False)}  # noqa: E731
    a, b = AN.H1
    if a in arms and b in arms:
        h = AN.h1_block(S, fam_of, drop_invalid, B, complete, drop_flag=flagged)
        p = h.get("primary")
        wording = h.get("wording") if p else NO_RETAINED
        reg_w = (registered_h1 or {}).get("wording")
        out["H1"] = {"contrast": h["contrast"], "available": p is not None, "primary": p, "family_weighted": h.get("family_weighted"),
                     "infra_paired": h.get("infra_paired"), "wording": wording, "retained": cnt(a, b),
                     "registered_wording": reg_w, "differs_from_registered": (reg_w is not None and wording != reg_w), "reset_limited": lim(a, b)}
    else:
        out["H1"] = {"available": False, "wording": f"unavailable: arms {a} and {b} both needed", "retained": cnt(a, b), "reset_limited": lim(a, b),
                     "differs_from_registered": False}
    hf = AN.holm_family(S, fam_of, drop_invalid, B, complete, drop_flag=flagged)
    out["holm_family"] = {}
    for hid, d in hf.items():
        reg = (registered_holm or {}).get(hid) or {}
        ha, hb = AN.FAMILY[hid][:2]
        out["holm_family"][hid] = dict(d, retained=cnt(ha, hb), registered_reject=reg.get("reject"), registered_estimate=reg.get("estimate"),
                                       differs_from_registered=("reject" in reg and reg.get("reject") != d.get("reject")),
                                       reset_limited=lim(ha, hb))
    out["holm_members"] = sorted(hf)
    ex, _ = AN.exploratory_contrasts(S, fam_of, drop_invalid, B, hf, drop_flag=flagged)
    out["exploratory_contrasts"] = {}
    for hid, d in ex.items():
        ea, eb = AN.EXPLORATORY[hid][:2]
        reg = (registered_exploratory or {}).get(hid) or {}
        out["exploratory_contrasts"][hid] = dict(d, retained=cnt(ea, eb), registered_estimate=reg.get("estimate"), reset_limited=lim(ea, eb))
    eq = AN.equivalence_block(S, fam_of, drop_invalid, B, complete, drop_flag=flagged)
    out["equivalence_exploratory"] = dict(eq, retained=cnt("G+M", "O+M"), reset_limited=lim("G+M", "O+M"),
                                          registered_wording=((registered_equivalence or {}).get("decision") or {}).get("wording"))
    if not eq.get("available"):
        out["equivalence_exploratory"]["wording"] = NO_RETAINED
    out["differences"] = ([f"H1 wording: amended '{out['H1'].get('wording')}' vs registered '{out['H1'].get('registered_wording')}'"]
                          if out["H1"].get("differs_from_registered") else []) + \
                         [f"{k} reject: amended {v.get('reject')} vs registered {v['registered_reject']}" for k, v in out["holm_family"].items()
                          if v["differs_from_registered"]]
    return out


def render_md(blk) -> list:
    if not blk or not blk.get("rule"):
        return []
    r3 = lambda x: "n/a" if x is None else f"{x:.3f}"            # noqa: E731
    pp = lambda x: "n/a" if x is None else f"{100 * x:+.1f} pp"    # noqa: E731
    rc = lambda t: f"{t['presentations']} / {t['items']} / {t['families']} / {t['K']}"   # noqa: E731
    L = ["", f"## A36 amended subset analysis ({blk['rule']}; {blk['status']})", "", blk["note"], "",
         "| arm | core cells | flagged | share | reset-limited | SR primary (items / pres) | SR registered (items / pres) |", "|---|---|---|---|---|---|---|"]
    for arm, v in blk["by_arm"].items():
        sp, sg = v.get("sr_primary") or {}, v.get("sr_registered") or {}
        L.append(f"| {arm} | {v['core_cells']} | {v['flagged']} | {100 * v['share']:.1f} % | {'yes' if v['reset_limited'] else 'no'} | "
                 f"{r3(sp.get('sr'))} ({sp.get('n_items', 0)} / {sp.get('n_presentations', 0)}) | {r3(sg.get('sr'))} ({sg.get('n_items', 0)} / {sg.get('n_presentations', 0)}) |")
    h = blk.get("H1") or {}
    t = (h.get("retained") or {}).get("amended") or {}
    L.append(f"\nH1 (amended subset): **{h.get('wording')}**; registered: {h.get('registered_wording')}; retained pres / items / families / K "
             f"{rc(t) if t else 'n/a'}" + (f"; reset-limited: {', '.join(h['reset_limited'])}" if h.get("reset_limited") else ""))
    L += ["", "| id | contrast | estimate (amended) | estimate (registered) | p Holm | reject (amended) | reject (registered) | pres / items / fam / K | reset-limited |",
          "|---|---|---|---|---|---|---|---|---|"]
    for hid, v in blk["holm_family"].items():
        L.append(f"| {hid} | {v.get('a')} - {v.get('b')} | {pp(v.get('estimate'))} | {pp(v.get('registered_estimate'))} | {v.get('p_holm', 1.0):.4f} | "
                 f"{v.get('reject')} | {v.get('registered_reject')} | {rc(v['retained']['amended'])} | {', '.join(v['reset_limited']) or 'no'} |")
    L += ["", "Exploratory (raw p, no Holm, no confirmatory conclusion):", "", "| id | contrast | estimate (amended) | estimate (registered) | pres / items / fam / K | reset-limited |",
          "|---|---|---|---|---|---|"]
    for hid, v in blk["exploratory_contrasts"].items():
        L.append(f"| {hid} | {v.get('a')} - {v.get('b')} | {pp(v.get('estimate'))} | {pp(v.get('registered_estimate'))} | {rc(v['retained']['amended'])} | "
                 f"{', '.join(v['reset_limited']) or 'no'} |")
    eq = blk.get("equivalence_exploratory") or {}
    L.append(f"\nEquivalence G+M vs O+M (exploratory): amended {((eq.get('decision') or {}).get('wording')) or eq.get('wording')}; registered "
             f"{eq.get('registered_wording')}; retained {rc(eq['retained']['amended'])}" + (f"; reset-limited: {', '.join(eq['reset_limited'])}" if eq.get("reset_limited") else ""))
    L.append("\nDifferences between the amended and the registered analysis: " + ("; ".join(blk["differences"]) if blk["differences"] else "none."))
    return L
