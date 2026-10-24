"""Cross-pipeline layout check (TODO A12; A28 item 5). Read-only.

Layout fingerprint per episode, from the stored episode records only: the sorted list of "instance name @ position"
of every settable instance, taken from ``episode_record["objects"]`` (name -> the objectId read after the spawn,
before that object's state is set). An AI2-THOR objectId is ``Type|x|y|z`` with the position after the spawn at
0.01 m (e.g. ``Apple|-02.01|+01.55|+01.14``); ``InitialRandomSpawn`` re-issues the ids of the objects it moves. So the
fingerprint is "sorted object name + position after the spawn rounded to 0.01 m" as registered. (A first version
used ``settled_metadata``: that is taken AFTER the state setup, so slicing / cracking changed the id set and the
default-layout TRAIN episodes of Kitchen 1 did not group -- it would have hidden collisions. Not used.)
Limitation (reported): stateless objects (e.g. Fork, SaltShaker) are not in the record, so two layouts that differ
only in stateless objects count as EQUAL -- the safe direction (a false collision, never a missed one).

Rule (A28 item 5): no val / test / holdout layout may equal ANY train layout, across all pipelines (main, top-up,
close, amb-v1, amb-v1-holdout). Train-train equality is expected (25 % of main TRAIN episodes keep the default
arrangement) and only counted. A collision is a STOP for that kitchen: no reseeding, no split reassignment, no extra
capture.

The holdout records are opened only to fingerprint them: the output holds episode numbers, pipeline names and
fingerprint hashes, never frame names or lists.

    python sim/kitchen_v3/k23/layout_fingerprint.py --scene FloorPlan15 [--thor-data <dir>] [--out report.json]
    exit 0 = no collision, 1 = collision, 2 = no episode found
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.abspath(os.path.join(HERE, "..", ".."))
if SIM not in sys.path:
    sys.path.insert(0, SIM)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import vocab  # noqa: E402


def default_thor_data() -> str:
    if os.name == "nt":
        return "//wsl.localhost/Ubuntu-22.04/home/arash/thor-data"
    return os.path.join(os.path.expanduser("~"), "thor-data")


def fingerprint(episode_record: dict) -> tuple:
    """(sha256, n_objects) of the episode's layout; see the module docstring.

    Source: ``episode_record["objects"]`` = {instance NAME: {"oid": objectId at state assignment}}. The oid is read
    after the spawn and BEFORE the object's own state is set, so a slice / crack / cook cannot change it, and the set
    of names is the probe's settable instances (the same in every episode of a kitchen and in both stores)."""
    ids = sorted(f"{name}@{(rec or {}).get('oid', '').split('|', 1)[-1]}"
                 for name, rec in (episode_record.get("objects") or {}).items())
    return hashlib.sha256("\n".join(ids).encode()).hexdigest(), len(ids)


POS_RE = re.compile(r"^[+-]?\d{2}\.\d{2}\|[+-]?\d{2}\.\d{2}\|[+-]?\d{2}\.\d{2}$")   # AI2-THOR writes zero as 00.00
MIN_COMPARABLE = 0.75     # share of the expected name set two records must share to be compared at all


def pipeline_of(er: dict) -> str:
    p = er.get("protocol") or "main"
    return {"main": "main", "topup": "topup-train"}.get(p, p)


def positions(er: dict) -> dict:
    return {name: (rec or {}).get("oid", "").split("|", 1)[-1] for name, rec in (er.get("objects") or {}).items()}


def expected_names(probe: dict) -> list:
    """The fixed name set every record should carry: the probe's settable instances (assign_states skips the
    knob-driven CONTROLLED types)."""
    out = []
    for name in probe.get("reachable") or {}:
        t = (probe.get("reset_ids") or {}).get(name, name).split("|", 1)[0]
        if vocab.base_type(t) not in vocab.CONTROLLED_TYPES:
            out.append(name)
    return sorted(out)


def episode_records(frames_path: str):
    """Only the episode-record rows; frame rows are skipped by a cheap prefix test (no JSON parse)."""
    with open(frames_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith('{"episode_record"'):
                yield json.loads(line)["episode_record"]


def scan(scene: str, thor_data: str) -> list:
    out = []
    for store in ("kitchen_v3", "kitchen_v3_amb"):
        p = os.path.join(thor_data, store, scene, "frames.jsonl")
        if not os.path.exists(p):
            continue
        for er in episode_records(p):
            fp, n = fingerprint(er)
            out.append({"store": store, "episode": er["episode"], "split": er["split"], "pipeline": pipeline_of(er),
                        "spawn_ok": (er.get("spawn") or {}).get("ok"), "fp": fp, "n_objects": n,
                        "pos": positions(er)})
    return out


def check(rows: list, expected: list | None = None) -> dict:
    """Collisions: a held-out record equal to a train record on every shared name, sharing >= MIN_COMPARABLE of the
    expected names. Records with unexpected names or malformed positions are PROBLEMS, and so is a held-out / train
    pair that shares too few names to be compared (the reduced fingerprint is only safe on a fixed, complete name set
    -- Astra K23-PREP 9). Failed held-out spawns and held-out duplicates make the check fail (investigate)."""
    exp = set(expected) if expected is not None else None
    n_exp = len(exp) if exp else 0
    problems = []
    for r in rows:
        bad_pos = sorted(n for n, p in r["pos"].items() if not POS_RE.match(p or ""))
        if bad_pos:
            problems.append(f"{r['pipeline']}/{r['episode']}: malformed positions {bad_pos[:3]}")
        r["coverage"], r["missing"] = 1.0, []
        if exp is not None:
            extra = sorted(set(r["pos"]) - exp)
            miss = sorted(exp - set(r["pos"]))
            if extra:
                problems.append(f"{r['pipeline']}/{r['episode']}: unexpected names {extra[:3]}")
            r["coverage"] = (n_exp - len(miss)) / n_exp if n_exp else 1.0
            r["missing"] = miss
    train = [r for r in rows if r["split"] == "train"]
    by_fp = {}
    for r in train:
        by_fp.setdefault(r["fp"], []).append(r)
    coll, undecidable = [], []
    held = [r for r in rows if r["split"] != "train"]
    for h in held:
        for t in train:
            common = set(h["pos"]) & set(t["pos"])
            need = MIN_COMPARABLE * (n_exp or max(len(h["pos"]), len(t["pos"]), 1))
            if len(common) < need:
                undecidable.append((h["pipeline"], h["episode"], t["pipeline"], t["episode"], len(common)))
                continue
            if all(h["pos"][n] == t["pos"][n] for n in common):
                coll.append({"pipeline": h["pipeline"], "split": h["split"], "episode": h["episode"],
                             "equals_train": [t["pipeline"], t["episode"]], "shared_names": len(common),
                             "exact": h["fp"] == t["fp"]})
    heldout_dups = []
    for i, h in enumerate(held):              # the same shared-name rule as held-out vs train
        for g in held[i + 1:]:
            common = set(h["pos"]) & set(g["pos"])
            if len(common) >= MIN_COMPARABLE * (n_exp or max(len(h["pos"]), len(g["pos"]), 1)) and                     all(h["pos"][n] == g["pos"][n] for n in common):
                heldout_dups.append({"episodes": [(h["pipeline"], h["split"], h["episode"]),
                                                  (g["pipeline"], g["split"], g["episode"])]})
    by_pipe = {}
    for r in rows:
        k = f"{r['pipeline']}/{r['split']}"
        by_pipe[k] = by_pipe.get(k, 0) + 1
    sizes = sorted((len(v) for v in by_fp.values()), reverse=True)
    failed_spawn = [(r["pipeline"], r["split"], r["episode"]) for r in held if r.get("spawn_ok") is False]
    incomplete = [(r["pipeline"], r["episode"], round(r["coverage"], 3)) for r in rows if r["coverage"] < 1.0]
    if incomplete:
        problems.append(f"{len(incomplete)} record(s) lack expected names (the reduced fingerprint needs the fixed "
                        f"name set): {incomplete[:3]}")
    if undecidable:
        problems.append(f"{len(undecidable)} held-out/train pairs share < {MIN_COMPARABLE:.0%} of the names: "
                        f"{undecidable[:3]}")
    return {"n_episodes": len(rows), "episodes_by_pipeline_split": dict(sorted(by_pipe.items())),
            "n_expected_names": n_exp, "n_train_layouts": len(by_fp),
            "largest_train_layout_group": sizes[0] if sizes else 0,
            "collisions": coll, "heldout_heldout_duplicates": heldout_dups, "spawn_failed_heldout": failed_spawn,
            "incomplete_records": incomplete, "min_coverage": min((r["coverage"] for r in rows), default=1.0),
            "problems": problems,
            "ok": not coll and not problems and not failed_spawn and not heldout_dups}


def quota_skip_unexplained(state: dict, episodes, plan_scene: str) -> dict:
    """{episode: why} for every 'done' episode WITHOUT a record that is not a DEMONSTRATED quota skip (Astra K23-PREP
    r3 2). stage_capture (frozen Kitchen 1 code) marks a planned TRAIN, non-amb episode 'done' without a record only
    when ``quota.split_done("train")``; quota counts only grow, so the quota stored in capture_state.json must still
    say every train target is met. Anything else (held-out, amb, unplanned, unreadable or unmet quota) is unexplained."""
    import kitchen_data as kdat
    plan = {e["episode"]: e for e in kdat.episode_plan(plan_scene, kdat.CAPTURE["episodes"])
            + kdat.topup_train_plan(plan_scene, kdat.TOPUP_TRAIN_MAX)}
    eps = sorted(set(episodes))
    if not eps:
        return {}
    try:
        met = bool(kdat.Quota.from_dict(state["quota"]).split_done("train"))
        qwhy = "stored quota: train targets NOT all met"
    except Exception as exc:
        met, qwhy = False, f"stored quota unreadable ({type(exc).__name__})"
    out = {}
    for e in eps:
        p = plan.get(e)
        if p is None:
            out[e] = "not a planned episode"
        elif p.get("split") != "train":
            out[e] = f"split {p.get('split')} (only TRAIN episodes are quota-skipped)"
        elif str(p.get("protocol", "")).startswith("amb"):
            out[e] = "amb protocol (never quota-skipped)"
        elif not met:
            out[e] = qwhy
    return out


def _parity(scene, kdat) -> list:
    """K2-PARITY: parity TRAIN episodes are TRAIN for the done-without-record rule (k23_plans)."""
    import k23_plans
    return k23_plans.parity_plan(scene, kdat)


def report(scene: str, thor_data: str, dry_alias: str | None = None, provenance: dict | None = None,
           qc_dir: str | None = None, qc_batches: int | None = None) -> dict:
    """The whole check for one kitchen: every store, the expected name set from the probe, and record coverage.

    ``provenance`` (Astra K23-PREP r4 2) = the K23 QC's per-episode quota-skip provenance (qc_state.json "skips",
    recorded when the QC first saw the episode). "Quota met now" does not prove that an EARLIER missing TRAIN record was
    a skip, so a main-store episode 'done' without a record must also be in it. None (the stand-alone CLI without
    --qc-state) = no provenance: every such episode is a problem (fail closed)."""
    rows = scan(scene, thor_data)
    probe_p = os.path.join(thor_data, "kitchen_v3", scene, "probe.json")
    probe = json.load(open(probe_p, encoding="utf-8")) if os.path.exists(probe_p) else None
    rep = check(rows, expected_names(probe) if probe else None)
    if probe is None:
        rep["problems"].append("no probe.json: the expected name set is unknown")
        rep["ok"] = False
    rep["done_without_record"] = {}
    for store in ("kitchen_v3", "kitchen_v3_amb"):
        sp = os.path.join(thor_data, store, scene, "capture_state.json")
        if os.path.exists(sp):
            done = set(json.load(open(sp, encoding="utf-8")).get("done_episodes") or [])
            have = {r["episode"] for r in rows if r["store"] == store}
            # a TRAIN episode skipped because every train target was met is 'done' without a record (Kitchen 1 rule)
            miss = sorted(done - have)
            rep["done_without_record"][store] = miss
            # only a quota-skipped TRAIN episode of the main plan may be 'done' without a record
            import kitchen_data as kdat
            split = {e["episode"]: e["split"] for e in
                     (kdat.episode_plan(dry_alias or scene, kdat.CAPTURE["episodes"]) + kdat.topup_train_plan(
                         dry_alias or scene, kdat.TOPUP_TRAIN_MAX) + _parity(dry_alias or scene, kdat))}                 if store == "kitchen_v3" else {}
            bad = [e for e in miss if split.get(e) != "train"]
            if bad:
                rep["problems"].append(f"{store}: episodes done without a record that are not TRAIN: {bad[:5]}")
                rep["ok"] = False
            # a TRAIN one must be a DEMONSTRATED quota skip (Astra K23-PREP r3 2), not merely 'train'
            if store == "kitchen_v3":
                st_ = json.load(open(sp, encoding="utf-8"))
                und = quota_skip_unexplained(st_, [e for e in miss if e not in bad], dry_alias or scene)
                if und:
                    rep["problems"].append(f"{store}: TRAIN episodes done without a record and without a demonstrated "
                                           f"quota skip: {sorted(und.items())[:5]}")
                    rep["ok"] = False
                prov = provenance if isinstance(provenance, dict) else {}
                import k23_decide
                noprov = [e for e in miss if e not in bad and e not in und and (
                    str(e) not in prov or k23_decide.skip_entry_problems(str(e), prov[str(e)], scene, qdir=qc_dir,
                                                                 batches=qc_batches if qc_batches is not None else -1))]
                if noprov:
                    rep["problems"].append(f"{store}: TRAIN episodes done without a record and without the QC's skip "
                                           f"provenance (r4 2): {noprov[:5]}")
                    rep["ok"] = False
    rep["scene"] = scene
    for r in rows:
        r.pop("pos", None)
    return rep


def summary(rep: dict) -> str:
    return (f"{rep['n_episodes']} episodes {rep['episodes_by_pipeline_split']}; expected names {rep['n_expected_names']}, "
            f"min record coverage {rep['min_coverage']:.2f}; {rep['n_train_layouts']} distinct train layouts (largest "
            f"group {rep['largest_train_layout_group']}); held-out = train collisions {len(rep['collisions'])} "
            f"{rep['collisions'][:2]}; held-out duplicates {len(rep['heldout_heldout_duplicates'])}; failed held-out "
            f"spawns {len(rep['spawn_failed_heldout'])}; problems {rep['problems'][:3] or 'none'}; done without a "
            f"record {rep.get('done_without_record')}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--thor-data", default=default_thor_data())
    ap.add_argument("--out", default=None)
    ap.add_argument("--qc-state", default=None, help="the K23 qc_state.json whose 'skips' prove earlier quota skips")
    a = ap.parse_args(argv)
    qs = json.load(open(a.qc_state, encoding="utf-8")) if a.qc_state else {}
    prov = (qs.get("skips") or {}) if a.qc_state else None
    rep = report(a.scene, a.thor_data, provenance=prov,
                 qc_dir=os.path.dirname(os.path.abspath(a.qc_state)) if a.qc_state else None,
                 qc_batches=qs.get("batches") if a.qc_state else None)
    if not rep["n_episodes"]:
        print(f"{a.scene}: no episode record found under {a.thor_data}")
        return 2
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
    print(f"{a.scene}: {summary(rep)} -> LAYOUT CHECK {'PASS' if rep['ok'] else 'FAIL (STOP the kitchen)'}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
