"""M9 -- the locked analysis (prereg s9, SPEC s12). Windows python, offline, no simulator, no LLM, no API.

    python sim/final_test/matrix/analyze.py --synthetic
    python sim/final_test/matrix/analyze.py --sealed docs/final-test/matrix/test-v1/score.jsonl --allow-test

Reads ONLY the sealed score journal (plus, optionally, plan.jsonl / practice.jsonl beside it, for cost). Never opens
outcome.sealed.jsonl, states/ or the dataset. Computes:
  H1 (A24.1 h1-rule-1 = C7, final_test_h1_stats.h1_rule_c7; plus the exploratory equivalence E1), the 3-test secondary Holm family (H2a, H2c, H3a; A24.5), the EXPLORATORY contrasts H4a-b / H5a-c (raw p, no Holm), the family-weighted sensitivity,
  the paired infra-failure sensitivity, per level / language rates, the REPEAT variance decomposition, cost per success,
  and the Codex table (exploratory, never pooled, never in Holm).
Synthetic mode runs the same code on generated outcomes with KNOWN effects; its output carries "synthetic": true and is
written to docs/final-test/matrix/synthetic/ so it can never overwrite a real analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, SIM)

import common as C                                                 # noqa: E402
import final_test_power as P                                       # noqa: E402
import final_test_h1_stats as H1S                                  # noqa: E402  (A24.1: the validated H1 rule h1-rule-1 = C7)
import a36 as A36                                                  # noqa: E402  (A36, signed: excluded_reset common-subset analysis)

MARGIN = P.MARGIN
ALPHA_HOLM = 0.05
LANGS = ("en", "fa")
CORE_NEEDED = C.CORE_ARMS

#: id -> (arm_a, arm_b, kind, delta0). D = SR(a) - SR(b).
H1 = ("G+M", "O0")
#: A24.5: the confirmatory-secondary Holm family has m = 3 members (alpha .05): H2a, H2c, H3a. (A23 had m = 8; A22 m = 11.)
FAMILY = {
    "H2a": ("G+M", "G0", "greater", 0.0), "H2c": ("O+M", "O0", "greater", 0.0),
    "H3a": ("O+M", "G+M", "two", 0.0),
}
#: A24.5: EXPLORATORY contrasts: no confirmatory conclusion can come from them, they are NOT in the Holm family (raw p only, no adjustment, no "reject").
#: H4a/b = package vs FS; H5a-c = the gpt-6-sol arms (A18). The p-value rule they had as Holm members (A23) is kept as `p_m8` so the report can print the
#: m = 8 adjustment next to the m = 3 verdict (the smaller m relaxes the Holm thresholds; that is disclosed).
EXPLORATORY = {
    "H4a": ("G+M", "G+FS", "two", 0.0), "H4b": ("O+M", "O+FS", "two", 0.0),
    # A18: Codex gpt-6-sol is a main executed comparator (X0 = no memory, X+M = memory snapshot S*).
    "H5a": ("X+M", "X0", "greater", 0.0),                 # memory gain on the Codex system
    "H5b": ("X+M", "G+M", "two", 0.0),                    # Codex vs Gemma, both with memory
    "H5c": ("G+M", "X0", "ni_x2", -MARGIN),               # Gemma+memory non-inferior to Codex without memory (descriptive; min(1, 2p) only inside the m = 8 comparison)
}
#: every pairwise contrast the report prints sensitivities for (Holm members first, then the exploratory ones)
ALL_CONTRASTS = {**FAMILY, **EXPLORATORY}
HOLM_M_A23 = 8                    # the A23 family (H2a, H2c, H3a + H4a-b + H5a-c), recomputed for transparency only
#: A23: hypotheses WITHDRAWN before any test call (Sonnet was removed from the protocol). The IDs are kept for traceability and are never
#: computed, never in the Holm family; the arms they name are in common.ARCHIVED_ARMS. A future namespace that tests Sonnet defines its own family.
WITHDRAWN = {"H2b": {"contrast": "SR(S+M) - SR(S0)", "meaning": "Sonnet memory gain"},
             "H3b": {"contrast": "SR(S+M) - SR(G+M)", "meaning": "Sonnet+memory vs Gemma+memory"},
             "H3c": {"contrast": "SR(G+M) - SR(S0)", "meaning": "Gemma+memory non-inferior to Sonnet without memory"}}
assert set(WITHDRAWN) == set(C.WITHDRAWN_HYPOTHESES) and not (set(WITHDRAWN) & set(ALL_CONTRASTS)) and not (set(FAMILY) & set(EXPLORATORY))
assert len(FAMILY) == 3 and len(FAMILY) + len(EXPLORATORY) == HOLM_M_A23


#: C5 (stage C5-SIMPLE, 2026-10-14; before any test call): how energy_j_gpu is produced. Part of A24_2_COST, so hashed with the analysis plan;
#: the producer is energy.py (its METHOD_ID must equal "id"; _ftm_energy_selftest checks it). It replaces the earlier c5-energy-3 method
#: (NVML + RAPL with interval checks; copy in sim/final_test/_checkpoints/C5-ENERGY/pre-simple/) on Arash's decision of 2026-10-14.
ENERGY_METHOD = {
    "id": "c5-simple-1",
    "decision": ("Arash, 2026-10-14 (docs/final-test/progress.md 'ARASH DECISION: simplify the energy method (C5)'): measure the GPU's "
                 "average power draw with nvidia-smi during Gemma planning, multiply by the planning time, price it at 0.20 USD/kWh; drop the "
                 "coverage / gap / backwards-clock checks; disclose it plainly as a simple, approximate method"),
    "method": ("a background nvidia-smi (--query-gpu=power.draw.instant -i 0 -lms 100) prints the GPU board power about every 100 ms for the "
               "whole plan phase. Per Gemma plan cell: planning_s = the time from just before plan.Plan._call_cell starts to just after it "
               "returns (every attempt, repair and parse of that cell); gpu_w_mean = the plain mean of the readings that arrived in that time "
               "(none there: the readings within 1 s of it); energy_j_gpu = gpu_w_mean x planning_s; USD = energy_j_gpu / 3,600,000 x "
               "usd_per_kwh. A cell with no reading at all has no energy figure (its cost is UNKNOWN)"),
    "not_measured": ("GPU only: the CPU, memory, display, storage, fans and power-supply losses are NOT measured (energy_j_cpu is None), so the "
                     "figure is LOWER than the laptop's total planning energy. The reading is the whole GPU board (every process on it, the idle "
                     "floor included, never subtracted); the idle floor measured when the meter opens is reported"),
    "accuracy": ("approximate: a mean of about 10 readings per second times the time; no calibration, no coverage or gap checks; the readings "
                 "are the driver's own power figure, not an external meter"),
    "code": ("sim/final_test/matrix/energy.py, installed for one launch (energy.measuring) on plan.Plan._call_cell by matrix_run.py (plan "
             "phase) and planfirst.py; mandatory in test-v1 (planning is refused when nvidia-smi cannot be read). plan.py is not edited"),
    "journal": ("energy.jsonl (energy.<pass id>.jsonl for --pass-id runs) beside plan.jsonl: one 'open' row per run (GPU name, idle mean) and "
                "one 'cell' row per Gemma cell, bound to its plan journal line by sha256"),
    "join": ("analyze.attach_energy: a Gemma plan row takes energy_j_gpu and planning_s from the one energy row whose bound plan line sha256 "
             "equals the sha256 of its plan journal line; no such row (or several) -> no energy figure -> UNKNOWN cost"),
    "unmeasured_rule": ("the 10 practice-v2 rows ran before the meter existed. On Arash's decision of 2026-10-14 ('energy estimate ok') each "
                        "is ESTIMATED as the run's measured Gemma rate x the row's own planning seconds and flagged 'estimated'; the measured "
                        "figure stays the primary one and the estimate is disclosed beside it. Every other Gemma row without an energy figure "
                        "keeps an UNKNOWN cost"),
    "unmeasured_rule_basis": "arash-decision:2026-10-14",
    # the binding terms of that decision (price_row applies them row by row): set to UNMEASURED_TERMS_APPROVED just below the class lists
    "unmeasured_rule_terms": None,
    "unmeasured_estimate": {"rate": "measured_rate",
                            "definition": ("rate_w = (sum of the measured energy_j_gpu) / (sum of their planning_s) over every Gemma plan row of "
                                           "the analysed run with an energy figure from the join (never estimates): the average measured Gemma "
                                           "GPU power per planning second; an eligible practice row's estimate = rate_w x its own planning "
                                           "seconds (planning_s / plan_secs; none -> UNKNOWN), assuming the measured and the earlier calls draw "
                                           "comparable power (same laptop, same model). Only the 10 practice-v2 rows named in the terms "
                                           "(bound to practice.jsonl by sha) are eligible"),
                            "min_cells": 20,
                            "too_few": "fewer than min_cells measured rows (or zero measured seconds) -> no rate -> the practice rows stay UNKNOWN",
                            "disclosure": ("reported per arm: measured joules (primary), estimated rows / seconds / joules / USD and the rate "
                                           "(n cells, seconds); markdown labels the estimate")},
    "disclosure": ("Planning energy is a simple, approximate figure: the GPU's average power draw reported by nvidia-smi (about 10 readings "
                   "per second) during each Gemma planning call, multiplied by the planning time and priced at 0.20 USD/kWh. Only the GPU is "
                   "counted (CPU and the rest of the laptop are not), so it understates total energy; idle power is included. The 10 practice "
                   "rows that ran before the meter existed are estimated from the measured rate and flagged."),
    "evidence": "docs/final-test/energy/c5-simple-measurement.json",
}

#: A24.2 item 1 -- cost-effectiveness accounting (stage P-cost, 2026-10-13; written BEFORE any test call, on synthetic data only). Part of ANALYSIS_PLAN,
#: so it is hashed with it. If freeze.json carries `a24_2_cost`, its numbers must equal these or the sealed analysis refuses (no silent override).
A24_2_COST = {
    "version": "a24.2-cost-1",
    "usd_per_kwh": 0.20,
    "hardware_price_usd": 399.0,
    "hardware_price_source": ("reference price, prereg A33 (2026-10-14, Arash decision 8; prospectively replaces A24.2's 'declared laptop purchase "
                              "price'): NVIDIA Jetson Orin Nano Super Developer Kit 8 GB, USD 399 (ledger 2026-10-12 'Hardware price for A24.2'); a "
                              "reference device price, not the run laptop's price, and not shown to run this system; recorded in freeze.json as "
                              "a24_2_cost before the first test call"),
    "hardware_price_basis": "amendment:A33",
    "hardware_price_status": ("recorded: prereg A33 (prospective; approved by Arash 2026-10-14, decision 8; signature pending at the freeze summary). D13 is closed. "
                              "freeze.json a24_2_cost must carry this basis; the freeze builder checks that the A33 heading exists in the prereg"),
    "hardware_price_disclosure": ("A33: planning time is measured on the run laptop (RTX 4060, 8 GB). The headline hardware allocation combines those "
                                  "measured hours with the USD 399 reference-device price. The energy term is the laptop GPU's energy during planning "
                                  "(energy_method c5-simple-1: nvidia-smi average power x planning time; GPU only, approximate). A Gemma plan "
                                  "row without an energy figure stays UNKNOWN and blocks the A24.2 statement. Sensitivity: the hw_time_3x row "
                                  "(3 x planning hours, laptop energy held fixed; factor 3 = an approved sensitivity assumption, not a measured "
                                  "slowdown); the reading rule uses the headline view only"),
    "operating_hours": 6000.0,
    "operating_hours_rule": "three years at 2,000 operating hours / year",
    "hardware_time_factor": 3.0,
    "hardware_time_note": ("view hw_time_3x = '3x hardware-time sensitivity; measured laptop energy held fixed' (ledger 2026-10-12: a Jetson-class device plans "
                           "more slowly). It is NOT an estimate of Jetson energy or of total operating cost"),
    "energy_rule": "GPU energy in joules (plan row energy_j_gpu = nvidia-smi mean power x planning_s) / 3,600,000 x usd_per_kwh (energy_method)",
    "energy_requirement": ("the producer takes the mean GPU power over the SAME interval that gives the planning seconds (energy_method c5-simple-1); "
                           "the energy method is recorded in freeze.json"),
    "energy_method": ENERGY_METHOD,
    "hardware_rule": "hardware_price_usd / operating_hours x attributable planning hours",
    "planning_seconds": ("plan row planning_s if present, else wall_s (the plan cell's wall time: every sequential attempt, repair and failed attempt, "
                         "counted once; never both the cell interval and its attempts); no other fallback (a sum of request latencies would omit parsing, "
                         "retrieval and package time) -> a Gemma row without either is UNKNOWN"),
    "arm_kinds": {"gemma": "G*: energy + hardware allocation (+ any cost_usd); a Gemma row without an energy figure (energy_j_gpu) makes the arm's cost UNKNOWN unless the practice-row estimate applies",
                  "api": "O* (and archived S* / H*): cost_usd of the plan journal (dated list prices; reasoning and repairs included, cache replays cost 0)",
                  "codex": "X*: CLI planning NOT priced (subscription: no incremental API expenditure; A24.2 prices local Gemma and API arms only); "
                           "its allocated preparation USD is kept and reported, its total stays incomplete, never zero"},
    "codex_pareto_exception": ("documented exception to 'showing all arms' (Astra P-cost Q6; no subscription allocation rule is registered): X arms appear "
                               "in the Pareto table and under the plot with their SR and known preparation USD, without a cost coordinate. A subscription "
                               "allocation would have to be fixed prospectively (PENDING ARASH if wanted)"),
    "workload": ("scored cells only: every plan row of (arm, canon presentation, r = 0, block main or unset) is summed (all incurred attempts); "
                 "plan rows of other blocks (REPEAT r > 0, curve, relocation) are totalled separately and are not in the per-task outcomes"),
    "allocation": ("shared / arm-specific preparation rows (practice.jsonl, priced as Gemma since practice is Gemma-planned, and optional prep_cost.jsonl "
                   "rows, which must name cost_kind gemma / api / codex): a row with `arm` (or `arms`) goes to those arms, a row without goes to the consumer arms "
                   "under allocation_rule; each arm's allocated total is spread equally over its attempted presentations. Planning seconds of a prep row = "
                   "planning_s or plan_secs (never the episode wall_s). A Gemma prep row without a validated measurement makes its targets' cost UNKNOWN (energy_method.unmeasured_rule; "
                   "practice-v2 has no energy measurement)"),
    "allocation_rule": {"mode": "split", "consumer_suffixes": ["+M", "+FS"], "practice_required": True,
                        "meaning": ("split: each shared row is allocated ONCE, divided among its target arms in proportion to their registered workload "
                                    "(the cohort presentations; equal workloads -> equal shares; with all 8 main arms the practice consumers are G+M, O+M, "
                                    "X+M, G+FS, O+FS = one fifth each; Astra P-cost Q1/Q2). full (each target carries the whole row) duplicates shared "
                                    "expenditure and is not reported. practice_required: consumer arms without any practice / preparation row are UNKNOWN")},
    "views": {"headline": "API USD + energy + hardware + allocated", "electricity_only": "headline without the hardware allocation",
              "hardware_only": "the hardware allocation alone (per arm only; no C is formed for it)",
              "hw_time_3x": "3x hardware-time sensitivity; measured laptop energy held fixed"},
    "cohort": ("the registered task cohort = every (item, language) presentation scored in any arm, minus INVALID presentations (dropped from every arm). "
               "Every outcome uses the same item weights: item value = mean over its cohort presentations; an arm missing a cohort presentation is "
               "reported (missing_vs_cohort), never silently dropped, and blocks the reading rule if it is G+M or O0"),
    "outcomes": {"a": "total planning USD / successful tasks per arm = sum of item-mean cost / sum of item-mean SR (equal to presentation totals / "
                      "successes under complete EN/FA coverage); the raw totals are reported too",
                 "b": "R = SR(G+M) / SR(O0) and C = mean planning USD per attempted task G+M / O0, item-weighted on the cohort",
                 "c": "Pareto data: item-weighted SR against mean planning USD per attempted task, all arms (pareto.svg beside analysis.json)",
                 "latency": "planning seconds per attempted task including repairs and failed attempts (mean, median, p90)"},
    "ci": ("95 % percentile interval (inverted-CDF order statistics, no interpolation), task-family cluster bootstrap, B = 20000, seed = final_test_power.SEED; "
           "ONE resampling matrix over the union of families for every arm, view and ratio"),
    "undefined_rule": ("a ratio with denominator 0 is +infinity when the numerator is > 0 and UNDEFINED when it is 0 (point estimate and every bootstrap "
                       "replicate). Infinite replicates stay in the ordering; if ANY replicate is undefined, or a needed cost is unknown, the CI bounds are "
                       "'not obtainable' (reported with the counts, never dropped) and the reading rule cannot be satisfied"),
    "claim": {"R_lower_min": 0.80, "C_upper_max": 0.10, "view": "headline",
              "text": ("Gemma with the package retained at least 80% of unassisted Opus's success rate at at most 10% of its specified planning cost"),
              "rule": ("permitted only when the lower 95 % bound of R >= 0.80 AND the upper 95 % bound of C <= 0.10, both obtainable and finite, in a "
                       "complete study, with G+M and O0 covering the cohort, and (real data) with freeze.json a24_2_cost recorded including the hardware "
                       "price basis; A24.2 makes this the statement available when H1 is not established; otherwise estimates and CIs only")},
    "scope": "planning cost only (executor / simulator excluded); CIs describe task-sampling uncertainty conditional on these prices and assumptions",
}

#: A21 F -- the analysis plan, frozen (hashed into freeze.json and into every analysis.json). Changing any of it is a protocol change.
LOW_DISCORDANCE = 0.25           # observed discordance below this: a non-inferiority member of the Holm family is not rejected (p := 1); chosen on ni_calibration.json (A21 F.4)
ANALYSIS_PLAN = {
    "version": "ftm-analysis-3",
    "h1": {"contrast": "SR(G+M) - SR(O0)", "alpha_one_sided": P.ALPHA_ONE_SIDED, "margin": MARGIN,
           "decision": "A24.1: h1-rule-1 = C7 (final_test_h1_stats.h1_rule_c7): CR2 + Satterthwaite, guard K >= 20 discordant families, one-sided computational "
                       "level .0175 per tail (nominal 96.5 % interval); NI iff L > -.05; superior iff NI and L > 0 (fixed sequence); materially inferior iff "
                       "U < -.05; 'statistically lower' iff NI and U < 0; se = 0 / undefined df / guard failure -> inconclusive. The registered wild+percentile "
                       "rule (final_test_power.h1_decision) failed its validation and is never used for a claim",
           "validation": "docs/final-test/h1_validation/v100_summary.C7.json (226 configurations x 100,000 datasets, 0/628 null cells above the frozen gate)",
           "estimation_only_if": "H1_PROCEDURE['validated'] is False (then estimate + interval only, no NI / superior / inferior wording)",
           "views": "primary = item-weighted (validated); family-weighted and infra-paired = sensitivities (same rule, not separately validated)",
           "exploratory_equivalence": "E1 (G+M vs O+M): CR2 + Satterthwaite 90 % interval + guard K >= 20, inside +/-5 pp; exploratory only",
           "family": "none -- H1 stands alone; it is NOT a member of the Holm family"},
    "holm": {"alpha": ALPHA_HOLM, "m": len(FAMILY), "members": {k: list(v[:2]) + [v[2], v[3]] for k, v in FAMILY.items()},
             "note": "m = 3 since A24.5 (H2a, H2c, H3a; all superiority or two-sided, no non-inferiority member); H4a-b and H5a-c are EXPLORATORY (raw p, listed in `exploratory`); "
                     "H2b, H3b and H3c are WITHDRAWN (Sonnet is archived, A23), their IDs are kept in `withdrawn`; the smaller m relaxes the Holm thresholds, which is disclosed "
                     "(the report prints the m = 8 adjustment of the A23 family next to the m = 3 verdict)",
             "exploratory": {k: list(v[:2]) + [v[2], v[3]] for k, v in EXPLORATORY.items()}, "withdrawn": WITHDRAWN},
    "p_values": {"greater": "one-sided wild cluster bootstrap, null imposed at delta0 (0 for superiority)",
                 "ni_x2": "H5c (EXPLORATORY since A24.5): p1 = one-sided wild test of H0: D <= -5 pp (the null BOUNDARY is imposed, not 0), reported raw; min(1, 2 p1) with the guard only in the m = 8 transparency comparison; "
                          "never from a test of D = 0",
                 "two": "min(1, 2 min(p_greater, p_less)) at delta0 = 0"},
    "ni_guard": {"observed_discordance_below": LOW_DISCORDANCE, "effect": "applied to NO m = 3 Holm member (none is a non-inferiority test); retained for the m = 8 transparency comparison, where H5c's p is set to 1 (not rejected)",
                 "scope": "H5c only, and only inside the m = 8 transparency comparison (A24.5 made H5c exploratory); NOT H1, whose registered calibration table (section 9.6) is unchanged",
                 "why": "docs/final-test/matrix/ni_calibration.json: type-I error of the complete non-inferiority procedure at the null boundary D = -5 pp, 4,000 simulations per cell "
                        "(Monte Carlo SE about .001), discordance .05-.30, family ICC 0 / .3 / .6; without the guard the first-Holm-step rate (nominal p1 < .00227) is up to .14 at "
                        "discordance .05, .054 at .10, .027 at .15, .009 at .20 and .0075 at .25; with the guard at .25 the largest first-step rate in the grid is .0022 and the largest "
                        "rate at the most lenient step (nominal .025) is .0213; guards at .20 and .15 left .0037 and .0047",
                 "residual": "no inflation above nominal was seen in the grid once the guard applies (largest rates are at or below nominal within Monte Carlo error); NI claims from H5c are still labelled approximate"},
    "dependence": {"unit_of_analysis": "item (mean of its EN and FA presentations)", "cluster": "the 42 test families (paired: every arm of a family moves together)",
                   "shared_executions": "plan cells with equal execution-unit keys share one simulator run; their outcome is ONE realisation. The family cluster contains every "
                                        "unit, so the intervals and tests already treat copies as dependent; the report adds the number of independent executions and a "
                                        "sensitivity that drops presentations where the two arms of a contrast share a unit (that estimate is performance CONDITIONAL on non-shared executions: it "
                                        "changes the population and cannot validate or replace the full-sample result)"},
    "sensitivities": ["family_weighted", "infra_paired (planner transport failure OR execution infrastructure failure of either arm)", "distinct_units_only",
                      "x_isolation_paired (H5: drop presentations whose X cell had a tool event or an identity mismatch)"],
    "x_disclosure": "X_DISCLOSURE (printed with every X result)",
    "seeds": {"master": C.MASTER_SEED, "B_tests": 9999, "B_interval": 20000},
    "a24_2_cost": A24_2_COST,
    "a36": {"rule": A36.A36_VERSION, "status": A36.A36_STATUS, "text": "docs/final-test/amendments/A36-DRAFT.md r2 (signatures.json key A36)",
            "eligibility": "excluded_reset (a36.excluded_reset, computed in score.py): a core cell (block main, canon, r0, run0) whose final class is "
                           "infra_exec and whose 3 attempts (0, 1, 2) of the CURRENT epoch (latest of the executor's epoch state, rows and starts) each "
                           "started once (exec_attempt_start) and stopped at the pre-execution reset gate (status blocked_state, registered comparison failed, "
                           "not overridden, prims == sweep prims, 0 action-resolve, no error, no outcome); anything mixed, malformed or missing -> not flagged",
            "retention": "paired common subset per contrast: a presentation flagged in EITHER arm leaves both; then the registered aggregation (per item the "
                         "mean of retained EN/FA differences, equal item weights, empty items / families dropped, K and N recomputed); every contrast "
                         "(H1, Holm m = 3, exploratory H4a-b / H5a-c, equivalence E1); exploratory stays exploratory",
            "unavailable": "fewer than 2 retained families -> the contrast is unavailable: H1 / E1 inconclusive (no claim), a Holm member enters with p = 1 "
                           "and m stays 3; counts are explicit zeros",
            "h1": "h1_rule_c7 unchanged (K >= 20 guard; K < 20 -> inconclusive); its validation did not cover this selection step",
            "per_arm_sr": "item-weighted (analyze.arm_rates) over non-flagged cells; item and presentation denominators printed",
            "reset_limited": f"an arm with MORE than {A36.RESET_LIMITED_SHARE:.0%} of its core cells flagged labels every contrast involving it",
            "registered_alongside": "A21: infra_exec = SR 0 in every denominator; differences between the two analyses are named"},
}


#: The sentence that accompanies every X0 / X+M result (A21 E). It names exactly what was and was not verified.
X_DISCLOSURE = ("X used codex-cli 0.162.1 configured for gpt-6-sol at high reasoning effort (the CLI-configured model; the model the server finally answered with was "
                "unobservable). Built-in tools remained available: isolation monitoring covered the events the CLI emitted, not invisible calls. The isolation probe "
                "(9 calls; 6 canary-read calls produced 24 visible tool events, mostly MCP resource listing) returned no secret token. Primary results include calls "
                "flagged for tool events; paired exclusions are reported separately. CLI scaffolding (about 10.6 k input tokens per call) and subscription execution are part "
                "of this comparator; 'free' means no incremental API expenditure only.")


def analysis_plan_sha() -> str:
    return C.sha256_json(ANALYSIS_PLAN)


class AnalysisError(RuntimeError):
    pass


# ================================================================================================ loading + seal
def read_jsonl(path):
    return C.Journal(path).rows()


def verify_seal(score_path: str, allow_partial: bool = False) -> dict:
    seal_path = os.path.join(os.path.dirname(score_path), "score.seal.json")
    try:
        with open(seal_path, encoding="utf-8") as f:
            seal = json.load(f)
    except (OSError, ValueError) as e:
        raise AnalysisError(f"score.seal.json missing or unreadable ({type(e).__name__}); refusing to analyse") from None
    sha = C.sha256_file(score_path)
    if seal.get("sha256") != sha:
        raise AnalysisError("score.jsonl does not match its seal (edited after sealing)")
    n = len(read_jsonl(score_path))
    if seal.get("n") != n:
        raise AnalysisError(f"seal says n={seal.get('n')} but score.jsonl has {n} rows")
    if not seal.get("complete") and not allow_partial:
        raise AnalysisError("score journal is not complete; analysis needs --partial and a recorded stopping rule")
    return seal


def build_index(rows, fam_of_manifest=None):
    """rows -> dict with S[(arm,item,lang)] = row for the scored cell (canon world, r=0, run=0), plus metadata."""
    S, fam_of, lvl_of, bad = {}, {}, {}, []
    for r in rows:
        try:
            p = C.parse_pres(r["pres"])
        except (ValueError, KeyError):
            bad.append(r.get("key"))
            continue
        if p["world"] != "canon" or r.get("r") != 0 or r.get("run") != 0:
            continue
        item = p["item"]
        fam = r.get("family") or (fam_of_manifest or {}).get(item)
        if fam is None:
            raise AnalysisError(f"item {item} has no family")
        if fam_of_manifest and item in fam_of_manifest and fam_of_manifest[item] != fam:
            raise AnalysisError(f"family of {item} disagrees with the split manifest")
        if fam_of.setdefault(item, fam) != fam:
            raise AnalysisError(f"item {item} appears in two families")
        lvl_of[item] = r.get("level") or lvl_of.get(item)
        S[(r["arm"], item, p["lang"])] = r              # a later row for the same cell replaces an earlier one (resume)
    return {"S": S, "fam": fam_of, "level": lvl_of, "bad": bad}


def is_invalid_row(r) -> bool:
    """score.py flags an unrecoverable INVALID presentation with `invalid_episode` (the ONLY thing analysis may drop a presentation for;
    `valid` means "reached scoring" and is True on every row). `valid is False` is kept for rows from before that split."""
    return bool(r.get("invalid_episode")) or r.get("valid") is False


def invalid_presentations(S):
    """INVALID presentations are dropped from EVERY arm (prereg s13), which keeps the pairing: one flagged row removes the presentation
    (item, language) from all arms and every contrast."""
    return {(i, l) for (a, i, l), r in S.items() if is_invalid_row(r)}


# ================================================================================================ contrasts
def is_infra(r) -> bool:
    """A21 A: infrastructure failure of a scored cell: planner transport failure after the retries OR an execution that stayed blocked / crashed after
    its re-runs. (`exec_failure` is score.py's flag for the second; rows from before A21 carry only `infra`.)"""
    return bool(r.get("infra") or r.get("exec_failure") or r.get("run_class") in ("infra_planner", "infra_exec"))


def item_diffs(S, a, b, drop_invalid, drop_infra=False, items=None, drop_shared=False, drop_flag=None):
    """-> ({item: diff}, info). item diff = mean over languages of (sr_a - sr_b) for presentations present in both arms.
    info also counts, over the presentations used: `n_shared` (both arms point at the SAME execution unit: the outcome is one copied realisation,
    the difference is 0 by construction) and `n_discordant` (the two arms' SR differ). `drop_shared` removes the shared ones (sensitivity);
    `drop_flag(row)` removes presentations whose row in either arm carries the flag (X isolation sensitivity)."""
    diffs, n_pres, n_drop_infra, n_missing, n_shared, n_disc, n_drop_flag = {}, 0, 0, 0, 0, 0, 0
    disc_items = set()
    cand = sorted({i for (arm, i, l) in S if arm in (a, b)} if items is None else items)
    for i in cand:
        ds = []
        for l in LANGS:
            if (i, l) in drop_invalid:
                continue
            ra, rb = S.get((a, i, l)), S.get((b, i, l))
            if ra is None or rb is None:
                n_missing += 1
                continue
            if drop_infra and (is_infra(ra) or is_infra(rb)):
                n_drop_infra += 1
                continue
            if drop_flag is not None and (drop_flag(ra) or drop_flag(rb)):
                n_drop_flag += 1
                continue
            shared = ra.get("unit") is not None and ra.get("unit") == rb.get("unit")
            if drop_shared and shared:
                continue
            n_shared += 1 if shared else 0
            if float(ra["sr"]) != float(rb["sr"]):
                n_disc += 1
                disc_items.add(i)
            ds.append(float(ra["sr"]) - float(rb["sr"]))
            n_pres += 1
        if ds:
            diffs[i] = sum(ds) / len(ds)
    return diffs, {"n_items": len(diffs), "n_pres": n_pres, "dropped_infra": n_drop_infra, "missing_pres": n_missing, "n_shared": n_shared,
                   "n_discordant": n_disc, "discordance": (n_disc / n_pres) if n_pres else None, "dropped_flagged": n_drop_flag,
                   "_disc_items": disc_items}


def discordant_families(S, a, b, drop_invalid, fam_of, **kw) -> int:
    """A22.1: K = the number of families with at least one discordant USED presentation (same filters as item_diffs)."""
    diffs, info = item_diffs(S, a, b, drop_invalid, **kw)
    return len({fam_of[i] for i in info["_disc_items"] if i in diffs})


def by_family(diffs, fam_of, weighted=False):
    """-> (family names, fam_tot, sizes). weighted=True: each family's mean counts once (sizes all 1)."""
    tot, n = {}, {}
    for i, d in diffs.items():
        f = fam_of[i]
        tot[f] = tot.get(f, 0.0) + d
        n[f] = n.get(f, 0) + 1
    fams = sorted(tot)
    if len(fams) < 2:
        raise AnalysisError(f"need at least 2 families, got {len(fams)}")
    if weighted:
        return fams, [tot[f] / n[f] for f in fams], [1] * len(fams)
    return fams, [tot[f] for f in fams], [n[f] for f in fams]


def win_tie_loss(fams, fam_tot, sizes):
    w = sum(1 for t in fam_tot if t > 1e-12)
    l = sum(1 for t in fam_tot if t < -1e-12)
    return {"families_a_better": w, "tie": len(fam_tot) - w - l, "families_b_better": l}


def contrast(S, fam_of, a, b, drop_invalid, B, *, drop_infra=False, weighted=False, drop_shared=False, drop_flag=None, amended=False):
    diffs, info = item_diffs(S, a, b, drop_invalid, drop_infra, drop_shared=drop_shared, drop_flag=drop_flag)
    if not diffs:
        return None
    if amended and len({fam_of[i] for i in diffs}) < 2:     # A36: a common subset with < 2 families is UNAVAILABLE evidence, never an abort
        return None
    fams, tot, sizes = by_family(diffs, fam_of, weighted)
    lo, hi = P.cluster_ci(tot, sizes)
    out = {"a": a, "b": b, "estimate": P.estimate(tot, sizes), "ci95": [lo, hi], "n_families": len(fams), **info,
           **win_tie_loss(fams, tot, sizes)}
    out["discordant_families"] = len({fam_of[i] for i in info["_disc_items"] if i in diffs})
    out["_tot"], out["_sizes"] = tot, sizes
    return out


def clean(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def holm(pvals: dict, alpha=ALPHA_HOLM):
    """Holm-Bonferroni step-down over a fixed family; pvals None -> treated as 1.0 (test not available)."""
    m = len(pvals)
    order = sorted(pvals, key=lambda k: (1.0 if pvals[k] is None else pvals[k], k))
    adj, run = {}, 0.0
    for rank, k in enumerate(order):
        p = 1.0 if pvals[k] is None else pvals[k]
        run = max(run, min(1.0, (m - rank) * p))
        adj[k] = run
    return {k: {"p": pvals[k], "p_holm": adj[k], "reject": adj[k] < alpha} for k in pvals}


def _member(S, fam_of, hid, a, b, kind, d0, drop_invalid, B, ni_guard=True, drop_flag=None):
    """One pairwise contrast with its p-value under the registered rule for its `kind` -> (p or None, detail). `drop_flag` (A36): paired subset."""
    c = contrast(S, fam_of, a, b, drop_invalid, B, drop_flag=drop_flag, amended=drop_flag is not None)
    if c is None:
        return None, {"a": a, "b": b, "available": False}
    tot, sizes = c["_tot"], c["_sizes"]
    pg = P.wcr_p(tot, sizes, d0, "greater", B)
    if kind == "greater":
        p, extra = pg, {"p_one_sided": pg}
    elif kind == "ni_x2":
        # A21 F: pg is the one-sided wild test of H0: D <= -5 pp (the null BOUNDARY imposed: delta0 = d0 = -MARGIN), never a test of D = 0.
        p, extra = min(1.0, 2 * pg), {"p_one_sided": pg, "null": f"D <= {d0}", "null_boundary_imposed": d0}
        disc = c.get("discordance")
        if ni_guard and disc is not None and disc < LOW_DISCORDANCE:
            extra["ni_guard"] = {"applied": True, "discordance": disc, "threshold": LOW_DISCORDANCE, "p_before_guard": p}
            p = 1.0
    else:
        pl = P.wcr_p(tot, sizes, d0, "less", B)
        p, extra = min(1.0, 2 * min(pg, pl)), {"p_greater": pg, "p_less": pl}
    return p, {**clean(c), "kind": kind, "delta0": d0, "available": True, **extra}


def holm_family(S, fam_of, drop_invalid, B, complete=True, drop_flag=None):
    """The SECONDARY confirmatory family, m = len(FAMILY) = 3 since A24.5 (H2a, H2c, H3a), Holm-Bonferroni at alpha .05. An unavailable test counts as p = 1
    and keeps m. H4a-b and H5a-c are NOT here (see `exploratory_contrasts`)."""
    ps, detail = {}, {}
    for hid, (a, b, kind, d0) in FAMILY.items():
        ps[hid], detail[hid] = _member(S, fam_of, hid, a, b, kind, d0, drop_invalid, B, drop_flag=drop_flag)
    res = holm(ps)
    for k in res:
        detail[k].update(res[k])
        if not complete:                  # an incomplete study makes no confirmatory claim: p-values stay descriptive, nothing is "rejected"
            detail[k]["reject"] = None
            detail[k]["note"] = "incomplete study: descriptive only (prereg s13)"
    return detail


def exploratory_contrasts(S, fam_of, drop_invalid, B, holm_detail=None, drop_flag=None):
    """A24.5: H4a-b (package vs FS) and H5a-c (gpt-6-sol arms) as EXPLORATORY contrasts: estimate, interval and RAW p only. No Holm adjustment, no `reject`,
    no confirmatory wording. `p_m8` is the p-value each had as an A23 Holm member (H5c: min(1, 2 p1), with the discordance guard) and `p_holm_m8` the
    8-test Holm adjustment of {H2a, H2c, H3a} + these five, printed only so the effect of the smaller m (3) on the thresholds is disclosed.
    -> (exploratory detail, {H2a/H2c/H3a: their p_holm if the family had been the A23 m = 8})"""
    ps8, out = {}, {}
    for hid, (a, b, kind, d0) in EXPLORATORY.items():
        p, d = _member(S, fam_of, hid, a, b, kind, d0, drop_invalid, B, drop_flag=drop_flag)
        ps8[hid] = p
        if d.get("available"):
            d["p_m8"] = p
            d["p"] = d["p_one_sided"] if kind == "ni_x2" else p     # descriptive: H5c is the one-sided boundary test, not doubled, never guarded
        out[hid] = dict(d, exploratory=True, in_holm=False, note="exploratory (A24.5): raw p, not adjusted; no confirmatory conclusion")
    for hid in FAMILY:
        ps8[hid] = ((holm_detail or {}).get(hid) or {}).get("p")
    h8 = holm(ps8)
    for hid in out:
        out[hid]["p_holm_m8"] = h8[hid]["p_holm"]
    return out, {k: h8[k]["p_holm"] for k in FAMILY}


INCOMPLETE_WORDING = "incomplete study: no confirmatory H1 decision (prereg s13)"


def fixed_sequence(dec: dict) -> dict:
    """HISTORY (not called since A24.1: h1-rule-1 builds the fixed sequence into its claims; kept for the old selftests and audits).
    Prereg 2 / 9.3: superiority is tested ONLY if non-inferiority holds (fixed sequence). `P.h1_decision` (locked) checks the superiority
    branch first, and the null-imposed wild-bootstrap p-values at -5 pp and 0 are not guaranteed monotone, so in a borderline sample it
    could print `superior` with p_ni >= .025. Guard here, without touching the locked file: a `superior` that lacks the non-inferiority
    conditions drops to the next applicable row (never to `materially inferior`: it cannot be that and superior at once)."""
    a = P.ALPHA_ONE_SIDED
    ni_holds = dec["p_ni"] < a and dec["ci95"][0] > -MARGIN
    dec["fixed_sequence"] = {"applied": True, "non_inferiority_holds": bool(ni_holds), "downgraded": False}
    if dec["wording"] == "superior" and not ni_holds:
        dec["fixed_sequence"].update(downgraded=True, was="superior")
        dec["wording"] = "non-inferiority unresolved (inconclusive)"
    return dec


#: A24.1 (stage H-H1-stats): the H1 procedure. `validated` is True because h1-rule-1 passed Stage V100 (all-or-nothing, frozen in the ledger before
#: it ran). Were it False, H1 would be ESTIMATION-ONLY: estimate + interval, no NI / superior / inferior wording (the frozen fallback).
H1_PROCEDURE = {"version": H1S.PROCEDURE_VERSION, "candidate": H1S.H1_CANDIDATE, "alpha_star": H1S.ALPHA_STAR, "guard_K": 20, "validated": True,
                "evidence": "docs/final-test/h1_validation/v100_summary.C7.json", "v100_stage_sha16": "e739bc27df577ba7"}
ESTIMATION_ONLY_WORDING = "estimation only: no validated H1 procedure (A24.1); no non-inferiority, superiority or inferiority claim"


def h1_decision(tot, sizes, K, procedure=None) -> dict:
    """One H1 view through h1-rule-1 (C7). The interval is the NOMINAL 96.5 % CR2 interval (alpha* .0175 per tail); the claims follow the fixed
    sequence (superior only if non-inferior). Estimation-only when the procedure is not validated."""
    proc = H1_PROCEDURE if procedure is None else procedure
    r = H1S.h1_rule_c7(tot, sizes, K)
    dec = {"procedure": proc["version"], "estimate": r["estimate"], "se": r["se"], "nu": r["nu"], "interval": r["interval"],
           "interval_nominal_level": r["interval_nominal_level"], "guard": r["guard"], "claims": r["claims"], "wording": r["wording"],
           "fixed_sequence": {"applied": True, "non_inferiority_holds": bool(r["claims"]["ni"])}}
    if not proc.get("validated"):
        dec.update(claims=None, wording=ESTIMATION_ONLY_WORDING, estimation_only=True,
                   interval_note="CR2 interval, not validated for nominal coverage (low-discordance coverage was .82 in V100)")
    return dec


def h1_block(S, fam_of, drop_invalid, B, complete=True, procedure=None, drop_flag=None):
    """H1 = SR(G+M) - SR(O0) under h1-rule-1 (A24.1). primary = item-weighted (the validated estimand); family-weighted and infra-paired are
    sensitivities through the same rule (not separately validated). The registered wild+percentile rule is not computed."""
    a, b = H1
    proc = H1_PROCEDURE if procedure is None else procedure
    out = {"contrast": f"SR({a}) - SR({b})", "procedure": dict(proc)}
    for name, kw in (("primary", {}), ("family_weighted", {"weighted": True}), ("infra_paired", {"drop_infra": True})):
        c = contrast(S, fam_of, a, b, drop_invalid, B, drop_flag=drop_flag, amended=drop_flag is not None, **kw)
        if c is None:
            out[name] = None
            continue
        dec = h1_decision(c["_tot"], c["_sizes"], c["discordant_families"], proc)
        dec["role"] = "primary (validated)" if name == "primary" else "sensitivity (same rule, not separately validated)"
        if not complete:                  # prereg s13: estimates over what was completed, WITHOUT the confirmatory H1 decision
            dec = {"descriptive_only": True, "procedure": dec["procedure"], "estimate": dec["estimate"], "interval": dec["interval"],
                   "interval_nominal_level": dec["interval_nominal_level"], "guard": dec["guard"], "claims": None, "wording": INCOMPLETE_WORDING}
        out[name] = {**clean(c), "decision": dec}
    if out.get("primary"):
        out["wording"] = out["primary"]["decision"]["wording"]
        for k in ("family_weighted", "infra_paired"):
            if out.get(k):
                out[k]["same_wording_as_primary"] = out[k]["decision"]["wording"] == out["wording"]
    return out


def equivalence_block(S, fam_of, drop_invalid, B, complete=True, drop_flag=None):
    """EXPLORATORY (A24 item 5, A24.1): |SR(G+M) - SR(O+M)| < 5 pp by E1 (CR2 + Satterthwaite 90 % interval + guard K >= 20). Never confirmatory;
    'not significantly different' is never 'equivalent'. In V100-style simulation its power was <= .02 at n = 125 items / 42 families."""
    c = contrast(S, fam_of, "G+M", "O+M", drop_invalid, B, drop_flag=drop_flag, amended=drop_flag is not None)
    if c is None:
        return {"available": False, "a": "G+M", "b": "O+M"}
    e = H1S.equiv_e1(c["_tot"], c["_sizes"], c["discordant_families"])
    if not complete:
        e = {**e, "equivalent": None, "wording": "incomplete study: descriptive only"}
    return {"available": True, **clean(c), "decision": e}


# ================================================================================================ descriptive blocks
def arm_rates(S, fam_of, lvl_of, arm, drop_invalid, level=None, lang=None):
    """Item-weighted SR of one arm with a family-cluster CI; lang=None -> mean over languages."""
    per_item = {}
    for (a, i, l), r in S.items():
        if a != arm or (i, l) in drop_invalid or (lang and l != lang):
            continue
        if level and lvl_of.get(i) != level:
            continue
        per_item.setdefault(i, []).append(float(r["sr"]))
    if not per_item:
        return None
    sc = {i: sum(v) / len(v) for i, v in per_item.items()}
    fams, tot, sizes = by_family(sc, fam_of) if len({fam_of[i] for i in sc}) >= 2 else (None, None, None)
    est = sum(sc.values()) / len(sc)
    ci = list(P.cluster_ci(tot, sizes)) if fams else None
    return {"sr": est, "ci95": ci, "n_items": len(sc), "n_families": len({fam_of[i] for i in sc})}


def metric_mean(S, lvl_of, arm, key, level, drop_invalid):
    v = [float(r[key]) for (a, i, l), r in S.items()
         if a == arm and lvl_of.get(i) == level and (i, l) not in drop_invalid and r.get(key) is not None]
    return {"mean": sum(v) / len(v), "n": len(v)} if v else None


def per_arm_tables(idx, arms, drop_invalid):
    S, fam, lvl = idx["S"], idx["fam"], idx["level"]
    levels = sorted({v for v in lvl.values() if v})
    out = {}
    for arm in arms:
        t = {"overall": arm_rates(S, fam, lvl, arm, drop_invalid), "level": {}, "lang": {}, "level_x_lang": {}, "diagnostics": {}}
        for L in levels:
            t["level"][L] = arm_rates(S, fam, lvl, arm, drop_invalid, level=L)
            for lg in LANGS:
                t["level_x_lang"][f"{L}/{lg}"] = arm_rates(S, fam, lvl, arm, drop_invalid, level=L, lang=lg)
        for lg in LANGS:
            t["lang"][lg] = arm_rates(S, fam, lvl, arm, drop_invalid, lang=lg)
        # the thesis's per-level metrics, as diagnostics next to SR: L1 SR, L2 PC, L3 TDS (final plan)
        t["diagnostics"] = {"L1_sr": t["level"].get("L1"),
                            "L2_pc_final": metric_mean(S, lvl, arm, "pc_final", "L2", drop_invalid),
                            "L3_tds_final": metric_mean(S, lvl, arm, "tds_final", "L3", drop_invalid),
                            "strict_sr": _plain_mean(S, arm, "strict_sr", drop_invalid),
                            "unrealistic_slice_rate": _plain_mean(S, arm, "unrealistic_slice", drop_invalid),
                            "prerequisite_realistic_sr": realistic_sr(S, fam, lvl, arm, drop_invalid),
                            "slice_flags": slice_flag_counts(S, arm, drop_invalid)}
        out[arm] = t
    return out


def realistic_sr(S, fam_of, lvl_of, arm, drop_invalid):
    """Prereg s8 (exploratory): prerequisite-realistic SR = SR with the known-unrealistic slice successes (the goal was met although no
    knife was held at a slice step) set to 0. Item-weighted with the family-cluster CI, like every SR. Rows whose holding was UNKNOWN
    (`unrealistic_slice` None) are NOT counted as unrealistic; `slice_flags` reports how many there are."""
    S2 = {k: (dict(r, sr=0.0) if r.get("unrealistic_slice") is True or (isinstance(r.get("unrealistic_slice"), (int, float)) and
                                                                         not isinstance(r.get("unrealistic_slice"), bool) and
                                                                         float(r["unrealistic_slice"]) == 1.0) else r)
          for k, r in S.items() if k[0] == arm}
    return arm_rates(S2, fam_of, lvl_of, arm, drop_invalid)


def slice_flag_counts(S, arm, drop_invalid):
    rs = [r for (a, i, l), r in S.items() if a == arm and (i, l) not in drop_invalid]
    return {"rows": len(rs), "unrealistic_true": sum(1 for r in rs if r.get("unrealistic_slice") is True),
            "unknown_none": sum(1 for r in rs if "unrealistic_slice" in r and r.get("unrealistic_slice") is None),
            "slice_rows": sum(1 for r in rs if int(r.get("n_slice") or 0) > 0)}


def _plain_mean(S, arm, key, drop_invalid):
    v = [float(r[key]) for (a, i, l), r in S.items() if a == arm and (i, l) not in drop_invalid and r.get(key) is not None]
    return {"mean": sum(v) / len(v), "n": len(v)} if v else None


def infra_table(S, arms):
    """Per arm: planner transport failures, execution infrastructure failures (blocked state / crash after the re-runs) and the A21 flag
    (more than INFRA_FLAG_RATE of the scored cells are infrastructure failures)."""
    out = {}
    for arm in arms:
        rs = [r for (a, i, l), r in S.items() if a == arm]
        n = len(rs)
        pl = sum(1 for r in rs if r.get("infra") or r.get("run_class") == "infra_planner")
        ex = sum(1 for r in rs if r.get("exec_failure") or r.get("run_class") == "infra_exec")
        tot = sum(1 for r in rs if is_infra(r))
        out[arm] = {"n": n, "infra": pl, "infra_rate": (pl / n) if n else None, "exec_infra": ex, "exec_infra_rate": (ex / n) if n else None,
                    "infra_total": tot, "infra_total_rate": (tot / n) if n else None,
                    "infrastructure_sensitive": bool(n and tot / n > C.INFRA_FLAG_RATE),
                    "invalid": sum(1 for r in rs if is_invalid_row(r))}
    return out


def dependence_block(S, arms) -> dict:
    """A21 D: how many scored cells are copies of another cell's execution. Per arm: cells, distinct units, cells whose unit is also used by another arm.
    A copy is never an independent replication; the independent-execution count of an arm is its number of distinct units."""
    by_unit = {}
    for (a, i, l), r in S.items():
        if r.get("unit") is not None:
            by_unit.setdefault(r["unit"], set()).add(a)
    out = {}
    for arm in arms:
        rs = [r for (a, i, l), r in S.items() if a == arm]
        units = {r.get("unit") for r in rs if r.get("unit") is not None}
        shared = sum(1 for r in rs if r.get("unit") is not None and len(by_unit.get(r["unit"], ())) > 1)
        out[arm] = {"cells": len(rs), "distinct_units": len(units), "cells_sharing_a_unit_with_another_arm": shared,
                    "shared_share": (shared / len(rs)) if rs else None}
    return out


def execution_link_problems(score_rows, exec_rows) -> list:
    """A21 D: every score row of an executed plan must be linked to its execution: a unit key, an exec record for (unit, run), and that record must
    list the plan cell. Without the link a copied outcome could not be told from an independent run. [] = every link holds."""
    ex = {}
    for e in exec_rows or ():
        if e.get("unit") is not None:
            ex[(e["unit"], int(e.get("run", 0)))] = e                              # latest record wins (resume)
    out = []
    for r in score_rows:
        if r.get("failure_class") in ("plan_rejected", "plan_infra", "plan_ceiling", "missing_exec") or r.get("exec_status") is None:
            continue
        u, run = r.get("unit"), int(r.get("run", 0))
        e = ex.get((u, run))
        if u is None:
            out.append(f"{r.get('key')}: no unit")
        elif e is None:
            out.append(f"{r.get('key')}: no exec record for unit {str(u)[:10]} run {run}")
        elif r.get("cell") not in (e.get("cells") or ()):
            out.append(f"{r.get('key')}: the exec record of its unit does not list the cell")
    return out


def execution_block(score_rows, exec_rows) -> dict:
    """A21 D: simulator time. The wall time of one unit run is split equally among the scored cells (r = 0, run 0) that share the unit; each arm is
    charged its share; the total counts every real run once. Reported as a shared benchmarking cost, not a per-arm deployment cost."""
    if not exec_rows:
        return {"available": False}
    wall = {}
    for e in exec_rows:
        if e.get("status") in ("done", "not_executed") and int(e.get("run", 0)) == 0 and e.get("unit"):
            wall[e["unit"]] = float(e.get("wall_s") or 0.0)           # latest record wins (resume); run 0 only
    cells = {}
    for r in score_rows:
        if r.get("scored") and r.get("unit") in wall:
            cells.setdefault(r["unit"], []).append(r["arm"])
    per_arm = {}
    for u, arms_ in cells.items():
        for a in arms_:
            per_arm[a] = per_arm.get(a, 0.0) + wall[u] / len(arms_)
    return {"available": True, "unique_runs": len(cells), "total_wall_s": round(sum(wall[u] for u in cells), 1),
            "per_arm_wall_s": {a: round(v, 1) for a, v in sorted(per_arm.items())},
            "rule": "unit wall time split equally among the scored cells that share the unit"}


def variance_decomposition(rows, arms):
    """REPEAT subset: planner variance across the 3 calls (r) at run 0, execution variance across the 3 runs of one cached
    plan. Everything is computed from the score rows: equal unit key <=> identical plan in an equivalent world."""
    cells = {}
    for r in rows:
        try:
            p = C.parse_pres(r["pres"])
        except (ValueError, KeyError):
            continue
        if p["world"] != "canon":
            continue
        cells.setdefault((r["arm"], r["pres"]), {}).setdefault(r["r"], {})[r["run"]] = r
    out = {}
    for arm in arms:
        pres_list = [(pid, rr) for (a, pid), rr in cells.items() if a == arm and len(rr) >= 2]
        if not pres_list:
            continue
        ident, spread, planner_var, exec_vars, exec_agree, n_exec = [], [], [], [], [], 0
        for pid, rr in pres_list:
            run0 = [rr[k][0] for k in sorted(rr) if 0 in rr[k]]
            if len(run0) >= 2:
                srs = [float(x["sr"]) for x in run0]
                ident.append(1.0 if len({x["unit"] for x in run0}) == 1 else 0.0)
                spread.append(max(srs) - min(srs))
                m = sum(srs) / len(srs)
                planner_var.append(sum((s - m) ** 2 for s in srs) / len(srs))
            for k in sorted(rr):
                runs = [float(rr[k][q]["sr"]) for q in sorted(rr[k])]
                if len(runs) >= 2:
                    n_exec += 1
                    m = sum(runs) / len(runs)
                    exec_vars.append(sum((s - m) ** 2 for s in runs) / len(runs))
                    exec_agree.append(1.0 if len(set(runs)) == 1 else 0.0)
        avg = lambda v: (sum(v) / len(v)) if v else None      # noqa: E731
        out[arm] = {"n_presentations": len(pres_list),
                    "planner": {"n": len(ident), "plan_identity_rate": avg(ident), "sr_spread_mean": avg(spread), "sr_variance_mean": avg(planner_var)},
                    "execution": {"n": n_exec, "outcome_agreement_rate": avg(exec_agree), "sr_variance_mean": avg(exec_vars)}}
    return out


def cost_block(arms, S, plan_rows, practice_rows):
    """Cost per success. Spend = plan-journal cost_usd per arm (every call, incl. repairs, REPEAT repeats, curve, reloc);
    practice cost is shared infrastructure of the +M arms and is added to each of them in full (an upper bound)."""
    prac = sum(float(r.get("cost_usd") or 0.0) for r in (practice_rows or []))
    out = {}
    for arm in arms:
        succ = sum(float(r["sr"]) for (a, i, l), r in S.items() if a == arm and not is_invalid_row(r))
        spend = sum(float(r.get("cost_usd") or 0.0) for r in (plan_rows or []) if r.get("arm") == arm)
        tok_in = sum(int(r.get("tokens_in") or 0) for r in (plan_rows or []) if r.get("arm") == arm)
        tok_out = sum(int(r.get("tokens_out") or 0) for r in (plan_rows or []) if r.get("arm") == arm)
        tot = spend + (prac if arm.endswith("+M") else 0.0)
        out[arm] = {"successes": succ, "plan_usd": spend, "practice_usd_added": prac if arm.endswith("+M") else 0.0,
                    "usd_per_success": (tot / succ) if succ > 0 and plan_rows is not None else None,
                    # A24.2: positive cost with zero successes is INFINITE, zero / zero is undefined; never a silent blank
                    "usd_per_success_state": ("unknown" if plan_rows is None else "finite" if succ > 0 else
                                              "infinite" if tot > 0 else "undefined"),
                    "tokens_in": tok_in, "tokens_out": tok_out}
    return out


def codex_block(idx, drop_invalid, B):
    """Exploratory and separate: X-<model>0 / X-<model>+M. Never pooled with API arms, never in the Holm family."""
    S, fam, lvl = idx["S"], idx["fam"], idx["level"]
    arms = sorted({a for (a, i, l) in S if a.startswith("X-")})
    out = {"note": "exploratory; model + CLI scaffold; never pooled with API arms; not in H1 or the Holm family", "arms": {}, "memory_gain": {}}
    for a in arms:
        out["arms"][a] = arm_rates(S, fam, lvl, a, drop_invalid)
    for a in arms:
        if a.endswith("+M"):
            base = a[:-2] + "0"
            if base in arms:
                c = contrast(S, fam, a, base, drop_invalid, B)
                if c:
                    out["memory_gain"][a] = clean(c)
    return out


# ================================================================================================ A24.2 cost-effectiveness
COST_VIEWS = ("headline", "electricity_only", "hardware_only", "hw_time_3x")
#: C is a ratio of TOTAL planning cost; a hardware-only C would divide by an API arm's zero hardware component, so it is not formed (the hardware part
#: is reported per arm instead).
C_VIEWS = ("headline", "electricity_only", "hw_time_3x")


def arm_kind(arm: str) -> str:
    """gemma (local, energy + hardware) / api (list-price USD) / codex (not costed). Prefix rule: G*, X*, everything else is API (O*, archived S* / H*)."""
    return "gemma" if arm.startswith("G") else "codex" if arm.startswith("X") else "api"


def integrate_power_j(samples, t0: float, t1: float) -> float | None:
    """Joules = trapezoid integral of sampled power (seconds, watts) over [t0, t1]; samples outside are used only to interpolate the window edges.
    For the producer of energy_j_gpu / energy_j_cpu (A24.2 'sampled power integrated over planning time'). None if the window has no coverage."""
    pts = sorted((float(t), float(w)) for t, w in samples if w is not None)
    if t1 <= t0 or len(pts) < 2 or pts[0][0] > t0 or pts[-1][0] < t1:
        return None

    def at(t):
        for (ta, wa), (tb, wb) in zip(pts, pts[1:]):
            if ta <= t <= tb:
                return wa if tb == ta else wa + (wb - wa) * (t - ta) / (tb - ta)
        return None
    xs = [(t0, at(t0))] + [p for p in pts if t0 < p[0] < t1] + [(t1, at(t1))]
    return sum((tb - ta) * (wa + wb) / 2.0 for (ta, wa), (tb, wb) in zip(xs, xs[1:]))


def _plan_seconds(r, prep=False):
    """Plan row: planning_s > wall_s (the plan cell's wall time), nothing else (Astra P-cost). Prep / practice row: planning_s > plan_secs only -- a
    practice row's wall_s is the whole EPISODE (execution included), and executor cost is excluded by A24.2."""
    if r.get("planning_s") is not None:
        return float(r["planning_s"])
    if prep:
        return float(r["plan_secs"]) if r.get("plan_secs") is not None else None
    if r.get("wall_s") is not None:
        return float(r["wall_s"])
    return None


UNMEASURED_CLASSES = ("practice", "preparation", "refused_interval", "no_energy_row")   # C5: why a Gemma row has no validated measurement
#   practice = a row of practice.jsonl, preparation = a row of prep_cost.jsonl (the SOURCE is set by analyse(), never read from the row);
#   a 'practice' row is eligible only if its content id (practice_row_id) is one of terms.rows.row_ids, used once (Astra C5 r4 #1)
UNMEASURED_CLAIMS = ("headline", "sensitivity_only")                       # headline: estimated rows count as known (flagged); sensitivity_only: not
#: the terms Arash approved in the ledger (docs/final-test/progress.md, 2026-10-14 "ARASH DECISION: energy estimate for the 10 unmeasured practice
#: rows = OK", confirmed in "ARASH DECISION: simplify the energy method (C5)"): only the 10 practice-v2 rows, estimated from the measured rate,
#: flagged and disclosed, the measured rows the primary figure. ACTIVE: copied into ENERGY_METHOD.unmeasured_rule_terms right after this table.
UNMEASURED_TERMS_APPROVED = {"claim": "headline", "eligible": ["practice"], "rate": "measured_rate",
                             "rows": {"journal": "docs/final-test/matrix/practice-v2/practice.jsonl",
                                      "journal_sha256": "ab84215a35cc082a770815d630949eeedbf39d8eaf63fe18ce47f9d846d31dda",
                                      "row_ids": [
                                         "1d0a3b6e730c510041a8606505bd3c798e2eed7b61baed949ad1898d21992f38",
                                         "2178d14b21d0856da776b465ce620d6870cd01623b360bf3bfb8043f1a32e372",
                                         "2dec82fa8cf40f5de41dc4fdff8223d44f9f9c8ac41fef7ef56a026ea83a9774",
                                         "2efa5590a87ebd75a498bd5de13e20b65aabdc971d16ef8ce28504a7364f9ea4",
                                         "3cbd72acca05775964fb3ed3ed4850effb89c389372c89245442bc1c9853dcc1",
                                         "777e908a5ac8254444bbb53f62bd1a2b65af4445c1fa1f0b10e4f4618c26d608",
                                         "7d361af12284598cbe8f72ed6b34312580e157f6de72d3d49f18bcca307753e5",
                                         "b7fc99c4a11b691fb864cd082ba69caa2f1c22593ede037e4223148cc44a81f4",
                                         "bd9f087c5d208556af0b907ff72ef1f99e31aaae4e745db750c56a25cf31ef66",
                                         "dba5a9f8d647f60bac20b7a18a1d38ebbb2b2408095c04e0a3e372e3d907fd62"]}}
ENERGY_METHOD["unmeasured_rule_terms"] = json.loads(json.dumps(UNMEASURED_TERMS_APPROVED))   # basis "arash-decision:2026-10-14"


def practice_row_id(r) -> str:
    """Content id of one practice row: sha256 of its canonical JSON (sorted keys, no spaces, UTF-8) without the energy fields and the
    cost_kind / cost_source added by the analysis. The approved rows are bound by these ids (terms.rows.row_ids)."""
    d = {k: v for k, v in r.items() if k not in ENERGY_FIELDS + ("cost_kind", "cost_source")}
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def unmeasured_terms_problems(terms, em) -> list:
    """C5: the structured terms of an unmeasured-rule amendment -> problems (empty = well formed and consistent with the energy method)."""
    if not isinstance(terms, dict):
        return ["unmeasured_rule_terms missing (a dict {rate, eligible, claim})"]
    why = []
    el = terms.get("eligible")
    keys = {"rate", "eligible", "claim"} | ({"rows"} if isinstance(el, list) and "practice" in el else set())
    if set(terms) != keys:
        why.append(f"unmeasured_rule_terms keys {sorted(terms)} != {sorted(keys)} (rows is required exactly when 'practice' is eligible)")
    est = em.get("unmeasured_estimate") if isinstance(em.get("unmeasured_estimate"), dict) else {}
    if not (isinstance(terms.get("rate"), str) and terms.get("rate") == est.get("rate")):
        why.append(f"unmeasured_rule_terms.rate {terms.get('rate')!r} != energy_method.unmeasured_estimate.rate {est.get('rate')!r}")
    if not (isinstance(est.get("min_cells"), int) and not isinstance(est.get("min_cells"), bool) and est["min_cells"] >= 1):
        why.append(f"energy_method.unmeasured_estimate.min_cells {est.get('min_cells')!r} is not a positive integer")
    if not (isinstance(el, list) and el and all(isinstance(x, str) and x in UNMEASURED_CLASSES for x in el) and len(set(el)) == len(el)):
        why.append(f"unmeasured_rule_terms.eligible {el!r} is not a non-empty subset of {list(UNMEASURED_CLASSES)}")
    if not (isinstance(terms.get("claim"), str) and terms.get("claim") in UNMEASURED_CLAIMS):
        why.append(f"unmeasured_rule_terms.claim {terms.get('claim')!r} not in {list(UNMEASURED_CLAIMS)}")
    if "rows" in terms:
        rw = terms.get("rows")
        hx = lambda x: isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)     # noqa: E731
        if not (isinstance(rw, dict) and set(rw) == {"journal", "journal_sha256", "row_ids"} and isinstance(rw.get("journal"), str)
                and rw["journal"] and hx(rw.get("journal_sha256")) and isinstance(rw.get("row_ids"), list) and rw["row_ids"]
                and all(hx(x) for x in rw["row_ids"]) and len(set(rw["row_ids"])) == len(rw["row_ids"])):
            why.append("unmeasured_rule_terms.rows must be {journal, journal_sha256, row_ids: distinct sha256 hex ids}")
    return why


def unmeasured_class(r, prep: bool = False) -> str:
    if prep:                                       # the source is set by analyse() (cost_source); anything else is plain preparation
        return "practice" if r.get("cost_source") == "practice" else "preparation"
    return "refused_interval" if r.get("energy_state") == "unmeasured" else "no_energy_row"


def approved_practice_ids(prep_rows, terms) -> set:
    """The practice-row ids the approved terms may estimate: in terms.rows.row_ids AND present exactly once among the practice-source rows
    (a duplicated approved row is never estimated twice; Astra C5 r4 #1)."""
    if not terms or "practice" not in terms.get("eligible", []) or not isinstance(terms.get("rows"), dict):
        return set()
    allowed = set(terms["rows"]["row_ids"])
    cnt = {}
    for r in prep_rows or ():
        if r.get("cost_source") == "practice":
            i = practice_row_id(r)
            cnt[i] = cnt.get(i, 0) + 1
    return {i for i, n in cnt.items() if n == 1 and i in allowed}


def approved_terms(cfg) -> dict | None:
    """The active, well-formed unmeasured-rule terms of the analysis plan, else None (rule not approved -> every unmeasured row UNKNOWN)."""
    em = cfg.get("energy_method") if isinstance(cfg.get("energy_method"), dict) else {}
    terms = em.get("unmeasured_rule_terms")
    if str(em.get("unmeasured_rule_basis") or "").startswith(("amendment:", "arash-decision:")) and not unmeasured_terms_problems(terms, em):
        return terms
    return None


def measured_rate(plan_rows, cfg) -> dict:
    """C5 unmeasured_estimate: the run's measured Gemma rate. Only rows with an energy figure from the join (energy_state 'measured', so never
    an estimate) and finite planning_s > 0 count. -> {rate_w or None, cells, joules, seconds, min_cells, reason}."""
    em = cfg.get("energy_method") if isinstance(cfg.get("energy_method"), dict) else {}
    est = em.get("unmeasured_estimate") if isinstance(em.get("unmeasured_estimate"), dict) else {}
    n, j, s = 0, 0.0, 0.0
    for r in plan_rows or ():
        if arm_kind(r.get("arm") or "") != "gemma" or r.get("energy_state") != "measured":
            continue
        if not (_has_energy(r) and _finite(r.get("planning_s"), 0, strict=True)):
            continue
        n, j, s = n + 1, j + _row_joules(r), s + float(r["planning_s"])
    mc = est.get("min_cells")
    ok = isinstance(mc, int) and n >= mc and s > 0
    return {"rate_w": (j / s) if ok else None, "cells": n, "joules": j, "seconds": s, "min_cells": mc,
            "reason": None if ok else f"{n} measured Gemma rows (< min_cells {mc}) or no measured seconds: no rate"}


def practice_binding_missing(prep_rows, terms) -> list:
    """The approved practice-row ids (terms.rows.row_ids) NOT present exactly once among the practice-source rows the analysis consumes
    (Astra C5 r5 #1): non-empty -> the practice cost of the consumer arms is incomplete (UNKNOWN), never silently smaller."""
    if not terms or not isinstance(terms.get("rows"), dict):
        return []
    return sorted(set(terms["rows"]["row_ids"]) - approved_practice_ids(prep_rows, terms))


def practice_journal_problems(root, terms) -> list:
    """Freeze-time check of terms.rows (Astra C5 r5 #1): the named journal exists under root, its bytes have journal_sha256, and its rows'
    content ids are exactly row_ids (no duplicates, none missing, none extra). -> problems."""
    if not isinstance(terms, dict) or "rows" not in terms:
        return []
    rw = terms["rows"] if isinstance(terms.get("rows"), dict) else {}
    jp = str(rw.get("journal") or "")
    parts = jp.replace("\\", "/").split("/")
    bad = (not jp or os.path.isabs(jp) or jp.startswith(("/", "\\")) or os.path.splitdrive(jp)[0] != "" or ":" in jp
           or any(x in ("..", ".", "") for x in parts))
    rr = os.path.realpath(root)
    fp = os.path.realpath(os.path.join(rr, *parts)) if not bad else ""
    if bad or os.path.commonpath([rr, fp]) != rr or fp == rr or not os.path.isfile(fp):
        return [f"terms.rows.journal {jp!r} is not an existing file inside the repo (relative, '/' separated, no '.' / '..' parts)"]
    raw = open(fp, "rb").read()
    why = []
    if hashlib.sha256(raw).hexdigest() != rw.get("journal_sha256"):
        why.append(f"terms.rows.journal_sha256 != the sha256 of {jp} on disk")
    ids = []
    try:
        lines = [ln for ln in raw.decode("utf-8").split("\n") if ln.strip()]
    except UnicodeDecodeError as e:
        return why + [f"{jp} is not UTF-8: {type(e).__name__}"]
    for n, ln in enumerate(lines, 1):
        try:
            r = json.loads(ln)
        except ValueError as e:
            why.append(f"{jp} row {n} does not parse: {type(e).__name__}")
            continue
        if not isinstance(r, dict):
            why.append(f"{jp} row {n} is not a JSON object ({type(r).__name__})")
            continue
        ids.append(practice_row_id(r))
    if len(ids) != len(lines):
        return why
    if len(set(ids)) != len(ids):
        why.append(f"{jp} holds duplicate rows")
    if sorted(set(ids)) != sorted(rw.get("row_ids") or []):
        why.append(f"the row ids of {jp} ({len(set(ids))}) != terms.rows.row_ids ({len(rw.get('row_ids') or [])})")
    return why


def price_row(r, kind: str, cfg: dict, prep: bool = False, rate_w=None, practice_ids=frozenset()) -> dict:
    """One plan / prep row -> its cost components (USD). energy None = not measured (a Gemma row then has an UNKNOWN cost). rate_w = the run's
    measured Gemma rate (measured_rate); it is used ONLY for a row whose unmeasured class the approved terms make eligible."""
    api = float(r.get("cost_usd") or 0.0)
    sec = _plan_seconds(r, prep)
    out = {"api": api, "energy": 0.0, "hardware": 0.0, "seconds": sec, "joules": None, "known": True}
    has_e = _has_energy(r)
    em = cfg.get("energy_method")
    em = em if isinstance(em, dict) else {}
    terms = approved_terms(cfg)
    if kind == "gemma" or has_e:
        if has_e:
            j = _row_joules(r)
            out["joules"], out["energy"], out["measured_joules"] = j, j / 3_600_000.0 * cfg["usd_per_kwh"], j
        elif kind == "gemma":
            cls = unmeasured_class(r, prep)
            if sec is not None:
                out["unmeasured_seconds"], out["unmeasured_class"] = sec, cls      # reported in every case
            eligible = terms is not None and cls in terms["eligible"] and (       # the approved scope, row by row (Astra C5 r3 #1, r4 #1)
                cls != "practice" or practice_row_id(r) in practice_ids)
            if eligible and sec is not None and _finite(rate_w, 0):
                j = float(rate_w) * sec
                out["estimated_joules"] = j                                         # flagged; never a measurement
                if terms["claim"] == "headline":
                    out["joules"], out["energy"], out["estimated"] = j, j / 3_600_000.0 * cfg["usd_per_kwh"], True
                else:
                    out["known"], out["energy"] = False, None                       # sensitivity_only: disclosed, stays UNKNOWN
            else:
                out["known"], out["energy"] = False, None
        if kind == "gemma":
            if sec is None:
                out["known"], out["hardware"] = False, None
            else:
                out["hardware"] = cfg["hardware_price_usd"] / cfg["operating_hours"] * (sec / 3600.0)
    return out


def plan_line_shas(path: str) -> list:
    """sha256 of every non-empty line of a JSONL journal (CR/LF stripped), in the order read_jsonl returns the rows (C5 energy join)."""
    out = []
    with open(path, "rb") as fh:
        lines = fh.read().split(b"\n")
    for i, ln in enumerate(lines):
        s = ln.rstrip(b"\r")
        if not s.strip():
            continue
        if i == len(lines) - 1:                                          # Journal.rows skips a torn (unparsable) tail
            try:
                json.loads(s.decode("utf-8"))
            except ValueError:
                continue
        out.append(hashlib.sha256(s).hexdigest())
    return out


def load_energy_rows(d: str) -> list:
    """Every row of energy*.jsonl in directory d (C5 meter output), identical copies once. Unparsable lines are skipped and counted."""
    rows, seen, bad = [], set(), 0
    if not d or not os.path.isdir(d):
        return rows
    for fn in sorted(os.listdir(d)):
        if not (fn.startswith("energy") and fn.endswith(".jsonl")):
            continue
        with open(os.path.join(d, fn), encoding="utf-8") as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                try:
                    r = json.loads(ln)
                except ValueError:
                    bad += 1
                    continue
                k = json.dumps(r, sort_keys=True)
                if k not in seen:
                    seen.add(k)
                    rows.append(r)
    if bad:
        rows.append({"kind": "_unparsable", "n": bad})
    return rows


def _f3(x):
    return None if x is None else round(float(x), 3)


ENERGY_FIELDS = ("energy_j_gpu", "energy_j_cpu", "energy_state", "planning_s", "energy_method", "energy_row_ts")


def strip_energy_fields(rows):
    """Sealed analysis: energy fields that rows carry THEMSELVES are not evidence (only the C5 energy journal is). -> (rows, n stripped)."""
    out, n = [], 0
    for r in rows or ():
        if any(k in r for k in ENERGY_FIELDS):
            n += 1
            r = {k: v for k, v in r.items() if k not in ENERGY_FIELDS}
        out.append(r)
    return out, n


def _finite(x, lo=None, strict=False):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    if v != v or v in (math.inf, -math.inf):
        return False
    return lo is None or (v > lo if strict else v >= lo)


def _has_energy(r) -> bool:
    """C5 simple method: a row has an energy figure when energy_j_gpu is a finite number >= 0; energy_j_cpu is optional (None = not measured,
    the c5-simple-1 case; a finite value >= 0 is added, e.g. synthetic data)."""
    return _finite(r.get("energy_j_gpu"), 0) and (r.get("energy_j_cpu") is None or _finite(r.get("energy_j_cpu"), 0))


def _row_joules(r) -> float:
    return float(r["energy_j_gpu"]) + (0.0 if r.get("energy_j_cpu") is None else float(r["energy_j_cpu"]))


def attach_energy(plan_rows, energy_rows, line_shas=None, cfg=None, strict=False):
    """C5 join (energy_method 'join'): Gemma plan rows <- their energy row. Returns (rows, report). strict (the sealed analysis): energy fields
    the rows carry themselves are removed first and line shas are required. A row matched to exactly one VALID energy row with a measured,
    problem-free interval gets planning_s + energy_j_gpu (+ energy_j_cpu when present) (energy_state 'measured'); a matched row whose interval was refused gets
    planning_s and energy_state 'unmeasured'; anything else is left without energy. price_row prices a row without energy as UNKNOWN unless the
    unmeasured rule is approved. Non-Gemma rows are never touched."""
    cfg = A24_2_COST if cfg is None else cfg
    em = cfg.get("energy_method")
    mid = em.get("id") if isinstance(em, dict) else None
    rep = {"method": mid, "strict": bool(strict), "energy_rows": 0, "open_rows": 0, "gemma_plan_rows": 0, "measured": 0, "refused_interval": 0,
           "no_energy_row": 0, "ambiguous": 0, "wrong_method": 0, "invalid_energy_row": 0, "unbound_energy_rows": 0,
           "interval_shorter_than_wall": 0, "fields_disagree": 0, "already_present": 0, "inline_fields_stripped": 0,
           "unparsable_lines": 0, "idle_floor": []}
    if strict:
        if line_shas is None:
            raise AnalysisError("C5 energy join (strict): the plan journal line shas are required")
        plan_rows, rep["inline_fields_stripped"] = strip_energy_fields(plan_rows)
    if line_shas is not None and len(line_shas) != len(plan_rows or ()):
        raise AnalysisError("C5 energy join: line_shas and plan rows differ in length")
    by_sha, by_tuple = {}, {}
    for e in energy_rows or ():
        if e.get("kind") == "_unparsable":
            rep["unparsable_lines"] += int(e.get("n") or 0)
            continue
        if e.get("kind") == "open":
            rep["open_rows"] += 1
            fl = ((e.get("meter") or {}).get("idle_floor") or {})
            rep["idle_floor"].append({"ts": e.get("ts"), "gpu_w": fl.get("gpu_w")})
            continue
        if e.get("kind") != "cell":
            continue
        rep["energy_rows"] += 1
        if mid is None or e.get("method") != mid:
            rep["wrong_method"] += 1
            continue
        b = e.get("bound")
        if not isinstance(b, dict) or not b.get("plan_line_sha256"):
            rep["unbound_energy_rows"] += 1
            continue
        by_sha.setdefault(b["plan_line_sha256"], []).append(e)
        by_tuple.setdefault((e.get("key"), b.get("ts"), b.get("request_sha"), b.get("status"), _f3(b.get("wall_s"))), []).append(e)
    out = []
    for n, r in enumerate(plan_rows or ()):
        if arm_kind(str(r.get("arm") or "")) != "gemma":
            out.append(r)
            continue
        rep["gemma_plan_rows"] += 1
        if not strict and _has_energy(r):
            rep["already_present"] += 1                                  # synthetic / test data only (never in the sealed analysis)
            out.append(r)
            continue
        r = {k: v for k, v in r.items() if k != "energy_state"}
        tup = (r.get("key"), r.get("ts"), r.get("request_sha"), r.get("status"), _f3(r.get("wall_s")))
        hits = by_sha.get(line_shas[n], []) if line_shas is not None else by_tuple.get(tup, [])
        if not hits:
            rep["no_energy_row"] += 1
            out.append(r)
            continue
        if len(hits) > 1:
            rep["ambiguous"] += 1
            out.append(r)
            continue
        e = hits[0]
        b = e["bound"]
        if (e.get("key"), b.get("ts"), b.get("request_sha"), b.get("status"), _f3(b.get("wall_s"))) != tup \
                or e.get("arm") != r.get("arm") or e.get("pres") != r.get("pres"):
            rep["fields_disagree"] += 1
            out.append(r)
            continue
        ps = e.get("planning_s")
        if e.get("schema") != "ftm.energy/1" or not _finite(ps, 0.0, strict=True):
            rep["invalid_energy_row"] += 1
            out.append(r)
            continue
        if r.get("wall_s") is not None and float(ps) + 1e-3 < float(r["wall_s"]):
            rep["interval_shorter_than_wall"] += 1
            out.append(r)
            continue
        nr = dict(r, planning_s=float(ps), energy_method=mid, energy_row_ts=e.get("ts"))
        good = not e.get("problems") and _has_energy(e)
        if good:
            nr.update(energy_j_gpu=float(e["energy_j_gpu"]), energy_j_cpu=None if e.get("energy_j_cpu") is None else float(e["energy_j_cpu"]),
                      energy_state="measured")
            rep["measured"] += 1
        else:
            if e.get("energy_j_gpu") is not None or e.get("energy_j_cpu") is not None:
                rep["invalid_energy_row"] += 1                           # energies alongside problems, negative or non-finite: not used
            nr["energy_state"] = "unmeasured"
            rep["refused_interval"] += 1
        out.append(nr)
    rep["complete"] = rep["gemma_plan_rows"] == rep["measured"] + rep["already_present"]
    return out, rep


def _view_usd(c: dict, view: str, cfg: dict):
    """Cost components -> USD under one A24.2 view; None if unknown."""
    if c.get("known") is False or c.get("energy") is None or c.get("hardware") is None:
        return None
    hw = c["hardware"] + c.get("alloc_hardware", 0.0)
    if view == "hardware_only":
        return hw
    base = c["api"] + c["energy"] + c.get("alloc_api", 0.0) + c.get("alloc_energy", 0.0)
    if view == "electricity_only":
        return base
    if view == "hw_time_3x":
        return base + hw * cfg["hardware_time_factor"]
    return base + hw


def _ratio_state(num: float, den: float):
    if den > 0:
        return num / den, "finite"
    return (math.inf, "infinite") if num > 0 else (None, "undefined")


def boot_ratio(num_f, den_f, W) -> dict:
    """Point estimate + 95 % percentile CI of sum(num) / sum(den) under the cluster bootstrap matrix W (families x replicates). A24.2: denominator 0 ->
    +inf (numerator > 0) or undefined (0 / 0); infinite replicates are kept, any undefined replicate makes the bounds not obtainable."""
    import numpy as np
    num, den = np.asarray(num_f, float), np.asarray(den_f, float)
    est, state = _ratio_state(float(num.sum()), float(den.sum()))
    nb, db = W @ num, W @ den
    undefined = (db <= 0) & (nb <= 0)
    infinite = (db <= 0) & (nb > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.where(db > 0, nb / np.where(db > 0, db, 1.0), np.inf)
    out = {"estimate": est, "state": state, "B": int(W.shape[0]), "n_infinite": int(infinite.sum()), "n_undefined": int(undefined.sum())}
    if out["n_undefined"]:
        out.update(ci95=None, bounds_obtainable=False, reason=f"{out['n_undefined']} bootstrap replicate(s) undefined (0 / 0)")
        return out
    lo, hi = np.quantile(stat, [0.025, 0.975], method="inverted_cdf")
    out.update(ci95=[float(lo), float(hi)], bounds_obtainable=bool(np.isfinite(lo)) or bool(np.isfinite(hi)))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        out["note"] = "a bound is infinite (replicates with cost and zero successes)"
    return out


def _cost_rows_by_cell(plan_rows):
    """plan rows -> ({(arm, item, lang): [rows]} for the scored workload, {arm: [rows]} for everything else)."""
    cells, other = {}, {}
    for r in plan_rows or ():
        arm = r.get("arm")
        try:
            p = C.parse_pres(r["pres"])
        except (ValueError, KeyError, AttributeError):
            other.setdefault(arm, []).append(r)
            continue
        if p["world"] == "canon" and int(r.get("r", 0) or 0) == 0 and r.get("block") in (None, "main"):
            cells.setdefault((arm, p["item"], p["lang"]), []).append(r)
        else:
            other.setdefault(arm, []).append(r)
    return cells, other


def consumer_arms(arms, cfg) -> list:
    """The arms that consume shared practice / preparation (A24.2 allocation): suffixes in cfg['allocation_rule']['consumer_suffixes']."""
    suf = tuple((cfg.get("allocation_rule") or {}).get("consumer_suffixes") or ("+M",))
    return [a for a in arms if a.endswith(suf)]


def _allocations(arms, prep_rows, cfg, n_pres, rate_w=None):
    """{arm: {api, energy, hardware, known, rows}}: each prep row is priced by its OWN cost_kind (practice is Gemma-planned) and assigned to its
    `arms` / `arm`, else to the consumer arms; mode 'split' shares it in proportion to the targets' attempted presentations, 'full' gives each target
    the whole row (upper bound)."""
    out = {a: {"api": 0.0, "energy": 0.0, "hardware": 0.0, "known": True, "rows": 0, "estimated_rows": 0, "estimated_joules": 0.0,
               "estimated_seconds": 0.0, "measured_joules": 0.0, "unmeasured_rows": 0, "unmeasured_seconds": 0.0} for a in arms}
    mode = (cfg.get("allocation_rule") or {}).get("mode", "split")
    terms = approved_terms(cfg)
    pids = approved_practice_ids(prep_rows, terms)
    missing = practice_binding_missing(prep_rows, terms)
    for r in prep_rows or ():
        targets = [a for a in (r.get("arms") or ([r["arm"]] if r.get("arm") else consumer_arms(arms, cfg))) if a in out]
        if not targets:
            continue
        kind = r.get("cost_kind")
        c = price_row(r, kind, cfg, prep=True, rate_w=rate_w, practice_ids=pids) if kind in ("gemma", "api", "codex") else {"known": False}
        tot_n = sum(n_pres.get(a, 0) for a in targets)
        for a in targets:
            if c.get("unmeasured_seconds") is not None:
                out[a]["unmeasured_rows"] += 1
                out[a]["unmeasured_seconds"] += c["unmeasured_seconds"] * (1.0 if mode == "full" else (
                    (n_pres.get(a, 0) / tot_n) if tot_n else 0.0))
            if not c["known"]:
                out[a]["known"] = False
                continue
            share = 1.0 if mode == "full" else ((n_pres.get(a, 0) / tot_n) if tot_n else 0.0)
            for k in ("api", "energy", "hardware"):
                out[a][k] += c[k] * share
            out[a]["rows"] += 1
            if c.get("estimated"):
                out[a]["estimated_rows"] += 1
                out[a]["estimated_joules"] += c["joules"] * share
                out[a]["estimated_seconds"] += c["seconds"] * share
            if c.get("measured_joules") is not None:
                out[a]["measured_joules"] += c["measured_joules"] * share
    if missing:                                     # Astra C5 r5 #1: an approved historical row not consumed exactly once -> incomplete
        for a in consumer_arms(arms, cfg):
            out[a]["known"] = False
            out[a]["approved_practice_rows_missing"] = len(missing)
    return out


def _unmeasured_summary(rows_s, secs_s, rows_a, secs_a, cfg, rate=None):
    rw = (rate or {}).get("rate_w")
    je = (secs_s + secs_a) * rw if rw is not None else None
    return {"scored_rows": rows_s, "scored_seconds": secs_s, "allocated_rows": rows_a, "allocated_seconds": secs_a,
            "joules_at_measured_rate": je,
            "usd_at_measured_rate": (je / 3_600_000.0 * cfg["usd_per_kwh"]) if je is not None else None, "measured_rate_w": rw,
            "note": "these rows have no energy figure: their cost is UNKNOWN unless the approved unmeasured rule makes them eligible (then "
                    "they are estimated at the measured rate and flagged)"}


def cost_effectiveness_block(S, fam_of, arms, drop_invalid, plan_rows, prep_rows=None, *, cfg=None, h1=None, complete=True, B=20000, seed=None,
                             freeze_required=False, freeze_recorded=False):
    """A24.2 item 1: (a) planning USD per success per arm, (b) R and C with 95 % family-cluster bootstrap CIs and the reading rule, (c) Pareto data, and
    planning latency per attempted task, under four cost views. Planning cost only. Never reads outcomes beyond the scored `sr` of S."""
    import numpy as np
    cfg = dict(A24_2_COST if cfg is None else cfg)
    out = {"assumptions": cfg, "available": plan_rows is not None}
    if plan_rows is None:
        out["reason"] = "no plan journal: costs unknown; the reading rule cannot be satisfied"
        out["claim"] = {"statement_permitted": False, "reason": out["reason"]}
        return out
    cells, other = _cost_rows_by_cell(plan_rows)
    pres = {a: sorted((i, l) for (x, i, l) in S if x == a and (i, l) not in drop_invalid) for a in arms}
    cohort = sorted({k for a in arms for k in pres[a]})
    missing_vs_cohort = {a: len(set(cohort) - set(pres[a])) for a in arms}
    rate = measured_rate(plan_rows, cfg)                                                  # C5 unmeasured_estimate (used only for eligible rows)
    out["energy_rate"] = dict(rate, active_terms=approved_terms(cfg))
    alloc = _allocations(arms, prep_rows, cfg, {a: len(cohort) for a in arms}, rate["rate_w"])   # registered workload = the cohort (equal shares)
    if (cfg.get("allocation_rule") or {}).get("practice_required") and not prep_rows:
        for a in consumer_arms(arms, cfg):
            alloc[a]["known"] = False
            alloc[a]["missing_practice_rows"] = True
    out["cohort_presentations"] = len(cohort)
    fams = sorted({fam_of[i] for a in arms for (i, l) in pres[a]})
    fidx = {f: k for k, f in enumerate(fams)}
    F = len(fams)
    W = np.random.default_rng(P.SEED if seed is None else seed).multinomial(F, [1.0 / F] * F, size=B).astype(float) if F else None

    # per presentation: cost components (allocation spread equally over the arm's attempted presentations), seconds, sr
    per = {}
    for a in arms:
        kind, n = arm_kind(a), len(pres[a])
        al = alloc[a]
        for (i, l) in pres[a]:
            rs = cells.get((a, i, l), [])
            c = {"api": 0.0, "energy": 0.0, "hardware": 0.0, "seconds": 0.0 if rs else None, "joules": 0.0, "known": bool(rs) and al["known"], "rows": len(rs),
                 "estimated_rows": 0, "estimated_joules": 0.0, "measured_joules": 0.0, "unmeasured_rows": 0, "unmeasured_seconds": 0.0,
                 **{"alloc_" + k: (al[k] / n if n else 0.0) for k in ("api", "energy", "hardware")}}
            for r in rs:
                pc = price_row(r, kind, cfg, rate_w=rate["rate_w"])
                c["joules"] = (c["joules"] + pc["joules"]) if (c["joules"] is not None and pc["joules"] is not None) else None
                if pc.get("estimated"):
                    c["estimated_rows"] += 1
                    c["estimated_joules"] += pc["joules"]
                if pc.get("measured_joules") is not None:
                    c["measured_joules"] += pc["measured_joules"]
                if pc.get("unmeasured_seconds") is not None:
                    c["unmeasured_rows"] += 1
                    c["unmeasured_seconds"] += pc["unmeasured_seconds"]
                # planning time is accumulated whether or not the cost is known (Astra C5 r2 #3: an UNKNOWN energy cost is not 0 s)
                c["seconds"] = (c["seconds"] + pc["seconds"]) if (c["seconds"] is not None and pc["seconds"] is not None) else None
                if not pc["known"]:
                    c["known"] = False
                    continue
                for k in ("api", "energy", "hardware"):
                    c[k] += pc[k]
            per[(a, i, l)] = c

    def fam_vec(items_vals):
        v = np.zeros(F)
        for f, x in items_vals:
            v[fidx[f]] += x
        return v

    def item_means(a, keys, val):
        """{item: mean over its presentations in `keys` of val(a, i, l)}; None if any value is None."""
        acc = {}
        for (i, l) in keys:
            x = val(a, i, l)
            if x is None:
                return None
            acc.setdefault(i, []).append(x)
        return {i: sum(v) / len(v) for i, v in acc.items()}

    sr = lambda a, i, l: float(S[(a, i, l)]["sr"])                       # noqa: E731
    by_arm, pareto = {}, []
    for a in arms:
        kind = arm_kind(a)
        keys = pres[a]
        missing = sum(1 for k in keys if per[(a,) + k]["rows"] == 0)
        unknown = sum(1 for k in keys if not per[(a,) + k]["known"] and per[(a,) + k]["rows"] > 0)
        succ_f = fam_vec((fam_of[i], sr(a, i, l)) for (i, l) in keys)
        im_sr = item_means(a, keys, sr) or {}
        n_f = fam_vec((fam_of[i], 1.0) for i in im_sr)
        sr_f = fam_vec((fam_of[i], v) for i, v in im_sr.items())
        row = {"kind": kind, "attempted_presentations": len(keys), "attempted_items": len(im_sr), "successes": float(succ_f.sum()),
               "missing_vs_cohort": missing_vs_cohort[a],
               "plan_rows_scored": sum(per[(a,) + k]["rows"] for k in keys), "presentations_without_plan_row": missing,
               "presentations_cost_unknown": unknown, "allocated": {k: alloc[a][k] for k in ("api", "energy", "hardware", "rows", "known", "estimated_rows", "estimated_joules")},
               "energy_measured": {"scored_joules": sum(per[(a,) + k]["measured_joules"] for k in keys), "allocated_joules": alloc[a]["measured_joules"],
                                   "note": "PRIMARY figure: validated measurements only"},
               "energy_estimated": {"scored_rows": sum(per[(a,) + k]["estimated_rows"] for k in keys),
                                    "scored_joules": sum(per[(a,) + k]["estimated_joules"] for k in keys),
                                    "allocated_rows": alloc[a]["estimated_rows"], "allocated_joules": alloc[a]["estimated_joules"],
                                    "allocated_seconds": alloc[a]["estimated_seconds"], "rate_w": rate["rate_w"], "rate_cells": rate["cells"],
                                    "rule": ("ESTIMATE (flagged, not a measurement): measured Gemma rate x planning seconds, ONLY for the unmeasured "
                                             "classes the approved unmeasured_rule_terms make eligible, counted as known only with claim 'headline'")},
               "energy_unmeasured": _unmeasured_summary(sum(per[(a,) + k]["unmeasured_rows"] for k in keys),
                                                        sum(per[(a,) + k]["unmeasured_seconds"] for k in keys),
                                                        alloc[a]["unmeasured_rows"], alloc[a]["unmeasured_seconds"], cfg, rate),
               "other_blocks_usd_api": sum(float(r.get("cost_usd") or 0.0) for r in other.get(a, [])), "other_blocks_rows": len(other.get(a, [])),
               "sr_item_weighted": boot_ratio(sr_f, n_f, W) if F else None, "views": {}}
        costed = kind != "codex" and unknown == 0 and missing == 0 and bool(keys) and alloc[a]["known"]
        if kind == "codex":
            row["cost_state"] = "incomplete (CLI planning unpriced: subscription; known allocated preparation USD reported)"
            row["known_preparation_usd"] = (alloc[a]["api"] + alloc[a]["energy"] + alloc[a]["hardware"]) if alloc[a]["known"] else None
        elif not costed:
            row["cost_state"] = (f"unknown ({missing} presentation(s) without a plan row, {unknown} with an unmeasured energy / time component"
                                 + ("; practice / preparation cost unmeasured or missing" if not alloc[a]["known"] else "") + ")")
        else:
            row["cost_state"] = "costed"
            row["components_usd"] = {k: sum(per[(a,) + q][k] for q in keys) for k in ("api", "energy", "hardware", "alloc_api", "alloc_energy", "alloc_hardware")}
            js = [per[(a,) + q]["joules"] for q in keys]
            row["joules_scored"] = None if kind != "gemma" or any(j is None for j in js) else sum(js)
            for v in COST_VIEWS:
                usd = lambda a_, i, l, v=v: _view_usd(per[(a_, i, l)], v, cfg)        # noqa: E731
                cost_f = fam_vec((fam_of[i], usd(a, i, l)) for (i, l) in keys)
                im_c = item_means(a, keys, usd)
                cm_f = fam_vec((fam_of[i], x) for i, x in im_c.items())
                row["views"][v] = {"total_usd": float(cost_f.sum()), "usd_per_success": boot_ratio(cm_f, sr_f, W),       # same item weights as (b), (c)
                                   "mean_usd_per_attempted_task": boot_ratio(cm_f, n_f, W)}
        secs = [per[(a,) + k]["seconds"] for k in keys]
        known_s = [s for s in secs if s is not None]
        if known_s and len(known_s) == len(secs):
            srt = sorted(known_s)
            im_s = item_means(a, keys, lambda a_, i, l: per[(a_, i, l)]["seconds"])
            row["latency"] = {"mean_s_per_presentation": sum(srt) / len(srt), "median_s": srt[len(srt) // 2],
                              "p90_s": srt[min(len(srt) - 1, int(math.ceil(0.9 * len(srt))) - 1)], "max_s": srt[-1],
                              "total_planning_hours": sum(srt) / 3600.0,
                              "mean_s_per_task_item_weighted": boot_ratio(fam_vec((fam_of[i], x) for i, x in im_s.items()), n_f, W),
                              "includes": "every attempt, repair and failed attempt of the scored cell"}
        else:
            row["latency"] = {"state": "unknown", "presentations_without_time": len(secs) - len(known_s)}
        by_arm[a] = row
        hv = row["views"].get("headline") or {}
        pareto.append({"arm": a, "kind": kind, "sr": (row["sr_item_weighted"] or {}).get("estimate"), "known_preparation_usd": row.get("known_preparation_usd"),
                       "sr_ci95": (row["sr_item_weighted"] or {}).get("ci95"),
                       "mean_usd_per_attempted_task": (hv.get("mean_usd_per_attempted_task") or {}).get("estimate"),
                       "usd_ci95": (hv.get("mean_usd_per_attempted_task") or {}).get("ci95"), "cost_state": row["cost_state"]})
    costed_pts = [p for p in pareto if p["mean_usd_per_attempted_task"] is not None and p["sr"] is not None]
    for p in pareto:
        p["pareto_efficient"] = (None if p not in costed_pts else not any(
            q is not p and q["sr"] >= p["sr"] and q["mean_usd_per_attempted_task"] <= p["mean_usd_per_attempted_task"]
            and (q["sr"] > p["sr"] or q["mean_usd_per_attempted_task"] < p["mean_usd_per_attempted_task"]) for q in costed_pts))
    out.update(by_arm=by_arm, pareto=pareto, n_families=F)

    # (b) R and C on the matched presentations of G+M and O0 (same task weights, same resampling matrix)
    a, b = H1
    rc = {"a": a, "b": b, "available": a in arms and b in arms}
    if rc["available"]:
        keys = sorted(set(pres[a]) & set(pres[b]))
        rc["matched_presentations"] = len(keys)
        rc["cohort_complete"] = missing_vs_cohort[a] == 0 and missing_vs_cohort[b] == 0
        ia, ib = item_means(a, keys, sr) or {}, item_means(b, keys, sr) or {}
        n_f = fam_vec((fam_of[i], 1.0) for i in ia)
        rc["matched_items"] = len(ia)
        rc["R"] = boot_ratio(fam_vec((fam_of[i], v) for i, v in ia.items()), fam_vec((fam_of[i], v) for i, v in ib.items()), W)
        rc["SR_a"], rc["SR_b"] = boot_ratio(fam_vec((fam_of[i], v) for i, v in ia.items()), n_f, W), boot_ratio(
            fam_vec((fam_of[i], v) for i, v in ib.items()), n_f, W)
        rc["C"] = {}
        for v in C_VIEWS:
            usd = lambda a_, i, l, v=v: _view_usd(per[(a_, i, l)], v, cfg) if per[(a_, i, l)]["rows"] else None      # noqa: E731
            ca, cb = item_means(a, keys, usd), item_means(b, keys, usd)
            if ca is None or cb is None:
                rc["C"][v] = {"estimate": None, "state": "unknown", "ci95": None, "bounds_obtainable": False,
                              "reason": "a matched presentation has no plan row or an unmeasured energy / time component"}
            else:
                rc["C"][v] = boot_ratio(fam_vec((fam_of[i], x) for i, x in ca.items()), fam_vec((fam_of[i], x) for i, x in cb.items()), W)
    out["R_C"] = rc

    # the A24.2 reading rule (headline view)
    cl = cfg["claim"]
    h1_ni = bool(((((h1 or {}).get("primary") or {}).get("decision") or {}).get("claims") or {}).get("ni"))
    claim = {"rule": cl["rule"], "text": cl["text"], "h1_established": h1_ni, "statement_permitted": False}
    R, Cv = rc.get("R") or {}, (rc.get("C") or {}).get(cl["view"]) or {}
    if not rc["available"]:
        claim["reason"] = "arms G+M and O0 both needed"
    elif not complete:
        claim["reason"] = "incomplete study: descriptive only (prereg s13)"
    elif not rc.get("cohort_complete"):
        claim["reason"] = f"G+M or O0 does not cover the cohort (missing {missing_vs_cohort[a]} / {missing_vs_cohort[b]} presentations)"
    elif freeze_required and not freeze_recorded:
        claim["reason"] = ("freeze.json has no complete a24_2_cost record (prices, hours and the hardware price basis): the accounting was not shown "
                           "to be fixed before testing")
    elif not (R.get("ci95") and R.get("bounds_obtainable")):
        claim["reason"] = "R: valid CI bounds not obtainable (" + str(R.get("reason") or R.get("state")) + ")"
    elif not (Cv.get("ci95") and Cv.get("bounds_obtainable")):
        claim["reason"] = "C: valid CI bounds not obtainable (" + str(Cv.get("reason") or Cv.get("state")) + ")"
    else:
        r_lo, c_hi = R["ci95"][0], Cv["ci95"][1]
        ok = math.isfinite(r_lo) and math.isfinite(c_hi) and r_lo >= cl["R_lower_min"] and c_hi <= cl["C_upper_max"]
        claim.update(R_lower=r_lo, C_upper=c_hi, statement_permitted=bool(ok),
                     reason=("reading rule met" if ok else f"reading rule not met (R lower {r_lo:.3f} vs >= {cl['R_lower_min']}, "
                                                           f"C upper {c_hi:.4f} vs <= {cl['C_upper_max']})"))
    if h1_ni:
        claim["note"] = "H1 non-inferiority is established; the A24.2 statement is a descriptive secondary statement beside it"
    est = {x: by_arm[x]["energy_estimated"] for x in (a, b) if x in by_arm and (by_arm[x]["energy_estimated"]["scored_rows"]
                                                                                or by_arm[x]["energy_estimated"]["allocated_rows"])}
    if est:                                                               # C5: an estimate inside the headline is always disclosed with it
        claim["energy_estimate_disclosure"] = {
            "arms": {x: {"rows": e["scored_rows"] + e["allocated_rows"], "joules": e["scored_joules"] + e["allocated_joules"],
                         "measured_joules": by_arm[x]["energy_measured"]["scored_joules"] + by_arm[x]["energy_measured"]["allocated_joules"]}
                     for x, e in est.items()},
            "rate_w": rate["rate_w"], "rate_cells": rate["cells"],
            "text": ("the headline cost includes ESTIMATED energy for unmeasured rows (measured Gemma rate x their planning seconds; "
                     "energy_method.unmeasured_rule_terms); the measured energy is the primary figure and is reported beside it")}
    out["claim"] = claim
    return out


def pareto_svg(pareto, title="SR vs mean planning USD per attempted task (A24.2, headline view)") -> str:
    """A small, dependency-free SVG scatter (log10 x axis). Arms without a positive USD value are listed under the plot, not placed on it."""
    pts = [p for p in pareto if p.get("sr") is not None and (p.get("mean_usd_per_attempted_task") or 0) > 0]
    off = [p for p in pareto if p not in pts]
    Wd, Ht, ml, mr, mt, mb = 640, 400, 70, 20, 40, 60
    if pts:
        xs = [math.log10(p["mean_usd_per_attempted_task"]) for p in pts]
        x0, x1 = math.floor(min(xs)), math.ceil(max(xs))
        x1 = x1 if x1 > x0 else x0 + 1
    else:
        x0, x1 = -6, 0
    X = lambda v: ml + (math.log10(v) - x0) / (x1 - x0) * (Wd - ml - mr)          # noqa: E731
    Y = lambda s: mt + (1 - s) * (Ht - mt - mb)                                     # noqa: E731
    e = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")   # noqa: E731
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{Wd}" height="{Ht + 16 * len(off)}" font-family="sans-serif" font-size="12">',
         f'<rect width="100%" height="100%" fill="#ffffff"/><text x="{ml}" y="22" font-size="14" fill="#111">{e(title)}</text>',
         f'<line x1="{ml}" y1="{Ht - mb}" x2="{Wd - mr}" y2="{Ht - mb}" stroke="#444"/><line x1="{ml}" y1="{mt}" x2="{ml}" y2="{Ht - mb}" stroke="#444"/>']
    for d in range(x0, x1 + 1):
        x = X(10.0 ** d)
        o.append(f'<line x1="{x:.1f}" y1="{Ht - mb}" x2="{x:.1f}" y2="{Ht - mb + 5}" stroke="#444"/><text x="{x:.1f}" y="{Ht - mb + 18}" text-anchor="middle" fill="#333">1e{d}</text>')
    for s in (0, 0.25, 0.5, 0.75, 1.0):
        o.append(f'<text x="{ml - 8}" y="{Y(s) + 4:.1f}" text-anchor="end" fill="#333">{int(s * 100)}%</text>')
    o.append(f'<text x="{(ml + Wd - mr) / 2}" y="{Ht - 18}" text-anchor="middle" fill="#111">mean planning USD per attempted task (log scale)</text>')
    o.append(f'<text x="16" y="{(mt + Ht - mb) / 2}" transform="rotate(-90 16 {(mt + Ht - mb) / 2})" text-anchor="middle" fill="#111">success rate</text>')
    for p in pts:
        x, y = X(p["mean_usd_per_attempted_task"]), Y(p["sr"])
        col = {"gemma": "#1b7837", "api": "#2c5aa0", "codex": "#8c510a"}.get(p["kind"], "#555")
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{6 if p.get("pareto_efficient") else 4}" fill="{col}"/>'
                 f'<text x="{x + 8:.1f}" y="{y - 6:.1f}" fill="#111">{e(p["arm"])}</text>')
    for k, p in enumerate(off):
        sr_txt = "n/a" if p.get("sr") is None else f"{100 * p['sr']:.1f}%"
        o.append(f'<text x="{ml}" y="{Ht + 4 + 16 * k}" fill="#555">{e(p["arm"])}: SR {sr_txt}; not placed ({e(p.get("cost_state"))})</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


def check_cost_freeze(freeze: dict | None, cfg=None) -> dict:
    """freeze.json `a24_2_cost` must match the analysis plan's numbers and name the hardware price basis and the energy method.
    -> {"recorded": bool, ...}; raises AnalysisError on a numeric mismatch."""
    cfg = A24_2_COST if cfg is None else cfg
    fz = (freeze or {}).get("a24_2_cost")
    if not fz:
        return {"recorded": False, "note": "freeze.json has no a24_2_cost entry: estimates use the analysis plan's numbers; the reading rule is blocked "
                                           "until the record (with the hardware price basis) is in freeze.json"}
    bad = [k for k in ("usd_per_kwh", "hardware_price_usd", "operating_hours", "hardware_time_factor")
           if k in fz and float(fz[k]) != float(cfg[k])]
    if bad:
        raise AnalysisError(f"freeze.json a24_2_cost disagrees with the analysis plan on {bad}; refusing (no silent override)")
    fem, cem = fz.get("energy_method"), cfg.get("energy_method")          # C5: the frozen energy method + unmeasured rule = the analysis plan's
    if isinstance(fem, dict) and isinstance(cem, dict) and any(fem.get(k) != cem.get(k) for k in ("id", "unmeasured_rule_basis", "unmeasured_rule_terms")):
        raise AnalysisError("freeze.json a24_2_cost energy_method (id / unmeasured_rule_basis / terms) disagrees with the analysis plan; "
                            "refusing (no silent override)")
    basis = str(fz.get("hardware_price_basis") or "")
    gaps = [k for k in ("usd_per_kwh", "hardware_price_usd", "operating_hours", "energy_method") if fz.get(k) in (None, "")]   # a null field is missing
    ub = str(cem.get("unmeasured_rule_basis") or "") if isinstance(cem, dict) else ""
    if ub.startswith(("amendment:", "arash-decision:")) and not (
            isinstance(fz.get("unmeasured_rule"), dict) and fz["unmeasured_rule"].get("terms") == cem.get("unmeasured_rule_terms")
            and (ub.startswith("arash-decision:") or fz["unmeasured_rule"].get("amendment_section_sha256"))):
        gaps.append("unmeasured_rule (the approved C5 rule with its terms; freeze_ft2 unmeasured_rule_check)")
    if not (basis == "laptop_purchase_price" or basis.startswith("amendment:")):
        gaps.append("hardware_price_basis (laptop_purchase_price or amendment:<id>; Astra P-cost Q4)")
    if gaps:
        return {"recorded": False, "missing": gaps, "note": "a24_2_cost in freeze.json is incomplete: the reading rule is blocked"}
    return {"recorded": True, "hardware_price_basis": basis}


# ================================================================================================ the analysis
def analyse(score_rows, *, fam_of_manifest=None, plan_rows=None, practice_rows=None, B=9999, meta=None, exec_rows=None, prep_rows=None, cost_cfg=None,
            B_cost=20000):
    idx = build_index(score_rows, fam_of_manifest)
    S = idx["S"]
    if not S:
        raise AnalysisError("no scored cells (canon world, r=0, run=0)")
    arms = sorted({a for (a, i, l) in S})
    drop_invalid = invalid_presentations(S)
    complete = bool((meta or {}).get("complete", True))
    core_missing = [a for a in CORE_NEEDED if a not in arms]
    items_per_arm = {a: len({i for (x, i, l) in S if x == a}) for a in arms}
    res = {"schema": "ftm.analysis/1", "spec": C.SPEC_VERSION, "meta": meta or {}, "arms": arms, "items_per_arm": items_per_arm,
           "n_families": len(set(idx["fam"].values())), "invalid_presentations_dropped": len(drop_invalid),
           "core_arms_missing": core_missing, "unparsable_rows": len(idx["bad"]), "complete": complete,
           "analysis_plan_sha": analysis_plan_sha(), "analysis_plan": ANALYSIS_PLAN}
    if "G+M" in arms and "O0" in arms:
        res["H1"] = h1_block(S, idx["fam"], drop_invalid, B, complete)
    else:
        res["H1"] = {"error": "arms G+M and O0 both needed"}
    res["equivalence_exploratory"] = (equivalence_block(S, idx["fam"], drop_invalid, B, complete) if ("G+M" in arms and "O+M" in arms)
                                      else {"available": False, "a": "G+M", "b": "O+M"})
    res["holm_family"] = holm_family(S, idx["fam"], drop_invalid, B, complete)
    res["holm_family_note"] = ("3 tests (H2a, H2c, H3a; A24.5), Holm-Bonferroni at alpha .05, separate from H1; an unavailable test counts as p=1 and keeps m=3. "
                               "H4a-b and H5a-c are exploratory (raw p, `exploratory_contrasts`). No study-wide .05 claim.")
    res["exploratory_contrasts"], res["holm_m8_for_transparency"] = exploratory_contrasts(S, idx["fam"], drop_invalid, B, res["holm_family"])
    res["withdrawn_hypotheses"] = {k: dict(v, status="withdrawn (A23): not computed") for k, v in WITHDRAWN.items()}
    res["infra_paired_sensitivity_holm"] = {
        h: _infra_sens(S, idx["fam"], a, b, drop_invalid) for h, (a, b, _, _) in ALL_CONTRASTS.items()}
    res["distinct_units_sensitivity"] = {h: _sens(S, idx["fam"], a, b, drop_invalid, drop_shared=True) for h, (a, b, _, _) in ALL_CONTRASTS.items()}
    res["distinct_units_sensitivity"]["H1"] = _sens(S, idx["fam"], H1[0], H1[1], drop_invalid, drop_shared=True)
    res["x_isolation_paired_sensitivity"] = {h: _sens(S, idx["fam"], a, b, drop_invalid, drop_flag=_x_flagged)
                                              for h, (a, b, _, _) in ALL_CONTRASTS.items() if h.startswith("H5")}
    res["arm_tables"] = per_arm_tables(idx, [a for a in arms if not a.startswith("X-")], drop_invalid)
    res["infra_by_arm"] = infra_table(S, arms)
    # A36 (signed): the amended common-subset analysis BESIDE the registered one (which is unchanged above); only when the score rows carry the flag
    if any("excluded_reset" in r for r in S.values()):
        res["a36"] = A36.a36_block(S, idx["fam"], drop_invalid, B, complete, lvl_of=idx["level"], registered_h1=res.get("H1"),
                                   registered_holm=res["holm_family"], registered_exploratory=res["exploratory_contrasts"],
                                   registered_equivalence=res["equivalence_exploratory"])
    res["dependence"] = {"plan": ANALYSIS_PLAN["dependence"], "by_arm": dependence_block(S, arms),
                         "contrasts": {h: _dep_row(S, idx["fam"], a, b, drop_invalid) for h, (a, b, _, _) in ALL_CONTRASTS.items()}}
    res["dependence"]["contrasts"]["H1"] = _dep_row(S, idx["fam"], H1[0], H1[1], drop_invalid)
    res["execution_time"] = execution_block(score_rows, exec_rows)
    lp = execution_link_problems(score_rows, exec_rows) if exec_rows else None
    res["execution_links"] = {"checked": None if lp is None else len(score_rows), "problems": None if lp is None else len(lp), "first": (lp or [])[:10]}
    if lp and complete:
        raise AnalysisError(f"{len(lp)} score row(s) are not linked to their execution (first: {lp[0]}); copied outcomes could not be told from independent runs")
    res["variance_decomposition"] = variance_decomposition(score_rows, [a for a in arms if not a.startswith("X-")])
    res["cost_per_success"] = cost_block([a for a in arms if not a.startswith("X-")], S, plan_rows, practice_rows)
    res["codex"] = codex_block(idx, drop_invalid, B)
    # A24.2 item 1 (stage P-cost): planning cost-effectiveness, every arm (X arms shown, not costed); practice + prep rows are the allocated costs
    prep = ([dict(r, cost_kind="gemma", cost_source="practice") for r in (practice_rows or [])]   # practice is Gemma-planned (prereg 6.2); never the row's label (Astra C5 r6 #1)
            + [dict(r, cost_source="preparation") for r in (prep_rows or [])])        # C5: the source is set HERE, never read from the row
    if cost_cfg is None and (meta or {}).get("synthetic"):              # C5-SIMPLE: synthetic data never holds the 10 real practice-v2 rows the
        em_ = dict(A24_2_COST["energy_method"], unmeasured_rule_basis=None, unmeasured_rule_terms=None)   # practice-row rule binds: rule off
        cost_cfg = dict(A24_2_COST, energy_method=em_)
    res["cost_effectiveness"] = cost_effectiveness_block(S, idx["fam"], arms, drop_invalid, plan_rows, prep,
                                                         cfg=cost_cfg, h1=res.get("H1"), complete=complete, B=B_cost,
                                                         freeze_required=not (meta or {}).get("synthetic"),
                                                         freeze_recorded=((meta or {}).get("a24_2_cost_freeze") or {}).get("recorded") is True)
    res["cost_effectiveness"]["energy_attach"] = (meta or {}).get("energy_attach")
    return res


def _x_flagged(r) -> bool:
    return bool(r.get("x_tool_events") or r.get("x_identity_problem"))


def _dep_row(S, fam, a, b, drop_invalid):
    c = contrast(S, fam, a, b, drop_invalid, 0)
    if c is None:
        return None
    return {"a": a, "b": b, "presentations": c["n_pres"], "shared_unit_presentations": c["n_shared"], "discordant": c["n_discordant"],
            "discordance": c["discordance"], "distinct_unit_presentations": c["n_pres"] - c["n_shared"],
            "note": "a presentation in which both arms point at the same unit has difference 0 by construction (one realisation, not agreement of two runs)"}


def _infra_sens(S, fam, a, b, drop_invalid):
    c = contrast(S, fam, a, b, drop_invalid, 0, drop_infra=True)
    return None if c is None else clean(c)


def _sens(S, fam, a, b, drop_invalid, **kw):
    c = contrast(S, fam, a, b, drop_invalid, 0, **kw)
    return None if c is None else clean(c)


# ================================================================================================ markdown
def pct(x):
    return "n/a" if x is None else f"{100 * x:+.1f} pp"


def rate(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render_md(res) -> str:
    L = []
    syn = res["meta"].get("synthetic")
    L.append("# Final-test analysis" + (" (SYNTHETIC -- generated outcomes, not evidence)" if syn else ""))
    L.append("")
    m = res["meta"]
    L.append(f"Namespace: `{m.get('namespace')}`. Score rows: {m.get('n_score_rows')}. Families: {res['n_families']}. "
             f"INVALID presentations dropped from every arm: {res['invalid_presentations_dropped']}. "
             f"Seal: {m.get('seal_sha256') or 'n/a'}. Complete: {m.get('complete')}.")
    if res["core_arms_missing"]:
        L.append(f"**Core arms missing:** {', '.join(res['core_arms_missing'])}.")
    if not res.get("complete", True):
        L.append(f"**{INCOMPLETE_WORDING}.** Estimates and intervals below cover the presentations that were completed; no test is "
                 f"called and nothing is rejected.")
    h1 = res["H1"]
    L += ["", "## H1 (primary): SR(G+M) - SR(O0)", ""]
    if "error" in h1:
        L.append(h1["error"])
    else:
        L.append(f"**Wording: {h1.get('wording')}**")
        pr = h1.get("procedure") or {}
        L.append(f"Procedure: {pr.get('version')} ({pr.get('candidate')}: CR2 + Satterthwaite, guard K >= 20 families with a discordant presentation, "
                 f"one-sided level {pr.get('alpha_star')} per tail = a nominal 96.5 % interval; NI iff the lower bound > -5 pp, superiority only after NI). "
                 + ("Validated in V100 (A24.1)." if pr.get("validated") else "NOT validated: estimation only."))
        L += ["", "| view | estimate | nominal 96.5% interval (C7) | K (guard >= 20) | wording |", "|---|---|---|---|---|"]
        for k, nm in (("primary", "primary (item-weighted, validated)"), ("family_weighted", "family-weighted (sensitivity)"),
                      ("infra_paired", "infra-paired (drop either-arm infra; sensitivity)")):
            v = h1.get(k)
            if not v:
                L.append(f"| {nm} | n/a | | | |")
                continue
            d = v["decision"]
            iv = d.get("interval") or [None, None]
            ivs = "undefined" if iv[0] is None else f"[{pct(iv[0])}, {pct(iv[1])}]"
            g = d.get("guard") or {}
            L.append(f"| {nm} | {pct(d['estimate'])} | {ivs} | {g.get('K')} ({'passed' if g.get('passed') else 'failed'}) | {d['wording']} |")
        p = h1.get("primary") or {}
        L.append(f"\nFamilies where G+M beats O0 / tie / loses: {p.get('families_a_better')} / {p.get('tie')} / {p.get('families_b_better')}. "
                 f"Items: {p.get('n_items')}; presentations: {p.get('n_pres')}.")
    L += ["", "## Multiplicity (frozen, A21 F)", "",
          "H1 stands alone (one-sided .025 non-inferiority, then superiority in a fixed sequence, h1-rule-1 of A24.1); it is NOT a member of the Holm family. The secondary family has m = 3 "
          "tests at alpha .05 (H2a, H2c, H3a; Holm-Bonferroni; A24.5 moved H4a-b and H5a-c to exploratory and A23 withdrew H2b, H3b, H3c); the smaller m relaxes the Holm "
          "thresholds, and the m = 8 adjustment is printed beside the verdict for transparency. There is no study-wide .05 claim. A member without validity evidence is "
          "estimation-only. Analysis plan sha " + str(res.get("analysis_plan_sha", ""))[:16] + "."]
    h1p = (res.get("H1") or {}).get("primary") if isinstance(res.get("H1"), dict) else None
    if h1p and h1p.get("discordance") is not None:
        L += ["", f"H1 observed discordance (share of presentations in which G+M and O0 differ): {h1p['discordance']:.2f}; families with a discordant "
              f"presentation K = {h1p.get('discordant_families')}. Below K = 20 h1-rule-1 makes no claim (the interval is not trustworthy there: its coverage "
              f"fell to .82 in the V100 low-discordance cells)."]
    eq = res.get("equivalence_exploratory") or {}
    if eq.get("available"):
        e = eq["decision"]
        iv = e.get("interval90") or [None, None]
        L += ["", "## Exploratory equivalence: SR(G+M) - SR(O+M) within +/-5 pp (E1, never confirmatory)", "",
              f"Estimate {pct(e['estimate'])}, CR2 90 % interval " + ("undefined" if iv[0] is None else f"[{pct(iv[0])}, {pct(iv[1])}]")
              + f", K = {(e.get('guard') or {}).get('K')}: **{e['wording']}**. 'Not significantly different' is never 'equivalent'; simulated power of E1 at a true "
              "difference of 0 was at most 2 % with this sample."]
    L += ["", "## Holm family (alpha .05, m=3)", "",
          "| id | contrast | estimate | 95% CI | p | p Holm | reject | discordance | shared-unit pres | missing pres |", "|---|---|---|---|---|---|---|---|---|---|"]
    for hid, v in res["holm_family"].items():
        if not v.get("available"):
            L.append(f"| {hid} | {v['a']} - {v['b']} | not available | | | 1.0000 | no | | | |")
            continue
        rej = "n/a (incomplete)" if v["reject"] is None else ("YES" if v["reject"] else "no")
        dsc = "n/a" if v.get("discordance") is None else f"{v['discordance']:.2f}" + (" (NI guard: p := 1)" if (v.get("ni_guard") or {}).get("applied") else "")
        L.append(f"| {hid} | {v['a']} - {v['b']} | {pct(v['estimate'])} | [{pct(v['ci95'][0])}, {pct(v['ci95'][1])}] | {v['p']:.4f} | {v['p_holm']:.4f} | {rej} | {dsc} | {v.get('n_shared', 0)} | {v.get('missing_pres', 0)} |")
    m8 = res.get("holm_m8_for_transparency") or {}
    if m8:
        L += ["", "For transparency (A24.5): the Holm-adjusted p of the same three tests if the family had been the A23 m = 8 (adding H4a-b, H5a-c): "
              + "; ".join(f"{k} {v:.4f}" for k, v in m8.items()) + ". Every m = 3 adjusted p is smaller than or equal to its m = 8 value."]
    ex = res.get("exploratory_contrasts") or {}
    if ex:
        L += ["", "## Exploratory contrasts (A24.5: raw p, NOT adjusted, no confirmatory conclusion)", "",
              "| id | contrast | estimate | 95% CI | raw p | p Holm if m = 8 | discordance |", "|---|---|---|---|---|---|---|"]
        for hid, v in ex.items():
            if not v.get("available"):
                L.append(f"| {hid} | {v['a']} - {v['b']} | not available | | | | |")
                continue
            dsc = "n/a" if v.get("discordance") is None else f"{v['discordance']:.2f}"
            L.append(f"| {hid} | {v['a']} - {v['b']} | {pct(v['estimate'])} | [{pct(v['ci95'][0])}, {pct(v['ci95'][1])}] | {v['p']:.4f} | {v['p_holm_m8']:.4f} | {dsc} |")
    wd = res.get("withdrawn_hypotheses") or {}
    if wd:
        L += ["", "Withdrawn before testing (A23, Sonnet archived; IDs kept, nothing computed): " + "; ".join(f"{k} {v['contrast']}" for k, v in wd.items()) + "."]
    L += ["", "Sensitivities (estimates): planner OR execution infrastructure failure of either arm dropped; presentations where the two arms share one execution dropped; "
          "H5 only, presentations with an X tool event or identity mismatch dropped:", "",
          "| id | infra-paired | items | dropped pres | distinct units only | items | X-isolation-paired |", "|---|---|---|---|---|---|---|"]
    for hid, v in res["infra_paired_sensitivity_holm"].items():
        d = (res.get("distinct_units_sensitivity") or {}).get(hid)
        x = (res.get("x_isolation_paired_sensitivity") or {}).get(hid)
        L.append(f"| {hid} | {pct(v['estimate']) if v else 'n/a'} | {v['n_items'] if v else ''} | {v['dropped_infra'] if v else ''} | "
                 f"{pct(d['estimate']) if d else 'n/a'} | {d['n_items'] if d else ''} | {pct(x['estimate']) if x else ''} |")
    dep = res.get("dependence") or {}
    if dep.get("by_arm"):
        L += ["", "## Shared executions (dependence, A21 D)", "", dep["plan"]["shared_executions"], "",
              "| arm | scored cells | distinct units | cells sharing a unit with another arm |", "|---|---|---|---|"]
        for arm, v in dep["by_arm"].items():
            L.append(f"| {arm} | {v['cells']} | {v['distinct_units']} | {v['cells_sharing_a_unit_with_another_arm']} ({rate(v['shared_share'])}) |")
        L += ["", "| contrast | presentations | with one shared unit | with distinct units | discordant |", "|---|---|---|---|---|"]
        for hid, v in dep["contrasts"].items():
            if v:
                L.append(f"| {hid} {v['a']} - {v['b']} | {v['presentations']} | {v['shared_unit_presentations']} | {v['distinct_unit_presentations']} | {v['discordant']} |")
    et = res.get("execution_time") or {}
    if et.get("available"):
        L += ["", f"Simulator time: {et['unique_runs']} unique runs, {et['total_wall_s']} s in total; per arm (shared runs split equally): "
              + ", ".join(f"{a} {v}" for a, v in et["per_arm_wall_s"].items()) + "."]
    L += ["", "## SR per arm, level and language (descriptive, family-cluster CI)", "",
          "| arm | overall | L1 | L2 | L3 | EN | FA | L2 PC | L3 TDS | strict SR | realistic SR | slice flags T/?/n |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, t in res["arm_tables"].items():
        def cell(x):
            if not x:
                return "n/a"
            ci = f" [{rate(x['ci95'][0])}, {rate(x['ci95'][1])}]" if x.get("ci95") else ""
            return rate(x["sr"]) + ci
        dg = t["diagnostics"]
        f = lambda v: "n/a" if not v else f"{v['mean']:.3f}"      # noqa: E731
        L.append(f"| {arm} | {cell(t['overall'])} | {cell(t['level'].get('L1'))} | {cell(t['level'].get('L2'))} | {cell(t['level'].get('L3'))} "
                 f"| {cell(t['lang'].get('en'))} | {cell(t['lang'].get('fa'))} | {f(dg['L2_pc_final'])} | {f(dg['L3_tds_final'])} | {f(dg['strict_sr'])} "
                 f"| {cell(dg.get('prerequisite_realistic_sr'))} | {dg['slice_flags']['unrealistic_true']}/{dg['slice_flags']['unknown_none']}/{dg['slice_flags']['slice_rows']} |")
    L += ["", "## Infrastructure failures per arm", "",
          "| arm | cells | planner infra | rate | execution infra | rate | total rate | flag | invalid |", "|---|---|---|---|---|---|---|---|---|"]
    for arm, v in res["infra_by_arm"].items():
        L.append(f"| {arm} | {v['n']} | {v['infra']} | {rate(v['infra_rate'])} | {v['exec_infra']} | {rate(v['exec_infra_rate'])} | {rate(v['infra_total_rate'])} | "
                 f"{'infrastructure-sensitive' if v['infrastructure_sensitive'] else ''} | {v['invalid']} |")
    L += ["", "## REPEAT variance decomposition", "",
          "| arm | pres | plan identity | SR spread (planner) | planner var | outcome agreement (exec) | exec var |", "|---|---|---|---|---|---|---|"]
    for arm, v in res["variance_decomposition"].items():
        pl, ex = v["planner"], v["execution"]
        g = lambda x: "n/a" if x is None else f"{x:.3f}"        # noqa: E731
        L.append(f"| {arm} | {v['n_presentations']} | {g(pl['plan_identity_rate'])} | {g(pl['sr_spread_mean'])} | {g(pl['sr_variance_mean'])} | {g(ex['outcome_agreement_rate'])} | {g(ex['sr_variance_mean'])} |")
    L += ["", "## API spend per success (plan journal cost_usd + shared practice cost; Gemma energy and hardware are in the A24.2 section below)", "", "| arm | successes | plan USD | practice USD added | USD / success | tokens in/out |", "|---|---|---|---|---|---|"]
    for arm, v in res["cost_per_success"].items():
        ps = {"infinite": "infinite (cost, 0 successes)", "undefined": "undefined (0 / 0)"}.get(v.get("usd_per_success_state"), "n/a")             if v["usd_per_success"] is None else f"{v['usd_per_success']:.4f}"
        L.append(f"| {arm} | {v['successes']:.1f} | {v['plan_usd']:.4f} | {v['practice_usd_added']:.4f} | {ps} | {v['tokens_in']}/{v['tokens_out']} |")
    L += render_cost_md(res.get("cost_effectiveness"))
    L += A36.render_md(res.get("a36"))
    if any(a in res.get("arms", ()) for a in C.X_MAIN_ARMS):
        L += ["", "## How to read the X0 / X+M results (A21 E)", "", X_DISCLOSURE]
    cx = res["codex"]
    L += ["", "## Codex CLI systems (exploratory, separate)", "", cx["note"], ""]
    if cx["arms"]:
        L += ["| arm | SR | items |", "|---|---|---|"]
        for a, v in cx["arms"].items():
            L.append(f"| {a} | {rate(v['sr']) if v else 'n/a'} | {v['n_items'] if v else ''} |")
        for a, v in cx["memory_gain"].items():
            L.append(f"\n{a} memory gain: {pct(v['estimate'])} [{pct(v['ci95'][0])}, {pct(v['ci95'][1])}]")
    else:
        L.append("No Codex arms in this journal.")
    return "\n".join(L) + "\n"


def _usd(x):
    return "n/a" if x is None else ("infinite" if x == math.inf else f"{x:.3g}")


def _ci_txt(bx, fmt=_usd, inf_label="infinite (cost, 0 successes)"):
    if not bx:
        return "n/a"
    if bx.get("state") in ("infinite", "undefined", "unknown") and bx.get("estimate") in (None, math.inf):
        est = {"infinite": inf_label, "undefined": "undefined (0 / 0)"}.get(bx["state"], "unknown")
    else:
        est = fmt(bx.get("estimate"))
    if bx.get("ci95"):
        return f"{est} [{fmt(bx['ci95'][0])}, {fmt(bx['ci95'][1])}]" + (f" ({bx['n_infinite']} inf. replicates)" if bx.get("n_infinite") else "")
    return est + (f" (CI not obtainable: {bx.get('reason')})" if bx.get("reason") else "")


def render_cost_md(ce) -> list:
    """Markdown lines for the A24.2 block."""
    if not ce:
        return []
    L = ["", "## Cost-effectiveness (A24.2 key secondary; planning cost only, descriptive, 95 % family-cluster bootstrap CIs)", ""]
    a = ce.get("assumptions") or {}
    L.append(f"Assumptions ({a.get('version')}): electricity {a.get('usd_per_kwh')} USD/kWh on GPU joules (simple, approximate; CPU not measured); hardware {a.get('hardware_price_usd')} USD / "
             f"{a.get('operating_hours')} h x planning hours (basis: {a.get('hardware_price_basis')}; {a.get('hardware_price_source', '')[:60]}...). "
             f"Codex (X) CLI planning is not priced (subscription); their allocated preparation USD is shown. {a.get('scope')}.")
    if str(a.get("hardware_price_basis", "")).endswith("pending"):
        L.append(f"**Hardware price basis pending:** {a.get('hardware_price_status')}")
    if a.get("hardware_price_disclosure"):
        L.append(f"**Hardware price disclosure (A33):** {a.get('hardware_price_disclosure')}")
    em = a.get("energy_method") if isinstance(a.get("energy_method"), dict) else {}
    if em:
        L.append(f"**Energy (C5, {em.get('id')}):** {em.get('disclosure')} Unmeasured Gemma rows: "
                 f"{('ESTIMATED at the measured Gemma rate (flagged) for ' + ', '.join((em.get('unmeasured_rule_terms') or {}).get('eligible') or []) + ' rows, claim ' + str((em.get('unmeasured_rule_terms') or {}).get('claim')) + ', under ' + str(em.get('unmeasured_rule_basis'))) if approved_terms(a) else 'cost UNKNOWN (no approved rule)'}.")
    er = ce.get("energy_rate")
    if er:
        L.append(f"Measured Gemma rate (C5 unmeasured_estimate): {('n/a' if er.get('rate_w') is None else format(er['rate_w'], '.2f') + ' W')} over "
                 f"{er.get('cells')} measured rows / {er.get('seconds', 0):.2f} s (min {er.get('min_cells')}); active terms: {er.get('active_terms')}.")
    ea = ce.get("energy_attach")
    if ea:
        fl = "; ".join(f"GPU {f.get('gpu_w')} W" for f in ea.get("idle_floor") or []) or "n/a"
        L.append(f"Energy join (strict={ea.get('strict')}): {ea.get('gemma_plan_rows')} Gemma plan rows, {ea.get('measured')} measured, "
                 f"{ea.get('refused_interval')} without a reading, {ea.get('no_energy_row')} without an energy row, {ea.get('ambiguous')} ambiguous, "
                 f"{ea.get('invalid_energy_row')} invalid; inline energy fields stripped from {ea.get('inline_fields_stripped')} rows; idle floor {fl}.")
    if not ce.get("available"):
        return L + ["", f"**Not available:** {ce.get('reason')}"]
    L += ["", f"Cohort: {ce.get('cohort_presentations')} presentations (same item weights in every outcome).", "",
          "| arm | kind | cost state | SR (item-weighted) | USD / success (headline) | mean USD / attempted task (headline) | electricity only / task | hardware only / task | 3x hardware time / task | Pareto |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    pe = {p["arm"]: p.get("pareto_efficient") for p in ce.get("pareto") or []}
    pc = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"          # noqa: E731
    for arm, r in (ce.get("by_arm") or {}).items():
        v = r.get("views") or {}
        g = lambda name, key: _ci_txt((v.get(name) or {}).get(key)) if v else "n/a"     # noqa: E731
        L.append(f"| {arm} | {r['kind']} | {r['cost_state']} | {_ci_txt(r.get('sr_item_weighted'), pc)} | {g('headline', 'usd_per_success')} | "
                 f"{g('headline', 'mean_usd_per_attempted_task')} | {g('electricity_only', 'mean_usd_per_attempted_task')} | "
                 f"{g('hardware_only', 'mean_usd_per_attempted_task')} | {g('hw_time_3x', 'mean_usd_per_attempted_task')} | "
                 f"{'' if pe.get(arm) is None else ('efficient' if pe[arm] else 'dominated')} |")
    for arm, r in (ce.get("by_arm") or {}).items():
        if r.get("kind") == "codex":
            L.append(f"\n{arm}: known allocated preparation USD {_usd(r.get('known_preparation_usd'))}; CLI planning unpriced, so its total is incomplete (never zero).")
        if r.get("missing_vs_cohort"):
            L.append(f"\n{arm}: {r['missing_vs_cohort']} cohort presentation(s) missing (not dropped silently).")
        um, es, ms = r.get("energy_unmeasured") or {}, r.get("energy_estimated") or {}, r.get("energy_measured") or {}
        if um.get("scored_rows") or um.get("allocated_rows") or es.get("scored_rows") or es.get("allocated_rows"):
            L.append(f"\n{arm}: measured Gemma energy (primary) {(ms.get('scored_joules') or 0) + (ms.get('allocated_joules') or 0):.1f} J; "
                     f"unmeasured Gemma energy in {um.get('scored_rows')} scored + {um.get('allocated_rows')} allocated rows "
                     f"({(um.get('scored_seconds') or 0) + (um.get('allocated_seconds') or 0):.2f} s; at the measured rate: "
                     f"{_usd(um.get('usd_at_measured_rate'))}); "
                     f"ESTIMATED (flagged, in the cost): {es.get('scored_rows')} scored + {es.get('allocated_rows')} allocated rows, "
                     f"{(es.get('scored_joules') or 0) + (es.get('allocated_joules') or 0):.1f} J.")
    rc = ce.get("R_C") or {}
    if rc.get("available"):
        L += ["", f"Matched G+M / O0 presentations: {rc.get('matched_presentations')} ({rc.get('matched_items')} items). "
                  f"R = SR(G+M) / SR(O0) = {_ci_txt(rc.get('R'), lambda x: 'n/a' if x is None else f'{x:.3f}', 'infinite (O0 SR 0)')}."]
        for v_ in C_VIEWS:
            L.append(f"- C ({v_}) = mean USD per attempted task G+M / O0 = "
                     f"{_ci_txt((rc.get('C') or {}).get(v_), lambda x: 'n/a' if x is None else f'{x:.4f}', 'infinite (O0 cost 0)')}")
    cl = ce.get("claim") or {}
    L += ["", f"**A24.2 reading rule (lower R >= 0.80 and upper C <= 0.10, headline view): "
              f"{'MET -- the thesis may state: ' + chr(34) + cl.get('text', '') + chr(34) if cl.get('statement_permitted') else 'not met'}** ({cl.get('reason')})."
              + (f" {cl['note']}." if cl.get("note") else "")]
    L += ["", "| arm | planning s / presentation (mean / median / p90) | planning hours | joules (Gemma, scored) |", "|---|---|---|---|"]
    for arm, r in (ce.get("by_arm") or {}).items():
        lt = r.get("latency") or {}
        if "mean_s_per_presentation" in lt:
            L.append(f"| {arm} | {lt['mean_s_per_presentation']:.2f} / {lt['median_s']:.2f} / {lt['p90_s']:.2f} | {lt['total_planning_hours']:.3f} | "
                     f"{'' if r.get('joules_scored') is None else round(r['joules_scored'], 1)} |")
        else:
            L.append(f"| {arm} | unknown | | |")
    L += ["", "Latency includes every attempt, repair and failed attempt of the scored cell. Pareto plot: pareto.svg beside analysis.json."]
    return L


# ================================================================================================ synthetic data
SYN_TRUE = {"G0": 0.40, "G+M": 0.58, "O0": 0.60, "O+M": 0.70, "G+FS": 0.50, "O+FS": 0.64, "X0": 0.57, "X+M": 0.66}


def synthetic_rows(seed=20261010, n_fam=42, truth=None, infra_rate=0.03):
    """Generated score rows with KNOWN item-level success probabilities: p(arm, item) = clip(base + family effect + item effect).
    Returns (rows, truth dict). Arms share the item effect (paired), the EN twin copies the FA outcome with prob .7."""
    truth = dict(SYN_TRUE if truth is None else truth)
    rg = random.Random(seed)
    rows, k = [], 0
    levels = ["L1"] * 12 + ["L2"] * 24 + ["L3"] * 6
    for fi in range(n_fam):
        fam = f"syn-fam-{fi:02d}"
        lvl = levels[fi % len(levels)]
        fe = rg.gauss(0, 0.08)
        for ii in range(rg.choice((2, 3, 3, 4))):
            item = f"syn-{fi:02d}-{ii}"
            ie = rg.gauss(0, 0.10)
            u = rg.random()
            for arm, base in truth.items():
                p = min(0.97, max(0.03, base + fe + ie))
                en = 1.0 if u < p else 0.0
                for lang in LANGS:
                    uu = u if (lang == "en" or rg.random() < 0.7) else rg.random()
                    infra = (arm[0] in "SO") and rg.random() < infra_rate
                    sr = 0.0 if infra else (1.0 if uu < p else 0.0)
                    k += 1
                    rows.append({"key": f"syn|{k}", "namespace": "rehearsal-v1", "arm": arm, "pres": C.pres_id(item, lang), "r": 0, "run": 0,
                                 "unit": f"u|{arm}|{item}|{lang}", "sr": sr, "strict_sr": sr, "valid": True, "infra": infra,
                                 "level": lvl, "lang": lang, "family": fam, "pc_final": (0.8 if sr else 0.4) if lvl == "L2" else None,
                                 "tds_final": (0.7 if sr else 0.3) if lvl == "L3" else None, "unrealistic_slice": 0.0, "_en": en})
    # REPEAT: first 10 items, r in 0..2, runs 0..2 (execution deterministic except a little noise)
    rep = sorted({C.parse_pres(r["pres"])["item"] for r in rows})[:10]
    extra = []
    for r0 in rows:
        p = C.parse_pres(r0["pres"])
        if p["item"] not in rep or r0["arm"] not in ("G0", "G+M", "O0", "O+M"):
            continue
        for rr in (0, 1, 2):
            for run in (0, 1, 2):
                if rr == 0 and run == 0:
                    continue
                flip = rg.random() < 0.10
                same_plan = rg.random() < 0.7 or rr == 0
                sr = r0["sr"] if (same_plan and not (run and flip)) else (1.0 - r0["sr"] if rg.random() < 0.3 else r0["sr"])
                x = dict(r0)
                x.update({"key": f"{r0['key']}|r{rr}|run{run}", "r": rr, "run": run, "sr": sr, "strict_sr": sr,
                          "unit": r0["unit"] if same_plan else f"{r0['unit']}|alt{rr}"})
                extra.append(x)
    rows += extra
    for r in rows:
        r.pop("_en", None)
    return rows, truth


def synthetic_plan_rows(rows, seed=1, energy=True):
    """Generated plan rows for the scored cells. cost_usd as before (paid arms only; the random stream is unchanged). A24.2 fields come from a SEPARATE
    stream: wall_s for every arm and, for Gemma arms when `energy`, measured-looking GPU / CPU joules (about 45-75 W GPU and 10-20 W CPU over the wall time)."""
    rg, rg2 = random.Random(seed), random.Random(seed * 7919 + 13)
    out = []
    for r in rows:
        if r["r"] != 0 or r["run"] != 0:
            continue
        paid = r["arm"][0] in "SO"
        row = {"key": "p|" + r["key"], "arm": r["arm"], "pres": r["pres"], "r": 0, "block": "main",
               "cost_usd": round(rg.uniform(0.005, 0.015), 5) if paid else 0.0, "tokens_in": 3000, "tokens_out": 400}
        kind = arm_kind(r["arm"])
        wall = rg2.uniform(0.4, 2.5) if kind == "gemma" else rg2.uniform(4.0, 14.0) if kind == "api" else rg2.uniform(20.0, 60.0)
        row["wall_s"] = round(wall, 3)
        if kind == "gemma" and energy:
            row["energy_j_gpu"] = round(wall * rg2.uniform(45.0, 75.0), 2)
            row["energy_j_cpu"] = round(wall * rg2.uniform(10.0, 20.0), 2)
        out.append(row)
    return out


# ================================================================================================ main
def write_outputs(res, outdir):
    os.makedirs(outdir, exist_ok=True)
    jp, mp = os.path.join(outdir, "analysis.json"), os.path.join(outdir, "analysis.md")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, sort_keys=True, ensure_ascii=False)
    with open(mp, "w", encoding="utf-8") as f:
        f.write(render_md(res))
    pa = (res.get("cost_effectiveness") or {}).get("pareto")
    if pa:
        with open(os.path.join(outdir, "pareto.svg"), "w", encoding="utf-8") as f:
            f.write(pareto_svg(pa, "SR vs mean planning USD per attempted task (A24.2, headline view)"
                               + (" -- SYNTHETIC" if (res.get("meta") or {}).get("synthetic") else "")))
    return jp, mp


def completeness_problems(rows, manifest, ns, invalid=frozenset()) -> list:
    """Complete seal => every core arm and every FS arm has a canonical r=0 run=0 row for each registered item x language, minus
    explicitly flagged INVALID presentations. `item_diffs` skips a presentation missing in one arm, so without this check a missing
    cell would quietly change the estimand away from 'the mean over the capped test items'."""
    have = {(r["arm"], C.parse_pres(r["pres"])["item"], C.parse_pres(r["pres"])["lang"]) for r in rows
            if C.parse_pres(r["pres"])["world"] == "canon" and r.get("r") == 0 and r.get("run") == 0}
    items = list(C.namespace_items(ns, manifest))
    out = []
    for arm in tuple(C.CORE_ARMS) + ("G+FS", "O+FS") + tuple(C.X_MAIN_ARMS):
        miss = [(i, l) for i in items for l in LANGS if (i, l) not in invalid and (arm, i, l) not in have]
        if miss:
            out.append(f"{arm}: {len(miss)} of {len(items) * len(LANGS)} presentations missing (first: {miss[0][0]}/{miss[0][1]})")
    return out


def run_sealed(score_path, *, allow_test=False, allow_partial=False, B=9999, root=None, outdir=None):
    seal = verify_seal(score_path, allow_partial)
    rows = read_jsonl(score_path)
    nss = {r.get("namespace") for r in rows}
    if len(nss) != 1 or None in nss:
        raise AnalysisError(f"score journal must hold exactly one namespace, found {sorted(map(str, nss))}")
    ns = nss.pop()
    items = sorted({C.parse_pres(r["pres"])["item"] for r in rows})
    C.guard_items(items, ns, allow_test=allow_test, root=root)             # refuses test items unless the full SPEC guard holds
    man = C.load_manifest(root)
    fam_of = {i: f["family"] for f in man["families"] for i in f["items"]}
    d = os.path.dirname(score_path)
    plan_rows = read_jsonl(os.path.join(d, "plan.jsonl")) if os.path.exists(os.path.join(d, "plan.jsonl")) else None
    prac_rows = read_jsonl(os.path.join(d, "practice.jsonl")) if os.path.exists(os.path.join(d, "practice.jsonl")) else None
    exec_rows = read_jsonl(os.path.join(d, "exec.jsonl")) if os.path.exists(os.path.join(d, "exec.jsonl")) else None
    meta = {"namespace": ns, "n_score_rows": len(rows), "seal_sha256": seal["sha256"], "complete": bool(seal.get("complete")),
            "synthetic": False, "B": B}
    if plan_rows is not None:                                               # C5: strict join of the energy journal(s) to the Gemma plan rows
        plan_rows, meta["energy_attach"] = attach_energy(plan_rows, load_energy_rows(d), plan_line_shas(os.path.join(d, "plan.jsonl")),
                                                         strict=True)
    if prac_rows is not None:                                               # practice energy can only come from a C5 journal: none exists
        prac_rows, meta["practice_energy_fields_stripped"] = strip_energy_fields(prac_rows)
    fz_path = C.paths(root)["freeze"]
    try:
        with open(fz_path, encoding="utf-8") as f:
            fz = json.load(f)
    except (OSError, ValueError):
        fz = None
    meta["a24_2_cost_freeze"] = check_cost_freeze(fz)                  # raises on a disagreement; flags a missing record
    prep_rows = read_jsonl(os.path.join(d, "prep_cost.jsonl")) if os.path.exists(os.path.join(d, "prep_cost.jsonl")) else None
    if prep_rows is not None:
        prep_rows, meta["prep_energy_fields_stripped"] = strip_energy_fields(prep_rows)
    if meta["complete"] and ns == "test-v1":
        inv = {(C.parse_pres(r["pres"])["item"], C.parse_pres(r["pres"])["lang"]) for r in rows if is_invalid_row(r)}
        miss = completeness_problems(rows, man, ns, inv)
        if miss:
            raise AnalysisError("the seal says complete but the score journal does not cover the registered estimand "
                                "(prereg 9.1: the mean over the capped test items): " + "; ".join(miss[:8]))
    res = analyse(rows, fam_of_manifest=fam_of, plan_rows=plan_rows, practice_rows=prac_rows, B=B, meta=meta, exec_rows=exec_rows, prep_rows=prep_rows)
    return res, write_outputs(res, outdir or d)


def synthetic_practice_rows(n=30, seed=3):
    """Generated Gemma practice rows (A24.2 fields: plan_secs, energy joules; wall_s is the whole episode and must NOT be priced)."""
    rg = random.Random(seed)
    out = []
    for k in range(n):
        ps = rg.uniform(1.0, 6.0)
        out.append({"key": f"syn-prac|{k}", "order": 0, "attempt": k + 1, "cost_usd": 0.0, "plan_secs": round(ps, 3), "wall_s": round(ps + rg.uniform(60, 200), 1),
                    "energy_j_gpu": round(ps * rg.uniform(45.0, 75.0), 2), "energy_j_cpu": round(ps * rg.uniform(10.0, 20.0), 2)})
    return out


def run_synthetic(*, B=9999, outdir=None, seed=20261010):
    rows, truth = synthetic_rows(seed)
    plan = synthetic_plan_rows(rows)
    meta = {"namespace": "synthetic", "synthetic": True, "n_score_rows": len(rows), "seal_sha256": None, "complete": True, "B": B,
            "true_arm_success_probabilities": truth}
    res = analyse(rows, plan_rows=plan, practice_rows=synthetic_practice_rows(), B=B, meta=meta)
    out = outdir or os.path.join(C.paths()["matrix_docs"], "synthetic")
    return res, write_outputs(res, out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--synthetic", action="store_true")
    g.add_argument("--sealed", metavar="SCORE_JSONL")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--B", type=int, default=9999)
    ap.add_argument("--outdir")
    a = ap.parse_args(argv)
    try:
        if a.synthetic:
            res, (jp, mp) = run_synthetic(B=a.B, outdir=a.outdir)
        else:
            res, (jp, mp) = run_sealed(a.sealed, allow_test=a.allow_test, allow_partial=a.partial, B=a.B, outdir=a.outdir)
    except (AnalysisError, C.TestItemRefused, C.JournalCorrupt) as e:
        print(f"REFUSED: {type(e).__name__}: {e}")
        return 2
    print(f"H1: {res['H1'].get('wording', res['H1'])}")
    print(f"wrote {jp}\nwrote {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
