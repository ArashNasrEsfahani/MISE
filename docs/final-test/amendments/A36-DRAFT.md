# DRAFT amendment A36 (r2) -- run the Kitchen 1 locked test after acceptance NOT PASS; exclude reset-gate failures

**Status: DRAFT, NOT SIGNED.** Drafted 2026-10-15 ~17:45 IST on Arash's instruction ("Go for it", after: "It's clearly the simulator
limitations, we just omit the runs that were caused because of that from the results"); r2 ~18:30 applies every blocking fix of Astra
A36-DRAFT-r1 (`docs/final-test/_astra/A36-DRAFT-r1.txt`). Nothing here is in force until Arash signs it (recorded by him in
`docs/final-test/signatures.json`; never written by code). `preregistration.md` is NOT edited.

**Honest label.** Written AFTER acceptance attempt 1 (NOT PASS: completeness, (e), (c)) and attempt 2 (NOT PASS: completeness, (e);
verdict sha eeda4ef8..., `protocol_stopped: true`, Astra ACC-V2-A2-verdict NOT PASS CONFIRMED), and after their exec rows were inspected.
Written BEFORE any Kitchen 1 locked-test planner call, execution or outcome. So it is blind to the locked-test outcomes, but NOT blind to
the acceptance results. It overrides the signed stop condition (Common Terms 3-4 of A34-A2 / A35) and the acceptance registration's
"Arash's proceed / stop decision cannot redefine the verdict". It does NOT redefine the verdict: both acceptance attempts stay NOT PASS.

**What the acceptance failures were (complete).** Attempt 1: reset-gate blocks (completeness, (e)) and the provenance-checker gap (c),
fixed by A35. Attempt 2: reset-gate blocks (completeness and two P0 cells of (e)) and, independently, the P0 reference plan of
microwave-bowl:fa did not reach its goal after passing the ordinary reset check ("heat Potato": potato not detected) -- a perception miss,
not a reset failure.

---

## Plain meaning for Arash

1. The acceptance verdict stays NOT PASS; the thesis says so in one paragraph, including the perception miss.
2. You authorise the Kitchen 1 locked test to run anyway, under this amended rule.
3. In the test, a run is dropped only if all 3 of its tries were stopped by the start-up reset check, before the robot did anything.
   Crashes or hangs while the robot is acting are NOT dropped (they could depend on the plan); they keep the registered scoring.
4. Every comparison between two arms uses only the tasks both arms kept. The original rule (dropped runs count as failures) is reported
   next to it. If the two give different answers, the thesis says so.
5. The thesis does not say "H1 confirmed"; it says what the numbers show under this amended rule (wording in rule 8).

---

## Rule text

1. **Proceed.** By Arash's decision, the Kitchen 1 (FloorPlan1) locked test under ft-v2 proceeds although acceptance-v2 ended terminally
   NOT PASS (attempts 1 and 2). This supersedes Common Term 4 of A34-A2 / A35 and the freeze precondition "acceptance PASS", **for this
   decision only**: the freeze records acceptance as NOT PASS and this waiver; it never marks acceptance PASS. No acceptance attempt 3; no
   acceptance row, verdict, bundle or journal is changed, re-scored or pooled across attempts. Every other part of the preregistration,
   A14-A35 and the freeze gates stays in force (roster, arms, items, treatment identity 7a2ee211..., prompts, package, retries,
   thresholds, budget ceilings 30 / 50 USD, A34-A2's override and its obs_drift record, the A21 infrastructure classes and flags).
2. **Exclusion flag `excluded_reset` (analysis only).** A planned core cell is flagged `excluded_reset` if and only if ALL hold:
   (a) its final registered class (A21 precedence) is `infra_exec`; (b) every one of its 3 registered attempts ended at the pre-execution
   reset gate (registered comparison failed and not overridden under A34-A2) and no plan step was dispatched in any attempt (a crash or hang
   during or after the first action counts as execution begun); (c) the executor's per-attempt evidence (stage, gate breaches, retry history)
   is present and complete. A mixed history, a crash / hang at any stage, or missing / uncertain evidence -> NOT flagged; registered scoring.
   The underlying `infra_exec` class is kept unchanged; the flag is an extra field. Core cells = the main-block r0 / run0 planned cells
   of each applicable arm.
3. **Primary analysis.** A flagged cell is treated as "presentation not available for that arm". Every contrast uses only presentations
   available in BOTH compared arms (paired), then the registered aggregation unchanged: per item the mean of its retained paired EN / FA
   differences; equal item weighting; items with none retained are dropped; families with none are dropped; K, N and the counts are
   recomputed. Contrast estimates use this common subset, never separately filtered per-arm SRs.
   - **H1** uses the current registered rule h1-rule-1 (C7, A24.1-H1): `sim/final_test_h1_stats.py` `h1_rule_c7`, nominal 96.5 %
     CR2-Satterthwaite interval, computational tail .0175, guard K >= 20 (K < 20 -> inconclusive), wordings unchanged. Its validation
     did not cover this selection step; that is stated.
   - **Secondary Holm family**: H2a SR(G+M)-SR(G0), H2c SR(O+M)-SR(O0), H3a SR(O+M)-SR(G+M); m = 3, alpha .05; a member without valid
     evidence enters with p = 1; existing validity requirements unchanged; every other contrast stays exploratory.
   - Per-arm SR tables use each arm's non-flagged cells and print the denominator.
4. **Registered analysis, reported alongside (same tables).** The A21 rule: every `infra_exec` cell counts as failure (SR 0) and stays in
   every denominator. If any H1 wording or Holm decision differs, both are reported and the difference is named. Agreement between the
   two does not show the exclusions were unbiased.
5. **Disclosure, always.** Per arm: the number and share of flagged cells, their presentations and gate breach kinds; per contrast: the
   retained presentation / item / family counts and K. The A34-A2 obs_drift tables and the A21 infrastructure tables as registered.
6. **Reporting flag (pre-set).** If more than 5 % of any arm's core cells are flagged, every contrast involving that arm is labelled
   "reset-limited" in tables and text, with its count. This is a reporting flag, not evidence that fewer exclusions are harmless; the run
   is not stopped or redone because of it. (Context: acceptance had 3 / 228 and 7 / 224 terminally blocked execution units; attempt 2
   blocked 14 cells, incl. 2 P0 cells; G0 lost 4 / 48.)
7. **P0 / P+M** stay diagnostics, unchanged. The acceptance P0 miss is reported as a perception failure.
8. **Status wording (thesis, required).** "Acceptance ended NOT PASS twice. After reviewing those results, we authorised a separate locked
   evaluation through A36, overriding the signed stop condition. Its exclusion rule was fixed before any locked-test calls or outcomes.
   Results describe performance under that amended rule on retained presentations; they do not establish H1 under the original
   preregistered protocol. The original failure-counting analysis is reported alongside." A threshold crossing is worded "meets the
   numerical criterion of H1 in the amended subset analysis", never "H1 confirmed".

## Why this is reasonable, and its limits

- Fixed before any locked-test outcome; the flag comes from pre-execution reset checks only, which never read plans or outcomes, and
  excludes nothing once the robot has started acting.
- Both arms of a contrast use the same retained presentations. Selection bias and a change in task mix remain possible (in acceptance the
  blocks clustered on 4 of 48 presentations: spoon-in-sink, dinner-done, snack-cleanup); reset eligibility can also depend on scheduling,
  shared execution units and preceding activity. The result describes the retained subset.
- It deviates from terms Arash signed the same day; the thesis says so.
- Under A34-A2, an overridden run still started from a slightly different picture than the plan was made from (obs_drift); unchanged.

## Implementation (after signature only; Astra-checked; gates re-run)

- Executor / scoring: per-attempt failure stage + retry history recorded and bound (raw classes, journals, shared-unit links kept); the
  `excluded_reset` flag computed in scoring and bound into the analysis seals.
- Analysis: common-subset contrasts through `h1_rule_c7` and the m = 3 Holm family; the registered table alongside; the 5 % flag.
  Selftests: mixed retries, first-action crash, missing evidence, shared units, one-language retention, empty subsets, K < 20,
  unchanged Holm membership, non-`infra_exec` never flagged.
- Spend: every attempt's cost kept; A21 infrastructure sensitivity / flags and A34 drift analyses kept.
- Freeze: binds A36's signature key, this file's sha, the revised analysis identity and the preserved acceptance verdict; the exception
  waives only the named prerequisite.
- Owed gate evidence, as separate hash-bound reports: tokens / context from verified attempt-2 evidence; G-iso's absent / present / absent
  comparison and 12.5's empty-plan control need their own runs (ordinary acceptance rows do not establish them); leakage its required audit;
  12.7 refreshed accounting probes; 12.6 after its components. Missing evidence stays blocking.
- Then FREEZE-SUMMARY.html with the real freeze sha for Arash's freeze signature; then the locked test (~18-26 h, ~8-15 USD).

## Reply to sign

`I sign A36 as drafted (A36-DRAFT.md r2).`
