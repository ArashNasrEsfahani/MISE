# Kitchens 2/3 replication analysis (A29 item 3)

Exploratory, estimation-focused replication. Kitchen 1 is never pooled. No superiority statement; G+M - O0 has two exploratory criteria, the other three contrasts are estimation-only and their intervals are not simultaneous. Intervals: nominal 96.5 % CR2-Satterthwaite interval; actual coverage is not assured.

- FloorPlan15 (`test-k2-v1`): 1200 score rows, 100 items, 40 families, levels {'L1': 47, 'L2': 35, 'L3': 18}, INVALID dropped 0, seal 28ed8672719c4f7e, complete True.
- FloorPlan17 (`test-k3-v1`): 1212 score rows, 101 items, 37 families, levels {'L1': 46, 'L2': 36, 'L3': 19}, INVALID dropped 0, seal 9de0ae338244e5c0, complete True.

Analysis plan k23-analysis-2 sha 673b4710c86ebfd8.

## The four contrasts (primary view)

| contrast | scope | estimate | nominal 96.5% interval | se | nu | K (guard >= 20) | items | families |
|---|---|---|---|---|---|---|---|---|
| G+M - O0 | FloorPlan15 | -10.5 pp | [-19.6 pp, -1.4 pp] | 0.0408 | 25.33 | 16 (failed) | 100 | 40 |
| G+M - O0 | FloorPlan17 | -15.3 pp | [-26.3 pp, -4.4 pp] | 0.0491 | 24.70 | 22 (passed) | 101 | 37 |
| G+M - O0 | **two-kitchen average** | -12.9 pp | [-21.9 pp, -3.9 pp] | 0.0403 | 25.37 | 27 (passed) | 201 | 41 clusters (36 shared) |
| G+M - G0 | FloorPlan15 | +14.0 pp | [+4.5 pp, +23.5 pp] | 0.0427 | 25.33 | 15 (failed) | 100 | 40 |
| G+M - G0 | FloorPlan17 | +11.4 pp | [+3.6 pp, +19.1 pp] | 0.0347 | 24.70 | 21 (passed) | 101 | 37 |
| G+M - G0 | **two-kitchen average** | +12.7 pp | [+5.7 pp, +19.7 pp] | 0.0313 | 25.37 | 24 (passed) | 201 | 41 clusters (36 shared) |
| O+M - O0 | FloorPlan15 | +0.5 pp | [-7.4 pp, +8.4 pp] | 0.0353 | 25.33 | 10 (failed) | 100 | 40 |
| O+M - O0 | FloorPlan17 | +0.0 pp | [-6.6 pp, +6.6 pp] | 0.0294 | 24.70 | 19 (failed) | 101 | 37 |
| O+M - O0 | **two-kitchen average** | +0.2 pp | [-5.5 pp, +6.0 pp] | 0.0259 | 25.37 | 22 (passed) | 201 | 41 clusters (36 shared) |
| G+M - O+M | FloorPlan15 | -11.0 pp | [-19.0 pp, -3.0 pp] | 0.0360 | 25.33 | 14 (failed) | 100 | 40 |
| G+M - O+M | FloorPlan17 | -15.3 pp | [-24.5 pp, -6.2 pp] | 0.0412 | 24.70 | 20 (passed) | 101 | 37 |
| G+M - O+M | **two-kitchen average** | -13.2 pp | [-21.0 pp, -5.3 pp] | 0.0353 | 25.37 | 24 (passed) | 201 | 41 clusters (36 shared) |

## G+M - O0 exploratory criteria: primary

- Average (pooled lower bound > -5 pp): **not met** (lower bound <= -5 pp (not met for this criterion only; not evidence of no difference)).
- Each kitchen (both lower bounds > -5 pp, a conjunction): **not met** (FloorPlan15: no threshold statement: guard failed (K = 16 < 20 families with a discordant presentation); FloorPlan17: lower bound <= -5 pp (not met for this criterion only; not evidence of no difference)).
  - FloorPlan15: not met (no threshold statement: guard failed (K = 16 < 20 families with a discordant presentation))
  - FloorPlan17: not met (lower bound <= -5 pp (not met for this criterion only; not evidence of no difference))
- Threshold statement 'upper bound < -5 pp' (exploratory): average False; FloorPlan15 None; FloorPlan17 False (None = no threshold statement permitted).
- Thesis wording: The exploratory average non-inferiority criterion was not met (not evidence of no difference). The exploratory each-kitchen non-inferiority criterion was not met (not evidence of no difference). Never: "was shown non-inferior".

## G+M - O0 exploratory criteria: infra-paired sensitivity (A21 A.5)

- Average (pooled lower bound > -5 pp): **not met** (lower bound <= -5 pp (not met for this criterion only; not evidence of no difference)).
- Each kitchen (both lower bounds > -5 pp, a conjunction): **not met** (FloorPlan15: no threshold statement: guard failed (K = 16 < 20 families with a discordant presentation); FloorPlan17: no threshold statement: guard failed (K = 18 < 20 families with a discordant presentation)).
  - FloorPlan15: not met (no threshold statement: guard failed (K = 16 < 20 families with a discordant presentation))
  - FloorPlan17: not met (no threshold statement: guard failed (K = 18 < 20 families with a discordant presentation))
- Threshold statement 'upper bound < -5 pp' (exploratory): average False; FloorPlan15 None; FloorPlan17 None (None = no threshold statement permitted).
- Thesis wording: The exploratory average non-inferiority criterion was not met (not evidence of no difference). The exploratory each-kitchen non-inferiority criterion was not met (not evidence of no difference). Never: "was shown non-inferior".

Infrastructure-sensitive arms (> 5 % infra cells): {'FloorPlan15': [], 'FloorPlan17': ['G+M', 'G0', 'O+M', 'O0', 'P+M', 'P0']}; primary and infra-paired wordings agree: True.

## Sensitivities (estimates; same CR2 algebra; no threshold statement)

| contrast | scope | infra-paired | items (dropped pres) | distinct units only | items |
|---|---|---|---|---|---|
| G+M - O0 | FloorPlan15 | -11.5 pp [-20.9 pp, -2.1 pp] | 100 (10) | -10.5 pp [-19.6 pp, -1.4 pp] | 100 |
| G+M - O0 | FloorPlan17 | -17.0 pp [-28.2 pp, -5.8 pp] | 97 (20) | -15.3 pp [-26.3 pp, -4.4 pp] | 101 |
| G+M - O0 | two-kitchen average | -14.3 pp [-23.6 pp, -4.9 pp] | 197 | -12.9 pp [-21.9 pp, -3.9 pp] | 201 |
| G+M - G0 | FloorPlan15 | +14.0 pp [+3.9 pp, +24.1 pp] | 100 (10) | +14.0 pp [+4.5 pp, +23.5 pp] | 100 |
| G+M - G0 | FloorPlan17 | +8.9 pp [+1.0 pp, +16.7 pp] | 96 (24) | +11.4 pp [+3.6 pp, +19.1 pp] | 101 |
| G+M - G0 | two-kitchen average | +11.4 pp [+4.2 pp, +18.7 pp] | 196 | +12.7 pp [+5.7 pp, +19.7 pp] | 201 |
| O+M - O0 | FloorPlan15 | -1.5 pp [-9.4 pp, +6.4 pp] | 100 (10) | +0.5 pp [-7.4 pp, +8.4 pp] | 100 |
| O+M - O0 | FloorPlan17 | +0.0 pp [-6.1 pp, +6.1 pp] | 95 (22) | +0.0 pp [-6.6 pp, +6.6 pp] | 101 |
| O+M - O0 | two-kitchen average | -0.8 pp [-6.3 pp, +4.8 pp] | 195 | +0.2 pp [-5.5 pp, +6.0 pp] | 201 |
| G+M - O+M | FloorPlan15 | -10.0 pp [-18.0 pp, -2.0 pp] | 100 (10) | -21.7 pp [-37.6 pp, -5.7 pp] | 60 |
| G+M - O+M | FloorPlan17 | -17.2 pp [-26.7 pp, -7.6 pp] | 96 (20) | -26.2 pp [-40.9 pp, -11.4 pp] | 65 |
| G+M - O+M | two-kitchen average | -13.6 pp [-21.6 pp, -5.6 pp] | 196 | -23.9 pp [-38.1 pp, -9.8 pp] | 125 |

## Common-item intersection (descriptive only)

Items: {'FloorPlan15': 100, 'FloorPlan17': 101}; common 89; identical sets: False. 

| contrast | FloorPlan15 | FloorPlan17 | two-kitchen average |
|---|---|---|---|
| G+M - O0 | -10.7 pp [-20.5 pp, -0.9 pp] | -13.5 pp [-25.2 pp, -1.7 pp] | -12.1 pp [-22.0 pp, -2.2 pp] |
| G+M - G0 | +13.5 pp [+3.3 pp, +23.7 pp] | +11.2 pp [+2.7 pp, +19.8 pp] | +12.4 pp [+4.8 pp, +19.9 pp] |
| O+M - O0 | +0.6 pp [-8.4 pp, +9.5 pp] | +0.6 pp [-6.4 pp, +7.5 pp] | +0.6 pp [-5.9 pp, +7.1 pp] |
| G+M - O+M | -11.2 pp [-20.0 pp, -2.5 pp] | -14.0 pp [-23.8 pp, -4.3 pp] | -12.6 pp [-21.4 pp, -3.9 pp] |

## FloorPlan15: SR per arm, language and level (descriptive, family-cluster CI)

| arm | overall | EN | FA | L1 | L2 | L3 |
|---|---|---|---|---|---|---|
| G+M | 77.0% [66.2%, 86.5%] | 78.0% [68.5%, 86.7%] | 76.0% [62.9%, 87.2%] | 87.2% [70.2%, 98.8%] | 87.1% [76.9%, 95.8%] | 30.6% [7.9%, 54.5%] |
| G0 | 63.0% [51.8%, 73.8%] | 72.0% [61.2%, 82.1%] | 54.0% [40.2%, 68.0%] | 75.5% [63.6%, 87.5%] | 68.6% [55.7%, 80.3%] | 19.4% [0.0%, 44.4%] |
| O+M | 88.0% [77.5%, 95.8%] | 88.0% [78.0%, 95.8%] | 88.0% [76.6%, 96.6%] | 88.3% [69.1%, 100.0%] | 91.4% [82.3%, 98.6%] | 80.6% [59.4%, 95.5%] |
| O0 | 87.5% [81.6%, 92.6%] | 88.0% [80.4%, 94.0%] | 87.0% [80.6%, 93.0%] | 91.5% [84.7%, 97.6%] | 87.1% [78.6%, 95.2%] | 77.8% [56.2%, 93.5%] |
| P+M | 94.0% [89.8%, 97.3%] | 94.0% [88.8%, 98.1%] | 94.0% [89.5%, 97.9%] | 96.8% [93.4%, 100.0%] | 91.4% [82.9%, 98.5%] | 91.7% [81.2%, 100.0%] |
| P0 | 93.0% [88.6%, 96.5%] | 94.0% [88.8%, 98.1%] | 92.0% [86.8%, 96.5%] | 95.7% [91.9%, 98.9%] | 88.6% [79.5%, 96.2%] | 94.4% [86.1%, 100.0%] |

| arm | cells | planner infra | execution infra | total rate | flag | invalid | distinct units | shared share | A36 flags (not applied) |
|---|---|---|---|---|---|---|---|---|---|
| G+M | 200 | 0 | 10 | 5.0% |  | 0 | 200 | 57.5% | 10 |
| G0 | 200 | 0 | 9 | 4.5% |  | 0 | 200 | 44.5% | 9 |
| O+M | 200 | 0 | 8 | 4.0% |  | 0 | 200 | 68.5% | 8 |
| O0 | 200 | 0 | 10 | 5.0% |  | 0 | 200 | 66.5% | 10 |
| P+M | 200 | 0 | 9 | 4.5% |  | 0 | 200 | 77.5% | 9 |
| P0 | 200 | 0 | 10 | 5.0% |  | 0 | 200 | 69.5% | 10 |

Cost (FloorPlan15, A29 item 4; planning only):

| arm | billed USD | standard-price USD | tokens in/out | calls / cell |
|---|---|---|---|---|
| G0 | 0.0000 | 0.0000 | 613360/15017 | 1.10 |
| G+M | 0.0000 | 0.0000 | 599805/13556 | 1.04 |
| O0 | 3.8722 | 3.8722 | 836142/26384 | 1.00 |
| O+M | 3.9098 | 3.9098 | 862629/22963 | 1.00 |

A24.2 (FloorPlan15, descriptive; statement not made): R = SR(G+M)/SR(O0) = 0.880 [0.781, 0.963]; C (headline) = unknown (CI not obtainable: a matched presentation has no plan row or an unmeasured energy / time component).

## FloorPlan17: SR per arm, language and level (descriptive, family-cluster CI)

| arm | overall | EN | FA | L1 | L2 | L3 |
|---|---|---|---|---|---|---|
| G+M | 70.8% [60.6%, 80.3%] | 74.3% [63.6%, 84.0%] | 67.3% [56.0%, 78.0%] | 77.2% [64.1%, 88.0%] | 86.1% [75.0%, 94.7%] | 26.3% [5.9%, 50.0%] |
| G0 | 59.4% [48.1%, 69.7%] | 64.4% [52.7%, 74.5%] | 54.5% [41.5%, 66.7%] | 69.6% [55.7%, 81.2%] | 65.3% [51.9%, 75.9%] | 23.7% [2.9%, 47.6%] |
| O+M | 86.1% [78.7%, 92.5%] | 86.1% [77.8%, 93.1%] | 86.1% [78.7%, 93.0%] | 88.0% [79.2%, 95.2%] | 88.9% [78.1%, 97.1%] | 76.3% [53.1%, 92.5%] |
| O0 | 86.1% [81.4%, 90.4%] | 84.2% [77.2%, 89.9%] | 88.1% [83.2%, 92.9%] | 84.8% [78.3%, 90.9%] | 88.9% [80.9%, 96.3%] | 84.2% [66.7%, 95.7%] |
| P+M | 89.6% [83.0%, 94.8%] | 89.1% [81.0%, 95.4%] | 90.1% [83.5%, 95.8%] | 87.0% [78.8%, 93.3%] | 91.7% [82.3%, 98.6%] | 92.1% [78.1%, 100.0%] |
| P0 | 93.1% [89.2%, 96.6%] | 92.1% [86.4%, 96.7%] | 94.1% [90.3%, 97.8%] | 91.3% [85.4%, 97.0%] | 91.7% [83.8%, 97.5%] | 100.0% [100.0%, 100.0%] |

| arm | cells | planner infra | execution infra | total rate | flag | invalid | distinct units | shared share | A36 flags (not applied) |
|---|---|---|---|---|---|---|---|---|---|
| G+M | 202 | 0 | 14 | 6.9% | infrastructure-sensitive | 0 | 201 | 54.5% | 14 |
| G0 | 202 | 0 | 18 | 8.9% | infrastructure-sensitive | 0 | 201 | 45.5% | 18 |
| O+M | 202 | 0 | 15 | 7.4% | infrastructure-sensitive | 0 | 202 | 67.3% | 15 |
| O0 | 202 | 0 | 15 | 7.4% | infrastructure-sensitive | 0 | 202 | 65.3% | 15 |
| P+M | 202 | 0 | 18 | 8.9% | infrastructure-sensitive | 0 | 202 | 76.7% | 18 |
| P0 | 202 | 0 | 12 | 5.9% | infrastructure-sensitive | 0 | 202 | 70.8% | 12 |

Cost (FloorPlan17, A29 item 4; planning only):

| arm | billed USD | standard-price USD | tokens in/out | calls / cell |
|---|---|---|---|---|
| G0 | 0.0000 | 0.0000 | 613041/14866 | 1.10 |
| G+M | 0.0000 | 0.0000 | 594350/13114 | 1.03 |
| O0 | 3.8446 | 3.8446 | 836195/24990 | 1.00 |
| O+M | 3.9037 | 3.9037 | 863298/22526 | 1.00 |

A24.2 (FloorPlan17, descriptive; statement not made): R = SR(G+M)/SR(O0) = 0.822 [0.710, 0.925]; C (headline) = unknown (CI not obtainable: a matched presentation has no plan row or an unmeasured energy / time component).

## Disclosures

- Kitchens 2/3 are an estimation-focused, exploratory replication (A29 item 3): power and coverage for these samples and the pooled estimator are unknown; >= 80 % power has not been demonstrated; an unmet criterion is never evidence of no difference.
- h1-rule-1 was validated on Kitchen 1's family sizes only (A24.1-H1 item 3); every interval is a nominal 96.5 % CR2-Satterthwaite interval whose actual coverage is not assured.
- Transfer of the registered task set to new kitchens, not independent task replication; claims cover applicable, reference-solvable tasks (A29 item 2).
- Kitchen 1 results are never pooled with Kitchens 2/3. The two kitchens are weighted equally regardless of size; their task compositions are disclosed.
- Estimates are unweighted item averages for the selected task sets, not sampling-weighted estimates for the eligible pool.
- Spend-adaptive sizing (A29 item 6) used accounting counters only; outcome-independent sampling and unconditional interval coverage are not claimed. With a fall-back size, the language and level analyses are descriptive only.
