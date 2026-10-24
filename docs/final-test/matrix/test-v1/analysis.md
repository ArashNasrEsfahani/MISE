# Final-test analysis

Namespace: `test-v1`. Score rows: 5930. Families: 42. INVALID presentations dropped from every arm: 0. Seal: 850e2c2e76680172a9da1cefa2104c1c2d9d8591e7f8d6b01fa9e112dc082c60. Complete: True.

## H1 (primary): SR(G+M) - SR(O0)

**Wording: non-inferiority unresolved (inconclusive)**
Procedure: h1-rule-1 (C7: CR2 + Satterthwaite, guard K >= 20 families with a discordant presentation, one-sided level 0.0175 per tail = a nominal 96.5 % interval; NI iff the lower bound > -5 pp, superiority only after NI). Validated in V100 (A24.1).

| view | estimate | nominal 96.5% interval (C7) | K (guard >= 20) | wording |
|---|---|---|---|---|
| primary (item-weighted, validated) | -9.6 pp | [-21.0 pp, +1.8 pp] | 20 (passed) | non-inferiority unresolved (inconclusive) |
| family-weighted (sensitivity) | -9.4 pp | [-18.8 pp, -0.0 pp] | 20 (passed) | non-inferiority unresolved (inconclusive) |
| infra-paired (drop either-arm infra; sensitivity) | -10.5 pp | [-22.4 pp, +1.3 pp] | 17 (failed) | non-inferiority unresolved (inconclusive): guard failed (K = 17 < 20 families with a discordant presentation) |

Families where G+M beats O0 / tie / loses: 3 / 23 / 16. Items: 125; presentations: 250.

## Multiplicity (frozen, A21 F)

H1 stands alone (one-sided .025 non-inferiority, then superiority in a fixed sequence, h1-rule-1 of A24.1); it is NOT a member of the Holm family. The secondary family has m = 3 tests at alpha .05 (H2a, H2c, H3a; Holm-Bonferroni; A24.5 moved H4a-b and H5a-c to exploratory and A23 withdrew H2b, H3b, H3c); the smaller m relaxes the Holm thresholds, and the m = 8 adjustment is printed beside the verdict for transparency. There is no study-wide .05 claim. A member without validity evidence is estimation-only. Analysis plan sha ad4b67bca59e5252.

H1 observed discordance (share of presentations in which G+M and O0 differ): 0.22; families with a discordant presentation K = 20. Below K = 20 h1-rule-1 makes no claim (the interval is not trustworthy there: its coverage fell to .82 in the V100 low-discordance cells).

## Exploratory equivalence: SR(G+M) - SR(O+M) within +/-5 pp (E1, never confirmatory)

Estimate -13.6 pp, CR2 90 % interval [-19.8 pp, -7.4 pp], K = 20: **equivalence not established**. 'Not significantly different' is never 'equivalent'; simulated power of E1 at a true difference of 0 was at most 2 % with this sample.

## Holm family (alpha .05, m=3)

| id | contrast | estimate | 95% CI | p | p Holm | reject | discordance | shared-unit pres | missing pres |
|---|---|---|---|---|---|---|---|---|---|
| H2a | G+M - G0 | +14.4 pp | [+7.4 pp, +21.6 pp] | 0.0002 | 0.0006 | YES | 0.24 | 0 | 0 |
| H2c | O+M - O0 | +4.0 pp | [-1.0 pp, +10.8 pp] | 0.1114 | 0.1114 | no | 0.11 | 0 | 0 |
| H3a | O+M - G+M | +13.6 pp | [+7.0 pp, +20.8 pp] | 0.0006 | 0.0012 | YES | 0.19 | 105 | 0 |

For transparency (A24.5): the Holm-adjusted p of the same three tests if the family had been the A23 m = 8 (adding H4a-b, H5a-c): H2a 0.0016; H2c 0.3342; H3a 0.0042. Every m = 3 adjusted p is smaller than or equal to its m = 8 value.

## Exploratory contrasts (A24.5: raw p, NOT adjusted, no confirmatory conclusion)

| id | contrast | estimate | 95% CI | raw p | p Holm if m = 8 | discordance |
|---|---|---|---|---|---|---|
| H4a | G+M - G+FS | +12.8 pp | [+5.3 pp, +20.3 pp] | 0.0040 | 0.0200 | 0.22 |
| H4b | O+M - O+FS | +3.2 pp | [-0.9 pp, +7.8 pp] | 0.1772 | 0.3544 | 0.11 |
| H5a | X+M - X0 | +4.0 pp | [+0.0 pp, +8.5 pp] | 0.0459 | 0.1836 | 0.12 |
| H5b | X+M - G+M | +12.0 pp | [+5.6 pp, +19.3 pp] | 0.0014 | 0.0084 | 0.19 |
| H5c | G+M - X0 | -8.0 pp | [-16.3 pp, -0.4 pp] | 0.7478 | 1.0000 | 0.22 |

Withdrawn before testing (A23, Sonnet archived; IDs kept, nothing computed): H2b SR(S+M) - SR(S0); H3b SR(S+M) - SR(G+M); H3c SR(G+M) - SR(S0).

Sensitivities (estimates): planner OR execution infrastructure failure of either arm dropped; presentations where the two arms share one execution dropped; H5 only, presentations with an X tool event or identity mismatch dropped:

| id | infra-paired | items | dropped pres | distinct units only | items | X-isolation-paired |
|---|---|---|---|---|---|---|
| H2a | +16.2 pp | 120 | 27 | +14.4 pp | 125 |  |
| H2c | +5.0 pp | 120 | 31 | +4.0 pp | 125 |  |
| H3a | +14.2 pp | 120 | 29 | +23.1 pp | 78 |  |
| H4a | +14.3 pp | 119 | 25 | +12.8 pp | 125 |  |
| H4b | +4.5 pp | 121 | 28 | +3.2 pp | 125 |  |
| H5a | +5.8 pp | 120 | 32 | +4.0 pp | 125 | +4.0 pp |
| H5b | +13.3 pp | 120 | 30 | +21.5 pp | 79 | +12.0 pp |
| H5c | -8.4 pp | 119 | 26 | -8.0 pp | 125 | -8.0 pp |

## Shared executions (dependence, A21 D)

plan cells with equal execution-unit keys share one simulator run; their outcome is ONE realisation. The family cluster contains every unit, so the intervals and tests already treat copies as dependent; the report adds the number of independent executions and a sensitivity that drops presentations where the two arms of a contrast share a unit (that estimate is performance CONDITIONAL on non-shared executions: it changes the population and cannot validate or replace the full-sample result)

| arm | scored cells | distinct units | cells sharing a unit with another arm |
|---|---|---|---|
| G+FS | 250 | 249 | 205 (82.0%) |
| G+M | 250 | 250 | 169 (67.6%) |
| G+M/p | 250 | 250 | 184 (73.6%) |
| G+M@o0k1 | 60 | 60 | 59 (98.3%) |
| G+M@o0k3 | 60 | 60 | 60 (100.0%) |
| G+M@o0k5 | 60 | 60 | 60 (100.0%) |
| G0 | 250 | 247 | 204 (81.6%) |
| O+FS | 250 | 250 | 226 (90.4%) |
| O+M | 250 | 250 | 221 (88.4%) |
| O+M/p | 250 | 250 | 228 (91.2%) |
| O+M@o0k1 | 60 | 60 | 57 (95.0%) |
| O+M@o0k3 | 60 | 60 | 57 (95.0%) |
| O+M@o0k5 | 60 | 60 | 59 (98.3%) |
| O0 | 250 | 250 | 224 (89.6%) |
| P+M | 250 | 250 | 193 (77.2%) |
| P0 | 250 | 250 | 211 (84.4%) |
| X+M | 250 | 250 | 206 (82.4%) |
| X0 | 250 | 250 | 204 (81.6%) |

| contrast | presentations | with one shared unit | with distinct units | discordant |
|---|---|---|---|---|
| H2a G+M - G0 | 250 | 0 | 250 | 60 |
| H2c O+M - O0 | 250 | 0 | 250 | 28 |
| H3a O+M - G+M | 250 | 105 | 145 | 48 |
| H4a G+M - G+FS | 250 | 0 | 250 | 54 |
| H4b O+M - O+FS | 250 | 0 | 250 | 28 |
| H5a X+M - X0 | 250 | 0 | 250 | 30 |
| H5b X+M - G+M | 250 | 104 | 146 | 48 |
| H5c G+M - X0 | 250 | 0 | 250 | 54 |
| H1 G+M - O0 | 250 | 0 | 250 | 54 |

Simulator time: 1273 unique runs, 15100.7 s in total; per arm (shared runs split equally): G+FS 850.6, G+M 1580.4, G+M/p 1374.2, G+M@o0k1 110.6, G+M@o0k3 117.2, G+M@o0k5 117.2, G0 1261.1, O+FS 776.7, O+M 1543.8, O+M/p 1385.2, O+M@o0k1 137.2, O+M@o0k3 142.5, O+M@o0k5 123.2, O0 1204.8, P+M 1477.2, P0 1357.0, X+M 866.1, X0 675.7.

## SR per arm, level and language (descriptive, family-cluster CI)

| arm | overall | L1 | L2 | L3 | EN | FA | L2 PC | L3 TDS | strict SR | realistic SR | slice flags T/?/n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G+FS | 55.6% [46.1%, 65.0%] | 72.3% [63.2%, 80.7%] | 58.0% [43.6%, 73.1%] | 23.2% [7.7%, 39.6%] | 60.0% [48.7%, 71.2%] | 51.2% [40.3%, 61.8%] | 0.537 | 0.317 | 0.552 | 55.6% [46.1%, 65.0%] | 0/0/5 |
| G+M | 68.4% [59.3%, 76.8%] | 86.2% [77.8%, 94.0%] | 78.0% [67.5%, 87.5%] | 21.4% [5.6%, 36.2%] | 66.4% [55.8%, 76.3%] | 70.4% [61.5%, 78.5%] | 0.808 | 0.391 | 0.684 | 68.4% [59.3%, 76.8%] | 0/0/7 |
| G+M/p | 70.0% [61.4%, 77.8%] | 87.2% [80.0%, 93.8%] | 81.0% [71.2%, 89.8%] | 21.4% [5.8%, 36.7%] | 70.4% [61.7%, 78.6%] | 69.6% [60.0%, 78.5%] | n/a | n/a | 0.696 | 70.0% [61.4%, 77.8%] | 0/0/7 |
| G+M@o0k1 | 58.3% [38.5%, 75.0%] | 85.0% [70.0%, 100.0%] | 70.0% [45.5%, 94.4%] | 20.0% [0.0%, 45.5%] | 56.7% [33.3%, 76.7%] | 60.0% [37.9%, 78.4%] | 0.758 | 0.373 | 0.583 | 58.3% [38.5%, 75.0%] | 0/0/0 |
| G+M@o0k3 | 60.0% [39.1%, 77.3%] | 90.0% [71.4%, 100.0%] | 70.0% [45.5%, 94.4%] | 20.0% [0.0%, 45.5%] | 60.0% [34.5%, 81.1%] | 60.0% [37.9%, 78.4%] | 0.758 | 0.373 | 0.600 | 60.0% [39.1%, 77.3%] | 0/0/0 |
| G+M@o0k5 | 60.0% [39.1%, 77.3%] | 90.0% [71.4%, 100.0%] | 70.0% [45.5%, 94.4%] | 20.0% [0.0%, 45.5%] | 60.0% [34.5%, 81.1%] | 60.0% [37.9%, 78.4%] | 0.758 | 0.373 | 0.600 | 60.0% [39.1%, 77.3%] | 0/0/0 |
| G0 | 54.0% [44.5%, 63.4%] | 69.1% [57.6%, 79.1%] | 57.0% [45.1%, 69.2%] | 23.2% [7.7%, 39.4%] | 60.8% [49.5%, 71.7%] | 47.2% [36.6%, 58.1%] | 0.612 | 0.340 | 0.540 | 54.0% [44.5%, 63.4%] | 0/0/3 |
| O+FS | 78.8% [72.6%, 84.4%] | 87.2% [81.0%, 93.5%] | 76.0% [66.3%, 86.4%] | 69.6% [54.3%, 82.7%] | 79.2% [71.7%, 86.3%] | 78.4% [71.5%, 84.3%] | 0.891 | 0.812 | 0.772 | 78.8% [72.6%, 84.4%] | 0/0/18 |
| O+M | 82.0% [76.1%, 87.3%] | 88.3% [80.0%, 96.4%] | 81.0% [73.1%, 87.9%] | 73.2% [60.7%, 85.0%] | 79.2% [71.0%, 86.6%] | 84.8% [78.8%, 90.0%] | 0.910 | 0.851 | 0.816 | 82.0% [76.1%, 87.3%] | 0/0/18 |
| O+M/p | 82.4% [76.3%, 87.7%] | 90.4% [83.3%, 96.7%] | 81.0% [71.4%, 89.3%] | 71.4% [57.7%, 83.9%] | 80.8% [73.3%, 87.6%] | 84.0% [77.5%, 89.4%] | n/a | n/a | 0.816 | 82.4% [76.3%, 87.7%] | 0/0/18 |
| O+M@o0k1 | 78.3% [64.8%, 90.6%] | 85.0% [70.0%, 100.0%] | 75.0% [50.0%, 95.5%] | 75.0% [46.2%, 100.0%] | 73.3% [57.1%, 89.7%] | 83.3% [68.8%, 94.9%] | 0.833 | 0.806 | 0.767 | 78.3% [64.8%, 90.6%] | 0/0/4 |
| O+M@o0k3 | 73.3% [59.4%, 86.7%] | 80.0% [60.0%, 100.0%] | 70.0% [45.0%, 90.9%] | 70.0% [45.5%, 93.8%] | 70.0% [51.6%, 88.5%] | 76.7% [62.1%, 89.7%] | 0.838 | 0.774 | 0.717 | 73.3% [59.4%, 86.7%] | 0/0/4 |
| O+M@o0k5 | 78.3% [66.1%, 89.4%] | 85.0% [70.0%, 100.0%] | 75.0% [50.0%, 95.5%] | 75.0% [50.0%, 95.5%] | 76.7% [63.3%, 90.6%] | 80.0% [64.0%, 93.3%] | 0.833 | 0.780 | 0.750 | 78.3% [66.1%, 89.4%] | 0/0/4 |
| O0 | 78.0% [69.9%, 84.7%] | 89.4% [82.4%, 96.2%] | 73.0% [59.4%, 86.0%] | 67.9% [51.8%, 81.9%] | 77.6% [68.6%, 85.6%] | 78.4% [69.8%, 85.5%] | 0.877 | 0.794 | 0.772 | 78.0% [69.9%, 84.7%] | 0/0/18 |
| P+M | 87.6% [82.5%, 91.9%] | 93.6% [87.5%, 100.0%] | 83.0% [74.6%, 90.0%] | 85.7% [75.0%, 94.6%] | 85.6% [78.1%, 91.9%] | 89.6% [84.9%, 93.9%] | n/a | n/a | 0.876 | 87.6% [82.5%, 91.9%] | 0/0/17 |
| P0 | 87.2% [82.0%, 91.9%] | 91.5% [85.6%, 97.2%] | 84.0% [75.6%, 91.5%] | 85.7% [76.6%, 94.4%] | 85.6% [78.7%, 92.0%] | 88.8% [83.3%, 93.6%] | n/a | n/a | 0.868 | 87.2% [82.0%, 91.9%] | 0/0/18 |
| X+M | 80.4% [73.8%, 86.3%] | 90.4% [82.4%, 98.6%] | 83.0% [74.5%, 90.4%] | 58.9% [42.6%, 73.9%] | 77.6% [68.5%, 85.7%] | 83.2% [76.9%, 88.6%] | 0.899 | 0.757 | 0.800 | 80.4% [73.8%, 86.3%] | 0/0/15 |
| X0 | 76.4% [69.9%, 82.4%] | 89.4% [84.0%, 95.0%] | 74.0% [65.6%, 83.7%] | 58.9% [40.5%, 75.0%] | 78.4% [70.3%, 85.9%] | 74.4% [66.9%, 81.1%] | 0.902 | 0.714 | 0.748 | 76.4% [69.9%, 82.4%] | 0/0/16 |

## Infrastructure failures per arm

| arm | cells | planner infra | rate | execution infra | rate | total rate | flag | invalid |
|---|---|---|---|---|---|---|---|---|
| G+FS | 250 | 0 | 0.0% | 15 | 6.0% | 6.0% | infrastructure-sensitive | 0 |
| G+M | 250 | 0 | 0.0% | 20 | 8.0% | 8.0% | infrastructure-sensitive | 0 |
| G+M/p | 250 | 0 | 0.0% | 16 | 6.4% | 6.4% | infrastructure-sensitive | 0 |
| G+M@o0k1 | 60 | 0 | 0.0% | 5 | 8.3% | 8.3% | infrastructure-sensitive | 0 |
| G+M@o0k3 | 60 | 0 | 0.0% | 4 | 6.7% | 6.7% | infrastructure-sensitive | 0 |
| G+M@o0k5 | 60 | 0 | 0.0% | 4 | 6.7% | 6.7% | infrastructure-sensitive | 0 |
| G0 | 250 | 0 | 0.0% | 16 | 6.4% | 6.4% | infrastructure-sensitive | 0 |
| O+FS | 250 | 0 | 0.0% | 17 | 6.8% | 6.8% | infrastructure-sensitive | 0 |
| O+M | 250 | 0 | 0.0% | 22 | 8.8% | 8.8% | infrastructure-sensitive | 0 |
| O+M/p | 250 | 0 | 0.0% | 17 | 6.8% | 6.8% | infrastructure-sensitive | 0 |
| O+M@o0k1 | 60 | 0 | 0.0% | 8 | 13.3% | 13.3% | infrastructure-sensitive | 0 |
| O+M@o0k3 | 60 | 0 | 0.0% | 9 | 15.0% | 15.0% | infrastructure-sensitive | 0 |
| O+M@o0k5 | 60 | 0 | 0.0% | 7 | 11.7% | 11.7% | infrastructure-sensitive | 0 |
| O0 | 250 | 0 | 0.0% | 19 | 7.6% | 7.6% | infrastructure-sensitive | 0 |
| P+M | 250 | 0 | 0.0% | 21 | 8.4% | 8.4% | infrastructure-sensitive | 0 |
| P0 | 250 | 0 | 0.0% | 16 | 6.4% | 6.4% | infrastructure-sensitive | 0 |
| X+M | 250 | 0 | 0.0% | 24 | 9.6% | 9.6% | infrastructure-sensitive | 0 |
| X0 | 250 | 0 | 0.0% | 18 | 7.2% | 7.2% | infrastructure-sensitive | 0 |

## REPEAT variance decomposition

| arm | pres | plan identity | SR spread (planner) | planner var | outcome agreement (exec) | exec var |
|---|---|---|---|---|---|---|
| G+M | 60 | 0.950 | 0.033 | 0.007 | 0.794 | 0.051 |
| G+M/p | 60 | 0.950 | 0.017 | 0.004 | 0.872 | 0.032 |
| G0 | 60 | 1.000 | 0.000 | 0.000 | 0.864 | 0.034 |
| O+M | 60 | 0.717 | 0.083 | 0.019 | 0.833 | 0.042 |
| O+M/p | 60 | 0.717 | 0.033 | 0.007 | 0.844 | 0.039 |
| O0 | 60 | 0.600 | 0.133 | 0.030 | 0.878 | 0.031 |

## API spend per success (plan journal cost_usd + shared practice cost; Gemma energy and hardware are in the A24.2 section below)

| arm | successes | plan USD | practice USD added | USD / success | tokens in/out |
|---|---|---|---|---|---|
| G+FS | 139.0 | 0.0000 | 0.0000 | 0.0000 | 771511/20425 |
| G+M | 171.0 | 0.0000 | 0.0000 | 0.0000 | 1092264/27620 |
| G+M/p | 175.0 | 0.0000 | 0.0000 | 0.0000 | 0/0 |
| G+M@o0k1 | 35.0 | 0.0000 | 0.0000 | 0.0000 | 165159/4307 |
| G+M@o0k3 | 36.0 | 0.0000 | 0.0000 | 0.0000 | 165159/4301 |
| G+M@o0k5 | 36.0 | 0.0000 | 0.0000 | 0.0000 | 165159/4297 |
| G0 | 135.0 | 0.0000 | 0.0000 | 0.0000 | 1195441/33719 |
| O+FS | 197.0 | 4.9791 | 0.0000 | 0.0253 | 1044629/40029 |
| O+M | 205.0 | 7.5166 | 0.0000 | 0.0367 | 1631419/55646 |
| O+M/p | 206.0 | 0.0000 | 0.0000 | 0.0000 | 0/0 |
| O+M@o0k1 | 47.0 | 1.1979 | 0.0000 | 0.0255 | 252254/9442 |
| O+M@o0k3 | 44.0 | 1.1949 | 0.0000 | 0.0272 | 252254/9293 |
| O+M@o0k5 | 47.0 | 1.1904 | 0.0000 | 0.0253 | 252254/9070 |
| O0 | 195.0 | 7.4598 | 0.0000 | 0.0383 | 1581804/61880 |
| P+M | 219.0 | 0.0000 | 0.0000 | 0.0000 | 0/0 |
| P0 | 218.0 | 0.0000 | 0.0000 | 0.0000 | 0/0 |
| X+M | 201.0 | 0.0000 | 0.0000 | 0.0000 | 3528745/37665 |
| X0 | 191.0 | 0.0000 | 0.0000 | 0.0000 | 3549001/39102 |

## Cost-effectiveness (A24.2 key secondary; planning cost only, descriptive, 95 % family-cluster bootstrap CIs)

Assumptions (a24.2-cost-1): electricity 0.2 USD/kWh on GPU joules (simple, approximate; CPU not measured); hardware 399.0 USD / 6000.0 h x planning hours (basis: amendment:A33; reference price, prereg A33 (2026-10-14, Arash decision 8; p...). Codex (X) CLI planning is not priced (subscription); their allocated preparation USD is shown. planning cost only (executor / simulator excluded); CIs describe task-sampling uncertainty conditional on these prices and assumptions.
**Hardware price disclosure (A33):** A33: planning time is measured on the run laptop (RTX 4060, 8 GB). The headline hardware allocation combines those measured hours with the USD 399 reference-device price. The energy term is the laptop GPU's energy during planning (energy_method c5-simple-1: nvidia-smi average power x planning time; GPU only, approximate). A Gemma plan row without an energy figure stays UNKNOWN and blocks the A24.2 statement. Sensitivity: the hw_time_3x row (3 x planning hours, laptop energy held fixed; factor 3 = an approved sensitivity assumption, not a measured slowdown); the reading rule uses the headline view only
**Energy (C5, c5-simple-1):** Planning energy is a simple, approximate figure: the GPU's average power draw reported by nvidia-smi (about 10 readings per second) during each Gemma planning call, multiplied by the planning time and priced at 0.20 USD/kWh. Only the GPU is counted (CPU and the rest of the laptop are not), so it understates total energy; idle power is included. The 10 practice rows that ran before the meter existed are estimated from the measured rate and flagged. Unmeasured Gemma rows: ESTIMATED at the measured Gemma rate (flagged) for practice rows, claim headline, under arash-decision:2026-10-14.
Measured Gemma rate (C5 unmeasured_estimate): 37.40 W over 1210 measured rows / 995.25 s (min 20); active terms: {'claim': 'headline', 'eligible': ['practice'], 'rate': 'measured_rate', 'rows': {'journal': 'docs/final-test/matrix/practice-v2/practice.jsonl', 'journal_sha256': 'ab84215a35cc082a770815d630949eeedbf39d8eaf63fe18ce47f9d846d31dda', 'row_ids': ['1d0a3b6e730c510041a8606505bd3c798e2eed7b61baed949ad1898d21992f38', '2178d14b21d0856da776b465ce620d6870cd01623b360bf3bfb8043f1a32e372', '2dec82fa8cf40f5de41dc4fdff8223d44f9f9c8ac41fef7ef56a026ea83a9774', '2efa5590a87ebd75a498bd5de13e20b65aabdc971d16ef8ce28504a7364f9ea4', '3cbd72acca05775964fb3ed3ed4850effb89c389372c89245442bc1c9853dcc1', '777e908a5ac8254444bbb53f62bd1a2b65af4445c1fa1f0b10e4f4618c26d608', '7d361af12284598cbe8f72ed6b34312580e157f6de72d3d49f18bcca307753e5', 'b7fc99c4a11b691fb864cd082ba69caa2f1c22593ede037e4223148cc44a81f4', 'bd9f087c5d208556af0b907ff72ef1f99e31aaae4e745db750c56a25cf31ef66', 'dba5a9f8d647f60bac20b7a18a1d38ebbb2b2408095c04e0a3e372e3d907fd62']}}.
Energy join (strict=True): 1210 Gemma plan rows, 1210 measured, 0 without a reading, 0 without an energy row, 0 ambiguous, 0 invalid; inline energy fields stripped from 0 rows; idle floor GPU 2.579 W.

Cohort: 250 presentations (same item weights in every outcome).

| arm | kind | cost state | SR (item-weighted) | USD / success (headline) | mean USD / attempted task (headline) | electricity only / task | hardware only / task | 3x hardware time / task | Pareto |
|---|---|---|---|---|---|---|---|---|---|
| G+FS | gemma | unknown (0 presentation(s) without a plan row, 250 with an unmeasured energy / time component; practice / preparation cost unmeasured or missing) | 55.6% [46.1%, 65.0%] | n/a | n/a | n/a | n/a | n/a |  |
| G+M | gemma | unknown (0 presentation(s) without a plan row, 250 with an unmeasured energy / time component; practice / preparation cost unmeasured or missing) | 68.4% [59.3%, 76.8%] | n/a | n/a | n/a | n/a | n/a |  |
| G+M/p | gemma | unknown (250 presentation(s) without a plan row, 0 with an unmeasured energy / time component) | 70.0% [61.4%, 77.8%] | n/a | n/a | n/a | n/a | n/a |  |
| G+M@o0k1 | gemma | unknown (60 presentation(s) without a plan row, 0 with an unmeasured energy / time component) | 58.3% [38.2%, 75.7%] | n/a | n/a | n/a | n/a | n/a |  |
| G+M@o0k3 | gemma | unknown (60 presentation(s) without a plan row, 0 with an unmeasured energy / time component) | 60.0% [38.9%, 77.7%] | n/a | n/a | n/a | n/a | n/a |  |
| G+M@o0k5 | gemma | unknown (60 presentation(s) without a plan row, 0 with an unmeasured energy / time component) | 60.0% [38.9%, 77.7%] | n/a | n/a | n/a | n/a | n/a |  |
| G0 | gemma | costed | 54.0% [44.5%, 63.4%] | 3.53e-05 [2.64e-05, 4.75e-05] | 1.91e-05 [1.59e-05, 2.28e-05] | 1.87e-06 [1.47e-06, 2.34e-06] | 1.72e-05 [1.44e-05, 2.05e-05] | 5.34e-05 [4.47e-05, 6.38e-05] | efficient |
| O+FS | api | unknown (0 presentation(s) without a plan row, 250 with an unmeasured energy / time component; practice / preparation cost unmeasured or missing) | 78.8% [72.6%, 84.4%] | n/a | n/a | n/a | n/a | n/a |  |
| O+M | api | unknown (0 presentation(s) without a plan row, 250 with an unmeasured energy / time component; practice / preparation cost unmeasured or missing) | 82.0% [76.1%, 87.3%] | n/a | n/a | n/a | n/a | n/a |  |
| O+M/p | api | unknown (250 presentation(s) without a plan row, 0 with an unmeasured energy / time component) | 82.4% [76.3%, 87.7%] | n/a | n/a | n/a | n/a | n/a |  |
| O+M@o0k1 | api | unknown (60 presentation(s) without a plan row, 0 with an unmeasured energy / time component) | 78.3% [64.5%, 90.9%] | n/a | n/a | n/a | n/a | n/a |  |
| O+M@o0k3 | api | unknown (60 presentation(s) without a plan row, 0 with an unmeasured energy / time component) | 73.3% [59.2%, 87.0%] | n/a | n/a | n/a | n/a | n/a |  |
| O+M@o0k5 | api | unknown (60 presentation(s) without a plan row, 0 with an unmeasured energy / time component) | 78.3% [66.0%, 89.7%] | n/a | n/a | n/a | n/a | n/a |  |
| O0 | api | costed | 78.0% [69.9%, 84.7%] | 0.024 [0.0219, 0.0272] | 0.0187 [0.0182, 0.0193] | 0.0187 [0.0182, 0.0193] | 0 [0, 0] | 0.0187 [0.0182, 0.0193] | efficient |
| P+M | api | unknown (250 presentation(s) without a plan row, 0 with an unmeasured energy / time component; practice / preparation cost unmeasured or missing) | 87.6% [82.5%, 91.9%] | n/a | n/a | n/a | n/a | n/a |  |
| P0 | api | unknown (250 presentation(s) without a plan row, 0 with an unmeasured energy / time component) | 87.2% [82.0%, 91.9%] | n/a | n/a | n/a | n/a | n/a |  |
| X+M | codex | incomplete (CLI planning unpriced: subscription; known allocated preparation USD reported) | 80.4% [73.8%, 86.3%] | n/a | n/a | n/a | n/a | n/a |  |
| X0 | codex | incomplete (CLI planning unpriced: subscription; known allocated preparation USD reported) | 76.4% [69.9%, 82.4%] | n/a | n/a | n/a | n/a | n/a |  |

G+M@o0k1: 190 cohort presentation(s) missing (not dropped silently).

G+M@o0k3: 190 cohort presentation(s) missing (not dropped silently).

G+M@o0k5: 190 cohort presentation(s) missing (not dropped silently).

O+M@o0k1: 190 cohort presentation(s) missing (not dropped silently).

O+M@o0k3: 190 cohort presentation(s) missing (not dropped silently).

O+M@o0k5: 190 cohort presentation(s) missing (not dropped silently).

X+M: known allocated preparation USD n/a; CLI planning unpriced, so its total is incomplete (never zero).

X0: known allocated preparation USD 0; CLI planning unpriced, so its total is incomplete (never zero).

Matched G+M / O0 presentations: 250 (125 items). R = SR(G+M) / SR(O0) = 0.877 [0.764, 1.010].
- C (headline) = mean USD per attempted task G+M / O0 = unknown (CI not obtainable: a matched presentation has no plan row or an unmeasured energy / time component)
- C (electricity_only) = mean USD per attempted task G+M / O0 = unknown (CI not obtainable: a matched presentation has no plan row or an unmeasured energy / time component)
- C (hw_time_3x) = mean USD per attempted task G+M / O0 = unknown (CI not obtainable: a matched presentation has no plan row or an unmeasured energy / time component)

**A24.2 reading rule (lower R >= 0.80 and upper C <= 0.10, headline view): not met** (C: valid CI bounds not obtainable (a matched presentation has no plan row or an unmeasured energy / time component)).

| arm | planning s / presentation (mean / median / p90) | planning hours | joules (Gemma, scored) |
|---|---|---|---|
| G+FS | 0.88 / 0.77 / 1.55 | 0.061 |  |
| G+M | 0.76 / 0.64 / 1.34 | 0.053 |  |
| G+M/p | unknown | | |
| G+M@o0k1 | unknown | | |
| G+M@o0k3 | unknown | | |
| G+M@o0k5 | unknown | | |
| G0 | 0.93 / 0.76 / 1.71 | 0.065 | 8417.5 |
| O+FS | 4.42 / 3.93 / 6.32 | 0.307 |  |
| O+M | 4.02 / 3.75 / 5.63 | 0.279 |  |
| O+M/p | unknown | | |
| O+M@o0k1 | unknown | | |
| O+M@o0k3 | unknown | | |
| O+M@o0k5 | unknown | | |
| O0 | 4.18 / 3.95 / 6.19 | 0.290 |  |
| P+M | unknown | | |
| P0 | unknown | | |
| X+M | 16.30 / 16.37 / 21.03 | 1.132 |  |
| X0 | 16.82 / 16.11 / 21.69 | 1.168 |  |

Latency includes every attempt, repair and failed attempt of the scored cell. Pareto plot: pareto.svg beside analysis.json.

## A36 amended subset analysis (a36-r2; SIGNED (Arash, 2026-10-15; signatures.json key A36; A36-DRAFT.md r2); binds the locked test once the freeze records it)

Primary = paired common subset (a flagged presentation is unavailable in that arm and leaves both arms of every contrast with it); registered = A21 (infra_exec = SR 0, in every denominator). Agreement between the two does not show the exclusions were unbiased. h1-rule-1's validation did not cover this selection step.

| arm | core cells | flagged | share | reset-limited | SR primary (items / pres) | SR registered (items / pres) |
|---|---|---|---|---|---|---|
| G+FS | 250 | 15 | 6.0 % | yes | 0.598 (123 / 235) | 0.556 (125 / 250) |
| G+M | 250 | 20 | 8.0 % | yes | 0.746 (122 / 230) | 0.684 (125 / 250) |
| G+M/p | 250 | 16 | 6.4 % | yes | 0.756 (123 / 234) | 0.700 (125 / 250) |
| G+M@o0k1 | 60 | 0 | 0.0 % | no | 0.583 (30 / 60) | 0.583 (30 / 60) |
| G+M@o0k3 | 60 | 0 | 0.0 % | no | 0.600 (30 / 60) | 0.600 (30 / 60) |
| G+M@o0k5 | 60 | 0 | 0.0 % | no | 0.600 (30 / 60) | 0.600 (30 / 60) |
| G0 | 250 | 16 | 6.4 % | yes | 0.589 (124 / 234) | 0.540 (125 / 250) |
| O+FS | 250 | 17 | 6.8 % | yes | 0.846 (123 / 233) | 0.788 (125 / 250) |
| O+M | 250 | 22 | 8.8 % | yes | 0.902 (123 / 228) | 0.820 (125 / 250) |
| O+M/p | 250 | 17 | 6.8 % | yes | 0.885 (122 / 233) | 0.824 (125 / 250) |
| O+M@o0k1 | 60 | 0 | 0.0 % | no | 0.783 (30 / 60) | 0.783 (30 / 60) |
| O+M@o0k3 | 60 | 0 | 0.0 % | no | 0.733 (30 / 60) | 0.733 (30 / 60) |
| O+M@o0k5 | 60 | 0 | 0.0 % | no | 0.783 (30 / 60) | 0.783 (30 / 60) |
| O0 | 250 | 19 | 7.6 % | yes | 0.837 (123 / 231) | 0.780 (125 / 250) |
| P+M | 250 | 21 | 8.4 % | yes | 0.955 (123 / 229) | 0.876 (125 / 250) |
| P0 | 250 | 16 | 6.4 % | yes | 0.934 (122 / 234) | 0.872 (125 / 250) |
| X+M | 250 | 24 | 9.6 % | yes | 0.886 (123 / 226) | 0.804 (125 / 250) |
| X0 | 250 | 18 | 7.2 % | yes | 0.817 (123 / 232) | 0.764 (125 / 250) |

H1 (amended subset): **non-inferiority unresolved (inconclusive): guard failed (K = 17 < 20 families with a discordant presentation)**; registered: non-inferiority unresolved (inconclusive); retained pres / items / families / K 223 / 119 / 42 / 17; reset-limited: G+M, O0

| id | contrast | estimate (amended) | estimate (registered) | p Holm | reject (amended) | reject (registered) | pres / items / fam / K | reset-limited |
|---|---|---|---|---|---|---|---|---|
| H2a | G+M - G0 | +16.2 pp | +14.4 pp | 0.0006 | True | True | 223 / 120 / 42 / 20 | G+M, G0 |
| H2c | O+M - O0 | +5.0 pp | +4.0 pp | 0.0155 | True | False | 219 / 120 / 42 / 5 | O+M, O0 |
| H3a | O+M - G+M | +14.2 pp | +13.6 pp | 0.0008 | True | True | 221 / 120 / 42 / 18 | O+M, G+M |

Exploratory (raw p, no Holm, no confirmatory conclusion):

| id | contrast | estimate (amended) | estimate (registered) | pres / items / fam / K | reset-limited |
|---|---|---|---|---|---|
| H4a | G+M - G+FS | +14.3 pp | +12.8 pp | 225 / 119 / 42 / 18 | G+M, G+FS |
| H4b | O+M - O+FS | +4.5 pp | +3.2 pp | 222 / 121 / 42 / 7 | O+M, O+FS |
| H5a | X+M - X0 | +5.8 pp | +4.0 pp | 218 / 120 / 42 / 8 | X+M, X0 |
| H5b | X+M - G+M | +13.3 pp | +12.0 pp | 220 / 120 / 42 / 17 | X+M, G+M |
| H5c | G+M - X0 | -8.4 pp | -8.0 pp | 224 / 119 / 42 / 21 | G+M, X0 |

Equivalence G+M vs O+M (exploratory): amended equivalence not established; registered equivalence not established; retained 221 / 120 / 42 / 18; reset-limited: G+M, O+M

Differences between the amended and the registered analysis: H1 wording: amended 'non-inferiority unresolved (inconclusive): guard failed (K = 17 < 20 families with a discordant presentation)' vs registered 'non-inferiority unresolved (inconclusive)'; H2c reject: amended True vs registered False

## How to read the X0 / X+M results (A21 E)

X used codex-cli 0.162.1 configured for gpt-6-sol at high reasoning effort (the CLI-configured model; the model the server finally answered with was unobservable). Built-in tools remained available: isolation monitoring covered the events the CLI emitted, not invisible calls. The isolation probe (9 calls; 6 canary-read calls produced 24 visible tool events, mostly MCP resource listing) returned no secret token. Primary results include calls flagged for tool events; paired exclusions are reported separately. CLI scaffolding (about 10.6 k input tokens per call) and subscription execution are part of this comparator; 'free' means no incremental API expenditure only.

## Codex CLI systems (exploratory, separate)

exploratory; model + CLI scaffold; never pooled with API arms; not in H1 or the Holm family

No Codex arms in this journal.
