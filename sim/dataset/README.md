# dataset/

The bilingual (Persian + English) evaluation dataset for the three-level command hierarchy. This
directory will hold JSONL files of command records, each tagged with its level — **L1** atomic
(one command → one function, scored by Success Rate), **L2** sequential (ordered chain of atomic
functions, scored by Procedural Correctness / normalized Levenshtein), and **L3** abstract goal
(needs reasoning + decomposition into a new plan, scored by Task Decomposition Score / F1 over
action sets). Each record carries the natural-language command (fa + en), its level, and the
ground-truth action(s) the eval harness compares against. The metrics are defined in `sim/eval.py` (METRICS).
