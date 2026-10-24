# Observability - total logging, action->database causality, and the trace viewer

This is the real implementation of the one promise CLAUDE.md already makes about the self-improving loop:

> *"Self-improving loop writes are auditable. When the agent edits its own knowledge base, log what/why so the
> proof-of-concept can be evaluated."*

Until now that promise was kept by two flat text files (`sim/knowledge_base_audit.log`, `sim/memory_audit.log`)
and one JSONL journal (`sim/loop_reports.jsonl`). They record *that* a write happened and a free-text `why` -
but nothing in them links a knowledge-base row back to the command, the LLM call, or the AI2-THOR action that
caused it. The author asked for exactly that link:

> *"create a system that logs everything, and also I can check which actions resulted in which additions to the
> database, and see the database itself visually, and more."*

Which decomposes into three deliverables, all built:

| # | Deliverable | Where it lives |
|---|---|---|
| **D1** | **Total logging** - a durable, append-only event log of every command, retrieval, LLM call, plan step, executed AI2-THOR primitive, perception run, feedback event, evaluation launch and knowledge/memory write | `sim/trace.py` -> `sim/trace.db` |
| **D2** | **Causality** - every database write attributable to the activity that caused it, navigable in **both** directions by a single indexed query | the `writes` ledger + the `edges` PROV graph |
| **D3** | **Visual browser** - the databases themselves (tables, rows, provenance), the event timeline, the span waterfall and the causal graph | `sim/web/trace.{html,css,js}` + the `/trace*` routes in `sim/webapp.py` |

Verified 2026-08-16: pure gate **72/72 checks** (`sim/_trace_selftest.py`), with the four protected phase
self-tests unchanged (`_phase7` 37/37, `_phase6` OK, `_phase6a` OK, `_eval` 64/64). No AI2-THOR, no LLM, no
network, no third-party package: `sim/trace.py` is stdlib-only (`sqlite3`, `json`, `threading`, `contextvars`,
`queue`, `hashlib`, `time`, `os`, `sys`, `contextlib`, `re`, `logging`).

The research backing - what mature systems model and why this schema is standard rather than idiosyncratic - is
`docs/observability-research.md`; the buildable specification this implements is `docs/observability-design.md`.

---

## 1. The design in one paragraph

`sim/trace.py` is an **observability sidecar**: the instrumented modules import it, it never imports them (the
dependency arrow points one way, so a trace failure cannot cascade). It writes to its **own** SQLite file,
`sim/trace.db`, in WAL mode - never to `memory.db`, never to the KB JSON - so the observer can never stall or
corrupt the observed. Every call site is wrapped in a swallow-everything guard mirroring `knowledge._audit`'s
IO guard, so **no hook can raise into the agent** (self-test T-29 runs the whole stack against a tracer that
raises on every call, and every function still returns its original value). Turning tracing off with
`TRACE_ENABLED=0` produces byte-identical stores and no `trace.db` at all (T-27).

The store is an **event-sourced** log: `events` is append-only and is the source of truth; the knowledge base
and `memory.db` are reframed as *projections* of it (`docs/observability-research.md` §3). Nothing is rebuilt
from events at runtime - the KB is live research data - so replay is a *verification* tool, not the boot path.

---

## 2. The trace data model

`sim/trace.db`, schema version 1, seven tables and 25 named indexes (asserted whole by T-1). The vocabulary is
borrowed, not invented: OpenTelemetry for the execution half, W3C PROV-O for the provenance half
(`docs/observability-research.md` §1, §4).

### 2.1 Concept mapping

| Table | OTel / product concept | PROV-O concept |
|---|---|---|
| `runs` | a trace (all spans sharing a trace id) + Langfuse's session level | `prov:Bundle` - a named, citable set of provenance descriptions |
| `spans` | an OTel span; `kind` is OpenInference's `openinference.span.kind`; `level` is Langfuse's verbosity knob | `prov:Activity` |
| `events` | OTel span *events* ("a structured log message at a point inside a span") | the immutable evidence base |
| `artifacts` | (nothing - no product models this) | `prov:Entity` |
| `writes` | (nothing) | the denormalised `wasGeneratedBy` ledger, with verdict + counter deltas |
| `edges` | OTel span *links* ("associate spans, implying a causal relationship") | the PROV relations themselves |
| `meta` | - | store bookkeeping (`schema_version`, `created_ts`, `trace_api_version`, `seq_hwm`) |

`artifacts` + `writes` + `edges` are the half **no LLM-observability product has**: LangSmith, Langfuse,
Phoenix, Braintrust, Weave and OpenLLMetry all trace *execution*, none traces an agent writing back into its
own long-term memory (`docs/observability-research.md` §2.1). That is contribution (c) of this thesis, so the
schema has to model it.

### 2.2 `runs` - PROV Bundle / OTel trace

| Field | Meaning |
|---|---|
| `run_id` | 32 lowercase hex, W3C trace-id format (`os.urandom(16).hex()`) |
| `parent_run_id` | set for an `AUTO_RETRY` re-entry and for an eval subprocess inheriting `TRACE_RUN_ID` |
| `kind` | `command` \| `eval` \| `feedback` \| `perception` \| `maint` \| `bench` \| `test` |
| `origin` | process discriminator: `webapp` \| `eval` \| `bench` \| `cli` \| `test` (inferred from `sys.argv[0]`, overridable with `TRACE_ORIGIN`) |
| `started_ts` / `ended_ts` | `prov:startedAtTime` / `prov:endedAtTime` |
| `status` | `UNSET` \| `OK` \| `ERROR`; `error` carries the terminal message |
| `scene`, `level`, `lang` | promoted to columns because CLAUDE.md makes the kitchen, the L1/L2/L3 hierarchy and the Persian/English axis first-class - every thesis query filters on them |
| `command` | the natural-language command verbatim (Persian stored as-is; T-47 asserts byte-identical round-trip, no `\uXXXX` escaping) |
| `arm`, `decision_id` | the bandit arm and `bandit_decisions.id`, so the trace joins to the Phase-6a bandit for free |
| `agent_id` | `gen_ai.agent.id` - the planner model, e.g. `gemma3:4b` |
| `seed`, `config_json`, `config_hash` | the frozen ablation snapshot; two runs with the same `config_hash` are comparable **by construction**, which is what the Phase-10 ablation matrix needs to assert |
| `code_version` | git rev, best-effort |
| `n_spans`, `n_events`, `n_writes` | denormalised counters filled at `end_run` (cheap UI) |

### 2.3 `spans` - PROV Activity / OTel span

`span_id` (16 hex, W3C parent-id format), `run_id`, `parent_span_id` (NULL => root; this is
`prov:wasInformedBy`), `seq` and `depth` (precomputed for the waterfall), `name`, `op`, `kind`, `step_ix`,
`start_ts` / `end_ts` / `duration_ms`, `status`, `level`, `error`, `scene`, `actor`, `input_json`,
`output_json`, `attrs_json`, `repeat`.

- **`kind`** (closed enum, what makes the UI filterable):
  `AGENT` · `LLM` · `RETRIEVER` · `TOOL` · `ACTION` · `PERCEPTION` · `LOOP` · `FEEDBACK` · `EVALUATOR` ·
  `MAINT`. Five of these ten are **not** in OpenInference's `openinference.span.kind` enum (LLM, EMBEDDING,
  CHAIN, RETRIEVER, RERANKER, TOOL, AGENT, GUARDRAIL, EVALUATOR, PROMPT): `ACTION` (a raw AI2-THOR primitive),
  `PERCEPTION`, `LOOP`, `FEEDBACK` and `MAINT` (prune / clear_nav / reindex). Conversely EMBEDDING, CHAIN,
  RERANKER, GUARDRAIL and PROMPT have no instance here. (An earlier revision said "two additions" - `ACTION` and
  `MAINT` only; that was wrong, and the thesis §3.9 carries the corrected count.)
- **`op`** = `gen_ai.operation.name`: `invoke_agent`, `chat`, `execute_tool`, `retrieval`, `embeddings` come
  from the GenAI semantic conventions verbatim; `thor_action`, `close_loop`, `observe`, `feedback`,
  `maintenance`, `scene_reset`, `evaluate` are project terms.
- **`input_json` / `output_json`** are separate columns rather than an attribute bag, because the primary UI
  drill-down is "what went in, what came out" - the one thing every product in the survey converged on. Both
  are capped at `TRACE_MAX_BLOB` (8 KB) with a truncation marker (T-34).
- **`step_ix`** rides the ambient context frame, so a write made three call levels below a plan step still
  lands with its step index and no signature had to change (T-57).
- **`repeat`** > 1 when a contiguous run of identical navigation primitives was collapsed into one span
  (T-33) - an L3 command can otherwise emit hundreds of `MoveAhead` spans.
- **`attrs_json`** uses `gen_ai.*` where a term exists, plus three local namespaces: `thor.*`
  (`thor.scene`, `thor.action`, `thor.object_type`, `thor.object_id`, `thor.success`, `thor.error`),
  `agent.*` (`agent.level`, `agent.lang`, `agent.arm`, `agent.rag_enabled`, `agent.loop_enabled`), `kb.*`
  (`kb.write.kind`, `kb.write.action`, `kb.entry.id`, `kb.why`, `kb.dedup_key`). Keeping the standard prefix
  for standard things and an obvious local prefix for embodied specifics is exactly how OpenInference layers
  on OTel.

### 2.4 `events` - the append-only source of truth

`seq` (global monotonic cursor and the viewer's paging key), `event_id` (sha256 content hash, `UNIQUE`, so a
double-instrumented or replayed call inserts one row - T-6), `ts`, `run_id` (NULL => **orphan**), `span_id`,
`type`, `actor`, `level`, `scene`, `summary` (one line, <=200 chars, safe to render densely), `payload_json`
(full detail; only served by `/trace_event`), `origin`, `pid`, `thread`.

`seq` is minted from a **disjoint 4096-value block** reserved off a `meta.seq_hwm` high-water mark inside a
`BEGIN IMMEDIATE` transaction, so two processes (webapp + eval subprocess) can never mint the same primary key
and no event is ever silently discarded by `INSERT OR IGNORE`. Within a process it stays gapless and monotonic;
across a re-open it does not restart at `MAX(seq)+1` (block tails are skipped by design). T-5, T-37.

Emitted types (21 today):

| Type | Emitted at | Level |
|---|---|---|
| `run.started` / `run.ended` | `start_run` / `end_run` | DEFAULT |
| `command.received` | `_plan_and_run`, before planning | DEFAULT |
| `retrieval.performed` | after the retrieval span closes | DEFAULT |
| `llm.call` | `record_llm`, around the planner call | DEFAULT |
| `plan.produced` / `plan.repaired` | after `parse_plan` / after each repair | DEFAULT / WARNING |
| `step.started` / `step.finished` | `_tracked_step` enter / exit | DEFAULT |
| `action.executed` | the `ThorAgent.do()` hook | DEBUG |
| `action.skipped_summary` | once per run: how many `Pass`/`Done` were filtered | DEBUG |
| `scene.reset` | the `reset()` / `set_sensors()` hooks | WARNING |
| `kb.write` | every `knowledge.py` write path | DEFAULT |
| `memory.write` | every `memory.py` writer | DEFAULT |
| `file.write` | `loop_reports.jsonl` append, eval result file | DEBUG |
| `feedback.received` | rating / teaching / demo / perception label | DEFAULT |
| `perception.run` | scene scan and YOLO detect pass | DEBUG |
| `loop.closed` | `improve.close_loop` return | DEFAULT |
| `audit.line` | mirror of every legacy `_audit()` line | DEBUG |
| `console.log` | tee of `webapp._log` (the volatile 80-line ring buffer, made durable) | DEBUG |
| `trace.warning` | the tracer degrading (unwritable store, dropped events) | WARNING |

`eval.scored` is **reserved in the schema but never emitted** - see the follow-up in §6.

### 2.5 `artifacts` - PROV Entity

One row per durable thing the agent produced, across *all* stores in one uniform id space:

```
artifact_id ::= <store> ':' <kind> ':' <safe_ref>          e.g.  kb:lesson:1
                                                                 mem:episode:2
                                                                 mem:failure:pickup|pickupobject failed mug not visible|Mug
                                                                 file:report:loop_reports.jsonl%231
```

Fields: `store` (`kb`|`mem`|`file`|`eval`|`bandit`), `kind`, `ref` (the untruncated natural key),
`created_seq`, `created_ts`, `updated_ts`, `content_hash`, `summary`, `source`, `scope`, `confidence`,
`revision`, `occurrences`, `dedup_key`, `alive`, `n_causes`.

Three properties worth stating:

- **`source` IS `prov:wasAttributedTo`.** The existing per-entry `source ∈ {human, demo, environment, prior}`
  needed no change - `trace.py` maps it to a PROV agent (`_SOURCE_ACTOR`) and emits the edge. Likewise the
  existing `ts` is `prov:generatedAtTime`. The code had independently arrived at a *subset* of PROV; what was
  missing was only the identifier that turns those loose fields into a graph.
- **`dedup_key` is computed, never persisted by the stores.** It records *why* a write merged into entry #N
  (e.g. `pick up the mug|pickupobject failed mug not visible` for `upsert_lesson`).
- **Rows are never removed.** A prune or delete sets `alive=0` and writes a `wasInvalidatedBy` edge, keeping
  the entry's whole prior history (T-23).
- **Artifact ids are namespaced per store file.** A write coming from a knowledge base other than the one
  being browsed becomes `kb@1a2b3c4d:lesson:1`, because KB ids restart at 1 after every reset and a shared id
  would fuse unrelated histories (T-58).

### 2.6 `writes` - the auditable ledger (this is D2)

The operational answer to both of the author's questions in **one indexed row read**:

`id`, `event_seq`, `run_id`, `span_id`, `step_ix`, `ts`, `store`, `table_or_kind`, `pointer`, `artifact_id`,
`verdict`, `dedup_key`, `before_json`, `after_json`, `why`, `source`, `audit_line`, `origin`.

- **`verdict`** (closed enum): `insert` · `reinforce` · `update` · `delete` · `noop`. A `reinforce` must be
  distinguishable from an `insert`, otherwise the causality view silently loses every repeat-learning event -
  which is the core Phase-7 evidence.
- **`audit_line`** stores the exact legacy line the same write produced, so the grep-able trail and the
  structured trail can never diverge. The line format itself is unchanged (T-26 asserts every op string).
- **`before_json` / `after_json`** carry the counter/field deltas, not the whole entry. For `prune` this is
  the one hook that records information existing **nowhere else on disk**: the raw *and* decayed confidence of
  each deleted spatial belief (T-52).
- Writes are **never dropped**. Under queue saturation the tracer evicts DEBUG events first and never a
  ledger op - the auditability evidence itself is not droppable (T-54).

### 2.7 `edges` - the PROV graph

`src`, `dst`, `rel`, `run_id`, `seq`, `ts`, `note`, with `UNIQUE(src, dst, rel)` so `INSERT OR IGNORE` makes
edge writing idempotent. Node ids are namespaced (`run:` | `span:` | `art:` | `agent:`) and may never contain
`/` - that character delimits the cycle-guard path in the recursive query.

Relations used: `wasGeneratedBy`, `used`, `wasDerivedFrom`, `wasRevisionOf`, `wasAttributedTo`,
`wasAssociatedWith`, `wasInformedBy`, `wasInvalidatedBy`. An unknown relation is dropped (T-21).

**Orientation is non-negotiable**: following PROV, an edge points from the dependent thing to the thing it
depends on. So `src -> dst` walks *backwards* in causality (ancestors: "what produced this?") and `dst -> src`
walks *forwards* (descendants: "what did this produce?"). Both are the same recursive CTE with the join
flipped - which is precisely the bidirectional navigation the author asked for.

The traversal carries a `'/'`-delimited visited path and excludes nodes already in it. This is not decoration:
*entry A is retrieved by a run that then reinforces A* is the normal operation of the self-improving loop, and
`UNION`'s built-in cycle protection deduplicates whole **rows**, not nodes - the moment the row carries the
`depth` column the lineage view needs, every revisit is a "new" row and the protection stops working. T-18
asserts the guarded form terminates and returns strictly fewer rows than the naive one.

No RDF/OWL: PROV-O is used as a vocabulary and a correctness check, stored relationally. A `rel` column holding
the literal string `wasGeneratedBy` is enough to claim conformance to the data model, and
`GET /trace_prov?run_id=…|node=…` exports PROV-JSON so a reviewer can load the graph into a standard PROV tool
(thesis appendix material).

### 2.8 Why both `writes` and `edges`

| Question | Answered by | Cost |
|---|---|---|
| "What did **this action** write?" | `writes WHERE span_id=?` (`ix_writes_span`) - `trace.span_effects()` | one index seek |
| "What produced **this KB entry**?" (direct causes, verdicts, deltas) | `writes WHERE store/table/pointer` (`ix_writes_ptr`) - `trace.artifact_causes()` | one index seek |
| "…transitively - through the episodes it was distilled from, the lessons the planner used, the agents responsible?" | the `edges` ancestor CTE - `trace.lineage(node, direction='up')` | ~depth index scans |
| "What did this entry go on to influence?" | the `edges` descendant CTE - `trace.lineage(node, direction='down')` | ~depth index scans |

Consistency between the two is guaranteed because both are written by the **same** function,
`trace.record_write()` - never by callers independently - and asserted by T-24 (after 200 randomised writes,
every `artifact_id` resolves, every `n_causes` matches `COUNT(*)`, and every insert has a `wasGeneratedBy`
edge).

---

## 3. Instrumentation map

### 3.1 Which module emits what

| Module | Call site (file:line) | Emits |
|---|---|---|
| `knowledge.py` | `_audit()` `:185` | `audit.line` (the legacy line, mirrored) |
| | `_add()` `:205` | `kb.write` **insert** for procedure/lesson/fact + artifact + `wasGeneratedBy` + `wasAttributedTo` |
| | `reinforce()` `:267` | `kb.write` **update** (the `uses` bump that previously had *no audit line at all* - an invisible mutation of research data; it now also writes a `REINFORCE procedure` audit line) |
| | `upsert_procedure()` `:316` | `kb.write` **reinforce** with the dedup key |
| | `upsert_lesson()` `:358`, `:372` | `kb.write` **reinforce**, and **update** for the `error_key` patch (a second, previously unaudited `_save()`) |
| `memory.py` | `_audit()` `:201` | `audit.line` |
| | `remember_location()` `:249` | `memory.write` spatial **insert** |
| | `observe_location()` `:320`, `:355` | spatial **insert** / **reinforce** |
| | `record_failure()` `:414` | failures **insert**/**reinforce** keyed on the signature |
| | `prune()` `:483`, `:489` | one **delete** write *per removed id*, each with raw + decayed confidence |
| | `log_episode()` `:520` | episodes **insert** |
| | `set_rating()` `:588`, `:592` | episodes **update** per rated id + `feedback.received` |
| | `add_perception_label()` `:645` | perception_labels **insert** |
| | `mark_cell()` `:676` | nav_cells **update** with before/after (the `INSERT OR REPLACE` previously destroyed prior state silently) |
| | `clear_nav()` `:718`, `:723` | nav_cells **delete** |
| `improve.py` | `close_loop()` `:376` | the `LOOP`/`close_loop` span (`actor=system:improve`) |
| | `_close_loop_body()` `:415` | `record_use` per retrieved id -> `prov:used` grounding edges |
| | `_close_loop_body()` `:488`, `:496` | `file.write` for the journal line (addressable as `file:report:<journal>#<line>`) + `loop.closed` |
| | `_link_episodes()` `:128` | `wasDerivedFrom` KB entry -> the episodes of the steps it was distilled from |
| `feedback.py` | `log_step()` `:107`, `:128` | the `FEEDBACK` span; `wasDerivedFrom` failure-counter -> failing episode |
| | `observe_scene()` `:155`, `:181` | the `PERCEPTION` span + one summary `memory.write` |
| `webapp.py` | `_install_trace_hooks()` `:103`/`:112`, `main()` `:2607` | installs the `ThorAgent.do/reset/set_sensors` hooks: one `ACTION` span per raw AI2-THOR primitive, `scene.reset` per reset |
| | `_log()` `:144` | `console.log` |
| | `_plan_and_run()` `:1203`-`:1368` | `start_run(kind='command')`, `command.received`, the `RETRIEVER` strategy + retrieval spans with `record_use` per injected hit, `retrieval.performed`, `record_llm` (in a `try/finally`, so a failed call is still recorded), `plan.produced`, `plan.repaired`, `annotate_run` (arm / decision_id / level), `end_run` |
| | `_tracked_step()` `:288`-`:343` | one `TOOL` span per plan step with `step_ix`, `step.started` / `step.finished`, and the context hand-off to the worker thread |
| | `_close_loop_after_run()` `:1071` | passes `run_id` / `decision_id` / the executed steps' `episode_ids` into `close_loop` |
| | `_rate_plan()` `:987`, `_store_human_teaching()` `:439`, `_stop_demo()` `:534`, `_store_perception_feedback()` `:757` | a dedicated `kind='feedback'` run + `feedback.received` for every human input |
| | `_scan_job()` `:1381`, detect pass `:1694`/`:1697` | `kind='perception'` run + `perception.run` |
| | `_run_eval_bg()` `:1618`-`:1661` | `kind='eval'` run, and `TRACE_DB` / `TRACE_RUN_ID` / `TRACE_ORIGIN` handed to the eval subprocess through the environment |

The cross-thread problem deserves its own note: `_plan_and_run` opens the run on the **plan thread** while
`_tracked_step` (and therefore every `episodes` / `failures` / `spatial` write of that run) executes on the
**AI2-THOR worker thread**, which a contextvar cannot cross. The webapp takes a `trace.snapshot()` on the plan
thread and re-binds it with `trace.bound(...)` inside the worker. Before this, a review measured **4079 of
4079 writes with `run_id IS NULL`** on the author's own store; T-64 asserts zero orphans for a full command.

### 3.2 Write sites covered

All of `knowledge.py` (`_add` for procedures/lessons/facts, `reinforce`, `upsert_procedure`, `upsert_lesson`
including the `error_key` patch) and all nine `memory.py` writers (`remember_location`, `observe_location`,
`record_failure`, `log_episode`, `set_rating`, `add_perception_label`, `mark_cell`, `clear_nav`, `prune`),
plus the `loop_reports.jsonl` journal append and the eval result file. Because the hooks live **inside the
writers**, every caller is covered transitively - including callers in files this workflow may not edit
(`seed_memory.py`, `control_gui.py`, `eval.py`, the benches). Those writes are recorded with full verdict,
`why`, before/after and audit line; when the caller opened no run they are recorded as **orphans** rather than
dropped, and the orphan counter in the Stats pane is the instrumentation-coverage signal.

### 3.3 Sites NOT covered, and why

| Not covered | Why | Consequence |
|---|---|---|
| `eval.scored` per record (SR/PC/TDS attached to a span) | `sim/eval.py` is owned by another workflow and is being rewritten concurrently | eval runs are traced at *process* granularity (the subprocess inherits `TRACE_RUN_ID`), not per record; the metrics live only in `eval_*.json`. Follow-up F1 |
| `select.py` bandit writes (`bandit_decisions` insert, `reward` update, `bandit_arms` upsert) | not our file | the bandit tables are **browsable** read-only in the Database pane and joined via `runs.decision_id`, but the *causal* `writes` rows are missing. Follow-up F3 |
| `thor_actions.ThorAgent.replay()` | re-issues a recorded macro straight through `self.controller.step`, bypassing the `do()` hook | replayed macro primitives are invisible (the enclosing step span is still recorded). Follow-up F2 |
| Token counts (`gen_ai.usage.*_tokens`) | `llm.OllamaProvider.chat` discards `eval_count` / `prompt_eval_count` / `total_duration` | `gen_ai.usage.input_chars` / `output_chars` stand in. The token attributes are **absent, never zero** - absent means unknown, zero would be a lie in a results table. Follow-up F4 |
| `rag.HybridRAG._save_embeddings` | not our file | the embeddings sidecar is derived state, inferable only from `report["reindexed"]`. Follow-up F5 |
| `control_gui.py` (`_learn`) | legacy Tk path, superseded by the webapp | its writes are captured (the hook is in `knowledge._add`, not the caller) but land as orphans. Follow-up F6 |
| `KnowledgeBase.search()` reads | **deliberate**: only entries actually injected into a planner prompt count as `prov:used`, or the grounding signal becomes meaningless | - |
| per-observation events in `observe_scene` | **deliberate**: a 40-object kitchen would emit 40 near-identical events per step; one summary event + the individual `writes` rows carry the same information at 1/40th the volume | - |
| `memory.py` schema changes | **deliberate non-goal**: no new columns, no new kwargs. `writes.step_ix` for memory rows is `NULL`; the step index lives on the enclosing span, one join away | - |

---

## 4. How to use it

### 4.1 Enable / disable

Tracing is **on by default** and configured entirely by environment variables (or `trace.configure(...)` in a
test):

| Variable | Default | Effect |
|---|---|---|
| `TRACE_ENABLED` | `1` | `0` disables everything: no store file is created, no writer thread starts, every API returns a benign value (T-27) |
| `TRACE_DB` | `sim/trace.db` | store path (relative paths resolve against `sim/`) |
| `TRACE_ORIGIN` | inferred from `sys.argv[0]` | `webapp` \| `eval` \| `bench` \| `cli` \| `test` |
| `TRACE_RUN_ID` | - | cross-process hand-off: a child process files its rows under the parent's run |
| `TRACE_LEVEL` | `DEBUG` | drop events below this level at emit time |
| `TRACE_ASYNC` | `1` | `0` writes synchronously (slower, deterministic - used by tests) |
| `TRACE_SKIP_ACTIONS` | `Pass,Done` | AI2-THOR primitives never recorded |
| `TRACE_COLLAPSE_NAV` | `1` | collapse contiguous identical navigation primitives into one span with `repeat=n` |
| `TRACE_MAX_BLOB` | `8192` | prompt/completion truncation cap |
| `TRACE_QUEUE_MAX` / `TRACE_FLUSH_MS` / `TRACE_BATCH` | `20000` / `250` / `200` | writer-thread backpressure and batching |
| `TRACE_AUDIT_MIRROR` | `1` | `0` drops the `audit.line` mirror (~49% of all events by count; `writes.audit_line` already carries the identical line) |

The store lives at **`sim/trace.db`** (plus `-wal`/`-shm` companions). It is gitignored - it is a binary,
regenerable log, and the research data it *describes* stays where it always was.

### 4.2 Open the viewer

The viewer is served by the existing console, which runs in WSL per CLAUDE.md:

```powershell
wsl -d Ubuntu-22.04 -- pkill -f ai2thor                                  # separate command from the launch
wsl -d Ubuntu-22.04 -- sh -c "cd /mnt/d/Personal/Master*/sim && MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 /home/arash/thor-env/bin/python webapp.py"
```

then open **<http://localhost:8000/trace>** (or the printed WSL IP from Windows). Five views, keyboard-switchable
1-5:

1. **Timeline** - the event feed, virtualised, filterable by type / level / actor / scene / time window / run
   id / free text, with a "live tail" toggle polling `/trace_tail`. Clicking a row opens the full payload and
   the writes that event produced.
2. **Runs** - runs newest-first; opening one gives the span waterfall (CSS-positioned bars, coloured by
   `kind`, indented by `depth`) and its span tree.
3. **Lineage** - the causal graph as inline SVG, laid out in layers straight from the query's `depth`, with
   the direction toggle (ancestors / descendants / both) and a click-to-recentre on any node.
4. **Database** - the 18 browsable tables across all three stores: `kb.procedures|lessons|facts|audit`,
   `mem.episodes|spatial|failures|nav_cells|perception_labels|bandit_arms|bandit_decisions`,
   `trace.runs|spans|events|artifacts|writes|edges`, `file.loop_reports`. Every row that maps to an artifact
   carries an `_artifact_id` so it links straight into its causes and its lineage.
5. **Stats** - counts by event type / level / span kind / write verdict / artifact source, the run tally,
   the **orphan counters**, the most-caused artifacts, and tracer health (queue depth, drops, write errors,
   emit p50/p99).

The read routes are `/trace_events`, `/trace_event`, `/trace_runs`, `/trace_run`, `/trace_entity`,
`/trace_lineage`, `/trace_span_effects`, `/trace_db`, `/trace_db_rows`, `/trace_db_row`, `/trace_stats`,
`/trace_tail`, `/trace_search`, `/trace_prov`. All are read-only, exception-safe, and return their own
`{"error": …}` envelope, so a trace failure can never 500 the console or stall its 400 ms `/status` poll. They
open `memory.db` strictly `mode=ro` and never create the trace store (T-45, T-59), and no user string ever
reaches SQL - an unknown table id or `order_by` column comes back as an error envelope, never as a fallback
(T-46, T-60).

### 4.3 Review the page with no backend: `?demo=1`

**<http://localhost:8000/trace?demo=1>** - and any `file://` load of `sim/web/trace.html` - renders from an
inline fixture generated in the browser, in the exact shapes of the real endpoints. A "demo data" chip appears
in the topbar so a screenshot can never be mistaken for real results. This is how the page is reviewed for
layout/design without WSL, AI2-THOR or a populated store. There is also a "paper" theme toggle next to it,
for taking thesis figures on a light background.

---

## 5. Worked example: a command fails and the agent writes a lesson

Below is **real output**, not a sketch: `sim/_trace_selftest.py` group H (T-51 / T-51b) drives exactly this
scenario, and the rows here were dumped from the resulting store by re-running that fixture with a print
harness. The command is `"pick up the mug"` on `FloorPlan1`, L1, English; step 0 (`go_to`) succeeds, step 1
(`pickup`) fails with `PickupObject failed: Mug|+01.2|-00.3 not visible`; `improve.close_loop` writes a lesson.

**The run** (`runs`):

```
run_id   7e1f9a8dce27341365a46928fcdd4442   kind command   status OK   scene FloorPlan1
level L1   lang en   command "pick up the mug"   agent_id gemma3:4b   config_hash bbc33ccd38a7c20f
n_spans 7   n_events 16   n_writes 6
```

**The spans** (`spans`, in `seq` order) - a `TOOL` span per plan step, the raw AI2-THOR primitive and the
memory write nested underneath it, then loop closure:

```
seq kind      op            name                  parent            step_ix  dur_ms
0   TOOL      execute_tool  execute_tool go_to    -                 0         2.055
1   ACTION    thor_action   MoveAhead             0ba106098274110f  -         0.037
2   FEEDBACK  feedback      log_step go_to        0ba106098274110f  0         1.861
3   TOOL      execute_tool  execute_tool pickup   -                 1         2.979
4   ACTION    thor_action   MoveAhead             d68aeac8e7c6af69  -         0.013
5   FEEDBACK  feedback      log_step pickup       d68aeac8e7c6af69  1         2.905
6   LOOP      close_loop    close_loop            -                 -        14.958
```

**The events** (`events`) - the whole command in 16 rows:

```
seq type              actor           level    summary
1   run.started       system:webapp   DEFAULT  command: pick up the mug
2   command.received  -               DEFAULT  pick up the mug
3   action.executed   env:ai2thor     DEBUG    MoveAhead ok
4   audit.line        env:ai2thor     DEBUG    2026-08-16 17:59:55  [LOG episode]  <environment>  [L1] go_to -> ok in FloorPlan1
5   memory.write      env:ai2thor     DEFAULT  insert episodes #1 — [L1] go_to -> ok
6   action.executed   env:ai2thor     DEBUG    MoveAhead ok
7   audit.line        env:ai2thor     DEBUG    2026-08-16 17:59:55  [LOG episode]  <environment>  [L1] pickup -> fail in FloorPlan1
8   memory.write      env:ai2thor     DEFAULT  insert episodes #2 — [L1] pickup -> fail
9   audit.line        env:ai2thor     DEBUG    2026-08-16 17:59:55  [RECORD failure]  <environment>  pickup [pickupobject failed mug not visible] on Mug in FloorPlan1 (x1)   WHY: pick up the mug
10  memory.write      env:ai2thor     DEFAULT  insert failures #pickup|pickupobject failed mug not visible|Mug
11  audit.line        env:ai2thor     DEBUG    2026-08-16 17:59:55  [ADD lesson]  <environment>  pick up the mug   WHY: loop closure: command failed (seen x1 in failure memory)
12  kb.write          env:ai2thor     DEFAULT  insert lesson #1 — pick up the mug
13  kb.write          env:ai2thor     DEFAULT  update lesson #1 — pick up the mug
14  file.write        system:improve  DEBUG    insert loop_reports.jsonl #loop_reports.jsonl#1 — pick up the mug -> 1 write(s), failure
15  loop.closed       system:improve  DEFAULT  failure (steps) -> insert lesson #1
16  run.ended         -               DEFAULT  OK
```

**The ledger** (`writes`) - six rows, none orphaned, each naming the span that caused it:

```
id  span              step_ix  store table_or_kind        pointer                                    verdict  why
1   a4d73b920e662e56  0        mem   episodes             1                                          insert   (log_step)
2   f357d71965870668  1        mem   episodes             2                                          insert   (log_step)
3   f357d71965870668  1        mem   failures             pickup|pickupobject failed mug not visible|Mug  insert   pick up the mug
4   309885c18fc79a55  -        kb    lesson               1                                          insert   loop closure: command failed (seen x1 in failure memory)
5   309885c18fc79a55  -        kb    lesson               1                                          update   error_key patch: persist the dedup basis after insert
6   309885c18fc79a55  -        file  loop_reports.jsonl   loop_reports.jsonl#1                       insert   loop report journaled
```

Row 5 is worth pausing on: the `error_key` patch was a second, entirely **unaudited** `_save()` before this
work. It now appears with `before_json={"error_key": null}`,
`after_json={"error_key": "PickupObject failed: Mug|+01.2|-00.3 not visible"}` and the computed
`dedup_key = "pick up the mug|pickupobject failed mug not visible"` - i.e. the trace records not just the
mutation but the basis on which the *next* identical failure will merge into this same lesson instead of
creating a new one.

### 5.1 "What produced this lesson?"

`trace.artifact_causes("kb:lesson:1")` - one indexed query, joined to the run and span:

```
verdict  run_id                            run_command      span_name   span_kind  ts                   why
insert   7e1f9a8dce27341365a46928fcdd4442  pick up the mug  close_loop  LOOP       2026-08-16 17:59:55  loop closure: command failed (seen x1 in failure memory)
update   7e1f9a8dce27341365a46928fcdd4442  pick up the mug  close_loop  LOOP       2026-08-16 17:59:55  error_key patch: persist the dedup basis after insert
```

`trace.lineage("art:kb:lesson:1", direction="up")` walks the graph and returns, by shortest depth:

```
depth 0  art:kb:lesson:1                     "pick up the mug"          kb.lesson #1
depth 1  span:309885c18fc79a55               close_loop                 LOOP / close_loop
depth 1  art:mem:episode:1                   "[L1] go_to -> ok"         mem.episode #1
depth 1  art:mem:episode:2                   "[L1] pickup -> fail"      mem.episode #2
depth 1  agent:env:ai2thor                                              PROV agent
depth 2  run:7e1f9a8dce27341365a46928fcdd4442 "pick up the mug"         command run
depth 2  span:a4d73b920e662e56               log_step go_to             FEEDBACK / feedback
depth 2  span:f357d71965870668               log_step pickup            FEEDBACK / feedback
depth 2  agent:system:improve                                           PROV agent
depth 3  span:0ba106098274110f               execute_tool go_to         TOOL / execute_tool
depth 3  span:d68aeac8e7c6af69               execute_tool pickup        TOOL / execute_tool
depth 3  agent:llm:gemma3:4b                                            PROV agent
```

over these edges:

```
art:kb:lesson:1  --wasGeneratedBy-->  span:309885c18fc79a55     "loop closure: command failed (seen x1 in failure memory)"
art:kb:lesson:1  --wasRevisionOf-->   span:309885c18fc79a55     "error_key patch: persist the dedup basis after insert"
art:kb:lesson:1  --wasDerivedFrom-->  art:mem:episode:1
art:kb:lesson:1  --wasDerivedFrom-->  art:mem:episode:2
art:kb:lesson:1  --wasAttributedTo--> agent:env:ai2thor          "environment"
art:mem:episode:2 --wasGeneratedBy--> span:f357d71965870668
span:f357d71965870668 --wasInformedBy--> span:d68aeac8e7c6af69
span:d68aeac8e7c6af69 --wasInformedBy--> run:7e1f9a8dce27341365a46928fcdd4442
run:…4442        --wasAssociatedWith--> agent:llm:gemma3:4b      "planner"
```

Read in prose, that is the author's question answered end to end: *lesson #1 was generated by the loop-closure
activity of run `7e1f9a8d…`, was derived from the two episodes the run executed - one of which is the failing
`pickup` - was attributed to the environment (not to a human), and the responsible planning agent was
`gemma3:4b`.*

The same graph also records what the failing action itself produced, which is the reverse direction:
`trace.span_effects("d68aeac8e7c6af69", include_descendants=True)` returns exactly the episode row #2 and the
failure counter `pickup|pickupobject failed mug not visible|Mug`, each with its `before/after` and its exact
legacy audit line.

### 5.2 The second identical failure

Running the same failing command again does **not** create a second lesson. `upsert_lesson` merges on the
dedup key, so the store keeps one artifact with `revision=2`, `n_causes=2` - and the ledger keeps two rows,
`insert` then `reinforce`, under two different `run_id`s (T-51, and T-48c for the procedure equivalent). That
distinction - one entity, many identified causes - is the shape the RQ3 self-improvement curve needs, and it
is why `reinforce` had to be a first-class verdict.

---

## 6. Manual follow-up: lines to add to files owned by another workflow

Three workflows are editing this repo concurrently. Everything below is a hook that could not be applied here
because another workflow owns the file. Function names are the stable anchors; line numbers drift. Every
snippet assumes the standard defensive import at module top:

```python
try:                                    # observability sidecar (sim/trace.py); optional by design
    import trace as _trace              # shadows the stdlib coverage module on sys.path
    if getattr(_trace, "TRACE_API_VERSION", 0) < 1:
        _trace = None                   # we resolved the STDLIB trace module: disable instrumentation
except Exception:
    _trace = None
```

> Do **not** write `_t(_trace.emit, ...)`. Dereferencing the attribute at the call site raises `AttributeError`
> when the sidecar is absent, *before* any guard can run. Name the function by string and resolve it with
> `getattr` inside the guard - see `knowledge._t`.

**F1 - `sim/eval.py`: per-record run + scores.** Wrap `_score_record`'s body in
`_trace.run(kind="eval_record", command=rec.command, scene=scene, level=level, lang=rec.lang, agent_id=model,
origin="eval", config={...})`, pass `run_id=_rid` to its `improve.close_loop(...)` call, and after scoring emit
`eval.scored` with `gen_ai.evaluation.name` / `gen_ai.evaluation.score.value` plus `SR`/`PC`/`TDS`. *Interim
behaviour, already working:* the webapp launches `eval.py` as a subprocess with `TRACE_DB` / `TRACE_RUN_ID` /
`TRACE_ORIGIN` in the environment, so its KB/memory writes are already captured and attributed to the eval run;
this upgrade moves it from per-process to per-record attribution and puts the metrics in the same trace as the
executions that produced them.

**F2 - `sim/thor_actions.py`: route `replay()` through `do()`.** In `replay(log)`, replace
`self.event = self.controller.step(action=action, **kwargs)` with `self.event = self.do(action, **kwargs)`
(`_logging_actions` is already `False` there, so behaviour is otherwise identical). Without it, replayed macro
primitives are invisible to the trace. `gpu_status()`'s `Pass` loop needs nothing - `Pass` is filtered by
default.

**F3 - `sim/select.py`: the bandit's own writes.** Three `record_write` calls: in `_insert_decision()` after
the commit (`verdict='insert'`, `table_or_kind='bandit_decisions'`), in `reward()` after the UPDATE
(`verdict='update'`, with `before`/`after`), and in `_persist_arm()` after the upsert
(`table_or_kind='bandit_arms'`, `verdict='update'`).

**F4 - `sim/llm.py`: real token telemetry.** `OllamaProvider.chat` already parses Ollama's response and throws
away `eval_count`, `prompt_eval_count` and `total_duration`. Stash them on the provider instance
(`self.last_usage = {...}` - no signature change) and `record_llm` can fill `gen_ai.usage.input_tokens` /
`output_tokens` instead of the char-count proxies.

**F5 - `sim/rag.py`: the embeddings sidecar.** One `record_write(store="file", table_or_kind="embeddings", …)`
in `HybridRAG._save_embeddings` would make the derived vector file first-class; it currently holds orphaned
vectors for ids from before the KB reset, which the Stats pane could then surface as a "stale embeddings"
signal.

**F6 - `sim/control_gui.py`: the legacy Tk loop.** `ControlGUI._learn` writes procedures/lessons directly,
bypassing `improve.close_loop`. The writes *are* captured (the hook is in `knowledge._add`) but land as
orphans. Either open a `trace.run(kind='gui')` around `_learn`, or accept the orphan attribution - the Tk GUI
is superseded by the webapp.

**F7 - `sim/web/index.html`: a link to the viewer.** The main console has no `/trace` link; the page is
reachable only by typing the URL. One anchor in the header nav is enough. (`index.html` is being rewritten by
the UI-redesign workflow right now, which is why it is not added here.)

---

## 7. Overhead, verification, limitations

### 7.1 Measured overhead

The self-test asserts a hard budget and fails the build if it is exceeded:

| Check | Budget asserted | Measured (Windows, Python 3.14, SQLite 3.50.4) |
|---|---|---|
| T-30 async `emit` (default mode) | average **< 0.10 ms**, p99 <= 100 µs, all 5000 rows land | **6.7 µs** average, p50 2.8 µs, **p99 6.2 µs** |
| T-31 sync `emit` (`TRACE_ASYNC=0`) | average < 1.0 ms, p99 <= 1000 µs | 13.8 µs average, p99 21.9 µs |
| T-32 filtered `Pass`/`Done` primitives | 10 000 in **< 50 ms**, zero events emitted | **3.7 ms**, 10 000 counted, 0 emitted |

So the instrumentation costs single-digit microseconds per event on the hot path, against AI2-THOR steps that
cost milliseconds and Ollama planning calls that cost seconds - i.e. it is not measurable in any thesis number.
The full worked-example command above produced 16 events, 7 spans, 6 writes, 5 artifacts and 27 edges.

Volume control, in the order it applies: `Pass`/`Done` are dropped before an event is built; contiguous
identical navigation primitives collapse to one span with `repeat=n`; `action.executed` and friends are
`DEBUG` and filtered out of the timeline by default; prompts/completions truncate at 8 KB; and
`TRACE_AUDIT_MIRROR=0` removes the largest single event class (~49% by count) without losing any information,
since `writes.audit_line` carries the identical string.

### 7.2 Verification

```powershell
python sim/_trace_selftest.py            # 72/72  - the observability gate
python sim/_trace_selftest.py --regress  # adds T-53: re-runs the four protected self-tests with TRACE_ENABLED=1 and 0
python sim/_phase7_selftest.py           # 37/37  - loop closure, unchanged
python sim/_phase6_selftest.py           # OK     - env feedback, unchanged
python sim/_phase6a_selftest.py          # OK     - bandit, unchanged
python sim/_eval_selftest.py             # 64/64  - eval harness, unchanged
python sim/trace.py                      # TRACE DEMO OK - 29/29 public entry points benign under degradation
python -c "import sys; sys.path.insert(0,'sim'); import webapp; print('webapp imports')"
```

All temp-store only: no AI2-THOR, no LLM, no network, and the real `knowledge_base.json` / `memory.db` /
`loop_reports.jsonl` are fingerprinted and asserted byte-identical (T-45 does 20 passes over every read API and
proves a write attempted through the browser path is rejected).

The 72 checks group as: store & schema (T-1..6), identity & context (T-7..13), the two causal directions
(T-14..24), safety & non-interference (T-25..29), performance (T-30..35), concurrency (T-36..38), the read API
(T-39..47), end-to-end loop scenarios (T-48..52), and the hardening group (T-54..66) added after review.

### 7.3 What review found, and what fixed it

| Defect found | Fix | Test |
|---|---|---|
| Cross-process `events.seq` collision: two processes minted the same primary key, so `INSERT OR IGNORE` silently discarded events and could re-point a `writes` row at another process's event | `_reserve_seq_block()` claims a disjoint 4096-value block off `meta.seq_hwm` in a `BEGIN IMMEDIATE` transaction on a short-lived private connection | T-37, T-5 |
| `sim/webapp.py` had **zero** write-side instrumentation - every write from the console was an orphan (4079/4079 measured) | the full run/span/event wiring listed in §3.1 | T-63, T-64 |
| `install_thor_hooks()` was dead code - never called | called at import and again in `main()`; `ThorAgent.do._traced` is now `True` after `import webapp` | T-62 |
| `improve.close_loop` was called with no observability context, so `_link_episodes` never fired and no `procedure -> episode` lineage existed | `_close_loop_after_run` now threads `run_id`, `decision_id` and the executed steps' `episode_ids` | T-65 |
| Ambient-run misattribution: a feedback run opened on an HTTP thread could steal the process default from an open command run, and `end_run` could reinstate an already-finished run | the process default is claimed only when no other run owns it (`_open_runs`), and the worker thread is bound explicitly from a plan-thread snapshot | T-55, T-56 |
| Artifact-id fusion: entry #1 of two different knowledge bases collapsed into one artifact (ids restart at 1 after every KB reset) | ids are minted by `trace.artifact_id()` and namespaced per store file (`kb@1a2b3c4d:…`); `artifact()` refuses to show live content from a foreign store | T-58 |
| `writes.step_ix` was never populated | `step_ix` rides the ambient context frame, set by `trace.span(step_ix=…)` | T-57 |
| The ledger was droppable: queue overflow could evict a `writes`/`artifacts`/`edges` op. A dropped op was also counted as enqueued but never committed, so `flush()` blocked for its full timeout after any drop | `_enqueue` only ever evicts events (DEBUG first); drops advance the commit counter | T-54, T-54b |
| `status()` / `stats()` created the store from routes documented as read-only | they report `pending=True` with a reason; the viewer renders "idle" | T-59 |
| `db_rows()` silently fell back to the primary key on an unknown `order_by` | returns an error envelope (the safety property is unchanged: no user string reaches SQL) | T-60 |
| Loop-report pointer collisions, and real data loss on Windows (the CRT's `O_APPEND` is not atomic - 8 concurrent appends produced 5 lines) | `_append_report_line()` appends under a thread lock plus a best-effort cross-process file lock and derives the line number from the byte offset of its own append; `_count_lines()` is incremental | T-61 |

### 7.4 Limitations, stated honestly

- **The author's existing `sim/trace.db` holds pre-fix data** (0 runs, 100% orphan writes, fused artifact ids)
  from before the webapp instrumentation and the id-namespacing landed. Nothing deletes it, but **thesis
  figures should come from a fresh store**: stop the console, move the old file aside, and re-run.
- **Resetting a store in place still shares artifact ids.** Namespacing is keyed on the store *path*, so a KB
  that is emptied and refilled at the same path will reuse `kb:lesson:1` for an unrelated entry. Cross-file
  fusion is fixed; same-file reset is not.
- **`ThorAgent.replay()` bypasses the action hook** (follow-up F2), so replayed macro primitives are invisible.
- **No token counts** until F4 lands; char-count proxies stand in and the token attributes are absent, not
  zero.
- **Eval metrics are not in the trace yet** (F1). Per-process attribution works; per-record does not.
- **A new file appears next to the journal:** `sim/loop_reports.jsonl.lock`, the cross-process append lock. It
  is zero-content bookkeeping and is gitignored.
- **`trace` is a stdlib module name.** `sim/trace.py` shadows the stdlib coverage tracer for anything importing
  from `sim/`. Every instrumented module therefore guards on `TRACE_API_VERSION` and disables instrumentation
  if it resolved the wrong module. If this ever bites in a differently-ordered `sys.path`, the fallback is
  `importlib.util.spec_from_file_location("agent_trace", os.path.join(HERE, "trace.py"))`.
- **Replay is a verification tool, not the boot path.** The KB is not rebuilt from events at runtime (it is
  live research data that `rag.py`, `improve.py` and `select.py` read directly). "Rebuild-and-diff" as a
  correctness check, and counterfactual replay for the Phase-10 ablation matrix, are available but not wired
  into any pipeline.
- **`sim/trace.db` is not a projection of anything else** - it is the only place some information exists
  (pruned beliefs' confidences, LLM latency, the console log). Deleting it loses that history permanently, and
  since it is gitignored it is not backed up by the repo.

---

## 8. For the thesis

Take **two screenshots from one fresh run**, both from the Lineage view in the "paper" theme:

1. **The lineage graph of a single KB entry** (`art:kb:procedure:N`), ancestors direction, showing the
   `wasGeneratedBy` edge to its `close_loop` span, the `wasDerivedFrom` edges to the episodes of the actions
   the plan actually executed, the `used` edges to the entries retrieval injected into the planner prompt, and
   the `wasAttributedTo` edge to `env:ai2thor`.
2. **The entity detail panel** for the same entry, showing `revision`, `n_causes` and the ledger rows -
   `insert` once, `reinforce` n times, each naming a different `run_id`.

Together they evidence the claim that upgrades contribution (c) from an assertion to an exhibit:

> *Every mutation of the agent's long-term memory is attributable to the specific command, plan step and
> AI2-THOR action that caused it, and every mutation can be traced back from the memory entry to its cause.*

They also give §5 (System architecture) a fifth, cross-cutting component - the **observability plane**, a
sidecar that observes the four components without participating in the loop, which is consistent with (and
strengthens) the interpretability argument for the modular, non-end-to-end design. For related work, the
positioning is in `docs/observability-research.md` §2.1: six commercial LLM-observability products all trace
execution, none traces an agent writing back into its own long-term memory, and the gap is independently named
as a methodological direction in the 2026 agent-provenance survey cited there. Modelling memory lineage in
PROV-O terms is therefore a defensible, citable position rather than a bespoke invention.

---

## Files

- `sim/trace.py` - the observability plane: store, context, writer thread, PROV helpers, read API (new).
- `sim/_trace_selftest.py` - the pure gate, 72 checks (new).
- `sim/knowledge.py`, `sim/memory.py`, `sim/improve.py`, `sim/feedback.py` - instrumented write paths.
- `sim/webapp.py` - runs/spans/events for commands, steps, feedback, perception and eval; the `/trace*` routes.
- `sim/web/trace.html`, `sim/web/trace.css`, `sim/web/trace.js` - the viewer (new, separate from the console).
- `sim/trace.db` - the store (created at runtime; gitignored).
- `docs/observability-research.md` - the survey and the standards this maps onto.
- `docs/observability-design.md` - the buildable specification.
