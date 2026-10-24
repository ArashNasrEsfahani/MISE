"""Offline selftest for sim/planners (final test, stage P1) -- mocked HTTP / subprocess, no network, no simulator,
no Ollama, no real key.

    python sim/_planners_selftest.py

Covers: byte equality of the contract with step4_gemma (and of every copied constant), identical application content
across the four providers, memory / few-shot blocks, per-model Anthropic request bodies and the schema transform,
reply parsing + shape repair, the runner's single repair, response cache hit / miss / replicate / corruption /
replay without a key, the cost ceiling (global, per provider, restart, reservations, ambiguous failures), retries
and backoff and rate limiting, secrets never reaching any output, a missing key -> clean skip, Ollama payload parity
with llm.OllamaProvider, the Codex isolation (empty temp dir, scrubbed env, deleted afterwards), and the price table.
"""
import glob
import json
import os
import re
import sys
import tempfile
import types as _types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import honest_skills as hs          # noqa: E402
import step4_gemma as g            # noqa: E402
import llm                         # noqa: E402
import planners as P               # noqa: E402
from planners import contract as C  # noqa: E402
from planners.anthropic_provider import AnthropicPlanner, PROFILES, to_anthropic_schema  # noqa: E402
from planners.codex_provider import CodexCliPlanner  # noqa: E402
from planners.keys import redact, Secret, get_openrouter_key  # noqa: E402
from planners.ollama_provider import OllamaPlanner  # noqa: E402
from planners import ollama_provider as OP  # noqa: E402
from planners.openrouter_provider import OpenRouterPlanner  # noqa: E402

PASS = FAIL = 0


def check(name, cond, info=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {info}")


FAKE_ANT = "s" + "k-ant-api03-" + "Zx9Qw3Lm7Rt5Vb2NqPz81"
FAKE_OR = "s" + "k-or-v1-" + "abcdef0123456789abcdef0123"
TYPES = ["Apple", "Fridge", "Knife", "Plate", "CounterTop", "Bread"]
GOOD = {"say": "ok", "plan": [{"action": "pick_and_place", "object": "Apple", "to": "Fridge", "arg": None}]}
BAD = {"say": "ok", "plan": [{"action": "fly", "object": "Apple", "to": None, "arg": None}]}


def anth_ok(obj, tin=3000, tout=1000, stop="end_turn", model="claude-opus-5-5", text=None):
    txt = text if text is not None else json.dumps(obj)
    return (200, {}, {"id": "msg_1", "model": model, "stop_reason": stop,
                      "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": txt}],
                      "usage": {"input_tokens": tin, "output_tokens": tout}})


class Script:
    """Scripted transport: items are (status, headers, body) tuples, exceptions (raised) or callables."""

    def __init__(self, *items):
        self.items, self.calls = list(items), []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": dict(headers), "body": body})
        it = self.items.pop(0) if len(self.items) > 1 else self.items[0]
        if isinstance(it, BaseException):
            raise it
        return it(url, headers, body) if callable(it) else it


class Env:
    def __init__(self, ceiling=None, provider_ceilings=None, ns="t"):
        self.td = tempfile.TemporaryDirectory()
        self.root = self.td.name
        self.cache = P.ResponseCache(os.path.join(self.root, "cache"), namespace=ns)
        self.ledger = P.CostLedger(os.path.join(self.root, "ledger.db"), ceiling, provider_ceilings)
        self.sleeps = []

    def kw(self, transport=None, **extra):
        return dict(cache=self.cache, ledger=self.ledger, transport=transport, sleep=self.sleeps.append, **extra)

    def all_text(self):
        out = []
        for dp, _, fs in os.walk(self.root):
            for f in fs:
                with open(os.path.join(dp, f), "rb") as fh:
                    out.append(fh.read().decode("utf-8", "replace"))
        return "\n".join(out)

    def close(self):
        self.td.cleanup()


CT = P.PlannerContract(TYPES, tools=hs.TOOLS)
MSGS = CT.messages("put the apple in the fridge", ["Apple [free] CounterTop (1.0, 0.9, 2.0)"], None)
SCHEMA = C.PLAN_SCHEMA

# ===================================================================================== 1. contract == step 4
cases = [("put the apple in the fridge", [], None, None),
         ("سیب را در یخچال بگذار", ["Apple [free] CounterTop (1.0, 0.9, 2.0)", "Knife [free] Drawer (0, 0, 0)"], "Apple", None),
         ("open the fridge", ["x"], None, ["pickup", "put_in", "open"])]
for cmd, mem, hold, tools in cases:
    mine = P.PlannerContract(TYPES, tools=tools or hs.TOOLS).messages(cmd, mem, hold)
    ref = g.build_messages(cmd, TYPES, mem, hold, tools)
    check(f"contract == step4 build_messages [{cmd[:12]}]", mine == ref)
    check("   byte-identical json", json.dumps(mine, ensure_ascii=False) == json.dumps(ref, ensure_ascii=False))
for nm in ("SYSTEM", "USER", "REPAIR", "TOOL_SPECS", "PLAN_SCHEMA", "MAX_STEPS"):
    check(f"constant {nm} equals step4_gemma", getattr(C, nm) == getattr(g, nm))
check("tool_lines equals step4", C.tool_lines(hs.TOOLS) == g.tool_lines(hs.TOOLS))
check("problems_text equals step4", C.problems_text([{"i": 1, "reason": "r"}, {"i": None, "reason": "q"}])
      == g.problems_text([{"i": 1, "reason": "r"}, {"i": None, "reason": "q"}]))
check("ollama MODEL/OPTIONS/THINK equal step4", (OP.MODEL, OP.OPTIONS, OP.THINK) == (g.MODEL, g.OPTIONS, g.THINK))
check("repair_messages equals step4 repair turn",
      C.PlannerContract.repair_messages(MSGS, "RAW", "- p")[2:] ==
      [{"role": "assistant", "content": "RAW"}, {"role": "user", "content": g.REPAIR.format(problems="- p")}])
check("no scene/goal leak in prompt", not re.search(r"FloorPlan|AI2-THOR|ai2thor|iTHOR|Unity|_physics|\|[-0-9.]+\|",
                                                     json.dumps(MSGS)))

# ===================================================================================== 2. blocks
m0 = CT.messages("open the fridge", ["x"], None)
mb = CT.messages("open the fridge", ["x"], None, memory_block="Apple is usually in Fridge.")
mf = CT.messages("open the fridge", ["x"], None, fewshot_block="episode 1: ok")
check("no block -> unchanged", m0 == g.build_messages("open the fridge", TYPES, ["x"], None, hs.TOOLS))
check("memory block appended under fixed header", mb[1]["content"].startswith(m0[1]["content"])
      and C.MEMORY_HEADER in mb[1]["content"] and mb[1]["content"].endswith("Apple is usually in Fridge."))
check("experience block header", C.EXPERIENCE_HEADER in mf[1]["content"] and C.MEMORY_HEADER not in mf[1]["content"])
check("system text unchanged by blocks", mb[0] == m0[0] == mf[0])
cb = P.PlannerContract(TYPES, tools=hs.TOOLS, block_position="before_command").messages(
    "open the fridge", ["x"], None, memory_block="B")
check("before_command keeps Command last", cb[1]["content"].endswith("Command: open the fridge") and "\nB\n\nCommand" in cb[1]["content"])
check("prompt hash differs with block", C.prompt_sha256(mb) != C.prompt_sha256(m0))

# ===================================================================================== 3. identical across providers
e = Env()
try:
    ant_t = Script(anth_ok(GOOD))
    a = AnthropicPlanner("claude-opus-5-5", env={"ANTHROPIC_API_KEY": FAKE_ANT}, **e.kw(ant_t))
    a.plan(MSGS, SCHEMA)
    ol_t = Script((200, {}, {"message": {"content": json.dumps(GOOD)}, "done_reason": "stop", "prompt_eval_count": 900, "eval_count": 60}))
    o = OllamaPlanner(**e.kw(ol_t))
    o.plan(MSGS, SCHEMA)
    envf = os.path.join(e.root, "k.env")
    open(envf, "w").write(f'# c\nexport OTHER=1\nOPENROUTER_API_KEY="{FAKE_OR}"\n')
    or_t = Script((200, {}, {"id": "g1", "model": "x/y", "provider": "P", "choices": [{"message": {"content": json.dumps(GOOD)}, "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 900, "completion_tokens": 60, "cost": 0.0003}}))
    r = OpenRouterPlanner("x/y", env={"OPENROUTER_ENV_FILE": envf}, **e.kw(or_t))
    r.plan(MSGS, SCHEMA)
    seen = {}

    def fake_run(cmd, inp, cwd, env, timeout):
        seen.update(cmd=cmd, inp=inp, cwd=cwd, env=env, listing=sorted(os.listdir(cwd)) if cwd and os.path.isdir(cwd) else None)
        if "--version" in cmd:
            return _types.SimpleNamespace(returncode=0, stdout="codex-cli 0.157.0\n", stderr="")
        open(cmd[cmd.index("-o") + 1], "w", encoding="utf-8").write(json.dumps(GOOD))
        ev = [{"type": "thread.started"}, {"type": "turn.completed", "usage": {"input_tokens": 5000, "output_tokens": 200}}]
        return _types.SimpleNamespace(returncode=0, stdout="\n".join(json.dumps(x) for x in ev), stderr="")
    cx = CodexCliPlanner("gpt-6-astra", runner=fake_run, which=lambda c: "/bin/codex", **e.kw())
    os.environ["ANTHROPIC_API_KEY"] = FAKE_ANT
    cres = cx.plan(MSGS, SCHEMA)
    os.environ.pop("ANTHROPIC_API_KEY", None)
    ab = ant_t.calls[0]["body"]
    sys_txt, usr_txt = MSGS[0]["content"], MSGS[1]["content"]
    check("anthropic: system top-level + one user turn identical", ab["system"] == sys_txt and ab["messages"] == [{"role": "user", "content": usr_txt}])
    check("ollama: chat roles identical", ol_t.calls[0]["body"]["messages"] == MSGS)
    check("openrouter: chat roles identical", or_t.calls[0]["body"]["messages"] == MSGS)
    check("codex: stdin == flat_prompt(messages) with both texts", seen["inp"] == C.flat_prompt(MSGS) and sys_txt in seen["inp"] and usr_txt in seen["inp"])
    check("same prompt hash everywhere", len({C.prompt_sha256(MSGS)}) == 1)
    # codex isolation
    check("codex: temp dir held only the schema at start", seen["listing"] == ["schema.json"], str(seen["listing"]))
    check("codex: temp dir deleted afterwards", not os.path.exists(seen["cwd"]))
    check("codex: flags", all(x in seen["cmd"] for x in ("exec", "--skip-git-repo-check", "--ephemeral", "--json", "--output-schema", "-o")) and seen["cmd"][seen["cmd"].index("-s") + 1] == "read-only" and seen["cmd"][seen["cmd"].index("-m") + 1] == "gpt-6-astra")
    check("codex: P3 isolation flags (user config + rules ignored, effort pinned)",
          "--ignore-user-config" in seen["cmd"] and "--ignore-rules" in seen["cmd"]
          and seen["cmd"][seen["cmd"].index("-c") + 1] == 'model_reasoning_effort="high"'
          and cx.decoding()["reasoning_effort"] == "high" and "--ignore-user-config" in cx.decoding()["cli_flags"])
    check("codex: child env has no key variables", not any(k in seen["env"] for k in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_ENV_FILE")))
    # P3: a per-call max_tokens override is part of the cache key (no silent cross-setting reuse)
    k_plain = o.plan(MSGS, SCHEMA)["cache_key"]
    k_over = OllamaPlanner(**e.kw(ol_t)).plan(MSGS, SCHEMA, budget={"max_tokens": 321})["cache_key"]
    check("cache key includes a budget max_tokens override", k_plain != k_over)
    # P3: optional strict schema for Ollama (off = step-4 behaviour)
    ol_s = Script((200, {}, {"message": {"content": json.dumps(GOOD)}, "done_reason": "stop", "prompt_eval_count": 900, "eval_count": 60}))
    OllamaPlanner(strict_schema=True, **e.kw(ol_s)).plan(MSGS, SCHEMA, use_cache=False)
    check("ollama strict_schema off by default = bare schema", ol_t.calls[0]["body"]["format"] == SCHEMA)
    check("ollama strict_schema on = the Anthropic/Codex transform",
          ol_s.calls[0]["body"]["format"] == to_anthropic_schema(SCHEMA)
          and OllamaPlanner(strict_schema=True, **e.kw(ol_s)).decoding()["format"] == "json-schema-strict")
    check("codex: result exploratory, version, usage, overhead", cres["exploratory"] and cres["extra"]["cli_version"] == "codex-cli 0.157.0"
          and cres["tokens_in"] == 5000 and cres["tokens_out"] == 200 and cres["extra"]["overhead_tokens_in"] is not None
          and cres["extra"]["tool_events"] == 0 and cres["cost_usd"] == 0.0 and cres["parsed"] == GOOD)
    check("role mapping + wire hash recorded", all(x["extra"].get("request_sha256") for x in (cres,)) and "top-level" in AnthropicPlanner.role_mapping)
    # codex failure modes
    def tool_run(cmd, inp, cwd, env, timeout):
        if "--version" in cmd:
            return _types.SimpleNamespace(returncode=0, stdout="v", stderr="")
        open(cmd[cmd.index("-o") + 1], "w").write(json.dumps(GOOD))
        return _types.SimpleNamespace(returncode=0, stdout=json.dumps({"type": "item.completed", "item": {"type": "command_execution"}}), stderr="")
    cv = CodexCliPlanner("gpt-6-luna", runner=tool_run, which=lambda c: "x", **e.kw()).plan(MSGS, SCHEMA)
    check("codex: tool event flags isolation violation", cv["extra"]["isolation_violation"] and cv["extra"]["tool_events"] == 1)
    def exit1(cmd, inp, cwd, env, timeout):
        return _types.SimpleNamespace(returncode=1 if "--version" not in cmd else 0, stdout="", stderr="boom")
    cf = CodexCliPlanner("gpt-6-sol", runner=exit1, which=lambda c: "x", **e.kw()).plan(MSGS, SCHEMA)
    check("codex: nonzero exit -> transport error, not cached", cf["error_kind"] == "transport" and e.cache.get(cf["cache_key"]) is None)
    try:
        CodexCliPlanner("gpt-6-sol", which=lambda c: None, **e.kw()).plan(MSGS, SCHEMA)
        check("codex: CLI missing -> PlannerSkip", False)
    except P.PlannerSkip:
        check("codex: CLI missing -> PlannerSkip", True)
    # ---- A21 E: the X arms' isolation and identity checks
    from planners import codex_provider as _cp
    SESS = ('2026-10-12T00:00:00Z  INFO codex.exec{otel.kind="internal" thread.id="t"}: codex_exec: Codex initialized with event: SessionConfiguredEvent { '
            'session_id: SessionId { uuid: u }, thread_id: ThreadId { uuid: u }, forked_from_id: None, parent_thread_id: None, thread_source: Some(User), thread_name: None, '
            'model: "gpt-6-sol", model_provider_id: "openai", service_tier: None, approval_policy: Never, approvals_reviewer: User, permission_profile: Managed { '
            'file_system: Restricted { entries: [FileSystemSandboxEntry { path: Special { value: Root }, access: Read, missing_path_behavior: None }], '
            'glob_scan_max_depth: None }, network: Restricted }, active_permission_profile: None, cwd: AbsolutePathBuf("C:\\\\Users\\\\someone\\\\Temp\\\\x"), '
            'reasoning_effort: Some(High), network_proxy: None, rollout_path: None }')
    ss = _cp.parse_session("Reading prompt from stdin...\n" + SESS + "\n")
    check("A21 E: parse_session reads model, effort, approval, read-only sandbox and restricted network from the CLI's own line, and stores no path",
          ss == {"found": True, "model": "gpt-6-sol", "reasoning_effort": "High", "approval_policy": "Never", "sandbox_read_only": True, "network_restricted": True}
          and "someone" not in json.dumps(ss))
    check("A21 E: no session line -> not found", _cp.parse_session("nothing here") == {"found": False})
    ok_id = _cp.identity_problems(ss, "gpt-6-sol", "high", "codex-cli 0.162.1", "codex-cli 0.162.1")
    check("A21 E: the frozen session has no identity problem", ok_id == [], str(ok_id))
    check("A21 E: a different model, a different effort, a write sandbox, open network, another approval policy, another CLI and a missing line are each a problem",
          _cp.identity_problems(dict(ss, model="gpt-5.6-terra"), "gpt-6-sol", "high", "v", None)
          and _cp.identity_problems(dict(ss, reasoning_effort="Low"), "gpt-6-sol", "high", "v", None)
          and _cp.identity_problems(dict(ss, sandbox_read_only=False), "gpt-6-sol", "high", "v", None)
          and _cp.identity_problems(dict(ss, network_restricted=False), "gpt-6-sol", "high", "v", None)
          and _cp.identity_problems(dict(ss, approval_policy="OnRequest"), "gpt-6-sol", "high", "v", None)
          and _cp.identity_problems(ss, "gpt-6-sol", "high", "codex-cli 0.163.0", "codex-cli 0.162.1")
          and _cp.identity_problems({"found": False}, "gpt-6-sol", "high", "v", None) == ["session_event_missing"])
    sandbox_w = SESS.replace("access: Read,", "access: Write,")
    check("A21 E: a session whose sandbox grants write access is not read-only", _cp.parse_session(sandbox_w)["sandbox_read_only"] is False)
    check("A21 E: tool events = known tool types or ANY non-message item type; plain messages and reasoning are not",
          _cp.tool_events([{"item": {"type": "agent_message"}}, {"item": {"type": "reasoning"}}, {"type": "turn.completed"}]) == 0
          and _cp.tool_events([{"item": {"type": "mcp_tool_call"}}]) == 1 and _cp.tool_events([{"item": {"type": "some_future_tool"}}]) == 1
          and _cp.tool_events([{"type": "command_execution"}]) == 1)
    seen2 = {}

    def run_id(cmd, inp, cwd, env, timeout):
        if "--version" in cmd:
            return _types.SimpleNamespace(returncode=0, stdout="codex-cli 0.162.1\n", stderr="")
        seen2.update(cmd=cmd, env=env)
        open(cmd[cmd.index("-o") + 1], "w", encoding="utf-8").write(json.dumps(GOOD))
        return _types.SimpleNamespace(returncode=0, stdout=json.dumps({"type": "turn.completed", "usage": {"input_tokens": 9000, "output_tokens": 100}}),
                                      stderr="Reading prompt from stdin...\n" + SESS + "\n")
    cx2 = CodexCliPlanner("gpt-6-sol", runner=run_id, which=lambda c: "/bin/codex", pin_cli_version="codex-cli 0.162.1", **e.kw())
    r2 = cx2.plan(MSGS, SCHEMA)
    x2 = r2["extra"]
    dis = [seen2["cmd"][i + 1] for i, a in enumerate(seen2["cmd"]) if a == "--disable"]
    check("A21 E: every call disables the hardened CLI features, sets the log filter, and logs the CLI-configured model (the server-side model stays unobserved) with identity ok",
          tuple(dis) == _cp.HARDEN_FEATURES and seen2["env"].get("RUST_LOG") == _cp.SESSION_LOG and x2["cli_configured_model"] == "gpt-6-sol" and x2["served_model"] is None and x2["requested_model"] == "gpt-6-sol"
          and x2["identity_violation"] is False and x2["identity_problems"] == [] and x2["pin_cli_version"] == "codex-cli 0.162.1"
          and x2["server_side_model"] == "not observable through the CLI" and r2["decoding"]["hardened"] is True and r2["decoding"]["pinned_cli_version"] == "codex-cli 0.162.1")
    check("A21 E: the stored stderr tail holds neither the session line nor a path", "SessionConfiguredEvent" not in x2["stderr_tail"] and "someone" not in x2["stderr_tail"])
    cx_wrong = CodexCliPlanner("gpt-6-luna", runner=run_id, which=lambda c: "/bin/codex", **e.kw())       # the fake session says gpt-6-sol: a substitution
    rw = cx_wrong.plan(MSGS, SCHEMA)
    check("A21 E: a session that reports another model than the requested one is an identity violation with the difference named",
          rw["extra"]["identity_violation"] and any("gpt-6-sol" in x and "gpt-6-luna" in x for x in rw["extra"]["identity_problems"]))
    try:
        CodexCliPlanner("gpt-6-sol", runner=run_id, which=lambda c: "/bin/codex", pin_cli_version="codex-cli 0.150.0", **e.kw()).plan(MSGS, SCHEMA)
        check("A21 E: a CLI whose version differs from the frozen pin makes no call", False)
    except P.PlannerSkip as ex_:
        check("A21 E: a CLI whose version differs from the frozen pin makes no call", "pin mismatch" in str(ex_))
    check("A21 E: the flag set (hardening included) is part of the cache key: a hardened and a plain provider never share a cached answer",
          CodexCliPlanner("gpt-6-sol", which=lambda c: "x", hardened=True, **e.kw()).decoding()["cli_flags"]
          != CodexCliPlanner("gpt-6-sol", which=lambda c: "x", hardened=False, **e.kw()).decoding()["cli_flags"])
finally:
    e.close()

# ===================================================================================== 4. anthropic bodies + schema
e = Env()
try:
    env = {"ANTHROPIC_API_KEY": FAKE_ANT}
    bo = AnthropicPlanner("claude-opus-5-5", env=env, **e.kw()).build_body(MSGS, SCHEMA)
    bs = AnthropicPlanner("claude-sonnet-5-5", env=env, **e.kw()).build_body(MSGS, SCHEMA)
    bh = AnthropicPlanner("claude-haiku-4-5-20251001", env=env, **e.kw()).build_body(MSGS, SCHEMA)
    check("opus: adaptive thinking, effort medium, no sampling, max 8000", bo["thinking"] == {"type": "adaptive"} and bo["output_config"]["effort"] == "medium" and "temperature" not in bo and bo["max_tokens"] == 8000)
    check("sonnet: adaptive thinking, effort high, no sampling", bs["thinking"] == {"type": "adaptive"} and bs["output_config"]["effort"] == "high" and "temperature" not in bs)
    check("haiku: no thinking, temperature 0, max 2000, no effort", "thinking" not in bh and bh["temperature"] == 0 and bh["max_tokens"] == 2000 and "effort" not in bh["output_config"])
    check("no fallbacks / betas in body", not any(k in bo for k in ("fallbacks", "betas")))
    check("structured output json_schema", bo["output_config"]["format"]["type"] == "json_schema")
    sch = bo["output_config"]["format"]["schema"]
    def walk(n, acc):
        if isinstance(n, dict):
            if isinstance(n.get("type"), list):
                acc.append("typelist")
            if n.get("type") == "object" and n.get("additionalProperties") is not False:
                acc.append("open object")
            for v in n.values():
                walk(v, acc)
        elif isinstance(n, list):
            for v in n:
                walk(v, acc)
        return acc
    check("schema transformed: no type lists, objects closed", walk(sch, []) == [], str(walk(sch, [])))
    arg = sch["properties"]["plan"]["items"]["properties"]["arg"]
    check("arg -> anyOf string|number|null", [v["type"] for v in arg["anyOf"]] == ["string", "number", "null"])
    check("original schema untouched", isinstance(SCHEMA["properties"]["plan"]["items"]["properties"]["object"]["type"], list) and "additionalProperties" not in SCHEMA)
    try:
        to_anthropic_schema({"type": "object", "properties": {"n": {"type": "string", "minLength": 2}}})
        check("unsupported keyword raises (never silently weakened)", False)
    except ValueError:
        check("unsupported keyword raises (never silently weakened)", True)
    try:
        AnthropicPlanner("claude-nope", env=env, **e.kw())
        check("unknown model rejected", False)
    except ValueError:
        check("unknown model rejected", True)
    # response handling
    t = Script(anth_ok(GOOD))
    p = AnthropicPlanner("claude-opus-5-5", env=env, **e.kw(t))
    r = p.plan(MSGS, SCHEMA)
    check("thinking block skipped, text parsed", r["parsed"] == GOOD and r["error"] == "")
    check("tokens and cost 3000/1000 opus = $0.032", r["tokens_in"] == 3000 and r["tokens_out"] == 1000 and abs(r["cost_usd"] - 0.032) < 1e-9, str(r["cost_usd"]))
    check("contract fields present", all(k in r for k in ("raw", "parsed", "tokens_in", "tokens_out", "cost_usd", "latency_s", "model_id", "provider", "decoding")) and r["provider"] == "anthropic" and r["model_id"] == "claude-opus-5-5")
    check("headers carry key + version", t.calls[0]["headers"]["x-api-key"] == FAKE_ANT and t.calls[0]["headers"]["anthropic-version"])
    for stop, kind in (("max_tokens", "length"), ("refusal", "refusal"), ("pause_turn", "stop")):
        rr = AnthropicPlanner("claude-sonnet-5-5", env=env, **e.kw(Script(anth_ok(GOOD, stop=stop)))).plan(
            C.PlannerContract(TYPES, tools=hs.TOOLS).messages("stop " + stop), SCHEMA)
        check(f"stop_reason {stop} -> error_kind {kind}", rr["error_kind"] == kind, rr["error_kind"])
finally:
    e.close()

# ===================================================================================== 5. parsing
check("parse clean", C.parse_reply('{"a":1}') == ({"a": 1}, ""))
check("parse fenced", C.parse_reply('```json\n{"a":1}\n```')[0] == {"a": 1})
check("parse prose-wrapped", C.parse_reply('Sure! {"a":1} done')[0] == {"a": 1} and "extracted" in C.parse_reply('Sure! {"a":1} done')[1])
check("parse garbage / empty / array", C.parse_reply("nope")[0] is None and C.parse_reply("")[0] is None and C.parse_reply("[1]")[0] is None)
check("schema ok", C.schema_problems(GOOD, SCHEMA) == [])
check("schema missing key", any("missing required field 'arg'" in s for s in C.schema_problems({"say": "x", "plan": [{"action": "a", "object": None, "to": None}]}, SCHEMA)))
check("schema wrong type", any("expected array" in s for s in C.schema_problems({"say": "x", "plan": "no"}, SCHEMA)))

# ===================================================================================== 6. cache
e = Env()
try:
    env = {"ANTHROPIC_API_KEY": FAKE_ANT}
    t = Script(anth_ok(GOOD))
    p = AnthropicPlanner("claude-opus-5-5", env=env, **e.kw(t))
    r1 = p.plan(MSGS, SCHEMA)
    r2 = p.plan(MSGS, SCHEMA)
    check("cache hit: one transport call", len(t.calls) == 1 and r2["cached"] and not r1["cached"])
    check("hit costs 0, keeps original cost, same raw/parsed", r2["cost_usd"] == 0.0 and abs(r2["original_cost_usd"] - 0.032) < 1e-9 and r2["raw"] == r1["raw"] and r2["parsed"] == GOOD)
    check("ledger billed only once", abs(e.ledger.billed("anthropic") - 0.032) < 1e-9)
    p.plan(MSGS, SCHEMA, replicate=1)
    check("replicate index -> separate entry (miss)", len(t.calls) == 2)
    p.plan(MSGS, SCHEMA, replicate=1)
    check("same replicate replays", len(t.calls) == 2)
    p.plan(MSGS, SCHEMA, stage="repair")
    check("stage in key", len(t.calls) == 3)
    AnthropicPlanner("claude-sonnet-5-5", env=env, **e.kw(t)).plan(MSGS, SCHEMA)
    check("other model -> miss", len(t.calls) == 4)
    AnthropicPlanner("claude-opus-5-5", effort="high", env=env, **e.kw(t)).plan(MSGS, SCHEMA)
    check("other decoding -> miss", len(t.calls) == 5)
    p.plan(CT.messages("a different command"), SCHEMA)
    check("other prompt -> miss", len(t.calls) == 6)
    other = Env(ns="other")
    k1 = e.cache.key("anthropic", "m", "h", {}, 0, SCHEMA)
    k2 = P.ResponseCache(e.cache.root, namespace="other").key("anthropic", "m", "h", {}, 0, SCHEMA)
    k3 = e.cache.key("anthropic", "m", "h", {}, 0, {"type": "object"})
    check("namespace and schema in key", k1 != k2 and k1 != k3)
    other.close()
    # replay without any key
    nokey = AnthropicPlanner("claude-opus-5-5", env={}, **e.kw(Script(AssertionError("no network"))))
    check("replay works without a key", nokey.plan(MSGS, SCHEMA)["cached"])
    # corruption
    path = e.cache._path(r1["cache_key"])
    open(path, "w").write("{not json")
    r3 = p.plan(MSGS, SCHEMA)
    check("corrupt entry -> miss then rewritten", not r3["cached"] and len(t.calls) == 7 and p.plan(MSGS, SCHEMA)["cached"])
    # transport failure is not cached
    bad = Script(RuntimeError("down"))
    pb = AnthropicPlanner("claude-opus-5-5", env=env, max_retries=1, **e.kw(bad))
    rb = pb.plan(CT.messages("never cached"), SCHEMA)
    check("transport failure not cached", rb["error_kind"] == "transport" and e.cache.get(rb["cache_key"]) is None)
    r4 = p.plan(MSGS, SCHEMA, use_cache=False)
    check("use_cache=False forces a live call", not r4["cached"] and len(t.calls) == 8)
finally:
    e.close()

# ===================================================================================== 7. cost ceiling
e = Env()
try:
    env = {"ANTHROPIC_API_KEY": FAKE_ANT}
    t = Script(anth_ok(GOOD))
    probe = AnthropicPlanner("claude-opus-5-5", env=env, **e.kw(t))
    r_usd = probe.reserve_usd(MSGS, {})
    check("reserve = chars/2 input + max_tokens output", abs(r_usd - ((len(MSGS[0]["content"]) + len(MSGS[1]["content"]) + 1) / 2 * 4e-6 + 8000 * 20e-6)) < 2e-5, str(r_usd))
    e.ledger.global_ceiling = r_usd + 0.016
    p = AnthropicPlanner("claude-opus-5-5", env=env, **e.kw(t))
    p.plan(MSGS, SCHEMA)
    m2 = CT.messages("put the apple in the fridge", ["Apple [free] CounterTop (1.0, 0.9, 2.0)"], None)
    m2[1]["content"] += " "                                  # same length class, different prompt -> miss
    try:
        p.plan(m2, SCHEMA)
        check("ceiling abort", False, "no exception")
    except P.CostCeilingExceeded as ex:
        check("ceiling abort raises, nothing sent", len(t.calls) == 1 and "nothing sent" in str(ex))
    check("no open reservation left after abort", abs(e.ledger.spent() - 0.032) < 1e-9, str(e.ledger.spent()))
    led2 = P.CostLedger(e.ledger.path, r_usd + 0.016)
    check("ledger persists across restart", abs(led2.spent("anthropic") - 0.032) < 1e-9)
    # per-provider ceiling
    e2 = Env(provider_ceilings={"anthropic": 0.01})
    try:
        pp = AnthropicPlanner("claude-opus-5-5", env=env, **e2.kw(Script(anth_ok(GOOD))))
        try:
            pp.plan(MSGS, SCHEMA)
            check("per-provider ceiling", False)
        except P.CostCeilingExceeded:
            check("per-provider ceiling", True)
    finally:
        e2.close()
    # smoke preset
    sm = P.CostLedger.smoke(os.path.join(e.root, "smoke.db"))
    check("smoke ceilings 1.00 anthropic / 0.10 openrouter", sm.provider_ceilings == {"anthropic": 1.0, "openrouter": 0.10})
    # openrouter hard cap
    envf = os.path.join(e.root, "k.env")
    open(envf, "w").write(f"OPENROUTER_API_KEY={FAKE_OR}\n")
    orp = OpenRouterPlanner("x/y", hard_cap_usd=0.0001, env={"OPENROUTER_ENV_FILE": envf}, **e.kw(Script(AssertionError("nope"))))
    try:
        orp.plan(MSGS, SCHEMA)
        check("openrouter hard cap", False)
    except P.CostCeilingExceeded:
        check("openrouter hard cap", True)
    # reservations are atomic across ledgers (two handles, one db)
    a1, a2 = P.CostLedger(os.path.join(e.root, "x.db"), 1.0), P.CostLedger(os.path.join(e.root, "x.db"), 1.0)
    rid = a1.reserve("anthropic", 0.7)
    try:
        a2.reserve("anthropic", 0.7)
        check("open reservation blocks a second handle", False)
    except P.CostCeilingExceeded:
        check("open reservation blocks a second handle", True)
    a1.release(rid)
    a2.reserve("anthropic", 0.7)
    check("release frees the reservation", True)
finally:
    e.close()

# ===================================================================================== 8. retries, backoff, rate limit
e = Env()
try:
    env = {"ANTHROPIC_API_KEY": FAKE_ANT}
    t = Script((429, {}, {"error": "rate"}), (429, {"Retry-After": "7"}, {"error": "rate"}), anth_ok(GOOD))
    p = AnthropicPlanner("claude-opus-5-5", env=env, **e.kw(t))
    r = p.plan(MSGS, SCHEMA)
    check("429 x2 then success", r["parsed"] == GOOD and len(t.calls) == 3)
    check("backoff 2s then Retry-After 7s", e.sleeps == [2.0, 7.0], str(e.sleeps))
    check("unbilled rejections released: spent == billed", abs(e.ledger.spent("anthropic") - 0.032) < 1e-9)
    e.sleeps.clear()
    t = Script((401, {}, {"error": "bad"}))
    r = AnthropicPlanner("claude-opus-5-5", env=env, **e.kw(t)).plan(CT.messages("x401"), SCHEMA)
    check("401 not retried, error recorded, reservation released", len(t.calls) == 1 and r["error_kind"] == "transport" and e.sleeps == []
          and abs(e.ledger.spent("anthropic") - 0.032) < 1e-9)
    t = Script(TimeoutError("timed out"))
    before = e.ledger.spent("anthropic")
    r = AnthropicPlanner("claude-opus-5-5", env=env, **e.kw(t)).plan(CT.messages("xtimeout"), SCHEMA)
    check("timeout: 4 transmissions, exponential backoff", len(t.calls) == 4 and e.sleeps == [2.0, 4.0, 8.0], str(e.sleeps))
    check("ambiguous failures keep their reservations (may be billed)", e.ledger.spent("anthropic") > before + 0.4, str(e.ledger.spent("anthropic")))
    check("failure result is clean", r["error_kind"] == "transport" and r["parsed"] is None and r["cost_usd"] == 0.0)
    # rate limiter
    e3 = Env()
    try:
        t = Script(anth_ok(GOOD))
        pr = AnthropicPlanner("claude-opus-5-5", env=env, min_interval_s=1.0, clock=lambda: 100.0, **e3.kw(t))
        pr.plan(CT.messages("r1"), SCHEMA)
        pr.plan(CT.messages("r2"), SCHEMA)
        check("min-interval rate limit sleeps before the 2nd call", e3.sleeps == [1.0], str(e3.sleeps))
    finally:
        e3.close()
finally:
    e.close()

# ===================================================================================== 9. secrets
e = Env()
try:
    keydir = tempfile.TemporaryDirectory()                  # the key file lives OUTSIDE the scanned tree
    envf = os.path.join(keydir.name, "keys.env")
    open(envf, "w").write(f"OPENROUTER_API_KEY='{FAKE_OR}'\n")
    echo_ant = Script((401, {}, {"error": {"message": f"invalid x-api-key {FAKE_ANT}"}}))
    pa = AnthropicPlanner("claude-opus-5-5", env={"ANTHROPIC_API_KEY": FAKE_ANT}, **e.kw(echo_ant))
    ra = pa.plan(CT.messages("s1"), SCHEMA)
    boom = Script(RuntimeError(f"connection died with {FAKE_ANT}"))
    ra2 = AnthropicPlanner("claude-opus-5-5", env={"ANTHROPIC_API_KEY": FAKE_ANT}, max_retries=0, **e.kw(boom)).plan(CT.messages("s2"), SCHEMA)
    echo_or = Script((500, {}, f"upstream said Bearer {FAKE_OR} sorry"))
    por = OpenRouterPlanner("x/y", env={"OPENROUTER_ENV_FILE": envf}, max_retries=0, **e.kw(echo_or))
    rr = por.plan(CT.messages("s3"), SCHEMA)
    ok_or = Script((200, {}, {"choices": [{"message": {"content": json.dumps(GOOD)}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}))
    rr2 = OpenRouterPlanner("x/y", env={"OPENROUTER_ENV_FILE": envf}, **e.kw(ok_or)).plan(CT.messages("s4"), SCHEMA)
    check("openrouter sent Bearer header (live path works)", ok_or.calls[0]["headers"]["Authorization"] == "Bearer " + FAKE_OR)
    check("openrouter cost without usage.cost falls back to the table", rr2["cost_usd"] > 0 and not rr2["extra"]["cost_from_response"])
    blob = e.all_text() + json.dumps([ra, ra2, rr, rr2], default=str) + repr(pa) + repr(por) + repr(Secret(FAKE_ANT)) + str(Secret(FAKE_OR))
    check("no Anthropic key anywhere (ledger, cache, results, repr)", FAKE_ANT not in blob)
    check("no OpenRouter key anywhere", FAKE_OR not in blob)
    check("no key-shaped string anywhere", not re.search(r"sk-[A-Za-z0-9_\-]{16,}", blob))
    check("error texts redacted but informative", "HTTP 401" in ra["error"] and "sk-" not in ra["error"].replace("sk-***", ""))
    check("redact patterns", "abc12345def" not in redact("Authorization: Bearer abc12345def") and "zzzzzzzz1" not in redact('{"x-api-key": "zzzzzzzz1"}') and "ccc" not in redact("my sk-ant-ccccccccccc here"))
    check("redact exact loaded value (even without a pattern)", "plainvalue99" not in redact("oops plainvalue99", extra=[Secret("plainvalue99")]))
    # missing keys -> clean skip, nothing sent, nothing reserved
    n0 = len(e.ledger.rows())
    t = Script(AssertionError("must not be called"))
    for nm, prov in (("anthropic unset", AnthropicPlanner("claude-opus-5-5", env={}, **e.kw(t))),
                     ("openrouter no file", OpenRouterPlanner("x/y", env={"OPENROUTER_ENV_FILE": os.path.join(e.root, "none.env")}, **e.kw(t)))):
        try:
            prov.plan(CT.messages("skip " + nm), SCHEMA)
            check(f"{nm} -> PlannerSkip", False)
        except P.PlannerSkip as ex:
            check(f"{nm} -> PlannerSkip", "skipped" in str(ex))
    open(os.path.join(e.root, "novar.env"), "w").write("OTHER=1\n")
    try:
        get_openrouter_key({"OPENROUTER_ENV_FILE": os.path.join(e.root, "novar.env")})
        check("openrouter var missing -> skip", False)
    except P.PlannerSkip:
        check("openrouter var missing -> skip", True)
    check("skip sends and reserves nothing", not t.calls and len(e.ledger.rows()) == n0)
    check("make_planner builds without a key", P.make_planner("opus", cache=e.cache, ledger=e.ledger).model == "claude-opus-5-5"
          and P.make_planner("gemma", cache=e.cache, ledger=e.ledger).provider == "ollama"
          and P.make_planner("codex:gpt-6-astra", cache=e.cache, ledger=e.ledger).exploratory
          and P.make_planner("openrouter:x/y", cache=e.cache, ledger=e.ledger).provider == "openrouter")
    try:
        P.make_planner("gpt-nope")
        check("make_planner unknown", False)
    except ValueError:
        check("make_planner unknown", True)
finally:
    keydir.cleanup()
    e.close()

# ===================================================================================== 10. ollama parity
e = Env()
try:
    cap = {}
    ref = llm.OllamaProvider("http://x")
    ref._post = lambda path, payload, timeout: (cap.update(path=path, payload=payload), {"message": {"content": "{}"}})[1]
    ref.chat(g.MODEL, MSGS, fmt=SCHEMA, options=g.OPTIONS, think=g.THINK)
    t = Script((200, {}, {"message": {"content": json.dumps(GOOD)}, "done_reason": "stop", "prompt_eval_count": 900, "eval_count": 60}))
    r = OllamaPlanner(**e.kw(t)).plan(MSGS, SCHEMA)
    check("ollama payload == llm.OllamaProvider payload", t.calls[0]["body"] == cap["payload"] and t.calls[0]["url"].endswith(cap["path"]))
    check("ollama tokens, zero cost, think False", r["tokens_in"] == 900 and r["tokens_out"] == 60 and r["cost_usd"] == 0.0 and t.calls[0]["body"]["think"] is False)
    r = OllamaPlanner(**e.kw(Script((200, {}, {"message": {"content": "{"}, "done_reason": "length"})))).plan(CT.messages("len"), SCHEMA)
    check("done_reason=length -> error_kind length", r["error_kind"] == "length" and "num_predict=768" in r["error"])
    r = OllamaPlanner(**e.kw(Script((200, {}, {"message": {"content": ""}, "done_reason": "stop"})))).plan(CT.messages("emp"), SCHEMA)
    check("empty reply -> error_kind empty", r["error_kind"] == "empty")
finally:
    e.close()

# ===================================================================================== 11. runner
e = Env()
try:
    env = {"ANTHROPIC_API_KEY": FAKE_ANT}
    t = Script(anth_ok(BAD), anth_ok(GOOD))
    p = AnthropicPlanner("claude-sonnet-5-5", env=env, **e.kw(t))
    run = P.PlannerRunner(p, CT)
    out = run.plan("put the apple in the fridge", ["x"], None)
    check("repair: invalid first, valid second", out["ok"] and out["repaired"] and not out["first_ok"] and out["calls"] == 2)
    check("plan normalised", out["plan"] == [{"action": "pick_and_place", "object": "Apple", "to": "Fridge", "arg": None}])
    msgs2 = t.calls[1]["body"]["messages"]
    check("repair turn carries failed reply + reasons", len(msgs2) == 3 and msgs2[1]["role"] == "assistant" and "unknown action" in msgs2[2]["content"] and "fly" in msgs2[1]["content"])
    out2 = run.plan("put the apple in the fridge", ["x"], None)
    check("runner replay fully cached, 0 USD", out2["cached_calls"] == 2 and out2["cost_usd"] == 0.0 and out2["ok"])
    t = Script(anth_ok(BAD))
    rj = P.PlannerRunner(AnthropicPlanner("claude-sonnet-5-5", env=env, **e.kw(t)), CT).plan("rej", [], None)
    check("rejected after one repair, nothing to execute", rj["rejected"] and rj["plan"] == [] and rj["calls"] == 2 and not rj["infra_failure"])
    t = Script(RuntimeError("down"))
    ri = P.PlannerRunner(AnthropicPlanner("claude-sonnet-5-5", env=env, max_retries=0, **e.kw(t)), CT).plan("infra", [], None)
    check("infra failure: no repair, not rejected", ri["infra_failure"] and ri["calls"] == 1 and not ri["rejected"] and not ri["ok"])
    t = Script(anth_ok(GOOD, stop="refusal"), anth_ok(GOOD))
    rf = P.PlannerRunner(AnthropicPlanner("claude-sonnet-5-5", env=env, **e.kw(t)), CT).plan("ref", [], None)
    check("refusal goes through the repair", rf["ok"] and rf["repaired"] and rf["calls"] == 2)
    t = Script(anth_ok(None, text="I think you should open it"), anth_ok(GOOD))
    rn = P.PlannerRunner(AnthropicPlanner("claude-sonnet-5-5", env=env, **e.kw(t)), CT).plan("prose", [], None)
    check("non-JSON reply repaired", rn["ok"] and rn["repaired"])
    t = Script(anth_ok({"say": "x", "plan": "no"}), anth_ok(GOOD))
    rs = P.PlannerRunner(AnthropicPlanner("claude-sonnet-5-5", env=env, **e.kw(t)), CT).plan("shape", [], None)
    check("shape error repaired", rs["ok"] and rs["repaired"] and "expected array" in t.calls[1]["body"]["messages"][2]["content"])
finally:
    e.close()

# ===================================================================================== 12. prices + hygiene
pt = P.PriceTable.load()
check("every Anthropic profile priced with source/date/status", all(m in pt.table["anthropic"] and all(pt.table["anthropic"][m].get(k) for k in ("source", "as_of", "status", "input", "output")) for m in PROFILES))
check("opus 3000/1000 = 0.032, sonnet 0.016, haiku 0.008", [round(pt.cost("anthropic", m, 3000, 1000), 6) for m in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5-20251001")] == [0.032, 0.016, 0.008])
try:
    pt.cost("anthropic", "claude-unknown", 1, 1)
    check("unknown Anthropic model never priced at 0", False)
except KeyError:
    check("unknown Anthropic model never priced at 0", True)
check("local / CLI providers free", pt.cost("ollama", "gemma4", 10 ** 6, 10 ** 6) == 0 and pt.cost("codex_cli", "gpt-6-sol", 10 ** 6, 10 ** 6) == 0)
leaks = []
for f in glob.glob(os.path.join(HERE, "planners", "*")):
    if os.path.isfile(f) and f.endswith((".py", ".json")):
        s = open(f, encoding="utf-8").read()
        leaks += [f for m in re.finditer(r"sk-[A-Za-z0-9_\-]{16,}", s)]
        if re.search(r"(?i)(api[_-]?key)\s*=\s*[\"'][A-Za-z0-9_\-]{12,}[\"']", s):
            leaks.append(f)
check("no key-shaped literal in sim/planners", not leaks, str(leaks))
check("package imports on bare stdlib (no anthropic / requests import)", not any(re.search(r"^\s*(import|from)\s+(anthropic|requests|openai)\b", open(f, encoding="utf-8").read(), re.M) for f in glob.glob(os.path.join(HERE, "planners", "*.py"))))

print(f"{PASS}/{PASS + FAIL}" + ("  OK" if not FAIL else "  FAILED"))
sys.exit(1 if FAIL else 0)
