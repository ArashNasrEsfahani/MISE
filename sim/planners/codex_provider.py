"""Codex CLI provider: a MODEL-PLUS-CLI SYSTEM, exploratory, reported separately (preregistration s3.4).

Each call is a fresh ``codex exec -m <model> --skip-git-repo-check --ephemeral -s read-only -C <empty temp dir>
--ignore-user-config --ignore-rules -c model_reasoning_effort="high" --json --output-schema <file> -o <file>`` with the
prompt on stdin (the two ignore flags and the pinned effort were added in P3: before that the user's config.toml set
the effort, personality and service tier silently). The temp dir holds only the schema file and is
deleted afterwards; the child environment has every API-key variable removed. There is no repository, gold file,
personal memory or browsing configured. What the CLI adds beyond the prompt (its own instructions, tool access) is
NOT controlled by these flags (Astra): the archive keeps the CLI version, the exact flags, the JSONL events, the
usage numbers, the wall time, the CLI-reported minus estimated input tokens (scaffolding overhead) and a count of
tool events -- a non-zero count marks the call as an isolation violation. The first call (r = 0) is the scored one;
decoding is not controllable, so repeats measure variance only.

A21 E (G-blockers, 2026-10-12; the X0 / X+M arms are MAIN comparators from A18):
  * hardened tool surface: sixteen CLI features are disabled (`HARDEN_FEATURES`: apps, shell tool, browser, computer use, multi-agent, plugins, hooks,
    image generation, skill search, image viewer, sleep, unified exec, goals, remote plugins, in-app browser). Measured on codex-cli 0.162.1: the
    default tool list had 24 entries and 14.1 k input tokens for a one-word prompt, the hardened one 15 entries and 9.0 k. Some built-in tools
    (code-mode exec, apply_patch, MCP resource listing) cannot be switched off from the command line; their USE is detected, not prevented;
  * clean sessions: `--ephemeral`, an empty temp cwd per call, no repository, user config and rules ignored, API-key variables removed, and the
    session is read-only (the sandbox may still allow reading outside the cwd; the canary probe `x_isolation_probe.py` checks that a file outside
    the cwd cannot be read, and every call is scanned for tool events);
  * identity: the CLI's own SessionConfiguredEvent (stderr, `RUST_LOG=codex_exec=info,off`; the user's account fields are never stored) gives the
    model, reasoning effort, approval policy and sandbox the CLI configured the session with (logged as `cli_configured_model`; `served_model` stays None). Any difference from the frozen values is an
    `identity_violation` (the planning phase aborts). The CLI event stream does NOT expose the model the server finally answered with, so silent
    SERVER-SIDE substitution cannot be excluded from here; it is disclosed, not claimed away;
  * the CLI version is pinned: with `pin_cli_version` set (freeze.json's `codex_cli_version`) a different version skips the planner before any call;
  * fixed retries: one transmission per attempt, no transport retries of our own (the CLI has its own, invisible); a transport failure is `infra`
    and is re-queued by plan.py in at most 2 later sessions (prereg s13). Reasoning effort is pinned to "high".
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time

from .base import PlannerProvider, PlannerSkip, TransportFailure, estimate_tokens, messages_text
from .contract import flat_prompt

_KEY_ENV = ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_ENV_FILE", "OPENAI_API_KEY")
#: pinned in P3 to the value the P2 smoke ran with (inherited then from the user config); Astra P3: pin "high"
REASONING_EFFORT = "high"
_TOOL_ITEMS = ("command_execution", "mcp_tool_call", "web_search", "file_change", "tool_call", "function_call")
#: items that are not tool use; any OTHER item type in the event stream counts as a tool event (unknown tool kinds are caught too)
_BENIGN_ITEMS = ("agent_message", "assistant_message", "reasoning", "error")
#: A21 E: CLI features switched off for every X call (names as printed by `codex features list` on 0.162.1; a name the CLI does not know is an error)
HARDEN_FEATURES = ("apps", "shell_tool", "browser_use", "browser_use_external", "computer_use", "multi_agent", "plugins", "hooks", "image_generation",
                   "skill_search", "view_image", "sleep_tool", "unified_exec", "goals", "remote_plugin", "in_app_browser")
#: the child's log filter: only the `codex_exec` target at info level (it prints SessionConfiguredEvent); everything else, including the
#: account fields of the other targets, stays off
SESSION_LOG = "codex_exec=info,off"
_SESSION_LINE = re.compile(r"SessionConfiguredEvent \{.*")


def parse_session(stderr: str) -> dict:
    """The facts of the session the CLI started, from its own SessionConfiguredEvent line. Stores NO path and NO account field.
    -> {'found', 'model', 'reasoning_effort', 'approval_policy', 'sandbox_read_only', 'network_restricted'}."""
    m = _SESSION_LINE.search(stderr or "")
    if not m:
        return {"found": False}
    line = m.group(0)

    def one(rx):
        mm = re.search(rx, line)
        return mm.group(1) if mm else None
    prof = line[line.find("permission_profile:"):] if "permission_profile:" in line else ""
    prof = prof.split("active_permission_profile", 1)[0]
    return {"found": True, "model": one(r'\bmodel: "([^"]+)"'), "reasoning_effort": one(r"reasoning_effort: (?:Some\()?(\w+)"),
            "approval_policy": one(r"approval_policy: (\w+)"),
            "sandbox_read_only": bool(prof) and "Managed" in prof and "access: Read" in prof and "access: Write" not in prof and "access: ReadWrite" not in prof,
            "network_restricted": "network: Restricted" in prof}


def identity_problems(sess: dict, model: str, effort, cli_version: str, pin) -> list:
    """Differences between the session the CLI started and the frozen configuration (empty list = identity ok)."""
    out = []
    if not sess.get("found"):
        return ["session_event_missing"]
    if sess.get("model") != model:
        out.append(f"model {sess.get('model')!r} != requested {model!r}")
    if effort and str(sess.get("reasoning_effort") or "").lower() != str(effort).lower():
        out.append(f"reasoning_effort {sess.get('reasoning_effort')!r} != {effort!r}")
    if sess.get("approval_policy") != "Never":
        out.append(f"approval_policy {sess.get('approval_policy')!r} != 'Never'")
    if not sess.get("sandbox_read_only"):
        out.append("sandbox is not the read-only managed profile")
    if not sess.get("network_restricted"):
        out.append("network is not restricted")
    if pin and cli_version != pin:
        out.append(f"cli version {cli_version!r} != pinned {pin!r}")
    return out


def tool_events(events) -> int:
    """Events that are tool use: the known tool item types, or ANY item type that is not plain message / reasoning text."""
    n = 0
    for ev in events:
        it = ev.get("item") if isinstance(ev.get("item"), dict) else None
        typ = str((it or {}).get("type", ev.get("type", "")))
        if typ in _TOOL_ITEMS or (it is not None and typ not in _BENIGN_ITEMS):
            n += 1
    return n


def strict_schema(schema: dict) -> dict:
    """The transmitted schema: OpenAI strict structured output needs ``additionalProperties: false`` on every object
    (found in the P2 smoke: Codex rejected the bare PLAN_SCHEMA, HTTP 400). Same transform as the Anthropic arm."""
    from .anthropic_provider import to_anthropic_schema
    return to_anthropic_schema(schema)


def _run(cmd, input_text, cwd, env, timeout):
    return subprocess.run(cmd, input=input_text, cwd=cwd, env=env, capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout)


class CodexCliPlanner(PlannerProvider):
    provider = "codex_cli"
    exploratory = True
    role_mapping = "system+user concatenated under '### SYSTEM' / '### USER' headers in one stdin prompt"

    def __init__(self, model: str, cli: str = "codex", sandbox: str = "read-only", extra_args=(), runner=None,
                 which=shutil.which, timeout_s: float = 300.0, tmp_root: str | None = None,
                 reasoning_effort: str | None = REASONING_EFFORT, pin_cli_version: str | None = None, hardened: bool = True,
                 verify_identity: bool = True, **kw):
        super().__init__(model, timeout_s=timeout_s, **kw)
        self.cli, self.sandbox, self.extra_args = cli, sandbox, list(extra_args)
        self._runner, self._which, self.tmp_root = runner or _run, which, tmp_root
        self.reasoning_effort = reasoning_effort
        self.pin_cli_version, self.hardened, self.verify_identity = pin_cli_version, bool(hardened), bool(verify_identity)
        self._version = None

    def _isolation_args(self) -> list:
        """P3 (Astra): the user's ~/.codex config.toml (reasoning effort, personality, service tier, ...) and execpolicy
        rules are NOT loaded; the reasoning effort is pinned explicitly so it is recorded, not inherited."""
        out = ["--ignore-user-config", "--ignore-rules"]
        if self.hardened:
            for f in HARDEN_FEATURES:                                         # A21 E: the tool surface that CAN be switched off is off
                out += ["--disable", f]
        if self.reasoning_effort:
            out += ["-c", f'model_reasoning_effort="{self.reasoning_effort}"']
        return out

    def decoding(self) -> dict:
        return {"cli_flags": ["exec", "-m", self.model, "--skip-git-repo-check", "--ephemeral", "-s", self.sandbox]
                + self._isolation_args() + ["--json", "--output-schema", "-o"] + self.extra_args,
                "cli_version": self.cli_version(), "reasoning_effort": self.reasoning_effort,
                "user_config": "ignored", "sampling": "not controllable (CLI defaults)", "hardened": self.hardened,
                "pinned_cli_version": self.pin_cli_version}

    def check_available(self):
        if not self._which(self.cli):
            raise PlannerSkip(f"Codex CLI {self.cli!r} not found on PATH: Codex planning skipped")
        if self.pin_cli_version and self.cli_version() != self.pin_cli_version:
            raise PlannerSkip(f"Codex CLI version pin mismatch: installed {self.cli_version()!r}, frozen {self.pin_cli_version!r}; "
                              "no call was made (A21 E: a different CLI is a different system)")

    def cli_version(self) -> str:
        if self._version is None:
            try:
                exe = self._which(self.cli) or self.cli
                r = self._runner([exe, "--version"], None, None, self._child_env(), 30)
                self._version = (r.stdout or r.stderr or "").strip().splitlines()[0] if (r.stdout or r.stderr) else "unknown"
            except Exception as e:
                self._version = f"unavailable ({type(e).__name__})"
        return self._version

    @staticmethod
    def _child_env() -> dict:
        env = {k: v for k, v in os.environ.items() if k not in _KEY_ENV}
        env["RUST_LOG"] = SESSION_LOG                       # A21 E: prints SessionConfiguredEvent and nothing from the account-bearing targets
        return env

    def _call(self, messages, schema, budget) -> dict:
        prompt = flat_prompt(messages)
        tmp = tempfile.mkdtemp(prefix="codex_planner_", dir=self.tmp_root)
        t0 = time.time()
        try:
            schema_path, out_path = os.path.join(tmp, "schema.json"), os.path.join(tmp, "out.txt")
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(strict_schema(schema), f)
            exe = self._which(self.cli) or self.cli
            cmd = [exe, "exec", "-m", self.model, "--skip-git-repo-check", "--ephemeral", "-s", self.sandbox, "-C", tmp] \
                + self._isolation_args() + ["--json", "--output-schema", schema_path, "-o", out_path] + self.extra_args
            self._wire_sha = _sha(prompt + "\n" + json.dumps(cmd[1:], sort_keys=True))
            rid = self.begin_transmission()                       # 0 USD, but the ledger sees the call
            try:
                r = self._runner(cmd, prompt, tmp, self._child_env(), self.timeout_s)
            except subprocess.TimeoutExpired:
                raise TransportFailure(f"codex exec timed out after {self.timeout_s:.0f}s") from None
            except OSError as e:
                raise TransportFailure(f"codex exec failed to start: {type(e).__name__}") from None
            self._ok_rid = rid
            events = []
            for line in (r.stdout or "").splitlines():
                try:
                    ev = json.loads(line)
                    if isinstance(ev, dict):
                        events.append(ev)
                except ValueError:
                    pass
            raw = ""
            if os.path.isfile(out_path):
                with open(out_path, encoding="utf-8") as f:
                    raw = f.read()
            if not raw.strip():
                raw = _last_agent_message(events)
            tin, tout = _usage(events)
            tools = tool_events(events)
            sess = parse_session(r.stderr or "")
            idp = identity_problems(sess, self.model, self.reasoning_effort, self.cli_version(), self.pin_cli_version) if self.verify_identity else []
            err_lines = [ln for ln in (r.stderr or "").splitlines() if ln.strip() and "SessionConfiguredEvent" not in ln and "Sent prompt" not in ln]
            est_in = estimate_tokens(messages_text(messages))
            err = kind = ""
            if r.returncode != 0:
                err, kind = f"codex exec exit code {r.returncode}", "transport"
            elif not raw.strip():
                err, kind = "empty reply", "empty"
            return {"raw": raw, "tokens_in": tin, "tokens_out": tout, "stop_reason": f"exit {r.returncode}",
                    "error": err, "error_kind": kind, "cost_usd": 0.0,
                    "extra": {"cli_version": self.cli_version(), "cli_flags": self.decoding()["cli_flags"],
                              "events_n": len(events), "tool_events": tools, "isolation_violation": tools > 0,
                              "wall_s": round(time.time() - t0, 2), "est_prompt_tokens": est_in,
                              "overhead_tokens_in": (tin - est_in) if tin else None,
                              "usage_reported": bool(tin or tout), "events": events[:200],
                              "stderr_tail": "\n".join(err_lines)[-300:],
                              "cli_configured_model": sess.get("model"), "served_model": None, "session": {k: v for k, v in sess.items() if k != "found"} if sess.get("found") else None,
                              "requested_model": self.model, "identity_violation": bool(idp), "identity_problems": idp,
                              "pin_cli_version": self.pin_cli_version, "server_side_model": "not observable through the CLI"}}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _last_agent_message(events) -> str:
    for ev in reversed(events):
        it = ev.get("item") or {}
        if it.get("type") in ("agent_message", "assistant_message") and it.get("text"):
            return it["text"]
        if ev.get("type") == "agent_message" and ev.get("message"):
            return ev["message"]
    return ""


def _usage(events):
    """Best-effort token usage from the JSONL event stream: the last event carrying input/output token counts."""
    tin = tout = 0

    def find(n):
        if isinstance(n, dict):
            if "input_tokens" in n and "output_tokens" in n:
                return int(n["input_tokens"] or 0), int(n["output_tokens"] or 0)
            for v in n.values():
                r = find(v)
                if r:
                    return r
        elif isinstance(n, list):
            for v in n:
                r = find(v)
                if r:
                    return r
        return None

    for ev in events:
        r = find(ev)
        if r:
            tin, tout = r
    return tin, tout
