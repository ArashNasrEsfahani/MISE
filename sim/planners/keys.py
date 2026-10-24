"""Key handling for the planner layer. Keys are read at call time, held only in a ``Secret`` (repr is masked) and
never written anywhere. ``redact`` scrubs key-like strings AND the exact loaded values from any text that may be
logged, cached or raised."""
from __future__ import annotations

import os
import re

from .base import PlannerSkip

DEFAULT_OPENROUTER_ENV_FILE = r"D:\KareOne\RAG Paper\.env"

_SK = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")                      # sk-ant-..., sk-or-v1-..., sk-proj-...
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{8,}")
_KV = re.compile(r"(?i)(x-api-key|api[_-]?key|authorization|token|secret)(\"?\s*[:=]\s*\"?)[A-Za-z0-9._\-]{8,}")
#: C9 (2026-10-13): the one benign `_SK` shape. Inside an English word ("task-sampling", "task-specific") `_SK` matched from the word's last
#: two letters, so registered prose (analyze.ANALYSIS_PLAN, hashed into freeze.json) read as a key. A match is left alone ONLY when the
#: character before it is an ASCII letter that does not end a JSON / Python escape (\n \t \r \b \f \uXXXX) AND everything after "sk-" is
#: lowercase letters and hyphens. Every known key format (sk-ant-api03-..., sk-or-v1-..., sk-proj-...) carries a digit, a capital or an
#: underscore, so it is still caught in every context, and the exact loaded secret is replaced first wherever it stands. Disclosed
#: narrowing: an all-lowercase, digit-free "sk-..." glued to a preceding letter is not flagged.
_SK_PROSE_TAIL = re.compile(r"[a-z\-]+")
_ESCAPE_END = re.compile(r"\\(?:[ntrbf]|u[0-9A-Fa-f]{4})$")


def _sk_sub(m) -> str:
    i, s = m.start(), m.string
    prev = s[i - 1] if i else ""
    if (prev.isascii() and prev.isalpha() and not _ESCAPE_END.search(s[max(0, i - 6):i])
            and _SK_PROSE_TAIL.fullmatch(m.group(0)[3:])):
        return m.group(0)
    return "sk-***"


class Secret:
    """Holds a key. ``str``/``repr`` are masked; ``reveal()`` is the only way to read it (building a header)."""
    __slots__ = ("_v",)

    def __init__(self, value: str):
        self._v = value

    def reveal(self) -> str:
        return self._v

    def __repr__(self):
        return "Secret(***)"

    __str__ = __repr__


def redact(text, extra=()) -> str:
    s = str(text)
    for e in extra:
        v = e.reveal() if isinstance(e, Secret) else e
        if v and len(v) >= 4:
            s = s.replace(v, "***")
    s = _SK.sub(_sk_sub, s)
    s = _BEARER.sub("Bearer ***", s)
    return _KV.sub(lambda m: m.group(1) + m.group(2) + "***", s)


def _windows_user_env(name: str) -> str:
    """Read a per-user environment variable from the Windows registry (HKCU\\Environment), read-only.
    `setx` writes there, but processes started before it never see it in os.environ."""
    if os.name != "nt":
        return ""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return str(winreg.QueryValueEx(k, name)[0] or "").strip()
    except OSError:
        return ""


def get_anthropic_key(env=None) -> Secret:
    # THESIS_ANTHROPIC_API_KEY first: a variable literally named ANTHROPIC_API_KEY is also picked up by Claude Code
    # itself (it then bills the API instead of the user's subscription), so the experiments use their own name.
    explicit = env is not None
    env = os.environ if env is None else env
    v = (env.get("THESIS_ANTHROPIC_API_KEY") or env.get("ANTHROPIC_API_KEY") or "").strip()
    if not v and not explicit:
        v = _windows_user_env("THESIS_ANTHROPIC_API_KEY") or _windows_user_env("ANTHROPIC_API_KEY")
    if not v:
        raise PlannerSkip("ANTHROPIC_API_KEY is not set: live Anthropic planning skipped")
    return Secret(v)


def parse_env_file(path: str) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:]
            k, _, v = line.partition("=")
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            out[k.strip()] = v
    return out


def get_openrouter_key(env=None) -> Secret:
    """Key from the env file named by OPENROUTER_ENV_FILE (default: the KareOne RAG Paper .env). The file is read at
    each call; the key is never copied anywhere."""
    env = os.environ if env is None else env
    path = env.get("OPENROUTER_ENV_FILE") or DEFAULT_OPENROUTER_ENV_FILE
    if not os.path.isfile(path):
        raise PlannerSkip("OpenRouter env file not found: OpenRouter planning skipped")
    try:
        v = (parse_env_file(path).get("OPENROUTER_API_KEY") or "").strip()
    except OSError:
        raise PlannerSkip("OpenRouter env file unreadable: OpenRouter planning skipped") from None
    if not v:
        raise PlannerSkip("OPENROUTER_API_KEY missing in the env file: OpenRouter planning skipped")
    return Secret(v)
