"""K23 process ownership and LIFETIME-SAFE signalling (Astra K23-PREP r4 3-4, r5 3-4). WSL / Linux only.

Ownership:
  * a process is TAGGED when /proc/<pid>/environ carries KCAP_OWNER = the invocation token (or a prefix ending in "*")
    -- capture_one_k23.sh sets it at the flock launch and every process of the batch inherits it (ai2thor copies
    os.environ for Unity); a reused pid is a new exec without the token;
  * a session is VERIFIED when, in the same /proc snapshot, a tagged process (or an already-verified process, see
    below) is one of its members: while any member lives, the session number cannot be reused, so EVERY member of that
    session at that instant -- tagged or not (a child started with a cleared environment) -- is ours;
  * descendants (ppid chain) of an owned process are owned;
  * across repeated snapshots during one stop, an owned process is remembered as (pid, start time); a session stays
    verified while such a remembered member is still alive with the same start time (so tokenless members are not lost
    when the tagged ones exit first).
Signalling: every signal goes through a pidfd (os.pidfd_open) whose process is RE-VERIFIED (start time from
/proc/<pid>/stat after the pidfd was opened) before signal.pidfd_send_signal: a pid that died and was reused between the
snapshot and the signal is never hit. Nothing is ever signalled by number alone, by command line alone or by a record.
A stop is TERM -> wait (grace) -> SIGSTOP of the whole owned set (no new forks escape) -> re-snapshot -> SIGKILL, every
wait bounded by --deadline (epoch). Exit 0 = nothing owned left; 1 = something left (listed; the caller must treat its
batch as failed / pending); 2 = could not run.

  python k23_procs.py list   --token T [--session S] [--cmd REGEX]
  python k23_procs.py signal --token T --sig TERM [--session S] [--cmd REGEX]
  python k23_procs.py stop   --token T [--session S] [--cmd REGEX] [--grace 5] [--deadline EPOCH]
  python k23_procs.py record --file F     (is a published "pid boot starttime token" record still a live owned one?)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time

VAR = "KCAP_OWNER"


def boot_id() -> str:
    try:
        return open("/proc/sys/kernel/random/boot_id", encoding="utf-8").read().strip()
    except OSError:
        return ""


def _stat(pid: int):
    """(ppid, pgid, sid, start ticks) or None."""
    try:
        s = open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace").read()
        f = s.rsplit(")", 1)[1].split()
        return int(f[1]), int(f[2]), int(f[3]), int(f[19])
    except (OSError, IndexError, ValueError):
        return None


def token_of(pid: int) -> str:
    try:
        env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
    except OSError:
        return ""
    key = VAR.encode() + b"="
    for e in env:
        if e.startswith(key):
            return e[len(key):].decode("utf-8", "replace")
    return ""


def cmdline(pid: int) -> str:
    try:
        return open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return ""


def matches(tok: str, want: str) -> bool:
    if not tok:
        return False
    return tok.startswith(want[:-1]) if want.endswith("*") else tok == want


def snapshot() -> dict:
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        p = int(d)
        st = _stat(p)
        if st is None:
            continue
        out[p] = {"ppid": st[0], "pgid": st[1], "sid": st[2], "start": st[3], "token": token_of(p)}
    return out


def _excluded() -> set:
    """This process and its ancestors (the caller's own chain is never a target)."""
    ex, p = set(), os.getpid()
    while p > 1 and p not in ex:
        ex.add(p)
        st = _stat(p)
        if st is None:
            break
        p = st[0]
    return ex


def owned(snap: dict, want: str, session: int | None = None, state: dict | None = None) -> dict:
    """{pid: start} of every owned process in this snapshot (see the module docstring). A session is verified only
    when its LEADER (pid == sid) is tagged -- the sessions K23 itself creates with `setsid` (the batch script, the
    capture, the QC) -- never the caller's session (capture_one's, a person's shell). ``state`` = {"rem": {pid: start},
    "vsess": set()} carried across the snapshots of one stop: a verified session stays verified while a remembered
    member of it is alive with the same start time."""
    ex = _excluded()
    st = state if state is not None else {"rem": {}, "vsess": set()}
    rem, vsess = st.setdefault("rem", {}), st.setdefault("vsess", set())
    tagged = {p for p, v in snap.items() if matches(v["token"], want)}
    alive_rem = {p for p, s0 in rem.items() if p in snap and snap[p]["start"] == s0}
    pinned = {snap[p]["sid"] for p in alive_rem if snap[p]["sid"] in vsess}
    # r6 4: a session is verified by its tagged leader ONLY if that leader is still the same process (start time,
    # token) AFTER the whole scan: then the session number was pinned during the scan and cannot have been reused
    tagged = {p for p in tagged if (_stat(p) or (0, 0, 0, -1))[3] == snap[p]["start"] and matches(token_of(p), want)}
    leaders = {p for p in tagged if snap[p]["sid"] == p}
    pinned = {sid for sid in pinned if any((_stat(p) or (0, 0, 0, -1))[3] == rem[p] for p in alive_rem
                                           if snap[p]["sid"] == sid)}
    sessions = leaders | pinned
    if session is not None:
        sessions &= {session}
    vsess |= sessions
    own = {p for p, v in snap.items() if v["sid"] in sessions}
    own |= {p for p in tagged | alive_rem if session is None or snap[p]["sid"] == session}
    # descendants of owned processes (wherever their session). r7 (Astra r7 4): a ppid link is ownership evidence only
    # when, re-read AFTER the scan, the child is still the same process with the same parent, it started no earlier
    # than that parent, and the parent is still the owned process (same start time) -- a parent pid reused during the
    # enumeration (its new owner's children) links nothing. A link that cannot be verified now is caught by a later
    # snapshot (an orphan is re-parented to the holder, an owned subreaper).
    starts = {q: snap[q]["start"] for q in own}
    changed = True
    while changed:
        changed = False
        for p, v in snap.items():
            q = v["ppid"]
            if p in own or q not in own or v["start"] < starts[q]:
                continue
            cs, ps = _stat(p), _stat(q)
            if cs is None or ps is None or cs[0] != q or cs[3] != v["start"] or ps[3] != starts[q]:
                continue
            own.add(p)
            starts[p] = v["start"]
            changed = True
    out = {p: snap[p]["start"] for p in own if p not in ex}
    rem.update(out)
    return out


def safe_signal(pid: int, start: int, sig: int) -> bool:
    """Signal exactly the process (pid, start): pidfd first, then re-verify, then pidfd_send_signal."""
    try:
        fd = os.pidfd_open(pid)
    except OSError:
        return False
    try:
        st = _stat(pid)
        if st is None or st[3] != start:
            return False                                 # the pid now names another process: never signal it
        signal.pidfd_send_signal(fd, sig)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


PID_ONLY = None          # set by --pid: restrict every list / signal / stop to that one (owned) process


def _filter(own: dict, cmd: str | None) -> dict:
    if PID_ONLY is not None:
        own = {p: s for p, s in own.items() if p == PID_ONLY}
    if not cmd:
        return own
    rx = re.compile(cmd)
    return {p: s for p, s in own.items() if rx.search(cmdline(p))}


def do_list(want, session=None, cmd=None) -> dict:
    return _filter(owned(snapshot(), want, session), cmd)


def do_signal(want, sig, session=None, cmd=None) -> list:
    tgt = do_list(want, session, cmd)
    return sorted(p for p, s in tgt.items() if safe_signal(p, s, sig))


def do_stop(want, session=None, cmd=None, grace=5.0, deadline=None) -> dict:
    end = deadline if deadline else time.time() + grace + 15
    state = {"rem": {}, "vsess": set()}
    rem = _filter(owned(snapshot(), want, session, state), cmd)
    found = sorted(rem)
    if not rem:
        return {"found": [], "left": []}
    for p, s in rem.items():
        safe_signal(p, s, signal.SIGTERM)
    t_g = min(time.time() + grace, end - 4)
    while time.time() < t_g:
        cur = _filter(owned(snapshot(), want, session, state), cmd)
        rem.update(cur)
        rem = {p: s for p, s in rem.items() if (_stat(p) or (0, 0, 0, -1))[3] == s}
        if not rem:
            return {"found": found, "left": []}
        time.sleep(0.5)
    # freeze the whole owned set, re-snapshot until stable (a fork before the freeze is caught), then KILL
    for _ in range(3):
        cur = _filter(owned(snapshot(), want, session, state), cmd)
        rem.update(cur)
        for p, s in list(rem.items()):
            safe_signal(p, s, signal.SIGSTOP)
    for p, s in rem.items():
        safe_signal(p, s, signal.SIGKILL)
    while time.time() < end:
        rem = {p: s for p, s in rem.items() if (_stat(p) or (0, 0, 0, -1))[3] == s and _state(p) != "Z"}
        if not rem:
            break
        time.sleep(0.2)
    left = sorted(p for p, s in rem.items() if (_stat(p) or (0, 0, 0, -1))[3] == s and _state(p) != "Z")
    return {"found": found, "left": left, "left_cmd": {p: cmdline(p)[:120] for p in left}}


def _state(pid: int) -> str:
    try:
        return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return "?"


def record_owner(path: str) -> int:
    """The pid of a published record "pid boot_id starttime token" when it is still a live owned one: same boot, the
    leader alive with the recorded start time and carrying the token, or (leader gone) a tagged member of its session
    alive. 0 otherwise (stale: retire it, never signal it)."""
    try:
        pid, boot, st, tok = open(path, encoding="utf-8").read().split()[:4]
        pid, st = int(pid), int(st)
    except Exception:
        return 0
    if not tok or boot != boot_id():
        return 0
    s = _stat(pid)
    if s is not None:
        return pid if (s[3] == st and matches(token_of(pid), tok)) else 0
    snap = snapshot()
    return pid if any(v["sid"] == pid and matches(v["token"], tok) for v in snap.values()) else 0


def hold(rc_file: str, until: float, cmd: list) -> int:
    """Session HOLDER (Astra K23-PREP r5 3): run as the leader of a session K23 creates (`setsid`), become a child
    subreaper (orphans of the command are re-parented HERE, so they stay descendants), run the command, write its exit
    code to ``rc_file`` (atomic), then stay alive -- reaping -- while any descendant lives, so the session stays verified
    by this tagged leader until the batch script has stopped every member (tokenless ones too). Ends by itself when it
    is the last process, or at ``until`` (epoch) at the latest."""
    import ctypes
    import subprocess

    def write_rc(v):
        with open(rc_file + ".tmp", "w", encoding="utf-8") as fh:
            fh.write(str(v) + chr(10))
        os.replace(rc_file + ".tmp", rc_file)
    try:                                                 # r6 3: containment is REQUIRED, never best effort
        ok_ = ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0     # PR_SET_CHILD_SUBREAPER
    except Exception:
        ok_ = False
    if not ok_:
        print("k23_procs hold: could not become a child subreaper -- the command is NOT started", file=sys.stderr)
        write_rc(125)
        return 125
    # r6 3: the holder survives TERM / HUP / INT (a Python handler, reset to default in the child at exec) so the
    # session stays verified while its members are stopped; only the final KILL of the stop ends it
    for sg in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sg, lambda *_a: None)
    try:
        child = subprocess.Popen(cmd)
    except Exception as exc:
        print(f"k23_procs hold: cannot start {cmd[:2]}: {exc!r}", file=sys.stderr)
        rc = 127
    else:
        try:                                             # r6 3: bounded by --until
            rc = child.wait(timeout=max(1.0, until - time.time()))
        except subprocess.TimeoutExpired:
            child.kill()                                 # our own unreaped child: its pid cannot be reused yet
            try:
                child.wait(timeout=10)                   # r7 (Astra r7 3): bounded too
            except subprocess.TimeoutExpired:
                pass
            rc = 124
        rc = rc if rc >= 0 else 128 - rc
    write_rc(rc)
    while time.time() < until:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return rc                                    # no child left: nothing of the session to keep verified
        if pid == 0:
            time.sleep(0.2)
    # r7 (Astra r7 3): --until reached with adopted descendants still alive -> never leave them without their owner:
    # stop every descendant of this holder (verified links, pidfd signals, STOP -> re-scan -> KILL), reap, bounded
    left = kill_descendants(os.getpid(), time.time() + 15)
    if left:
        print(f"k23_procs hold: descendants left after --until: {left}", file=sys.stderr)
        return 99
    return rc


def descendants(root: int, snap: dict) -> dict:
    """{pid: start} of the verified descendants of ``root`` (same post-scan link rule as owned())."""
    st0 = _stat(root)
    if st0 is None:
        return {}
    own, starts = {root}, {root: st0[3]}
    changed = True
    while changed:
        changed = False
        for p, v in snap.items():
            q = v["ppid"]
            if p in own or q not in own or v["start"] < starts[q]:
                continue
            cs, ps = _stat(p), _stat(q)
            if cs is None or ps is None or cs[0] != q or cs[3] != v["start"] or ps[3] != starts[q]:
                continue
            own.add(p)
            starts[p] = v["start"]
            changed = True
    return {p: starts[p] for p in own if p != root}


def kill_descendants(root: int, end: float) -> list:
    """STOP every verified descendant (re-scanned 3 times: a fork before the freeze is caught), KILL, reap; bounded
    by ``end``. Returns the pids still alive (not zombies) at the end."""
    rem = {}
    for _ in range(3):
        rem.update(descendants(root, snapshot()))
        for p, s_ in list(rem.items()):
            safe_signal(p, s_, signal.SIGSTOP)
    for p, s_ in rem.items():
        safe_signal(p, s_, signal.SIGKILL)
    while time.time() < end:
        try:
            while os.waitpid(-1, os.WNOHANG)[0] > 0:
                pass
        except ChildProcessError:
            pass
        rem = {p: s_ for p, s_ in rem.items() if (_stat(p) or (0, 0, 0, -1))[3] == s_ and _state(p) != "Z"}
        if not rem:
            return []
        time.sleep(0.2)
    return sorted(rem)


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["hold"]:
        if "--" not in argv:
            print("usage: hold --rc-file F --until EPOCH -- cmd ...", file=sys.stderr)
            return 2
        i = argv.index("--")
        hp = argparse.ArgumentParser()
        hp.add_argument("--rc-file", required=True)
        hp.add_argument("--until", type=float, required=True)
        ha = hp.parse_args(argv[1:i])
        return hold(ha.rc_file, ha.until, argv[i + 1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("list", "signal", "stop", "record"))
    ap.add_argument("--token", default="")
    ap.add_argument("--session", type=int, default=None)
    ap.add_argument("--cmd", dest="cmdrx", default=None)
    ap.add_argument("--sig", default="TERM")
    ap.add_argument("--grace", type=float, default=5.0)
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--file", default=None)
    ap.add_argument("--pid", type=int, default=None, help="only this pid, and only when it is owned (a batch child)")
    a = ap.parse_args(argv)
    global PID_ONLY
    PID_ONLY = a.pid
    try:
        if a.cmd == "record":
            p = record_owner(a.file)
            print(p)
            return 0 if p else 1
        if not a.token or a.token in ("*", "k23:*"):
            print(json.dumps({"error": "a specific token (or scene prefix k23:<scene>:*) is required"}))
            return 2
        if a.cmd == "list":
            r = do_list(a.token, a.session, a.cmdrx)
            print(" ".join(str(p) for p in sorted(r)))
            return 0
        if a.cmd == "signal":
            print(json.dumps({"signalled": do_signal(a.token, getattr(signal, "SIG" + a.sig), a.session, a.cmdrx)}))
            return 0
        r = do_stop(a.token, a.session, a.cmdrx, a.grace, a.deadline)
        print(json.dumps(r))
        return 0 if not r["left"] else 1
    except Exception as exc:
        print(json.dumps({"error": repr(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
