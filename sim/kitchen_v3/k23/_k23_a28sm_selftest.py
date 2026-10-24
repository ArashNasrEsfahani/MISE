"""Hermetic checks for the A28-ii-SM amendment in k23_decide (Arash 2026-10-19 ~00:27; Astra K23-A28SM-r1 REJECT -> r2):
a pause is exempt (does not end the window) only for the ONE marker pinned with its guard CSV by a bound record, with
conditions recomputed from those files, Arash's heading and a pinned affirmative review of THIS code; stop records are
lifted only when all their evidence is that pause. Temp dirs only; no WSL, no capture.
    python sim/kitchen_v3/k23/_k23_a28sm_selftest.py"""
import hashlib
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import k23_decide as D  # noqa: E402

N = [0, 0]


def ok(c, msg):
    N[0] += 1
    N[1] += bool(c)
    print(("PASS: " if c else "FAIL: ") + msg)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


INV = "abcd1234ef567890abcd1234ef567890"
I8 = INV[:8]
HEAD = [f"## {D.A28SM_ENTRY}"]
MF = f"pause_20261019-001102-{I8}.json"
CF = f"session_20261019-001102-{I8}.csv"
REPLAY = [(870.0, 99.0), (885.0, 93.0), (795.0, 95.0), (885.0, 100.0), (780.0, 92.0)]
REASON = D.A28SM_REASON + " (" + str(REPLAY) + ")"
TMP_ASTRA = tempfile.mkdtemp()
CODE = [sha(os.path.join(HERE, "k23_decide.py")), sha(os.path.join(HERE, "thermal_guard.py"))]


def astra(text):
    p = os.path.join(TMP_ASTRA, "K23-A28SM-r2.txt")
    open(p, "w", encoding="utf-8").write(text)
    D.A28SM_ASTRA = p
    return p


GOOD_ASTRA = "VERDICT: APPROVE\ncode sha256: " + " ".join(CODE) + "\n"


def build(temps=(56.0,) * 5, reason=REASON, test=False, induced=False, flag="Active, Not Active", flag_over=None,
          rows_over=None, rec_over=None, inv8=I8, astra_text=GOOD_ASTRA, extra_lines=""):
    root = tempfile.mkdtemp()
    d = os.path.join(root, "thermal")
    os.makedirs(d)
    mf, cf = f"pause_20261019-001102-{inv8}.json", f"session_20261019-001102-{inv8}.csv"
    samples = [{"ts": f"2026/10/19 00:1{2 + i}:04.4{i}1", "temp": t, "sm": REPLAY[i % 5][0], "util": REPLAY[i % 5][1]}
               for i, t in enumerate(temps)]
    mk = {"paused_at": "2026-10-19 00:16:04", "reason": reason, "test": test, "induced": induced, "last_samples": samples}
    json.dump(mk, open(os.path.join(d, mf), "w", encoding="utf-8"))
    with open(os.path.join(d, cf), "w", encoding="utf-8") as fh:
        fh.write("wall_time,timestamp,temperature.gpu,clocks.sm,power.draw,utilization.gpu\n# session start x\n")
        fh.write("2026-10-19 00:11:04,2026/10/19 00:11:04.330, 55, 780, 18.80, 87\n# corun True thermal_flags "
                 f"{flag!r} -> True hw False\n")
        for i, s in enumerate(samples):
            row = f"2026-10-19 00:1{2 + i}:04,{s['ts']}, {s['temp']:g}, {s['sm']:g}, 23.00, {s['util']:g}\n"
            fh.write((rows_over or {}).get(i, row))
            fh.write(f"# corun True thermal_flags {(flag_over or {}).get(i, flag)!r} -> True hw False\n")
        fh.write(extra_lines)
        fh.write("# PAUSE x\n")
    ap = astra(astra_text)
    rec = {"marker": mf, "marker_sha256": sha(os.path.join(d, mf)), "guard_csv": cf,
           "guard_csv_sha256": sha(os.path.join(d, cf)), "astra_file": ap, "astra_sha256": sha(ap)}
    rec.update(rec_over or {})
    json.dump(rec, open(os.path.join(d, f"a28sm_{I8}.json"), "w", encoding="utf-8"))
    return root, d


root, d = build()
ok(D.a28sm_exempt("X", I8, d, HEAD) == [], "valid record, 56 C, hardware flag inactive in every row: exempt")
ok(D.a28sm_exempt("X", I8, d, []) != [], "no Arash ledger heading: not exempt")
ok(D.a28sm_exempt("X", I8, d, HEAD, marker_file=MF) == [], "the pinned marker itself: exempt")
ok(D.a28sm_exempt("X", I8, d, HEAD, marker_file=f"pause_20261019-002000-{I8}.json") != [],
   "another marker of the same invocation: not exempt")
CASES = [
    ("record missing", "rm"),
    ("review not pinned (other path)", dict(rec_over={"astra_file": "docs/final-test/_astra/K3-AMB-B4-r1.txt"})),
    ("review sha mismatch", dict(rec_over={"astra_sha256": "0" * 64})),
    ("review is a REJECT", dict(astra_text="VERDICT: REJECT\n" + " ".join(CODE))),
    ("review of other code", dict(astra_text="VERDICT: APPROVE\ncode sha256: " + "1" * 64)),
    ("one sample at 80 C", dict(temps=(56.0, 56.0, 80.0, 56.0, 56.0))),
    ("NaN temperature", dict(temps=(56.0, 56.0, float("nan"), 56.0, 56.0))),
    ("only 4 samples", dict(temps=(56.0,) * 4)),
    ("temperature rule", dict(reason="GPU temperature >= 87 C in 5 consecutive samples ([87, 87, 87, 87, 87])")),
    ("no-telemetry rule", dict(reason="no telemetry for 301 s (>= 300 s)")),
    ("pause-test marker", dict(test=True, induced=True)),
    ("hardware flag active in one row", dict(flag_over={2: "Active, Active"})),
    ("unreadable flag pair", dict(flag_over={4: "[N/A], [N/A]"})),
    ("flag line without a pair (None)", dict(extra_lines="2026-10-19 00:17:04,2026/10/19 00:17:04.1, 56, 800, 20, 95\n"
                                                         "# corun True thermal_flags None -> None hw None\n")),
    ("a NO_TELEMETRY row", dict(extra_lines="2026-10-19 00:17:04,NO_TELEMETRY\n")),
    ("a row with empty rule fields", dict(extra_lines="2026-10-19 00:06:00,t6,,,,\n# corun True thermal_flags "
                                                      "'Active, Not Active' -> True\n")),
    ("an orphan [N/A] flag line", dict(extra_lines="# corun True thermal_flags '[N/A], [N/A]' -> None\n")),
    ("an orphan Active, Active flag line", dict(extra_lines="# corun True thermal_flags 'Active, Active' -> True\n")),
    ("an orphan readable flag line", dict(extra_lines="# corun True thermal_flags 'Active, Not Active' -> True\n")),
    ("a row without its flag line", dict(extra_lines="2026-10-19 00:17:04,2026/10/19 00:17:04.1, 56, 800, 20, 95\n")),
    ("pause sample not in the CSV", dict(rows_over={1: "2026-10-19 00:13:04,2026/10/19 00:13:04.999, 56, 885, 23, 93\n"})),
    ("pause sample with other values", dict(rows_over={3: "2026-10-19 00:15:04,2026/10/19 00:15:04.431, 79, 885, 23, 100\n"})),
    ("marker of another invocation", dict(inv8="ffff0000")),
]
for name, kw in CASES:
    if kw == "rm":
        r2, d2 = build()
        os.remove(os.path.join(d2, f"a28sm_{I8}.json"))
    else:
        r2, d2 = build(**kw)
    ok(D.a28sm_exempt("X", I8, d2, HEAD) != [], f"not exempt: {name}")
    shutil.rmtree(r2, ignore_errors=True)
for fname in (MF, CF):
    r3, d3 = build()
    open(os.path.join(d3, fname), "a").write(" ")
    ok(D.a28sm_exempt("X", I8, d3, HEAD) != [], f"not exempt: {fname[:7]} changed after the record (sha)")
    shutil.rmtree(r3, ignore_errors=True)

# consumers: pause_markers flags only the pinned marker; real_pauses ignores it, counts everything else
root, d = build()
mks = D.pause_markers("X", "2026-10-20", d)
ok(len(mks) == 1 and mks[0].get("exempt") is True, "pause_markers: the pinned marker is flagged exempt")
ok(D.real_pauses(mks, None, "amb", 0) == [], "real_pauses: the exempt marker is not a real pause")
json.dump({"paused_at": "x", "reason": "GPU temperature >= 87 C in 5 consecutive samples ([87,87,87,87,87])", "test": False,
           "induced": False, "last_samples": []}, open(os.path.join(d, f"pause_20261019-002000-{I8}.json"), "w"))
mks2 = D.pause_markers("X", "2026-10-20", d)
hot = [m for m in mks2 if m["file"].endswith("002000-" + I8 + ".json")]
ok(len(mks2) == 2 and hot and not hot[0].get("exempt"), "pause_markers: a second (87 C) marker of the same invocation stays real")
ok(D.real_pauses(mks2, None, "amb", 0) != [], "real_pauses: the second marker counts")
os.remove(os.path.join(d, f"pause_20261019-002000-{I8}.json"))
ok(D.pause_markers("X", "2026-10-22", d) == [], "a marker two days before the deadline does not belong to it (unchanged)")

# stop records (scene_dir pointed at the temp scene folder)
EVID = (f"marker {MF}: {REASON[:120]}; the receipt carries a pause marker that is not the pause-test trigger "
        f"({REASON[:120]}); the receipt / exit code (wsl rc 86) shows a pause that is not a proven pause-test trigger")
orig = D.scene_dir
try:
    D.scene_dir = lambda s: root
    T1 = "THERMAL PAUSE: a pause ends this capture window (no restart): " + EVID
    T2 = "THERMAL PAUSE (persisted first): " + EVID

    def lift(why, inv=INV):
        sess = {"terminal": {"why": why, "t": 1, "invocation": inv, "window_ended": True}, "resolutions": []}
        D.resolve_terminal("X", sess)
        return sess["terminal"] is None, sess
    lifted, sess = lift(T1 + " | earlier: " + T2)
    ok(lifted and sess["resolutions"][-1]["rule"] == "A28-ii-SM",
       "resolve_terminal: duplicate persistence of the one exempt pause is lifted (recorded)")
    ok(lift(T2)[0], "resolve_terminal: a single exempt-pause stop is lifted")
    ok(not lift("no telemetry for 301 s (>= 300 s)")[0], "resolve_terminal: a no-telemetry stop of that invocation is NOT lifted")
    ok(not lift(T1 + " | earlier: THERMAL PAUSE (persisted first): marker x.json: GPU temperature >= 87 C")[0],
       "resolve_terminal: an independent earlier stop keeps the window closed")
    ok(not lift(T1 + "; the receipt shows something else")[0], "resolve_terminal: an extra cause in the segment keeps it closed")
    for tail in (" AND no telemetry for 301 s", " AND GPU temperature >= 87 C"):
        ok(not lift(T2.replace(f"({REASON[:120]})", f"({REASON[:120]}){tail}"))[0],
           f"resolve_terminal: text appended to an evidence line ({tail.strip()}) keeps it closed")
        bad_m = T2.replace(f"marker {MF}: {REASON[:120]}", f"marker {MF}: {REASON[:120]}{tail}")
        ok(not lift(bad_m)[0], f"resolve_terminal: text appended to the marker line ({tail.strip()}) keeps it closed")
    ok(not lift(T2, inv="ffff0000" + INV[8:])[0], "resolve_terminal: a stop of another invocation is not lifted")
    cp = D.closure_path("X", "2026-10-20")
    json.dump({"window_ended": True, "events": [{"t": 1, "invocation": INV, "why": EVID}]}, open(cp, "w"))
    ok(D.closure_exempt("X", "2026-10-20") is True, "closure_exempt: only the exempt pause's evidence -> not blocking")
    json.dump({"window_ended": True, "events": [{"t": 1, "invocation": INV, "why": "no telemetry for 301 s (>= 300 s)"}]},
              open(cp, "w"))
    ok(D.closure_exempt("X", "2026-10-20") is False, "closure_exempt: a no-telemetry event of that invocation keeps it closed")
    json.dump({"window_ended": True, "events": [{"t": 1, "invocation": INV, "why": EVID + " AND no telemetry for 301 s"}]},
              open(cp, "w"))
    ok(D.closure_exempt("X", "2026-10-20") is False, "closure_exempt: text appended to the evidence keeps it closed")
    json.dump({"window_ended": True, "events": [{"t": 1, "invocation": INV, "why": EVID}, {"t": 2, "invocation": "", "why": "pre"}]},
              open(cp, "w"))
    ok(D.closure_exempt("X", "2026-10-20") is False, "closure_exempt: an unbound event keeps the window closed")
    open(cp, "w").write("{bad")
    ok(D.closure_exempt("X", "2026-10-20") is False, "closure_exempt: unreadable closure keeps the window closed")
finally:
    D.scene_dir = orig
shutil.rmtree(root, ignore_errors=True)
shutil.rmtree(TMP_ASTRA, ignore_errors=True)
print(f"_k23_a28sm_selftest: {N[1]}/{N[0]} {'OK' if N[0] == N[1] else 'FAILED'}")
sys.exit(0 if N[0] == N[1] else 1)
