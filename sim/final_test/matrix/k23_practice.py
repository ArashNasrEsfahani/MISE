"""K23 practice entry point: Gemma practice for Kitchen 2 / 3 (namespaces practice-k2-v1 = FloorPlan15, practice-k3-v1 = FloorPlan17)
through the UNCHANGED practice_k.py engine (A29 item 1: K = 10, practice order 0, the kitchen's own package; Astra K23-NAMESPACES-r1).

practice_k.py is not edited (Kitchen 1 bytes and gates stay identical). Like fg_runner.py for capture, this process only:
  (1) admits THIS K23 practice namespace to practice_k.PRACTICE_NAMESPACES (in this process, for this namespace only);
  (2) replaces build_live's hook-B hint function, which practice_k reads from the FloorPlan1 briefing (treatment.briefing_hint_table's
      default scene), by the hint function of THE NAMESPACE'S kitchen briefing (k23_entry.briefing_hints) -- the executor, packager,
      contract, snapshot root and identity are already scene-aware in practice_k / treatment;
  (3) refuses a LIVE run unless: both kitchens' A29 sample manifests are frozen (A29 item 2: "before practice in either kitchen"), the
      kitchen's package inputs are final (uses table + inventory 'k23-v1', approved briefing), its perception freeze verifies, the
      treatment pins hold, the orders are [0] and K includes 10;
  (4) --status verifies the journal and the snapshots in the kitchen's OWN ft-v2 tree (practice_k.status defaults to the legacy
      Kitchen 1 tree).

    python sim/final_test/matrix/k23_practice.py --namespace practice-k2-v1 --dry-run        (any python: orders + guard + readiness)
    python sim/final_test/matrix/k23_practice.py --namespace practice-k2-v1 --status         (any python: journal + snapshots)
    python sim/final_test/matrix/k23_practice.py --namespace practice-k2-v1 --ollama http://<win host>:11434   (WSL, under the sim lock:
           use sim/final_test/k23_run/practice/launch.py, which adds the lock, the hang guard and the free count_tokens key)

Defaults: --orders 0 --k 10 (A29: the K = 10 package; Kitchen 1's K = 1 / 3 / 5 curve snapshots are Kitchen-1-only). Exit codes: practice_k's,
2 = refused here (nothing run). Python 3.10 syntax; the simulator / Ollama are imported only by the live path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                       # noqa: E402
import k23_entry as E                                    # noqa: E402
import k23_ns as K                                       # noqa: E402

REFUSED = 2
#: Astra K23-TOOLING-G1-r2 1: practice_k's argparse accepts ABBREVIATED options (--gemma -> --gemma-only-counter, --allow -> --allow-test,
#: ...), so a blacklist is bypassable. Only these options (with their arity) are forwarded to practice_k; everything else is refused.
PASSTHROUGH = {"--ollama": 1, "--limit": 1, "--steal-lock": 0}


def passthrough(rest: list) -> list:
    out, i = [], 0
    while i < len(rest):
        tok = rest[i]
        name = tok.split("=", 1)[0]
        if name not in PASSTHROUGH:
            raise ValueError(f"option {tok!r} is not forwarded to practice_k (allowed: {', '.join(sorted(PASSTHROUGH))}; abbreviations refused)")
        n = PASSTHROUGH[name] if "=" not in tok else 0
        if n and i + n >= len(rest):
            raise ValueError(f"option {tok} needs a value")
        out += rest[i:i + 1 + n]
        i += 1 + n
    return out
#: prereg A39.2 (Arash 2026-10-20 ~16:15; Astra K23-TOOLING-T4-r1): Kitchen 2 practice starts only after the FloorPlan15 COCO run,
#: --add-coco and k23_coco_meta.py --write; readiness() refuses before that (k23_entry.order_problems) and a live run records the order
#: evidence the agent freeze checks.
A38_NOTE = ("prereg A39.2: Kitchen 2 practice and rehearsal start only after the FloorPlan15 COCO run, --add-coco and "
            "k23_coco_meta.py --write (enforced by readiness; the order evidence is checked at the agent freeze).")
DEFAULT_ORDERS = "0"
DEFAULT_KS = "10"


def admit(ns: str):
    """practice_k with THIS K23 practice namespace admitted and the scene-aware hint function installed (in this process only)."""
    if E.kind(ns) != "practice":
        raise ValueError(f"{ns!r} is not a K23 practice namespace ({', '.join(E.k23_namespaces('practice'))})")
    import practice_k as PK
    if ns not in PK.PRACTICE_NAMESPACES:
        PK.PRACTICE_NAMESPACES = tuple(PK.PRACTICE_NAMESPACES) + (ns,)
    if not getattr(PK.build_live, "k23_scene_hints", False):
        orig = PK.build_live

        def build_live(args):
            ex, hook, counter, params, hints = orig(args)
            if args.namespace in E.k23_namespaces("practice"):          # resolved per call (Astra K23-TOOLING-G1 1: any admission order)
                hints = E.briefing_hints(args.namespace)                # the kitchen's own briefing table, not FloorPlan1's
            return ex, hook, counter, params, hints
        build_live.k23_scene_hints = True
        build_live.k23_orig = orig
        PK.build_live = build_live
    return PK


def readiness(ns: str, orders, ks, root: str | None = None, *, perception: bool = True) -> list:
    """Every reason a LIVE practice run of `ns` is refused ([] = ready). perception=False only for the Windows-side launcher preflight:
    the perception freeze record lists ~/thor-data files by their WSL paths, so it is verified inside WSL (the live run always does)."""
    import treatment as T
    scene = E.scene_of(ns)
    out = []
    if list(orders) != [0]:
        out.append(f"orders {list(orders)}: K23 practice is order 0 only (A29 item 1)")
    if 10 not in set(ks):
        out.append(f"K {list(ks)}: the registered K23 package is K = 10 (A29 item 1)")
    out += E.samples_problems(root)
    out += E.order_problems(ns, root)                          # prereg A39.2: Kitchen 2 practice only after COCO + --add-coco + meta
    out += E.package_problems(scene, root)
    if perception:
        out += E.perception_problems(scene, root)                  # the WSL-share mapping on Windows, the plain check in WSL
    out += [f"treatment pin: {p}" for p in T.pin_problems()]
    return out


def status(ns: str, orders, ks, root: str | None = None) -> dict:
    import practice_k as PK
    import treatment as T
    st = PK.status(ns, orders, ks, root=root, snap_root=T.snapshot_root(root, scene=E.scene_of(ns)))
    st["snapshot_root"] = os.path.relpath(T.snapshot_root(root, scene=E.scene_of(ns)), C.paths(root)["root"]).replace(os.sep, "/")
    rows = [r for r in C.Journal(C.journal_path(ns, "practice", root)).rows() if r.get("order") in orders]
    ids = sorted({r.get("treatment_id") for r in rows if r.get("treatment_id")})
    st["treatment_ids"] = ids
    if rows:
        try:
            cur = T.identity(ns, root)["sha"]
            st["identity_current"] = cur
            bad = T.journal_problems(rows, ns, cur)
            st["identity_problems"] = bad[:5]
            st["ok"] = bool(st["ok"]) and not bad
        except Exception as e:                            # noqa: BLE001  (pins moved: the identity fails closed)
            st["identity_problems"] = [f"{type(e).__name__}: {str(e)[:200]}"]
            st["ok"] = False
    return st


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description="K23 practice entry point (practice_k.py engine, unchanged)", add_help=True, allow_abbrev=False)
    ap.add_argument("--namespace", required=True, choices=E.k23_namespaces("practice"))
    ap.add_argument("--orders", default=DEFAULT_ORDERS)
    ap.add_argument("--k", default=DEFAULT_KS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--channels", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--allow-test", action="store_true", help=argparse.SUPPRESS)
    a, rest = ap.parse_known_args(argv)
    if a.channels is not None:
        print("REFUSED: --channels is an ft-v1 option; K23 practice renders the package", file=sys.stderr)
        return REFUSED
    if a.allow_test:
        print("REFUSED: practice never touches test items (--allow-test is not passed through)", file=sys.stderr)
        return REFUSED
    if "--kitchen" in rest:
        print("REFUSED: the kitchen is the namespace's scene (no --kitchen)", file=sys.stderr)
        return REFUSED
    if "--gemma-only-counter" in rest:                              # Astra K23-TOOLING-G1 3: the registered two-tokenizer counter only
        print("REFUSED: --gemma-only-counter would change the registered token counting (Gemma + Anthropic count_tokens)", file=sys.stderr)
        return REFUSED
    orders = [int(x) for x in a.orders.split(",") if x.strip()]
    ks = [int(x) for x in a.k.split(",") if x.strip()]
    if a.status:
        st = status(a.namespace, orders, ks)
        print(json.dumps(st, indent=1))
        return 0 if st["ok"] else 1
    PK = admit(a.namespace)
    try:
        rest = passthrough(rest)
    except ValueError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return REFUSED
    pk_argv = ["--namespace", a.namespace, "--orders", a.orders, "--k", a.k] + rest
    if a.dry_run:
        code = PK.main(pk_argv + ["--dry-run"])
        why = readiness(a.namespace, orders, ks)
        print(json.dumps({"k23_practice": a.namespace, "scene": E.scene_of(a.namespace), "live_ready": not why, "live_blockers": why,
                          "perception_checked": True, "a38_note": A38_NOTE if E.scene_of(a.namespace) == E.COCO_SCENE else None,
                          "hint_table_sha": __import__("treatment").hint_table_sha(E.briefing_table(a.namespace))}))
        return code
    why = readiness(a.namespace, orders, ks)
    if why:
        print("REFUSED (nothing run): " + "; ".join(why[:8]), file=sys.stderr)
        return REFUSED
    E.record_order_ok(a.namespace, "practice")                     # A39.2 evidence (FloorPlan15 only), before the first attempt
    import k23_leakage as KL
    if KL.install():                                                # practice-snapshot own-fragment rule (Arash's decision K23-PRACTICE-LEAK)
        print("k23_leakage: own-fragment rule active (decision K23-PRACTICE-LEAK)", file=sys.stderr)
    import k23_fa_examples as FAX
    if FAX.install():                                               # A24.2: Persian-practised examples in English (Arash's decision)
        FAX.record_event(a.namespace, "practice")                  # the activation boundary, checked at the agent freeze
        print("k23_fa_examples: active (decision K23-FA-EXAMPLES)", file=sys.stderr)
    return PK.main(pk_argv)


if __name__ == "__main__":
    sys.exit(main())
