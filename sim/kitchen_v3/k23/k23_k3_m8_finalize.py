"""K3 M8: finalize the Kitchen 3 uses table + homeowner inventory after the K3 perception freeze, exactly like Kitchen 2's 58b6a37:
version k23-v1-draft -> k23-v1, the source / vocabulary / written texts name the frozen B5 list (docs/final-test/matrix/detector_types_FloorPlan17.json)
and the perception freeze record sha. Refuses unless: the freeze record + meta exist, the B5 list's weights sha equals the primary's sha in the
record, and both files list exactly the frozen types minus Knife (the D2 exclusion). Refuses if already final. Preserves line endings.
    python sim/kitchen_v3/k23/k23_k3_m8_finalize.py
"""
import hashlib, json, os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
P = lambda *a: os.path.join(ROOT, *a)
INV, USES = P("sim", "step5", "homeowner_inventory_FloorPlan17.json"), P("sim", "step5", "uses_table_FloorPlan17.json")
B5 = P("docs", "final-test", "matrix", "detector_types_FloorPlan17.json")
REC, META = P("sim", "kitchen_v3", "FloorPlan17", "frozen_weights.sha256"), P("sim", "kitchen_v3", "FloorPlan17", "frozen_weights.meta.json")


def load(p):
    raw = open(p, "rb").read()
    return json.loads(raw.decode("utf-8")), (b"\r\n" in raw)


def save(p, d, crlf):
    s = json.dumps(d, indent=1, ensure_ascii=False) + "\n"
    open(p, "wb").write((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))


def main():
    for f in (INV, USES, B5, REC, META):
        if not os.path.exists(f):
            sys.exit(f"REFUSED: missing {os.path.relpath(f, ROOT)}")
    inv, inv_crlf = load(INV); uses, uses_crlf = load(USES); b5 = json.load(open(B5, encoding="utf-8")); meta = json.load(open(META, encoding="utf-8"))
    if inv["version"] != "k23-v1-draft" or uses["version"] != "k23-v1-draft":
        sys.exit("REFUSED: already final (or unexpected version)")
    rec_sha = hashlib.sha256(open(REC, "rb").read()).hexdigest()
    if meta.get("record_sha256") and meta["record_sha256"] != rec_sha:
        sys.exit("REFUSED: the freeze meta does not match the record")
    rec = {l.split("  ", 1)[1].strip(): l.split("  ", 1)[0] for l in open(REC, encoding="utf-8") if "  " in l}
    if rec.get("sim/weights/v3/FloorPlan17.yolo26s.pt") != b5["weights_sha"]:
        sys.exit("REFUSED: the B5 list's weights sha is not the frozen primary's")
    want = set(b5["types"]) - {"Knife"}
    k = inv["kitchens"]; v = k.get("FloorPlan17", k) if isinstance(k, dict) else k
    inv_types = set(v) if not isinstance(v, dict) else set(next(v[x] for x in ("types", "inventory", "items") if x in v))
    uses_types = set(r if isinstance(r, str) else (r.get("type") or r.get("name")) for r in uses["rows"])
    if inv_types != want or uses_types != want:
        sys.exit(f"REFUSED: type lists differ from the frozen list minus Knife (inventory {len(inv_types)}, uses {len(uses_types)}, frozen {len(want)})")
    sha12, n = b5["weights_sha"][:12], b5["n"]
    stamp = time.strftime("%Y-%m-%d %H:%M")
    inv["version"] = "k23-v1"
    inv["source"] = (f"FloorPlan17: the frozen primary detector type list, docs/final-test/matrix/detector_types_FloorPlan17.json (frozen primary sha "
                     f"{sha12}..., {n} types) minus the D2 exclusion")
    uses["version"] = "k23-v1"
    uses["written"] = (f"2026-10-18, K23 agent-side, before any Kitchen 3 practice, rehearsal or test result; checked against the frozen Kitchen 3 "
                       f"primary {stamp} (perception freeze record {rec_sha[:8]}...)")
    uses["vocabulary"] = (f"FloorPlan17's frozen primary detector class list: docs/final-test/matrix/detector_types_FloorPlan17.json (frozen primary sha "
                          f"{sha12}..., {n} types) minus the D2 exclusion; type names only")
    save(INV, inv, inv_crlf); save(USES, uses, uses_crlf)
    print(f"finalized k23-v1: inventory + uses table, {len(want)} types, freeze record {rec_sha[:12]}, primary {sha12}")


if __name__ == "__main__":
    main()
