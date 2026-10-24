"""Offline gate for k23_coco_meta.py on a fixture root (never the real FloorPlan15 record)."""
import hashlib, json, os, shutil, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import k23_coco_meta as M
ok = fail = 0


def check(n, c):
    global ok, fail
    if c: ok += 1
    else: fail += 1; print("FAIL", n)


fx = tempfile.mkdtemp(prefix="k23coco_")
P = M.paths(fx)
os.makedirs(P["dir"]); os.makedirs(os.path.join(fx, "sim", "weights", "v3"))
coco = os.path.join(fx, *M.COCO_REL.split("/"))
open(coco, "wb").write(b"coco-weights")
pre = b"aa" * 32 + b"  sim/weights/v3/FloorPlan15.yolo26s.pt\n" + b"bb" * 32 + b"  sim/perception.py\n"
open(P["record"], "wb").write(pre)
meta0 = {"scene": "FloorPlan15", "n_files": 2, "record_sha256": hashlib.sha256(pre).hexdigest(), "coco_pending": True, "test_evaluation_blocked": True}
json.dump(meta0, open(P["meta"], "w"))
check("before --add-coco: invalid (coco pending)", M.inspect(fx)["state"] == "invalid")
# what k23_freeze.py --add-coco does: append the line, flip the flags, keep record_sha256 / n_files
open(P["record"], "ab").write(M.sha_file(coco).encode() + b"  " + M.COCO_REL.encode() + b"\n")
json.dump(dict(meta0, coco_pending=False, test_evaluation_blocked=False, coco_added="t"), open(P["meta"], "w"))
st = M.inspect(fx)
check("after --add-coco: the stale metadata needs a successor", st["state"] == "needs_successor" and st["n_lines"] == 3)
m = M.write(fx, now="20261020T2300")
check("successor: record sha / n_files describe the record now", m["record_sha256"] == hashlib.sha256(open(P["record"], "rb").read()).hexdigest()
      and m["n_files"] == 3)
check("successor keeps the previous values in coco_append", m["coco_append"]["previous_record_sha256"] == meta0["record_sha256"]
      and m["coco_append"]["previous_n_files"] == 2 and m["coco_append"]["coco_sha256"] == M.sha_file(coco))
check("the previous metadata is archived", os.path.exists(os.path.join(P["dir"], "supersession", "20261020T2300-coco", "frozen_weights.meta.json")))
check("then: the metadata is current", M.inspect(fx)["state"] == "current")
try:
    M.write(fx); refused = False
except RuntimeError:
    refused = True
check("a second write is refused", refused)
# tampering cases
json.dump(dict(meta0, coco_pending=False, test_evaluation_blocked=False), open(P["meta"], "w"))
open(P["record"], "wb").write(pre.replace(b"bb", b"cc") + M.sha_file(coco).encode() + b"  " + M.COCO_REL.encode() + b"\n")
check("a record changed beyond the COCO line is invalid", M.inspect(fx)["state"] == "invalid")
open(P["record"], "wb").write(pre + b"\n" + M.sha_file(coco).encode() + b"  " + M.COCO_REL.encode() + b"\n")
check("an extra blank line before the COCO line is invalid (byte-exact predecessor, Astra G3 4)", M.inspect(fx)["state"] == "invalid")
open(P["record"], "wb").write(pre + M.sha_file(coco).encode() + b"  " + M.COCO_REL.encode() + b"\n" + b"\n")
check("a trailing blank line after the COCO line is invalid", M.inspect(fx)["state"] == "invalid")
open(P["record"], "wb").write(pre + b"00" * 32 + b"  " + M.COCO_REL.encode() + b"\n")
check("a COCO line whose sha is not the file's is invalid", M.inspect(fx)["state"] == "invalid")
open(P["record"], "wb").write(pre + M.sha_file(coco).encode() + b"  sim/weights/v3/other.pt\n")
check("a last line that is not the COCO checkpoint is invalid", M.inspect(fx)["state"] == "invalid")
shutil.rmtree(fx, ignore_errors=True)
print(f"{ok}/{ok + fail}"); sys.exit(1 if fail else 0)
