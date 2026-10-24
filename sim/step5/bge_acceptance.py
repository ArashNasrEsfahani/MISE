"""G-dense preliminary acceptance measurement for bge-m3 (docs/step5-plan.md 8.5). Run ONLY with the isolated venv:

    sim/step5/.venv_bge/Scripts/python.exe sim/step5/bge_acceptance.py

CPU only (CUDA hidden before torch is imported), threads capped, max_seq_length capped, offline from the local Hugging Face cache, pinned
package versions and model revision. Measures model load time, single-query embedding latency (p50/p95), batch embedding of ~100 synthetic
bilingual records, and peak working-set RAM of this process. Also reports an INFORMATIONAL cross-lingual top-1 check; it is not the G-dense
recall gate (that needs the real record set, workflow B). Writes sim/step5/bge_acceptance.json. Never touches the WSL environment.
"""
import ctypes
import json
import os
import sys
import time
from ctypes import wintypes
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""                 # CPU only, set BEFORE torch is imported
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
THREADS, MAX_LEN = 4, 128
REV = "5617a9f61b028005a4858fdac845db406aefb181"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import ltm_concepts as lc  # noqa: E402


def peak_ram_mb():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    k32, ps = ctypes.WinDLL("kernel32", use_last_error=True), ctypes.WinDLL("psapi", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    ps.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return round(c.PeakWorkingSetSize / 2 ** 20, 1), round(c.WorkingSetSize / 2 ** 20, 1)


def main():
    import numpy as np
    import sentence_transformers
    import torch
    import transformers
    torch.set_num_threads(THREADS)
    lex = lc.default_lexicon()
    objs = ["Apple", "Bread", "Egg", "Mug", "Cup", "Plate", "Bowl", "Tomato", "Potato", "Lettuce"]
    dests = ["Fridge", "Sink", "Microwave", "Cabinet", "Drawer", "Toaster", "GarbageCan", "Pot", "Pan", "CounterTop"]
    verbs = ["put", "pick"]
    recs = []
    for o in objs:
        for d in dests:
            recs.append([{"verb": "put", "req": "-", "obj": o, "ostate": "-", "src": "-", "dest": d, "dstate": "-", "neg": 0}])
    texts = [lc.key_text(r, lex) for r in recs]
    assert len(texts) == 100
    t0 = time.perf_counter()
    model_dir = Path.home() / ".cache/huggingface/hub/models--BAAI--bge-m3/snapshots" / REV
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(model_dir), device="cpu")
    model.max_seq_length = MAX_LEN
    load_s = time.perf_counter() - t0
    peak_load, _ = peak_ram_mb()
    # warm-up, then batch
    model.encode(["warm up"], normalize_embeddings=True)
    t0 = time.perf_counter()
    doc = model.encode(texts, batch_size=16, normalize_embeddings=True)
    batch_s = time.perf_counter() - t0
    # single-query latency over 60 queries (EN and FA instructions)
    qs = []
    for i, r in enumerate(recs[:30]):
        o, d = r[0]["obj"], r[0]["dest"]
        qs.append((f"put the {lex.type_name(o, 'en')} in the {lex.type_name(d, 'en')}", i))
        qs.append((f"{lex.type_name(o, 'fa')} را در {lex.type_name(d, 'fa')} بگذار", i))
    lat = []
    hit_en = hit_fa = 0
    for j, (q, gold) in enumerate(qs):
        t0 = time.perf_counter()
        v = model.encode([q], normalize_embeddings=True)[0]
        lat.append((time.perf_counter() - t0) * 1000.0)
        top = int(np.argmax(doc @ v))
        if j % 2 == 0:
            hit_en += top == gold
        else:
            hit_fa += top == gold
    lat_sorted = sorted(lat)
    peak, cur = peak_ram_mb()
    out = {
        "date": time.strftime("%Y-%m-%d"), "model": "BAAI/bge-m3", "model_revision": REV, "device": "cpu", "threads": THREADS, "max_seq_length": MAX_LEN,
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "sentence_transformers": sentence_transformers.__version__,
                     "transformers": transformers.__version__, "numpy": np.__version__}, "dim": int(doc.shape[1]),
        "n_records": len(texts), "n_queries": len(qs), "load_seconds": round(load_s, 2), "batch_embed_100_records_seconds": round(batch_s, 2),
        "query_latency_ms": {"p50": round(lat_sorted[len(lat_sorted) // 2], 1), "p95": round(lat_sorted[int(len(lat_sorted) * 0.95) - 1], 1),
                             "max": round(lat_sorted[-1], 1), "mean": round(sum(lat) / len(lat), 1)},
        "peak_ram_mb_after_load": peak_load, "peak_ram_mb": peak, "ram_mb_at_end": cur,
        "informational_top1_cross_lingual": {"en_instruction_vs_bilingual_key": round(hit_en / (len(qs) // 2), 3),
                                              "fa_instruction_vs_bilingual_key": round(hit_fa / (len(qs) // 2), 3),
                                              "note": "NOT the G-dense recall gate; synthetic 100 put-records, raw instruction vs key text"},
        "key_text_sample": texts[0],
    }
    (HERE / "bge_acceptance.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
