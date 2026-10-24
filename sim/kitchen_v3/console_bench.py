"""Measure what the USER sees from the web console (run inside WSL against a running webapp on :8000).

1. delivered fps: frames received on /stream over 8 s while idle;
2. click-to-picture latency: send /action (rotate_right / rotate_left alternately) and time until a streamed frame
   differs clearly from the frame before the click (mean abs pixel difference > 4), 6 trials;
3. frame size.
Same script for the old (git HEAD, July) and the current console, so the comparison is like for like.

    ~/thor-env/bin/python sim/kitchen_v3/console_bench.py <label>
"""
import io
import json
import sys
import threading
import time
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

B = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"
LABEL = sys.argv[1] if len(sys.argv) > 1 else "console"
frames = []                                     # (t, jpeg bytes)
stop = False


def reader():
    r = urllib.request.urlopen(B + "/stream", timeout=30)
    buf = b""
    while not stop:
        chunk = r.read1(65536) if hasattr(r, "read1") else r.read(16384)
        if not chunk:
            break
        buf += chunk
        while True:
            a = buf.find(b"\xff\xd8")
            b = buf.find(b"\xff\xd9", a + 2) if a >= 0 else -1
            if a < 0 or b < 0:
                break
            frames.append((time.time(), buf[a:b + 2]))
            buf = buf[b + 2:]


def img(j):
    return np.asarray(Image.open(io.BytesIO(j)).convert("L").resize((160, 160)), dtype=np.float32)


for _ in range(60):
    try:
        st = json.loads(urllib.request.urlopen(B + "/status", timeout=5).read())
        if st.get("scene"):
            break
    except Exception:
        pass
    time.sleep(2)
th = threading.Thread(target=reader, daemon=True)
th.start()
time.sleep(3)
n0 = len(frames)
time.sleep(8)
idle_fps = (len(frames) - n0) / 8.0
sizes = [len(j) for _t, j in frames[-20:]]
lat, comp = [], []
for k in range(6):
    time.sleep(1.5)
    base = img(frames[-1][1])
    t_click = time.time()
    urllib.request.urlopen(B + "/action?name=" + ("rotate_right" if k % 2 == 0 else "rotate_left"), timeout=10).read()
    got = None
    while time.time() - t_click < 15:
        for t, j in list(frames):
            if t > t_click and float(np.abs(img(j) - base).mean()) > 4:
                got = t - t_click
                break
        if got is not None:
            break
        time.sleep(0.02)
    # completion: time until the picture has been stable for 0.4 s
    done = None
    while time.time() - t_click < 15:
        fr = [(t, j) for t, j in list(frames) if t > t_click]
        if len(fr) >= 2:
            last_change = t_click
            for (t1, j1), (t2, j2) in zip(fr, fr[1:]):
                if float(np.abs(img(j2) - img(j1)).mean()) > 1.0:
                    last_change = t2
            if time.time() - last_change > 0.4 and last_change > t_click:
                done = last_change - t_click
                break
        time.sleep(0.1)
    lat.append(round(got, 3) if got is not None else None)
    comp.append(round(done, 3) if done is not None else None)
# composite actions: walk to an object (the "Go To" button) and open it, as the console buttons do
def wait_stable(t0, limit=40):
    while time.time() - t0 < limit:
        fr = [(t, j) for t, j in list(frames) if t > t0]
        if len(fr) >= 2:
            last = t0
            for (t1, j1), (t2, j2) in zip(fr, fr[1:]):
                if float(np.abs(img(j2) - img(j1)).mean()) > 1.0:
                    last = t2
            if time.time() - last > 0.8 and last > t0:
                return round(last - t0, 2)
        time.sleep(0.1)
    return None
objs = json.loads(urllib.request.urlopen(B + "/objects", timeout=10).read())
objs = objs.get("objects", objs) if isinstance(objs, dict) else objs
comp_actions = {}
for typ, act in (("Toaster", "go to"), ("Fridge", "go to"), ("Fridge", "open"), ("Fridge", "close")):
    o = next((x for x in objs if x.get("type") == typ), None)
    if not o:
        continue
    time.sleep(1.0)
    t0 = time.time()
    urllib.request.urlopen(B + "/act_obj?id=" + urllib.parse.quote(o["id"]) + "&action=" + urllib.parse.quote(act),
                           timeout=10).read()
    comp_actions[f"{act} {typ}"] = wait_stable(t0)
stop = True
st = json.loads(urllib.request.urlopen(B + "/status", timeout=5).read())
print(json.dumps({"label": LABEL, "server_fps": st.get("fps"), "delivered_idle_fps": round(idle_fps, 1),
                  "click_to_picture_s": lat, "median_latency_s": sorted(x for x in lat if x is not None)[len(lat) // 2]
                  if any(x is not None for x in lat) else None, "move_complete_s": comp, "composite_s": comp_actions, "jpeg_kb": round(sum(sizes) / max(1, len(sizes)) / 1024)}))
