"""Log console fps over time (run inside WSL next to a running webapp): one line every 10 s to ~/fps_watch.log.

    ~/thor-env/bin/python sim/kitchen_v3/fps_watch.py
Columns: time, server fps, frame age, honest on/phase, view, thor-Linux64 %CPU (from ps).
"""
import json
import subprocess
import time
import urllib.request

LOG = "/home/arash/fps_watch.log"
while True:
    try:
        st = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/status", timeout=5).read())
        h = st.get("honest") or {}
        cpu = subprocess.run("ps -C thor-Linux64 -o %cpu= | head -1", shell=True, capture_output=True,
                             text=True).stdout.strip()
        line = (f"{time.strftime('%H:%M:%S')} fps={st.get('fps')} age={(st.get('sim') or {}).get('frame_age_s')} "
                f"honest={h.get('on')}/{h.get('phase')} view={st.get('view')} busy={st.get('busy')} thorcpu={cpu}")
    except Exception as exc:
        line = f"{time.strftime('%H:%M:%S')} down {type(exc).__name__}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    time.sleep(10)
