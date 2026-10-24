"""AI2-THOR smoke test — confirms the simulator boots, renders a frame, and reports throughput.

Run inside WSL (Ubuntu) with the WSLg X display, forcing the NVIDIA GPU via Mesa's D3D12 layer:

    cd /mnt/d/Personal/Master*/sim
    MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 ~/thor-env/bin/python thor_smoke.py

Success = a (H, W, 3) frame prints, first_frame.png is written, and FPS is reported.
"""
import time

from ai2thor.controller import Controller

print("Booting AI2-THOR (first run downloads the Unity build ~hundreds of MB, please wait)...", flush=True)
controller = Controller(scene="FloorPlan1", width=600, height=600)
print("Controller started.", flush=True)

event = controller.step(action="Pass")
print(f"Initial frame: {event.frame.shape} | lastActionSuccess: {event.metadata['lastActionSuccess']}", flush=True)
print(f"Objects in scene: {len(event.metadata['objects'])}", flush=True)

# Save the first rendered frame so it can be eyeballed from Windows.
try:
    from PIL import Image

    Image.fromarray(event.frame).save("first_frame.png")
    print("Saved first_frame.png", flush=True)
except Exception as exc:  # pragma: no cover - diagnostic only
    print(f"Could not save frame image: {exc}", flush=True)

# Measure rendering throughput over a short action loop.
actions = ["MoveAhead", "RotateRight", "MoveAhead", "RotateLeft"]
steps = 60
start = time.time()
for i in range(steps):
    controller.step(action=actions[i % len(actions)])
elapsed = time.time() - start
print(f"Ran {steps} steps in {elapsed:.2f}s -> {steps / elapsed:.1f} FPS", flush=True)

controller.stop()
print("DONE - ai2thor render OK", flush=True)
