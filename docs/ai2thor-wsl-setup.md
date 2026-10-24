# AI2-THOR on WSL2 — Setup Guide & Progress Log

Goal: get the AI2-THOR simulator rendering frames inside WSL2 on the **NVIDIA RTX 4060**. This is step 0 of the
thesis implementation.

> **Status: ✅ working (2026-06-23).** `ai2thor 5.0.0` renders `FloorPlan1` at **~40 FPS @ 600×600** via the WSLg X
> display + Mesa's D3D12 driver pointed at the NVIDIA GPU. See [Rendering approach](#rendering-approach-what-works-here).

## This machine (verified 2026-06-23)

| Component | Value | Implication |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop (4 GB) + Intel UHD | Render via Mesa-D3D12→NVIDIA (the NVIDIA **Vulkan ICD is absent** in WSL, so CloudRendering would be CPU-only) |
| RAM / CPU | 24 GB / 20 logical cores | Plenty for AI2-THOR |
| OS | Windows 11 (build 22631) | — |
| WSL | 2.6.1.0, kernel 6.6.87.2-1, WSLg 1.0.66, D3D12 1.611 | Modern; GPU + GUI passthrough available |
| Distros installed | only `docker-desktop` | **Need to install a real Linux distro (Ubuntu)** |

## Rendering approach (what works here)

AI2-THOR is a Unity build that needs a GPU. Tested on this machine:
- **CloudRendering** (Vulkan/EGL, headless) — ❌ **not usable here.** `vulkaninfo` only finds Mesa `llvmpipe` (CPU);
  the Windows NVIDIA driver exposes CUDA/encode libs to WSL but **no GL/Vulkan ICD**, so CloudRendering renders on CPU.
- **WSLg X display + Mesa-D3D12** — ✅ **chosen.** WSLg provides an X server (`DISPLAY=:0`) and Mesa's D3D12 driver
  gives GPU-accelerated OpenGL. It defaults to the Intel iGPU; we force the RTX 4060 with
  `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`. Confirmed ~40 FPS.
- `Xvfb` — not needed (WSLg already supplies a display).

## Why Ubuntu 22.04 (not 24.04)

Ubuntu 22.04 ships **Python 3.10**, the most battle-tested version for the `ai2thor` pip package. 24.04 ships Python
3.12 which can hit packaging edge cases. If you prefer 24.04, we'll pin Python via a venv/conda — but 22.04 is the
low-friction default.

---

## Step-by-step

### Step 1 — Install Ubuntu 22.04 (interactive — run yourself)

This first launch prompts for a new UNIX username + password and shows a live download bar — both need a **real
terminal** (Windows Terminal / PowerShell), not a detached/non-interactive shell (which hangs silently):

```
wsl --install -d Ubuntu-22.04
```

When it opens the Ubuntu shell, create a username (e.g. `arash`) and password. Then close and reopen if needed.
Verify from PowerShell:

```
wsl -l -v          # Ubuntu-22.04 should appear, VERSION 2
```

### Step 2 — Base packages + GPU/Vulkan stack (inside Ubuntu)

```bash
sudo apt update && sudo apt -y upgrade
sudo apt install -y python3 python3-venv python3-pip python3-dev build-essential git pciutils \
                    libvulkan1 vulkan-tools mesa-vulkan-drivers libgl1 libglib2.0-0
```

### Step 3 — Confirm the GPU is visible in WSL

```bash
nvidia-smi             # ✅ should list the RTX 4060 (driver comes from the Windows host — do NOT apt-install it)
# OpenGL is the path that matters here. Default picks Intel; force NVIDIA via the Mesa env var:
MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 glxinfo -B | grep -iE 'Device:|renderer string'
# -> Device: D3D12 (NVIDIA GeForce RTX 4060 Laptop GPU)
# Note: `vulkaninfo --summary` only shows Mesa llvmpipe (CPU) — NVIDIA has no Vulkan ICD in WSL. That's expected here.
```
If `nvidia-smi` is missing: update the NVIDIA driver on **Windows**; never install Linux NVIDIA drivers inside WSL.

### Step 4 — Python env + ai2thor

```bash
python3 -m venv ~/thor-env
source ~/thor-env/bin/activate
pip install --upgrade pip wheel
pip install ai2thor          # pin later once a known-good version is confirmed
```

### Step 5 — Smoke test (downloads the ~769 MB Unity build on first run)

The test script lives in the repo at `sim/thor_smoke.py` (default build, no `platform=` — it uses the WSLg X display).
Run it with the two env vars that select the GPU:

```bash
cd /mnt/d/Personal/Master*/sim          # the Master* glob avoids quoting the apostrophe + space in the path
MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 ~/thor-env/bin/python thor_smoke.py
```

✅ **Confirmed 2026-06-23:** `Initial frame: (600, 600, 3) | lastActionSuccess: True`, 77 objects, **40.5 FPS**,
`first_frame.png` written. That completes step 0.

> **These two env vars are required for every GPU run:** `DISPLAY=:0` (WSLg X server) and
> `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` (else it falls back to the Intel iGPU). Without `DISPLAY` the Unity build
> can't open a rendering context.

### Troubleshooting

- Frame is black / ~1–2 FPS → the NVIDIA hint didn't take; re-check with the `glxinfo` line in Step 3.
- `unable to open display` → `DISPLAY` not set; `export DISPLAY=:0`.
- Last resort headless: `sudo apt install -y xvfb` then `xvfb-run -a -s "-screen 0 1280x1024x24" python thor_smoke.py`.

---

## Where code vs. cache lives

- **Project code** stays on Windows at `D:\Personal\Master's Project`, reachable from WSL at
  `/mnt/d/Personal/Master's\ Project`. Claude Code runs on Windows and edits there.
- **Heavy IO** (the venv and the `~/.ai2thor` build cache) lives in the **Linux home** (`~`), *not* under `/mnt/d` —
  cross-filesystem IO is slow.

## Progress log

| Date | Status |
|---|---|
| 2026-06-23 | System surveyed (RTX 4060, WSL 2.6.1, Ubuntu 22.04 chosen). Only `docker-desktop` distro present. |
| 2026-06-23 | Attempted `wsl --install -d Ubuntu-22.04 --no-launch` detached/in-background → **stalled** (processes idle at ~0 CPU, no download). Lesson: `wsl --install` hangs when run non-interactively; **run it in a real terminal**. Killed it; nothing partially registered. |
| 2026-06-23 | User installed Ubuntu 22.04 (user `arash`) in a real terminal. Steps 2–4 done as root/arash: Python 3.10.12, build tools, `ai2thor 5.0.0`. `nvidia-smi` sees the RTX 4060. **Vulkan only finds llvmpipe (CPU)** → pivoted from CloudRendering to WSLg-X + Mesa-D3D12-NVIDIA. |
| 2026-06-23 | ✅ **Step 0 complete.** `sim/thor_smoke.py` renders `FloorPlan1` at **40.5 FPS @ 600×600** on the RTX 4060; `first_frame.png` saved. Next: design the Function Library that wraps AI2-THOR actions as atomic skills. |
