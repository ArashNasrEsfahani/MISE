# Hard-coded paths: what to set on a new machine

The code was developed on one laptop (Windows 11 + WSL2 Ubuntu-22.04, user `arash`) and many scripts carry that
machine's paths literally. **The Python files are not edited in this release on purpose**: the preregistration freeze
(`docs/final-test/freeze.json`, `docs/final-test/k23/freeze_<scene>.json`) pins the exact bytes of the evaluation code,
so changing a path inside a pinned file makes the freeze check refuse to run. Instead, recreate the same layout (fastest)
or edit the few launcher constants below in your own fork (and accept that the freeze check will then report the change).

## 1. What differs between machines

| What | Value on the original machine | Where it is assumed |
|---|---|---|
| WSL distro | `Ubuntu-22.04` | every `wsl -d ...` call (launchers, `START-TESTING.bat`) |
| WSL Python venv | `/home/arash/thor-env/bin/python` (`~/thor-env`) | launchers, `.sh` runners, `START-TESTING.bat` |
| Data root (captures, YOLO datasets) | `~/thor-data/` = `/home/arash/thor-data/` | training / dataset scripts, perception freeze records |
| Repository location | `D:\Personal\Master's Project` = `/mnt/d/Personal/Master's Project` | `.sh` runners (`REPO=$(ls -d /mnt/d/Personal/Master*/ ...)`), some launchers |
| Ollama host | `http://127.0.0.1:11434` from Windows; the Windows host IP (e.g. `172.20.144.1`, changes on reboot) from WSL | `sim/gui_config.json`, `--ollama` flags |

The least-effort setup is to reproduce this layout: create the WSL user's venv at `~/thor-env`
(`environment.yml` / `sim/requirements.txt`), keep data under `~/thor-data`, and clone the repository so that WSL sees
it at a path matching `/mnt/d/Personal/Master*/`. If your WSL user is not `arash`, a symlink
`sudo ln -s /home/<you> /home/arash` covers every `/home/arash/...` reference at once.

## 2. Main offenders (edit these if you do not mirror the layout)

| File | Constant / line | Set it to |
|---|---|---|
| `sim/final_test/matrix/matrix_run.py` | `WSL_DISTRO = "Ubuntu-22.04"`, `WSL_PYTHON = "/home/arash/thor-env/bin/python"` | your distro and venv python (used by `--wsl` to run capture / execute in WSL) |
| `sim/final_test/matrix/k23_entry.py` | `WSL_SHARE = "//wsl.localhost/Ubuntu-22.04"` | your distro's share; the Kitchen 2/3 perception freeze records list `~/thor-data` files by their WSL paths and Windows reads them through this share |
| `sim/planners/keys.py` | `DEFAULT_OPENROUTER_ENV_FILE` (a path on the author's machine) | do not edit: set the environment variable `OPENROUTER_ENV_FILE` to your own `.env` file holding `OPENROUTER_API_KEY=...`. Anthropic keys come from `THESIS_ANTHROPIC_API_KEY` or `ANTHROPIC_API_KEY` (process environment, or the Windows per-user environment). Keys are never written to disk by the code |
| `START-TESTING.bat` | `WSLDISTRO`, `WSLSIM` (`/mnt/d/Personal/Master's Project/sim`), `VENVPY` (`/home/arash/thor-env/bin/python`), `PORT` | your distro, the WSL path of `sim/` in your clone, your venv python |
| `sim/gui_config.json` | `"endpoint": "http://172.20.144.1:11434"` | the Ollama URL as seen from WSL (`ip route show default` gives the Windows host IP). `START-TESTING.bat` re-derives and repairs it on every start (backing the file up first) |
| YOLO `data.yaml` files (written by `sim/build_kitchen_data.py`, not shipped) | first line `path: /home/arash/thor-data/kitchen_v3/<scene>/yolo_primary` | the absolute dataset directory on your machine. Regenerating the dataset writes the right path; a dataset copied from elsewhere needs this line edited |
| `sim/build_kitchen_data.py`, `sim/train_kitchen.py`, `sim/train_amb.py`, `sim/train_state.py`, `sim/state_data.py` | `os.path.expanduser("~")/thor-data/...` | nothing to edit: they follow `~` (`build_kitchen_data.py` / `train_kitchen.py` also take `--root`, `state_data.py` takes `--store` / `--out`; `train_amb.py` and `train_state.py` use `~/thor-data` only) |
| `sim/final_test/k23_run/run_k23_analyze.py` | `M = r"D:\Personal\Master's Project\sim\final_test\matrix"` | your clone; or call `sim/final_test/matrix/k23_analyze.py` directly (see `scripts/reproduce_analysis.py`) |
| `sim/final_test/test_run/launch_test.py`, `sim/final_test/k23_run/launch_k23*.py` | `wsl -d Ubuntu-22.04 --cd ...` | distro / clone path (the K23 launchers already derive the WSL path from their own checkout) |
| `sim/final_test/matrix/k23_fp1_equivalence.py` | `/mnt/d/Personal/MP-base`, `/mnt/d/Personal/MP-agent-side` | one-off evidence trees of the Kitchen-1 equivalence check; only needed to repeat that check |

About 57 `.sh` / `.ps1` runners under `sim/kitchen_v3/`, `sim/kitchen_v3/k23/` and `sim/final_test/k23_run/` also
hard-code `/home/arash/thor-env/bin/python`, `/home/arash/thor-data` and `/mnt/d/Personal/Master*`; they are the
orchestration actually used for the evaluation runs and are kept verbatim as a record. Grep for them before reusing one:

```bash
grep -rlE "/home/arash|/mnt/d/Personal|D:\\\\Personal" sim scripts START-TESTING.bat
```

## 3. Other machine assumptions

- AI2-THOR runs only inside WSL with `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0` (WSLg + Mesa D3D12 on an
  NVIDIA GPU); see `docs/ai2thor-wsl-setup.md`. Only one AI2-THOR controller at a time on an 8 GB GPU.
- Run Python from the repository root, never from `sim/` (`sim/trace.py` shadows the standard-library `trace`).
- Long runs take a global lock at `/home/arash/sim.lock` (`flock`) in WSL.
- The torch build must match the WSL CUDA driver (the original setup used CUDA 12.7 with the `cu124` wheels).
- `sim/trace.db` / `sim/memory.db` are created at runtime (git-ignored). On `/mnt/...` (DrvFs) the trace store must not
  use SQLite WAL mode; `sim/trace.py` picks the journal mode from the filesystem.
