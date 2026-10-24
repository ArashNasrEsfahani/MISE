# K3 practice resume (attempt 9 onwards, K23-FA-EXAMPLES active) + K3 rehearsal NOW, in a COCO pause (Arash 2026-10-20: K3 first; the first
# live run of the K23 rehearsal tooling happens while attended). Pause COCO at an epoch boundary -> GPU idle check -> Ollama up -> practice-k3-v1
# -> status -> rehearsal-k3-v1 -> Ollama down -> COCO resumes (verified on the GPU). Failures are logged; COCO always resumes.
#   powershell -ExecutionPolicy Bypass -File sim\kitchen_v3\k23\k23_k3_practice_now.ps1   (launch hidden / detached)
$ErrorActionPreference = "Continue"
$R = "D:\Personal\Master's Project"
Set-Location $R
$LOG = "$R\sim\kitchen_v3\k23\FloorPlan17\practice_now.log"
$S = "/mnt/c/Users/arash/AppData/Local/Temp/claude/D--Personal-Master-s-Project/d6bc583a-1876-43c2-9178-94630f4d195f/scratchpad"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONDONTWRITEBYTECODE = "1"
function Note($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File -FilePath $LOG -Append -Encoding utf8 }
function GpuMiB { try { [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits).Trim()) } catch { 0 } }
function OllamaDown { Get-Process "ollama app", "ollama", "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep -Seconds 3 }
function WslRun([string]$cmd) {
    for ($t = 1; $t -le 10; $t++) {
        $out = wsl -d Ubuntu-22.04 -- bash -c $cmd 2>&1
        $rc = $LASTEXITCODE
        $out | Out-File -FilePath $LOG -Append -Encoding utf8
        if (-not ((($out -join " ") -match "Wsl/Service|0x8007274c") -or $rc -eq -1 -or $rc -eq 255)) { return $rc }
        Note "WSL refused the connection (try $t), retrying in 60 s"; Start-Sleep -Seconds 60
    }
    return 255
}
function ResumeCoco {
    OllamaDown
    for ($t = 1; $t -le 10; $t++) {
        Start-Process wsl.exe -ArgumentList '-d Ubuntu-22.04 -- bash -c "bash /mnt/d/Personal/Master*/sim/kitchen_v3/k23/train_k23.sh FloorPlan15 yolo26s_coco"' -WindowStyle Hidden
        Start-Sleep -Seconds 120
        if ((GpuMiB) -gt 3000) { Note "K2 COCO resumed (GPU $(GpuMiB) MiB)"; return }
        Note "COCO resume not visible on the GPU (try $t, $(GpuMiB) MiB), retrying"
    }
    Note "STOP: COCO did not resume after 10 tries (start it by hand)"
}
Note "pausing COCO at the next epoch boundary"
$prc = WslRun "bash $S/pause_coco_epoch.sh"
Start-Sleep -Seconds 65
Note "COCO pause rc=$prc, GPU $(GpuMiB) MiB"
if ((GpuMiB) -ge 2000) { Note "STOP: GPU not idle after the pause -> nothing run"; exit 2 }
Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList serve -WindowStyle Hidden
$ok = $false
for ($i = 0; $i -lt 20; $i++) { Start-Sleep -Seconds 2; try { if ((Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 http://127.0.0.1:11434/api/tags).Content -match "gemma4:e2b-it-q4_K_M") { $ok = $true; break } } catch {} }
if (-not $ok) { Note "STOP: Ollama did not come up"; ResumeCoco; exit 3 }
Note "Ollama up; K3 practice resume"
python sim\final_test\k23_run\practice\launch.py practice-k3-v1 *>> $LOG
$p1 = $LASTEXITCODE
if ($p1 -eq -1 -or $p1 -eq 255) { Note "practice rc=$p1 (WSL connection signature): one retry in 90 s"; Start-Sleep -Seconds 90; python sim\final_test\k23_run\practice\launch.py practice-k3-v1 *>> $LOG; $p1 = $LASTEXITCODE }
python sim\final_test\matrix\k23_practice.py --namespace practice-k3-v1 --status *>> $LOG
Note "K3 practice rc=$p1"
if ($p1 -ne 0) { Note "STOP: K3 practice failed -> COCO resumes (fix by hand)"; ResumeCoco; exit 4 }
Note "K3 rehearsal starts"
& "C:\Program Files\Git\bin\bash.exe" sim/final_test/k23_run/chain_k23.sh rehearsal-k3-v1 *>> $LOG
$r1 = $LASTEXITCODE
Note "K3 rehearsal rc=$r1"
ResumeCoco
Note "PRACTICE-NOW DONE (practice rc=$p1, rehearsal rc=$r1)"
