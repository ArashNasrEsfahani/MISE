# K3 rehearsal NOW (practice-k3-v1 complete 20:28; capture freeze view merged), in a COCO pause: pause COCO at an epoch boundary -> GPU idle
# check -> Ollama up -> chain_k23.sh rehearsal-k3-v1 (resumes at capture) -> Ollama down -> COCO resumes (verified on the GPU).
#   powershell -ExecutionPolicy Bypass -File sim\kitchen_v3\k23\k23_k3_rehearsal_now.ps1   (launch hidden / detached)
$ErrorActionPreference = "Continue"
$R = "D:\Personal\Master's Project"
Set-Location $R
$LOG = "$R\sim\kitchen_v3\k23\FloorPlan17\rehearsal_now.log"
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
Note "Ollama up"
Note "K3 rehearsal starts"
& "C:\Program Files\Git\bin\bash.exe" sim/final_test/k23_run/chain_k23.sh rehearsal-k3-v1 *>> $LOG
$r1 = $LASTEXITCODE
Note "K3 rehearsal rc=$r1"
ResumeCoco
Note "REHEARSAL-NOW DONE (rehearsal rc=$r1)"
