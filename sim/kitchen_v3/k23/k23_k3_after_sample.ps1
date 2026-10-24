# Windows runner after k23_k3_post_state2.sh (Arash 2026-10-20 16:15: K2 practice/rehearsal only after COCO -> K3 practice + rehearsal first):
# waits for the exact line "K3-POST2 DONE sample=written" -> K3 M8 (uses table + inventory k23-v1) + commit -> FA equivalence K2 + K3 -> Ollama up -> K3 practice
# (launch.py practice-k3-v1) -> status -> K3 rehearsal (chain_k23.sh rehearsal-k3-v1) -> Ollama down -> K2 COCO in WSL. Every failure is
# logged and the runner still ends by starting COCO (the GPU is not left idle). K2 practice / rehearsal / freezes / tests stay manual.
#   powershell -ExecutionPolicy Bypass -File sim\kitchen_v3\k23\k23_k3_after_sample.ps1   (launch hidden / detached)
$ErrorActionPreference = "Continue"
$R = "D:\Personal\Master's Project"
Set-Location $R
$CL = "$R\sim\kitchen_v3\k23\FloorPlan17\chain_gpu.log"
$LOG = "$R\sim\kitchen_v3\k23\FloorPlan17\after_sample.log"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONDONTWRITEBYTECODE = "1"
function Note($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File -FilePath $LOG -Append -Encoding utf8 }
function OllamaDown { Get-Process "ollama app", "ollama", "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep -Seconds 3 }
function GpuMiB { try { [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits).Trim()) } catch { 0 } }
function WslRun([string]$cmd) {
    # the WSL service sometimes refuses NEW connections under memory pressure (Wsl/Service/0x8007274c): retry up to 10 x 60 s
    for ($t = 1; $t -le 10; $t++) {
        $out = wsl -d Ubuntu-22.04 -- bash -c $cmd 2>&1
        $rc = $LASTEXITCODE
        $out | Out-File -FilePath $LOG -Append -Encoding utf8
        if (-not ((($out -join " ") -match "Wsl/Service|0x8007274c") -or $rc -eq -1 -or $rc -eq 255)) { return $rc }
        Note "WSL refused the connection (try $t), retrying in 60 s"; Start-Sleep -Seconds 60
    }
    return 255
}
function StartCoco {
    OllamaDown
    for ($t = 1; $t -le 10; $t++) {
        Start-Process wsl.exe -ArgumentList '-d Ubuntu-22.04 -- bash -c "bash /mnt/d/Personal/Master*/sim/kitchen_v3/k23/train_k23.sh FloorPlan15 yolo26s_coco"' -WindowStyle Hidden
        Start-Sleep -Seconds 120
        if ((GpuMiB) -gt 3000) { Note "K2 COCO started (resumes from epoch 33; GPU $(GpuMiB) MiB)"; return }
        Note "COCO start not visible on the GPU (try $t, $(GpuMiB) MiB), retrying"
    }
    Note "STOP: COCO did not start after 10 tries (start it by hand)"
}
Note "waiting for 'K3-POST2 DONE'"
$line = $null
while (-not $line) {
    Start-Sleep -Seconds 60
    $line = Select-String -Path $CL -Pattern '^\d{4}-\d{2}-\d{2} [\d:]{8} K3-POST2 DONE sample=(written|missing)$' | Select-Object -Last 1
}
Note "seen: $($line.Line)"
if ($line.Line -notmatch "sample=written") { Note "STOP: no K3 sample -> COCO only"; StartCoco; exit 2 }

# M8
python sim\kitchen_v3\k23\k23_k3_m8_finalize.py *>> $LOG
if ($LASTEXITCODE -ne 0) { Note "STOP: M8 finalize refused -> COCO only"; StartCoco; exit 3 }
git add sim/step5/uses_table_FloorPlan17.json sim/step5/homeowner_inventory_FloorPlan17.json 2>$null
git commit -q -m "K3 M8: uses table + homeowner inventory finalized k23-v1 (frozen B5 list, perception freeze record) by k23_k3_m8_finalize.py

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>" 2>$null
Note "M8 done and committed"

# A29.1 item 4: FA equivalence runs (10 items per kitchen, reference plans, no model; owed for K2), GPU + sim lock free here
foreach ($k in @("FloorPlan15", "FloorPlan17")) {
    $erc = WslRun "bash /mnt/d/Personal/Master*/sim/kitchen_v3/k23/a29_equiv.sh $k"
    Note "FA equivalence $k rc=$erc"
}

# Ollama up
Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList serve -WindowStyle Hidden
$ok = $false
for ($i = 0; $i -lt 20; $i++) { Start-Sleep -Seconds 2; try { if ((Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 http://127.0.0.1:11434/api/tags).Content -match "gemma4:e2b-it-q4_K_M") { $ok = $true; break } } catch {} }
if (-not $ok) { Note "STOP: Ollama did not come up -> COCO only"; StartCoco; exit 4 }
Note "Ollama up"

# K3 practice
python sim\final_test\k23_run\practice\launch.py practice-k3-v1 *>> $LOG
$prc = $LASTEXITCODE
if ($prc -eq -1 -or $prc -eq 255) {
    Note "K3 practice rc=$prc (WSL connection failure signature): one retry in 90 s"; Start-Sleep -Seconds 90
    python sim\final_test\k23_run\practice\launch.py practice-k3-v1 *>> $LOG
    $prc = $LASTEXITCODE
}
python sim\final_test\matrix\k23_practice.py --namespace practice-k3-v1 --status *>> $LOG
Note "K3 practice launch rc=$prc"
if ($prc -ne 0) { Note "STOP: K3 practice failed -> COCO (fix by hand)"; StartCoco; exit 5 }

# K3 rehearsal
& "C:\Program Files\Git\bin\bash.exe" sim/final_test/k23_run/chain_k23.sh rehearsal-k3-v1 *>> $LOG
$rrc = $LASTEXITCODE
Note "K3 rehearsal chain rc=$rrc"
StartCoco
Note "AFTER-SAMPLE DONE (practice rc=$prc, rehearsal rc=$rrc)"
