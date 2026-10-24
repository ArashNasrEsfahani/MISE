# While Arash is away (2026-10-20 ~21:40): wait for the K2 COCO run to END rc=0 -> --add-coco + k23_coco_meta (WSL) -> Ollama up -> K3 context fit
# (rehearsal-k3-v1, alone; never re-run chain_k23 for K3) -> K2 practice (practice-k2-v1; the A39.2 order check needs the steps before) -> status
# -> Ollama down. STRICT: stops at the first failure and leaves everything for the main session (no rehearsal, no freeze, no test here).
#   powershell -ExecutionPolicy Bypass -File sim\kitchen_v3\k23\k23_after_coco.ps1   (launch hidden / detached)
$ErrorActionPreference = "Continue"
$R = "D:\Personal\Master's Project"
Set-Location $R
$LOG = "$R\sim\kitchen_v3\k23\FloorPlan15\after_coco.log"
$CL = "$R\sim\kitchen_v3\k23\FloorPlan15\train\train_yolo26s_coco.log"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONDONTWRITEBYTECODE = "1"
function Note($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File -FilePath $LOG -Append -Encoding utf8 }
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
Note "waiting for the K2 COCO run to end"
while ($true) {
    Start-Sleep -Seconds 60
    $last = (Select-String -Path $CL -Pattern '\d{4}-\d{2}-\d{2} [\d:]{8} (START|END rc=\d+)' -AllMatches | ForEach-Object { $_.Matches } | Select-Object -Last 1).Value
    if ($last -match "END rc=(\d+)") { break }
}
Note "COCO: $last"
if ($last -notmatch "END rc=0") { Note "STOP: COCO did not end with rc 0"; exit 2 }

$rc = WslRun "bash /mnt/d/Personal/Master*/sim/kitchen_v3/k23/k23_coco_freeze_steps.sh add-coco"
Note "add-coco rc=$rc"; if ($rc -ne 0) { Note "STOP: --add-coco failed"; exit 3 }
$rc = WslRun "bash /mnt/d/Personal/Master*/sim/kitchen_v3/k23/k23_coco_freeze_steps.sh meta-write"
Note "coco_meta --write rc=$rc"; if ($rc -ne 0) { Note "STOP: k23_coco_meta --write failed"; exit 4 }
$rc = WslRun "bash /mnt/d/Personal/Master*/sim/kitchen_v3/k23/k23_coco_freeze_steps.sh meta-inspect"
Note "coco_meta inspect rc=$rc (0 = current)"; if ($rc -ne 0) { Note "STOP: k23_coco_meta state not current"; exit 5 }

Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList serve -WindowStyle Hidden
$ok = $false
for ($i = 0; $i -lt 20; $i++) { Start-Sleep -Seconds 2; try { if ((Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 http://127.0.0.1:11434/api/tags).Content -match "gemma4:e2b-it-q4_K_M") { $ok = $true; break } } catch {} }
if (-not $ok) { Note "STOP: Ollama did not come up"; exit 6 }
Note "Ollama up; K3 context fit"
python sim\final_test\matrix\k23_run.py --phase context --namespace rehearsal-k3-v1 --ollama http://127.0.0.1:11434 *>> $LOG
$c = $LASTEXITCODE
Note "K3 context fit rc=$c"
if ($c -ne 0) { Note "STOP: K3 context fit failed"; OllamaDown; exit 7 }

Note "K2 practice"
python sim\final_test\k23_run\practice\launch.py practice-k2-v1 *>> $LOG
$p = $LASTEXITCODE
python sim\final_test\matrix\k23_practice.py --namespace practice-k2-v1 --status *>> $LOG
Note "K2 practice rc=$p"
OllamaDown
if ($p -ne 0) { Note "STOP: K2 practice failed"; exit 8 }
Note "AFTER-COCO DONE (add-coco, coco_meta, K3 context fit, K2 practice all rc 0)"
