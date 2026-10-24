param([string]$Scene = "FloorPlan1", [string[]]$Models = @("yolo26s", "yolo11s"))
# Train + evaluate the primary detectors on an already balanced kitchen (the smoke passed 2026-09-30 19:24).
# SAFE TO STOP AND RERUN: a model whose evaluation report exists is skipped; a model stopped mid-training resumes from
# the checkpoint ckpt_backup.sh copies out of /tmp every 2 minutes (at most ~1 epoch is lost).
$log = "D:\Personal\Master's Project\sim\kitchen_v3\balance_train.log"
Add-Content -Path $log -Value "START train-only $Scene $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
foreach ($m in $Models) {
    if (Test-Path "D:\Personal\Master's Project\sim\kitchen_v3\$Scene\eval_$m.json") {
        Add-Content -Path $log -Value "SKIP $m - already trained and evaluated"; continue }
    wsl --shutdown; Start-Sleep -Seconds 8
    Start-Process wsl -WindowStyle Hidden -ArgumentList '-d','Ubuntu-22.04','--','bash','/home/arash/ckpt_backup.sh'
    wsl -d Ubuntu-22.04 -- bash /home/arash/balance_train_v3.sh train $Scene $m
    if ($LASTEXITCODE -ne 0) { Add-Content -Path $log -Value "STOPPED: train $m exited $LASTEXITCODE"; exit 1 }
}
# Any model whose LIVE evaluation used fewer than 150 poses (YOLO26s, 2026-10-01: a smoke run had frozen 10) is re-evaluated
foreach ($m in $Models) {
    $f = "D:\Personal\Master's Project\sim\kitchen_v3\$Scene\eval_$m.json"
    if ((Test-Path $f) -and ((Get-Content $f -Raw | ConvertFrom-Json).live.poses -lt 150)) {
        Add-Content -Path $log -Value "RE-EVAL $m - its live evaluation used fewer than 150 poses"
        wsl --shutdown; Start-Sleep -Seconds 8
        wsl -d Ubuntu-22.04 -- bash /home/arash/balance_train_v3.sh eval $Scene $m
    }
}
Add-Content -Path $log -Value "ALL BALANCE+TRAIN DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
