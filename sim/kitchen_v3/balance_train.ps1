param([string]$Scene = "FloorPlan1")
# v3 pilot (Arash's go 2026-09-30 "run 1 and 2 back to back"): balance the kitchen (top-ups + close protocol + detection
# gate), prove the training chain with a 1-epoch smoke, then train + evaluate YOLO26s and YOLO11s.
# Resumable: the balance phase resumes per episode; rerun this script after a stop.
$log = "D:\Personal\Master's Project\sim\kitchen_v3\balance_train.log"
Add-Content -Path $log -Value "START balance+train $Scene $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$idle = 0
while ($true) {
    wsl --shutdown; Start-Sleep -Seconds 8
    wsl -d Ubuntu-22.04 -- bash /home/arash/balance_train_v3.sh balance $Scene
    $rc = $LASTEXITCODE
    if ($rc -eq 0) { break }
    $tail = (Get-Content -Path $log -Tail 8) -join " "
    if ($rc -eq 3 -and $tail -match "render guard" -and $idle -lt 3) { $idle++; continue }
    Add-Content -Path $log -Value "STOPPED: balance $Scene exited $rc - training NOT started"
    exit $rc
}
wsl --shutdown; Start-Sleep -Seconds 8
wsl -d Ubuntu-22.04 -- bash /home/arash/balance_train_v3.sh smoke $Scene yolo26s
if ($LASTEXITCODE -ne 0) {
    Add-Content -Path $log -Value "STOPPED: training smoke failed ($LASTEXITCODE) - real training NOT started"
    exit 1
}
foreach ($m in @("yolo26s", "yolo11s")) {
    wsl --shutdown; Start-Sleep -Seconds 8
    wsl -d Ubuntu-22.04 -- bash /home/arash/balance_train_v3.sh train $Scene $m
    if ($LASTEXITCODE -ne 0) { Add-Content -Path $log -Value "STOPPED: train $m exited $LASTEXITCODE"; exit 1 }
}
Add-Content -Path $log -Value "ALL BALANCE+TRAIN DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
