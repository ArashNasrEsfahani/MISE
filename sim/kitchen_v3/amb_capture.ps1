# Windows driver for the ambiguity capture: fresh WSL per run; a render-guard abort is reset and resumed while runs
# keep committing episodes (same policy as capture_all.ps1). -Scene defaults to FloorPlan1 (K23-PREP 2026-10-14).
param([string]$Scene = "FloorPlan1")
$log = "D:\Personal\Master's Project\sim\kitchen_v3\amb.log"
Add-Content -Path $log -Value "START amb (ps1) $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$idle = 0
while ($true) {
    $before = (Select-String -Path $log -Pattern "$($Scene): ep " -SimpleMatch | Measure-Object).Count
    wsl --shutdown; Start-Sleep -Seconds 8
    wsl -d Ubuntu-22.04 -- bash /home/arash/amb_capture.sh $Scene
    $rc = $LASTEXITCODE
    if ($rc -eq 0) { break }
    $made = (Select-String -Path $log -Pattern "$($Scene): ep " -SimpleMatch | Measure-Object).Count - $before
    if ($made -gt 0) { $idle = 0 } else { $idle++ }
    if ($idle -lt 2) { Add-Content -Path $log -Value "amb run exited $rc after $made episode(s) - resetting WSL and resuming"; continue }
    Add-Content -Path $log -Value "STOPPED: amb exited $rc twice without progress"; exit $rc
}
Add-Content -Path $log -Value "AMB ALL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
