param([string[]]$Scenes = @("FloorPlan1"))
# Windows-side driver: for each kitchen, reset WSL (fresh display), then capture it.
# A fail-fast abort caused by the render guard (a glitch BURST) gets a WSL reset and a resume. Bursts recur every
# ~15 min on this machine (2026-09-29), so the budget is PROGRESS-based, not a fixed count: keep resetting while each
# run commits at least one new episode; stop only after two consecutive runs that commit nothing (a genuinely broken
# display) or after any abort that is not the render guard.
$log = "D:\Personal\Master's Project\sim\kitchen_v3\capture_all.log"
Add-Content -Path $log -Value "START (ps1) $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
function Count-Episodes($scene) {
    return (Select-String -Path $log -Pattern "$($scene): ep " -SimpleMatch | Measure-Object).Count
}
foreach ($s in $Scenes) {
    $tries = 0
    $idle = 0
    while ($true) {
        $before = Count-Episodes $s
        wsl --shutdown
        Start-Sleep -Seconds 8
        $tries++
        Add-Content -Path $log -Value "WSL reset before $s (try $tries) $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        wsl -d Ubuntu-22.04 -- bash /home/arash/capture_one_v3.sh $s
        $rc = $LASTEXITCODE
        if ($rc -eq 0) { break }
        $made = (Count-Episodes $s) - $before
        if ($made -gt 0) { $idle = 0 } else { $idle++ }
        $tail = (Get-Content -Path $log -Tail 6) -join " "
        if ($rc -eq 3 -and $tail -match "render guard" -and $idle -lt 2) {
            Add-Content -Path $log -Value "guard burst in $s after $made new episode(s) - resetting WSL and resuming"
            continue
        }
        Add-Content -Path $log -Value "STOPPED: $s exited $rc (fail-fast; $idle run(s) in a row without progress) - remaining kitchens NOT started"
        exit $rc
    }
}
Add-Content -Path $log -Value "ALL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
