# K23 OVERNIGHT runner launcher (Windows). Runs sim\final_test\k23_run\k23_overnight.py: Arash's Kitchen 2 / 3 sequence, unattended and
# STRICT (stops at the first failure and leaves everything for the morning; never forces, never skips a gate, never edits
# preregistration.md; git add + commit of evidence only). Steps, rules and logs: see the docstring of k23_overnight.py.
#
#   Dry run (foreground; checks every precondition, prints the plan, runs nothing):
#     powershell -NoProfile -ExecutionPolicy Bypass -File sim\final_test\k23_run\k23_overnight.ps1 -DryRun
#   Launch hidden (from the repo root, e.g. D:\Personal\Master's Project):
#     Start-Process powershell -WorkingDirectory (Get-Location) -WindowStyle Hidden -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File sim\final_test\k23_run\k23_overnight.ps1'
#   Watch: sim\final_test\k23_run\overnight\{overnight.log, status.json, STOPPED.json | DONE.json}
#   Options: -NoCocoResume (stop instead of resuming a paused / WSL-killed COCO run), -MaxWaitHours N (default 36)
param([switch]$DryRun, [switch]$NoCocoResume, [double]$MaxWaitHours = 36)
$ErrorActionPreference = "Continue"
$R = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $R
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONDONTWRITEBYTECODE = "1"
$py = @("sim\final_test\k23_run\k23_overnight.py")
if ($DryRun) {
    python @py --dry-run
    exit $LASTEXITCODE
}
$py += @("--run", "--max-wait-hours", "$MaxWaitHours")
if ($NoCocoResume) { $py += "--no-coco-resume" }
$out = Join-Path $R "sim\final_test\k23_run\overnight"
New-Item -ItemType Directory -Force $out | Out-Null
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') launcher start (pid $PID): python $($py -join ' ')" | Out-File -FilePath (Join-Path $out "console.txt") -Append -Encoding utf8
python @py *>> (Join-Path $out "console.txt")
$rc = $LASTEXITCODE
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') launcher end rc=$rc" | Out-File -FilePath (Join-Path $out "console.txt") -Append -Encoding utf8
exit $rc
