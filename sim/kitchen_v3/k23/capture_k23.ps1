# K23 capture supervisor (Windows side, EXCLUSIVE). It EXECUTES; every decision is made by k23_decide.py (Python,
# selftested offline). FloorPlan15 / FloorPlan17 perception capture, pipelines probe | thermaltest | main only.
#
# Once per run: authorization record valid now (k23_decide auth: dated, scoped, script hashes pinned, ledger headings,
# A28 item-15 items closed) -> no other capture / training supervisor -> Kitchen 1 unchanged gate.
# Per batch: decide pre (signed window + deadline, outstanding / terminal / unresolved QC, budgets across the whole
# window) -> delete the old receipt -> capture_one_k23.sh <scene> <pipeline> <n> <deadline_epoch> <invocation>
# -> decide post (receipt validated: invocation id, scene, pipeline, typed fields, WSL rc, QC report) -> act.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File sim\kitchen_v3\k23\capture_k23.ps1 -Scene FloorPlan15 `
#       -Pipeline main -Deadline "2026-10-14 10:30" [-FirstBatchOnly] [-BatchSize 8] [-MaxBatches 12]
#   ... -ResetWslNow   (a PERSON's action: preflight, then `wsl --shutdown`, counted; the supervisor never resets by itself)
#   ... -SelfTest      (offline: the Preflight function returns exactly one Boolean on every path; nothing else runs)
#
# Exit 0 = stopped as planned; 1 = STOP for a person; 3 = YIELD (a Kitchen 1 stage / another job needs the
# simulator or the GPU: nothing started, resume later -- A28 15.9).
param(
    [ValidateSet("FloorPlan15", "FloorPlan17")][string]$Scene = "FloorPlan15",
    [ValidateSet("probe", "thermaltest", "main", "topup", "close", "amb", "parity")][string]$Pipeline = "main",
    [string]$Deadline = "",
    [int]$BatchSize = 8,
    [int]$MaxBatches = 12,
    [int]$BatchMinutes = 30,
    [int]$MinFreeGbWsl = 40,
    [int]$MinFreeGbWin = 15,      # C: holds the WSL disk image (38 GB); 34 GB free on 2026-10-14; one kitchen main ~5 GB
    [switch]$FirstBatchOnly,
    [switch]$ResetWslNow,
    [switch]$SelfTest
)
$ErrorActionPreference = "Continue"      # native stderr must not throw (PowerShell 5.1); every check is explicit
$Repo = "D:\Personal\Master's Project"
$K23 = Join-Path $Repo "sim\kitchen_v3\k23"
$Dir = Join-Path $K23 $Scene
New-Item -ItemType Directory -Force -Path $Dir | Out-Null
$Log = Join-Path $Dir "supervisor.log"
$WslRepo = "/mnt/d/Personal/Master's Project"
$Decide = Join-Path $K23 "k23_decide.py"

# Logging NEVER goes to the success stream (a function's return value would include it: Astra K23-PREP 2).
function Say([string]$m) { $l = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"; Add-Content -Path $Log -Value $l; Write-Host $l }
function Stop-Run([string]$why) { Say "STOP: $why"; exit 1 }
function Decide([string[]]$argv) {
    $out = & python $Decide @argv 2>$null
    $rc = $LASTEXITCODE
    $line = @($out | Where-Object { $_ -match '^\{' } | Select-Object -Last 1)
    if ($rc -ne 0 -or $line.Count -ne 1) { return $null }
    try { return ($line[0] | ConvertFrom-Json) } catch { return $null }
}

# The ONLY continuation actions are "continue" and "retry" (case-sensitive); "done" ends cleanly; anything else --
# "stop", unknown, empty, missing, a non-string -- is a STOP for a person (Astra K23-PREP r2 6).
function Next-Step($post) {
    if ($null -eq $post) { return "stop" }
    $act = $post.action
    if (-not ($act -is [string])) { return "stop" }
    if ($act -ceq "continue" -or $act -ceq "retry") { return "loop" }
    if ($act -ceq "done") { return "done" }
    return "stop"
}

# Preflight: returns exactly ONE [bool]; any doubt = $false. $script:Mock* are the selftest hooks.
$script:MockWslRc = $null; $script:MockGpuApps = $null; $script:MockFreeGb = $null
function Preflight {
    [OutputType([bool])] param()
    if ($null -ne $script:MockGpuApps) { $gpuApps = @($script:MockGpuApps) }
    else { $gpuApps = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -match '^(ollama|ollama_llama_server|llama-server)$' } | ForEach-Object { $_.ProcessName }) }
    if ($gpuApps.Count -gt 0) { Say "  preflight FAIL: GPU apps on Windows: $($gpuApps -join ',')"; return $false }
    $drive = (Get-Item $env:LOCALAPPDATA).PSDrive.Name          # the WSL disk image lives on this drive
    foreach ($d in @($drive, "D")) {
        if ($null -ne $script:MockFreeGb) { $free = [int]$script:MockFreeGb } else { $free = [math]::Floor((Get-PSDrive $d).Free / 1GB) }
        if ($free -lt $MinFreeGbWin) { Say "  preflight FAIL: ${d}: has $free GB free (< $MinFreeGbWin)"; return $false }
        Say "  preflight ok: ${d}: $free GB free"
    }
    if ($null -ne $script:MockWslRc) { $rc = [int]$script:MockWslRc }
    else {
        $out = & wsl -d Ubuntu-22.04 --cd $WslRepo -- bash sim/kitchen_v3/k23/preflight_k23.sh $MinFreeGbWsl 2>&1
        $rc = $LASTEXITCODE
        foreach ($l in @($out)) { Say "  preflight: $l" }
    }
    if ($rc -ne 0) { Say "  preflight FAIL: WSL preflight rc $rc"; return $false }
    return $true
}

if ($SelfTest) {
    $fails = 0
    $Log = Join-Path $env:TEMP "k23_selftest_supervisor.log"
    foreach ($case in @(@{w = 1; g = @(); f = 100; exp = $false; n = "WSL preflight fails (lock busy)" },
                        @{w = 0; g = @("ollama"); f = 100; exp = $false; n = "Ollama running" },
                        @{w = 0; g = @(); f = 3; exp = $false; n = "disk low" },
                        @{w = 0; g = @(); f = 100; exp = $true; n = "all clear" })) {
        $script:MockWslRc = $case.w; $script:MockGpuApps = $case.g; $script:MockFreeGb = $case.f
        $r = @(Preflight)
        $okType = ($r.Count -eq 1) -and ($r[0] -is [bool])
        if (-not $okType -or $r[0] -ne $case.exp) { $fails++; Write-Host "SELFTEST FAIL: $($case.n): got $($r.Count) value(s) [$($r -join ',')]" }
        else { Write-Host "selftest ok: $($case.n) -> $($r[0])" }
        $branch = $false
        if (Preflight) { $branch = $true }                      # the if-test itself (the original bug) must follow the bool
        if ($branch -ne $case.exp) { $fails++; Write-Host "SELFTEST FAIL: $($case.n) in an if-test took the wrong branch" }
    }
    foreach ($c in @(@{j = '{"action": "continue"}'; e = "loop" }, @{j = '{"action": "retry"}'; e = "loop" },
                     @{j = '{"action": "done"}'; e = "done" }, @{j = '{"action": "stop"}'; e = "stop" },
                     @{j = '{"action": "Continue"}'; e = "stop" }, @{j = '{"action": "continue "}'; e = "stop" },
                     @{j = '{"action": "launch"}'; e = "stop" }, @{j = '{"action": ""}'; e = "stop" },
                     @{j = '{"action": null}'; e = "stop" }, @{j = '{"why": "no action"}'; e = "stop" },
                     @{j = '{"action": ["continue"]}'; e = "stop" }, @{j = '{"action": 1}'; e = "stop" })) {
        $got = Next-Step ($c.j | ConvertFrom-Json)
        if ($got -cne $c.e) { $fails++; Write-Host "SELFTEST FAIL: post $($c.j) -> $got (expected $($c.e))" }
        else { Write-Host "selftest ok: post $($c.j) -> $got" }
    }
    if ((Next-Step $null) -cne "stop") { $fails++; Write-Host "SELFTEST FAIL: a missing post decision did not stop" }
    $bad = Decide @("pre", "--scene", $Scene, "--pipeline", "main", "--deadline", "not-a-date")
    if ($null -ne $bad -and $bad.action -ne "stop") { $fails++; Write-Host "SELFTEST FAIL: a broken decide call did not stop" }
    if ($fails) { Write-Host "SUPERVISOR SELFTEST FAILED ($fails)"; exit 1 }
    Write-Host "SUPERVISOR SELFTEST OK"; exit 0
}

if (-not $Deadline) { Stop-Run "-Deadline 'YYYY-MM-DD HH:MM' is required (an absolute, dated hard stop)" }
Say "SUPERVISOR START $Scene pipeline=$Pipeline batch=$BatchSize maxBatches=$MaxBatches deadline=$Deadline"

# ---- once per run
$authArgs = @("auth", "--scene", $Scene, "--pipeline", $Pipeline)
if ($ResetWslNow) { $authArgs += "--check-only" }
$auth = Decide $authArgs
if ($null -eq $auth -or -not $auth.ok) { Stop-Run "not authorized: $(if ($auth) { $auth.why -join '; ' } else { 'decide failed' })" }
Say "authorized by ledger entry: $($auth.ledger_entry)"
$me = $PID
$others = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.ProcessId -ne $me -and $_.CommandLine -match 'capture_k23\.ps1|capture_all\.ps1|amb_capture\.ps1|balance_train\.ps1|train_only\.ps1|amb_train\.ps1' })
if ($others.Count -gt 0) { Stop-Run "another capture / training supervisor is running (pid $($others.ProcessId -join ','))" }
$k1 = & python (Join-Path $K23 "k1_unchanged.py") 2>&1
$k1rc = $LASTEXITCODE
foreach ($l in @($k1)) { Say "  k1: $l" }
if ($k1rc -ne 0) { Stop-Run "Kitchen 1 unchanged gate FAILED" }

if ($ResetWslNow -and $auth.precondition -ne "freeze_summary") {
    # A28 15.9 condition 3 / Astra r6: under the parallel basis other jobs (Kitchen 1) may use WSL at any time; the
    # preflight is not an exclusive reservation, so `wsl --shutdown` is never allowed here
    Stop-Run "-ResetWslNow is refused under the parallel basis ($($auth.precondition)): another job may be using WSL"
}
if ($ResetWslNow) {
    # a PERSON resets WSL (fresh WSLg display, the Kitchen 1 recipe) -- at the start of the window, or after a stop for
    # a render-guard burst. Never automatic: `wsl --shutdown` kills every WSL job (Astra K23-PREP r2 5).
    $pf = Preflight
    if (-not ($pf -is [bool]) -or -not $pf) { Stop-Run "preflight failed: NOT shutting WSL down (another job may be running)" }
    & wsl --shutdown
    if ($LASTEXITCODE -ne 0) { Stop-Run "wsl --shutdown exited $LASTEXITCODE" }
    Start-Sleep -Seconds 8
    $rs = Decide @("reset-done", "--scene", $Scene, "--deadline", $Deadline)
    if ($null -eq $rs) { Stop-Run "could not record the WSL reset" }
    if ([int]$rs.resets -gt [int]$rs.cap) { Stop-Run "restart cap exceeded ($($rs.resets) > $($rs.cap) WSL resets in this window)" }
    Say "WSL reset $($rs.resets) of max $($rs.cap) done after a passed preflight"; exit 0
}

$n = 0
$Prio = Join-Path $K23 "k1_priority_k23.py"
while ($true) {
    if ($n -ge $MaxBatches) { Say "planned batches done ($n)"; exit 0 }
    # A28 15.9 condition 2 (Astra K23-PREP r5 6): Kitchen 1 first -- checked before EVERY batch, fail closed. Busy or
    # undecidable = YIELD: stop at this batch boundary (the capture state is saved per episode), exit 3, poll later.
    $pr = & python $Prio 2>$null
    $prc = $LASTEXITCODE
    $pline = @($pr | Where-Object { $_ -match '^\{' } | Select-Object -Last 1)
    if ($prc -ne 0 -or $pline.Count -ne 1) { Say "YIELD (Kitchen 1 priority, rc $prc): $($pline -join ' ')"; exit 3 }
    try { $pj = $pline[0] | ConvertFrom-Json } catch { $pj = $null }
    if ($null -eq $pj -or $pj.free -ne $true) { Say "YIELD (Kitchen 1 priority): $($pline -join ' ')"; exit 3 }
    $pre = Decide @("pre", "--scene", $Scene, "--pipeline", $Pipeline, "--deadline", $Deadline, "--batch-minutes", "$BatchMinutes")
    if ($null -eq $pre) { Stop-Run "decide pre failed" }
    if ($pre.action -ceq "done") { Say $pre.why; exit 0 }
    if (-not ($pre.action -is [string]) -or $pre.action -cne "launch") { Stop-Run "pre: $($pre.why)" }
    $inv = [string]$pre.invocation
    if ($inv.Length -ne 32) { Stop-Run "no valid invocation id" }
    $receipt = Join-Path $Dir "last_batch.json"
    Remove-Item -Force -ErrorAction SilentlyContinue $receipt, "$receipt.tmp"
    if (Test-Path $receipt) { Stop-Run "could not delete the old receipt" }
    Say "batch $($n + 1): $Pipeline max_new=$BatchSize invocation=$inv"
    & wsl -d Ubuntu-22.04 --cd $WslRepo -- bash sim/kitchen_v3/k23/capture_one_k23.sh $Scene $Pipeline $BatchSize $pre.deadline_epoch $inv
    $wslrc = $LASTEXITCODE
    $pargs = @("post", "--scene", $Scene, "--pipeline", $Pipeline, "--deadline", $Deadline, "--invocation", $inv, "--wsl-rc", "$wslrc")
    if ($FirstBatchOnly) { $pargs += "--first-batch-only" }
    $post = Decide $pargs
    if ($null -eq $post) { Stop-Run "decide post failed (WSL rc $wslrc)" }
    Say "result: $($post.action) -- $($post.why)"
    $step = Next-Step $post
    if ($step -ceq "done") { exit 0 }
    if ($step -ceq "loop") { $n++; continue }
    exit 1                                    # "stop" and anything unknown or missing
}
