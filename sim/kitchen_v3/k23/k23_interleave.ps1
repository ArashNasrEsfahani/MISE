# K23 GPU interleave orchestrator (Arash 2026-10-18 ~00:40: "make sure some lighter simulation dataset collection work is between
# them for kitchen 3 so that the gpu has some lighter work in between heavier works"; approve 1-5 item 3 "interleaving").
# Order: wait for the K2 primary (END rc=0 + new weights) -> K3 capture block -> K2 specialist -> K3 capture block -> K2 control
# (A38 audit first) -> K3 capture block -> K2 state models -> K3 capture blocks until the K3 pipelines are done or the window ends.
# A capture block = wait_free.py (Kitchen 1 priority + operator thermal hold) then capture_k23.ps1 -MaxBatches $Block (every
# decision stays in the supervisor / k23_decide.py). A capture STOP (exit 1) disables further capture blocks (a person / the main
# session resolves it) but the training jobs go on. Training jobs run through their own launchers (heat watch 87 C x 5, resume
# from _ckpt). One GPU job at a time, never two. Never resets WSL, never signals a process.
#   Start-Process powershell -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File "...\k23_interleave.ps1"' -WindowStyle Hidden
param(
    [string]$Deadline = "2026-10-19 00:30",
    [int]$Block = 6,
    [string]$K2 = "FloorPlan15",
    [string]$K3 = "FloorPlan17",
    [switch]$ResumeControl,
    [switch]$CaptureOnly,
    [string]$StartPipeline = "",
    [switch]$OneBlock
)
$ErrorActionPreference = "Continue"
$Repo = "D:\Personal\Master's Project"
$K23 = Join-Path $Repo "sim\kitchen_v3\k23"
$Log = Join-Path $K23 "interleave.log"
$WaitFree = Join-Path $Repo "sim\final_test\_checkpoints\K2-CAPTURE\wait_free.py"
$Pipes = @("main", "topup", "close", "amb", "parity")
$script:pi = 0
$script:captureOn = $true
function Say([string]$m) { Add-Content -Path $Log -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" }
$dl = [datetime]::ParseExact($Deadline, "yyyy-MM-dd HH:mm", $null)

function Wsl([string]$cmd) {
    & wsl.exe -d Ubuntu-22.04 -- bash -c $cmd | Out-Null
    return $LASTEXITCODE
}

function Capture-Block {
    if (-not $script:captureOn) { Say "capture block skipped (capture stopped earlier)"; return }
    if ($script:pi -ge $Pipes.Count) { Say "capture block skipped (all K3 pipelines done)"; return }
    if ((Get-Date) -ge $dl.AddMinutes(-35)) { Say "capture block skipped (window ends $Deadline)"; return }
    for ($w = 0; $w -lt 4; $w++) {
        & python $WaitFree 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { break }
        Say "wait_free busy/held (try $w)"
    }
    if ($LASTEXITCODE -ne 0) { Say "capture block skipped: GPU not free after 4 waits"; return }
    $pipe = $Pipes[$script:pi]
    Say "CAPTURE BLOCK $K3 pipeline=$pipe max=$Block"
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $K23 "capture_k23.ps1") -Scene $K3 -Pipeline $pipe `
        -Deadline $Deadline -MaxBatches $Block | Out-Null
    $rc = $LASTEXITCODE
    $last = [string](@(Get-Content (Join-Path $K23 "$K3\supervisor.log") -Tail 1))[-1]
    Say "capture rc=$rc : $last"
    if ($last -match "COMPLETE") { $script:pi++; Say "pipeline $pipe COMPLETE -> next $($Pipes[[math]::Min($script:pi, $Pipes.Count - 1)])" }
    elseif ($rc -eq 1) { $script:captureOn = $false; Say "CAPTURE STOPPED for a person: further capture blocks off; training continues" }
}

function GpuTemp {
    $t = & wsl.exe -d Ubuntu-22.04 -- bash -c "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1"
    $v = 0; [void][int]::TryParse(([string](@($t)[0])).Trim(), [ref]$v); return $v
}

function Cool {
    # Arash 2026-10-18 ~01:35: "even if it stops training it will resume after cooling": wait until <= 72 C (the operator resume level)
    $n = 0
    while ($true) { $c = GpuTemp; if ($c -gt 0 -and $c -le 72) { break }; if ($n % 5 -eq 0) { Say "cooling: GPU $c C (resume at <= 72 C)" }; $n++; Start-Sleep -Seconds 60 }
    Say "cooled: GPU $c C"
}

function HeatPausedLast([string]$logpath) {
    # $true if the log's LAST run (after its last START line) ended with a HEAT PAUSE
    if (-not (Test-Path $logpath)) { return $false }
    $lines = @(Get-Content $logpath)
    $i = -1; for ($k = $lines.Count - 1; $k -ge 0; $k--) { if ($lines[$k] -match " START ") { $i = $k; break } }
    if ($i -lt 0) { return $false }
    return [bool](@($lines[$i..($lines.Count - 1)] | Where-Object { $_ -match "HEAT PAUSE" }).Count)
}

function Train([string]$name, [string]$launcher, [string]$logpath) {
    for ($try = 1; $try -le 10; $try++) {
        Say "TRAIN $name start (try $try)"
        $rc = Wsl "bash /mnt/d/Personal/Master*/sim/kitchen_v3/k23/$launcher"
        Say "TRAIN $name launcher rc=$rc"
        if ($rc -eq 0) { return 0 }
        if (HeatPausedLast $logpath) { Say "TRAIN $name HEAT PAUSE -> cool, then resume from its checkpoint"; Cool; continue }
        return $rc
    }
    Say "TRAIN ${name} - 10 tries used"; return 1
}

Say "INTERLEAVE START pid=$PID deadline=$Deadline block=$Block resumeControl=$ResumeControl captureOnly=$CaptureOnly"
if ($StartPipeline) {
    # Astra K3-IDLE-2 (2026-10-18): restart at the current pipeline; every skipped predecessor needs completion evidence
    # (an interleave.log 'pipeline <p> COMPLETE' line), else refuse. Completed pipelines are never re-run (they count as idle runs).
    $ix = [array]::IndexOf($Pipes, $StartPipeline)
    if ($ix -lt 0) { Say "REFUSED: -StartPipeline '$StartPipeline' is not one of $($Pipes -join ',')"; exit 2 }
    $logTxt = Get-Content $Log
    for ($j = 0; $j -lt $ix; $j++) {
        $p = $Pipes[$j]
        if (-not ($logTxt | Where-Object { $_ -match "pipeline $p COMPLETE" })) { Say "REFUSED: -StartPipeline $StartPipeline but no completion evidence for '$p'"; exit 2 }
    }
    $script:pi = $ix
    Say "START PIPELINE $StartPipeline (index $ix; predecessors complete: $($Pipes[0..([math]::Max(0,$ix-1))] -join ','))"
}
if ($CaptureOnly -and $OneBlock) {
    # one supervised block (use with -Block 1 for exactly one batch), then stop for review
    Capture-Block
    Say "ONE BLOCK DONE (supervised; review the QC before the next restart)"; exit 0
}
if ($CaptureOnly) {
    # K2 models all trained (control done 2026-10-18 15:44): only K3 capture blocks remain (a completed pipeline is skipped by the
    # supervisor's COMPLETE line, which advances to the next one: main -> topup -> close -> amb -> parity)
    while ($script:captureOn -and $script:pi -lt $Pipes.Count -and (Get-Date) -lt $dl.AddMinutes(-35)) { Capture-Block }
    Say "INTERLEAVE DONE"; exit 0
}
if ($ResumeControl) {
    # Arash 2026-10-18 13:07 "Resume the full session" after the 12:15 full stop: K2 primary/specialist/state are done; resume the
    # control from its _ckpt (heat auto-resume through Train), then K3 capture blocks.
    $rc = Train "control" "train_k23_amb.sh $K2 control" (Join-Path $K23 "$K2\amb\train_control.log")
    while ($script:captureOn -and $script:pi -lt $Pipes.Count -and (Get-Date) -lt $dl.AddMinutes(-35)) { Capture-Block }
    Say "INTERLEAVE DONE"; exit 0
}
# 1. wait for the K2 primary
$plog = Join-Path $K23 "$K2\train\train_yolo26s.log"
$w = Join-Path $Repo "sim\weights\v3\$K2.yolo26s.pt"
$t0 = Get-Date
while ($true) {
    $ev = @(Select-String -Path $plog -Pattern "START|END rc=" | ForEach-Object { $_.Line })
    $lastEv = [string]$ev[-1]
    if ($lastEv -match "END rc=0" -and (Test-Path $w) -and (Get-Item $w).LastWriteTime -gt $t0.AddHours(-12)) { break }
    if ($lastEv -match "END rc=[1-9]") {
        if (HeatPausedLast $plog) {
            Say "primary HEAT PAUSE -> cool, then resume from its checkpoint"; Cool
            $r = Wsl "bash /mnt/d/Personal/Master*/sim/kitchen_v3/k23/train_k23.sh $K2 yolo26s"
            Say "primary resumed run ended rc=$r"; continue
        }
        Say "primary ended rc!=0 ($lastEv), not a heat pause: waiting for the main session"
    }
    Start-Sleep -Seconds 120
}
Say "K2 primary done -> look-alike map (CPU) + interleave"
Wsl "cd /mnt/d/Personal/Master*/ && /home/arash/thor-env/bin/python sim/kitchen_v3/k23/k23_lookalike.py --scene $K2 >> sim/kitchen_v3/k23/$K2/lookalike.log 2>&1" | Out-Null
Capture-Block
$rc = Train "specialist" "train_k23_amb.sh $K2 amb26m" (Join-Path $K23 "$K2\amb\train_amb26m.log")
Capture-Block
if ($rc -eq 0) {
    $a = Wsl "cd /mnt/d/Personal/Master*/ && CUDA_VISIBLE_DEVICES= /home/arash/thor-env/bin/python sim/kitchen_v3/k23/k23_amb_train.py audit --scene $K2 --model control >> sim/kitchen_v3/k23/$K2/amb/control_audit.log 2>&1"
    Say "control audit rc=$a"
    if ($a -eq 0) { $rc = Train "control" "train_k23_amb.sh $K2 control" (Join-Path $K23 "$K2\amb\train_control.log") } else { Say "control skipped (audit failed)" }
} else { Say "control skipped (specialist rc=$rc; resume the specialist first)" }
Capture-Block
$rc = Train "state" "train_k23_state.sh $K2" (Join-Path $K23 "$K2\state\train_state.log")
while ($script:captureOn -and $script:pi -lt $Pipes.Count -and (Get-Date) -lt $dl.AddMinutes(-35)) { Capture-Block }
Say "INTERLEAVE DONE"
