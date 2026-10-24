# A/B from WINDOWS (the browser's path into WSL): old console (git HEAD worktree) vs current, measured by
# console_bench.py running in Windows Python against the WSL IP.
$bench = "D:\Personal\Master's Project\sim\kitchen_v3\console_bench.py"
$wip = (wsl -d Ubuntu-22.04 -- hostname -I).Trim().Split(' ')[0]
foreach ($v in @(@("old", "/mnt/d/Personal/oldconsole/sim"), @("new", "/mnt/d/Personal/Master's Project/sim"))) {
    wsl -d Ubuntu-22.04 -- pkill -f "[w]ebapp.py"; wsl -d Ubuntu-22.04 -- pkill -f "[t]hor-Linux64"; Start-Sleep 3
    $script = "cd '" + $v[1].Replace("'", "'\''") + "' && MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA DISPLAY=:0 exec /home/arash/thor-env/bin/python webapp.py > /home/arash/abw_" + $v[0] + ".log 2>&1"
    Set-Content -Path "$env:TEMP\abw_run.sh" -Value $script -NoNewline -Encoding ascii
    wsl -d Ubuntu-22.04 -- bash -c "cp /mnt/c/Users/arash/AppData/Local/Temp/abw_run.sh ~/abw_run.sh"
    Start-Process wsl -WindowStyle Hidden -ArgumentList '-d','Ubuntu-22.04','--','bash','/home/arash/abw_run.sh'
    Start-Sleep 5
    python $bench $v[0] "http://$wip`:8000"
}
wsl -d Ubuntu-22.04 -- pkill -f "[w]ebapp.py"; wsl -d Ubuntu-22.04 -- pkill -f "[t]hor-Linux64"
