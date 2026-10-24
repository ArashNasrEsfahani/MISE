param([string]$Cmd = "lcurve")
# Fresh WSL, checkpoint backup running, then one specialist-pipeline step (resumes an unfinished training run).
$log = "D:\Personal\Master's Project\sim\kitchen_v3\amb.log"
Add-Content -Path $log -Value "START amb step $Cmd $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
wsl --shutdown; Start-Sleep -Seconds 8
Start-Process wsl -WindowStyle Hidden -ArgumentList '-d','Ubuntu-22.04','--','bash','/home/arash/ckpt_backup.sh'
wsl -d Ubuntu-22.04 -- bash /home/arash/amb_train.sh $Cmd
