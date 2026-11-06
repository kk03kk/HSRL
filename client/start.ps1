# 默认离线检查；不启动游戏、不创建窗口后端、不发送输入。
param([ValidateSet('Doctor','Rehearsal','Decide')][string]$Mode='Doctor',[string]$Snapshot='')
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$python=Join-Path $root '.venv/Scripts/python.exe'
if(-not (Test-Path -LiteralPath $python)){throw '请先运行deploy/setup_windows.ps1安装本工作区环境'}
$env:PYTHONDONTWRITEBYTECODE='1';$env:PYTHONUTF8='1'
$stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
if($Mode -eq 'Rehearsal') {
 & $python (Join-Path $PSScriptRoot 'offline_rehearsal.py') --out (Join-Path $root "artifacts/local/rehearsal-$stamp")
} elseif($Mode -eq 'Decide') {
 if(-not $Snapshot){throw 'Decide需要-Snapshot指定离线公开快照JSON'}
 & $python (Join-Path $PSScriptRoot 'live_245.py') decide --snapshot $Snapshot --out (Join-Path $root "artifacts/local/decision-$stamp.json")
} else {
 & $python (Join-Path $PSScriptRoot 'live_245.py') doctor --infer --out (Join-Path $root "artifacts/local/client-doctor-$stamp.json")
}
if($LASTEXITCODE -ne 0){throw "离线检查失败，退出码$LASTEXITCODE；未启动实机"}
