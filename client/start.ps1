# 默认离线检查；不启动游戏、不创建窗口后端、不发送输入。
param(
 [ValidateSet('Doctor','Rehearsal','Decide')][string]$Mode='Doctor',
 [string]$Snapshot='',
 [string]$Trace='',
 [string]$GameDir=''
)
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$python=Join-Path $root '.venv/Scripts/python.exe'
if(-not (Test-Path -LiteralPath $python)){
 $shared = & git -C $root rev-parse --path-format=absolute --git-common-dir
 if($LASTEXITCODE -eq 0){$python=Join-Path (Split-Path $shared -Parent) '.venv/Scripts/python.exe'}
}
if(-not (Test-Path -LiteralPath $python)){throw '请先运行deploy/setup_windows.ps1安装本工作区环境'}
$settings=Get-Content -LiteralPath (Join-Path $root 'config/live/desktop-models.json') -Raw -Encoding utf8 | ConvertFrom-Json
$engine=Join-Path $root $settings.engine
if(-not (Test-Path -LiteralPath $engine -PathType Container)){throw '缺少已校验的 Player150 引擎'}
$originalPythonPath=$env:PYTHONPATH
$env:PYTHONPATH=$engine
$stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
try {
 if($Mode -eq 'Rehearsal') {
  if(-not $Trace){throw 'Player150 演练需要 -Trace 指定 decisions-full.jsonl.gz'}
  & $python -X utf8 (Join-Path $PSScriptRoot 'offline_rehearsal.py') --trace $Trace --limit 0 --out (Join-Path $root "artifacts/local/rehearsal-$stamp")
 } elseif($Mode -eq 'Decide') {
  if(-not $Snapshot){throw 'Decide 需要 -Snapshot 指定 Player150 离线公开快照 JSON'}
  & $python -X utf8 (Join-Path $PSScriptRoot 'live_245.py') decide --snapshot $Snapshot --out (Join-Path $root "artifacts/local/decision-$stamp.json")
 } else {
  $doctorArgs=@('doctor','--infer','--out',(Join-Path $root "artifacts/local/client-doctor-$stamp.json"))
  if($GameDir){$doctorArgs+=@('--game-dir',$GameDir)}
  & $python -X utf8 (Join-Path $PSScriptRoot 'live_245.py') @doctorArgs
 }
 if($LASTEXITCODE -ne 0){throw "离线检查失败，退出码$LASTEXITCODE；未启动实机"}
} finally {
 $env:PYTHONPATH=$originalPythonPath
}
