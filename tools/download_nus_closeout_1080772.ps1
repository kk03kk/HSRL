# Read-only server transfers. Run manually; no jobs, remote writes, or remote Python.
param([string]$CaptureDirectory = '', [switch]$PlanOnly)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$base = Join-Path $root 'artifacts\imports\nus-1080772-closeout'
if (-not $CaptureDirectory) { $CaptureDirectory = Join-Path $base 'capture-final' }
$capture = [IO.Path]::GetFullPath($CaptureDirectory)
$prefix = [IO.Path]::GetFullPath($base).TrimEnd('\') + '\'
if (-not $capture.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) { throw 'CaptureDirectory must be inside artifacts/imports/nus-1080772-closeout' }
$manifest = Get-Content -LiteralPath (Join-Path $root 'config\nus-1080772-closeout-download.json') -Raw -Encoding UTF8 | ConvertFrom-Json
New-Item -ItemType Directory -Path $capture -Force | Out-Null
$commands = [Collections.Generic.List[string]]::new()
$commands.Add('progress')
$skipped = 0
foreach ($item in $manifest.items) {
    $dest = [IO.Path]::GetFullPath((Join-Path $capture $item.destination))
    if (-not $dest.StartsWith($capture.TrimEnd('\') + '\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe manifest path' }
    New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($dest)) -Force | Out-Null
    $mode = 'get'
    if ($item.sha256 -and (Test-Path -LiteralPath $dest)) {
        $hash = (Get-FileHash -LiteralPath $dest -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($hash -eq $item.sha256) { $skipped++; continue }
        if ((Get-Item -LiteralPath $dest).Length -lt $item.bytes) { $mode = 'get -a' }
        else {
            if ($PlanOnly) { throw "Hash mismatch; retained: $dest" }
            Copy-Item -LiteralPath $dest -Destination ($dest + '.mismatch-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
        }
    }
    $local = $dest.Replace('\','/')
    if ($local.Contains('"') -or $local.Contains("`n") -or $item.source.Contains('"')) { throw 'Unsafe SFTP path' }
    $commands.Add($mode + ' "' + $item.source + '" "' + $local + '"')
}
$commands.Add('bye')
$batch = Join-Path $capture 'download.sftp.txt'
[IO.File]::WriteAllLines($batch,$commands.ToArray(),[Text.UTF8Encoding]::new($false))
Write-Host "Capture: $capture"
Write-Host "Files in manifest: $($manifest.items.Count); SHA-verified skipped: $skipped"
if ($PlanOnly) { Write-Host "Plan only; no network access: $batch"; return }
& sftp.exe -N -oBatchMode=no -oServerAliveInterval=30 -oServerAliveCountMax=3 -J hengzhi@sjump.comp.nus.edu.sg -l 16000 -b $batch hengzhi@xlogin.comp.nus.edu.sg
if ($LASTEXITCODE -ne 0) { throw "Transfer incomplete. Files retained. Run this script again: $capture" }
$checks = foreach ($item in $manifest.items) {
    $dest = Join-Path $capture $item.destination
    if (-not (Test-Path -LiteralPath $dest)) { throw "Missing: $dest" }
    $hash = (Get-FileHash -LiteralPath $dest -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($item.sha256 -and $hash -ne $item.sha256) { throw "Pinned SHA mismatch: $dest" }
    [pscustomobject]@{ path=$item.destination; bytes=(Get-Item -LiteralPath $dest).Length; sha256=$hash; pinned=[bool]$item.sha256 }
}
$cp = Get-Content -LiteralPath (Join-Path $capture 'milestones\decisions-558819917\checkpoint.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($cp.version -ne 7357) { throw 'Checkpoint version is not7357' }
foreach ($name in @('model.pt','trainer.pt')) {
    $item = $checks | Where-Object { $_.path -eq "milestones/decisions-558819917/$name" }
    if ($item.sha256 -ne $cp.sha256.PSObject.Properties[$name].Value) { throw "Checkpoint SHA mismatch: $name" }
}
[pscustomobject]@{schema='hsai.closeout_capture/v1';job=1080772;files=@($checks);download_complete=$true;trainer_semantics_verified=$false} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $capture 'capture.json') -Encoding UTF8
Write-Host 'Download and pinned7357 SHA checks passed. Trainer/model and training analysis remain pending.'
Write-Host (Join-Path $capture 'capture.json')
