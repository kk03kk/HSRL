# Manual transfer of one compressed bundle; old partial per-file downloads are retained.
param([ValidateSet('Prepare','Download')][string]$Mode='Prepare', [string]$BundleJob='')
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$remote='hengzhi@xlogin.comp.nus.edu.sg'
$jump='hengzhi@sjump.comp.nus.edu.sg'
$stage='/home/h/hengzhi/projects/HSRL/personal/artifacts/exports/closeout-transfer'
$local=Join-Path $root 'artifacts\imports\nus-1080772-closeout\bundles'
New-Item -ItemType Directory -Path $local -Force | Out-Null
$lines=[Collections.Generic.List[string]]::new()
if ($Mode -eq 'Prepare') {
    $lines.Add('-mkdir "' + $stage + '"')
    foreach ($pair in @(@('tools/pack_nus_closeout_1080772.py','pack_nus_closeout_1080772.py'),@('deploy/slurm/closeout_pack.sbatch','closeout_pack.sbatch'))) {
        $source=(Join-Path $root $pair[0]).Replace('\','/')
        $lines.Add('put "' + $source + '" "' + $stage + '/' + $pair[1] + '"')
    }
} else {
    if ($BundleJob -notmatch '^\d+$') { throw 'Specify the CPU packing job number with -BundleJob' }
    $local=Join-Path $local ('bundle-'+$BundleJob)
    New-Item -ItemType Directory -Path $local -Force | Out-Null
    $lines.Add('get "' + $stage + '/bundle-' + $BundleJob + '/nus-1080772-closeout.tar.gz.sha256" "' + $local.Replace('\','/') + '/nus-1080772-closeout.tar.gz.sha256"')
    $lines.Add('get -a "' + $stage + '/bundle-' + $BundleJob + '/nus-1080772-closeout.tar.gz" "' + $local.Replace('\','/') + '/nus-1080772-closeout.tar.gz"')
}
$lines.Add('bye')
$batch=Join-Path $local ('transfer-'+$Mode+'.sftp.txt')
[IO.File]::WriteAllLines($batch,$lines.ToArray(),[Text.UTF8Encoding]::new($false))
& sftp.exe -N -oBatchMode=no -oServerAliveInterval=30 -oServerAliveCountMax=3 -J $jump -b $batch $remote
if ($LASTEXITCODE -ne 0) { throw 'Transfer incomplete. Files retained. Repeat the same command to resume the bundle.' }
if ($Mode -eq 'Prepare') {
    Write-Host 'Uploaded two small files. On NUS: cd ~/projects/HSRL && mkdir -p personal/runs/slurm && sbatch personal/artifacts/exports/closeout-transfer/closeout_pack.sbatch'
} else {
    & (Join-Path $root '.venv\Scripts\python.exe') (Join-Path $root 'tools\unpack_nus_closeout.py') --archive (Join-Path $local 'nus-1080772-closeout.tar.gz') --checksum (Join-Path $local 'nus-1080772-closeout.tar.gz.sha256') --out (Join-Path $local 'extracted')
    if ($LASTEXITCODE -ne 0) { throw 'Integrity/unpack verification failed; all evidence retained.' }
}
