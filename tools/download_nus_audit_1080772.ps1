# 用户手工执行：单次SFTP会话，只读NUS冻结产物；可接着已有目录下载。
param([string]$CaptureDirectory = '', [switch]$PlanOnly)
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$imports = Join-Path $workspace 'artifacts\imports\nus-1080772'
if (-not $CaptureDirectory) {
    $partial = @(Get-ChildItem -LiteralPath $imports -Directory -ErrorAction SilentlyContinue | Where-Object {
        (Test-Path -LiteralPath (Join-Path $_.FullName 'current\plan.json')) -and
        -not (Test-Path -LiteralPath (Join-Path $_.FullName 'capture.json'))
    } | Sort-Object Name -Descending)
    if ($partial.Count) { $CaptureDirectory = $partial[0].FullName }
    else { $CaptureDirectory = Join-Path $imports ('capture-' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
}
$capture = [IO.Path]::GetFullPath($CaptureDirectory)
$prefix = [IO.Path]::GetFullPath($imports).TrimEnd('\') + '\'
if (-not $capture.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) { throw '下载目录必须位于本项目artifacts/imports/nus-1080772内' }
New-Item -ItemType Directory -Path $capture -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$remote = 'hengzhi@xlogin.comp.nus.edu.sg'
$jump = 'hengzhi@sjump.comp.nus.edu.sg'
$serverRoot = '/home/h/hengzhi/projects/HSRL'
$run = 'personal/runs/nus-4159-economy-long-002'
$bank = 'personal/artifacts/libraries/bot-firestone-3702-economy-v2-live-v1'
$audit = 'personal/runs/nus-4159-engine-audit-002'
$transfers = [Collections.Generic.List[object]]::new()
$commands = [Collections.Generic.List[string]]::new()
$commands.Add('progress')
$queue = [Collections.Generic.List[object]]::new()
function Add-Files([string[]]$Paths,[string]$Directory) {
    New-Item -ItemType Directory -Path $Directory -Force | Out-Null
    foreach ($path in $Paths) {
        $destination = Join-Path $Directory ($path.Split('/')[-1])
        $queue.Add([pscustomobject]@{ source="$serverRoot/$path"; destination=$destination })
    }
}
$current = Join-Path $capture 'current'
Add-Files @("$run/plan.json","$run/train.config.json","$run/engineering-link.json") $current
Add-Files @("$run/initial-state/model.pt","$run/initial-state/trainer.pt") (Join-Path $current 'initial-state')
Add-Files @("$run/baselines/diagnostics.json","$run/baselines/hard-12-11000000.json","$run/baselines/hard-12-11000000.economy.json") (Join-Path $current 'baselines')
foreach ($target in @('348819917','408819917')) {
    foreach ($name in @('model.pt','trainer.pt','checkpoint.json','report.json','diagnostics.json','hard.json','hard.economy.json')) {
        Add-Files @("$run/milestones/decisions-$target/$name") (Join-Path $current "decisions-$target")
    }
}
Add-Files @("$audit/report.json","$audit/source-fork.json","$audit/train.config.json") (Join-Path $capture 'source-audit')
Add-Files @("$audit/initial-state/migration.json") (Join-Path $capture 'source-audit\initial-state')
Add-Files @("$bank/report.json","$bank/strict-audit.json") (Join-Path $capture 'bank-metadata')
foreach ($split in @('train','dev','test')) {
    Add-Files @("$bank/$split/manifest.json","$bank/$split/calibration.json") (Join-Path $capture "bank-metadata\$split")
}
# 精确固定本次回传的六个大文件。通过SHA的文件不再请求。
$pins = @{
 'initial-state/model.pt'='110e03305bdbd058d510dbc28f59d0afbc242fd033de8ad4441c10111dd2a334'
 'initial-state/trainer.pt'='0c3a221a436f6e0e08f91035371acf6cd16fe902f2ad8c0798d81515b610df70'
 'decisions-348819917/model.pt'='3ebbc28b6cd3ce10ce74707766fc63f878af74a98299e1d5285507f32ac5d9b2'
 'decisions-348819917/trainer.pt'='9cbdd490c93af693be345fc9514f2af61b20be7798693ad401a87191dcbc2e04'
 'decisions-408819917/model.pt'='eb91fd41fa9957bd259d6f0fba7cb5eecdddc91dd85a8f212d2dafaa79ad3f66'
 'decisions-408819917/trainer.pt'='718e6c43fc4abafddba0e506a2a97b4220ae44d2568ef6e7c9c1875cdd03017f'
}
$sizes = @{ 'initial-state/model.pt'=67862268L; 'initial-state/trainer.pt'=254305001L;
 'decisions-348819917/model.pt'=67864188L; 'decisions-348819917/trainer.pt'=254312873L;
 'decisions-408819917/model.pt'=67864188L; 'decisions-408819917/trainer.pt'=254312873L }
$skipped = 0
foreach ($item in $queue) {
    $relative = $item.destination.Substring($current.Length).TrimStart('\').Replace('\','/')
    $mode = 'get -a'
    if ($pins.ContainsKey($relative) -and (Test-Path -LiteralPath $item.destination)) {
        $actual = (Get-FileHash -LiteralPath $item.destination -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -eq $pins[$relative]) { $skipped++; Write-Host "已核SHA，跳过：$relative"; continue }
        # 完整长度但错误的内容不能盲目追加；保留坏副本，下载新文件。
        if ((Get-Item -LiteralPath $item.destination).Length -ge $sizes[$relative]) {
            if ($PlanOnly) { throw "发现完整但SHA不符的文件；计划模式不移动它：$($item.destination)" }
            $targetPath = [IO.Path]::GetFullPath($item.destination)
            if (-not $targetPath.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) { throw '路径越界' }
            Move-Item -LiteralPath $targetPath -Destination ($targetPath + '.mismatch-' + $stamp)
            $mode = 'get'
        }
    }
    $localPath = $item.destination.Replace('\','/')
    if ($localPath.Contains('"') -or $localPath.Contains("`n")) { throw '本地路径不能含引号或换行' }
    $commands.Add($mode + ' "' + $item.source + '" "' + $localPath + '"')
    $transfers.Add($item)
}
$commands.Add('bye')
$batch = Join-Path $capture ('sftp-' + $stamp + '.txt')
[IO.File]::WriteAllLines($batch,$commands.ToArray(),[Text.UTF8Encoding]::new($false))
Write-Host "目录：$capture"
Write-Host "已跳过 $skipped 个通过SHA的大文件；其余使用同一SFTP会话续传。"
if ($PlanOnly) { Write-Host "仅生成计划，未连接服务器：$batch"; return }
# BatchMode=no置于-b之前，保留密码交互；整个列表只建立一次SFTP会话。
& sftp.exe -N -oBatchMode=no -oServerAliveInterval=30 -oServerAliveCountMax=3 -J $jump -l 16000 -b $batch $remote
if ($LASTEXITCODE -ne 0) { throw "连接或传输失败，文件已保留。用同一-CaptureDirectory重新运行：$capture" }

# 只在本地核权重内容；不在登录节点加载模型或读取大权重校验。
$plan = Get-Content -LiteralPath (Join-Path $current 'plan.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($plan.inherited.version -ne 4159 -or $plan.reference_sha256 -ne '110e03305bdbd058d510dbc28f59d0afbc242fd033de8ad4441c10111dd2a334') { throw '来源与本次回传审计不一致' }
$pinnedModels = @{
    'initial-state'='110e03305bdbd058d510dbc28f59d0afbc242fd033de8ad4441c10111dd2a334'
    'decisions-348819917'='3ebbc28b6cd3ce10ce74707766fc63f878af74a98299e1d5285507f32ac5d9b2'
    'decisions-408819917'='eb91fd41fa9957bd259d6f0fba7cb5eecdddc91dd85a8f212d2dafaa79ad3f66'
}
$checks = [Collections.Generic.List[object]]::new()
foreach ($folder in @('initial-state','decisions-348819917','decisions-408819917')) {
    $directory = Join-Path $current $folder
    if ($folder -eq 'initial-state') { $expected = $plan.inherited.sha256 }
    else { $expected = (Get-Content -LiteralPath (Join-Path $directory 'checkpoint.json') -Raw -Encoding UTF8 | ConvertFrom-Json).sha256 }
    if ($expected.PSObject.Properties['model.pt'].Value -ne $pinnedModels[$folder]) { throw "阶段模型与本次审计不一致：$folder" }
    foreach ($name in @('model.pt','trainer.pt')) {
        $file = Join-Path $directory $name
        $hash = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
        $want = $expected.PSObject.Properties[$name].Value
        if ($hash -ne $want) { throw "哈希不一致，保留文件，不接入训练：$file" }
        $checks.Add([pscustomobject]@{ path=$file; sha256=$hash; verified=$true })
    }
}
foreach ($pair in @(@('train','train'),@('dev','hard'))) {
    $manifest = Get-Content -LiteralPath (Join-Path $capture "bank-metadata\$($pair[0])\manifest.json") -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($manifest.digest -ne $plan.libraries.PSObject.Properties[$pair[1]].Value) { throw '库元数据与计划不一致' }
}
[pscustomobject]@{
    schema='hsai.nus-import/v1'; job_id='1080772'; capture_time=(Get-Date -Format o)
    source_root=$serverRoot; transfers=$queue.ToArray(); requested_transfers=$transfers.ToArray(); verified_skipped=$skipped; weight_checks=$checks.ToArray()
    scope='迁移后4159初始完整状态、已冻结5506/6039及评测和库元数据；不含旧版原始4159或最新轮转状态、完整库帧'
    ready_for_exact_resume=$false
} | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $capture 'capture.json') -Encoding UTF8
Write-Host "下载并校验完成：$capture"
Write-Host '请回传capture.json；后续再登记到模型/数据注册表。没有修改或停止NUS作业。'
