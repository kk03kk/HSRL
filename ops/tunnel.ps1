<#
.SYNOPSIS
    建立到 NUS 计算集群的 SSH 端口转发隧道（唯一需要人工输密码的步骤）。

.DESCRIPTION
    跳板机 sjump.comp.nus.edu.sg 只接受密码认证，登录节点 xlogin2 在校园网内不可
    直连，NUS VPN 又不路由 192.168.5x —— 所以必须用端口转发把本机 2222 映射到
    xlogin2:22。一次密码，隧道存活期间所有操作免密。

    原理与排障见 docs/architecture/TRANSPORT.md。

.PARAMETER Force
    已有一条隧道在跑时仍然重建（默认直接返回成功）。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1
#>
[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

function Test-Tunnel {
    $probe = & ssh hsai-nus 'echo TUNNEL_OK' 2>$null
    return ($LASTEXITCODE -eq 0 -and $probe -match 'TUNNEL_OK')
}

if (Test-Tunnel) {
    if (-not $Force) {
        Write-Host 'Tunnel already up (hsai-nus -> xlogin2). Nothing to do.' -ForegroundColor Green
        Write-Host 'Use -Force to rebuild it.' -ForegroundColor DarkGray
        exit 0
    }
    Write-Host 'Rebuilding an existing tunnel...' -ForegroundColor Yellow
    # 只杀我们自己的转发：按命令行精确匹配，避免误伤用户其他 ssh 会话
    Get-CimInstance Win32_Process -Filter "Name='ssh.exe'" |
        Where-Object { $_.CommandLine -match '-L\s+2222:xlogin2\.comp\.nus\.edu\.sg:22' } |
        ForEach-Object {
            Write-Host ("  stopping pid {0}" -f $_.ProcessId) -ForegroundColor DarkGray
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
    Start-Sleep -Milliseconds 800
}

Write-Host 'Creating tunnel. You will be asked for the sjump password ONCE.' -ForegroundColor Cyan

$sshArgs = @(
    '-N', '-f',
    '-L', '2222:xlogin2.comp.nus.edu.sg:22',
    '-o', 'ServerAliveInterval=30',
    '-o', 'ServerAliveCountMax=10',
    '-o', 'ExitOnForwardFailure=yes',
    'hengzhi@sjump.comp.nus.edu.sg'
)

& ssh @sshArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Host "ssh exited with $LASTEXITCODE - tunnel NOT established." -ForegroundColor Red
    Write-Host 'Common causes:' -ForegroundColor Yellow
    Write-Host '  * wrong sjump password'
    Write-Host '  * port 2222 already taken by a stale process (try -Force)'
    Write-Host '  * no network route to sjump.comp.nus.edu.sg (VPN not needed for the bastion)'
    exit $LASTEXITCODE
}

# ssh -f 会 fork 到后台，稍等一下再确认
$ok = $false
foreach ($attempt in 1..5) {
    Start-Sleep -Seconds 1
    if (Test-Tunnel) { $ok = $true; break }
}

if ($ok) {
    Write-Host 'Tunnel is up. Verify with:  python ops/hsai_ops.py doctor' -ForegroundColor Green
    exit 0
}

Write-Host 'ssh returned success but the tunnel does not answer.' -ForegroundColor Red
Write-Host 'Check ~/.ssh/config has the hsai-nus block, then retry.' -ForegroundColor Yellow
exit 1
