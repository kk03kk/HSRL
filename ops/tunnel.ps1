<#
.SYNOPSIS
    建立（或自愈）到 NUS 计算集群的 SSH 端口转发隧道。

.DESCRIPTION
    跳板机 sjump.comp.nus.edu.sg 只接受密码认证，登录节点 xlogin2 在校园网内不可
    直连，NUS VPN 又不路由 192.168.5x —— 所以必须用端口转发把本机 2222 映射到
    xlogin2:22。原理与排障见 docs\architecture\TRANSPORT.md。

    两种模式：

    1. **已登记凭据（推荐，可无人值守）**
       先跑一次 `ops\tunnel.ps1 -Setup` 把跳板机密码用 Windows DPAPI 加密存到本机，
       之后本脚本可以完全自动重建隧道，Agent 也能自己调用。

    2. **未登记凭据**
       直接在当前终端提示输入密码（等价于以前的行为）。

.PARAMETER Setup
    登记密码：安全提示输入 → DPAPI 加密保存 → 编译 askpass 助手 → 实测一次认证。
    密码不会出现在命令行、进程列表、日志或本会话记录里。

.PARAMETER Forget
    删除本机保存的密码。

.PARAMETER Status
    只报告凭据与隧道状态，不做任何改动。

.PARAMETER Force
    隧道已在跑时仍然重建；并精确清理同端口的陈旧转发进程。

.PARAMETER Port
    本地监听端口，默认 2222。稽核/演练时可以换成别的端口，不影响现有隧道。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Setup     # 只需一次
    powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1            # 以后都可以
    powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Status
#>
[CmdletBinding()]
param(
    [switch]$Setup,
    [switch]$Forget,
    [switch]$Status,
    [switch]$Force,
    [int]$Port = 2222
)

$ErrorActionPreference = 'Continue'

$authScript = Join-Path $PSScriptRoot 'tunnel-auth.ps1'
if (-not (Test-Path $authScript)) { throw "missing helper: $authScript" }
. $authScript

$target = 'xlogin2.comp.nus.edu.sg'
$bastion = 'hengzhi@sjump.comp.nus.edu.sg'

function Test-TunnelUp {
    param([int]$p)
    if ($p -eq 2222) {
        $probe = & ssh hsai-nus 'echo TUNNEL_OK' 2>$null
        return ($LASTEXITCODE -eq 0 -and $probe -match 'TUNNEL_OK')
    }
    $probe = & ssh -p $p -o HostKeyAlias=$target -o BatchMode=yes `
                   -o ConnectTimeout=8 hengzhi@127.0.0.1 'echo TUNNEL_OK' 2>$null
    return ($LASTEXITCODE -eq 0 -and $probe -match 'TUNNEL_OK')
}

function Stop-StaleForward {
    param([int]$p)
    Get-CimInstance Win32_Process -Filter "Name='ssh.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match ("-L\s+{0}:{1}:22" -f $p, [regex]::Escape($target)) } |
        ForEach-Object {
            Write-Host ("  stopping stale pid {0}" -f $_.ProcessId) -ForegroundColor DarkGray
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
    Start-Sleep -Milliseconds 800
}

function New-Tunnel {
    param([int]$p, [switch]$Auto)
    $sshArgs = @('-N', '-f',
                 '-L', "${p}:${target}:22",
                 '-o', 'ServerAliveInterval=30',
                 '-o', 'ServerAliveCountMax=10',
                 '-o', 'ExitOnForwardFailure=yes')
    if ($Auto) {
        # askpass 只在没有终端、或强制时使用；force 在 Windows 上是唯一可靠的方式
        $sshArgs += @('-o', 'NumberOfPasswordPrompts=1')
        $env:SSH_ASKPASS = (Build-HsaiAskPass)
        $env:SSH_ASKPASS_REQUIRE = 'force'
        $env:DISPLAY = 'localhost:0'
        $env:HSAI_TUNNEL_CRED = (Get-HsaiTunnelPaths).CredPath
    }
    $sshArgs += $bastion
    & ssh @sshArgs
    $rc = $LASTEXITCODE
    if ($Auto) {
        Remove-Item Env:\SSH_ASKPASS, Env:\SSH_ASKPASS_REQUIRE, Env:\DISPLAY, Env:\HSAI_TUNNEL_CRED -ErrorAction SilentlyContinue
    }
    return $rc
}

function Wait-Tunnel {
    param([int]$p)
    foreach ($i in 1..5) {
        Start-Sleep -Seconds 1
        if (Test-TunnelUp -p $p) { return $true }
    }
    return $false
}

# ────────────────────────── -Status ──────────────────────────

if ($Status) {
    $paths = Get-HsaiTunnelPaths
    $secret = Test-HsaiSecret
    Write-Host 'HSAI tunnel status' -ForegroundColor Cyan
    Write-Host ("  helper     : {0}  {1}" -f $paths.HelperExe, $(if (Test-Path $paths.HelperExe) { 'present' } else { 'MISSING' }))
    Write-Host ("  ascii path : {0}" -f $paths.HelperAscii)
    Write-Host ("  credential : {0}  {1}" -f $paths.CredPath, $(if ($secret.Ok) { "ok ($($secret.Bytes) bytes)" } else { "NOT USABLE ($($secret.Reason))" }))
    Write-Host ("  tunnel 2222: {0}" -f $(if (Test-TunnelUp -p 2222) { 'UP' } else { 'DOWN' }))
    Write-Host ("  self-heal  : {0}" -f $(if ($secret.Ok -and (Test-Path $paths.HelperExe)) { 'available' } else { 'not available - run -Setup' }))
    exit 0
}

# ────────────────────────── -Forget ──────────────────────────

if ($Forget) {
    $was = Remove-HsaiSecret
    if ($was) { Write-Host 'Stored credential deleted. The tunnel keeps running until it drops.' -ForegroundColor Green }
    else      { Write-Host 'No stored credential found.' -ForegroundColor Yellow }
    exit 0
}

# ────────────────────────── -Setup ──────────────────────────

if ($Setup) {
    Write-Host 'Enrolling the sjump password for unattended tunnel rebuild.' -ForegroundColor Cyan
    Write-Host 'The password is encrypted with Windows DPAPI (current user only).' -ForegroundColor DarkGray
    Write-Host 'It will NOT appear in your command history, the process list, or any log.' -ForegroundColor DarkGray
    Write-Host ''

    $ss = Read-Host -AsSecureString -Prompt 'sjump password for hengzhi'
    if ($ss.Length -eq 0) { Write-Host 'Empty password - nothing saved.' -ForegroundColor Red; exit 2 }

    $null = Set-HsaiSecret -Secret $ss
    $ss = $null
    $check = Test-HsaiSecret
    if (-not $check.Ok) { Write-Host "Saved but cannot be decrypted back: $($check.Reason)" -ForegroundColor Red; exit 2 }
    Write-Host ("  credential stored ({0} bytes)" -f $check.Bytes) -ForegroundColor Green

    $exe = Build-HsaiAskPass
    Write-Host ("  askpass helper built: {0}" -f $exe) -ForegroundColor Green

    Write-Host ''
    Write-Host 'Verifying with a real authentication attempt against the bastion...' -ForegroundColor Cyan
    $env:SSH_ASKPASS = $exe
    $env:SSH_ASKPASS_REQUIRE = 'force'
    $env:DISPLAY = 'localhost:0'
    $env:HSAI_TUNNEL_CRED = (Get-HsaiTunnelPaths).CredPath
    $probe = cmd /c "ssh -o BatchMode=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=20 hsai-bastion ""echo HSAI_AUTH_OK"" < nul 2>&1"
    $rc = $LASTEXITCODE
    Remove-Item Env:\SSH_ASKPASS, Env:\SSH_ASKPASS_REQUIRE, Env:\DISPLAY, Env:\HSAI_TUNNEL_CRED -ErrorAction SilentlyContinue

    if ($rc -eq 0 -and ($probe -join "`n") -match 'HSAI_AUTH_OK') {
        Write-Host '  authentication OK - self-heal is armed.' -ForegroundColor Green
        Write-Host ''
        Write-Host 'From now on just run:  powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1' -ForegroundColor Green
        exit 0
    }

    Write-Host '  authentication FAILED. The password is probably wrong.' -ForegroundColor Red
    $probe | Select-Object -First 4 | ForEach-Object { Write-Host ("    | $_") -ForegroundColor DarkGray }
    Write-Host 'Re-run -Setup to try again.' -ForegroundColor Yellow
    exit 1
}

# ────────────────────────── 默认：确保隧道 ──────────────────────────

if ((Test-TunnelUp -p $Port) -and -not $Force) {
    Write-Host ("Tunnel on port {0} already up." -f $Port) -ForegroundColor Green
    Write-Host 'Use -Force to rebuild it.' -ForegroundColor DarkGray
    exit 0
}

if ($Force) {
    Write-Host ("Rebuilding tunnel on port {0}..." -f $Port) -ForegroundColor Yellow
    Stop-StaleForward -p $Port
}

$secret = Test-HsaiSecret
if ($secret.Ok) {
    Write-Host ("Creating tunnel on port {0} using the stored credential (unattended)..." -f $Port) -ForegroundColor Cyan
    $rc = New-Tunnel -p $Port -Auto
}
else {
    if ([Console]::IsInputRedirected) {
        Write-Host 'No stored credential, and there is no terminal to prompt on.' -ForegroundColor Red
        Write-Host 'Run this once in your own PowerShell window:' -ForegroundColor Yellow
        Write-Host '  powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Setup' -ForegroundColor Yellow
        exit 3
    }
    Write-Host ("Creating tunnel on port {0}. Enter the sjump password once." -f $Port) -ForegroundColor Cyan
    $rc = New-Tunnel -p $Port
}

if ($rc -ne 0) {
    Write-Host ("ssh exited {0} - tunnel NOT established." -f $rc) -ForegroundColor Red
    Write-Host 'Checks:' -ForegroundColor Yellow
    Write-Host '  * wrong or rotated sjump password   -> re-run -Setup'
    Write-Host ("  * port {0} already in use          -> try -Force" -f $Port)
    Write-Host '  * no route to sjump.comp.nus.edu.sg'
    exit $rc
}

if (Wait-Tunnel -p $Port) {
    Write-Host ("Tunnel on port {0} is up." -f $Port) -ForegroundColor Green
    Write-Host 'Verify with:  python ops/hsai_ops.py doctor' -ForegroundColor DarkGray
    exit 0
}

Write-Host 'ssh reported success but the tunnel does not answer.' -ForegroundColor Red
Write-Host 'Check ~/.ssh/config for the hsai-nus block, then retry.' -ForegroundColor Yellow
exit 1
