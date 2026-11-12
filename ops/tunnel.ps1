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

.PARAMETER Verify
    用已登记的凭据在**临时端口**上真的建一条转发、探通、再拆掉，
    证明自愈可用。不碰现有的 2222 隧道。

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
    [switch]$Verify,
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
    # ConnectTimeout 只覆盖 TCP 连接；对端死掉但端口仍在监听时，卡住的是 SSH banner，
    # 所以要配合 Stop-StaleForward 使用，否则每次探测都会等满 ConnectTimeout。
    if ($p -eq 2222) {
        $probe = & ssh hsai-nus 'echo TUNNEL_OK' 2>$null
        return ($LASTEXITCODE -eq 0 -and $probe -match 'TUNNEL_OK')
    }
    $probe = & ssh -p $p -o HostKeyAlias=$target -o BatchMode=yes `
                   -o ConnectionAttempts=1 -o ConnectTimeout=8 `
                   hengzhi@127.0.0.1 'echo TUNNEL_OK' 2>$null
    return ($LASTEXITCODE -eq 0 -and $probe -match 'TUNNEL_OK')
}

function Get-FreePort {
    $l = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, 0)
    $l.Start()
    $p = $l.LocalEndpoint.Port
    $l.Stop()
    return $p
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
    <#
      起一个后台端口转发。

      注意两点（都踩过）：
      * **不要用 ssh -f 并把输出接到管道**。fork 出来的子进程会继承那个 pipe，
        cmd/PowerShell 于是永远等不到 EOF，命令挂死。
      * 因此用 Start-Process 把 ssh 完全独立出去，输出重定向到文件，不 -Wait。
    #>
    param([int]$p, [switch]$Auto)

    $logDir = 'C:\ProgramData\HSAI'
    if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force -Path $logDir | Out-Null }
    $log = Join-Path $logDir ("tunnel-{0}.log" -f $p)

    $sshArgs = @('-N',
                 '-L', "${p}:${target}:22",
                 '-o', 'ServerAliveInterval=30',
                 '-o', 'ServerAliveCountMax=10',
                 '-o', 'ExitOnForwardFailure=yes')
    if ($Auto) { $sshArgs += @('-o', 'NumberOfPasswordPrompts=1') }
    $sshArgs += $bastion

    $saved = @{}
    if ($Auto) {
        $saved['SSH_ASKPASS'] = $env:SSH_ASKPASS
        $saved['SSH_ASKPASS_REQUIRE'] = $env:SSH_ASKPASS_REQUIRE
        $saved['DISPLAY'] = $env:DISPLAY
        $saved['HSAI_TUNNEL_CRED'] = $env:HSAI_TUNNEL_CRED
        $env:SSH_ASKPASS = (Build-HsaiAskPass)
        $env:SSH_ASKPASS_REQUIRE = 'force'
        $env:DISPLAY = 'localhost:0'
        $env:HSAI_TUNNEL_CRED = (Get-HsaiTunnelPaths).CredPath
    }

    try {
        $proc = Start-Process -FilePath 'ssh' -ArgumentList $sshArgs `
                              -WindowStyle Hidden -PassThru `
                              -RedirectStandardOutput ($log + '.out') `
                              -RedirectStandardError $log
    }
    finally {
        if ($Auto) {
            foreach ($k in $saved.Keys) {
                if ($null -eq $saved[$k]) { Remove-Item ("Env:\" + $k) -ErrorAction SilentlyContinue }
                else { Set-Item ("Env:\" + $k) $saved[$k] }
            }
        }
    }
    return [pscustomobject]@{ Process = $proc; Log = $log }
}

function Wait-Tunnel {
    param([int]$p, [object]$Proc)
    foreach ($i in 1..8) {
        Start-Sleep -Milliseconds 700
        if (Test-TunnelUp -p $p) { return $true }
        if ($Proc -and $Proc.HasExited) { return $false }
    }
    return $false
}

function Show-TunnelLog {
    param([string]$LogPath)
    if ($LogPath -and (Test-Path $LogPath)) {
        $text = Get-Content $LogPath -Raw -ErrorAction SilentlyContinue
        if ($text) {
            $text.Trim().Split("`n") | Select-Object -First 6 |
                ForEach-Object { Write-Host ("    | " + $_.TrimEnd()) -ForegroundColor DarkGray }
        }
    }
}

function Test-BastionAuth {
    <#
      验证已登记的凭据能否真正打通隧道。

      **不能用 `ssh hsai-bastion 'echo ok'`**：sjump 是一个只做端口转发、没有 shell 的
      跳板机，认证成功后会返回 `This account is currently not available.`，看起来像失败
      其实是正常现象。唯一可靠的验证是真的建一条转发，从里面探一次，再拆掉。
    #>
    param([int]$ProbePort)

    if (-not $ProbePort -or $ProbePort -le 0) { $ProbePort = Get-FreePort }
    Stop-StaleForward -p $ProbePort
    $r = New-Tunnel -p $ProbePort -Auto
    if (-not (Wait-Tunnel -p $ProbePort -Proc $r.Process)) {
        return [pscustomobject]@{ Ok = $false; Log = $r.Log }
    }
    $ok = Test-TunnelUp -p $ProbePort
    Stop-StaleForward -p $ProbePort
    return [pscustomobject]@{ Ok = $ok; Log = $r.Log }
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
    Write-Host 'Verifying with a real port-forward (sjump has no shell, so a plain command cannot be used)...' -ForegroundColor Cyan
    $auth = Test-BastionAuth
    if ($auth.Ok) {
        Write-Host '  authentication OK - self-heal is armed.' -ForegroundColor Green
        Write-Host ''
        Write-Host 'From now on just run:  powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1' -ForegroundColor Green
        exit 0
    }

    Write-Host '  authentication FAILED.' -ForegroundColor Red
    Show-TunnelLog -LogPath $auth.Log
    Write-Host ''
    Write-Host 'Reading the log: "Permission denied (password)" means the password is wrong.' -ForegroundColor Yellow
    Write-Host 'Re-run -Setup to try again.' -ForegroundColor Yellow
    exit 1
}

# ────────────────────────── -Verify（不碰现有隧道） ──────────────────────────

if ($Verify) {
    $secret = Test-HsaiSecret
    if (-not $secret.Ok) {
        Write-Host ("No usable credential ({0}). Run -Setup first." -f $secret.Reason) -ForegroundColor Red
        exit 2
    }
    Write-Host 'Checking the stored credential with a throwaway forward on a free port...' -ForegroundColor Cyan
    $auth = Test-BastionAuth
    if ($auth.Ok) {
        Write-Host 'self-heal verified: the stored credential can rebuild the tunnel.' -ForegroundColor Green
        exit 0
    }
    Write-Host 'self-heal FAILED:' -ForegroundColor Red
    Show-TunnelLog -LogPath $auth.Log
    exit 1
}

# ────────────────────────── 默认：确保隧道 ──────────────────────────

if ((Test-TunnelUp -p $Port) -and -not $Force) {
    Write-Host ("Tunnel on port {0} already up." -f $Port) -ForegroundColor Green
    Write-Host 'Use -Force to rebuild it.' -ForegroundColor DarkGray
    exit 0
}

# 探测失败时端口可能**仍然被一个死掉的转发进程占着**：TCP 能连上，但远端的
# SSH banner 永远不来，于是每次探测都要等满 ConnectTimeout（实测 15s × 8 次 = 120s
# 挂死）。所以先清掉陈旧转发再重建，让后续探测立刻被 refused。
if (-not $Force) {
    Write-Host ("Port {0} does not answer; clearing any stale forward first." -f $Port) -ForegroundColor DarkGray
}
Stop-StaleForward -p $Port

$secret = Test-HsaiSecret
$auto = $secret.Ok
if ($auto) {
    Write-Host ("Creating tunnel on port {0} using the stored credential (unattended)..." -f $Port) -ForegroundColor Cyan
}
else {
    if ([Console]::IsInputRedirected) {
        Write-Host 'No stored credential, and there is no terminal to prompt on.' -ForegroundColor Red
        Write-Host 'Run this once in your own PowerShell window:' -ForegroundColor Yellow
        Write-Host '  powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Setup' -ForegroundColor Yellow
        exit 3
    }
    Write-Host ("Creating tunnel on port {0}. Enter the sjump password once." -f $Port) -ForegroundColor Cyan
}

$r = New-Tunnel -p $Port -Auto:$auto

if (Wait-Tunnel -p $Port -Proc $r.Process) {
    Write-Host ("Tunnel on port {0} is up." -f $Port) -ForegroundColor Green
    Write-Host 'Verify with:  python ops/hsai_ops.py doctor' -ForegroundColor DarkGray
    exit 0
}

Write-Host ("Tunnel on port {0} did NOT come up." -f $Port) -ForegroundColor Red
Show-TunnelLog -LogPath $r.Log
Write-Host 'Checks:' -ForegroundColor Yellow
Write-Host '  * wrong or rotated sjump password   -> re-run -Setup'
Write-Host ("  * port {0} already in use          -> try -Force" -f $Port)
Write-Host '  * no route to sjump.comp.nus.edu.sg'
exit 1
