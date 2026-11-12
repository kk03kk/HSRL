<#
.SYNOPSIS
    隧道凭据：DPAPI 加密存储 + SSH_ASKPASS 助手。被 ops\tunnel.ps1 点源引用。

.DESCRIPTION
    为什么需要它：跳板机 sjump 只接受密码认证，隧道断了就必须有人输一次密码。
    把密码用 Windows DPAPI（CurrentUser 作用域）加密存到本机后，askpass 助手可以
    在无人值守时把它交给 ssh，隧道就能自愈。

    实测得到的两条硬约束（2026-11-12，OpenSSH_for_Windows_9.5p2）：

    1. Win32-OpenSSH **支持** SSH_ASKPASS，但必须 SSH_ASKPASS_REQUIRE=force，
       否则它只在没有终端时才用（Windows 上判断不可靠）。
    2. askpass 助手的**可执行文件路径必须全是 ASCII**。放在 C:\Users\黄\... 下时
       ssh 报 `CreateProcessW failed error:2` / `posix_spawnp: No such file or directory`。
       所以 exe 放 C:\ProgramData\HSAI\，而密钥料（DPAPI 密文）留在用户配置目录
       —— .NET 读 Unicode 路径没问题，只有 CreateProcessW 不行。

    安全边界：DPAPI CurrentUser 意味着密文只有当前 Windows 账户能解开；同一台机器的
    其他用户即使读到文件也解不开。明文只在内存里出现，用完立即清零。

    编码注意：本文件是 **UTF-8 with BOM**。Windows PowerShell 5.1 对没有 BOM 的脚本按
    系统 ANSI 代码页读取，中文会被错误解码并打乱行结构（曾导致上面的 C# here-string
    整段被当成 PowerShell 代码）。tests/test_ops_guard.py 有对应的守卫。
#>

# 注意：这里**不要**设置 $ErrorActionPreference。本文件是被点源引用的库，
# 改调用者的错误策略会让调用者意外中止（曾因此让测试脚本提前退出）。

# Windows PowerShell 5.1 不会自动加载 System.Security，必须显式引入才能用 ProtectedData
try { Add-Type -AssemblyName System.Security -ErrorAction Stop } catch {
    Write-Warning "无法加载 System.Security，DPAPI 不可用: $($_.Exception.Message)"
}

#: askpass 助手所在目录。**必须是纯 ASCII**（见上面第 2 条）。
$script:HsaiHelperDir = 'C:\ProgramData\HSAI'
$script:HsaiHelperExe = Join-Path $script:HsaiHelperDir 'hsai-askpass.exe'
$script:HsaiCredPath  = Join-Path $env:LOCALAPPDATA 'HSAI\tunnel.cred'

#: 助手源码：读取 DPAPI 密文、解密、把明文写到 stdout。纯 ASCII，避免编译期编码问题。
$script:HsaiAskPassSource = @'
using System;
using System.IO;
using System.Security.Cryptography;

public class HsaiAskPass
{
    public static int Main(string[] args)
    {
        string path = Environment.GetEnvironmentVariable("HSAI_TUNNEL_CRED");
        if (string.IsNullOrEmpty(path))
        {
            path = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "HSAI", "tunnel.cred");
        }
        if (!File.Exists(path))
        {
            Console.Error.WriteLine("hsai-askpass: credential not found: " + path);
            return 2;
        }

        byte[] blob = File.ReadAllBytes(path);
        byte[] plain = null;
        try
        {
            plain = ProtectedData.Unprotect(blob, null, DataProtectionScope.CurrentUser);
        }
        catch (Exception e)
        {
            Console.Error.WriteLine("hsai-askpass: cannot decrypt (different Windows user, " +
                                    "or the credential was re-enrolled): " + e.Message);
            return 3;
        }

        try
        {
            using (Stream stdout = Console.OpenStandardOutput())
            {
                stdout.Write(plain, 0, plain.Length);
                stdout.Flush();
            }
        }
        finally
        {
            Array.Clear(plain, 0, plain.Length);
        }
        return 0;
    }
}
'@

function Get-HsaiTunnelPaths {
    [pscustomobject]@{
        HelperDir  = $script:HsaiHelperDir
        HelperExe  = $script:HsaiHelperExe
        CredPath   = $script:HsaiCredPath
        HelperAscii = ($script:HsaiHelperExe -match '^[\x20-\x7E]+$')
    }
}

function Build-HsaiAskPass {
    <#
      编译 askpass 助手。已存在且比本脚本新就跳过（除非 -Force）。
      返回 exe 路径。
    #>
    param([switch]$Force)

    $paths = Get-HsaiTunnelPaths
    if (-not $paths.HelperAscii) {
        throw "askpass 助手路径含非 ASCII 字符，ssh 无法启动它: $($paths.HelperExe)"
    }
    if (-not (Test-Path $paths.HelperDir)) {
        New-Item -ItemType Directory -Force -Path $paths.HelperDir | Out-Null
    }

    $stale = $true
    if (Test-Path $paths.HelperExe) {
        $exeTime = (Get-Item $paths.HelperExe).LastWriteTimeUtc
        $srcTime = (Get-Item $PSCommandPath).LastWriteTimeUtc
        $stale = $exeTime -lt $srcTime
        if (-not $Force -and -not $stale) { return $paths.HelperExe }
    }

    if (Test-Path $paths.HelperExe) { Remove-Item $paths.HelperExe -Force }
    Add-Type -TypeDefinition $script:HsaiAskPassSource `
             -OutputAssembly $paths.HelperExe `
             -OutputType ConsoleApplication `
             -ReferencedAssemblies 'System.dll', 'System.Security.dll'
    return $paths.HelperExe
}

function Set-HsaiSecret {
    <#
      把 SecureString 用 DPAPI(CurrentUser) 加密后写到 $script:HsaiCredPath。
      只接受 SecureString —— 明文不进参数、不进历史、不进日志。
    #>
    param([Parameter(Mandatory = $true)][Security.SecureString]$Secret)

    if ($Secret.Length -eq 0) { throw "密码为空，拒绝保存" }

    $bstr = [IntPtr]::Zero
    $plain = $null
    try {
        $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secret)
        # 直接用 BSTR 的原始字节，不经过 .NET 字符串，减少明文副本
        $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
        $bytes = [Text.Encoding]::UTF8.GetBytes($plain)
        $blob = [Security.Cryptography.ProtectedData]::Protect(
            $bytes, $null, [Security.Cryptography.DataProtectionScope]::CurrentUser)
    }
    finally {
        if ($bstr -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
        $plain = $null
        if ($bytes) { [Array]::Clear($bytes, 0, $bytes.Length) }
    }

    $dir = Split-Path -Parent $script:HsaiCredPath
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    [IO.File]::WriteAllBytes($script:HsaiCredPath, $blob)
    [Array]::Clear($blob, 0, $blob.Length)

    # 仅当前用户可读写（用户配置目录本来就是这样，显式再钉一次）
    try {
        $me = [Security.Principal.WindowsIdentity]::GetCurrent().Name
        & icacls $script:HsaiCredPath /inheritance:r /grant:r "${me}:(R,W)" | Out-Null
    }
    catch {
        Write-Warning "无法收紧凭据文件 ACL（不影响功能）: $($_.Exception.Message)"
    }

    return $script:HsaiCredPath
}

function Test-HsaiSecret {
    <#
      验证凭据存在且能解密。**只返回长度，绝不返回密码本身。**
    #>
    if (-not (Test-Path $script:HsaiCredPath)) {
        return [pscustomobject]@{ Ok = $false; Bytes = 0; Reason = 'not enrolled' }
    }
    try {
        $blob = [IO.File]::ReadAllBytes($script:HsaiCredPath)
        $plain = [Security.Cryptography.ProtectedData]::Unprotect(
            $blob, $null, [Security.Cryptography.DataProtectionScope]::CurrentUser)
        try {
            return [pscustomobject]@{ Ok = ($plain.Length -gt 0); Bytes = $plain.Length; Reason = 'ok' }
        }
        finally {
            [Array]::Clear($plain, 0, $plain.Length)
        }
    }
    catch {
        return [pscustomobject]@{
            Ok = $false; Bytes = 0
            Reason = "decrypt failed: $($_.Exception.Message)"
        }
    }
}

function Remove-HsaiSecret {
    if (Test-Path $script:HsaiCredPath) {
        Remove-Item $script:HsaiCredPath -Force
        return $true
    }
    return $false
}
