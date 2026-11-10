# Deployment build of the Windows hsbg_engine wheel.
#
# Architecture (docs/ARCHITECTURE.md, S3): the NUS server owns the simulator
# source; this Windows box only *deploys* the resulting wheel. Run this after
# the server has bumped engine/ and recorded the new source_hash in
# docs/sync/ENGINE_RELEASES.md.
#
#   powershell -ExecutionPolicy Bypass -File client/tools/build_engine.ps1
#   powershell -ExecutionPolicy Bypass -File client/tools/build_engine.ps1 -Install
#
# Requires: rustup toolchain (engine/rust-toolchain.toml), MSVC C++ build tools
# (link.exe + Windows SDK) and maturin (Anaconda's python already has it).
param(
    [switch]$Install,          # pip-install the built wheel into .venv
    [string]$Python = "",      # python used for maturin; default: Anaconda
    [string]$CargoHome = "D:\tmp\hsai-migration\cargo"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # client/tools -> repo root
if (-not (Test-Path (Join-Path $repo "engine\Cargo.toml"))) { throw "run from the HSAI checkout (engine/ not found under $repo)" }

# 1. MSVC environment (vcvars64.bat) -------------------------------------------------
$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = $null
if (Test-Path $vswhere) {
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
$vcvars = if ($vsPath) { Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat" } else { $null }
if (-not $vcvars -or -not (Test-Path $vcvars)) {
    throw "MSVC C++ build tools not found. Install 'Desktop development with C++' via Visual Studio Installer (see docs/ARCHITECTURE.md S3)."
}
Write-Host "[vcvars] $vcvars"

# 2. Rust / cargo -------------------------------------------------------------------
$cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
if (-not (Test-Path (Join-Path $cargoBin "cargo.exe"))) { throw "cargo not found at $cargoBin (install rustup)" }
if ($CargoHome -and (Test-Path $CargoHome)) { $env:CARGO_HOME = $CargoHome; Write-Host "[cargo] CARGO_HOME=$CargoHome (reusing cache)" }

# 3. maturin ------------------------------------------------------------------------
if (-not $Python) {
    $Python = if (Test-Path "D:\Anaconda\python.exe") { "D:\Anaconda\python.exe" } else { "python" }
}
& $Python -m maturin --version | Out-Null
if ($LASTEXITCODE -ne 0) { throw "maturin missing in $Python (pip install maturin)" }

# 4. Build inside the vcvars environment --------------------------------------------
$commands = @(
    "call `"$vcvars`"",
    "set PATH=$cargoBin;%PATH%",
    "cd /d `"$repo\engine`"",
    "`"$Python`" -m maturin build --release --out `"$repo\artifacts\local\wheels`""
) -join " && "
Write-Host "[build] maturin build --release"
cmd /c $commands
if ($LASTEXITCODE -ne 0) { throw "maturin build failed (exit $LASTEXITCODE)" }

$wheel = Get-ChildItem (Join-Path $repo "artifacts\local\wheels") -Filter "hsbg_engine-*.whl" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $wheel) { throw "no wheel produced" }
Write-Host "[wheel] $($wheel.FullName)"
Write-Host ("[wheel] sha256 {0}" -f (Get-FileHash $wheel.FullName -Algorithm SHA256).Hash.ToLower())

if ($Install) {
    & (Join-Path $repo ".venv\Scripts\python.exe") -m pip install --force-reinstall --no-deps $wheel.FullName
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
    $hash = & (Join-Path $repo ".venv\Scripts\python.exe") -c "import hsbg_engine as E; print(E.build_info()['source_hash'])"
    Write-Host "[verify] runtime source_hash = $hash"
    Write-Host "[verify] record it in config/live/engine.json and docs/sync/ENGINE_RELEASES.md"
}
