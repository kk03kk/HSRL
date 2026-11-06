param([string]$Python = "python")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "创建新工作区虚拟环境失败" }
}
$EnvPython = Join-Path (Get-Location) ".venv/Scripts/python.exe"
& $EnvPython -m pip install --upgrade pip
& $EnvPython -m pip install -e ".[train,dev]" maturin
if ($LASTEXITCODE -ne 0) { throw "安装依赖失败" }
& $EnvPython -m maturin build --release --auditwheel repair --manifest-path engine/crates/hsbg-engine-py/Cargo.toml --out artifacts/local/wheels --interpreter $EnvPython
if ($LASTEXITCODE -ne 0) { throw "Rust构建失败，请先安装Rust 1.98.1及对应链接工具" }
$Wheel = Get-ChildItem artifacts/local/wheels/hsbg_engine-*.whl | Sort-Object LastWriteTime -Descending | Select-Object -First 1
& $EnvPython -m pip install --force-reinstall --no-deps $Wheel.FullName
& $EnvPython -m pip install -e './client[live]'
if ($LASTEXITCODE -ne 0) { throw "客户端安装失败" }
