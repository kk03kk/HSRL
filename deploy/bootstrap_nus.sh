#!/usr/bin/env bash
# 在独立 clone 中安装；不复制 Windows 二进制，也不修改朋友的环境。
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/slurm/common.sh
hsrl_require_compute
if [[ "$(uname -m)" != x86_64 ]]; then
  echo '本配方只验证 x86_64；aarch64 节点需重新制定 Torch/原生扩展配方。' >&2
  exit 2
fi
mkdir -p artifacts/local
if [[ -f "$HOME/.cargo/env" ]]; then source "$HOME/.cargo/env"; fi
if ! command -v rustup >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o artifacts/local/rustup-init.sh
  sh artifacts/local/rustup-init.sh -y --profile minimal --default-toolchain none --no-modify-path
  source "$HOME/.cargo/env"
fi
rustup toolchain install 1.98.1 --profile minimal
if [[ ! -x .venv/bin/python ]]; then python3.12 -m venv .venv; fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'numpy>=1.24,<3' 'pytest>=8,<10' 'maturin>=1.15,<2'
command -v cargo >/dev/null || { echo '先安装 rustup，并 source "$HOME/.cargo/env"'; exit 2; }
python -m pip install -e '.[dev,train]'
python -m pip install -e ./engine
mkdir -p artifacts/runs/slurm artifacts/models artifacts
python -m pip freeze > artifacts/local/requirements-nus-resolved.txt
python -c "from hsbg.az.train.personal import training_source_digest; print(training_source_digest())" > artifacts/local/source-revision.txt
echo '安装完成；保留当前环境，后续setup默认复用。GPU检查通过Slurm执行。'
