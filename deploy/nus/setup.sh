#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p runs/setup
system_python="${HSRL_SYSTEM_PYTHON:-/usr/bin/python3}"
fetch() {
  "$system_python" - "$1" "$2" <<'PY'
import shutil, sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=60) as response, open(sys.argv[2], 'wb') as target:
    shutil.copyfileobj(response, target)
PY
}
# No sudo, no global pip writes, no changes to the user's shell startup files.
if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  if ! "$system_python" -m venv .venv; then
    fetch https://bootstrap.pypa.io/virtualenv.pyz runs/setup/virtualenv.pyz
    sha256sum runs/setup/virtualenv.pyz > runs/setup/virtualenv.sha256
    "$system_python" runs/setup/virtualenv.pyz --no-periodic-update .venv
  fi
fi
if [[ ! -x "$CARGO_HOME/bin/rustup" ]]; then
  fetch https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init runs/setup/rustup-init
  sha256sum runs/setup/rustup-init > runs/setup/rustup-init.sha256
  chmod u+x runs/setup/rustup-init
  runs/setup/rustup-init -y --no-modify-path --profile minimal --default-toolchain 1.98.1
fi
rustup toolchain install 1.98.1 --profile minimal --component rustfmt --component clippy
python -m pip install --upgrade 'pip==25.3'
python -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/nus.txt
python -m maturin develop --release --locked
cargo fmt --all -- --check
cargo clippy --release --locked --all-targets --jobs 2 -- -D warnings
cargo test --release --locked -p taverngym --jobs 2
python -m pytest tests -q
python -m taverngym.rl.doctor --config configs/ppo_smoke.json --device cpu --out runs/setup/doctor-cpu.json
python -m pip freeze > runs/setup/pip-freeze.txt
rustc -Vv > runs/setup/rustc.txt
printf '\nSetup complete; CUDA is validated by the GPU smoke job.\n'
