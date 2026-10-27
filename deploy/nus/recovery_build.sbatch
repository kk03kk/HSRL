#!/usr/bin/env bash
#SBATCH --job-name=hsrl-recovery-build
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:45:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/nus/env.sh
python tools/package_nus.py --verify .
# Build into the new release, without pip install or changing the shared environment.
python -m maturin build --release --locked --out runs/recovery-wheel
wheel=$(find runs/recovery-wheel -maxdepth 1 -name '*.whl' -print -quit)
python tools/install_native_wheel.py "$wheel"
cargo fmt --all -- --check
cargo test --release --locked -p taverngym --jobs 2
python -m pytest tests -q
python -m taverngym.rl.doctor --config configs/ppo_recovery_cpu.json --device cpu
