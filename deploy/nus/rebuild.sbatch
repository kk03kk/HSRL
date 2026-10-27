#!/usr/bin/env bash
#SBATCH --job-name=hsrl-rebuild
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/nus/env.sh
python tools/package_nus.py --verify .
printf 'Content-addressed native cache: %s\n' "$CARGO_TARGET_DIR"
python -m maturin develop --release --locked
cargo fmt --all -- --check
cargo clippy --release --locked --all-targets --jobs 2 -- -D warnings
cargo test --release --locked -p taverngym --jobs 2
python -m pytest tests -q
python tools/validate_training.py --games 256 --out runs/qualification-v2/cpu-stress.json
python -m taverngym.rl.doctor --config configs/ppo_smoke.json --device cpu
