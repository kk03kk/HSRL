#!/usr/bin/env bash
#SBATCH --job-name=hsrl-strategy
#SBATCH --partition=normal
#SBATCH --cpus-per-task=12
#SBATCH --mem=14G
#SBATCH --time=00:40:00

set -euo pipefail
run="$(realpath "${1:?provide a run directory}")"
start="${2:?provide the first seed}"
games="${3:?provide the number of paired seeds}"
panel="${4:?provide a new output folder name}"
mode="${5:-hsrl}"
cd "$(git rev-parse --show-toplevel)"
test -z "$(git status --porcelain)"
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS="$SLURM_CPUS_PER_TASK"
export CARGO_TARGET_DIR="$run/target"

printf 'commit=%s\njob_id=%s\nseed_start=%s\nseed_count=%s\n' \
    "$(git rev-parse HEAD)" "$SLURM_JOB_ID" "$start" "$games" > "$run/$panel-manifest.txt"
cargo run --locked --release -p taverngym --example strategy_experiment -- \
    "$start" "$games" "$SLURM_CPUS_PER_TASK" "$run/$panel" "$mode"
