#!/usr/bin/env bash
#SBATCH --job-name=hsrl-simulator
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=00:45:00

set -euo pipefail
run="$(realpath "${1:?provide a run directory}")"
cd "$(git rev-parse --show-toplevel)"
test -z "$(git status --porcelain)"
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS="$SLURM_CPUS_PER_TASK"
export CARGO_TARGET_DIR="$run/target"
printf 'job_id=%s\ncommit=%s\nhost=%s\n' "$SLURM_JOB_ID" "$(git rev-parse HEAD)" "$(hostname)" > "$run/build.txt"
cargo test --locked --release 2>&1 | tee "$run/rust-tests.log"
cargo run --locked --release -p taverngym --example hsrl_validate -- \
    "$run" "${2:-8192}" "${3:-512}" "$SLURM_CPUS_PER_TASK" 2>&1 | tee "$run/validation.log"
