#!/usr/bin/env bash
#SBATCH --job-name=hsai-asd0750-eval
#SBATCH --partition=normal
#SBATCH --cpus-per-task=24
#SBATCH --mem=28G
#SBATCH --time=02:30:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
out="${HSAI_ASD_OUT:?}"
build="${out}.build"
mkdir "$build"
export PYO3_PYTHON="$PWD/.venv/bin/python"
export CARGO_TARGET_DIR="$PWD/$build/target"
export PATH="$HOME/.cargo/bin:$PATH"
cargo +1.98.1 build --manifest-path tools/asd_obs2_native/Cargo.toml --locked --offline --release
cp "$CARGO_TARGET_DIR/release/lib_asd_obs2.so" "$build/_asd_obs2.abi3.so"
export PYTHONPATH="$PWD/$build"
exec .venv/bin/python workflows/asd0750_paired_eval.py \
    --out "$out" --seeds 96 --workers "${SLURM_CPUS_PER_TASK:?}" \
    --asd artifacts/models/asd-policy-0750-public-20261121/policy-0750.pt \
    --static artifacts/models/asd-policy-0750-public-20261121/author-card-static.f32 \
    --r1 artifacts/runs/nus-peer-rounds-production-20261120-1/round-01/train-1170497/fit/train/snapshots/final_000647/model.pt \
    --final artifacts/runs/nus-player150-five-20261120-01/round-05/train/fit/train/snapshots/final_003286/model.pt
