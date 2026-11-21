#!/usr/bin/env bash
#SBATCH --job-name=hsai-pirate
#SBATCH --partition=normal
#SBATCH --cpus-per-task=24
#SBATCH --mem=28G
#SBATCH --time=02:30:00
#SBATCH --output=artifacts/runs/slurm/%x-%A_%a.out
#SBATCH --error=artifacts/runs/slurm/%x-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/artifacts/runs/nus-asd0750-paired-eval-20261121-01.build"
stage=${1:?}
block=${SLURM_ARRAY_TASK_ID:-0}
out="artifacts/runs/nus-pirate-$stage-20261121-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-$block"
exec .venv/bin/python workflows/pirate_takeover.py "$stage" "$out" --workers "$SLURM_CPUS_PER_TASK" --start "$((block * 40))" --count 40
