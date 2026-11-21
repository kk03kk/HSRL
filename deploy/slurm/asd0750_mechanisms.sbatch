#!/usr/bin/env bash
#SBATCH --job-name=hsai-asd0750-replay
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec .venv/bin/python tools/replay_asd0750_mechanisms.py "${1:?shared evidence}" "${2:?replay output}" \
    --workers "${SLURM_CPUS_PER_TASK:?}"
