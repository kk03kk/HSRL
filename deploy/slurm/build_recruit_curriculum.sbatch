#!/usr/bin/env bash
#SBATCH --job-name=hsai-start-bank
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
.venv/bin/python -m pytest tests/test_recruit_start.py tests/test_process_learning.py -q
exec .venv/bin/python workflows/build_recruit_curriculum.py \
  --bank artifacts/runs/nus-recruit-bank-20261118-1 \
  --out "${HSAI_RECRUIT_BUILD_OUT:?}"
