#!/usr/bin/env bash
#SBATCH --job-name=hsai-recruit-resume
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
exec .venv/bin/python workflows/probe_recruit_resume.py \
  --out "${HSAI_RECRUIT_RESUME_OUT:?}" \
  --bank artifacts/runs/nus-recruit-curriculum-20261118-1/endowed \
  --checkpoint artifacts/runs/nus-process-base-20261118-2/model.pt
