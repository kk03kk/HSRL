#!/usr/bin/env bash
#SBATCH --job-name=hsai-recruit-bank
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1
.venv/bin/python workflows/collect_recruit_states.py \
  --checkpoint artifacts/models/arena-v1.4-r1/model.pt \
  --out "${HSAI_STATEBANK_OUT:?set new output directory}" --games 96 --device cpu
