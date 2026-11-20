#!/bin/bash
#SBATCH -A allusers
#SBATCH -q normal
#SBATCH -p gpu,gpu-long
#SBATCH --gres=gpu:h100-47:2
#SBATCH --exclude=xgpi11,xgpi15,xgpi17
#SBATCH --propagate=NONE
#SBATCH -c 32
#SBATCH --mem=96G
#SBATCH -t 01:50:00
#SBATCH -J peer_budget_ppo
#SBATCH --output=artifacts/runs/slurm/%x-%j.log
set -euo pipefail
source deploy/slurm/common.sh
hsrl_require_compute
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
out="${1:?provide a fresh output directory}"
recipe=config/research/peer-action-budget-ppo.json
.venv/bin/python workflows/peer_open_ppo.py train --out "$out" --recipe "$recipe"
.venv/bin/python workflows/peer_open_ppo.py resume --out "$out" --recipe "$recipe"
