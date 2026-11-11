#!/usr/bin/env bash
#SBATCH --job-name=hsai-behavior
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --propagate=NONE
#SBATCH --time=01:30:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HSRL_TORCH_THREADS=2 PYTHONUTF8=1 PYTHONUNBUFFERED=1

out="${HSRL_BEHAVIOR_OUT:-artifacts/runs/audit-behavior-20261111-02}"
python -u workflows/audit_model_behavior.py \
  --model V1.1=artifacts/models/v1.1/model.pt \
  --model league-latest-12311=artifacts/models/league-latest-20261110-step012311/model.pt \
  --model league-latest-13806=artifacts/models/league-latest-20261111-step013806/model.pt \
  --model league-final-14858=artifacts/models/league-final-20261111-step014858/model.pt \
  --games 48 --seats 0,1,7 --seed 950000 --examples 600 \
  --out "$out" --device cuda
nvidia-smi > "$out/nvidia-smi.txt" || true
