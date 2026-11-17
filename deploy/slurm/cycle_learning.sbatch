#!/usr/bin/env bash
#SBATCH --job-name=hsai-cycle-learn
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude=xgpi11,xgpi15,xgpi17
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
out="${HSAI_CYCLE_OUT:?}"
test ! -e "$out"
mkdir -p "$out"
.venv/bin/python workflows/upgrade_activation_checkpoint.py \
  --checkpoint artifacts/models/arena-v1.4-r1/model.pt --out "$out/initial"
exec .venv/bin/python workflows/learn_cycle_process.py \
  --checkpoint "$out/initial/model.pt" --dataset "${HSAI_CYCLE_DATASET:?}" \
  --out "$out/learn" --mode "${HSAI_CYCLE_MODE:?}" --lr "${HSAI_CYCLE_LR:?}" \
  --updates 400 --anchor-kl 10 --per-kind-batch 16 --device cuda:0
