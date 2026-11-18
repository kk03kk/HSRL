#!/usr/bin/env bash
#SBATCH --job-name=hsai-native-triple-bridge
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude=xgpi11,xgpi15,xgpi17
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
extra=()
if [ "${HSAI_BRIDGE_PROBE:-0}" = 1 ]; then extra+=(--probe); fi
exec .venv/bin/python workflows/probe_economy_suffix.py \
  --plan config/research/native-triple-bridge-20261118.json \
  --out "${HSAI_BRIDGE_OUT:?}" --device cuda:0 --workers 4 "${extra[@]}"
