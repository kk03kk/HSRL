#!/usr/bin/env bash
#SBATCH --job-name=hsai-native-triple-analysis
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
exec .venv/bin/python tools/analyze_native_bridge.py \
  --plan config/research/native-triple-bridge-20261118.json \
  --root "${HSAI_BRIDGE_OUT:?}" --out "${HSAI_BRIDGE_ANALYSIS_OUT:?}"
