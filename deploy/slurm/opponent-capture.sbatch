#!/usr/bin/env bash
#SBATCH --job-name=hsrl-bank-capture
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --propagate=NONE
#SBATCH --time=04:30:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HSRL_TORCH_THREADS=2
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTHONUTF8=1 PYTHONUNBUFFERED=1
out="${HSRL_RESEARCH_OUT:-artifacts/runs/opponent-research-001}"
mkdir -p "$out/job-$SLURM_JOB_ID"
python -c "from hsbg.az.train.personal import training_source_digest; print(training_source_digest())" > "$out/job-$SLURM_JOB_ID/revision.txt"
nvidia-smi > "$out/job-$SLURM_JOB_ID/nvidia-smi.txt"
python -u workflows/prepare_opponent_research.py --phase capture --minutes 240 --out "$out"
