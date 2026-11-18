#!/usr/bin/env bash
#SBATCH --job-name=hsai-turn-close
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -eu
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HSRL_FAILURE_DIR="${HSAI_CLOSE_OUT:?}/failed"
exec .venv/bin/python workflows/probe_turn_close.py
