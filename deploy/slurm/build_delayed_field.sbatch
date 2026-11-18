#!/usr/bin/env bash
#SBATCH --job-name=hsai-delay-field
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -eu
cd "${SLURM_SUBMIT_DIR:?}"
exec .venv/bin/python workflows/build_delayed_field.py
