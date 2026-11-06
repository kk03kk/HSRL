#!/usr/bin/env bash
#SBATCH --job-name=hsrl-closeout-pack
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --propagate=NONE
#SBATCH --time=00:30:00
#SBATCH --output=personal/runs/slurm/%x-%j.out
#SBATCH --error=personal/runs/slurm/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
/usr/bin/python3 -S personal/artifacts/exports/closeout-transfer/pack_nus_closeout_1080772.py \
  --out "personal/artifacts/exports/closeout-transfer/bundle-${SLURM_JOB_ID}"
