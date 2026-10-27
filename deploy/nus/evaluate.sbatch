#!/usr/bin/env bash
#SBATCH --job-name=hsrl-eval
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/nus/env.sh
python -m taverngym.rl.doctor --config "${HSRL_CONFIG:-configs/ppo_nus.json}"
python -m taverngym.rl.train evaluate \
  --checkpoint "${HSRL_CHECKPOINT:-runs/nus-ppo-v2/best.pt}" \
  --device cuda --games 64 --split test \
  --out "${HSRL_EVAL_OUT:-runs/nus-ppo-v2/heldout.json}"
