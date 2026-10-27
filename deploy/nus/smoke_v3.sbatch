#!/usr/bin/env bash
#SBATCH --job-name=hsrl-smoke3
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/nus/env.sh
export HSRL_CONFIG=configs/ppo_smoke.json
export HSRL_RUN=runs/nus-smoke-v3
exec bash deploy/nus/train.sbatch
