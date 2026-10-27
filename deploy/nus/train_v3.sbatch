#!/usr/bin/env bash
#SBATCH --job-name=hsrl-ppo3
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-23:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/nus/env.sh
export HSRL_CONFIG=configs/ppo_nus_v3.json
export HSRL_RUN=runs/nus-ppo-v3
export HSRL_INIT_FROM="$PWD/runs/nus-ppo-v2/bc-anchor.pt"
exec bash deploy/nus/train.sbatch
