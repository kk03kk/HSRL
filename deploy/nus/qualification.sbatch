#!/usr/bin/env bash
#SBATCH --job-name=hsrl-qualify
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:45:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/nus/env.sh
export HSRL_CONFIG=configs/ppo_qualification.json
export HSRL_RUN=runs/nus-qualification-v2
export HSRL_REQUIRE_ITERATIONS=40
# Keep the batch PID so the signal traps in train.sbatch receive Slurm's B:USR1.
exec bash deploy/nus/train.sbatch
