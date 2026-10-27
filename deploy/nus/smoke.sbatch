#!/usr/bin/env bash
#SBATCH --job-name=hsrl-smoke
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
python -m taverngym.rl.doctor --config configs/ppo_smoke.json
args=()
[[ ! -f runs/nus-smoke-v2/latest.pt ]] || args+=(--resume)
python -m taverngym.rl.train train --config configs/ppo_smoke.json --out runs/nus-smoke-v2 "${args[@]}"
