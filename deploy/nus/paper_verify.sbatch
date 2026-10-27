#!/usr/bin/env bash
#SBATCH --job-name=hsrl-paper-check
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/nus/env.sh
python tools/package_nus.py --verify .
python -m pytest tests -q
# Native sources are unchanged. Never silently rebuild or accept another binary.
python -m taverngym.rl.doctor --config configs/ppo_smoke.json --device cpu
