#!/usr/bin/env bash
#SBATCH --job-name=hsrl-setup
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash deploy/nus/setup.sh
