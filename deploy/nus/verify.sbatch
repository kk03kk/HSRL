#!/usr/bin/env bash
#SBATCH --job-name=hsrl-verify
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
# Recovery gate: reuse existing venv/native extension; no installation or Rust rebuild.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/nus/env.sh
test -x .venv/bin/python
mkdir -p runs/setup
python tools/package_nus.py --verify .
python -m pytest tests -q
python -m taverngym.rl.doctor --config configs/ppo_smoke.json --device cpu --out runs/setup/doctor-cpu.json
python -m pip freeze > runs/setup/pip-freeze.txt
rustc -Vv > runs/setup/rustc.txt
printf '\nVerification complete; reuse the successful native build.\n'
