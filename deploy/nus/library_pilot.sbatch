#!/usr/bin/env bash
#SBATCH --job-name=hsrl-entity18m-pilot
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=02:45:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/nus/env.sh
python tools/package_nus.py --verify .
cfg="$PWD/runs/library-data-v2/train-config.json"
python -m taverngym.rl.doctor --config "$cfg" --device cuda
child=""
trap '[[ -z "$child" ]] || kill -USR1 "$child" 2>/dev/null || true' USR1
trap '[[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true' TERM INT
python tools/run_library_pilot.py --config "$cfg" \
  --out "$PWD/runs/entity-library-pilot-$SLURM_JOB_ID" --budget 50000 --bc-waves 16 &
child=$!
set +e
wait "$child"; code=$?
if kill -0 "$child" 2>/dev/null; then wait "$child"; code=$?; fi
set -e
exit "$code"
