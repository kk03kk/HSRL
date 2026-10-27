#!/usr/bin/env bash
#SBATCH --job-name=hsrl-recovery
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
python -m taverngym.rl.doctor --config configs/ppo_recovery.json --device cuda
base="$HOME/projects/hearthstone-rl/AutoTavern"
source_model="$base/runs/releases/paper-stage-v1/AutoTavern/runs/nus-paper-pilot-20261027-132048/compact_gru128-s701/latest.pt"
run="$PWD/runs/recovery-$SLURM_JOB_ID"
test -f "$source_model"
test ! -e "$run"
mkdir -p "$run"
child=""
trap '[[ -z "$child" ]] || kill -USR1 "$child" 2>/dev/null || true' USR1
trap '[[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true' TERM INT
run_child() {
  "$@" & child=$!
  set +e; wait "$child"; result=$?
  if kill -0 "$child" 2>/dev/null; then wait "$child"; result=$?; fi
  set -e
  return "$result"
}
if run_child python tools/bootstrap_model.py --config configs/ppo_recovery.json \
  --init-from "$source_model" --out "$run/bootstrap" --waves 16 --epochs 4 \
  --bc-lr 0.0003 --batch-chunks 8 --gate-games 8; then code=0; else code=$?; fi
if [[ "$code" == 3 ]]; then
  printf '\nBOOTSTRAP_GATE_NOT_READY: candidate preserved; PPO was not started.\n'
  printf 'Inspect %s/bootstrap/gate.json\n' "$run"
  exit 0
fi
if [[ "$code" != 0 ]]; then exit "$code"; fi
run_child python -m taverngym.rl.train train --config configs/ppo_recovery.json \
  --init-from "$run/bootstrap/bc-ready.pt" --out "$run/ppo"
run_child python -m taverngym.research exam --checkpoint "$run/bootstrap/bc-ready.pt" \
  --device cuda --games 16 --out "$run/independent-bc.json"
run_child python -m taverngym.research exam --checkpoint "$run/ppo/latest.pt" \
  --device cuda --games 16 --out "$run/independent-final.json"
printf '\nRecovery artifacts: %s\n' "$run"
