#!/usr/bin/env bash
#SBATCH --job-name=hsrl-ppo
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
config="${HSRL_CONFIG:-configs/ppo_nus.json}"
out="${HSRL_RUN:-runs/nus-ppo-v2}"
args=()
if [[ -f "$out/latest.pt" ]]; then
  args+=(--resume)
elif [[ -n "${HSRL_INIT_FROM:-}" ]]; then
  args+=(--init-from "$HSRL_INIT_FROM")
fi
python -m taverngym.rl.train train --config "$config" --out "$out" "${args[@]}" &
child=$!
trap 'kill -USR1 "$child" 2>/dev/null || true' USR1
trap 'kill -TERM "$child" 2>/dev/null || true' TERM INT
set +e
wait "$child"
status=$?
if kill -0 "$child" 2>/dev/null; then
  wait "$child"
  status=$?
fi
if [[ "$status" -eq 0 && -n "${HSRL_REQUIRE_ITERATIONS:-}" ]]; then
  python - "$out/latest.pt" "$HSRL_REQUIRE_ITERATIONS" <<'PY'
import sys, torch
s=torch.load(sys.argv[1],map_location='cpu',weights_only=True)
assert s['progress']['iteration'] == int(sys.argv[2]), s['progress']
print('Qualification passed:', s['progress']['iteration'], 'complete PPO waves.')
PY
  status=$?
fi
exit "$status"
