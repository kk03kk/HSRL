#!/usr/bin/env bash
#SBATCH --job-name=hsai-unfreeze-ab
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --propagate=NONE
#SBATCH --time=00:40:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HSRL_TORCH_THREADS=2 PYTHONUTF8=1 PYTHONUNBUFFERED=1
out="${HSRL_UNFREEZE_OUT:-artifacts/runs/audit-unfreeze-20261111-01}"
if [[ -e "$out" ]]; then echo "output exists: $out" >&2; exit 2; fi
python -u - "$out" <<'PY'
import json, sys, time
from pathlib import Path
sys.path.insert(0, '.')
from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
from hsbg.az.train.model import load_checkpoint
from hsbg.az.train.ppo import PPOEvaluator, PPOGreedyPolicy
out = Path(sys.argv[1]); out.mkdir(parents=True)
model, meta = load_checkpoint(
    'artifacts/models/league-final-20261111-step014858/model.pt', device='cuda')
rows = []
for seat in (0, 7):
    for flag in (False, True):
        ev = PPOEvaluator(model, device='cuda')
        policy = PPOGreedyPolicy(ev, max_actions_per_turn=80, move_guard=True,
                                 move_cap=5, move_repeat_limit=2,
                                 forbid_cancel=True, forbid_unfreeze=flag)
        t = time.time()
        res = evaluate_policy(policy, EvalConfig(
            games=128, seed=992000, policy_seat=seat, max_steps=50000,
            collect_action_hist=True))
        hist = res.get('action_hist') or {}
        row = {'seat': seat, 'forbid_unfreeze': flag,
               'mean_rank': res['mean_rank'], 'top4_rate': res['top4_rate'],
               'win_rate': res['win_rate'], 'rank_counts': res['rank_counts'],
               'decisions': res['decisions'], 'freeze_actions': int(hist.get('2', 0)),
               'illegal_actions': res['illegal_actions'],
               'truncated_games': res['truncated_games'],
               'seconds': time.time() - t}
        rows.append(row)
        print(json.dumps(row), flush=True)
report = {'schema': 'hsai.forbid_unfreeze_ab/v1', 'games_per_arm': 128,
          'seed': 992000, 'model': 'league-final-20261111-step014858',
          'note': 'eval-only flag; default training semantics unchanged',
          'arms': rows}
(out / 'unfreeze-ab.json').write_text(json.dumps(report, indent=2) + '\n')
PY
