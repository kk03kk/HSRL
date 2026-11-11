#!/usr/bin/env bash
#SBATCH --job-name=hsai-actor-scale
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:2
#SBATCH --exclude=xgpi15,xgpi17
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --propagate=NONE
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# Bounded actor-scaling A/B: 8 vs 16 worker processes, same final state,
# same library, p75band mirror sampling. Trainer on cuda:0, actors on cuda:1.
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export HSRL_TORCH_THREADS=2 PYTHONUTF8=1 PYTHONUNBUFFERED=1

out="${HSRL_SCALE_OUT:-artifacts/runs/audit-scaling-20261111-01}"
if [[ -e "$out" ]]; then echo "output exists: $out" >&2; exit 2; fi
mkdir -p "$out"
nvidia-smi -L > "$out/nvidia-smi.txt"
python - "$out" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
base = json.loads(Path('config/league-36h.json').read_text())
for p in (8, 16):
    cfg = dict(base)
    cfg.update({'games': 512, 'iterations': 4, 'minutes': 10, 'processes': p,
                'pipeline': 1, 'amp': 'bf16', 'leaf-amp': 'bf16',
                'anchor-amp': 1, 'ipc-mode': 'shm', 'gpus': '1',
                'gpu-index-space': 'visible', 'mirror-sample': 'p75band',
                'mirror-strict': 1, 'reset-anchor': 1,
                'snapshot-every': 5, 'keep-snapshots': 3})
    (out / f'config-p{p}.json').write_text(json.dumps(cfg, indent=2) + '\n')
PY
for p in 8 16; do
  python -u -m hsbg.az.train.personal train \
    --config "$out/config-p$p.json" \
    --checkpoint artifacts/models/league-final-20261111-step014858/model.pt \
    --initial-trainer-state artifacts/models/league-final-20261111-step014858/trainer.pt \
    --library artifacts/libraries/league-eot-v1.1-20261109/train \
    --out "$out/p$p" --device cuda:0 --run-iterations 4 --run-minutes 10
done
python - "$out" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for p in (8, 16):
    hist = [json.loads(x) for x in (out / f'p{p}' / 'history.jsonl').read_text().splitlines()]
    hist = [r for r in hist if 'iter' in r][-3:]
    def mean(key):
        vals = [r['collect'].get(key) for r in hist if r['collect'].get(key) is not None]
        return sum(vals) / len(vals) if vals else None
    iters = []
    for r in hist:
        tl = r.get('profile', {}).get('timeline', {})
        if 't_iter_start' in tl and 't_history_done' in tl:
            iters.append(tl['t_history_done'] - tl['t_iter_start'])
    rows.append({'processes': p, 'iterations': len(hist),
                 'decisions_per_s': mean('decisions_per_s'),
                 'wall_s': mean('wall_s'),
                 'samples_per_s': sum(r['ppo'].get('samples_per_s', 0) for r in hist) / len(hist),
                 'iteration_s': sum(iters) / len(iters) if iters else None,
                 'decisions': mean('decisions')})
report = {'schema': 'hsai.actor_scaling_audit/v1', 'arms': rows,
          'note': 'bounded 4-iteration A/B; p75band mirror; final trainer; not a long run'}
(out / 'scaling.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
