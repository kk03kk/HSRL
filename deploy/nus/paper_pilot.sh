#!/usr/bin/env bash
# User-run, one-time Slurm submission. Upload this file to HOME; no source replacement.
set -euo pipefail
cd "${HSRL_PROJECT:-$HOME/projects/hearthstone-rl/AutoTavern/runs/releases/paper-stage-v1/AutoTavern}"
python3 tools/package_nus.py --verify .
pilot_root="runs/nus-paper-pilot-$(date +%Y%m%d-%H%M%S)"
if [[ -e "$pilot_root" ]]; then printf 'Pilot directory already exists; refusing reuse\n'; exit 1; fi
python3 - "$pilot_root" <<'PILOT_CONFIG'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
configs=root/'configs';configs.mkdir()
plan=[]
for model in ('compact_gru128','compact_ff128'):
    for seed in (701,1701):
        config=json.loads(Path(f'configs/paper_stage/{model}.json').read_text())
        config.update(seed=seed,bc_waves=16,iterations=10000,decision_budget=200000,
                      eval_every=25,eval_games=16,eval_seeds=[900000000,900100000])
        name=f'{model}-s{seed}'
        path=configs/f'{name}.json';path.write_text(json.dumps(config,indent=2)+'\n')
        plan.append({'name':name,'config':str(path),'run':str(root/name),
                     'model':model,'seed':seed,'budget':200000})
(root/'plan.json').write_text(json.dumps({'format':'hsrl.equal-budget-pilot.v1','runs':plan,
    'purpose':'architecture learning comparison, not human-strength training',
    'compare':'BC baseline vs final fixed-budget checkpoint; do not rank from the 3-wave benchmark'},indent=2)+'\n')
(root/'plan.tsv').write_text(''.join(f"{r['name']}\t{r['config']}\t{r['run']}\n" for r in plan))
wrapper=r'''#!/usr/bin/env bash
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/nus/env.sh
python tools/package_nus.py --verify .
child=""
trap '[[ -z "$child" ]] || kill -USR1 "$child" 2>/dev/null || true' USR1
trap '[[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true' TERM INT
run_child() {
  "$@" & child=$!
  set +e
  wait "$child"; status=$?
  if kill -0 "$child" 2>/dev/null; then wait "$child"; status=$?; fi
  set -e
  return "$status"
}
args=()
if [[ -f "$HSRL_RUN/latest.pt" ]]; then args+=(--resume); fi
run_child python -m taverngym.rl.train train --config "$HSRL_CONFIG" --out "$HSRL_RUN" "${args[@]}"
python3 - "$HSRL_RUN/metrics.jsonl" <<'CHECK_COMPLETE'
import json,sys
from pathlib import Path
rows=[json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines()]
done=[x for x in rows if x.get('phase')=='completed']
assert done and done[-1]['ppo_decisions']>=200000, 'Pilot incomplete; checkpoint preserved, do not interpret as full budget'
print('Full sample budget completed:',done[-1]['ppo_decisions'])
CHECK_COMPLETE
run_child python -m taverngym.research exam --checkpoint "$HSRL_RUN/bc-anchor.pt" --device cuda --games 16 --out "$HSRL_RUN/independent-bc.json"
run_child python -m taverngym.research exam --checkpoint "$HSRL_RUN/latest.pt" --device cuda --games 16 --out "$HSRL_RUN/independent-final.json"
'''
(root/'checked-train.sbatch').write_text(wrapper)
print('Prepared four independent runs:', root)
PILOT_CONFIG
last_job=""
while IFS=$'\t' read -r name config run; do
  dependencies=()
  if [[ -n "$last_job" ]]; then dependencies=(--dependency="afterany:$last_job"); fi
  job=$(sbatch --parsable "${dependencies[@]}" \
    --partition=gpu --time=02:45:00 --job-name="hsrl-$name" \
    --account=allusers --qos=normal --gres=gpu:a100-40:1 \
    --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G --signal=B:USR1@180 \
    --output="$pilot_root/slurm-%x-%j.out" \
    --export="ALL,HSRL_CONFIG=$PWD/$config,HSRL_RUN=$PWD/$run,HSRL_INIT_FROM=,HSRL_REQUIRE_ITERATIONS=" \
    "$pilot_root/checked-train.sbatch")
  job="${job%%;*}"
  printf '%s\t%s\t%s\n' "$job" "$name" "$run" | tee -a "$pilot_root/jobs.tsv"
  last_job="$job"
done < "$pilot_root/plan.tsv"
printf '\nPilot root: %s\nAt most one GPU job runs at a time; each seed is independent.\n' "$pilot_root"
squeue -u "$(id -un)"
