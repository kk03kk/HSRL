#!/usr/bin/env bash
set -euo pipefail
run=$1; assets=$2; bc=$3; parent=$4
source=$run/source
entry=$source/deploy/slurm/ppo_night.sh
test -s "$run/preflight-receipt.json"
if grep -q '^stage1=' "$run/jobs.txt"; then
    echo "A training chain is already registered in $run/jobs.txt" >&2
    exit 1
fi
test -s "$run/gpu-preflight.json"
jq -e '.confirmed_resume and .confirmed_recipe_switch and .frames > 0' "$run/gpu-preflight.json" >/dev/null
smoke=$(sed -n 's/^smoke=//p' "$run/jobs.txt")
fresh=$(sed -n 's/^fresh=//p' "$run/jobs.txt")
train_opts=(-p gpu --gres=gpu:h100-96:1 --exclude=xgpi4,xgpi5 -c 10 --mem=38G --time=02:59:00 --chdir="$source")
control_opts=(-p gpu --gres=gpu:h100-47:1 --exclude=xgpi11,xgpi15 -c 10 --mem=26G --chdir="$source")
run_job() {
    local label=$1
    shift
    local id
    id=$(sbatch --parsable "$@")
    printf '%s=%s\n' "$label" "$id" >> "$run/jobs.txt"
    printf '%s\n' "$id"
}
first=$(run_job stage1 "${train_opts[@]}" -J hsrl-night-1 -d "afterok:$smoke" --kill-on-invalid-dep=yes \
    -o "$run/logs/stage1-%j.log" "$entry" "$run" "$assets" "$bc" "$parent" segment 1 10680)
second=$(run_job stage2 "${train_opts[@]}" -J hsrl-night-2 -d "afterany:$first" \
    -o "$run/logs/stage2-%j.log" "$entry" "$run" "$assets" "$bc" "$parent" segment 2 10680)
third=$(run_job stage3 "${train_opts[@]}" -J hsrl-night-3 -d "afterany:$second" \
    -o "$run/logs/stage3-%j.log" "$entry" "$run" "$assets" "$bc" "$parent" segment 3 10680)
control=$(run_job control "${control_opts[@]}" --time=01:40:00 -a 0-1 -J hsrl-night-replica \
    -d "after:$first" --kill-on-invalid-dep=yes -o "$run/logs/replica-%A-%a.log" "$entry" "$run" "$assets" "$bc" "$parent" control)
seal=$(run_job seal -p normal -c 2 --mem=5G --time=00:15:00 --chdir="$source" -J hsrl-night-seal \
    -d "afterany:$third:$control:$fresh" -o "$run/logs/seal-%j.log" "$entry" "$run" "$assets" "$bc" "$parent" seal)
eval=$(run_job evaluation "${control_opts[@]}" --time=00:37:00 -a 0-4%2 -J hsrl-night-holdout \
    -d "afterok:$seal" --kill-on-invalid-dep=yes -o "$run/logs/holdout-%A-%a.log" "$entry" "$run" "$assets" "$bc" "$parent" evaluate)
report=$(run_job report -p normal -c 3 --mem=8G --time=00:30:00 --chdir="$source" -J hsrl-night-report \
    -d "afterany:$eval" -o "$run/logs/report-%j.log" "$entry" "$run" "$assets" "$bc" "$parent" report)
cat "$run/jobs.txt"
