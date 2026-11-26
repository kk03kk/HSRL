#!/usr/bin/env bash
set -euo pipefail

run="$(realpath "${1:?PPO output directory required}")"
bc="$(realpath "${2:?admitted BC run directory required}")"
mode="${3:?start or resume required}"
through="${4:?final PPO iteration for this segment required}"
cd "$(git rev-parse --show-toplevel)"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
commit="$(git rev-parse HEAD)"
test "$(cat "$run/source_commit")" = "$commit"
flags=()
case "$mode" in
    start) test ! -e "$run/train/latest.pt" ;;
    resume) test -f "$run/train/latest.pt"; flags+=(--resume) ;;
    *) printf 'Unknown training mode: %s\n' "$mode" >&2; exit 2 ;;
esac
if [[ -f "$run/policy_batch_size" ]]; then
    flags+=(--policy-batch-size "$(cat "$run/policy_batch_size")")
fi
python="$bc/env/venv/bin/python"
export BC_EXPECTED_GPU='NVIDIA H100 NVL'
"$python" -c 'import torch; assert torch.cuda.device_count() == 1; device = torch.cuda.get_device_properties(0); assert device.name == "NVIDIA H100 NVL" and device.total_memory >= 90 * 2**30; print(f"{device.name}, {device.total_memory / 2**30:.1f} GiB")'
nvidia-smi -L
nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu,utilization.memory,power.draw \
    --format=csv -l 15 >"$run/logs/gpu-$SLURM_JOB_ID.csv" 2>&1 &
watch_pid=$!
trap 'kill "$watch_pid" 2>/dev/null || true' EXIT
"$python" -u tools/run_frozen_python.py python/taverngym/ppo_train.py train \
    --init "$bc/admission-1/ppo-init.pt" --admission "$bc/admission-1/ppo-admission.json" \
    --code-commit "$commit" --seed 2601126 --first-seed 12000000 \
    --games 1024 --concurrent 1024 --workers "$SLURM_CPUS_PER_TASK" --actors 1 \
    --minibatch 2048 --microbatch 1024 --precision bf16 --resident-rollout \
    --lr-scale 1.5 --bc-kl-scale 1 --kl-check-every 4 \
    --iterations 1024 --stop-after "$through" --time-budget-seconds 10440 \
    --snapshot-every 20 --out "$run/train" "${flags[@]}"
