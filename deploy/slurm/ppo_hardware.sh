#!/usr/bin/env bash
set -euo pipefail

stage="${1:?stage required}"
run="$(realpath "${2:?hardware output directory required}")"
bc="$(realpath "${3:?BC run directory required}")"
cd "$(git rev-parse --show-toplevel)"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python="$bc/env/venv/bin/python"
commit="$(git rev-parse HEAD)"
src=(tools/run_frozen_python.py python/taverngym/ppo_train.py)
reference=(--init "$bc/admission-1/ppo-init.pt" --admission "$bc/admission-1/ppo-admission.json")

trainer() {
    "$python" -u "${src[@]}" train "${reference[@]}" --code-commit "$commit" \
        --seed 2601126 --minibatch 2048 --microbatch 1024 --lr-scale 1.5 --kl-check-every 4 "$@"
}
compare() { "$python" -u "${src[@]}" compare "$@"; }
watch_gpu() {
    local path="$run/logs/gpu-$SLURM_JOB_ID.csv"
    nvidia-smi -L
    nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu,utilization.memory,power.draw \
        --format=csv -l 2 >"$path" 2>&1 &
    watch_pid=$!
    trap 'kill "$watch_pid" 2>/dev/null || true' EXIT
}
case "$stage" in
    full)
        watch_gpu
        common=(--workers 10 --expected-gpu 'NVIDIA H100 NVL')
        small=(--first-seed 11000000 --games 128 --concurrent 128 --iterations 2)
        trainer "${common[@]}" "${small[@]}" --out "$run/preflight-host"
        trainer "${common[@]}" "${small[@]}" --resident-rollout --out "$run/preflight-resident" --stop-after 1
        trainer "${common[@]}" "${small[@]}" --resident-rollout --out "$run/preflight-resident" --resume
        compare --compare-residency --out "$run/preflight-exact.json" \
            "$run/preflight-host/latest.pt" "$run/preflight-resident/latest.pt"
        target=(--first-seed 11010000 --games 1024 --concurrent 1024 --iterations 6)
        trainer "${common[@]}" "${target[@]}" --out "$run/full-host"
        trainer "${common[@]}" "${target[@]}" --resident-rollout --out "$run/full-resident"
        compare --compare-residency --out "$run/target-exact.json" \
            "$run/full-host/latest.pt" "$run/full-resident/latest.pt"
        ;;
    mig)
        # Slurm's numbered MIG mask can enumerate only one instance. Use the
        # exact two UUIDs visible to this job before importing CUDA.
        mapfile -t uuids < <(nvidia-smi -L | grep -oE 'MIG-[[:xdigit:]-]+')
        if [[ ${#uuids[@]} -ne 2 ]]; then
            printf 'Expected exactly two allocated MIG UUIDs, got %s\n' "${#uuids[@]}" >&2
            exit 1
        fi
        export CUDA_VISIBLE_DEVICES="${uuids[0]},${uuids[1]}"
        watch_gpu
        common=(--workers 10 --expected-gpu 'NVIDIA H100 NVL MIG 3g.47gb' --resident-rollout)
        small=(--first-seed 11000000 --games 128 --concurrent 128 --iterations 2 --actors 2)
        trainer "${common[@]}" "${small[@]}" --out "$run/dual-preflight"
        trainer "${common[@]}" "${small[@]}" --out "$run/dual-resumed" --stop-after 1
        trainer "${common[@]}" "${small[@]}" --out "$run/dual-resumed" --resume
        compare --out "$run/dual-preflight-exact.json" "$run/dual-preflight/latest.pt" "$run/dual-resumed/latest.pt"
        target=(--first-seed 11010000 --games 1024 --iterations 6)
        trainer "${common[@]}" "${target[@]}" --concurrent 512 --out "$run/mig-single"
        trainer "${common[@]}" "${target[@]}" --concurrent 1024 --actors 2 --out "$run/mig-dual"
        ;;
    panel)
        case "${SLURM_ARRAY_TASK_ID:?panel task required}" in
            0) field=original; first=13500000 ;;
            1) field=mixed; first=13510000 ;;
            *) exit 2 ;;
        esac
        export CARGO_TARGET_DIR="$bc/native/target"
        data="$run/panel-$field"
        "$CARGO_TARGET_DIR/release/examples/bc_collect" "$first" 2048 "$SLURM_CPUS_PER_TASK" "$data" "$field" metrics
        "$bc/native/venv/bin/python" tools/bc_prepare_data.py "$data" --source-commit "$(cat "$bc/native/source_commit")"
        ;;
    evaluate)
        case "${SLURM_ARRAY_TASK_ID:?variant task required}" in
            0) arm=full-resident ;;
            1) arm=mig-single ;;
            2) arm=mig-dual ;;
            *) exit 2 ;;
        esac
        export BC_EXPECTED_GPU='NVIDIA H100 NVL MIG 3g.47gb'
        "$python" -u "${src[@]}" evaluate "${reference[@]}" --workers 10 --arena-batch 384 \
            --panel development --test "$run/panel-original" "$run/panel-mixed" \
            --checkpoint "$run/$arm/latest.pt" --out "$run/eval-$arm"
        ;;
    *) printf 'Unknown hardware stage: %s\n' "$stage" >&2; exit 2 ;;
esac
