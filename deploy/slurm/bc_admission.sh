#!/usr/bin/env bash
set -euo pipefail

mode="${1:?stage required}"
run="$(realpath "${2:?frozen BC run required}")"
shift 2
root="$(git rev-parse --show-toplevel)"
cd "$root"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python="$run/env/venv/bin/python"
workers="${SLURM_CPUS_PER_TASK:-10}"
arena="${BC_ARENA_BATCH:-384}"
data="$run/data"

case "$mode" in
    screen)
        out="${1:?development output directory required}"
        shift
        "$python" -u python/taverngym/bc_screen.py --out "$out" --gate \
            --panel "$data/screen-original" "$data/screen-mixed" --workers "$workers" \
            --arena-batch "$arena" --checkpoints "$@"
        ;;
    dagger)
        checkpoint="${1:?checkpoint for corrective sampling required}"
        case "${SLURM_ARRAY_TASK_ID:?requires a two-field job array}" in
            0) field=original; first=9600000 ;;
            1) field=mixed; first=9610000 ;;
            *) exit 2 ;;
        esac
        "$python" -u python/taverngym/bc_dagger.py --checkpoint "$checkpoint" \
            --field "$field" --first "$first" --games 1536 --beta 0.10 \
            --batch "$arena" --workers "$workers" --out "$data/dagger-$field"
        ;;
    refine)
        checkpoint="${1:?initial checkpoint required}"
        shift
        "$python" -u python/taverngym/bc_train.py train --balance-fields --save-epochs \
            --dev-games 256 --arena-batch "$arena" --late-weight 0.5 --correction-weight 0.75 \
            --lr 6e-5 --epochs 2 --save-every 36 --workers "$workers" --initialize "$checkpoint" \
            --train "$data/train-original" "$data/train-mixed" "$data/dagger-original" "$data/dagger-mixed" \
            --dev "$data/dev-original" "$data/dev-mixed" --out "$run/refine" "$@"
        ;;
    holdout)
        out="${1:?frozen selection output directory required}"
        case "${SLURM_ARRAY_TASK_ID:?requires a two-field job array}" in
            0) field=original ;;
            1) field=mixed ;;
            *) exit 2 ;;
        esac
        "$python" -u python/taverngym/bc_train.py evaluate --checkpoint "$out/selected.pt" \
            --test "$data/${2:-test}-$field" --out "$out/field-$field" \
            --arena-batch "$arena" --workers "$workers"
        ;;
    handoff)
        out="${1:?frozen selection output directory required}"
        "$python" -u tools/bc_handoff.py "$out" --workers "$workers" --arena-batch "$arena"
        python3 tools/bc_admission.py admit "$out"
        ;;
    *)
        printf 'Unknown BC admission stage: %s\n' "$mode" >&2
        exit 2
        ;;
esac
