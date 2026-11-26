#!/usr/bin/env bash
set -euo pipefail

mode="${1:?mode required}"
run="$(realpath "${2:?run directory required}")"
shift 2
root="$(git rev-parse --show-toplevel)"
cd "$root"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS="${SLURM_CPUS_PER_TASK:-1}"
export CARGO_TARGET_DIR="$run/native/target"
native="$run/native/venv/bin/python"
python="$run/env/venv/bin/python"

case "$mode" in
    native)
        mkdir -p "$run/native/wheels"
        python3 -m venv "$run/native/venv"
        "$native" -m pip install --disable-pip-version-check -r deploy/slurm/bootstrap-requirements.txt
        cargo build --locked --release -p taverngym --example bc_collect
        "$run/native/venv/bin/maturin" build --locked --release -i "$native" --out "$run/native/wheels"
        "$0" verify-native "$run"
        ;;
    verify-native)
        "$native" -m pip install --no-deps "$run"/native/wheels/*.whl
        "$native" -c 'from pathlib import Path; import taverngym; from taverngym._core import BcArena; assert taverngym.build_is_current(Path.cwd()) is True; assert len(BcArena([9000000],True,1).observe()[0])==4; print(taverngym.build_info())'
        test -x "$CARGO_TARGET_DIR/release/examples/bc_collect"
        git rev-parse HEAD > "$run/native/source_commit"
        (cd "$run/native" && sha256sum wheels/*.whl > wheels.sha256)
        touch "$run/native/READY"
        ;;
    env)
        mkdir -p "$run/env"
        python3 -m venv "$run/env/venv"
        "$python" -m pip install --disable-pip-version-check --extra-index-url https://download.pytorch.org/whl/cu128 -r deploy/slurm/bc_requirements.nus.txt
        "$python" -m pip check
        "$python" -m pip freeze > "$run/env/requirements.freeze"
        touch "$run/env/READY"
        ;;
    install-model-wheel)
        test -f "$run/env/READY"
        test -f "$run/native/READY"
        (cd "$run/native" && sha256sum -c wheels.sha256)
        "$python" -m pip install --no-deps "$run"/native/wheels/*.whl
        "$python" -c 'from pathlib import Path; import taverngym; assert taverngym.build_is_current(Path.cwd()) is True'
        touch "$run/env/ENGINE_READY"
        ;;
    collect)
        test -f "$run/native/READY"
        split="${1:?split name required}"
        read -r name first count field capture < <(awk -v wanted="$split" '$1==wanted {print; exit}' deploy/slurm/bc_splits.tsv)
        test "$name" = "$split"
        args=()
        if [[ "$capture" == metrics ]]; then args=(metrics); fi
        mkdir -p "$run/data"
        "$CARGO_TARGET_DIR/release/examples/bc_collect" "$first" "$count" "$CARGO_BUILD_JOBS" "$run/data/$split" "$field" "${args[@]}"
        "$native" tools/bc_prepare_data.py "$run/data/$split" --source-commit "$(cat "$run/native/source_commit")"
        ;;
    preflight)
        test -f "$run/env/ENGINE_READY"
        device="${1:-cuda}"
        "$python" tools/bc_preflight.py "$run/data/pilot-original" "$run/data/pilot-mixed" --device "$device" --out "$run/preflight-$device.json"
        ;;
    train|resume)
        test -f "$run/READY_FOR_BC"
        test -f "$run/env/ENGINE_READY"
        flags=(--balance-fields --save-epochs --dev-games 256 --late-weight 0.35)
        if [[ "$mode" == resume ]]; then flags+=(--resume); fi
        "$python" -u python/taverngym/bc_train.py train "${flags[@]}" \
            --train "$run/data/train-original" "$run/data/train-mixed" \
            --dev "$run/data/dev-original" "$run/data/dev-mixed" \
            --out "$run/train" --workers "$CARGO_BUILD_JOBS" "$@"
        ;;
    evaluate)
        test -f "$run/READY_FOR_BC"
        "$python" -u python/taverngym/bc_train.py evaluate \
            --checkpoint "${1:?checkpoint required}" --out "${2:?output directory required}" \
            --test "$run/data/test-original" "$run/data/test-mixed" --workers "$CARGO_BUILD_JOBS"
        ;;
    *)
        printf 'Unknown BC mode: %s\n' "$mode" >&2
        exit 2
        ;;
esac
