#!/usr/bin/env bash
set -euo pipefail
root=$1; assets=$2; bc=$3; parent=$4; mode=$5
shift 5
unset PYTHONHOME VIRTUAL_ENV
export PYTHONPATH="$assets/wheel-coverage"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "$mode" == control || "$mode" == evaluate ]]; then
    set -- "$SLURM_ARRAY_TASK_ID"
fi
exec "$bc/env/venv/bin/python" -u tools/run_frozen_python.py tools/ppo_night.py "$root" "$assets" "$bc" "$parent" "$mode" "$@"
