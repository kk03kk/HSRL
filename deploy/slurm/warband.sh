#!/usr/bin/env bash
set -euo pipefail
root=$1; bc=$2
shift 2
unset PYTHONHOME VIRTUAL_ENV
export PYTHONPATH="$root/wheel-coverage"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
python="$bc/env/venv/bin/python"
case "$1" in
collect)
    shift
    splits=(train dev test)
    exec "$python" -u tools/run_frozen_python.py tools/warband_research.py "$root" collect "${splits[$SLURM_ARRAY_TASK_ID]}" "$@"
    ;;
research)
    shift
    exec "$python" -u tools/run_frozen_python.py tools/warband_research.py "$root" "$@"
    ;;
train)
    shift
    exec "$python" -u tools/run_frozen_python.py python/taverngym/ppo_train.py train "$@"
    ;;
*) exit 1;;
esac
