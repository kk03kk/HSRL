#!/bin/bash
set -euo pipefail
source deploy/slurm/common.sh
hsrl_require_compute
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec .venv/bin/python workflows/peer_training_rounds.py "$@"
