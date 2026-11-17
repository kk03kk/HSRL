#!/usr/bin/env bash
#SBATCH --job-name=hsai-mechanism-probe
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:2
#SBATCH --exclude=xgpi15,xgpi17,xgpi11
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --propagate=NONE
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
root="${HSAI_RESEARCH_ROOT:?set new research output root}"
test ! -e "$root"
mkdir -p "$root"
.venv/bin/python workflows/upgrade_mechanism_checkpoint.py \
  --checkpoint artifacts/models/arena-v1.4-r1/model.pt --out "$root/initial"
.venv/bin/python -m pytest tests/test_mechanism_observation.py \
  tests/test_az_ppo.py tests/test_checkpoint_roundtrip.py -q \
  --junitxml="$root/tests.xml"
export HSAI_ARENA_CKPT="$root/initial/model.pt"
export HSAI_ARENA_OUT="$root/smoke"
export HSAI_ARENA_MINUTES=15 HSAI_ARENA_ITERATIONS=2 HSAI_ARENA_GAMES=64
export HSAI_ARENA_PROCS=4 HSAI_ARENA_SEED=1181701
bash deploy/slurm/stage_arena.sbatch
echo "MECHANISM_PROBE_COMPLETE $(date -Is)"
