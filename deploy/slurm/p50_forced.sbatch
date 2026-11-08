#!/usr/bin/env bash
#SBATCH --job-name=hsai-p50-forced
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:2
#SBATCH --exclude=xgpi15,xgpi17
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --propagate=NONE
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export HSRL_TORCH_THREADS=2 PYTHONUTF8=1 PYTHONUNBUFFERED=1
# inputs() 必须绑定到本入口：否则 --source-7357 会被当成旧四阶段入口（1093247）
export HSAI_INPUTS_RUNNER=workflows.run_forced_tier_long
out="${HSAI_LONG_OUT:-artifacts/runs/p50-forced-tier-long-001}"
export HSRL_POOL_EVIDENCE_DIR="$(realpath -m "$out")/pool-evidence"
python -S tools/nus_submit.py --phase forced --check-only
hsrl_gpu_job python -u workflows/launch_nus_refinement.py --task p50-forced --out "$out"
