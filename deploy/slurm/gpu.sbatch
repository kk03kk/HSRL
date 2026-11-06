#!/usr/bin/env bash
#SBATCH --job-name=hsrl-smoke
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --propagate=NONE
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HSRL_TORCH_THREADS=2
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
stage="${HSRL_STAGE:-preflight}"
model="${HSRL_MODEL:-artifacts/models/s14_best_main_model.pt}"
library="${HSRL_LIBRARY:-artifacts/libraries/bot-v1}"
out="${HSRL_OUT:-artifacts/runs/$stage-$SLURM_JOB_ID}"
mkdir -p "$out"
uname -a
nvidia-smi
python -m hsbg.az.train.personal preflight --checkpoint "$model" --device cuda \
  --out "$out/gpu-preflight.json"
if [[ "$stage" == preflight ]]; then exit 0; fi
if [[ "$stage" != smoke && "$stage" != pilot && "$stage" != resume ]]; then
  echo 'HSRL_STAGE 只接受 preflight/smoke/pilot/resume' >&2; exit 2
fi
if [[ ! -f "$library/train/manifest.json" ]]; then
  echo '请先生成并检查战团库，使用 workflows/build_3702_bot_bank.py' >&2; exit 2
fi
default_config=config/smoke.json
if [[ "$stage" == pilot ]]; then default_config=config/pilot.json; fi
config="${HSRL_CONFIG:-$default_config}"
resume=()
if [[ "$stage" == resume ]]; then resume=(--resume); fi
# 时间上限中断时恢复最近完整发布的快照；不承诺保存正在采集的一局。
python -m hsbg.az.train.personal train --config "$config" --checkpoint "$model" \
  --library "$library/train" --device cuda --out "$out/train" "${resume[@]}"
python -m hsbg.az.train.personal eval --checkpoint "$out/train/model.pt" \
  --library "$library/dev" --games 2 --device cuda --out "$out/dev-smoke.json"
