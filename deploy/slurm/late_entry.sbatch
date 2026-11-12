#!/usr/bin/env bash
#SBATCH --job-name=hsai-late-entry
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:2
#SBATCH --exclude=xgpi15,xgpi17
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --propagate=NONE
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# 中后期入场课程（docs/LATE_GAME_ENTRY.md）。
#
#   HSAI_LE_OUT       输出目录（必须不存在）
#   HSAI_LE_CONFIG    配方（默认 config/league-entry.json）
#   HSAI_LE_BANK      镜像/对手记录用的已验证库（默认 league-eot-v1.1-20261109）
#   HSAI_LE_SOURCE    起点模型目录（默认 league-final-20261111-step014858）
#   HSAI_LE_SEGMENTS  段数（默认 2）
#   HSAI_LE_MINUTES   每段分钟数（默认 30）
#
# 注意：段与段之间会 --resume，续训守卫要求训练代码 SHA 不变，
# 所以这个作业运行期间**不要提交 hsbg/ 或 workflows/ 的改动**（只改 docs/config/tools）。
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export HSRL_TORCH_THREADS=2 PYTHONUTF8=1 PYTHONUNBUFFERED=1

out="${HSAI_LE_OUT:?必须指定 HSAI_LE_OUT（新目录）}"
cfg="${HSAI_LE_CONFIG:-config/league-entry.json}"
lib="${HSAI_LE_BANK:-artifacts/libraries/league-eot-v1.1-20261109}"
src="${HSAI_LE_SOURCE:-artifacts/models/league-final-20261111-step014858}"
segments="${HSAI_LE_SEGMENTS:-2}"
minutes="${HSAI_LE_MINUTES:-30}"

if [[ -e "$out" ]]; then echo "output exists: $out" >&2; exit 2; fi
if [[ ! -f "$cfg" ]]; then echo "missing config: $cfg" >&2; exit 2; fi
if [[ ! -f "$lib/train/manifest.json" ]]; then echo "missing library: $lib/train" >&2; exit 2; fi
if [[ ! -f "$src/trainer.pt" ]]; then echo "missing trainer: $src" >&2; exit 2; fi

mkdir -p "$out"
export HSRL_POOL_EVIDENCE_DIR="$(realpath -m "$out")/pool-evidence"
{
  echo "start=$(date -Is)"
  echo "config=$cfg"
  echo "library=$lib"
  echo "source=$src"
  echo "segments=$segments minutes=$minutes"
  echo "job=${SLURM_JOB_ID:-local} node=$(hostname)"
  echo "gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  echo "git=$(git rev-parse HEAD)"
  echo "train_sha=$(python -c 'from hsbg.az.train.personal import training_source_digest as d;print(d())')"
} | tee "$out/run.env"

for i in $(seq 1 "$segments"); do
  echo "segment $i/$segments start $(date -Is)"
  if [[ "$i" == "1" ]]; then
    python -m hsbg.az.train.personal train \
      --config "$cfg" \
      --checkpoint "$src/model.pt" \
      --initial-trainer-state "$src/trainer.pt" \
      --library "$lib/train" --out "$out/train" --device cuda \
      --run-minutes "$minutes"
  else
    python -m hsbg.az.train.personal train \
      --config "$cfg" --resume \
      --library "$lib/train" --out "$out/train" --device cuda \
      --run-minutes "$minutes"
  fi
  echo "segment $i/$segments done $(date -Is)"
done
echo "done=$(date -Is)" | tee -a "$out/run.env"
