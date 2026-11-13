#!/usr/bin/env bash
#SBATCH --job-name=hsai-bc
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude=xgpi15,xgpi17
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --propagate=NONE
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# 监督学习阶段（BC / 行为克隆）：用真人动作样本把当前策略往"人类怎么打"上推一段。
#
# 为什么要有这个作业：第三方成功案例（PrentendYan/hearthstone-bg-ppo）的配方是
# **BC 初始化 → PPO**，而我们所有从联赛 fork 的臂都是**纯 PPO**。这是流程上最大的
# 一处不对称，所以值得用我们自己的样本池做一次受控实验
# （证据与结论见 docs/analysis/THIRD_PARTY_PPO_20261113.md）。
#
# 风险控制（重要）：bc_pretrain 的 `--anchor-beta` 默认是 **0**（不约束）。
# 直接对 champion 跑无约束 BC 可能把它推坏，所以这里默认给一个 KL 锚。
#
#   HSAI_BC_INIT    起点 checkpoint（默认 champion）
#   HSAI_BC_OUT     输出目录（必须不存在；目录内产出 model.pt + report）
#   HSAI_BC_MMR     样本 MMR 下限（默认 6000：≥8k 只有 60 局，太薄）
#   HSAI_BC_EPOCHS  轮数（默认 4）
#   HSAI_BC_LR      学习率（默认 3e-5，比默认 1e-4 更保守）
#   HSAI_BC_ANCHOR  KL 锚权重（默认 0.05；0 = 关闭，不建议对 champion 关）
#   HSAI_BC_LIMIT   样本上限（0 = 全部）
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HSRL_TORCH_THREADS=2

init="${HSAI_BC_INIT:-artifacts/models/league-final-20261111-step014858/model.pt}"
out="${HSAI_BC_OUT:?必须指定 HSAI_BC_OUT（新目录）}"
mmr="${HSAI_BC_MMR:-6000}"
epochs="${HSAI_BC_EPOCHS:-4}"
lr="${HSAI_BC_LR:-3e-5}"
anchor="${HSAI_BC_ANCHOR:-0.05}"
limit="${HSAI_BC_LIMIT:-0}"

if [[ -e "$out" ]]; then echo "output exists: $out" >&2; exit 2; fi
if [[ ! -f "$init" ]]; then echo "missing init: $init" >&2; exit 2; fi
mkdir -p "$out"
{
  echo "start=$(date -Is)"
  echo "init=$init"
  echo "mmr_min=$mmr epochs=$epochs lr=$lr anchor_beta=$anchor limit=$limit"
  echo "job=${SLURM_JOB_ID:-local} node=$(hostname)"
  echo "gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  echo "git=$(git rev-parse HEAD)"
} | tee "$out/run.env"

PYTHONPATH=engine/python:. python workflows/bc_pretrain.py \
  --samples data/firestone_bc/bc_samples.jsonl.gz \
  --mmr-min "$mmr" \
  --init "$init" \
  --device cuda \
  --lr "$lr" \
  --batch 256 \
  --epochs "$epochs" \
  --limit "$limit" \
  --anchor-beta "$anchor" \
  --anchor-scope all \
  --out "$out" \
  --report "$out/report.json" 2>&1 | tee "$out/bc.log"

echo "done=$(date -Is)" | tee -a "$out/run.env"
