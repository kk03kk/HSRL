#!/usr/bin/env bash
# 2 模型 + 6 bot 的 128 局配对评测（纯评测：不训练、不写 checkpoint、不改引擎）。
# 协议与字段见 config/eval-duel-7357-vs-10004.json + docs/EVAL_DUEL.md。
#SBATCH --job-name=hsai-duel-eval
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude=xgpi15,xgpi17
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --propagate=NONE
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
if [[ ! -x .venv/bin/python ]]; then
  echo '缺少 .venv：先提交 --phase cpu（bootstrap 装 Torch/编译 Rust 只能在计算节点执行）' >&2
  exit 2
fi
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export HSRL_TORCH_THREADS=4 PYTHONUTF8=1 PYTHONUNBUFFERED=1
out="${HSAI_EVAL_OUT:-artifacts/runs/duel-eval-$SLURM_JOB_ID}"
export HSAI_EVAL_OUT="$out"
if [[ -e "$out" ]]; then
  echo "评测输出目录已存在，请改用新目录并保留旧基线：$out" >&2
  exit 2
fi
# 登录节点与作业内都不猜设备：先由启动器留证，再由它 exec 评测入口（禁用 srun 包裹）。
# 两道预检都在作业内跑，任何一项失败都不会把半个报告留在目录里。
python -S tools/nus_submit.py --phase eval --check-only
python tools/stage_duel_models.py --check-only
hsrl_gpu_job python -u workflows/launch_nus_refinement.py --task eval-duel --out "$out"
echo "[eval] report.json/report.md -> $out"
