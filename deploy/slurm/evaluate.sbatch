#!/usr/bin/env bash
#SBATCH --job-name=hsrl-fixed-eval
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --propagate=NONE
#SBATCH --time=02:30:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
: "${HSRL_CANDIDATE:?请设置已完成Pilot的model.pt路径}"
test -f "$HSRL_CANDIDATE"
source .venv/bin/activate
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HSRL_TORCH_THREADS=2 PYTHONUTF8=1 PYTHONUNBUFFERED=1
out="${HSRL_EVAL_OUT:-artifacts/runs/fixed-eval-$SLURM_JOB_ID}"
[[ ! -e "$out" ]] || { echo '评测输出目录已存在，请使用新目录保留基线' >&2; exit 2; }
mkdir -p "$out"
nvidia-smi > "$out/nvidia-smi.txt"
python -m pip freeze > "$out/requirements.txt"
python -c "from hsbg.az.train.personal import training_source_digest; print(training_source_digest())" > "$out/revision.txt"
lib=artifacts/libraries/bot-v1/dev
games="${HSRL_EVAL_GAMES:-64}"
# 仅复用原始Best的完整冻结结果；后续配对仍严格检查全部协议及seed/seat。
if [[ -n "${HSRL_REFERENCE:-}" ]]; then
  python workflows/reuse_eval_reference.py --reference "$HSRL_REFERENCE" --out "$out/best.json" --games "$games"
else
  python -m hsbg.az.train.personal eval --checkpoint artifacts/models/s14_best_main_model.pt \
    --library "$lib" --games "$games" --device cuda --out "$out/best.json"
fi
python -m hsbg.az.train.personal eval --checkpoint "$HSRL_CANDIDATE" \
  --library "$lib" --games "$games" --device cuda --out "$out/candidate.json"
# baseline也声明相同推理运行环境，使protocol可严格比较。
baselines=()
if [[ "${HSRL_EVAL_BASELINES:-1}" == 1 ]]; then baselines=(bot random); fi
results=("$out/best.json" "$out/candidate.json")
for policy in "${baselines[@]}"; do
  python -m hsbg.az.train.personal eval --policy "$policy" --library "$lib" \
    --games "$games" --device cuda --out "$out/$policy.json"
  results+=("$out/$policy.json")
done
python workflows/paired_panel_delta.py --before "$out/best.json" \
  --after "$out/candidate.json" --out "$out/paired.json"
python -m hsbg.az.train.personal leaderboard "${results[@]}" --out "$out/leaderboard.json"
