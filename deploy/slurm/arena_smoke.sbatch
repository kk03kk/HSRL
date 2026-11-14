#!/usr/bin/env bash
#SBATCH --job-name=hsai-arena-smoke
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude=xgpi15,xgpi17
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --propagate=NONE
#SBATCH --time=00:40:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# 动态难度课程的第 0 步：先把两样**测量**做出来，再谈训练。
#   1) 真人战团库的强度曲线（每回合分位表）→ artifacts/state/human_curve.json
#      这是"能打赢 p25/p50"这类绝对刻度的锚，也是控制器的初始档位来源。
#   2) 一个 6 分钟/64 局的短冒烟：验证新增的**逐回合遥测**真的进了 history
#      （tel_own_str_* / tel_tier_* / tel_cum_dmg_* / tel_reach*），且
#      "把对手换成 p25 真人战团"这条路径注入率正常（human_board_applied 不为 0）。
#
# 冒烟参数刻意与正式臂一致（除了时长/局数）：mirror-p=0 + 全 7 席真人战团。
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export HSRL_TORCH_THREADS=2 PYTHONUTF8=1 PYTHONUNBUFFERED=1
export PYTHONPATH="engine/python:${PYTHONPATH:-}"

echo "=== [1/2] human strength curve ==="
python tools/pool_strength_curve.py \
  --pool "${HSAI_HUMAN_BOARDS:-data/real_library/firestone-bank-6k/train-m6k.jsonl}" \
  --out "${HSAI_CURVE_OUT:-artifacts/state/human_curve.json}"

echo "=== [2/2] telemetry smoke (6 min / 64 games) ==="
export HSAI_LR_OUT="${HSAI_SMOKE_OUT:-artifacts/runs/nus-arena-smoke-20261114-1}"
export HSAI_LR_MINUTES="${HSAI_SMOKE_MINUTES:-6}"
export HSAI_LR_GAMES="${HSAI_SMOKE_GAMES:-64}"
export HSAI_MIRROR_P="${HSAI_SMOKE_MIRROR_P:-0}"
export HSAI_HUMAN_SAMPLE="${HSAI_SMOKE_SAMPLE:-p25}"
export HSAI_HUMAN_BOARD_P=1.0
export HSAI_ENTRY_P=0
bash deploy/slurm/spell_l5.sbatch

echo "=== [3/3] telemetry keys in history ==="
python - <<'PY'
import json, os
run = os.environ.get("HSAI_SMOKE_OUT", "artifacts/runs/nus-arena-smoke-20261114-1")
path = os.path.join(run, "train", "history.jsonl")
if not os.path.isfile(path):
    cand = [os.path.join(run, d, "history.jsonl") for d in os.listdir(run)] if os.path.isdir(run) else []
    path = next((c for c in cand if os.path.isfile(c)), path)
rows = []
if os.path.isfile(path):
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
print(f"history={path} rows={len(rows)}")
if rows:
    last = rows[-1]
    keys = [k for k in sorted(last) if k.startswith(("tel_", "human_board_", "arena_"))]
    for k in keys:
        v = last[k]
        if isinstance(v, (dict, list)) and len(str(v)) > 120:
            v = "<big>"
        print(f"  {k} = {v}")
    need = ["tel_own_str_8", "tel_own_str_10", "tel_own_str_12", "tel_reach9",
            "tel_reach12", "tel_tier_10"]
    missing = [k for k in need if k not in last]
    print("MISSING:", missing if missing else "none")
PY
