#!/usr/bin/env bash
#SBATCH --job-name=hsai-emerge
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --exclude=xgpi15,xgpi17,xgpi11
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --propagate=NONE
#SBATCH --time=00:40:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# 合并 N 个「1 卡 8 进程」作业的全部子面板（node*/p*/panel.json）。
# 纯 CPU，秒级；缺哪个就跳过哪个（不覆盖任何已有结果）。
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/HSAI}"
source .venv/bin/activate
export PYTHONUTF8=1 PYTHONUNBUFFERED=1

out="${HSAI_GAP_OUT:?需要 HSAI_GAP_OUT}"
nodes="${HSAI_NODES:-3}"
procs="${HSAI_NODE_PROCS:-8}"
[[ -d "$out" ]] || { echo "no such dir: $out" >&2; exit 2; }

panels=()
for i in $(seq 0 $((nodes-1))); do
  for j in $(seq 0 $((procs-1))); do
    p="$out/node$i/p$j/panel.json"
    [ -s "$p" ] && panels+=("$p")
  done
done
echo "collected ${#panels[@]} / $((nodes*procs)) sub-panels"
[[ ${#panels[@]} -eq 0 ]] && { echo "no sub-panels" >&2; exit 3; }
python tools/merge_panels.py "${panels[@]}" --out "$out/panel.json"
echo "merged -> $out/panel.json"
