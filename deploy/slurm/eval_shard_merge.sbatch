#!/usr/bin/env bash
#SBATCH --job-name=hsai-smerge
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=cpu
#SBATCH --exclude=xgpi15,xgpi17,xgpi11
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --propagate=NONE
#SBATCH --time=00:40:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# 合并 N 个分片作业的结果（纯 CPU，秒级）。缺哪个分片就跳过哪个，不覆盖已有 panel.json。
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/HSAI}"
source .venv/bin/activate
export PYTHONUTF8=1 PYTHONUNBUFFERED=1

out="${HSAI_GAP_OUT:?需要 HSAI_GAP_OUT}"
nshard="${HSAI_SHARDS:-8}"
[[ -d "$out" ]] || { echo "no such dir: $out" >&2; exit 2; }

shards=()
for i in $(seq 0 $((nshard-1))); do
  [ -s "$out/shard$i/panel.json" ] && shards+=("$out/shard$i/panel.json")
done
echo "collected ${#shards[@]}/$nshard shards"
[[ ${#shards[@]} -eq 0 ]] && { echo "no shard panels" >&2; exit 3; }
python tools/merge_panels.py "${shards[@]}" --out "$out/panel.json"
echo "merged -> $out/panel.json"
