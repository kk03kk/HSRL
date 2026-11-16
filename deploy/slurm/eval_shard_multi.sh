#!/usr/bin/env bash
# 把一次评测拆成 N 个**独立作业**（各占一块 GPU，避免 GPU 前向被串行化），
# 全部结束后自动合并。实测依据见 deploy/slurm/eval_shard_one.sbatch 顶部注释。
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/HSAI}"

total="${HSAI_GAP_GAMES:-512}"
nshard="${HSAI_SHARDS:-8}"
base="${HSAI_GAP_SEED:-930000}"
out="${HSAI_GAP_OUT:?需要 HSAI_GAP_OUT（新目录）}"
export HSAI_SHARDS="$nshard" HSAI_GAP_OUT="$out"
[[ -e "$out" ]] && { echo "output exists: $out" >&2; exit 2; }
mkdir -p "$out"

per=$(( (total + nshard - 1) / nshard ))
echo "total=$total shards=$nshard per_shard=$per base_seed=$base out=$out"

jids=()
for i in $(seq 0 $((nshard-1))); do
  s=$(( base + i * per ))
  jid=$(HSAI_GAP_GAMES="$per" HSAI_GAP_SEED="$s" HSAI_GAP_OUT="$out/shard$i" \
        sbatch --parsable --export=ALL --job-name="hsai-s$i" \
        deploy/slurm/eval_shard_one.sbatch | tail -1)
  jids+=("$jid")
  echo "  shard$i job=$jid seed=$s games=$per"
done

deps=$(IFS=:; echo "${jids[*]}")
m_jid=$(sbatch --parsable --dependency=afterany:"$deps" \
        --export=ALL,HSAI_GAP_OUT="$out",HSAI_SHARDS="$nshard" \
        deploy/slurm/eval_shard_merge.sbatch | tail -1)
echo "MERGE job=$m_jid  (afterany:$deps)"
echo "=== done ==="
