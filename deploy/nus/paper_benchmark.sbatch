#!/usr/bin/env bash
#SBATCH --job-name=hsrl-paper-bench
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:45:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source deploy/nus/env.sh
python tools/package_nus.py --verify .
python -m taverngym.research benchmark \
  --configs configs/paper_stage/legacy128.json configs/paper_stage/compact_gru128.json \
            configs/paper_stage/compact_ff128.json configs/paper_stage/compact_ff192.json \
  --device cuda --out "runs/nus-paper-benchmark-v1/job-$SLURM_JOB_ID" &
child=$!
trap 'kill -USR1 "$child" 2>/dev/null || true' USR1
trap 'kill -TERM "$child" 2>/dev/null || true' TERM INT
set +e
wait "$child"
status=$?
if kill -0 "$child" 2>/dev/null; then
  wait "$child"
  status=$?
fi
exit "$status"
