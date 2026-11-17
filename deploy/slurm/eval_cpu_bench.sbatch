#!/usr/bin/env bash
#SBATCH --job-name=hsai-cpueval
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --propagate=NONE
#SBATCH --time=00:40:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# CPU 推理基准（**不占 GPU**）：量出「每局耗时」与「8 进程能否真并行」。
# 若成立，以后评测就不必抢 GPU 卡。
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/HSAI}"
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUTF8=1 PYTHONUNBUFFERED=1

out="${HSAI_BENCH_OUT:?需要 HSAI_BENCH_OUT}"
procs="${HSAI_BENCH_PROCS:-8}"
per="${HSAI_BENCH_PER:-16}"
base="${HSAI_BENCH_SEED:-930000}"
model="${HSAI_BENCH_MODEL:-artifacts/models/arena-v13-20261116/model.pt}"
snaps="${HSAI_BENCH_SNAPS:-artifacts/libraries/arena-cal-v13hold-snap}"
[[ -d "$snaps" ]] || { echo "no snaps: $snaps" >&2; exit 2; }
[[ -e "$out" ]] && { echo "exists: $out" >&2; exit 2; }
mkdir -p "$out"

echo "procs=$procs per=$per total=$((procs*per)) device=cpu node=$(hostname) job=${SLURM_JOB_ID:-local}" | tee "$out/bench.env"
t0=$(date +%s)
for i in $(seq 0 $((procs-1))); do
  s=$(( base + i * per ))
  (
    PYTHONPATH=engine/python:. python workflows/eval_human_gap.py \
      --checkpoint "bench=$model" \
      --games "$per" --seed "$s" --device cpu \
      --forbid-unfreeze 1 --freeze-cap 1 \
      --mirror-snaps "$snaps" --mirror-sample p50band --mirror-strict 0 \
      --out "$out/p$i/panel.json" > "$out/p$i.log" 2>&1
    echo "$i rc=$?"
  ) &
done
wait
t1=$(date +%s)
el=$((t1-t0))
tot=$((procs*per))
python - "$out" "$el" "$tot" <<'PY'
import glob, json, sys
out, el, tot = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
games = 0; dec = 0
for p in glob.glob(out + "/p*/panel.json"):
    m = json.load(open(p, encoding="utf-8"))["models"][0]
    games += int(m.get("games") or 0)
    dec += int((m.get("outcome") or {}).get("decisions") or 0)
print()
print("=== CPU 基准结果 ===")
print("  并发进程      : %d" % (len(glob.glob(out + '/p*/panel.json'))))
print("  完成局数      : %d" % games)
print("  总决策        : %d" % dec)
print("  墙钟          : %d s" % el)
print("  吞吐          : %.2f 局/s   %.0f 决策/s" % (games/max(el,1), dec/max(el,1)))
print("  折合 512 局   : %.0f s" % (512.0/max(games/max(el,1), 1e-9)))
PY
echo "done=$(date -Is) elapsed=${el}s" | tee -a "$out/bench.env"
