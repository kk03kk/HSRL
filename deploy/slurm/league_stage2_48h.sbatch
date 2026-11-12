#!/usr/bin/env bash
#SBATCH --job-name=hsai-league-stage2-48h
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:2
#SBATCH --exclude=xgpi15,xgpi17
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --propagate=NONE
#SBATCH --time=50:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# 48h stage-2 league run (94 x 30 min segments + periodic validation).
# Prerequisite: artifacts/models/stage2-proposal-20261111 was created by
# workflows/fork_stage.py from league-final-14858; it resets the LR horizon and
# anchor because the restored trainer otherwise wins over the new config.
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export HSRL_TORCH_THREADS=2 PYTHONUTF8=1 PYTHONUNBUFFERED=1

out="${HSAI_LEAGUE_OUT:-artifacts/runs/league-stage2-48h-$(date +%Y%m%d-%H%M%S)}"
config="${HSRL_CONFIG:-config/league-stage2.json}"
lib="${HSRL_LEAGUE_BANK:-artifacts/libraries/league-eot-v1.1-20261109}"
source_state="${HSAI_LEAGUE_SOURCE:-artifacts/models/stage2-proposal-20261111}"
validate_every="${HSRL_VALIDATE_EVERY:-4}"
validate_games="${HSRL_VALIDATE_GAMES:-32}"
validate_test_at="${HSRL_VALIDATE_TEST_AT:-47 94}"
segments="${HSRL_SEGMENTS:-94}"
minutes="${HSRL_SEGMENT_MINUTES:-30}"
mirror_sample="$(CONFIG="$config" python -c 'import json,os;print(json.load(open(os.environ["CONFIG"]))["mirror-sample"])')"

if [[ -e "$out" ]]; then echo "output exists: $out" >&2; exit 2; fi
if [[ ! -f "$source_state/model.pt" || ! -f "$source_state/trainer.pt" ]]; then
  echo "missing stage fork: $source_state" >&2; exit 2
fi
mkdir -p "$out/eval"
export HSRL_POOL_EVIDENCE_DIR="$(realpath -m "$out")/pool-evidence"
{
  echo "start=$(date -Is)"
  echo "host=$(hostname)"
  echo "slurm_job=${SLURM_JOB_ID:-}"
  echo "git_head=$(git rev-parse HEAD)"
  echo "out=$out"
  echo "library=$lib"
  echo "source_state=$source_state"
  echo "config=$config"
  echo "mirror_sample=$mirror_sample"
  echo "segments=$segments minutes=$minutes validate_every=$validate_every"
  echo "--- nvidia-smi -L ---"
  nvidia-smi -L
  echo "--- nvidia-smi ---"
  nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used --format=csv,noheader
} > "$out/run-info.txt" 2>&1

validate_split() {
  local split="$1" seed="$2" tag="$3"
  local out_json="$out/eval/${tag}-${split}.json"
  echo "validate ${split} ${tag} start $(date -Is)"
  if ! python -u workflows/validate_league_checkpoint.py \
      --checkpoint "$out/train/model.pt" \
      --library "$lib" --split "$split" \
      --games "$validate_games" --seed "$seed" \
      --sample "$mirror_sample" --forbid-unfreeze 1 \
      --device cuda --out "$out_json" ; then
    echo "validate ${split} ${tag} FAILED; preserving workspace" >&2
    return 3
  fi
  echo "validate ${split} ${tag} done $(date -Is)"
}

prev_version="$(python -c 'import json;print(json.load(open("'"$source_state"'/checkpoint.json"))["version"])')"
for i in $(seq 1 "$segments"); do
  if [[ -e "$out/STOP_REQUESTED" ]]; then
    echo "STOP_REQUESTED at segment $i; stopping at boundary $(date -Is)"
    break
  fi
  echo "segment $i/$segments start $(date -Is) prev_version=$prev_version"
  if [[ "$i" == "1" ]]; then
    python -m hsbg.az.train.personal train \
      --config "$config" \
      --checkpoint "$source_state/model.pt" \
      --initial-trainer-state "$source_state/trainer.pt" \
      --library "$lib/train" --out "$out/train" --device cuda \
      --run-minutes "$minutes"
  else
    python -m hsbg.az.train.personal train \
      --config "$config" --resume \
      --library "$lib/train" --out "$out/train" --device cuda \
      --run-minutes "$minutes"
  fi
  echo "segment $i/$segments training done $(date -Is)"
  python -u workflows/league_health.py \
    --history "$out/train/history.jsonl" \
    --latest "$out/train/latest.json" \
    --min-version "$prev_version" --expect-mirror 1 \
    > "$out/eval/segment-$(printf '%02d' "$i")-health.json"
  prev_version="$(python -c 'import json;print(json.load(open("'"$out"'/train/latest.json"))["version"])')"
  echo "segment $i/$segments health OK version=$prev_version $(date -Is)"
  if (( i % validate_every == 0 )); then
    validate_split dev 930000 "segment-$(printf '%02d' "$i")"
    if [[ " $validate_test_at " == *" $i "* ]]; then
      validate_split test 940000 "segment-$(printf '%02d' "$i")"
    fi
  fi
done

python - "$out" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
out = {"finished_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z")}
for name in ("model.pt", "trainer.pt"):
    p = root / "train" / name
    if p.is_file():
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[name] = {"path": str(p), "bytes": p.stat().st_size, "sha256": h.hexdigest()}
latest = root / "train" / "latest.json"
if latest.is_file():
    out["latest"] = json.loads(latest.read_text(encoding="utf-8"))
(root / "final-status.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
echo "long run loop complete $(date -Is)"
