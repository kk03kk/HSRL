#!/usr/bin/env bash
#SBATCH --job-name=hsai-migprobe
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:2
#SBATCH --exclude=xgpi15,xgpi17
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --propagate=NONE
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# Probe: can we expose TWO MIG slices as two CUDA devices by rewriting
# CUDA_VISIBLE_DEVICES with the MIG UUIDs the job was granted?
set -uo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source .venv/bin/activate
echo "host=$(hostname) job=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "CVD_original=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "--- nvidia-smi -L (as allocated) ---"
nvidia-smi -L
echo "--- torch as allocated ---"
python - <<'PY'
import torch
print("device_count =", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print("  dev", i, torch.cuda.get_device_name(i), round(p.total_memory / 2**30, 1), "GiB")
PY
echo "--- extract MIG UUIDs from nvidia-smi -L ---"
MIG_IDS=$(nvidia-smi -L 2>/dev/null | grep -o 'MIG-[0-9a-f-]*' | head -2 | paste -sd, -)
echo "MIG_IDS=$MIG_IDS"
echo "--- torch with CVD = the two MIG UUIDs ---"
if [ -n "$MIG_IDS" ]; then
  CUDA_VISIBLE_DEVICES="$MIG_IDS" python - <<'PY'
import os, torch
print("CVD =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("device_count =", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    try:
        import torch.cuda as tc
        uuid = tc.get_device_properties(i).uuid if hasattr(tc.get_device_properties(i), "uuid") else "?"
    except Exception:
        uuid = "?"
    print("  dev", i, torch.cuda.get_device_name(i), round(p.total_memory / 2**30, 1), "GiB")
PY
else
  echo "NO MIG UUIDs FOUND"
fi
echo "=== done ==="
