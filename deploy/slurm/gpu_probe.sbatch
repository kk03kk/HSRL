#!/usr/bin/env bash
#SBATCH --job-name=hsai-gpuprobe
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
# 30-second diagnostic. Reports what this site hands out for the GRES on the
# command line (override with: sbatch --gres=gpu:<type>:2 gpu_probe.sbatch).
# Also lists the node's FULL physical GPU inventory by unsetting
# CUDA_VISIBLE_DEVICES for the diagnostic only (never done for training).
set -uo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source .venv/bin/activate
echo "host=$(hostname) job=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
echo "--- scontrol alloc ---"
scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | tr ' ' '\n' | grep -E 'ReqTRES|AllocTRES' | head -4
echo "--- nvidia-smi -L (as allocated) ---"
nvidia-smi -L
echo "--- nvidia-smi -L (FULL node inventory, CVD unset) ---"
env -u CUDA_VISIBLE_DEVICES nvidia-smi -L
echo "--- torch (as allocated) ---"
python - <<'PY'
import os
import torch
print("device_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("  dev", i, torch.cuda.get_device_name(i),
          round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1), "GiB")
print("CVD", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY
echo "--- torch (FULL node inventory, CVD unset) ---"
env -u CUDA_VISIBLE_DEVICES python - <<'PY'
import torch
print("device_count_all", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("  all dev", i, torch.cuda.get_device_name(i),
          round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1), "GiB")
PY
echo "=== done ==="
