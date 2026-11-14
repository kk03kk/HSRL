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
# 30-second diagnostic: what GPU visibility does this site actually hand out for
# --gres=gpu:h100-47:2 ? Prints Slurm + CUDA + torch views. No training.
set -uo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source .venv/bin/activate
echo "host=$(hostname) job=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "SLURM_GPUS=${SLURM_GPUS:-<unset>}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<unset>}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
echo "--- nvidia-smi -L ---"
nvidia-smi -L
echo "count=$(nvidia-smi -L | wc -l)"
echo "--- torch ---"
python - <<'PY'
import os
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("is_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("  dev", i, torch.cuda.get_device_name(i))
print("CVD(os.environ)", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY
echo "--- scontrol ---"
scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | tr ' ' '\n' | grep -E 'AllocTRES|ReqTRES' | head -4
echo "=== done ==="
