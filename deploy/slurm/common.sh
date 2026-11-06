#!/usr/bin/env bash
# NUS重任务统一边界：不解除登录节点限制，只在实际分配的计算节点运行。
hsrl_require_compute() {
  local host limit
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo '需要Slurm计算节点分配；登录节点只做git、编辑、传输和sbatch。' >&2
    return 2
  fi
  host="$(hostname)"
  host="${host%%.*}"
  limit="$(ulimit -v)"
  if [[ "$host" == *login* || "$host" == sjump* ]]; then
    echo '需要在Slurm计算节点运行；登录节点只做git、编辑、传输和sbatch。' >&2
    return 2
  fi
  if [[ "$(uname -m)" != x86_64 ]]; then
    echo '此Slurm配方仅验证x86_64，不能复用到aarch64节点。' >&2
    return 2
  fi
  if [[ "$limit" != unlimited && "$limit" =~ ^[0-9]+$ && "$limit" -le 1048576 ]]; then
    echo 'Slurm当前进程虚拟地址空间不超过1 GiB，拒绝安装/编译/测试；请检查节点和作业限制。' >&2
    return 2
  fi
}

# 在已有sbatch分配内运行。本站实测srun子step会将两个MIG缩成一个，不能再包装。
# 不修改CVD、设备cgroup或CPU配额；启动器仍须验证原始Driver及映射后的实际两设备。
hsrl_gpu_job() {
  if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_CPUS_PER_TASK:-}" ]]; then
    echo 'GPU作业需要Slurm作业号及明确的CPU配额。' >&2
    return 2
  fi
  python -u workflows/launch_nus_refinement.py --probe-only \
    --out "artifacts/runs/gpu-visibility-${SLURM_JOB_ID}" || return $?
  "$@"
}
