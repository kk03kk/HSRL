# HSRL 的 NUS 工作区

NUS 上的 `~/projects/HSRL` 用于编辑、构建和实验。Windows 的 `D:\HSRL` 用于保留起始快照和未来的实机测试。Windows 的 SSH 配置提供连接。

## 连接

使用账户上已配置的 SSH 别名登录。连接拒绝时先核对别名、跳板与登录节点可用性。将连接配置和凭据保存在系统账户下。进入 HSRL 后，用 `pwd`、`git status --short --branch` 和 `git rev-parse HEAD` 核对位置及版本。

## 接入上游和建基线

每次接入新上游时先记录 URL、完整提交、上游分支、游戏补丁和卡表版本。首次直接在 NUS 克隆，在该提交创建研究分支。已有研究目录时，在独立分支和检出中验证新版本，保留旧运行所需的 wheel 和产物。当前演练使用的上游提交为 `1c2086a997afde46d68c6c5992cab194ddcfce88`。

## 构建和冒烟测试

登录节点只做轻量操作。以下命令在登录节点准备冻结检出并提交 CPU 作业，编译和测试由 Slurm 执行。

```bash
cd ~/projects/HSRL
git status --short --branch
commit=$(git rev-parse HEAD)
release="$PWD/artifacts/releases/$commit"
run="$PWD/artifacts/bootstrap/$(date -u +%Y%m%dT%H%M%SZ)-${commit:0:12}"
mkdir -p "$PWD/artifacts/releases"
if [ ! -d "$release" ]; then
    git worktree add --detach "$release" "$commit"
fi
mkdir -p "$run"
sbatch --chdir="$release" --output="$run/slurm-%j.log" \
    "$release/deploy/slurm/bootstrap.sh" "$run"
```

用 `squeue -u "$USER"` 看当前状态，用 `sacct -j <作业号> --format=JobID,State,ExitCode,Elapsed` 查最终结果，失败时读对应日志。脚本在独立 Python 环境构建 Linux release wheel，安装后直接验证轮子和对局，再运行上游 Python 测试。运行目录保留 wheel、哈希、环境版本、固定种子结果和作业日志。构建产物留在共享盘。

GPU 作业提交前用 `sinfo` 查分区和时限，并在作业内记录实际 GPU 型号和显存。CPU 建基线作业使用 `normal`，无需申请 GPU。作业依赖卡住时，先查前置作业是否成功及其输出路径是否可写。

## 日常同步与作业恢复

- NUS 的 `origin` 指向 HSRL 发布仓库，`upstream` 保留 AutoTavern 源。检查双方提交后以普通快进推送同步代码。Windows 需要实机验证时检出对应版本，并单独构建其原生扩展。
- 作业按固定检出和解释器运行。标准 GPU 分区时限为 3 小时，实际排队由调度器决定。九小时工作流按三段提交，Slurm 负责依赖和终评，连接断开后仍继续。
- 断点恢复打开对应分支的原子检查点。先读 `sacct`、训练日志和 `events.jsonl`，区分调度时间、程序退出、数值保护及引擎非法终局。确认冻结配置与断点一致后再续训。
- 研究记录见 [文档索引](README.md)。生产自动化见 [九小时运行手册](OVERNIGHT.md)。
