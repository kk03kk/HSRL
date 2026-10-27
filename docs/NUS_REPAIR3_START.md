# NUS repair3：评测截断修复与本地工具交付

1022803资格测试已完成40波（18:11，0:0）。1022804在BC8后第0波基线评测失败（3:28，1:0），尚未进入PPO。A100-PCIE-40GB与原生35275a3e7711a471正常预检；索引20大厅达到60回合上限，未出现动作拒绝。详见修复报告。

本包原生引擎源码不变，不需重建已通过的Linux扩展；改动训练Python的评测异常处理，因此不能在旧run直接 --resume。保留 `runs/nus-ppo-v2`，新run为v3。通过已验证的 --init-from 仅复用旧BC权重，重置优化器/RNG/进度/对手池，bc_waves=0，不重复8波BC。

## 1. 上传并校验

上传本地主项目 `runs/dist/AutoTavern-nus-repair3.tar.gz` 和同名 `.sha256` 到服务器HOME。不要覆盖旧runs或删除环境。

```bash
cd "$HOME"
sha256sum -c AutoTavern-nus-repair3.tar.gz.sha256 &&
cd "$HOME/projects/hearthstone-rl/AutoTavern" &&
python3 tools/package_nus.py --verify . &&
tar --no-same-owner -xzf "$HOME/AutoTavern-nus-repair3.tar.gz" -C .. &&
python3 tools/package_nus.py --verify .
test -s runs/nus-ppo-v2/bc-anchor.pt
```

这是完整源码包；旧归档manifest升级后由新的SOURCE_MANIFEST取代。旧模型保留原身份，不能称为新训练断点。

## 2. 提交独立新链

```bash
cd "$HOME/projects/hearthstone-rl/AutoTavern"
verify_job=$(sbatch --parsable deploy/nus/verify.sbatch) &&
smoke_job=$(sbatch --parsable --dependency=afterok:"$verify_job" deploy/nus/smoke_v3.sbatch) &&
qualification_job=$(sbatch --parsable --dependency=afterok:"$smoke_job" deploy/nus/qualification_v3.sbatch) &&
train_job=$(sbatch --parsable --dependency=afterok:"$qualification_job" deploy/nus/train_v3.sbatch) &&
printf 'verify=%s smoke=%s qualification=%s train=%s\n' \
  "$verify_job" "$smoke_job" "$qualification_job" "$train_job"
squeue -u "$(id -un)"
```

- verify：CPU，复用现有venv/native，完整Python测试和doctor；若原生指纹不符则阻断，不偷偷重建/忽略。
- smoke3：小模型GPU两波测试，新目录 `runs/nus-smoke-v3`。
- qual3：单A100-40、16CPU、64G，40波；复用旧生产BC权重，**每seed32大厅、两组生产验证种子、每10波评测**，与正式配置保持一致。旧资格配置只有8局和另一组种子，不能覆盖本次索引20案例。
- ppo3：相同资源、71小时时限、1000波上限，新目录 `runs/nus-ppo-v3`。从相同旧BC权重独立开始，不接着资格模型训练。

新run已有checkpoint时脚本严格resume该run，不能跨身份复用。第一次若发现v3目录已含其他实验，应改明确的新输出目录，不删除模型来绕过检查。

## 3. 检查重点

基线或后续评测可能记录 truncated_games>0；该面板只报告自然完整局的条件均值，**不会把截断算第八，不参与best选优**。selection_score=null、eligible_for_selection=false表示本次评测不合格，不代表训练采样报错。latest仍保存，best只有完整合格评测后才可能出现。

截断样例保存到 `runs/nus-ppo-v3/validation/iteration-XXXXXX/<panel>-<mode>-<seed>/failure-*.json`。这让我们能检查为什么对局拖到60回合。此次只修复评测故障处理，尚未认定达到回合上限究竟来自弱策略、局面循环还是其他规则差异。

真正动作拒绝、战斗截断/引擎停止原因仍为硬错误；训练rollout回合截断也继续严格拒收。出现这些错误需读取日志与快照，不能通过降低保护继续。

日志名为 `slurm-hsrl-smoke3-ID.out`、`slurm-hsrl-qual3-ID.out`、`slurm-hsrl-ppo3-ID.out`。本地工具见 `docs/LOCAL_OFFLINE.md`。本轮没有代用户上传或提交作业。
