# 首个模型失败后的CPU诊断

2026-10-27。目标是取得可复现的失败证据和模型行为，当前不续训、不修改已验收发布版，不重新安装依赖或构建Rust。

## 已知事实

1023992运行32:20后FAILED/1:0，PPO已完成139轮；第140轮seed14001121、第14大厅出现一次truncated_combats。绑定diagnostics顺序是turn_limit_end、truncated_combats、stopped_reason；所以(False,1,None)不是回合上限，也不是CUDA内存错误。具体可能是攻击次数、效果深度或异常停滞保护，尚需失败快照逐步重放。

latest/best/BC权重和failure-seed-14001121-game-14.json已保存。latest按代码保存顺序应对应第139轮，下一诊断会实际读取确认；不是20万预算已完成。固定机器人考试长期接近第8，不具备直接部署强模型的证据。附件只含部分日志，不能把可见134条PPO求和当作完整训练预算。

## 在本机PowerShell上传

```powershell
scp -J YOUR_NUS_USER@sjump.comp.nus.edu.sg "D:\HearthstoneRL\AutoTavern\tools\diagnose_pilot.py" "D:\HearthstoneRL\AutoTavern\deploy\nus\diagnose_first_model.sh" "D:\HearthstoneRL\AutoTavern\runs\dist\first_model_diagnosis.sha256" YOUR_NUS_USER@xlogin.comp.nus.edu.sg:~/
```

## 服务器执行一次

如果尚未取消后三项架构对照，先停止它们，已有checkpoint和日志保留：

```bash
scancel 1023993 1023994 1023995
```

提交诊断：

```bash
cd "$HOME"
sha256sum -c first_model_diagnosis.sha256 &&
bash "$HOME/diagnose_first_model.sh"
```

脚本进入既有paper-stage-v1独立发布目录，验证原源码清单，提交1个normal分区作业：8CPU、16GB、最长45分钟，无GPU。只复用现有虚拟环境和原生扩展。新工具复制到独立audit目录，不加入原发布清单，也不修改rl/，严格checkpoint身份检查仍有效。

打印audit_job与audit_directory后，把输出发回即可；不要重复提交。三个模型各跑greedy/sample×两个固定seed，每panel2大厅，另有脚本teacher状态上的动作一致率。只做推理与评测，不创建优化器、不更新权重。

## 诊断内容

1. 汇总完整metrics、实际checkpoint进度和有效样本数。
2. 从初始状态和原动作序列精确重放失败，比较最终状态哈希，保存触发前状态；不忽略保护或更换native源码。
3. 比较BC/latest/best的teacher动作一致率、买卖/上场/升本/刷新/结束回合分布、每回合战团数量与结束时金币。
4. 失败或截断单列，不偷偷算成第8；检查不匹配身份时明确失败。
5. 自动打包模型、原始失败快照、manifest、metrics及新诊断报告，检查原始训练文件未变化。

normal作业COMPLETED仅表示诊断完成，不表示模拟器已修好或棋力通过。单个panel异常会在报告中保留并让检查失败，但仍尝试导出证据包。

## 诊断结束后取回文件

首次导出默认保存服务器HOME下 `AutoTavern-first-model-evidence-1023992.tar.gz` 与 `.sha256`。已有同名文件会保留，届时以作业日志打印的新audit目录为准，不覆盖旧证据。

本机PowerShell：

```powershell
New-Item -ItemType Directory -Force "D:\HearthstoneRL\AutoTavern\runs\nus-first-model-1023992" | Out-Null
scp -J YOUR_NUS_USER@sjump.comp.nus.edu.sg 'YOUR_NUS_USER@xlogin.comp.nus.edu.sg:~/AutoTavern-first-model-evidence-1023992.tar.gz*' "D:\HearthstoneRL\AutoTavern\runs\nus-first-model-1023992\"
```

取回后由主项目校验、解包和分析。第一模型已经有权重，不需为取得模型而盲目重跑BC；是否继续PPO、修规则或调整冷启动，取决于此次真实诊断。实机接入先做匹配身份评测与影子编码，不把当前第8名策略直接标成可无人部署。

## 本地验收

诊断/训练/论文阶段相关回归29项通过；最后4项诊断回归通过。刻意触发原生战斗保护的真实引擎样例已完整重放，post哈希一致；匹配的本地功能checkpoint完成三模型、12个小panel，原权重未改。不匹配的旧本地checkpoint被严格拒绝。以上是工具功能验证，不冒充服务器模型棋力或服务器失败根因已查明。
