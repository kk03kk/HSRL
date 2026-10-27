# 本地离线闭环使用说明

主目录：`D:\HearthstoneRL\AutoTavern`。本工具不启动游戏、不连接旧客户端、不发送 OS 输入，也不依赖朋友仓库或旧工程的 Python 包。

## 一键验收

在主目录、已安装本项目原生扩展与 train/test 依赖的 Python 环境中执行：

```powershell
python tools/validate_local_offline.py --out runs/offline-check-001
```

输出目录必须是新的主项目 `runs/` 子目录。执行 Python 全量测试、CPU 小规模 BC/两波 PPO、checkpoint 保存、bot/PPO 对局导出及精确重放、成对评测、原训练四分面独立 test、模拟动作回执和旧客户端记录审查。最终 `result.json` 的 verified=true 代表离线功能验收，不代表棋力或实机通过。

Rust 全量回归另外执行 `cargo test -p taverngym --locked`。Windows 需使用已配置好的工具链/链接器；本机为 1.98.1 gnullvm。新增 Node 页面逻辑测试是可选项，无 Node 的环境会明确 skip。

启动器自动登记当前 Python 的 Windows DLL 目录及本机已有 LLVM 目录。其他本机可用 `TAVERNGYM_DLL_DIRS` 指定额外目录；Linux 按正常动态链接环境运行。没有安装原生扩展时应先构建，不能回退 mock。

## 导出与查看

```powershell
python tools/local_offline.py capture --games 1 --seed 73001 --out runs/bot-review.jsonl
python tools/local_offline.py capture --checkpoint runs/local-delivery/release-accepted/learning/latest.pt --games 1 --sample --out runs/model-review.jsonl
python tools/local_offline.py replay runs/model-review.jsonl
python tools/local_offline.py report runs/model-review.jsonl --out runs/model-review-stats.json
python tools/local_offline.py viewer --out runs/replay-viewer.html
```

用本地浏览器打开 `runs/replay-viewer.html`，通过文件输入导入 JSONL。支持前后步骤、回合/大厅/席位跳转、简洁局面、完整玩家字段、合法动作与策略 top-k。批注按钮导出 annotation.v1，填写 recommended_action、acceptable_actions 和 reason。批注默认 training_approved=false，不能未经检查自动加入训练。

轨迹初始全局状态用于重放、含特权信息；策略只接收 observation，页面只显示单独过滤的玩家视图。bot 概率为空是“没有记录”，不代表概率为零。格式错误或缺 summary 的轨迹会拒收。

原生源码指纹必须一致才能精确重放。旧格式2失败快照可转换；旧格式1没有动作序列，无法恢复精确决策过程：

```powershell
python tools/local_offline.py import-failure runs/failure.json --out runs/failure-review.jsonl
```

如果失败快照来自旧引擎，应在匹配的冻结环境执行；不能绕过身份后声称精确复现。

## 独立评测

```powershell
python tools/local_offline.py compare --checkpoint bc=runs/local-delivery/release-accepted/learning/bc-anchor.pt --checkpoint ppo=runs/local-delivery/release-accepted/learning/latest.pt --seeds 1902000000 1902000001 --out runs/comparison.json
python tools/local_offline.py selection-eval --checkpoint ppo=runs/local-delivery/release-accepted/learning/latest.pt --anchor runs/local-delivery/release-accepted/learning/bc-anchor.pt --split test --games 32 --out runs/selection-test.json
python tools/local_offline.py import-metrics runs/local-delivery/release-accepted/learning/metrics.jsonl --out runs/training-summary.json
```

compare 是每个 seed 一个大厅、轮换 learner 席位、其余七个原生 bot 的成对协议。greedy/sample 分开统计，按大厅 bootstrap；失败与截断单列，成对差值仅使用双方都有效的配对，并报告有效 n。n=1 不提供置信区间。这个简化协议不能代替训练时的 best 分数。

selection-eval 直接调用现有训练 evaluate，复核 bots/anchor_mix × greedy/sample 四分面；test 使用未参与选优的两组种子，validation 保留 checkpoint 的验证种子。--games 是每 seed 的大厅数。不同引擎、规则、架构或记忆身份不得混在同一次评测中。

服务器模型取回后替换 --checkpoint/--anchor 路径即可；如本地未来修改训练语义，改用匹配的冻结基线运行，不能关闭身份校验。导入日志仅汇总已有内容，不能从文字日志推导未保存的动作概率。

## 离线自动化适配

```powershell
python tools/local_offline.py client-audit tests/fixtures/offline/legacy-client.json --build 255130 --build 37.0.0.255130 --out runs/client-audit.json
python tools/local_offline.py sandbox --checkpoint runs/local-delivery/release-accepted/learning/latest.pt --steps 12000 --out runs/sandbox.json
```

sandbox 使用原生模拟器投影，显式 source=native_fixture、real_client_verified=false。它证明新模型与语义动作/回执链能对接，不证明真实旧客户端观测已经足够。

真实历史样例可直接审查，但当前缺少 policy_context，禁止自动补零输入模型。完整上下文、来源和语义动作绑定契约见 [玩家观测与客户端离线桥契约](OFFLINE_INPUT_CONTRACT.md)。观察器/历史重建实现属于实际接入前的后续工作，旧系统本轮只读。

已实现 PublicHistory 的确定性计数重建：带 game_start/turn_start/receipt/public_counter/gap 的事件流可通过 `rebuild-history INPUT.jsonl --out runs/history.json` 转为当前计数。只按明确 applied 计数；日志中途开始、漏回合或出现 gap 时输出未知。它不自动推断复杂卡牌的全部内部计数。真实样例的build为254477，除字段缺失外也有明确版本差异；放宽版本白名单不代表规则已经一致。

## 冻结与发布

```powershell
python tools/freeze_local_baseline.py --checkpoint runs/local-delivery/release-accepted/learning/latest.pt --out runs/baseline-next
python tools/package_nus.py --out runs/dist/AutoTavern-nus-repair3.tar.gz
```

冻结目录含独立源码、同平台原生扩展、checkpoint 与逐文件 SHA256。Linux/CUDA 服务器仍应从匹配源码构建自己的扩展，不能使用 Windows pyd。源码发布包附同名 .sha256；与当前服务器 repair2 包分开，不自动上传或提交作业。

本地工具完成后，用户补充1022804在基线评测触发回合上限而失败。本轮追加修复评测截断处理，训练源码身份因此改变、原生规则未变。旧模型请使用 `python tools/local_offline.py --baseline runs/local-delivery/baseline ...` 在冻结rl/原生环境审阅；不要绕过严格校验。新训练使用新run与明确权重初始化，见 `docs/NUS_REPAIR3_START.md`。
