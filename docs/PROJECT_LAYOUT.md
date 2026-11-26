# 项目文件框架

```text
HSRL/
├── crates/        Rust 模拟器、策略教师、PyO3 绑定和 Rust 测试
├── python/        taverngym 包、18M 模型、BC、PPO 及环境适配
├── deploy/        Slurm 作业入口、依赖和数据切分配置
├── tools/         跨阶段数据管线、复现、诊断、建库和评测工具
├── tests/         Python 回归及接口测试
├── tavernbench/   上游 36.6.3 评测任务和工具
├── docs/          研究手册、分阶段说明、历史结果和上游参考
├── artifacts/     NUS 的 wheel、数据、权重、日志和冻结检出，Git 忽略
└── runs/          本地轻量输出和实机工作，Git 忽略
```

## 从功能找源码

| 要修改或核对的模块 | 入口 |
| --- | --- |
| 游戏规则、卡牌池、观测和动作 | [引擎](../crates/taverngym/src/)中的 `engine.rs`、`flow.rs`、`config.rs`、`actions.rs`、`obs.rs` 和 `content/` |
| 当前卡表和补丁生成 | [卡表](../crates/taverngym/data/)与[生成器](../tools/gen_carddb.py)、[补丁增量](../tools/overlays/) |
| 手写教师、流派与原版 Bot | [策略实现](../crates/taverngym/src/strategy/)的 `TEACHER_POLICY` 和 `PolicyState`、[原版 Bot](../crates/taverngym/src/bot.rs) |
| 教师轨迹、整局竞技场与对照 | [Rust 竞技场](../crates/taverngym/src/bc.rs)、[策略及采样示例](../crates/taverngym/examples/) |
| Python 扩展和实体模型 | [PyO3 绑定](../crates/taverngym-py/src/lib.rs)、[模型](../python/taverngym/entity_policy.py)、[Python 包](../python/taverngym/) |
| BC 自然数据、纠错与准入 | [训练与评测](../python/taverngym/bc_train.py)、[纠错](../python/taverngym/bc_dagger.py)、[准入](../tools/bc_admission.py) |
| PPO 采样、更新、检查点 | [PPO 入口](../python/taverngym/ppo_train.py)、[离线动作审计](../crates/taverngym/examples/ppo_audit.rs) |
| 战团导入、采样和实战分布 | [战团库](../crates/taverngym/src/warband.rs)、[场面注入](../crates/taverngym/src/snapshot.rs)、[建库与评测](../tools/warband_research.py) |
| NUS 作业与自动续训 | [Slurm 入口](../deploy/slurm/)、[分段控制器](../tools/ppo_night.py) |
| 测试及真实规则夹具 | [Rust 测试](../crates/taverngym/tests/)、[Python 测试](../tests/) |

根目录的 `Cargo.toml`、`Cargo.lock` 和 `rust-toolchain.toml` 定义 Rust 工作区、依赖和工具链。 `pyproject.toml` 定义 Python 环境与 wheel。 `.github/workflows/` 管理持续集成。许可、第三方权利和上游引用保存在 `LICENSE`、`NOTICE`、`CITATION.cff`。

## 执行位置与产物

NUS 的 `~/projects/HSRL` 是开发与训练工作区。Windows 本地用于查看、轻量测试和后续实机验证。每次训练与正式评测都从固定提交和已核对的本平台 wheel 运行。

| 路径 | 留存内容 |
| --- | --- |
| `artifacts/releases/<commit>/` | 不随主线移动的冻结源码检出 |
| `artifacts/bootstrap/`、`artifacts/simulator/` | 轮子、回归报告、日志、规则与牌表证据 |
| `artifacts/strategy/`、`artifacts/bc/` | 教师结果、示范和纠错数据、准入权重 |
| `artifacts/ppo/` | 训练清单、模型和优化器、逐轮指标、独立验收 |
| `artifacts/warband/` | 素材原局清单、库切分、压力标定、对照与轮子 |
| `runs/` | 本机少量报告与验证记录 |

同一来源只保留必要的轻量证明进入 Git。跨补丁的原始日志、旧权重和旧 wheel 按版本保存，项目运行使用 HSRL 已生成并登记的资产。
