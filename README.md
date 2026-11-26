# HSRL 酒馆战棋 AI 研究

HSRL 以 [AutoTavern](https://github.com/MCG-NJU/AutoTavern) 的 TavernGym 为模拟器基础。仓库记录从服务器规则验收、手写策略教师、行为克隆，到 PPO、战团库和无人值守训练的完整工作流。30 多套攻略流派用于教师开发和终局阵容审计。

**当前学习基线是 37.0.3 热修、八席、十族抽五、每招募回合 100 次操作。** 18M 实体模型已通过 BC 准入。脚本场 PPO 第 288 轮已通过独立棋力验收。战团库与短训验证完成，九小时混合训练的最终结论以该作业的独立验收为准。

| 要做什么 | 从哪里开始 |
| --- | --- |
| 回顾整轮如何完成，准备接入新补丁 | [研究流程与迁移手册](docs/WORKFLOW.md) |
| 找到引擎、教师、Python 模型、作业和产物 | [项目目录](docs/PROJECT_LAYOUT.md) |
| 查环境和 Slurm 操作 | [NUS 工作手册](docs/NUS.md) |
| 看实际棋力、训练曲线和遗留问题 | [37.0.3 研究记录](docs/reports/37.0.3.md) |
| 查脚本参数、门槛或基准接口 | [文档索引](docs/README.md) |

## 快速运行模拟器

需要 Python 3.10+ 和 [rustup](https://rustup.rs)。Rust 版本锁在 [rust-toolchain.toml](rust-toolchain.toml)。NUS 上通过 [构建与验收入口](docs/NUS.md) 在计算节点编译。

```bash
git clone https://github.com/kk03kk/HSRL.git
cd HSRL
python -m pip install .
```

```python
import taverngym

game = taverngym.Game(seed=1, config_json=taverngym.hsrl_config_json())
while not game.is_over():
    for seat in range(game.num_players):
        if game.needs_action(seat):
            game.step(seat, game.bot_action(seat))

print(list(game.placements()))
```

## 上游与许可

上游模拟器与论文基准的使用说明、引用和历史版本见 [中文说明](docs/upstream/README.zh-CN.md) 和 [English reference](docs/upstream/README.en.md)。代码采用 [MIT 许可证](LICENSE)，第三方游戏内容权利见 [NOTICE](NOTICE)。GitHub 发布源码和可公开的说明，权重、原始回放与完整训练产物保存在独立实验目录。
