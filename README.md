# HSRL

**Hearthstone Battlegrounds Reinforcement Learning** — 基于 [AutoTavern](https://github.com/MCG-NJU/AutoTavern) 的个人酒馆战棋强化学习项目。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/kk03kk/HSRL/actions/workflows/ci.yml/badge.svg)](https://github.com/kk03kk/HSRL/actions/workflows/ci.yml)

目标是训练能够独立经营与决策的模型，随后接入本地自动化。当前已完成离线训练链路与功能验证，正在进行服务器资格训练；尚未宣称达到高手水平或完成真实游戏接入。

## 当前路线

```text
Rust模拟器 → 机器人源对局 / 人工战团记录 → 分离的训练、验证、测试库
                                              ↓
                              Entity Transformer → BC初始化
                                              ↓
                                  基础行为验收 → 库对手PPO
                                              ↓
                             独立评测 / checkpoint / 离线回放
```

- **环境**：作者36.6.3模拟器基准，修正等数量先攻，并针对37.0、37.0.1公告进行定向修复与校验。
- **模型**：17,962,378参数，384维、8层、8头、49实体token的前馈Entity Transformer；离线TF-IDF/LSA128规则表示。
- **训练**：BC初始化后进行PPO；学习者正常经营，七个对手使用上下文匹配的战团库。目前基础库主要来自脚本机器人。
- **约束**：245个语义动作、合法动作掩码、每回合200次训练工程预算；保持随机大厅，不单独限制换位次数。
- **验证**：源对局分离、固定评测协议、严格模型身份校验、执行回执与轨迹回放。本地Python回归181项通过；原生回归877项通过。

这些是本项目的实现和验证状态，不代表完整复刻作者私有训练系统，也不是线上棋力成绩。

## 目录

| 路径 | 内容 |
|---|---|
| [`crates/taverngym`](crates/taverngym) | Rust模拟器、卡牌实现与原生测试 |
| [`crates/taverngym-py`](crates/taverngym-py) | PyO3原生接口 |
| [`python/taverngym/rl`](python/taverngym/rl) | 模型、采样、BC/PPO、评测与checkpoint |
| [`python/taverngym/research`](python/taverngym/research) | 战团库生成、BC验收与资源预检 |
| [`python/taverngym/offline`](python/taverngym/offline) | 离线协议、轨迹、回放与诊断 |
| [`configs`](configs) | 模型与实验配置 |
| [`assets`](assets) | 带指纹的规则输入资产与整理后的战团代理 |
| [`deploy/nus`](deploy/nus) | NUS Slurm准备、训练与评测脚本 |
| [`tools`](tools) | 采集、数据准备、校验与发布工具 |
| [`tests`](tests) | Python与页面逻辑回归 |
| [`tavernbench`](tavernbench) | 保留的上游决策基准与上游发布结果 |
| [`docs`](docs) | 当前使用说明、架构及历史文档索引 |

运行数据默认写入`runs/`。日志、模型、环境、客户端提取数据与个人工作区记录不纳入Git。

## 本地开始

需要Python 3.10+及[`rust-toolchain.toml`](rust-toolchain.toml)指定的Rust工具链。正式训练环境目前使用Python 3.12与PyTorch 2.10。

```bash
git clone https://github.com/kk03kk/HSRL.git
cd HSRL

# 仅需运行模拟器
pip install -e .
```

不依赖PyTorch的模拟器示例：

```python
import taverngym

game = taverngym.Game(seed=1)
while not game.is_over():
    for seat in range(game.num_players):
        if game.needs_action(seat):
            game.step(seat, game.bot_action(seat))
print(list(game.placements()))
```

离线开发与测试可以使用CPU版PyTorch：

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test]"
pytest -q
cargo test --release -p taverngym
```

确认原生构建与1796万模型前向：

```bash
python -m taverngym.rl.doctor --config configs/entity_library_nus.json --device cpu
```

`Game()`使用原生默认规则；当前训练使用`Config().rules`及`configs/`中的配置，包含已确认修复和200次预算。GPU环境安装见部署文档，不要在GPU训练环境照搬CPU版PyTorch安装命令。

## 使用入口

- [当前完整训练流程](docs/TRAINING_PIPELINE_CURRENT.md)
- [NUS：CPU准备 → 单A100资格训练](docs/NUS_ENTITY_LIBRARY_START.md)
- [训练操作预算](docs/TRAINING_ACTION_BUDGET.md)
- [战团录入工具](docs/WARBAND_COLLECTOR.md)
- [自动对手记录采集](docs/OPPONENT_CAPTURE.md)
- [离线回放与适配](docs/LOCAL_OFFLINE.md)
- [全部文档与历史阶段索引](docs/README.md)

保留`taverngym`包名和发布包中的`AutoTavern`目录名，以兼容已有环境、实验和部署脚本；Git仓库名为HSRL，本地文件夹也可以保留原名。

## 来源与许可

模拟器、Python原生接口、TavernBench及基础文档源自[MCG-NJU/AutoTavern](https://github.com/MCG-NJU/AutoTavern)，基线提交为`fb61201a557ddd36a411cf44aac51ce4d9b606ce`。保留上游Git历史、版权、[MIT许可证](LICENSE)、[NOTICE](NOTICE)与[论文引用信息](CITATION.cff)。

本项目增加训练器、战团库流程、采集/回放工具及部署配置；TavernBench中原作者发布的结果不属于本项目的训练成绩。游戏内容权利与项目声明见[NOTICE](NOTICE)。
