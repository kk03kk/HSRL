# 配置入口索引

- `engine-upstream-lock.json`：引擎源码身份，不得绕过。
- `live/`：真实游戏 profile 和模型 pin；离线评测不自动改 pin。
- `evaluation/`：新评测的显式计划。模型 SHA、seed、局数、协议和停止规则在提交前确定。
- `environments/`：按 Windows CPU / NUS CUDA 角色锁定的包清单；引擎轮子/构建单独验收。
- `reference-assets.json`：可选外部参考源码的内容清单；缺少资产明确失败，不作为训练依赖。
- `league-*.json`、早期 curriculum/benchmark 配置是已保留的特定实验入口，**不等于下一轮
  已获批准的训练方案**。当前配方指针见 `docs/TRAINING_RECIPE.md` 与最新决议。

不为整理而移动仍被代码引用的科学配置。历史配置保留可复现性；开启其中任何新实验仍需明确授权。
