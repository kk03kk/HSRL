# 服务器交接：架构调整（2026-11-10）

```yaml
status: pending          # 等服务器 Agent 执行后改 confirmed
owner: local             # 本地写，服务器执行并在 SERVER.md 回复
updated: 2026-11-10 22:05 +0800
base_commit: 见本文件所在提交
```

## 0. 决定与边界（先读 `docs/ARCHITECTURE.md`）

从本轮起：**服务器统一负责模拟器与强化学习的全部开发、审计、测试、训练**；
**本地只负责真实游戏桥接**。GitHub 只同步代码与文档。

| 服务器拥有写权 | 本地拥有写权 |
|---|---|
| `engine/`、`hsbg/`、`workflows/`、`deploy/`、`data/`、`tools/`、`tests/`（除 `test_live_*`）、`pyproject.toml`、`requirements.txt`、`README.md`、`AGENTS.md`、`docs/`（除下列） | `client/`、`config/live/`、`artifacts/runs/live-*`、`records/`、`docs/sync/LOCAL.md`、`docs/live/`、`docs/LIVE_V1_1_AUDIT.md`、`docs/E_PHASE_RUNBOOK.md`、`docs/analysis/` |

**服务器不要改** `client/`、`config/live/`；**本地不再改** `engine/`、`hsbg/` 及训练/评测代码。

## 1. 服务器P0（本周内，不涉及训练）

1. `git pull --rebase`，确认拿到 `docs/ARCHITECTURE.md`、`AGENTS.md` 更新、`docs/analysis/` 三份报告。
2. 在**计算节点**（不要在登录节点）跑一次全量回归并记录：
   ```bash
   srun --pty bash      # 或提交一个短作业
   python -m pytest tests/ -q --ignore=tests/test_live_supervisor.py \
       --ignore=tests/test_live_execution_trace.py \
       --ignore=tests/test_live_resource_migration.py \
       --ignore=tests/test_live_simulator_check.py
   cargo test --release --manifest-path engine/Cargo.toml -p hsbg-engine
   python tools/verify_workspace.py
   ```
   把"通过/失败 + 失败清单"写进 `docs/sync/SERVER.md` 新开的一节（服务器自己的文件）。
3. **建立引擎发版记录**：新建 `docs/sync/ENGINE_RELEASES.md`（**服务器写**），每次 `engine/` 有变更就追加一行：
   `日期 | git commit | source_hash | db_info(=N_CARDS, client_version, server_patch) | 变更摘要`。
   理由：本地需要用它决定"要不要重新构建 Windows 轮子"，这是两端唯一会不一致的产物。

## 2. 服务器P1（两周内）

4. **接收实机日志**：本地会把 `live-<date>-<seq>.tar.gz`（含 Power.log / 结果屏 / summary）放到
   `artifacts/imports/live-<date>/`。请用它做两件事：
   - 校验模拟器规则（尤其：罐装宝石触发次数、饰品报价与买不起的场景、Kiri 是否出现）；
   - 把发现写进 `SERVER.md`，需要改规则的再走 `DECISIONS.md`。
5. **文档归位**：模拟器/训练类文档（`docs/SIMULATOR_AUDIT_2026-11-09.md`、`docs/ELO_EVAL.md`、
   `docs/EVAL_DUEL.md`、`docs/nus/**`）今后由服务器维护；如需重写，请在服务器侧完成并 push。

## 3. 明确不做的事

- 不要为了"对齐 HSBRSIM"去改规则：`docs/analysis/ENGINE_RULES_FINAL_DECISION.md` 已判定**不改**。
- 不要把 `client/` 或 `config/live/` 纳入服务器维护范围（实机只在本地跑）。
- 不要在登录节点跑 `pytest`/`cargo`/torch（1 GiB 限制）。

## 4. 验收标准

- `SERVER.md` 有一条"2026-11 全量回归"记录：Python/Rust/verify_workspace 三项结果齐全；
- `docs/sync/ENGINE_RELEASES.md` 存在且首行是当前 `c4c71014e3d41760` 的记录；
- `git status` 干净（除运行中作业的冻结副本）；
- `docs/sync/manifests/server.json` 用 v2 脚本重新生成并提交。

## 5. 待确认问题（请在 SERVER.md 里回答）

1. 服务器当前构建的扩展 `build_info().source_hash` 是多少？与 `c4c71014e3d41760` 是否一致？
2. 服务器上有没有可用的 Windows 构建机（用于产出本地轮子）？若没有，只能由本地部署构建（需要 MSVC）。
3. `pytest` 全量在计算节点跑一次大约多久、有多少已知失败？（本地有 22 个预存在失败，怀疑是测试落后于代码。）
