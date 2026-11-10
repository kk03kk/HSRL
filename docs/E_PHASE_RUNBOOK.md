# E 阶段执行手册（本地实机连续验收）

状态：**计划**（未授权执行）。触发条件：用户明确说"开始 E 阶段"。
前置：D 阶段已完成（`docs/LIVE_V1_1_AUDIT.md` §7）；本地代码 = `main` 最新；游戏已登录并停在大厅。

## 0. 原则

- 准确性优先：任何 S0 立即停输入，不重试未知投递。
- 效率其次：S1 只在回合末/局末处理，不为单点异常废掉整局。
- 一局一目录，证据不覆盖；失败用新目录重跑。

## 1. E-0 · 单局 canary（约 25 分钟）

目的：验证 D 阶段两个"只有单测"的分支在真实对局中的行为。

```powershell
git pull --rebase
python -m pytest tests/test_live_*.py -q          # 期望 58 passed
python client/live_245.py doctor --game-dir D:\Hearthstone --infer
python client/live_245.py autoplay --game-dir D:\Hearthstone `
    --out artifacts/runs/live-e0-$(Get-Date -Format yyyyMMdd)-001 `
    --seconds 3600 --games 1
```

**观察点（每 5 分钟看一次 `health.json`）**

| 事件 | 期望 | 说明 |
|---|---|---|
| `retired_stale_inflight` | 不出现（干净开局） | 若出现，说明上一局账本被继承，属于预期修复 |
| `rejected_reconciliation` → `expired_unsent_input` proof=`server_did_not_take_still_legal` | ≤1 次且对局继续 | 关键新分支；出现时保存该局目录 |
| `receipt_overdue` | 0–3 次 | >3 次记 S1 |
| `halted` | 0 | 出现即失败 |
| `duplicate_gesture_suppressed` | 只出现在"已被回执确认"的手势后 | 记录次数与手势类型 |

**验收**：S0=0；S1≤2；`result.json` 的 `placement` 与游戏内结果屏一致；`decisions.jsonl` 行数 = `decisions-full` 决策行数。

失败处理：保留目录 → 写 `records/research/e0-<date>.md`（症状/证据/推断）→ 离线修 → 重跑 E-0。

## 2. E-1 · 连续三局（约 75 分钟）

```powershell
python client/live_245.py autoplay --game-dir D:\Hearthstone `
    --out artifacts/runs/live-e1-$(Get-Date -Format yyyyMMdd)-001 `
    --seconds 3600 --games 3 --max-failures 2 --max-same-failure 1
```

**验收（全部满足才算通过）**

1. 3 局连续、**0 次人工介入**、0 次整批中止。
2. 每局 S0=0、S1≤2；`halted`=0。
3. 三局 `result.json` 名次全部与结果屏一致。
4. 局间自动续局成功（无需手动点"开始"）。
5. 重复手势=0；`cancelled_before_press` 每局 ≤20。
6. 后期（回合 ≥10）动作速率 ≥6 有效动作/分钟，动作 p95 ≤800 ms。

任一条不满足：停批、保留现场、离线归因，不打第三批。

## 3. 记录与回收

- 每局结束把 `summary.json`/`performance.json`/`result.json` 摘要贴进 `records/research/e-phase-<date>.md`。
- `artifacts/runs/live-*` 全量保留；用 `python tools/audit_live_run.py --run <dir>` 复查。
- 通过后：把结论写入 `docs/PROJECT_STATE.md`，并在 `docs/sync/LOCAL.md` 记录"E 阶段通过"。

## 4. 停止条件（任一即停）

- 出现 S0（未知投递、越界点击、重复执行、状态损坏）。
- 同一 S1 签名连续 2 局出现。
- 游戏客户端崩溃/掉线 2 次以上。
- 用户叫停。

## 5. 已知风险

- 英雄选择悬停取消（历史 2–12 次/局）仍在，属 S1 噪声。
- `other` 等待分类仍按上一条状态归因，不能当真实阻塞时长。
- 若客户端掉线（长时间无人操作），本批会以 S3 结束——这是正确行为，不要绕过。
