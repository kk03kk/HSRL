# SERVER —— 给 NUS 服务器 Agent 的交接任务书

```yaml
status: pending        # 由服务器 Agent 改成 draft → confirmed
owner: server
updated: 2026-11-10 18:10   # 服务器 Agent 处理后改成自己的时间
base_commit: 待填
```

> 这份文件是**本地 Agent 写给服务器 Agent**的一次性任务书。规则见 `docs/SYNC_MODEL.md`，
> 状态板见 `docs/sync/README.md`。请不要改写 `LOCAL.md`，只在 `SERVER.md` 与
> `DECISIONS.md` 里写你的结论。

## 0. 背景（你需要的全部上下文）

- 2026-11-09/10 完成了一轮 V1.1 真机测试与审计，本地侧已完成 D 阶段工程改造（批处理监督器、
  投递对账、名次竞态、赛后续局、重复手势闸门）。
- 本地 Agent 无法查看服务器；服务器 Agent 无法查看本地。两边只能通过 GitHub 上的
  `docs/sync/` 交换信息。
- 用户要求：**先盘点、后规范**。不要在没有盘点证据的情况下修改三端结构。

## 1. 你要做的事（按顺序）

### 1.1 拉取并阅读（只读）

```bash
cd ~/projects/HSAI
git fetch origin && git status --porcelain=v1 -b
git log --oneline -5
```

阅读顺序：`README.md` → `AGENTS.md` → `docs/SYNC_MODEL.md` → `docs/sync/LOCAL.md`
→ `docs/sync/DECISIONS.md` → `docs/INVENTORY_LOCAL.md`。

> 如果 `docs/SYNC_MODEL.md` 不存在，说明本地改动还没 push；先看 `git log` 确认，
> 并在 `SERVER.md` 里记录 `blocked: 本地文档未同步`。

### 1.2 服务器盘点（登录节点安全；**不要**跑 torch/训练）

```bash
python tools/sync_manifest.py --side server --out docs/sync/manifests/server.json
```

该脚本只用标准库、只做统计与少量 SHA256，不会导入 torch/numpy，适合登录节点。

同时收集以下事实（命令可按需裁剪，避免 GB 级遍历）：

```bash
du -sh artifacts/* records/* 2>/dev/null
ls -la artifacts/models/ | head -50
find artifacts/models -maxdepth 2 -name manifest.json -printf '%h\n' 2>/dev/null | head -50
squeue -u "$USER" ; sacct -S 2026-11-01 -X -o JobID,JobName,State,Elapsed,AllocTRES%40 | tail -40
sha256sum artifacts/models/*/model.pt 2>/dev/null | head -20
grep -o '"source_hash"[^,]*' config/engine-upstream-lock.json | head -3
```

### 1.3 把结果写回来

1. 编辑 `docs/sync/SERVER.md`（把下面的模板填满，`status: draft`）。
2. 在 `docs/sync/DECISIONS.md` **追加**你的决议（编号接着写，不要改旧条目）。
3. `git add docs/sync && git commit -m "sync(server): inventory" && git push`。

**禁止**：强推、改 `LOCAL.md`、在服务器上执行历史未决输入账本、删除任何产物、启动训练/评测。

## 2. 必须回答的问题（来自 `LOCAL.md` §5）

| # | 问题 | 证据要求 |
|---|---|---|
| 1 | `git rev-parse HEAD` / 是否干净 / 是否已含 D 阶段改动 | 命令输出 |
| 2 | `artifacts/models/` 下每个模型目录的 `model.pt` SHA256（至少 final `68a8a587…`、30M `52bcddbb…`、240M `caf0fe56…`、`v9786` `988fdc7f…`） | `sha256sum` 输出 |
| 3 | `artifacts/models/leaderboard-v1/` 是否存在；`reports/20261109-live-v1.json` 与 `manifest-2026-11-09.json` 是否在 | 目录清单 + 文件 SHA |
| 4 | 战团库有几份、多大、cohort 数是否等于本地 39 | `du -sh` + `wc -l index.csv` |
| 5 | `engine/` 源码指纹是否仍为 `c4c71014e3d41760`；是否重新编译过 37.0.2 | `config/engine-upstream-lock.json` 摘录 + 构建时间 |
| 6 | 已完成/进行中/可清理的 Slurm 作业与产物目录 | `sacct` 摘要 + 目录清单 |
| 7 | 是否残留旧项目路径或重复代码副本（`~/projects/hearthstone-rl`、`AutoTavern`、`HSBG`…） | `ls` + `du -sh` |
| 8 | `artifacts/`、`records/` 总占用与磁盘余量 | `du -sh` + `df -h ~` |

## 3. 建议的 SERVER.md 主体（复制后填写）

```markdown
## 1. 计费/资源现状
- 登录节点磁盘：df -h 输出；artifacts 总占用；records 总占用

## 2. 代码版本
- HEAD：…；dirty：…；与本地 e4d4dfc 的关系：…

## 3. 模型与权重
| 目录 | 大小 | model.pt SHA256 | 用途 | 是否最新 |
|---|---|---|---|---|

## 4. 战团库 / 数据集 / 评测产物
| 资产 | 路径 | 大小 | 版本锚点 | 可否清理 |
|---|---|---|---|---|

## 5. 作业与产物
| 作业 | 状态 | 输出目录 | 是否正式结果 |
|---|---|---|---|

## 6. 与本地 manifest 的差异
（列出 trees/pins 里不一致的字段，以及判断：正常差异 / 需要同步 / 需要清理）

## 7. 我建议的三端分工调整
（只写建议，不动手）
```

## 4. 验收标准（本地 Agent 会按这个复核）

1. `docs/sync/manifests/server.json` 存在，`pins` 里能对上的条目与本地**同 SHA**，
   对不上的条目有文字解释。
2. §2 的 8 个问题都有命令输出或明确写“无法确认 + 原因”。
3. `SERVER.md` 的 `status` 至少到 `draft`，`base_commit` 已填。
4. `DECISIONS.md` 至少追加 1 条决议（含日期、侧别、结论）。
5. 没有删除/移动服务器上的任何产物，没有启动训练/评测。

## 5. 之后会发生什么

本地 Agent 拉取你的提交 → 逐条核对 → 把 `status` 改 `confirmed` → 双方在
`DECISIONS.md` 里冻结三端规范 → 用户决定是否按规范迁移/清理文件。
