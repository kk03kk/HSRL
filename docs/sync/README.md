# docs/sync —— 两个 Agent 的异步协作区

状态：**当前**。规则见 `docs/SYNC_MODEL.md` §5。

## 文件所有权（不要改写对方的文件）

| 文件 | 写者 | 状态 |
|---|---|---|
| `LOCAL.md` | 本地 Agent | draft（2026-11-10 首次盘点） |
| `SERVER.md` | 服务器 Agent | **pending**（等待服务器盘点） |
| `DECISIONS.md` | 双方追加 | active（只追加，不改历史条目） |
| `manifests/local.json` | `tools/sync_manifest.py` | 已生成（2026-11-10） |
| `manifests/server.json` | `tools/sync_manifest.py` | **缺失**（等服务器跑同一脚本） |

## 一轮交互怎么做

```text
本地: 编辑 LOCAL.md / 追加 DECISIONS.md
      python tools/sync_manifest.py --side local  --out docs/sync/manifests/local.json
      git add docs/sync && git commit -m "sync(local): ..." && git push
服务器: git pull --rebase
      读 LOCAL.md + DECISIONS.md → 盘点服务器
      python tools/sync_manifest.py --side server --out docs/sync/manifests/server.json
      编辑 SERVER.md / 追加 DECISIONS.md
      git add docs/sync && git commit -m "sync(server): ..." && git push
本地: git pull --rebase → 复核 SERVER.md → 把确认过的条目标 confirmed
```

## 判读两份 manifest

`manifests/{local,server}.json` 结构相同，逐字段比较：

| 字段 | 含义 | 期望 |
|---|---|---|
| `git.head` / `git.dirty` | 该端代码版本 | 同一 `head`；`dirty` 应为 false 或已说明 |
| `trees.*.files/bytes` | 各源码树规模 | 除 `client/`（本地多实机证据）外应一致 |
| `pins.*.sha256` | 版本锚点 | 同一权重/卡表/配置必须同 SHA |
| `artifacts.*` | 大目录规模 | 只做“有没有/多大”的对照，不要求相等 |
| `problems` | 关键路径存在性 | 全部 `exists: true` |
| `env` | 解释器/依赖版本 | 记录差异即可（两端环境本就不同） |

## 当前状态板

| 事项 | 本地 | 服务器 | 结论 |
|---|---|---|---|
| 代码基线 | `e4d4dfc` + 未提交改动 | 待确认 | 待对齐 |
| V1.1 权重 SHA | `68a8a587…0848e` | 待确认 | 待对齐 |
| 引擎指纹 | `c4c71014e3d41760`（`config/engine-upstream-lock.json`） | 待确认 | 待对齐 |
| 战团库 `library-v1` | 有（39 cohort，3.4 GB） | 待确认 | 待对齐 |
| 评测报告 `leaderboard-v1` | **无** | 待确认（应在服务器） | 待对齐 |
| 实机证据 | 有（本地唯一） | 不应有 | 已明确 |
| 未提交改动 | 57 个条目（昨夜审计 + D 阶段） | — | **需用户决定是否 push** |
