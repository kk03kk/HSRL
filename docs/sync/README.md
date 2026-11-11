# docs/sync —— 两个 Agent 的异步协作区

状态：**当前**。规则见 `docs/SYNC_MODEL.md` §5。

## 文件所有权（不要改写对方的文件）

| 文件 | 写者 | 状态 |
|---|---|---|
| `LOCAL.md` | 本地 Agent | confirmed（2026-11-10 首次盘点） |
| `SERVER.md` | 服务器 Agent | confirmed（2026-11-11 追加 §14） |
| `DECISIONS.md` | 双方追加 | active（只追加，不改历史条目） |
| `manifests/local.json` | `tools/sync_manifest.py` | 已生成（`ed809ab`，待作业后重跑） |
| `manifests/server.json` | `tools/sync_manifest.py` | 已生成（`02ecc57`，待作业后重跑） |

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

## 当前状态板（2026-11-11 23:25 更新）

| 事项 | 本地 | 服务器 | 结论 |
|---|---|---|---|
| 代码基线 | 本地 manifest 待刷新 | 服务器已提交审计/E1/交接（本次 push 后取新 HEAD） | 本地 `pull --rebase` 后重跑 `sync_manifest.py` |
| 36h 训练 | 无 | `1102669` COMPLETED，final `v14858`，累计 `1,552,463,499` | 已结束；final 资产已保存 |
| E1 评测 | 无 | r2 正式报告（2,688 局，协议 `0fbf28d2…`），final 点估计冠军 1086.53 | **只跑一轮，不做挑战赛**（D-027） |
| 实机模型 | 待更新 `config/live/model.json` | live 包 `league-final-14858-live.tgz`，SHA `32ccdabd…` | 交接文档 `SERVER_TO_LOCAL_20261111.md`；先 E-0 canary |
| 引擎指纹 | `c4c71014e3d41760` | 同 | **一致**；无需重建 wheel |
| 战团库 | `library-v1`（实机） | `league-eot-v1.1-20261109`（训练） | 按用途分开（D-012） |
| 未提交改动 | 待 pull | 本次审计代码/文档已准备 push | push 后两端重跑 manifest |

## 下一步（本地）

1. `git pull --rebase` 拿到本次审计/E1/交接提交。
2. 按 `docs/sync/SERVER_TO_LOCAL_20261111.md` 下载/解包 live 包，更新
   `config/live/model.json`，跑 doctor + E-0 canary。
3. 回传 E-0 的 `summary/health/result/actions` 与结果屏截图；再决定 E-1。
4. 本地复核 `SERVER.md` §15–16 与 D-026/D-027 后，把自己的状态改为 confirmed。

## 下一步（服务器）

1. 本次 push 后重跑 `tools/sync_manifest.py --side server`，等本地复核。
2. 下一轮长训只做准备：`config/league-stage2.json`、
   `deploy/slurm/league_stage2.sbatch`、`artifacts/models/stage2-proposal-20261111/`；
   **未启动，等用户确认**。
3. 不再跑第二轮 E1、不做冠军挑战赛；只有 invalid/工程错误才重跑并先请示。
