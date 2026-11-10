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

## 当前状态板（2026-11-10 19:55 更新）

| 事项 | 本地 | 服务器 | 结论 |
|---|---|---|---|
| 代码基线 | `711f604` 干净 | `224cb49` + 联赛冻结副本 | **已对齐**（本地领先 1 个 sync 提交；D 阶段 `d10e352` 两边都有） |
| V1.1 权重 SHA | `68a8a587…0848e` | `68a8a587…0848e` | **一致**（另有服务器 `trainer.pt` `dfcb59ce…`，本地无） |
| 引擎指纹 | `c4c71014e3d41760` | 同（旧榜为 `c7b5f27d`，已归档） | **一致**；旧榜降级为历史（D-011） |
| 战团库 | `library-v1` 39 cohort（实机用） | `league-eot-v1.1-20261109` 4 源（训练用） | **按用途分开**（D-012），不复制 |
| 评测报告 | 无（本地不需要） | `1099277` 1,792 局、`1107182` 3,072 局 | **当前引擎有效结果只在服务器** |
| 运行中作业 | 无 | `1102669` 36h 联赛 + `1097487` allocation | **冻结**，不要打扰（D-013） |
| 实机证据 | `artifacts/runs/live-*`（唯一） | 不应有 | 已明确（D-007） |
| 清单一致性 | v2 清单已提交 | v1 清单（缺 `git_blob`） | **待服务器重跑 v2**（D-017） |
| 未提交改动 | 0 | 11 个联赛冻结副本 | 作业结束后再定 |

## 下一步（服务器）

```bash
git pull --rebase                       # 拿到 711f604 之后的本地复核与 v2 脚本
python tools/sync_manifest.py --side server --out docs/sync/manifests/server.json
python tools/sync_compare.py docs/sync/manifests/local.json docs/sync/manifests/server.json
```

期望：pins 全部 `OK`（四个 CRLF 文件不再假差异），`artifacts/*` 只显示量级差异，
`result: consistent`。若仍有硬漂移，在 `SERVER.md` 追加一节说明。

### 36h 联赛（`1102669`）结束后的顺序

1. `squeue`/`sacct` 确认 `1102669` 已是 `COMPLETED`，把 36h 产物写进 `SERVER.md`（最新 version、累计决策、最终 SHA）。
2. 把冻结工作副本（1 个已跟踪修改 + 10 个未跟踪）整理成一次正常提交回到 `main`，或明确丢弃；不要长期留 dirty。
3. 按 D-015 执行服务器清理（`artifacts/exports` 889 MB、`artifacts/local/v1.1-*.tgz` 333 MB、旧面板约 185 MB），
   在目录留 `cleanup-<date>.txt`；权重、联赛库、`records/` 原始证据不删。
4. 用**当前引擎** `c4c71014e3d41760` 重跑 live-v1 全榜（旧 `46e7013d…` 是 `c7b5f27d` 引擎，见 D-011），
   协议 SHA 变化时在 `DECISIONS.md` 追加新条目。
5. 重新生成 `manifests/server.json` 并 push，等本地复核。
