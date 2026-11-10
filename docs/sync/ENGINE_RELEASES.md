# ENGINE_RELEASES —— 引擎发版记录

```yaml
status: current
owner: server
updated: 2026-11-10 22:20 +0800
base_commit: 402480b
```

服务器拥有写权。每次 `engine/` 有变更就追加一行；本地据此决定是否重新构建 Windows 轮子（S3 部署构建）。

列：`日期 | git commit | source_hash | db_info(=cards, client_version, server_patch) | 变更摘要`

| 日期 | git commit | source_hash | db_info | 变更摘要 |
|---|---|---|---|---|
| 2026-11-10 | `402480b` | `c4c71014e3d41760` | `[3983, "37.0.2.256589", "37.0.2"]`（`N_CARDS=3984`） | 当前服务器构建。2026-11-09 审计修复 `Jar o' Gems` 每战斗触发一次后的指纹；`build_info().source_hash=c4c71014e3d41760`，`build_is_current=True`；`engine/python/hsbg_engine/_core.abi3.so` 构建于 2026-11-09 08:55。训练/评测/回归均使用此二进制。 |
| 2026-11-09（审计前） | — | `c7b5f27d6f40d1d2` | `[3983, "37.0.2.256589", "37.0.2"]`（`N_CARDS=3984`） | 本地 Windows 扩展仍是此版本（2026-11-06 构建）。不用于当前服务器训练/评测；旧 live-v1 报告/榜单锚定此指纹，已降级为历史证据（D-011）。 |

## 备注

- `source_hash` 由 `engine/crates/hsbg-engine/src`、`engine/crates/hsbg-engine-py/src` 与编译期卡表计算；换数据或改源码后必须重编译并重跑回归。
- `db_info()` 实测返回 `(3983, '37.0.2.256589', '37.0.2')`；`N_CARDS=3984`（含空位）。
- 当前 `config/live/engine.json` 尚未建立；S3 部署构建时由本地写入 `expected_source_hash` 与 wheel SHA。
