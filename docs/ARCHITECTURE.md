# HSAI 架构：服务器统一开发，本地专注实机（2026-11-10 起）

状态：**已生效（文档与所有权部分）**，代码/流程部分见 §3 的分步迁移。
决策依据：`docs/analysis/` 三份审计报告 + 用户 2026-11-10 指示。

---

## 0. 判断：这个调整合理，且能解决已经发生过的真实问题

**合理的理由**

1. 模拟器规则只有一个真源。此前已经出现"本地跑 `c7b5f27d`、服务器源码 `c4c71014`"的错位，
   本地任何模拟器结论都可能与服务器不一致——这正是本轮架构要消除的。
2. 本地机器的定位很清晰：它**唯一不可替代**的能力是"对真实客户端读屏+操作"。
   模拟器/训练/评测本来就在 NUS 上跑，本地重复维护只会产生两套环境。
3. 训练与评测是重活（H100 + 大库），本地无 GPU、无数据库、无 Slurm，重复建设没有收益。
4. 实机链路对模拟器的依赖比想象小：`client/` 只用到 3 个纯常量模块 + 1 个原生扩展（见 §2.3），
   完全不需要本地编译或修改模拟器。

**风险与缓解**

| 风险 | 缓解 |
|---|---|
| 本地需要 `hsbg_engine` 的 Windows 轮子（服务器是 Linux，不能替本地出 Windows 产物） | 保留"**部署构建**"例外：仅在服务器升级引擎版本时，用固定脚本在本机构建一次并写入版本锚（§3-S3） |
| 本地不再跑模拟器测试，改错了没人发现 | 模拟器测试全部在服务器跑；本地只跑 `tests/test_live_*.py`（实机链路） |
| 两端重复开发同一条规则 | 单写权 + `client/tools/check_ownership.py` 自检 + PR 级清单（§2.1） |
| 真实游戏日志无法用于规则验证 | 新增"日志回传"通道，只走 tar.gz + sha256，不进 Git（§2.4） |

---

## 1. 职责矩阵（唯一写权）

| 层 | 服务器 `~/projects/HSAI` | 本地 `D:\HSAI` | GitHub `kk03kk/HSRL` |
|---|---|---|---|
| 模拟器内核 `engine/` | **写**（规则/编译/测试/版本） | 只读镜像 | 同步 |
| 动作空间/观测 `hsbg/env/` | **写** | 只读（运行时依赖） | 同步 |
| 训练/损失/采样 `hsbg/az/` | **写** | 不参与 | 同步 |
| 评测 `hsbg/eval/`、`workflows/`、`deploy/` | **写** | — | 同步 |
| 卡表/数据 `data/` | **写** | 只读 | 同步 |
| 测试 `tests/` | **写**（除 `test_live_*`） | 只跑 `test_live_*` | 同步 |
| 实机桥接 `client/` | 只读 | **写** | 同步 |
| 实机配置 `config/live/` | 只读 | **写** | 同步 |
| 实机证据 `artifacts/runs/live-*`、`records/evidence/` | 接收（导入件） | **写** | ❌ 不进 Git |
| 文档 `docs/`（除下列） | **写** | 只读 | 同步 |
| `docs/sync/LOCAL.md`、`docs/live/**`、`docs/LIVE_V1_1_AUDIT.md`、`docs/E_PHASE_RUNBOOK.md` | 只读 | **写** | 同步 |
| `docs/analysis/**`（差异审计） | 可写 | 可写 | 同步 |
| `AGENTS.md`、`README.md` | **写** | 只读 | 同步 |
| 模型权重 / 战团库 / 数据集 | **写** | 只读子集 | ❌ |

**一句话**：**规则与学习系统只在服务器改；实机桥接只在本地改；GitHub 只同步代码与文档。**

---

## 2. 边界怎么落地

### 2.1 单写权自检

本地改动前/提交前跑：

```powershell
python client/tools/check_ownership.py           # 只看 git status
python client/tools/check_ownership.py --strict  # 有越界改动时退出码 1
```

它列出所有"本地无权修改"的路径（`engine/** hsbg/** workflows/** deploy/** data/** tests/**（除 test_live_*）
tools/** pyproject.toml requirements.txt`）。越界改动**不会自动回滚**，只是明确提示"这应该由服务器改"。

### 2.2 文档头

每份跨端文档头部保留 `status / owner / updated / base_commit`；`owner: server` 的文档本地只读。

### 2.3 本地运行时依赖（保持最小）

实机链路实际只需要：

| 依赖 | 用途 | 来源 |
|---|---|---|
| `client/**`（含 `model.py`/`encoder.py`/`actor.py`） | 读屏、决策、输入、回执 | 本地自有 |
| `hsbg/env/obs_std.py`、`hsbg/env/action_space_std.py` | 观测/动作布局常量 | 只读镜像 |
| `hsbg/az/train/common.py` | 245/3227/124 等固定常量（numpy） | 只读镜像 |
| `hsbg_engine` 原生扩展（Windows wheel） | 卡表/契约校验 | **部署构建**，见 S3 |
| `numpy torch pillow rapidocr onnxruntime opencv` | 推理与视觉 | 各端自建 |

不需要：Rust 编译、`hsbg/az` 训练栈、评测器、`workflows/`、Slurm。

### 2.4 真实日志回传（不进 Git）

```powershell
# 本地：打包一局的证据（Power.log + 截图 + decisions trace）
python client/tools/pack_evidence.py --run artifacts/runs/live-v1.1-d-phase-006 --out D:\xfer
# 产出 live-<date>-<seq>.tar.gz + .sha256（默认只含 Power.log / 结果屏 / summary，不含模型）
```

服务器侧解包到 `artifacts/imports/live-<date>/`（与现有 `artifacts/imports/nus-*` 风格一致），
用于规则验证与问题排查。传输方式任选（scp/网盘），**不提交到 Git**。

### 2.5 引擎版本锚

`config/live/engine.json` 记录"本地实机链路期望的引擎构建"：

```json
{"schema": "hsai.live-engine/v1", "expected_source_hash": "c4c71014e3d41760",
 "db_info": [3983, "37.0.2.256589", "37.0.2"], "built_on": "server", "wheel_sha256": "..."}
```

`doctor` 读取它并与 `hsbg_engine.build_info()` 对比；不一致时**报告为警告并写进 `health.json`**，
默认不阻断 `autoplay`（避免影响正在进行的实机验收），需要严格时把 `require_match` 设为 `true`。

---

## 3. 迁移方案（分 5 步，S1 本轮已完成）

| 步 | 内容 | 负责端 | 状态 |
|---|---|---|---|
| **S1** | 文档与所有权：本文件、`AGENTS.md`、`docs/SYNC_MODEL.md`、`docs/README.md`、`docs/sync/LOCAL.md`；根目录清理；`check_ownership.py` | 本地 | ✅ 本轮完成 |
| **S2** | 本地瘦身：停止在本地跑模拟器/训练测试；`tests/` 只保留/运行 `test_live_*`；不再本地编译 Rust（除非 S3 部署构建） | 本地 | ✅ 规则已定，本轮只做文档 |
| **S3** | 引擎版本锚 + 部署构建脚本（`client/tools/build_engine.ps1`）：服务器 bump 引擎 → 本地一条命令重建 wheel → 写 `engine.json` | 本地（脚本）+ 服务器（发版） | ⏳ 待做，见交接 |
| **S4** | 日志回传通道：`pack_evidence.py` + 服务器侧 `artifacts/imports/live-*` 约定 | 双方 | ⏳ 待做 |
| **S5** | 服务器侧验收：`main` 上 `pytest`/`cargo test`/`verify_workspace` 全绿；训练与评测按新边界运行 | 服务器 | ⏳ 见交接 |

**兼容性**：本调整不改任何规则/动作/观测/模型，引擎指纹保持 `c4c71014e3d41760`；
S3 的"构建一次 Windows wheel"只是把已有产物对齐到源码，不改变训练语义。
**回退**：S1/S2 全部是文档与新增脚本，删除即可回到旧流程。

---

## 4. 与既有结论的关系

- 引擎规则审计结论不变：**不修改模拟器**（`docs/analysis/ENGINE_RULES_FINAL_DECISION.md`）。
- 本地扩展仍是 11-06 构建（`c7b5f27d`）；按 S3，它应由服务器发版 + 本地部署构建对齐，而不是本地重复审计。
- 训练/评测的唯一执行地仍是 NUS；本地不再产生任何训练或评测结论。
