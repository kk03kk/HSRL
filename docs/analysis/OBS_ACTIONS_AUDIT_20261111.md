# 观测编码与动作规则：HSAI × HSBRSIM 深度审计（2026-11-11）

范围：`engine/crates/hsbg-engine/src/{obs.rs, actions.rs, types.rs, env.rs, snapshot.rs}` + Python 镜像
`hsbg/env/{obs_std,action_space_std}.py` + 实机编码器 `client/hsrl_agent/live/encoding.py`。
性质：客观差异 + 哪方合理；**未修改任何代码**。

---

## 0. 结论

| 模块 | 结论 |
|---|---|
| **观测编码** | 两边 `obs.rs` **逐字节相同**（SHA256 `906b2612ffc6…`，425 行）。Python 镜像常量与 Rust 一致。**没有差异可判**；真正未审的风险在"实机投影填值"（§3）。 |
| **动作规则** | `actions.rs` 有 **2 处、20 行**差异，都围绕同一个问题：**挂起菜单出现"零合法动作"时怎么办**。判 **HSBRSIM 的写法更安全**（§2.3）。 |

---

## 1. 观测编码（obs.rs）

### 1.1 文件级证据

| 文件 | HSAI | HSBRSIM | 结论 |
|---|---|---|---|
| `obs.rs` | `906b2612ffc6` | `906b2612ffc6` | **完全相同** |
| `types.rs` | `1b3bb9e4dd33` | `1b3bb9e4dd33` | 相同 |
| `env.rs` | `99dfdd9dfe92` | `99dfdd9dfe92` | 相同 |
| `snapshot.rs` | `a4cd22b23f14` | `a4cd22b23f14` | 相同 |
| `combat.rs` | `101b429ab3c8` | `101b429ab3c8` | 相同 |

### 1.2 布局常量（Rust ↔ Python 逐项核对）

| 常量 | `obs.rs` | `obs_std.py` | 一致 |
|---|---|---|---|
| `MINION_F` | `8 + Kw::NAMES.len() + Races::COUNT` = 8+12+11 = **31** | `8 + 12 + 11` | ✅ |
| `ITEM_F` | `9 + MINION_F` = **40** | `9 + MINION_F` | ✅ |
| `GLOBAL_F` | `42 + Races::COUNT` = **53** | `42 + 11` | ✅ |
| `N_QUESTS` / `QUEST_F` | 3 / 2 | 3 / 2 | ✅ |
| `N_SECRETS` | `MAX_SECRETS` = 5 | 5 | ✅ |
| `OPP_F` / `N_OPP` | 9 / 7 | 9 / 7 | ✅ |
| `OPP_BOARD_F` | `N_OPP*N_BOARD*MINION_F` | 同 | ✅ |
| `PENDING_KINDS` | 9 | 9 | ✅ |
| `OBS_FLOATS` / `OBS_CARDS` | 3227 / 124 | 3227 / 124 | ✅ |

`obs_std.py` 与 `obs.rs` 唯一差异是**注释里的数据版本文案**（HSAI 写 37.0.2.256589，HSBRSIM 写 37.0.0.255790），常量与字段顺序完全相同。

### 1.3 实机编码器的结构性保障

`client/hsrl_agent/live/encoding.py` 不手写偏移量，而是：

```python
from hsbg.env import obs_std as O
if set(O.GLOBAL_FIELD_NAMES) - set(g): raise Blocked('global fields missing')
f=[float(g[k])/O.GLOBAL_FIELD_SCALE.get(k,1.) for k in O.GLOBAL_FIELD_NAMES]
```

即**布局由 `obs_std` 驱动**，字段缺失会直接 `Blocked`。→ 布局漂移在结构上被阻止。

---

## 2. 动作规则（actions.rs）

### 2.1 相同部分

动作分段与编码完全一致：`N_SHOP=8 / N_BOARD=7 / N_HAND=10 / N_CHOICE=16 / N_POS=8`，
`A_END=0 … A_DARK_GIFT=6`，`N_ACTIONS=245`（运行时 `hsbg_engine.N_ACTIONS=245` 实测）。
`encode/decode`、`option_legal`、菜单/目标/分支授权逻辑相同。

### 2.2 差异（全部 2 处，20 行）

**差异 A — `apply()` 之后是否主动清理买不起的饰品单**

| | HSAI（`actions.rs:151-158`） | HSBRSIM |
|---|---|---|
| 行为 | 每次动作**成功**后，遍历 8 个座位调用 `drop_unaffordable_trinket_offers(seat)` | 无此调用（只在 `take_trinket` 之后与 `cancel()` 内清理） |

**差异 B — 待选菜单零合法选项时是否放行 Cancel**

| | HSAI（`actions.rs:227-231`） | HSBRSIM（`actions.rs:219-231`） |
|---|---|---|
| 行为 | `m[A_CHOOSE+i]=option_legal(...)`；`m[A_CANCEL]=top.cancellable()` | 同前，但额外累计 `any`，然后 `m[A_CANCEL]=top.cancellable() \|\| !any` |

### 2.3 哪方合理

**判 HSBRSIM 更安全，理由是可证据化的，不是偏好：**

1. 两处针对的是**同一个失效模式**：`Pending::Trinket{free:false}` 挂起后金币被花光 → 四个选项都不可选、
   而饰品单 `cancellable()` 恒为 false → `legal_mask` 全 False，但 `needs_action=true` → 整局卡死。
   HSBRSIM 的提交 `063fe69` 记录了 **194 个现场全复现**；我们的 `docs/B05` 同类崩溃记录也出现过
   （`存在无合法动作的行（mask 全 False）`）。
2. **HSAI 的写法是"预防"**：只要清理时机漏掉任何一条路径（例如某个回合末效果在 `apply()` 之外创建报价），
   空掩码仍会出现。我本轮用只读探针 `artifacts/local/engine-audit/probe_trinket_mask.py` 实测：
   把已挂起的饰品单金币改为 0 后，HSAI 的 `legal_mask` **确实全 False**（`needs_action=True`）；
   同一状态在 HSBRSIM 的写法下至少有 `A_CANCEL` 一个合法位。
3. **HSBRSIM 的写法是"兜底"**：`|| !any` **只在"一个选项都不可选"时**才生效，
   **不改变任何正常局面的掩码**（正常局面 `any=true`，结果与 HSAI 完全一致）。
4. 两者可共存：HSAI 的主动清理保留，加上 `|| !any` 只是多了一层最后防线。

**副作用评估（若采用）**：新增的合法动作是 `A_CANCEL`（编号 7，早已存在，不是新动作）；
我们的训练配置 `forbid_cancel=1` 会屏蔽 Cancel，但当它是唯一合法动作时必须放行——这与实机
`bind_actions` 的既有约定一致（"Cancel 只在它是唯一授权动作时绑定"）。
→ **不改变观测、不改变动作编号、不改变正常掩码**，只把"死局"变成"可退出"。

### 2.4 相关但不同的差异（`state.rs`，仅记录）

`state.rs` 另有 22 行差异（HSAI 独有 hand-link 清理 `unlink_removed_hand_card` /
`prune_removed_hand_links` / 不变量断言）。它不改变 245 个动作的语义，但会改变**手牌索引的合法性**：
悬空 link 若被后续效果按 id 命中，可能让"打出第 N 张手牌"指向错误对象。由于 `alloc_eid` 单调递增、
id 不复用，实际影响概率低（此前已判定）。

---

## 3. 未覆盖、但对"模型下什么"影响更大的部分

| 项 | 状态 | 说明 |
|---|---|---|
| `client/.../projection.py`（Power.log → 状态） | **未审** | 决定 3227 个值填什么；填错等于给模型假观测 |
| `client/.../binding`（动作 → 界面元素） | **未审** | 昨夜 7 次 S0 全在这一层：动作合法≠界面可点 |
| 归一化 `norm_mean/norm_std` | 已随 checkpoint 加载 | 需确认训练/实机用同一份（`load_champion_policy` 里已读取） |
| 观测里的"可见性过滤"（对手战团/隐藏信息） | 未审 | 若实机投影喂了模拟器不该看到的信息，等于信息泄漏 |

---

## 4. 建议（不改代码的结论）

1. **动作规则**：采纳 HSBRSIM 的 `m[A_CANCEL] = cancellable() || !any`（10 行内改动 + 一个空掩码回归测试），
   HSAI 现有主动清理保留。这是本轮唯一"有证据支持应该改"的差异。
2. **观测编码**：两边同源，无需改；把精力转到 §3 的 `projection.py` 逐字段对齐（用同一局面比较
   `Game.observe()` 与实机投影向量）。
3. `state.rs` 的 hand-link 差异维持现状（低风险、已有不变量保护）。
