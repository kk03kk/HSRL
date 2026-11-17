# 第三方 PPO 模型接入评估（`PrentendYan/hearthstone-bg-ppo`）

日期：2026-11-17 · 状态：**只做研究与验证，未改训练代码、未动运行中的作业**

---

## 0. 结论摘要

| 问题 | 结论 | 依据 |
|---|---|---|
| 能否用我们的模拟器跑它的推理？ | **能，且不需要改引擎** | 观测/动作规格同源（见 §2） |
| 需要写什么？ | 一个**约 80 行的适配策略类**（加载 ckpt + argmax + 两条掩码） | §4 |
| 卡牌索引能否对齐？ | **能，已实证对齐** | 36 个阵容种族富集 2.4×–35×（§3.2） |
| 能否当我们的 PPO 初始化权重？ | **不能直接加载**（架构不同）；可做**蒸馏**或**冻结对手** | §5 |
| 最大风险 | **许可证 CC BY-NC 4.0（禁止商用）** + `keep_start_board` 掩码我们没实现 | §6 |

---

## 1. 权重实证（已下载并校验）

```
文件      data/thirdparty/hsbg-ppo/model.pt
大小      39,145,094 B
MD5       6ea4d35fdf436513d315d04fe9cbc3c8   ← 与 README 声明一致 ✅
sha256    0454a99dda1d95b03922aa5015c827b1d499c8e952893284a3b3a0e87bf0352b
格式      tg-ckpt-v1（dict: format / net_config / state_dict / meta）
```

**net_config（逐项读出，非猜测）**

| 键 | 值 |
|---|---|
| `d` / `layers` / `heads` | **352 / 4 / 8** |
| `n_cards` | **3984**（含索引 0 空位） |
| `attr` / `attr_f` | **True / 311**（311 维卡牌属性表，作为 frozen buffer 存在 state_dict 里） |
| `zcz` | True |
| `z_vocab` | **46 个阵容名**（如 `menagerie`、`quilboar-choose-one`、`hsreplay:beasts-beetles`…） |
| `attr_sha256` / `zcard_sha256` | 有，`load()` 会校验 |

**参数量（实测）**

```
state_dict 元素总数        9,778,188
− attr  (3984×311)         1,239,024
− zcard (46×3984×3)          549,792
= 可训练参数               7,989,372   ← 与 meta["params"] 完全一致 ✅（README「约 8.0M」）
```

**meta（作者自述，值得注意）**

```json
{"stage":"ppo","iter":1000,"params":7989372,
 "reward":{"kind":"util","util":[1.0,0.2,0.0,0.0,0.0,0.0,-0.1,-0.2]},
 "engine_source_hash":"205739107f78dc5a",
 "required_masks":["no_cancel_choose=1","keep_start_board=1 (sell restriction)"]}
```

→ 他们的奖励是 **「1 名 1.0 / 2 名 0.2 / 3–6 名 0 / 7 名 −0.1 / 8 名 −0.2」**：只奖前二、中段归零。
与我们这一轮刚采用的 `[1.00, 0.35, −1.00, −1.15, …]` **方向完全一致**（他们更极端，中段直接 0）。

---

## 2. 观测与动作规格：同源（这是能接入的根本原因）

作者的 `hsbg_net/net.py` 文档字符串给出**精确布局**：

```
floats  global 53 | quests 3×2 | pending 9 | board 7×31 | hand 10×40 | shop 8×40 |
        options 16×40 | opponents 7×9 | opponent boards 49×31          = 3227 ✓
cards   hero 1 | powers 2 | deity 1 | pending source 1 | quests 3×(quest,reward,owned) |
        secrets 5 | board 7 | hand 10 | shop 8 | options 16 | trinkets 8 |
        opp heroes 7 | opp boards 49                                    = 124 ✓
```

**我们的 `hsbg/env/obs_std.py` 明确写着同一件事**：

> 「标准观测 profile（AutoTavern / TavernGym 3227 floats + 124 cards）」
> 「`OBS_FLOATS=3227`、`OBS_CARDS=124`、`N_ACTIONS=245` 与上游一致」
> 「卡牌块偏移与上游 `OBS_CARDS` 注释顺序一致」
> 块宽：global 53（42 标量 + 11 种族位）、board 7×31、hand 10×40、shop 8×40、choices 16×40、opponents 7×9、opp_boards 7×7×31

**两边逐块宽度完全一致**，连 global token 的 68（53+6+9）都相同（他们 `GLOBAL=(0,1,68)`）。

**动作 245 分段也一致**（对照他们的 head 注释与我们的 `action_space_std`）：

| 段 | 他们 | 我们 |
|---|---|---|
| 0–7 | end, refresh, freeze, upgrade, hp0, hp1, gift, cancel | `A_END/A_REFRESH/A_FREEZE/A_UPGRADE/A_HERO_POWER(2)/A_DARK_GIFT/A_CANCEL` **同序** |
| 8–15 | buy（shop 8 槽） | `A_BUY 8..15` ✅ |
| 16–22 | sell（board 7 槽） | `A_SELL 16..22` ✅ |
| 23–29 | activate（board 7） | `A_ACTIVATE 23..29` ✅ |
| 30–45 | choose（options 16） | `A_CHOOSE 30..45` ✅ |
| 46–125 | play（hand10 × pos8） | `A_PLAY 46..125` ✅ |
| 126–195 | magnetize（hand10 × board7，**双线性**） | `A_MAGNETIZE 126..195` ✅ |
| 196–244 | move（board7 × pos7） | `A_MOVE 196..244` ✅ |

---

## 3. 卡牌索引对齐：已**实证**通过（不是靠维度猜）

### 3.1 索引空间与 padding 约定一致

- 他们 `n_cards=3984`，`attr` 形状 **(3984, 311)**，**第 0 行全零**（= padding），**3983 行非零**。
- 我们 `obs_std.py`：`N_CARDS_REFERENCE = 3984`，注释「当前 37.0.2.256589 构建的 `hsbg_engine.N_CARDS`（= **3983 张卡 + 空位 0**）」。
- 我们的引擎索引约定：**索引 = cards.json 位置 + 1，0 为空位**（库构造里出现过 `card_info((index & 0x0fff) + 1)`）。

⇒ **索引空间、padding 位置完全一致。**

### 3.2 卡序一致性检验（可证伪）

方法：`zcard[46, 3984, 3]` 是「阵容 × 卡 × 3 flags」，46 个阵容名里带种族词。
**若两家卡序一致**，则某阵容标记的卡应在对应种族上显著富集；**若顺序不同**，富集倍数应 ≈ 1。

实测（36 个可判定的阵容）：

| 阵容 | 标记卡 | 种族命中 | 该族基准率 | **富集倍数** |
|---|---:|---:|---:|---:|
| `beasts-leviathan` | 4 | 4/4 | 8.0% | **12.5×** |
| `hsreplay:quilboar-choose-one` | 10 | 10/10 | 4.3% | **23.4×** |
| `hsreplay:undead-attack-scaling` | 10 | 10/10 | 7.0% | **14.3×** |
| `hsreplay:pirates-apm-golden` | 6 | 6/6 | 5.2% | **19.3×** |
| `hsreplay:aberrations-sludge` | 10 | 6/10 | 1.7% | **35.1×** |
| `dragons-copebringer` | 4 | 4/4 | 4.9% | **20.3×** |
| …（共 36 个） | | | | **2.4× – 35×，中位 ~12×** |

**结论：卡序一致。** 若不一致，36 个阵容应全部塌到 ~1.0。

### 3.3 补丁谱系

`zcard` 在我们 `BG34_140`（飞行专家，37.0.2 已移出牌池）的索引 **2028** 上有 1 次标记
⇒ 他们的卡表**同样包含这张卡且索引相同** ⇒ 与我们同一卡表谱系（37.0.2 卡表，池子去掉它）。

> 注：他们的 `engine_source_hash=205739107f78dc5a` 与我们的 `c4c71014e3d41760` 不同——
> 那是**引擎源码指纹**（各自构建），不是卡表版本；卡表已由 §3.2/§3.3 独立证明一致。

---

## 4. 适配层设计（不需要改引擎）

模型是**纯函数**：`(features[3227], cards[124], mask[245], z∈[0,45]) → logits[245], value`。

我们已有全部输入：

| 输入 | 我们的来源 |
|---|---|
| `features` | `game.observe(seat)` → `obs_arrays()` 给出 3227 floats（`obs_std` 同源布局） |
| `cards` | 同上的 124 个 i32 索引（**索引语义已证对齐**） |
| `mask` | `game.legal_mask(seat)` | 
| `z` | 传 **45 = none**（公平测试用"无阵容先验"） |

**适配类草图（约 80 行，放在 `hsbg/az/train/thirdparty_policy.py`）**

```python
class ThirdPartyPPOPolicy(Policy):
    def __init__(self, ckpt_path, z=45, no_cancel_choose=True, keep_start_board=True):
        from hsbg_net import load           # 把作者的 hsbg_net/ 目录原样 vendor 进来
        self.net, self.ck, _ = load(ckpt_path, device="cpu")
        self.net.eval(); self.z = int(z)
        self.keep_start = {}                # (game, seat, turn) -> 起始板 eids
    def reset(self): self.keep_start.clear()
    def decide(self, game, seat):
        f, c = obs_arrays(game.observe(seat))
        mask = mask_from_bytes(game.legal_mask(seat))
        if self.no_cancel_choose: mask = without_cancel_in_choice(mask)   # 复用现有 pending_guard
        if self.keep_start_board: mask = mask_out_start_board_sells(game, seat, mask)
        with torch.no_grad():
            logits, value = self.net(torch.from_numpy(f)[None],
                                     torch.from_numpy(c)[None],
                                     torch.from_numpy(mask)[None],
                                     z=torch.tensor([self.z]))
        a = int(logits[0].argmax())
        return Decision(action=a, info={"src": "thirdparty", "value": float(value)})
```

**要新增的唯一实质逻辑**：`keep_start_board`（回合开始时已在场上的随从不能被卖）。
实现只需在策略层记录"本回合开始时的 board eid 集合"，把对应 `A_SELL` 槽置 False
⇒ **不改引擎**（引擎本来就允许卖，我们只是自己不选）。

**要新 vendor 的代码**：作者仓库的 `hsbg_net/`（5 个文件，约 14 KB，MIT 之外的 CC BY-NC 4.0）放进
`thirdparty/hsbg_ppo/hsbg_net/`，仅测试用。

---

## 5. 能否作为我们的 PPO 初始化权重

**不能直接加载**，架构不同：

| | 他们 | 我们（`arena-v1.2`） |
|---|---|---|
| d / layers / heads | 352 / 4 / 8 | 384 / 8 / 8 |
| 可训练参数 | **7,989,372** | **16,436,034** |
| 头 | policy(245) + scalar value | policy(245, seq head) + **rank(8) 名次头** + value |
| 卡表示 | id 嵌入 + **311 维属性投影** | id 嵌入 + 规则文本嵌入 |
| 归一化/压缩 | 无（Linear 直接吃 raw） | 输入做 symmetric log 压缩 |

可行的三条路：

| 方案 | 做法 | 代价 | 价值 |
|---|---|---|---|
| **A. 冻结对手（推荐先做）** | 直接当对手，与我们的模型同场对打 | 1 个适配类 | **立刻可测棋力**，零训练风险 |
| **B. 蒸馏（DAgger/BC）** | 用它的 logits 作 teacher，训我们的网络（我们已有 `teacher_action` + CE 管线） | 1 轮训练 | 把它的"阵容偏好"迁移到我们的架构 |
| **C. 架构借鉴** | ①**46 阵容 z-token 条件化**；②311 维卡牌属性投影；③magnetize 用双线性头 | 改模型定义 | 结构性改进，不依赖它的权重 |

**B 的关键风险**：它的 logits 是针对**它的 action 掩码**（含 `no_cancel_choose`/`keep_start_board`）训练的；做蒸馏时必须用同一套掩码，否则 teacher 分布被我们自己掩码扭曲。

---

## 6. 不兼容 / 风险清单（全部列出）

| # | 项 | 严重度 | 说明与对策 |
|---|---|---|---|
| **1** | **许可证 CC BY-NC 4.0** | **高** | **禁止商业使用**，需署名。任何对外/商用产品都不能带它。内部研究对局可用。 |
| 2 | `keep_start_board=1` 未实现 | 中 | 作者实测**差 0.10–0.14 名次**。需按 §4 加一条卖牌掩码（不改引擎） |
| 3 | `no_cancel_choose=1` | 低 | 我们已有 `forbid-cancel 1`；需确认它是否等价于"choose 阶段也不许取消" |
| 4 | `z` 未指定时=45 | 低 | 公平测试一律传 45（无先验）；作者 README 说它**阵容偏好窄（偏野猪人）**，给 z 反而更不公平 |
| 5 | attr / zcard 是 frozen buffer | 无 | 已在 ckpt 内，我们无需复现 |
| 6 | 卡表版本 | **已排除** | §3.2 富集检验 + §3.3 索引 2028 双证一致 |
| 7 | obs 数值标定（是否 log 压缩） | **待确认** | 他们的 `f_global/f_board/...` 直接吃 raw。**我们的压缩在我们自己的模型内部**（`entity_model` 文档：「输入做 symmetric log 压缩」），所以喂给他们时应给**引擎原始 3227**。做完 §7 的 A/B 冒烟即可确认（若 argmax 分布与随机无异 ⇒ 标定不符） |
| 8 | 引擎源码指纹不同 | 无影响 | 卡表/动作/观测三项已独立验证 |
| 9 | 它只有 **iter=1000** 的训练 | 参考 | 与我们的 600+ 轮不在同一量级，但它是 8M 参数的窄模型 |

---

## 7. 公平对局评估方案（与我们当前模型同条件）

**与主训评测完全相同的口径**：

```
场地      arena-cal-v14hold-snap（held-out 真人战团，T1–T20）
局数/种子  1024 局，base seed 930000，同一套 3 节点 × 8 进程分片
强制升本   无（自由升本）
动作契约   max-actions 80 / move-guard 1(cap5,repeat2) / forbid-cancel 1 / freeze-cap 2
参赛模型  thirdparty(z=45) + v1.2 + r1 + r2 + r3（共 5 个，同场同种子）
指标      逐回合中位战力(M1) / tier(M2) / 名次 / 前四 / 吃鸡 / 非法 / 截断
```

**分三步走（先便宜后昂贵）**：

| 步 | 内容 | 成本 | 判据 |
|---|---|---|---|
| **S0 冒烟** | 64 局，只跑它一个 | ~1 分钟 | ①非法动作=0 ②argmax 动作分布不塌（例如不出现 >95% 全选 EndTurn）③各动作族都有覆盖 |
| **S1 正规对局** | 256 局，它 vs 我们的 v1.2 | ~2 分钟 | 配对名次 delta + CI |
| **S2 全量** | 1024 局 × 5 模型（§7 表） | ~8 分钟（3 节点） | 正式判胜 |

**S0 是硬门槛**：如果 S0 的动作分布塌陷，说明输入标定不符（风险 #7），**必须先解决再谈对局**——绝不能用随机输入的前向通过来当"能跑"的证据。

---

## 8. 能借鉴的设计（不依赖其权重，价值最高）

1. **阵容条件化 token `z`（46 类）**：把"想打什么阵容"作为输入条件，模型可被引导走指定流派。
   我们目前没有这个机制，而我们的分析已经证明"模型在赌上限"——**给定阵容先验可能比加压更有效**。
2. **311 维卡牌属性 + id 嵌入双路**：对**新卡/冷门卡**比纯 id 嵌入更稳（我们靠规则文本嵌入，方向类似但更贵）。
3. **magnetize 用双线性头（hand × board）**：把 70 维磁力动作拆成 10×7 的双线性打分，
   与我们"整 245 维平铺打分"相比参数更省、结构更对。
4. **奖励 `[1.0, 0.2, 0, 0, 0, 0, −0.1, −0.2]`**：独立印证了我们这轮"只奖前二"的方向。
5. **`keep_start_board` 卖牌限制**：作者实测 +0.10–0.14 名次，**这条我们也可以直接加**（与第三方权重无关）。

---

## 9. 明确回答

- **能否直接接入？** **推理能**（写 ~80 行适配类 + vendor 5 个源文件），**初始化权重不能**（架构不同）。
- **要改什么？** 不改引擎；新增 `thirdparty_policy.py` + `keep_start_board` 掩码 + vendor 目录。
- **能借鉴什么？** 阵容 z-token、卡牌属性双路、双线性磁力头、top-2 奖励、卖牌限制（后两条与我们权重无关，可直接用）。
- **最合理的实际测试方案？** §7 的 **S0 冒烟（64 局，1 分钟）→ S1（256 局）→ S2（1024 局 × 5 模型同场同种子）**，
  其中 S0 的动作分布检查是**不可跳过**的门槛。
- **最大风险**：**CC BY-NC 4.0 禁商用**；其次是输入标定（风险 #7）需 S0 实证。
