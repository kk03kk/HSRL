"""Training CLI schema. Defaults preserved from the existing trainer."""
import argparse


def build_parser():
    ap = argparse.ArgumentParser(description="Entity PPO（持久 trainer + 进程 pool + 流水线）")
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--stop-at-decisions", type=float, default=0.0,
                    help="累计 learner decisions 达到该值后停止（0=不启用）；与 --minutes 取先到者")
    ap.add_argument("--iterations", type=int, default=0)
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--arch", default="entity", choices=["tiny", "entity", "entity18m"])
    # ── entity 尺寸（只影响**从零构建**；带 --checkpoint/--trainer-state 时以 checkpoint
    #    meta 为准，本组只做一致性校验）。默认 0 = build_net 的历史默认（384/8/8），
    #    因此不传时与原路径逐位等价。0.667M = --d-model 128 --n-layers 2 --n-heads 4。
    ap.add_argument("--d-model", type=int, default=0,
                    help="entity 宽度（0=默认 384）；从零构建时生效")
    ap.add_argument("--n-layers", type=int, default=0,
                    help="entity Transformer 层数（0=默认 8）；从零构建时生效")
    ap.add_argument("--n-heads", type=int, default=0,
                    help="entity 注意力头数（0=默认 8）；从零构建时生效")
    ap.add_argument("--move-head-split", type=int, default=0, choices=[0, 1],
                    help="1=Move 段（196..244）用有向双投影（move_from/move_to），"
                         "方向差不再只由 move_bias 决定；旧 checkpoint 自动只初始化新投影。"
                         "默认 0=保持对称实现（逐位兼容旧 ckpt）")
    ap.add_argument("--move-guard", type=int, default=1, choices=[0, 1],
                    help="1=换位去抖守卫：采样前屏蔽「逆换位 / 同动作超限 / 本回合换位"
                         "超 cap」的 Move 动作位（见 docs/MOVE_GUARD.md）。"
                         "默认 0=关闭（与旧路径逐位等价）")
    ap.add_argument("--forbid-unfreeze", type=int, default=1, choices=[0, 1],
                    help="商店全部冻结时屏蔽 Freeze（与实机桥接一致；默认 1）")
    ap.add_argument("--freeze-cap", type=int, default=1,
                    help="每回合最多按几次 Freeze（0=不限制，引擎上限 10）。"
                         "2026-11-17 D-070：默认 2→1（每回合只能按一次冻结键）")
    ap.add_argument("--ban-trinkets", type=int, default=1, choices=[0, 1],
                    help="1=禁用加重护手(BG36_MagicItem_414)等黑名单待选卡（默认开，不改模拟器）")
    ap.add_argument("--lobby-exclude", default="",
                    help="禁掉的 lobby 种族（逗号分隔，如 MECHANICAL,DRAGON；空=不过滤）。"
                         "用确定性种子过滤实现：含被禁种族的局会被跳过换派生种子，"
                         "引擎与卡表都不改（见 hsbg/az/train/lobby_filter.py）")
    ap.add_argument("--lobby-filter-probability", type=float, default=1.0,
                    help="Training fraction assigned the lobby exclusion before observing the game.")
    # ── 借鉴第三方模型的三项"可选结构"（2026-11-17 D-072；默认全关 = 逐位等价旧模型）──
    # 以前这三项在 build_entity_net 里存在，但训练入口**没有任何 CLI** ⇒ 永远打不开。
    ap.add_argument("--use-z", type=int, default=0, choices=[0, 1],
                    help="1=启用 46 阵容条件化 token（z_global + z_tok/zcard）。"
                         "注意：需要 zcard 有真表（当前为空 buffer ⇒ 只有 z_global 生效）")
    ap.add_argument("--n-z", type=int, default=46, help="z 槽位数（默认 46，45=中立）")
    ap.add_argument("--use-attr", type=int, default=0, choices=[0, 1],
                    help="1=启用 311 维卡牌属性投影（attr buffer 当前为空 ⇒ 恒等，需先灌表）")
    ap.add_argument("--attr-f", type=int, default=311, help="卡牌属性维度（默认 311）")
    ap.add_argument("--mag-query", type=int, default=0, choices=[0, 1],
                    help="1=磁力头加一条可学双线性项（零初始化 ⇒ 起点等价旧版；**唯一"
                         "无需外部数据就能训的一项**）")
    ap.add_argument("--forbid-cancel", type=int, default=1, choices=[0, 1],
                    help="1=always remove Cancel action 7; fail if no legal action remains")
    ap.add_argument("--move-cap", type=int, default=2,
                    help="每回合换位次数上限：>=cap 后屏蔽全部 Move"
                         "（2026-11-17 D-070：默认 5→2；0=本回合禁止换位）")
    ap.add_argument("--move-repeat-limit", type=int, default=2,
                    help="同一 (from,to) 本回合出现 >=N 次后屏蔽（默认 2；1=不允许重复）")
    # ── 流派（archetype）软偏置（默认关闭；见 docs/ARCHETYPE_BIAS.md）──
    ap.add_argument("--archetype-bias-coef", type=float, default=0.0,
                    help="流派软偏置系数：0=关闭（逐位等价旧路径）。对当前流派在当前 "
                         "turn 的偏好动作加 coef*相对增益 的 logit（只加不减、不屏蔽），"
                         "建议 0.5–2.0")
    ap.add_argument("--archetype-min-prob", type=float, default=0.6,
                    help="检测置信度门槛：max_prob >= 该值才偏置（低置信不偏置，默认 0.6）")
    ap.add_argument("--archetype-table", default="",
                    help="偏好表路径（空= data/archetype/archetype_table.json）")
    ap.add_argument("--archetype-model", default="",
                    help="分类器模型路径（空= data/archetype/archetype_model.npz）")
    ap.add_argument("--archetype-cache", type=int, default=0,
                    help="1=流派特征/分类走输入哈希缓存（输出逐位相同，但实测命中率≈0，"
                         "默认 0 关闭；见 docs/PERF_COLLECT.md）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sims", type=int, default=0)
    ap.add_argument("--determinizations", type=int, default=1)
    ap.add_argument("--opponent", default="bot", choices=["bot", "self", "weak"])
    ap.add_argument("--behavior", default="net", choices=["net", "mcts", "mixed"])
    # ── 搜索作为行为策略：μ=(1−ε)π+ε·π_search（默认 eps=0 = 关闭、逐位等价旧路径）──
    #   见 hsbg/az/train/behavior_mix.py 与 docs/BEHAVIOR_MIX_PPO.md。
    ap.add_argument("--behavior-mix-eps", type=float, default=0.0,
                    help="关键决策以该概率改用 π_search 采样（μ=(1−ε)π+ε·π_search）；"
                         "0=关闭（默认，逐位等价旧路径）；建议 0.1–0.3")
    ap.add_argument("--behavior-mix-when", default="carrier_or_spell",
                    help="关键决策口径（sandbox_rollout.SEARCH_WHEN_NAMES；可逗号组合，"
                         "默认 carrier_or_spell，与 t6/AB 面板同口径）")
    ap.add_argument("--behavior-mix-max-per-turn", type=int, default=6,
                    help="每回合最多搜索几次（0=不限；默认 6 = t6 配置）")
    ap.add_argument("--behavior-mix-topk", type=int, default=8,
                    help="根候选数 = π top-k 非 Move 合法动作（默认 8）")
    ap.add_argument("--behavior-mix-turns", type=int, default=1,
                    help="叶值 horizon：1=到本回合末（默认，沿用 s17 的 t1 口径）")
    ap.add_argument("--behavior-mix-max-ms", type=float, default=500.0,
                    help="单次搜索墙钟上限（ms；0=不限）：超时放弃搜索、回退按 π 采样")
    ap.add_argument("--behavior-mix-exact", type=int, default=1, choices=[0, 1],
                    help="1=精确模式（默认）：每个关键决策都跑搜索，logp_mu 对全部"
                         "分支精确（IS 比 π/μ 无偏，权重 ≤1/(1−ε)）；"
                         "0=只在 ε 命中时搜索（成本 ×ε，跨分支不再等价 marginal μ）")
    ap.add_argument("--behavior-mix-report", default="",
                    help="可选：逐 iteration 的混合统计 JSONL 落盘路径（空=只写进 "
                         "history.jsonl 的 collect 段）")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-min-ratio", type=float, default=0.1)
    ap.add_argument("--max-updates", type=int, default=100000)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--target-kl", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=0.95,
                    help="GAE λ（与折扣 γ 无关；论文 0.995）")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="训练折扣 γ（默认 1.0：8 座单局终局效用；不是 GAE 的 λ）")
    # ── 信用分配（§B；默认 gae = 旧路径逐位等价）──
    ap.add_argument("--adv-estimator", default="gae",
                    choices=["gae", "power", "geom"],
                    help="advantage 时间核：gae=旧 GAE；power=幂率/分数阶；geom=几何对照")
    ap.add_argument("--power-alpha", type=float, default=0.25,
                    help="幂率核 α>=0（w_l=(1+l/κ)^(−α)）；仅 adv-estimator=power 生效")
    ap.add_argument("--power-kappa", type=float, default=10.0,
                    help="幂率核 κ>0；离线推荐 α=0.25, κ=10（见 docs/CREDIT_ASSIGNMENT.md）")
    ap.add_argument("--power-normalize", type=int, default=1, choices=[0, 1],
                    help="1=horizon 归一化（§B.1）；离线诊断建议 A/B 用 0"
                         "（归一化会把早期 |A| 压成 ~1/L）")
    ap.add_argument("--power-cutoff", type=int, default=0,
                    help="lag 截断（0=不截断）")
    ap.add_argument("--credit-timebase", default="decision",
                    choices=["decision", "turn"],
                    help="decision=按动作步；turn=未来 elapsed turn 的 SMDP 折扣")
    ap.add_argument("--baseline-action-dependent", type=int, default=0,
                    choices=[0, 1],
                    help="动作条件 baseline：仅允许 0（actor baseline 只用 V(s)）。"
                         "1 会直接报错，见 §B.3")
    ap.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="梯度裁剪范数（论文 1.0）")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="梯度累积步数：有效 minibatch = minibatch * grad_accum（论文 8192）")
    ap.add_argument("--leaf-compile", type=int, default=0,
                    help="1=叶前向 torch.compile + TF32（实测 B=64 5.4→0.7 ms）")
    ap.add_argument("--leaf-amp", default="off", choices=["off", "bf16"],
                    help="采集/评测叶前向精度（bf16 实测 collect 约 -5%%）")
    ap.add_argument("--leaf-wait-ms", type=float, default=2.0,
                    help="批量叶服务器聚批等待（ms）；单客户端建议 0.1")
    ap.add_argument("--rc-threads", type=int, default=8,
                    help="Rust 采集器内部引擎线程数（BatchEnv threads）")
    ap.add_argument("--rust-collector", type=int, default=0,
                    help="1=Rust 全采集器（BatchEnv 驱动，记录打包在 Rust）")
    ap.add_argument("--library-pool", default="",
                    help="镜像对手库 JSONL（空=不用）；见 docs/LIBRARY_POOL_PLAN.md")
    ap.add_argument("--frozen-seats", type=int, default=0,
                    help="由冻结对手驱动的 bot 座位数（1–2 性价比最高；7=全部，慢 ~25x）")
    ap.add_argument("--frozen-opponent", default="",
                    help="冻结对手模型 checkpoint（空=引擎 bot）；bot 座位由它驱动（分裂模型/自对弈）")
    ap.add_argument("--trajectory-pool", default="",
                    help="对手场面轨迹库目录（空=不用）；见 docs/LIBRARY_POOL_PLAN.md")
    ap.add_argument("--trajectory-p", type=float, default=0.0,
                    help="每局被轨迹镜像替换的概率（建议 0.1 起）")
    ap.add_argument("--mirror-strict", type=int, choices=[0, 1], default=0, help="活席必须使用校准库，禁止Bot回退")
    ap.add_argument("--mirror-snaps", default="",
                    help="回合末场面快照库目录（snap_*.json.gz；空=不用）；见 "
                         "docs/MIRROR_SNAPSHOT_DESIGN.md")
    ap.add_argument("--mirror-seats", type=int, default=0,
                    help="回合末快照镜像的座位数：给 seat 1..N 注入（0=关闭，最大 7）")
    ap.add_argument("--mirror-p", type=float, default=0.0,
                    help="每座位每回合注入快照的概率（默认 0=关闭，零影响）")
    ap.add_argument("--mirror-sample", "--sample", dest="mirror_sample", default="p50",
                    metavar="MODE",
                    help="快照桶内采样：p50/p75/p90=强度分位课程，p75band 等=分位带内"
                         "均匀取样（保留难度且恢复对手多样性），turnband:<lo-hi=pct,...>="
                         "按回合的分位带宽课程，max/strong=旧上限")
    ap.add_argument("--mirror-p-ramp", default="",
                    help="线性 p ramp，如 0.1@0,0.25@0.3,0.25@1.0（x 与 iteration 同单位）；"
                         "空=用固定 --mirror-p")
    ap.add_argument("--mirror-seat-random", type=int, default=0,
                    help="1=每回合从全部非 learner 座位随机抽 --mirror-seats 个（课程模式）")
    ap.add_argument("--mirror-max-turn", type=int, default=0,
                    help="只在 turn <= N 时注入镜像快照（0=不限）。中后期入场课程用它把"
                         "镜像对手关在入场回合之前")
    ap.add_argument("--human-boards", default="",
                    help="人类战团（board-only）JSONL（默认 "
                         "data/real_library/library_real_degraded.jsonl；空=不用）；见 "
                         "docs/HUMAN_BOARD_OPPONENT.md")
    ap.add_argument("--human-board-p", type=float, default=0.0,
                    help="每个候选座位每回合注入人类战团的概率（默认 0=关闭，零影响）")
    ap.add_argument("--human-board-seats", type=int, default=2,
                    help="每回合从全部非 learner 座位里随机抽几个座位注入（默认 2）")
    ap.add_argument("--arena-config", default="",
                    help="动态难度课程配置 JSON（enabled/curve/init_q/spec）；空=关闭。"
                         "见 docs/DECISIONS.md D-054")
    ap.add_argument("--arena-state", default="",
                    help="难度档位持久化路径（默认 <out>/arena_state.json）。"
                         "续训必须指向同一个文件，否则每段都会从头降难度")
    ap.add_argument("--human-board-sample", default="auto",
                    choices=["auto", "p10", "p20", "p25", "p33", "p40", "p50",
                             "p60", "p67", "p75", "p80", "p90", "max"],
                    help="人类战团桶内分位：auto=按 turn 走难度曲线（t<=8 p50 / 9..12 p75 "
                         "/ t>=13 p90，默认）；pXX=固定分位（越低对手越弱，动态难度课程用）")
    ap.add_argument("--human-board-set-tier", type=int, default=1,
                    help="1=同时写入按 turn 派生的酒馆等级（人类数据没有 tier 字段；默认 1）")
    ap.add_argument("--human-board-fidelity", default="board",
                    choices=["board", "hand", "full"],
                    help="人类战团注入保真度：board=board-only（默认，逐位等价现状）；"
                         "hand=board+手牌；full=board+hand+可推断 counters"
                         "（关键词位/金身/圣盾层数/tavern tier）。hand/full 需要 "
                         "data/real_library/library_real_quasi_full.jsonl"
                         "（workflows/build_fidelity_library.py 生成）；见 "
                         "docs/FIDELITY_INJECTION.md")
    ap.add_argument("--human-board-fast-inject", type=int, default=0,
                    help="1=人类战团注入走快速路径（payload 注入视图缓存 + orjson + "
                         "ended 位字符串替换）；**与默认路径逐位等价**（同 seed 动作序列/"
                         "状态 JSON 完全相同），实测采集侧 Python 成本降约四成。默认 0=旧路径")
    ap.add_argument("--human-board-min-turn", type=int, default=1,
                    help="只在 turn >= N 时注入人类战团（默认 1=逐位等价旧行为）")
    # ---- 中后期入场课程（learner 席注入真人完整状态；见 docs/LATE_GAME_ENTRY.md）----
    ap.add_argument("--entry-boards", default="",
                    help="入场注入库（与 --human-boards 同格式的准完整 JSONL）；空=关闭")
    ap.add_argument("--entry-p", type=float, default=0.0,
                    help="逐局入场概率（默认 0=关闭，零影响）；未命中的局保持原配方")
    ap.add_argument("--entry-turn-min", type=int, default=0,
                    help="入场回合区间下界（<=0=关闭）。T0 之前 learner 由 bot 代打，"
                         "不产生任何 learner 记录")
    ap.add_argument("--entry-turn-max", type=int, default=0,
                    help="入场回合区间上界（0=等于下界）")
    ap.add_argument("--entry-sample", default="auto",
                    choices=["auto", "p50", "p75", "p90", "max"],
                    help="入场帧桶内分位（默认 auto=按 turn 走难度曲线）")
    ap.add_argument("--entry-fidelity", default="full",
                    choices=["board", "hand", "full"],
                    help="入场注入保真度（默认 full；需要准完整库）")
    ap.add_argument("--entry-fast-inject", type=int, default=0,
                    help="1=入场注入走快速路径（与 --human-board-fast-inject 逐位等价）")
    ap.add_argument("--entry-set-health", type=int, default=0,
                    help="1=把观测到的 hero_hp 一起写回 learner 席；默认 0（血量随局演进）")
    ap.add_argument("--entry-min-hand", type=int, default=0,
                    help="入场帧的最低观测手牌数（0=不过滤）。真人后期每回合 play 22/sell 10.4，"
                         "模型只有 8/1.2 —— 用它给「资源在手」的起点")
    ap.add_argument("--entry-require-cards", default="",
                    help="入场帧必须含这些引擎卡 card_id（逗号/分号分隔；空=不过滤）。"
                         "用于'引擎入场课程'：真人 T12 有 64%% 战团含引擎卡，模型只有 9%%")
    ap.add_argument("--entry-min-tier", type=int, default=0,
                    help="只使用观测等级 >= 本值的入场帧（0=不过滤）。"
                         "专家先验：T8 左右到五本 → 0/1 都不过滤，5 = 只在五本以上开课")
    # ---- 专家相对过程奖励（默认 coef=0 = 关闭；见 hsbg/az/train/board_target.py）----
    ap.add_argument("--board-target-coef", type=float, default=0.0,
                    help="每回合 (强度差 + 阵容召回) 奖励的系数；0=关闭。需要入场库带 "
                         "chain_* 字段（tools/build_human_chain.py）")
    ap.add_argument("--board-target-w-card", type=float, default=1.0,
                    help="阵容召回率权重（默认 1.0）")
    ap.add_argument("--board-target-w-str", type=float, default=0.5,
                    help="强度差权重（默认 0.5；单独用召回率会被囤牌刷高）")
    ap.add_argument("--board-target-cap", type=float, default=1.0,
                    help="强度差裁剪上限（默认 1.0）")
    # ---- 对手强度课程：human-board 席数/概率/分位按累计 decisions 爬坡 ----
    #     默认（空 spec + adaptive=0）完全恒等：CollectConfig 里仍是 --human-board-* 的值。
    #     见 docs/BOARD_RAMP.md。
    ap.add_argument("--human-board-ramp", default="",
                    help="对手强度课程（席数/概率）：decisions:seats:p,...，例 "
                         "\"0:1:0.30,10M:2:0.50,20M:3:0.70,30M:5:0.90\"。按**累计 learner "
                         "decisions** 求值，decisions 支持 K/M/B 后缀；空=用固定 "
                         "--human-board-seats/--human-board-p（默认，逐位等价）")
    ap.add_argument("--human-board-quantile-ramp", default="",
                    help="分位课程（与席数/p 独立）：decisions:quantile,...，例 "
                         "\"0:p75,20M:p90\"；取值 auto/p50/p75/p90/max；空=用固定 "
                         "--human-board-sample（默认不爬坡）")
    ap.add_argument("--ramp-mode", default="step", choices=["step", "linear"],
                    help="档位插值方式：step=分段常数（默认，边界突变）；linear=段内线性"
                         "（seats 四舍五入、p 连续），用于平滑过渡实验")
    ap.add_argument("--ramp-adaptive", type=int, default=0, choices=[0, 1],
                    help="1=自适应门控：用最近 --ramp-window 个 iter 的「存活到 t12」与"
                         "「tier>=4」比例决定提前升档（连续达标）/推迟下一档（不达标）；"
                         "默认 0=严格按 decisions 走课程")
    ap.add_argument("--ramp-window", type=int, default=5,
                    help="自适应门控的滑动窗口 iteration 数（默认 5）")
    ap.add_argument("--ramp-survive-threshold", type=float, default=0.25,
                    help="升档门槛：窗口内存活到 t12 比例 >= 该值（默认 0.25）")
    ap.add_argument("--ramp-tier4-threshold", type=float, default=0.30,
                    help="升档门槛：窗口内 tier>=4 比例 >= 该值（默认 0.30）")
    ap.add_argument("--ramp-delay-decisions", default="2M",
                    help="不达标时把后续档位整体推迟多少 decisions（默认 2M；每档最多一次）")
    ap.add_argument("--ramp-max-forced-per-game", type=float, default=12.0,
                    help="门控保护：每局「强制结束回合」均值超过该值说明存活是动作上限"
                         "刷出来的假象，不允许提前升档（默认 12；s15 真实值 ≈0.16）")
    ap.add_argument("--ramp-start-decisions", default="0",
                    help="ramp 的起始累计 decisions 偏移（续训时要传上一次的累计值，"
                         "否则课程会从 band0 重新开始；默认 0）")
    ap.add_argument("--ramp-quality-json", default="",
                    help="可选：离线质量统计 JSON（workflows/quality_monitor.py 输出）。给了就"
                         "优先用它的 survive_t12_ratio / tier_ge4_ratio 做门控指标；"
                         "空=用采集轨迹代理（max_turn>=12 / max_turn>=8）")
    # ---- 势函数奖励塑形（默认关闭，见 docs/REWARD_SHAPING.md）----
    ap.add_argument("--shaping-coef", type=float, default=0.0,
                    help="势函数奖励塑形系数 λ；0=关闭（默认，records/GAE 与旧路径逐位等价）")
    ap.add_argument("--shaping-gamma", type=float, default=None,
                    help="塑形折扣 γ_s（默认空=跟随训练 --gamma；只有与训练 γ 严格相等才满足"
                         "策略不变性；显式不一致会启动期警告并记录累计偏差）")
    ap.add_argument("--shaping-w-str", type=float, default=1.0,
                    help="Φ 的 Σ(atk+hp) 权重（默认 1.0；log_strength 口径下乘 L(Σ)）")
    ap.add_argument("--shaping-w-tier", type=float, default=0.5,
                    help="Φ 的 TIER_BONUS 权重（默认 0.5；log_strength 口径建议 0）")
    ap.add_argument("--shaping-mode", default="sigma", choices=["sigma", "log_strength"],
                    help="塑形强度口径：sigma（默认，原始 Σ）/ log_strength（L(Σ)，与"
                         "沙盒/推理期叶值同曲线；整局望远镜和 = shaping_coef·L(Σ_T)）")
    ap.add_argument("--rank-utility-coef", type=float, default=1.0,
                    help="终局名次效用系数（默认 1.0=旧行为）。混合奖励用："
                         "0.3 = 0.7·强度 + 0.3·名次")
    ap.add_argument("--win-bonus", type=float, default=0.0,
                    help="冠军奖金 b：名次效用 u(1) 由 1.0 变成 1+b，其余名次不变。"
                         "同一张表同时用于终局奖励与 V(s)=Σp·u，两边恒等一致。"
                         "0=论文原值（默认）。v13 起用 0.7（冠军/亚军 2.38×）")
    ap.add_argument("--rank-utility-table", default="",
                    help="直接指定整张名次效用表（8 个逗号分隔的浮点数，如 "
                         "'1.00,0.35,-1.00,-1.15,-1.30,-1.45,-1.60,-1.75'）。"
                         "给了它就以整表为准，--win-bonus 被覆盖。空=不用（默认）")
    ap.add_argument("--terminal-strength-coef", type=float, default=0.0,
                    help="终局强度奖励系数（默认 0=关闭）：末步额外加 coef·L(Σ_末决策)；"
                         "0.7 = 0.7·强度 + 0.3·名次")
    ap.add_argument("--shaping-tier-bonus", default="0,0,20,60,150,400,900",
                    help="按 tier 直接索引的奖励表（逗号分隔；索引 0=占位，越界按 0）")
    # ---- 法术自举补贴（动作条件奖励，默认关闭；见 docs/analysis/NEXT_INTERVENTIONS_20261114.md）----
    ap.add_argument("--spell-bootstrap-coef", type=float, default=0.0,
                    help="「买法术」动作的前 K 次小额补贴系数；0=关闭（默认，逐位等价旧路径）。"
                         "注意这是**动作条件**奖励，不满足策略不变性")
    ap.add_argument("--spell-bootstrap-per-game", type=int, default=3,
                    help="每局最多补贴几次买法术（配额上限，防止刷廉价法术骗分）")
    # ---- 等级进度奖励（tier-target，默认关闭，见 docs/TIER_TARGET.md）----
    ap.add_argument("--tier-target-coef", type=float, default=0.0,
                    help="等级进度奖励系数；0=关闭（默认，records/GAE 与旧路径逐位等价）")
    ap.add_argument("--tier-penalty", type=float, default=0.02,
                    help="某回合 tier 未达目标时该回合第一条 decision 的惩罚（乘 coef；默认 0.02）")
    ap.add_argument("--tier-bonus", type=float, default=0.15,
                    help="首次达标某回合目标时的一次性奖励（乘 coef；默认 0.15）")
    ap.add_argument("--tier-target-table", default="7:4,9:5,12:6",
                    help="目标曲线 turn:tier，逗号分隔（默认 7:4,9:5,12:6）")
    ap.add_argument("--tier-gate-k", type=float, default=0.0,
                    help="板面门控系数：只有板面 Σ >= k*参考曲线才计入该回合（0=关闭）。"
                         "无门控的等级奖励会被「只升本不买人」打穿（2026-11-13 实测）")
    ap.add_argument("--tier-gate-ref", default="",
                    help="门控参考曲线 turn:strength；空=用 champion 面板 p50 默认曲线")
    # ---- 留存随从滚雪球奖励（retained-growth，默认关闭；见 hsbg/az/train/retained_growth.py）----
    ap.add_argument("--retained-growth-coef", type=float, default=0.0,
                    help="留存随从增重奖励系数；0=关闭（默认，records/GAE 与旧路径逐位等价）。"
                         "按'每只留存随从每回合涨多少'发钱，每局上界 = --retained-growth-cap")
    ap.add_argument("--retained-growth-min-turn", type=int, default=9,
                    help="从第几回合开始计分（默认 9）")
    ap.add_argument("--retained-growth-normalizer", type=float, default=690.0,
                    help="归一化基准：每只留存随从每回合增重达该值即拿满该回合（真人实测 690）")
    ap.add_argument("--retained-growth-cap", type=float, default=0.5,
                    help="每局总奖励上界（默认 0.5 ≈ 名次全距 2.0 的 25%%）。"
                         "注意 argparse 会对 help 文本做百分号格式化，"
                         "所以这里必须写成两个百分号。")
    # ---- 阵容覆盖奖励（comp-target，默认关闭；见 hsbg/az/train/comp_target.py）----
    ap.add_argument("--comp-target-coef", type=float, default=0.0,
                    help="阵容覆盖奖励系数；0=关闭（默认，records/GAE 与旧路径逐位等价）。"
                         "每局奖励上界 = coef（拼满一个 S 级流派）")
    ap.add_argument("--comp-target-min-turn", type=int, default=8,
                    help="从第几回合开始打分（默认 8；更早阵容还没成形）")
    ap.add_argument("--comp-target-cards", default="config/strategy_cards.json",
                    help="流派库路径（33 个流派的核心卡；默认 config/strategy_cards.json）")
    ap.add_argument("--comp-target-tier-w", default="S:1.0,A:0.6,B:0.3",
                    help="流派等级权重（默认 S:1.0,A:0.6,B:0.3）")
    ap.add_argument("--comp-target-cap", type=float, default=1.0,
                    help="单局完成度上界（默认 1.0）")
    ap.add_argument("--comp-target-min-hits", type=int, default=2,
                    help="反刷分门槛：命中核心卡数不足就不给分（默认 2）")
    # ---- 强制升本课程（默认关闭，见 docs/FORCED_TIER.md）----
    ap.add_argument("--forced-tier-plan", default="",
                    help="强制升本计划 turn:tier，逗号分隔（默认 2:2,5:3,7:4,8:5，不含 6 本）；"
                         "空=关闭（默认，records/GAE 与旧路径逐位等价）")
    ap.add_argument("--forced-tier-carry", type=int, default=1,
                    help="目标回合没做到时允许顺延的回合数（记为 deferred；默认 1）")
    ap.add_argument("--forced-tier-schedule", default="",
                    help="强制概率随累计 decisions 的分段常数计划，如 "
                         "'0:1.0,40M:0.75,80M:0.4,120M:0.0'；空=全程用 --forced-tier-prob")
    ap.add_argument("--forced-tier-prob", type=float, default=1.0,
                    help="未给 schedule 时的强制概率（1.0=每回合必强制）")
    # ---- 强制买法术探索课程（默认关闭；见 docs/analysis/DESIGN_FORCED_SPELL_20261114.md）----
    ap.add_argument("--forced-spell-per-game", type=int, default=0,
                    help="每局最多强制买几次法术；0=关闭（默认，旧路径逐位等价）。"
                         "零频动作只能靠强制注入打开（动作条件奖励对它无效）")
    ap.add_argument("--forced-spell-turn-min", type=int, default=6,
                    help="从第几回合开始允许强制买法术（默认 6）")
    ap.add_argument("--forced-spell-prob", type=float, default=1.0,
                    help="配额内每次合法机会的强制概率（1.0=必强制）")
    ap.add_argument("--forced-spell-teacher", type=int, default=0, choices=[0, 1],
                    help="1=未被强制但法术槽合法时写 teacher_action 标签（默认 0）")
    ap.add_argument("--forced-teacher-until", default="",
                    help="在这段累计 decisions 之前，目标未达成且 Upgrade 合法时写 "
                         "teacher 标签（DAgger），如 '160M'；空=不写")
    ap.add_argument("--forced-teacher-coef", type=float, default=0.0,
                    help="teacher 标签的 CE 系数（>0 才真的产生梯度；与 "
                         "--teacher-bc-coef 取较大值）")
    ap.add_argument("--library-p", type=float, default=0.0,
                    help="每回合每个 bot 座位的注入概率（建议 0.3 起）")
    ap.add_argument("--lockstep", type=int, default=0,
                    help="1=lock-step 批量 actor（每 worker 内多局同步推进）")
    ap.add_argument("--fast-act", type=int, default=1,
                    help="1=worker 侧采样（省 245 维 prior 回传）；0=旧路径（A/B 用）")
    ap.add_argument("--local-batch-inference", type=int, choices=[0, 1], default=0,
                    help="单进程单worker lockstep复用fused_act；新实验显式开启")
    ap.add_argument("--amp", default="off", choices=["off", "bf16"],
                    help="PPO 前向/反向精度；bf16 实测 2.03×（loss 侧仍 fp32）")
    ap.add_argument("--compile-train", type=int, default=0, choices=[0, 1, 2],
                    help="训练侧 torch.compile（只 learner 的 fwd+bwd + anchor 参考前向）："
                         "0=关（默认）1=默认模式 2=max-autotune。失败自动回退 eager；"
                         "叶前向/自对弈/评测路径不受影响（docs/TRAINING_COMPILE.md）")
    ap.add_argument("--anchor-amp", type=int, default=0, choices=[0, 1],
                    help="1=anchor 参考前向也用 --amp 的精度（bf16，实测约 1.8×）；"
                         "只影响 KL 正则项参考分布，默认 0=旧行为 fp32")
    ap.add_argument("--pi-coef", type=float, default=0.1)
    # ---- 动作序列辅助任务（默认关闭，见 docs/ACTION_SEQUENCE_AUX.md）----
    ap.add_argument("--seq-coef", type=float, default=0.0,
                    help="动作序列辅助头（该 seat 在本 turn 内动作 multi-hot 的 BCE）权重；"
                         "0=关闭（默认，records→batch→loss 与旧路径逐字节等价）")
    ap.add_argument("--seq-coef-ramp", default="",
                    help='按 decisions 比例的线性 ramp，如 "0.02@0,0.05@0.5"；'
                         "给了 ramp 时本 iteration 的系数由 ramp 决定")
    ap.add_argument("--seq-ramp-total-decisions", type=float, default=0.0,
                    help="ramp 的 100%% 进度对应的累计 decisions（0=用 --stop-at-decisions，"
                         "再缺省 30M）")
    # ---- 真实人类 BC 辅助正则（默认关闭，见 docs/BC_AUX.md）----
    ap.add_argument("--bc-coef", type=float, default=0.0,
                    help="真实人类对局（Firestone）BC 辅助 CE 权重（形态 A）；"
                         "0=关闭（默认，不加载 BC、不采样、loss 与旧路径逐位等价）")
    ap.add_argument("--bc-coef-ramp", default="",
                    help='BC 系数 ramp（与 --seq-coef-ramp 同语法），如 "0.005@0,0.02@0.3"；'
                         "按累计 decisions 求值后写回 bc_coef")
    ap.add_argument("--bc-mmr-min", type=int, default=6000,
                    help="BC 样本 MMR 下限（低于该值丢弃；默认 6000）")
    ap.add_argument("--bc-mmr-boost", type=float, default=1.0,
                    help="MMR 线性增强：1=不加权（默认）；1.5=在 6000..8000 上从 1.0 线性升到 1.5")
    ap.add_argument("--bc-class-weight", default="",
                    help='稀有动作类别权重覆盖，如 "upgrade=4,hero_power=2"；'
                         "在 class-balanced 权重之上再乘。空=现状（默认）")
    ap.add_argument("--bc-samples", default="data/firestone_bc/bc_samples.jsonl.gz",
                    help="BC 样本 JSONL(.gz) 路径（默认仓库 data/firestone_bc/bc_samples.jsonl.gz）")
    ap.add_argument("--bc-batch", type=int, default=128,
                    help="每个 PPO 微批抽取/编码的 BC 样本数（默认 128；固定形状编译）")
    ap.add_argument("--bc-every", type=int, default=1,
                    help="每 N 个 PPO 微批做一次 BC（默认 1=每个微批）；N>1 时该步 "
                         "BC 权重乘 N 保持等效平均权重，跳过步不跑 forward_bc")
    ap.add_argument("--bc-compile", type=int, default=1, choices=[0, 1],
                    help="--compile-train>0 时是否也把 BC 批次单独 torch.compile（默认 1）；"
                         "0=BC 保持 eager（A/B 基线，其余不变）")
    ap.add_argument("--bc-type-weight", type=float, default=1.0,
                    help="分层 BC 的类型级 CE 权重（9 类 log-sum-exp 聚合；默认 1.0）")
    ap.add_argument("--bc-card-weight", type=float, default=0.5,
                    help="分层 BC 的卡牌身份 CE 权重（3983 类 card_head，仅 buy/play；"
                         "默认 0.5；0=不建/不用 card_head）")
    ap.add_argument("--dpo-coef", type=float, default=0.0,
                    help="upgrade 子集 one-sided DPO 系数（0=关闭，默认）；每 "
                         "--dpo-every 个 iteration 把 coef*L_dpo 加进同一次 backward，"
                         "见 docs/DPO_UPGRADE.md")
    ap.add_argument("--dpo-beta", type=float, default=0.1,
                    help="DPO 温度 β（-logsigmoid(β·Δlogp)；默认 0.1）")
    ap.add_argument("--dpo-batch", type=int, default=128,
                    help="每个 DPO 步从 upgrade 池无放回抽样的样本数（默认 128）")
    ap.add_argument("--dpo-every", type=int, default=4,
                    help="DPO 周期，按 **iteration** 计（默认 4=每 4 个 iteration 一次）；"
                         "系数不做周期折算，平均梯度占比 ≈ coef/every")
    ap.add_argument("--dpo-samples", default="data/firestone_bc/bc_samples.jsonl.gz",
                    help="upgrade 样本来源 JSONL(.gz)（只读 action_type==upgrade 的行）；"
                         "默认 data/firestone_bc/bc_samples.jsonl.gz")
    # ---- 场面题（自对弈关键局面 + 人类参考解）与强度辅助头（默认关闭）----
    # 见 docs/RESEARCH_PLAN_V2.md §4.1 与 docs/BOARD_PROBLEMS.md。
    ap.add_argument("--problem-pool", default="data/board_problems/problems.jsonl",
                    help="场面题池（workflows/build_board_problems.py 产出）；"
                         "两个系数都为 0 时不读该文件")
    ap.add_argument("--problem-bc-coef", type=float, default=0.0,
                    help="场面题 soft-label CE 系数（0=关闭，默认）；只在题目状态上算，"
                         "与 BC/DPO 正交。>0 需要 --problem-pool 存在且模型支持 "
                         "forward_problem")
    ap.add_argument("--problem-bc-every", type=int, default=1,
                    help="场面题 CE 周期（按 iteration；默认 1=每次）")
    ap.add_argument("--problem-batch", type=int, default=128,
                    help="每批题目行数（题目 CE 与强度辅助头共用同一批 + 同一次前向）")
    ap.add_argument("--problem-turn-weight", default="",
                    help="场面题 turn 加权（默认空 = 不加权，逐位等价）："
                         "'10:2,12:3,14:4,15:4'；未列出的 turn 权重 1，权重 0 = "
                         "该 turn 的题从池里剔除。非法语法直接报错退出")
    ap.add_argument("--problem-min-strength-delta", type=float, default=0.0,
                    help="场面题强度增量过滤（默认 0 = 不过滤）：只保留 "
                         "label.strength_delta（t+3 log1p 增量）>= 阈值的题")
    ap.add_argument("--strength-aux-coef", type=float, default=0.0,
                    help="强度辅助头系数（0=关闭，默认）：t+3 战团强度增量的 log1p "
                         "回归 + tier+1 二分类，只在有标注（t+3 时存活）的题目上算")
    ap.add_argument("--strength-aux-every", type=int, default=1,
                    help="强度辅助头周期（按 iteration；默认 1=每次）")
    ap.add_argument("--strength-tier-pos-weight", type=float, default=1.0,
                    help="强度辅助头 tier+1 BCE 的正样本权重（默认 1.0）")
    ap.add_argument("--strength-head-init", default="fit", choices=["fit", "random"],
                    help="强度辅助头初始化：fit=在题目池上离线拟合线性头（CV 选 lambda "
                         "的 ridge + logistic，并冻结；推荐，40-60 iter 的小 lr 下随机头"
                         "无法校准）、random=保持 nn.Linear 随机初始化（纯消融）")
    ap.add_argument("--strength-head-ridge", type=float, default=1.0,
                    help="离线拟合的 ridge 强度（乘以样本数；默认 1.0）")
    # ---- 经济（金币）课程（默认关闭 = 逐位等价；见 docs/CURRICULUM_SCALING.md）----
    ap.add_argument("--gold-curriculum", type=int, default=0,
                    help="1=启用经济课程（--curriculum-stage 仍为 none 时按 warmup 处理）；"
                         "默认 0=关闭")
    ap.add_argument('--recruit-start-bank',default='',help='Verified whole-state curriculum directory')
    ap.add_argument('--recruit-start-p',type=float,default=0.,help='Episode probability for verified recruit starts')
    ap.add_argument("--curriculum-stage", default="none",
                    choices=["none", "warmup", "anneal"],
                    help="none=关闭（默认，逐位等价）；warmup=金币收入 ×mult 在 "
                         "--curriculum-anneal-decisions 内线性降到 1.0；anneal=先保持 "
                         "mult 半个窗口再线性降到 1.0")
    ap.add_argument("--curriculum-gold-mult", type=float, default=1.0,
                    help="金币收入倍数（默认 1.0；warmup 建议 3.0）")
    ap.add_argument("--curriculum-anneal-decisions", type=int, default=10_000_000,
                    help="线性退火窗口（累计 decisions；默认 10M）")
    # ---- 缩放教师 DAgger / 过程价值（默认关闭 = 逐位等价）----
    ap.add_argument("--teacher-bc-coef", type=float, default=0.0,
                    help="在线缩放教师 CE 系数（0=关闭，默认）；采集侧只在 "
                         "gold>=10 且 tier>=4 的决策上产出教师标签")
    ap.add_argument("--process-value-coef", type=float, default=0.0,
                    help="过程价值辅助头系数（0=关闭，默认）：预测 log1p(酒馆永久 buff) / "
                         "log1p(施法数) / log1p(consume 数)")
    ap.add_argument("--entropy-coef", type=float, default=0.005)
    ap.add_argument("--max-actions-per-turn", type=int, default=100,
                    help="每回合最多几次决策（超限强制 EndTurn）。"
                         "2026-11-17 D-070：80→100（原值几乎从不触发，见 "
                         "docs/analysis/ACTION_CAP_20261117.md）")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=1,
                    help=">1 时使用并行采集 + 共享批量叶 server（非 pool 路径）")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--processes", type=int, default=1,
                    help=">1 时用进程级 actor pool（每进程绑定一张 GPU）")
    ap.add_argument("--gpus", default="",
                    help="actor GPU编号列表；默认physical，Slurm须配合--gpu-index-space visible使用逻辑编号")
    ap.add_argument('--gpu-index-space', choices=['physical', 'visible'], default='physical',
                    help='visible=保留Slurm CUDA_VISIBLE_DEVICES，gpus与device使用逻辑编号')
    ap.add_argument('--leaf-fused', type=int, choices=[0,1], default=0,
                    help='pool纯动作批保留logits在GPU，减少往返；默认关闭')
    ap.add_argument('--pipeline-zero-step-fallback',type=int,choices=[0,1],default=0,
                    help='旧策略批次PPO零更新时，本进程改用同步采集，保留下一已发批次；不放宽KL保护')
    ap.add_argument("--gpu-free-check", type=int, default=1,
                    help="1=启动 actor pool 前用 nvidia-smi 剔除已被占用的 GPU"
                         "（邻居抢占；0=不检查）")
    ap.add_argument("--gpu-free-mib", type=int, default=1024,
                    help="--gpu-free-check 阈值：已用显存 > 该值（MiB）视为被占用")
    ap.add_argument("--threads", type=int, default=16,
                    help="每个 actor 进程内的采集线程数（每轮 games 应 >= processes*threads）")
    # ── records 数据通路（默认 auto/fp32：进程 pool 走 shm，features 仍 fp32 存储；
    #    `--records-dtype fp16` 是可选省字节项，但**不保证逐位等价**。见 docs/TRAIN_THROUGHPUT.md）──
    ap.add_argument("--records-dtype", default="fp32", choices=["fp32", "fp16"],
                    help="records 存储精度：fp32（默认，逐位等价）；fp16 只把 features(=3227 "
                         "floats, 占 88%% 字节) 降精度存，父进程读入后立刻还原 fp32。"
                         "实测：小配置 12 iter 动作流 sha 全同，但 384 局/8 进程配置下第 3 iter "
                         "起与 fp32 分歧 ⇒ 省 ~2-3%% 时间，不用于需要逐位一致的正式训练")
    ap.add_argument("--ipc-mode", default="auto", choices=["auto", "disk", "shm"],
                    help="worker→父进程的 records 通路：auto（默认）= 有进程 pool 且 /dev/shm "
                         "可用时走 shm，否则 disk；shm=/dev/shm 环形缓冲 + 父进程零拷贝映射"
                         "（省掉 335-438 MB/iter 写盘+读盘）；disk=npz 落盘再读回（历史路径）")
    ap.add_argument("--ipc-shm-slots", type=int, default=2,
                    help="--ipc-mode shm 时每个 worker 的环形 slot 数（>=2 让父进程读上一轮时"
                         " worker 能写下一轮；容量按 32 MiB 向上取整）")
    ap.add_argument("--collect-sha", type=int, default=0, choices=[0, 1],
                    help="1=把每个 iteration 采集批次的「动作流 sha256」写进 history（actions/"
                         "turns/game_ids/seats 拼接后哈希）；默认 0=不写，字段与旧版一致")
    ap.add_argument("--pipeline", type=int, default=0,
                    help="1=double-buffer：collect(k+1) 与 update(k) 并行（需 --processes>1）")
    ap.add_argument("--pipeline-guard", default="warn", choices=["warn", "abort"],
                    help="流水线的 ratio 守卫触发时：warn 只记录，abort 立即保存并退出")
    ap.add_argument("--keep-versions", type=int, default=3,
                    help="保留几个历史权重文件（流水线用；每个 61 MB）")
    ap.add_argument("--snapshot-every", type=int, default=1,
                    help="每 N 个 iteration 落一次快照（进程 actor 下建议 5）")
    ap.add_argument("--keep-snapshots", type=int, default=3,
                    help="只保留最近 K 个 snapshot 目录（0=不清理）")
    ap.add_argument("--anchor-beta", type=float, default=0.0,
                    help="KL anchor 系数 β（论文 Table 7 = 0.1；0=关闭）")
    ap.add_argument("--reset-anchor", type=int, default=0,
                    help="1=把 anchor 重建为当前策略（fork/续训时修正「anchor 指向更早旧策略」）")
    ap.add_argument("--anchor-iterations", type=int, default=100,
                    help="β 生效的前 N 个 iteration（论文 300 个中的前 100）")
    ap.add_argument("--anchor-beta-tail", type=float, default=0.0,
                    help="前 N 个 iteration 之后保留的 anchor β（stage-3 建议 0.02；0=旧行为）")
    ap.add_argument("--eval-every-decisions", type=int, default=0,
                    help="每累计 N decisions 做一次 C0 评测（0=关闭）")
    ap.add_argument("--eval-games", type=int, default=512)
    ap.add_argument("--eval-seats", type=int, default=8)
    ap.add_argument("--eval-seed-base", type=int, default=420000)
    ap.add_argument("--eval-async", type=int, default=0,
                    help="1=异步评测：ckpt 交给独立子进程跑 c0_gate，主循环不等待"
                         "（并发上限 1；默认 0=同步阻塞，旧行为）")
    ap.add_argument("--eval-device", default="",
                    help="评测用的 GPU（同步默认跟随 --device；异步默认 cuda:4）")
    ap.add_argument("--eval-drain", type=int, default=0,
                    help="1=训练退出时等待在跑的异步评测完成（默认 0=直接终止，不拖 wall）")
    ap.add_argument("--eval-stop-mean-rank", type=float, default=0.0,
                    help="评测 mean_rank <= 该值时在下一个 iteration 边界停止训练"
                         "（0=关闭；异步/同步共用，结果到达时判定）")
    ap.add_argument("--eval-panel", default="both", choices=["bot", "human", "both"],
                    help="周期评测面板：bot=256/512 局 bot 面板（已饱和）；human=人类战团"
                         "对手面板（新主指标）；both=两个面板各起一个子进程串行排队"
                         "（默认 both；并发上限仍为 1）")
    ap.add_argument("--eval-human-quantile", default="auto",
                    choices=["auto", "p50", "p75", "p90"],
                    help="human 面板的对手强度分位：auto=t<=8 p50 / 9-12 p75 / >=13 p90"
                         "（默认，见 docs/EVAL_PANELS.md）")
    ap.add_argument("--eval-human-boards", default="",
                    help="human 面板使用的人类战团 JSONL（空=默认 "
                         "data/real_library/library_real_degraded.jsonl）")
    ap.add_argument("--eval-human-baseline-bot", type=int, default=0,
                    help="1=human 面板同 seed 再跑 bot 基线（配对 CI；默认 0=只评 learner，"
                         "省一半 wall；TRIGGER/早停只看 bot 面板）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default="")
    # ── critic（价值/名次头）预热：BC 只训策略 ⇒ critic 头仍是随机初始化，此时 GAE 的
    #    advantage 与真实优势近似无关，PPO 第一个 iteration 就会把策略推歪
    #    （沙盒实测 anchor_kl 首行 0.205、130k 后 Σ 1646→524）。iters>0 时在**第一个
    #    iteration 的 update 之前**只训 critic 头（冻结策略与 trunk，策略漂移由构造为 0）。
    #    默认 0 = 关闭，逐位等价旧路径。
    ap.add_argument("--value-warmup-iters", type=int, default=0,
                    help="critic 预热遍数：只训 rank 头（--value-head-coef>0 时含标量 "
                         "value 头），冻结策略/trunk；0=关闭（默认）")
    ap.add_argument("--value-warmup-lr", type=float, default=1e-3,
                    help="critic 预热的 Adam lr（默认 1e-3；仅 --value-warmup-iters>0 生效）")
    ap.add_argument("--value-warmup-minibatch", type=int, default=1024,
                    help="critic 预热的 minibatch 行数（默认 1024；仅 iters>0 生效）")
    ap.add_argument("--value-warmup-every", type=int, default=0,
                    help="周期性重热 critic 的周期（按 iteration；0=只在第一个 iteration 预热）。"
                         "预热的效果会被后续 PPO 的 trunk 漂移抹掉，长训练建议 50–100 重热一次")
    ap.add_argument("--upgrade-reward", type=float, default=0.0,
                    help="给 **upgrade 动作（action_space_std.A_UPGRADE=3）** 的即时奖励 w"
                         "（默认 0=关闭，逐位等价）。rank-only 下升本收益要 5–8 回合后才兑现，"
                         "在 GAE λ=0.995 下被压没；这个'局部可行'信号直接解决'升本动作没有"
                         "正收益'。实现：加在 reward 上并**重算 GAE**（advantages/returns）")
    ap.add_argument("--win-only-reward", type=float, default=0.0,
                    help="终局**只有第 1 名** +w、其余 0（默认 0=关闭，逐位等价）。"
                         "用于「减弱名次奖励、只奖励夺冠」，通常配 --rank-utility-coef 0")
    ap.add_argument("--strength-reward-coef", type=float, default=0.0,
                    help="每回合 ΔL(Σ) 密集强度奖励系数（默认 0=关闭，逐位等价）："
                         "L=log1p(Σ)/log1p(30000)，回合末决策加 c×(L_t−L_{t−1})")
    # ── 动态课程（默认 0 = 关闭，逐位等价）───────────────────────────────────
    # 规则（Lead 定）：起点 start 席 × p；**每当真实强度口径（5 席 p0.9 面板的 t15 Σ p50）
    # 比上一次评测提升 >= threshold，就回加一席**（上限 max）；回加后若比上一次评测
    # 回退 > threshold，则退回一档并保持 hold-decisions。训练剂量只作参考，主判据用真实口径。
    ap.add_argument("--dyn-curriculum", type=int, default=0, choices=[0, 1],
                    help="1=启用动态对手强度课程（按真实强度口径 t15 Σ 自动加/减席）")
    ap.add_argument("--dyn-start-seats", type=int, default=2,
                    help="动态课程起始人类战团席数（默认 2）")
    ap.add_argument("--dyn-max-seats", type=int, default=5,
                    help="席数上限（默认 5）")
    ap.add_argument("--dyn-p", type=float, default=0.75,
                    help="动态课程使用的注入概率 p（默认 0.75）")
    ap.add_argument("--dyn-threshold", type=float, default=0.10,
                    help="加/减席的判定阈值（相对上一次评测的 t15 Σ 变化；默认 0.10=10%%）")
    ap.add_argument("--dyn-hold-decisions", type=float, default=10000000.0,
                    help="减席后至少保持多少 decisions 再考虑加席（默认 10M）")
    ap.add_argument("--dyn-quality-games", type=int, default=96,
                    help="真实强度口径面板局数（5 席 p0.9，默认 96）")
    ap.add_argument("--trainer-state", default="")
    # ── 健康门（2026-11-17 审计 P2-11）：把 `TRAINING_RECIPE §5` 的红线变成代码 ──
    # 硬门（工程错误，fail-closed，默认开）：非法动作/无效局/截断/引擎停止/注入率崩塌/
    # 非有限损失/非法动作概率泄漏。软警告（只记录）：熵、KL、吞吐。
    ap.add_argument("--health-gate", type=int, default=1, choices=[0, 1],
                    help="1=启用健康门（默认）；0=关闭（仅用于对照/复现旧作业）")
    ap.add_argument("--health-illegal-actions", type=int, default=0)
    ap.add_argument("--health-invalid-games", type=int, default=0)
    ap.add_argument("--health-truncated-games", type=int, default=0)
    ap.add_argument("--health-truncated-combats", type=int, default=0)
    ap.add_argument("--health-engine-stopped", type=int, default=0)
    ap.add_argument("--health-mirror-apply-min", type=float, default=0.99,
                    help="镜像注入成功率下限（0=关闭该项检查；纯 bot 作业应设 0）")
    ap.add_argument("--health-entropy-warn-below", type=float, default=0.05,
                    help="策略熵软警告阈值（只记录，不停机）。默认 0.05 = 只抓崩塌："
                         "本游戏早期仅 2 个合法动作，最大熵 ln2=0.693，0.8 会每轮误报")
    ap.add_argument("--health-kl-warn-above", type=float, default=0.03,
                    help="近似 KL 软警告阈值（只记录，不停机）")
    ap.add_argument("--health-throughput-warn-below", type=float, default=0.0,
                    help="吞吐软警告阈值 决策/秒（只记录，不停机；**默认 0=关闭**，"
                         "小规模 smoke 必然低于固定阈值）")
    ap.add_argument("--teacher-loss-normalize", default="labeled",
                    choices=["labeled", "valid"],
                    help="教师 CE 的分母：labeled=改动前语义（对有标签行取均值）；"
                         "valid=与策略项同分母（coef 即恒定权重比）")
    ap.add_argument("--out", default="runs/az_entity_sync")
    return ap
