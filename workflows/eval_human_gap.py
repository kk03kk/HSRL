#!/usr/bin/env python3
"""Human-gap panel: measure our models against the Firestone human baseline.

Why this exists
---------------
Every claim of the form "the model is weak late game" is meaningless until it is
expressed in the same units as a human. This workflow plays N games per
checkpoint against the seven engine bots, records the per-turn economy rows
(``EvalConfig.collect_economy``) and reduces them to the exact same statistics
that :mod:`tools.analyze_human_baseline` extracts from the replays:

* tavern tier by turn (level-up tempo);
* board strength by turn, where strength is ``Σ(atk + hp)`` of the
  recruitment-end board — byte-for-byte the same definition the human data uses
  (``stage == combat_start``);
* gold, board size, actions per turn;
* how deep games actually go (turn 12 / 14 survival).

Caveats this panel cannot remove (stated so nobody over-reads it)
----------------------------------------------------------------
* Humans played 8 real opponents; our models play 7 engine bots. Tier tempo and
  board-growth are mostly *intrinsic* to the player, but rank outcomes are not
  comparable at all — compare curves, never mean placement.
* The human feed is winner-biased (816/995 are 1st place), so the human curve is
  "what winners do", i.e. a demanding yardstick, not an average.

Usage
-----
::

    python workflows/eval_human_gap.py \
        --checkpoint v1.1=artifacts/models/v1.1/model.pt \
        --checkpoint final14858=artifacts/models/league-final-20261111-step014858/model.pt \
        --games 128 --device cuda:0 \
        --human-baseline artifacts/runs/human-baseline-20261112/baseline.json \
        --out artifacts/runs/human-gap-<date>/panel.json

    # tempo probe: force the human tier curve and see how much of the gap closes
    python workflows/eval_human_gap.py ... --forced-tier-plan "2:2,5:3,6:4,8:5"
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsbg.az.train.common import engine_info  # noqa: E402
from hsbg.az.train.eval_policy import (  # noqa: E402
    EvalProtocol,
    make_eval_policy,
    protocol_from_args,
)
from hsbg.az.train.evaluate import EvalConfig, evaluate_policy  # noqa: E402
from hsbg.az.train.lobby_filter import parse_race_list  # noqa: E402
from hsbg.az.train.thirdparty_policy import is_thirdparty_spec  # noqa: E402
from hsbg.evaluation.metrics import (
    _pct,
    _r,
    economy_curves,
    survival,
    action_mix_by_turn,
    combat_by_turn,
    final_boards,
    mechanics_by_turn,
    retained_gain_by_turn,
    boards_by_turn,
    _strength_by_seed,
    _roster,
    _trace,
    _counts_by_turn,
    _counts_all,
    _opponents_by_turn,
    _gold_by_turn,
    compare
)


def sha256_file(path) -> str:
    # `thirdparty:/path` 这种 spec 不是真实路径；哈希要落到真文件上
    if is_thirdparty_spec(str(path)):
        from hsbg.az.train.thirdparty_policy import split_thirdparty
        path = Path(split_thirdparty(str(path)))
    elif not isinstance(path, Path):
        path = Path(str(path))
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_name(name: str) -> str:
    """模型名 → 安全文件名片段（只留字母数字与 ``._-``）。"""
    keep = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(name))
    return keep.strip("._-")[:64]


def _preflight_outputs(names, out, dump_rows=True, record_decisions=False):
    """Reject ambiguous names and existing evidence before loading any model."""
    safe = [_safe_name(name) for name in names]
    if any(not name for name in safe) or len({s.casefold() for s in safe}) != len(safe):
        raise SystemExit("checkpoint names must have unique nonempty safe filenames")
    out = Path(out)
    targets = [out]
    if record_decisions:
        for name in safe:
            trace = out.parent / f'decisions-{name}.jsonl.gz'
            targets.extend([trace, Path(str(trace)+'.partial'), Path(str(trace)+'.manifest.json')])
    if dump_rows:
        for suffix in [""] + [f"-{name}" for name in safe]:
            targets.extend([out.parent / f"rows{suffix}.jsonl.gz",
                            out.parent / f"extra{suffix}.json"])
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise SystemExit("refusing to overwrite evaluation evidence: " + ", ".join(existing))
    if len({str(p.resolve()).casefold() for p in targets}) != len(targets):
        raise SystemExit("panel output conflicts with row/extra output")
    out.parent.mkdir(parents=True, exist_ok=True)


def _field_stats(result: Dict[str, Any], mirror_pool: Any) -> Dict[str, Any]:
    """对手场是否**真的**生效（审计 P1-3：以前算完就丢，面板无法自证）。

    关键量：``mirror_applied/attempt``（成功注入率）、``missing/rejected``（退回 bot 的
    座位数）、``pending_resolved``（注入前被动结算的次数）、以及库本身的规模。
    只要 ``applied/attempt`` 明显低于 1，这一轮的"对手是真人战团"就不成立。
    """
    out: Dict[str, Any] = {}
    mirror = result.get("mirror")
    if isinstance(mirror, dict):
        out.update({f"mirror_{k}" if not str(k).startswith("mirror_") else str(k): v
                    for k, v in mirror.items()})
    for key in ("mirror_skipped_ended", "mirror_pending_resolved",
                "mirror_strict_counts", "human_board", "forced_tier",
                "forced_tier_actions"):
        if key in result and result[key] is not None:
            out[key] = result[key]
    attempt = int(out.get("mirror_attempt") or 0)
    applied = int(out.get("mirror_applied") or 0)
    out["mirror_apply_rate"] = (round(applied / attempt, 4) if attempt else None)
    if mirror_pool is not None:
        try:
            stats = mirror_pool.stats()
            out["library"] = {
                "root": str(getattr(mirror_pool, "root", "") or ""),
                "frames": int(stats.get("frames") or 0),
                "keys": int(stats.get("keys") or 0),
                "turn_min": stats.get("turn_min"),
                "turn_max": stats.get("turn_max"),
                "sample": stats.get("sample"),
                # 每个 (turn,seat) 桶的帧数中位数：桶太小 → 7 席可能拿到同一帧。
                "frames_per_key_p50": stats.get("frames_per_key_p50"),
                "schema_versions": stats.get("schema_versions"),
            }
        except Exception as exc:      # noqa: BLE001 - 诊断字段缺失不影响评测
            out["library"] = {"error": repr(exc)[:200]}
    return out


def run_checkpoint(name: str, path: Path, args,
                   protocol: Optional[EvalProtocol] = None) -> Dict[str, Any]:
    """评测一个 checkpoint，返回面板里的单模型条目。

    **策略构造**统一走 :func:`hsbg.az.train.eval_policy.make_eval_policy`：
    2026-11-17 审计发现这里曾把 ``PPOEvaluator`` 的构造删掉却留下
    ``if evaluator is None:``，导致**所有自家 checkpoint** 都 ``UnboundLocalError``
    （P0-1）。根因是"构造策略"这件事散落在 800 行脚本里，现在收敛到那个模块。
    """
    if protocol is None:
        protocol = protocol_from_args(args)
    # 契约与训练不一致就直接拒绝出分（fail-closed）。第三方权重走的是作者的
    # 训练契约，契约项同样需要先自检，所以这里对所有 spec 都做。
    protocol.assert_training_contract()

    built = make_eval_policy(str(path), protocol=protocol, device=args.device)
    policy, model, meta = built["policy"], built["model"], built["meta"]
    if built["kind"] == "thirdparty":
        print(f"[{name}] 第三方权重：{built['real_path']}  z={getattr(policy, 'z', '?')}",
              flush=True)
    pool = None
    if str(getattr(args, "human_opponents", "") or "").strip():
        from hsbg.az.train.human_board_pool import HumanBoardPool
        pool = HumanBoardPool(str(args.human_opponents),
                              sample=str(getattr(args, "human_opponent_sample",
                                                 "auto") or "auto"))
        print(f"[{name}] 对手换成真人战团：{args.human_opponents} "
              f"sample={getattr(args, 'human_opponent_sample', 'auto')} "
              f"turns={sorted(pool.turns())[:3]}...", flush=True)
    mirror_pool = None
    if str(getattr(args, "mirror_snaps", "") or "").strip():
        from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
        mirror_pool = MirrorSnapshotPool(
            str(args.mirror_snaps), sample=str(getattr(args, "mirror_sample", "p50band")))
        if int(getattr(mirror_pool, "frames", 0) or 0) <= 0:
            raise SystemExit(f"快照库为空/读不了: {args.mirror_snaps}")
        print(f"[{name}] 对手换成快照库（36h 同款注入）：{args.mirror_snaps} "
              f"frames={mirror_pool.frames} sample={args.mirror_sample} "
              f"strict={int(getattr(args, 'mirror_strict', 0) or 0)}", flush=True)
    cfg = EvalConfig(
        games=int(args.games),
        seed=int(args.seed),
        policy_seat=0,
        max_steps=int(args.max_steps),
        collect_economy=True,
        collect_action_hist=True,
        # 专家先验 G1：Σ(atk+hp) 不是战力，必须同时看**实际战斗胜负**。
        collect_exposure=True,
        forced_tier_plan=str(args.forced_tier_plan or ""),
        # 2026-11-17（D-071）：与训练同口径的 lobby 种族排除。
        lobby_exclude=parse_race_list(getattr(args, "lobby_exclude", "")),
        human_board_pool=pool,
        human_board_set_tier=True,
        mirror_pool=mirror_pool,
        mirror_strict=bool(int(getattr(args, "mirror_strict", 0) or 0)),
        # 记录进协议的这两项必须真的生效（默认 7 / 1.0 = 改动前写死的取值）。
        mirror_seats=protocol.mirror_seats,
        mirror_p=protocol.mirror_p,
    )
    checkpoint_sha = sha256_file(path)
    recorder = None
    if bool(getattr(args, 'record_decisions', False)):
        from hsbg.evaluation.trace import DecisionRecorder
        recorder = DecisionRecorder(Path(args.out).parent / f'decisions-{_safe_name(name)}.jsonl.gz',
                                    identity={'name': name, 'checkpoint_sha256': checkpoint_sha,
                                              'protocol': {**protocol.as_dict(), 'protocol_sha256': protocol.sha256()}})
        cfg.decision_recorder = recorder
    started = time.time()
    try:
        result = evaluate_policy(policy, cfg)
        trace_manifest = recorder.finish(result) if recorder is not None else None
    except BaseException:
        if recorder is not None:
            recorder.abort()
        raise
    # evaluate_policy returns end_turn_economy / action_hist at the TOP level
    # (the seats[] shape belongs to the saved multi-seat report format).
    rows = result.get("end_turn_economy") or []
    hist = result.get("action_hist")
    # 2026-11-16：分片并行评测需要「无损合并」。做法是**不重写聚合逻辑**，而是把
    # 喂给它们的原始 rows（以及 outcome/exposure）原样落盘；合并时把各分片的 rows
    # 拼起来，再调用本文件**同一批函数**重算 —— 等价性由构造保证，不靠人工对齐口径。
    #
    # 2026-11-17 审计 P0-3：文件名原来固定成 `rows.jsonl.gz` / `extra.json`，一个进程
    # 跑多个 `--checkpoint` 时**后者覆盖前者**，合并工具只能重建出最后一个模型
    # （`gap-v1.2-plus3-1024` 的 panel.json 就只剩 v1.3_r3）。现在按模型名分文件；
    # 旧名（无模型后缀）仍**照写一份**，保证只认旧名的老合并脚本不失效。
    extra_payload = {
        "name": name,
        "decision_trace": trace_manifest,
        "outcome": {k: result.get(k) for k in
                    ("mean_rank", "top4_rate", "win_rate", "illegal_actions",
                     "truncated_games", "decisions", "decisions_per_game")},
        "rank_sequence": result.get("rank_sequence"),
        # 逐局种子（与 rank_sequence 一一对应）——配对统计必须按 seed 配对，
        # 不能按下标（审计 P2-6：缺一个中间分片就会整体错位）。
        "seed_sequence": result.get("seed_sequence"),
        "action_hist": hist,
        "combat_exposure": result.get("combat_exposure") or [],
        "seconds": round(time.time() - started, 1),
        "games": int(args.games),
        "seed": int(args.seed),
        "protocol_sha256": protocol.sha256(),
        # 2026-11-17（验证发现）：分片的 extra 以前不含**模型身份**，于是合并后的
        # 判胜面板丢了 checkpoint / sha256 / arch —— 无法自证"评的是哪个权重"。
        "checkpoint": str(path),
        "checkpoint_sha256": checkpoint_sha,
        "arch": meta.get("arch"),
        "seed_start": int(args.seed),
        "seed_end": int(args.seed) + int(args.games) - 1,
        # 注入是否真的生效（审计 P1-3：以前算完就丢，面板无法自证对手是真人战团）。
        "field_stats": _field_stats(result, mirror_pool),
    }
    if bool(getattr(args, "dump_rows", 1)):
        _d = Path(args.out).parent
        # 2026-11-17：以前这里假设 dir 已存在（sbatch 里确实会 mkdir），但
        # `--out <新目录>/panel.json` 直接跑就会 FileNotFoundError 并**静默丢掉**
        # rows/extra —— 而合并工具依赖这两个文件。自己建一下。
        _d.mkdir(parents=True, exist_ok=True)
        _safe = _safe_name(name)
        # 命名顺序固定：先按模型名的新文件，再写旧名兼容文件。
        for suffix in ([f"-{_safe}"] if _safe else []) + [""]:
            rows_path = _d / (f"rows{suffix}.jsonl.gz")
            extra_path = _d / (f"extra{suffix}.json")
            try:
                with gzip.open(rows_path, "wt", encoding="utf-8") as _gz:
                    for _r in rows:
                        _gz.write(json.dumps(_r, ensure_ascii=False) + "\n")
                extra_path.write_text(json.dumps(extra_payload, ensure_ascii=False),
                                      encoding="utf-8")
            except OSError as exc:
                raise OSError(f"evaluation evidence export failed: {rows_path}") from exc
    return {
        "name": name,
        "decision_trace": trace_manifest,
        "checkpoint": str(path),
        "checkpoint_sha256": checkpoint_sha,
        "arch": meta.get("arch"),
        "games": int(args.games),
        "seed_start": int(args.seed),
        "seed_end": int(args.seed) + int(args.games) - 1,
        "seconds": round(time.time() - started, 1),
        "outcome": {k: result.get(k) for k in
                    ("mean_rank", "top4_rate", "win_rate", "illegal_actions",
                     "truncated_games", "decisions", "decisions_per_game")},
        # 逐局名次（同 `--seed` 下各模型打的是同一批种子，所以可以**配对**比较，
        # 显著性强于比两个均值；见 docs/analysis/ACCEPTANCE_BASELINE.md §4.5）。
        "rank_sequence": result.get("rank_sequence"),
        # 逐局种子：配对的**键**。`compare_panels` 优先按 seed 求交集配对，
        # 缺失时才退回按下标（并显式告警）。
        "seed_sequence": result.get("seed_sequence"),
        "action_hist": hist,
        # 注入/对手场的实际生效证据（applied/attempt/missing/rejected/pool）。
        "field_stats": extra_payload["field_stats"],
        "survival": survival(rows),
        "curves": economy_curves(rows),
        "action_mix_by_turn": action_mix_by_turn(rows),
        "combat_by_turn": combat_by_turn(result.get("combat_exposure") or []),
        "mechanics_by_turn": mechanics_by_turn(rows),
        "retained_gain_by_turn": retained_gain_by_turn(rows),
        "final_boards": final_boards(rows),
        "boards_by_turn": boards_by_turn(rows),
        # ── 2026-11-16 全量留档（用户要求：一次评测尽量多给信息）──────────────
        # 1) 逐局逐回合战力（带 seed，可做前后回合归因）
        "strength_by_seed": _strength_by_seed(rows),
        # 2) 完整追踪：每局每回合一条精简记录（身材/金币/手牌/饰品/英雄/动作数）
        "trace": _trace(rows),
        # 3) 饰品选择：逐回合 各饰品被选中的次数
        "trinkets_by_turn": _counts_by_turn(rows, "trinkets"),
        # 4) 实际遇到的对手强度（注入帧的 source_strength），逐回合分布
        "opponents_by_turn": _opponents_by_turn(result.get("combat_exposure") or []),
        # 5) 金币收支：逐回合 获得/花费（买牌/刷新/升本/其它）
        "gold_by_turn": _gold_by_turn(rows),
        # 6) 英雄分布
        "heroes": _counts_all(rows, "hero"),
    }


def _opponent_label(args: Any) -> str:
    """对手场的**如实**标签（以前用快照库时也写 "engine_bot x7"）。"""
    if str(getattr(args, "human_opponents", "") or "").strip():
        return (f"human_board_pool x{int(getattr(args, 'mirror_seats', 7))} "
                f"({args.human_opponents}, sample={getattr(args, 'human_opponent_sample', 'auto')})")
    if str(getattr(args, "mirror_snaps", "") or "").strip():
        return (f"snapshot_library x{int(getattr(args, 'mirror_seats', 7))} "
                f"({args.mirror_snaps}, sample={getattr(args, 'mirror_sample', '')}, "
                f"strict={int(getattr(args, 'mirror_strict', 0))}, "
                f"p={float(getattr(args, 'mirror_p', 1.0))})")
    return "engine_bot x7"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="human-gap panel")
    ap.add_argument("--checkpoint", action="append", default=[], required=True,
                    metavar="NAME=PATH", help="repeatable")
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--seed", type=int, default=930000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-steps", type=int, default=50000)
    ap.add_argument("--max-actions-per-turn", type=int, default=100)
    ap.add_argument("--forbid-unfreeze", type=int, default=1, choices=[0, 1])
    ap.add_argument("--forbid-cancel", type=int, default=1, choices=[0, 1],
                    help="冻结契约：全局禁用 Cancel（与训练 --forbid-cancel 1 一致）")
    ap.add_argument("--move-guard", type=int, default=1, choices=[0, 1],
                    help="冻结契约：换位守卫（与训练 --move-guard 1 一致）")
    ap.add_argument("--move-cap", type=int, default=2)
    ap.add_argument("--move-repeat-limit", type=int, default=2)
    ap.add_argument("--no-cancel-choose", type=int, default=0, choices=[0, 1],
                    help="只在存在待选时禁 Cancel（第三方作者要求；训练默认 0）")
    ap.add_argument("--freeze-cap", type=int, default=1,
                    help="每回合最多按几次 Freeze。**默认 1 = 训练口径**"
                         "（2026-11-17 D-070：2→1，Freeze 键每回合只能按一次）")
    ap.add_argument("--forced-tier-plan", default="",
                    help='e.g. "2:2,5:3,6:4,8:5" — diagnostic only, never for ranking')
    ap.add_argument("--forced-tier-prob", type=float, default=0.0,
                    help="诊断用：强制升本的命中概率（训练默认 0 = 纯教师）。"
                         ">0 时必须与训练一致，否则两侧不可比")
    ap.add_argument("--mirror-p", type=float, default=1.0,
                    help="每座位注入概率（与训练 --mirror-p 一致）")
    ap.add_argument("--mirror-seats", type=int, default=7,
                    help="注入到几个非学习席（与训练 --mirror-seats 一致）")
    ap.add_argument("--human-baseline", default="")
    #: 把 7 个对手从引擎 bot 换成**真人战团**（`HumanBoardPool`，按回合注入）。
    #: 为什么需要：对 7 个引擎 bot，champion 已经是 `mean_rank 1.177`（饱和），
    #: 名次类指标无法区分优劣；换成真人战团后名次重新有信息量。
    ap.add_argument("--human-opponents", default="",
                    help="真人战团库 JSONL；给空=保持 engine_bot 对手（默认，历史协议）")
    ap.add_argument("--human-opponent-sample", default="auto",
                    choices=["auto", "p50", "p75", "p90", "max"],
                    help="真人对手强度档（auto = turn<=8 p50 / 9..12 p75 / >=13 p90）")
    #: 用**快照库**（`--mirror-snaps`）当对手：这正是 36h 冠军用的注入方式
    #: （整席位快照 → ended_turn=true → 只带这块板子进战斗，回合开始引擎复位）。
    #: 与 `--human-opponents` 互斥；把真人库用 tools/build_snapshot_library.py 转换后即可喂进来。
    ap.add_argument("--mirror-snaps", default="",
                    help="快照库目录（含 snap_*.json.gz）；给空=不用快照对手")
    ap.add_argument("--mirror-sample", default="p50band",
                    help="桶内抽样：p50band/p75band/p90band 为带宽采样（默认）")
    ap.add_argument("--mirror-strict", type=int, default=0, choices=[0, 1],
                    help="1=活席必须使用库，禁止 bot 回退")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp-forbid-cancel", type=int, default=0, choices=[0, 1],
                    help="仅对第三方权重生效：是否全局禁用 Cancel（作者只要求 no_cancel_choose）")
    ap.add_argument("--keep-start-board", type=int, default=1, choices=[0, 1],
                    help="1=回合开始已在场的随从不可卖（第三方权重要求 keep_start_board=1）")
    ap.add_argument("--ban-trinkets", type=int, default=1, choices=[0, 1],
                    help="1=禁用加重护手(BG36_MagicItem_414)等黑名单待选卡（默认开）")
    ap.add_argument("--lobby-exclude", default="",
                    help="与训练同口径的 lobby 种族排除（如 MECHANICAL,DRAGON；空=不过滤）")
    ap.add_argument('--record-decisions', type=int, default=1, choices=[0, 1],
                    help='stream every learner action, exact pre/post state and observation')
    ap.add_argument("--dump-rows", type=int, default=1, choices=[0, 1],
                    help="把喂给聚合的原始 rows/outcome 落盘（rows.jsonl.gz + extra.json），"
                         "供分片并行评测无损合并；0=关闭")
    args = ap.parse_args(argv)

    specs = []
    for spec in args.checkpoint:
        if "=" not in spec:
            raise SystemExit(f"--checkpoint needs NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        if is_thirdparty_spec(str(path)):
            specs.append((name, str(path)))   # thirdparty: 前缀不是文件路径
            continue
        p = Path(path)
        if not p.is_file():
            raise SystemExit(f"checkpoint not found: {p}")
        specs.append((name, p))

    _preflight_outputs([name for name, _ in specs], args.out, bool(args.dump_rows), bool(args.record_decisions))

    human = {}
    if args.human_baseline:
        hp = Path(args.human_baseline)
        if hp.is_file():
            human = json.loads(hp.read_text(encoding="utf-8"))
        else:
            print(f"WARN: human baseline missing at {hp}; comparison disabled", file=sys.stderr)

    engine = engine_info()
    # ── 协议指纹（审计 P0-2）：`docs/EVAL_PROTOCOL.md` §4 规则 5 要求"不同
    # protocol_sha256 禁止混比"，而这条规则以前在 gap 面板路径上**完全没实现**。
    # 注意 seed/games 是分片级参数，按设计**不进**指纹（见 eval_policy 的说明）。
    protocol = protocol_from_args(args, engine=engine, mirror_seats=args.mirror_seats)
    protocol.assert_training_contract()
    protocol_payload = protocol.as_dict()
    protocol_sha = protocol.sha256()

    payload: Dict[str, Any] = {
        "schema": "hsai.human_gap_panel/v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "engine": engine,
        # 协议（可比性的唯一依据）：指纹 + 全部字段，供 compare_panels 校验。
        "protocol": {**protocol_payload, "protocol_sha256": protocol_sha},
        "panel": {
            # 2026-11-17：以前这里恒为 "engine_bot x7"（用快照库时也是），
            # `docs/EVAL_PROTOCOL.md` §8.6 只能叫读者"别信这个字段"。现在如实填写。
            "opponent": _opponent_label(args),
            "games": int(args.games),
            "seed": int(args.seed),
            "seed_end": int(args.seed) + int(args.games) - 1,
            "max_actions_per_turn": int(args.max_actions_per_turn),
            "forbid_unfreeze": bool(int(args.forbid_unfreeze)),
            "forbid_cancel": bool(int(args.forbid_cancel)),
            "move_guard": bool(int(args.move_guard)),
            "move_cap": int(args.move_cap),
            "move_repeat_limit": int(args.move_repeat_limit),
            "freeze_cap": int(args.freeze_cap),
            "ban_trinkets": bool(int(args.ban_trinkets)),
            "no_cancel_choose": bool(int(args.no_cancel_choose)),
            "mirror_snaps": str(args.mirror_snaps or ""),
            "mirror_sample": str(args.mirror_sample or ""),
            "mirror_strict": int(args.mirror_strict),
            "mirror_seats": int(args.mirror_seats),
            "mirror_p": float(args.mirror_p),
            "forced_tier_plan": str(args.forced_tier_plan or ""),
            "forced_tier_prob": float(args.forced_tier_prob),
            "protocol_sha256": protocol_sha,
        },
        "caveats": [
            "humans faced 8 real opponents; models face injected static warbands - "
            "compare curves, not ranks",
            "human feed is winner-biased (816/995 1st place): a demanding yardstick, not an average",
        ],
        "models": [],
    }

    for name, path in specs:
        print(f"[{name}] evaluating {args.games} games ...", flush=True)
        entry = run_checkpoint(name, path, args, protocol=protocol)
        if human:
            entry["vs_human"] = compare(entry, human)
        payload["models"].append(entry)
        o = entry["outcome"]
        fs = entry.get("field_stats") or {}
        print(f"    mean_rank={o.get('mean_rank')} top4={o.get('top4_rate')} "
              f"win={o.get('win_rate')} illegal={o.get('illegal_actions')} "
              f"inject={fs.get('mirror_apply_rate')} {entry['seconds']}s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2))

    # ---- compact side-by-side print ----
    if human:
        hc = human.get("strength_by_turn") or {}
        ht = human.get("tier_by_turn") or {}
        turns = [t for t in ("6", "8", "10", "12", "14") if t in hc]
        print()
        print("HUMAN-GAP TABLE  (board strength = sum(atk+hp) at recruitment end)")
        head = f"{'turn':>4} {'human p50':>10} {'human tier':>10}"
        for m in payload["models"]:
            head += f" {m['name'][:14]:>14}"
        print(head)
        for t in turns:
            h = hc[t]
            row = f"{t:>4} {h['p50']:>10.0f} {ht.get(t, {}).get('mean', float('nan')):>10.2f}"
            for m in payload["models"]:
                c = m["curves"].get(t)
                row += f" {c['strength_p50'] if c else float('nan'):>14.0f}"
            print(row)
        print()
        print(f"{'turn':>4} {'strength ratio (model/human)':>32}")
        for t in turns:
            line = f"{t:>4}"
            for m in payload["models"]:
                c = m["curves"].get(t)
                r = (c["strength_p50"] / hc[t]["p50"]) if (c and c.get("strength_p50")) else None
                line += f"  {m['name'][:10]}={r:.2f}" if r else f"  {m['name'][:10]}=NA"
            print(line)
        print()
        print(f"{'model':<16} {'survive>t12':>12} {'survive>t14':>12} {'T6@t12':>8}")
        for m in payload["models"]:
            s = m["survival"]["reached"]
            c = m["curves"].get("12") or {}
            print(f"{m['name']:<16} {str(s.get('12')):>12} {str(s.get('14')):>12} "
                  f"{str(c.get('tier6_share')):>8}")

    print(f"\nwritten: {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
