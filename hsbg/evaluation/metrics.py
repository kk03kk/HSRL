"""Pure panel reductions, shared by evaluation and shard aggregation."""
from __future__ import annotations
import collections
from typing import Any, Dict, List, Optional


def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return float(s[k])


def _r(x, nd=1):
    return None if x is None else round(float(x), nd)


def economy_curves(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Turn -> statistics. Uses the recruitment-end board, same as the human data."""
    by_turn: Dict[int, Dict[str, List[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    tier_hist: Dict[int, collections.Counter] = collections.defaultdict(collections.Counter)

    for r in rows:
        t = int(r["turn"])
        b = by_turn[t]
        if isinstance(r.get("tier"), int):
            b["tier"].append(float(r["tier"]))
            tier_hist[t][int(r["tier"])] += 1
        if isinstance(r.get("gold"), int):
            b["gold"].append(float(r["gold"]))
        if isinstance(r.get("actions"), int):
            b["actions"].append(float(r["actions"]))
        if isinstance(r.get("hand_n"), int):
            b["hand"].append(float(r["hand_n"]))
        # recruitment-end board == what the human snapshots call combat_start
        atk, hp = r.get("eot_attack"), r.get("eot_board_hp")
        if isinstance(atk, int) and isinstance(hp, int):
            b["strength"].append(float(atk + hp))
        board = r.get("eot_board") or r.get("board")
        if isinstance(board, list):
            b["board_n"].append(float(len(board)))

    out: Dict[str, Any] = {}
    for t in sorted(by_turn):
        b = by_turn[t]
        if len(b["tier"]) < 10:
            continue
        hist = tier_hist[t]
        n = len(b["tier"])
        out[str(t)] = {
            "n": n,
            "tier_mean": _r(sum(b["tier"]) / n, 2),
            "tier_mode": hist.most_common(1)[0][0],
            "tier6_share": _r(sum(1 for x in b["tier"] if x >= 6) / n, 3),
            "tier5plus_share": _r(sum(1 for x in b["tier"] if x >= 5) / n, 3),
            "tier_hist": dict(sorted(hist.items())),
            "gold_p50": _r(_pct(b["gold"], 0.5)),
            "hand_p50": _r(_pct(b["hand"], 0.5)),
            "strength_p50": _r(_pct(b["strength"], 0.5)),
            "strength_p90": _r(_pct(b["strength"], 0.90)),
            # 2026-11-16 用户要求：**逐回合所有局的战力全部保留**（原来只留 p50/p90/mean，
            # 原始分布算完就丢了）。有了这个，"前 N 名 / 任意分位 / 直方图 / 分布形状"
            # 都能事后从 panel.json 直接算，不用再重跑评测。
            # 体积很小：512 局 × 18 回合 ≈ 9k 个数 ≈ 70 KB。
            "strength_all": [_r(x) for x in sorted(b["strength"], reverse=True)],
            "strength_top10": [_r(x) for x in sorted(b["strength"], reverse=True)[:10]],
            "strength_p99": _r(_pct(b["strength"], 0.99)),
            "strength_max": _r(max(b["strength"])) if b["strength"] else None,
            "strength_min": _r(min(b["strength"])) if b["strength"] else None,
            "strength_mean": _r(sum(b["strength"]) / len(b["strength"])) if b["strength"] else None,
            "board_n_p50": _r(_pct(b["board_n"], 0.5)),
            "actions_mean": _r(sum(b["actions"]) / len(b["actions"])) if b["actions"] else None,
        }
    return out


def survival(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    reached = collections.defaultdict(int)
    games: Dict[Any, int] = {}
    for r in rows:
        g = r.get("seed")
        games[g] = max(games.get(g, 0), int(r["turn"]))
    for g, t in games.items():
        for milestone in (8, 10, 12, 14):
            if t >= milestone:
                reached[milestone] += 1
    n = max(len(games), 1)
    return {
        "games": len(games),
        "max_turn_p50": _pct(list(games.values()), 0.5),
        "reached": {str(k): _r(v / n, 3) for k, v in sorted(reached.items())},
    }


def action_mix_by_turn(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-turn action-group counts, averaged over **all** games alive that turn.

    This is the sensitive instrument for the churn hypothesis: human PMS gives
    ``minionsSoldOverTurn`` / ``rerollsOverTurn`` per turn, so the model side has to
    be in the same shape to be comparable. Board strength moves too slowly to
    resolve a short run; sell/reroll rate does not.

    ⚠️ 2026-11-13 修：早先版本只在**该组键存在**时把计数收进分母，所以 ``mean`` 其实是
    "用过它的局里的均值"（旧读数里 ``sell T3 = 1.33`` 被误读成"每局都卖 1.33 个"）。
    现在同时给三个口径：

    - ``mean``：以**当回合全部存活局**为分母（没用过就是 0）——判断"每局平均几次"用它；
    - ``used_rate``：当回合用过该动作的局占比——判断"有多少局会做这件事"用它；
    - ``mean_used``：只在用过它的局里平均——判断"做的话做几次"用它。
    """
    per_turn: Dict[int, Dict[Any, Dict[str, float]]] = collections.defaultdict(dict)
    for r in rows:
        t = int(r["turn"])
        counts = r.get("action_counts") or {}
        if not isinstance(counts, dict):
            continue
        seed = r.get("seed")
        cur = per_turn[t].get(seed)
        if cur is None:
            cur = per_turn[t][seed] = {}
        for group, n in counts.items():
            if isinstance(n, int):
                cur[group] = cur.get(group, 0.0) + float(n)
    out: Dict[str, Any] = {}
    for t in sorted(per_turn):
        games = per_turn[t]
        n_games = len(games)
        if n_games < 5:
            continue
        groups = sorted({g for counts in games.values() for g in counts})
        mean, used_rate, mean_used = {}, {}, {}
        for g in groups:
            vals = [c.get(g, 0.0) for c in games.values()]
            used = [v for v in vals if v > 0]
            mean[g] = _r(sum(vals) / n_games, 2)
            used_rate[g] = _r(len(used) / n_games, 3)
            mean_used[g] = _r(sum(used) / len(used), 2) if used else 0.0
        out[str(t)] = {
            "games": n_games,
            "mean": mean,
            "used_rate": used_rate,
            "mean_used": mean_used,
        }
    return out


def combat_by_turn(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """逐回合**实际战斗结果**（专家先验 G1：Σ(atk+hp) 不能代表战力，要看真打赢没赢）。

    每局每回合未必有战斗记录（还没打就被淘汰、或该回合没有对阵），所以分母是
    **该回合有战斗记录的局数**，而不是存活局数；同时给出 `games_alive` 便于判读。
    """
    by_turn: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            t = int(r.get("turn"))
        except (TypeError, ValueError):
            continue
        by_turn[t].append(r)
    out: Dict[str, Any] = {}
    for t in sorted(by_turn):
        rs = by_turn[t]
        if len(rs) < 5:
            continue
        wins = sum(1 for r in rs if r.get("win"))
        ties = sum(1 for r in rs if r.get("tie"))
        dmg = [float(r.get("damage_taken") or 0) for r in rs]
        out[str(t)] = {
            "combats": len(rs),
            "win": _r(wins / len(rs), 3),
            "tie": _r(ties / len(rs), 3),
            "damage_taken_mean": _r(sum(dmg) / max(len(dmg), 1), 2),
            "ghost": _r(sum(1 for r in rs if r.get("ghost")) / len(rs), 3),
            "mirror_applied": _r(sum(1 for r in rs if r.get("mirror_applied")) / len(rs), 3),
        }
    return out


def final_boards(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Each game's last recorded board, as card ids. Feeds composition coverage.

    Card ids carry a ``_G`` suffix for golden minions; strip it so a golden copy
    still matches the base id in config/strategy_cards.json.
    """
    last: Dict[Any, Dict[str, Any]] = {}
    for r in rows:
        g = r.get("seed")
        t = int(r["turn"])
        if g not in last or t >= int(last[g]["turn"]):
            last[g] = r
    out = []
    for g, r in sorted(last.items(), key=lambda kv: str(kv[0])):
        board = r.get("eot_board") or r.get("board") or []
        ids = []
        for m in board:
            cid = m.get("card_id")
            if not isinstance(cid, str):
                continue
            ids.append(cid[:-2] if cid.endswith("_G") else cid)
        out.append({
            "seed": g,
            "turn": int(r["turn"]),
            "tier": r.get("tier"),
            "strength": (r.get("eot_attack") or 0) + (r.get("eot_board_hp") or 0),
            "card_ids": ids,
        })
    return out


def mechanics_by_turn(rows: List[Dict[str, Any]], min_turn: int = 6) -> Dict[str, Any]:
    """逐回合**机制密度**：金卡数 / 每格攻血。

    为什么需要（专家先验 C1）：后期每格强度差 4–14 倍，靠的是乘法机制（三连金卡、宝石、
    磁力、成长引擎），不是"买更多随从"。真人库同名口径：
    `tools/report_board_mechanics.py`（T12 约 **2.55 张金卡/局、90% 的局有金卡**、
    每格 atk 598 / hp 529）。模型侧用同一张表对照。

    ``describe_board`` 的 ``card_id`` 对金卡带 ``_G`` 后缀（``final_boards`` 会剥掉，
    这里**必须保留**才能数金卡）。
    """
    per_turn: Dict[int, Dict[str, int]] = collections.defaultdict(
        lambda: {"boards": 0, "minions": 0, "golden": 0, "with_golden": 0,
                 "atk": 0, "hp": 0})
    for r in rows:
        t = int(r["turn"])
        if t < int(min_turn):
            continue
        board = r.get("eot_board") or r.get("board") or []
        if not board:
            continue
        d = per_turn[t]
        d["boards"] += 1
        g = 0
        for m in board:
            if not isinstance(m, dict):
                continue
            d["minions"] += 1
            d["atk"] += int(m.get("atk") or 0)
            d["hp"] += int(m.get("hp") or 0)
            cid = m.get("card_id")
            if isinstance(cid, str) and cid.endswith("_G"):
                g += 1
        d["golden"] += g
        if g:
            d["with_golden"] += 1
        # 投资集中度：最强那一格占战团总 Σ 的比例（"养大哥" vs "均匀铺开"）
        sums = [int(m.get("atk") or 0) + int(m.get("hp") or 0) for m in board
                if isinstance(m, dict)]
        tot = sum(sums)
        if sums and tot > 0:
            d["top1"] = d.get("top1", 0) + max(sums) / tot
            d["conc_n"] = d.get("conc_n", 0) + 1
    out: Dict[str, Any] = {}
    for t in sorted(per_turn):
        d = per_turn[t]
        n = max(d["boards"], 1)
        mn = max(d["minions"], 1)
        out[str(t)] = {
            "boards": d["boards"],
            "golden_per_game": _r(d["golden"] / n, 2),
            "with_golden": _r(d["with_golden"] / n, 3),
            "atk_per_minion": _r(d["atk"] / mn, 1),
            "hp_per_minion": _r(d["hp"] / mn, 1),
            "top1_share": _r(d["top1"] / max(d.get("conc_n", 1), 1), 3),
        }
    return out


def retained_gain_by_turn(rows: List[Dict[str, Any]], min_turn: int = 6) -> Dict[str, Any]:
    """模型侧「留存随从每回合涨多少」——与真人库同一口径（RUNS §5.48）。

    真人 T10→11：每只留存随从平均 **+690 Σ**，整局增长几乎全部来自这一项。
    面板里用 `board_stats`（逐卡 atk/hp）按 seed 串成回合序列，
    把相邻回合都在场的卡（按 card_id）配对，算平均增重。
    """
    by_seed: Dict[str, Dict[int, Dict[str, List[int]]]] = collections.defaultdict(dict)
    for r in rows:
        t = int(r["turn"])
        if t < int(min_turn):
            continue
        stats = r.get("board_stats")
        if not isinstance(stats, dict) or not stats:
            continue
        by_seed[str(r.get("seed"))][t] = {
            str(k): [int(v[0]), int(v[1])] for k, v in stats.items()
            if isinstance(v, (list, tuple)) and len(v) >= 2
        }
    acc: Dict[int, Dict[str, float]] = collections.defaultdict(
        lambda: {"n": 0, "gain": 0.0, "before": 0.0, "after": 0.0})
    for turns in by_seed.values():
        for t in sorted(turns):
            a, b = turns.get(t), turns.get(t + 1)
            if not a or not b:
                continue
            kept = [k for k in a if k in b]
            if not kept:
                continue
            d = acc[t]
            d["n"] += 1
            d["gain"] += sum((b[k][0] + b[k][1]) - (a[k][0] + a[k][1]) for k in kept) / len(kept)
            d["before"] += sum(a[k][0] + a[k][1] for k in kept) / len(kept)
            d["after"] += sum(b[k][0] + b[k][1] for k in kept) / len(kept)
    out: Dict[str, Any] = {}
    for t, d in acc.items():
        n = max(d["n"], 1)
        out[str(t)] = {"games": d["n"], "kept_gain": _r(d["gain"] / n, 1),
                       "kept_before": _r(d["before"] / n, 1),
                       "kept_after": _r(d["after"] / n, 1)}
    return out


def boards_by_turn(rows: List[Dict[str, Any]],
                   min_turn: int = 6) -> Dict[str, Dict[str, List[str]]]:
    """``seed -> turn -> [card_id]``（只留 ``turn >= min_turn``）。

    为什么需要：``final_boards`` 只有"每局最后一帧"，无法回答"什么时候拼齐了阵容"
    "中途转过几次型"。有了逐回合卡 id，就能用**与训练奖励完全相同的口径**
    （``hsbg/az/train/comp_target.py`` 的等级权重 · 命中/需求）算出阵容完成度曲线，
    从而把"阵容覆盖奖励有没有生效"和"棋力有没有变强"分开看。

    体积：96 局 × ~10 回合 × ≤7 个 id，单个模型几百 KB，可接受。
    """
    out: Dict[str, Dict[str, List[str]]] = {}
    for r in rows:
        t = int(r["turn"])
        if t < int(min_turn):
            continue
        board = r.get("eot_board") or r.get("board") or []
        ids = []
        for m in board:
            cid = m.get("card_id")
            if not isinstance(cid, str):
                continue
            ids.append(cid[:-2] if cid.endswith("_G") else cid)
        out.setdefault(str(r.get("seed")), {})[str(t)] = ids
    return out


def _strength_by_seed(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """{seed: {turn: 战力}} —— 保留"哪一局"，可做前后回合归因。"""
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        atk, hp = r.get("eot_attack"), r.get("eot_board_hp")
        if not (isinstance(atk, int) and isinstance(hp, int)):
            continue
        try:
            t = int(r["turn"])
        except (KeyError, TypeError, ValueError):
            continue
        s = str(r.get("seed"))
        out.setdefault(s, {})[str(t)] = float(atk + hp)
    return out


def _roster(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把「回合结束那一刻」的完整阵容逐只展开（含身材/金卡/关键字/站位）。

    2026-11-16 用户要求「每局每回合的阵容必须完整记录」。这里**直接从 eot_board
    取**（与 boards_by_turn 同源），不再用 board_stats 这个按卡牌 id 归并的字典——
    那是个字典，重复卡会塌成一份，而且它取的是**回合开始**的板子，跟回合结束的
    阵容对不上，导致本回合新上场的随从查不到身材（表现为 `—`）。
    """
    out: List[Dict[str, Any]] = []
    board = row.get("eot_board") or row.get("board") or []
    for m in board:
        if not isinstance(m, dict):
            continue
        kw = m.get("keywords") or m.get("kw") or []
        out.append({
            "pos": m.get("position"),
            "card_id": m.get("card_id") or m.get("card"),
            "name": m.get("name"),
            "atk": m.get("atk"), "hp": m.get("hp"),
            "golden": bool(m.get("golden")),
            "keywords": sorted(kw) if isinstance(kw, (list, tuple, set)) else kw,
            "tier": m.get("tier"),
        })
    return out


def _trace(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """每局每回合一条精简记录：一切能解释"为什么强/为什么弱"的字段。

    体积控制：只保留数值/短列表，单条 ~200 B ⇒ 512 局 × 18 回合 ≈ 2 MB。
    """
    out: List[Dict[str, Any]] = []
    for r in rows:
        atk, hp = r.get("eot_attack"), r.get("eot_board_hp")
        out.append({
            "seed": r.get("seed"), "turn": r.get("turn"),
            "tier": r.get("tier"), "gold": r.get("gold"), "health": r.get("health"),
            "hand_n": r.get("hand_n"), "actions": r.get("actions"),
            "strength": (int(atk) + int(hp)) if isinstance(atk, int) and isinstance(hp, int) else None,
            "board_stats": r.get("board_stats"),
            "roster": _roster(r),
            "hero": r.get("hero"), "trinkets": r.get("trinkets"),
            "rewards": r.get("rewards"),
            "repeats": r.get("consecutive_repeats"),
            # 金币明细不逐条存（体积大），用 gold_by_turn 的逐回合聚合代替
        })
    return out


def _counts_by_turn(rows: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    """{turn: {取值: 次数}} —— 用于逐回合看某字段的分布（饰品/奖励等列表字段）。"""
    acc: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in rows:
        v = r.get(field)
        if not isinstance(v, list):
            continue
        try:
            t = str(int(r["turn"]))
        except (KeyError, TypeError, ValueError):
            continue
        for x in v:
            acc[t][str(x)] += 1
    return {t: dict(c.most_common()) for t, c in sorted(acc.items(), key=lambda kv: int(kv[0]))}


def _counts_all(rows: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    c: collections.Counter = collections.Counter()
    for r in rows:
        v = r.get(field)
        if v is not None:
            c[str(v)] += 1
    return dict(c.most_common())


def _opponents_by_turn(exposure: List[Dict[str, Any]]) -> Dict[str, Any]:
    """逐回合**实际遇到**的对手强度（注入帧的 source_strength）分布。

    这是难度闭环的唯一直接度量：库里写的是目标，这里才是它真的撞上了什么。
    """
    acc: Dict[str, List[float]] = collections.defaultdict(list)
    for e in exposure:
        s = e.get("source_strength")
        if not isinstance(s, (int, float)):
            continue
        acc[str(int(e.get("turn") or 0))].append(float(s))
    out: Dict[str, Any] = {}
    for t, v in sorted(acc.items(), key=lambda kv: int(kv[0])):
        v.sort()
        out[t] = {"n": len(v), "p10": _r(_pct(v, 0.10)), "p50": _r(_pct(v, 0.50)),
                  "p90": _r(_pct(v, 0.90)), "mean": _r(sum(v) / len(v))}
    return out


def _gold_by_turn(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """逐回合金币收支：按动作分组（买牌/刷新/升本/出售…）累加净变化。"""
    acc: Dict[str, Dict[str, float]] = collections.defaultdict(
        lambda: collections.defaultdict(float))
    for r in rows:
        evs = r.get("gold_events")
        if not isinstance(evs, list):
            continue
        try:
            t = str(int(r["turn"]))
        except (KeyError, TypeError, ValueError):
            continue
        for e in evs:
            if not isinstance(e, dict):
                continue
            k = str(e.get("kind") or "other")
            try:
                acc[t][k] += float(e.get("net_change") or 0)
            except (TypeError, ValueError):
                continue
    return {t: {k: _r(v) for k, v in d.items()} for t, d in sorted(acc.items(), key=lambda kv: int(kv[0]))}


def compare(model: Dict[str, Any], human: Dict[str, Any]) -> Dict[str, Any]:
    """Per-turn model-vs-human deltas. Human side is the winner-biased baseline."""
    mc = model["curves"]
    hc = human.get("strength_by_turn") or {}
    ht = human.get("tier_by_turn") or {}
    rows = []
    for t in sorted(set(mc) & set(hc), key=int):
        m, h = mc[t], hc[t]
        tier_h = ht.get(t) or {}
        rows.append({
            "turn": int(t),
            "model_tier_mean": m["tier_mean"],
            "human_tier_mean": tier_h.get("mean"),
            "tier_gap": _r((m["tier_mean"] or 0) - (tier_h.get("mean") or 0), 2),
            "model_strength_p50": m["strength_p50"],
            "human_strength_p50": h.get("p50"),
            "strength_ratio": _r(m["strength_p50"] / h["p50"], 3)
            if m.get("strength_p50") and h.get("p50") else None,
            "model_strength_p90": m["strength_p90"],
            "human_strength_p90": h.get("p90"),
            "model_board_n_p50": m["board_n_p50"],
        })
    return {"per_turn": rows}
