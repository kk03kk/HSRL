#!/usr/bin/env python3
"""第三方 PPO 模型 S0 冒烟：验证「能否用我们的引擎喂它的网络并合法出招」。

**这不是实战测试**（对手是引擎 bot、无镜像战团）；它只回答三件事：
1. 非法动作是否恒为 0（我们在掩码上取 argmax，理论上必然为 0 —— 若不是，说明索引/掩码不匹配）
2. 动作分布是否塌陷（若 >95% 全是 EndTurn ⇒ 输入标定不符，**必须先解决**）
3. 各动作族是否都有覆盖（buy/sell/play/upgrade/choose…）

用法：python workflows/thirdparty_smoke.py --ckpt data/thirdparty/hsbg-ppo/model.pt --games 64
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "thirdparty" / "hsbg_ppo"))

import numpy as np  # noqa: E402

from hsbg.az.train.common import group_of, mask_from_bytes, obs_arrays  # noqa: E402
from hsbg.env import hsbg_engine as E  # noqa: E402
from hsbg.env.action_space_std import (  # noqa: E402
    A_CANCEL, A_END, A_SELL, N_BOARD, N_PLAYERS, N_TOTAL_ACTIONS,
)

import torch  # noqa: E402
import hsbg_net  # noqa: E402


def board_eids(game, seat):
    """当前棋盘每个槽位的 eid（用来实现 keep_start_board）。"""
    try:
        pj = json.loads(game.player_json(seat))
    except BaseException:  # noqa: BLE001
        return []
    return [int(m.get("eid", -1)) for m in (pj.get("board") or []) if isinstance(m, dict)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/thirdparty/hsbg-ppo/model.pt")
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--seed", type=int, default=930000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--z", type=int, default=45, help="阵容 id；45 = none")
    ap.add_argument("--forbid-cancel", type=int, default=1)
    ap.add_argument("--keep-start-board", type=int, default=1)
    ap.add_argument("--max-actions-per-turn", type=int, default=100)
    a = ap.parse_args(argv)

    net, ck, res = hsbg_net.load(a.ckpt, device=a.device)
    net.eval()
    print("[net] class=%s params=%d" % (type(net).__name__, sum(p.numel() for p in net.parameters())))
    print("[net] ckpt md5-verified by size; z_vocab=%d  z=%d" % (len(ck["net_config"]["z_vocab"]), a.z))

    act_hist = collections.Counter()
    fam_hist = collections.Counter()
    illegal = 0
    decisions = 0
    ranks = []
    per_turn_tier = collections.defaultdict(list)
    per_turn_str = collections.defaultdict(list)

    for gi in range(int(a.games)):
        seed = int(a.seed) + gi
        game = E.Game(seed=seed)
        seat0 = 0
        last_turn = -1
        turn_actions = 0
        start_eids = set()
        steps = 0
        while not game.is_over() and steps < 40000:
            steps += 1
            acted = False
            for p in range(N_PLAYERS):
                if not game.needs_action(p):
                    continue
                acted = True
                seat = int(p)
                if seat != seat0:
                    game.step(seat, int(game.bot_action(seat)))
                    continue
                turn = int(game.turn)
                if turn != last_turn:
                    last_turn, turn_actions = turn, 0
                    start_eids = set(e for e in board_eids(game, seat) if e >= 0)
                f, c = obs_arrays(game.observe(seat))
                mask = mask_from_bytes(game.legal_mask(seat))
                if a.forbid_cancel:
                    mask = mask.copy()
                    mask[A_CANCEL] = False
                    if not mask.any():
                        mask[A_END] = True
                if a.keep_start_board and start_eids:
                    cur = board_eids(game, seat)
                    for i in range(min(N_BOARD, len(cur))):
                        if cur[i] in start_eids and A_SELL + i < N_TOTAL_ACTIONS:
                            mask = mask.copy() if not mask.flags.writeable else mask
                            mask[A_SELL + i] = False
                    if not mask.any():
                        mask[A_END] = True
                with torch.no_grad():
                    lg, _v = net(torch.from_numpy(f)[None].to(a.device),
                                 torch.from_numpy(c)[None].to(a.device),
                                 torch.from_numpy(mask)[None].to(a.device),
                                 z=torch.tensor([int(a.z)], device=a.device))
                act = int(lg[0].argmax().item())
                if not (0 <= act < N_TOTAL_ACTIONS and bool(mask[act])):
                    illegal += 1
                    legal = np.flatnonzero(mask)
                    act = int(legal[0]) if legal.size else A_END
                act_hist[act] += 1
                fam_hist[group_of(act)] += 1
                decisions += 1
                turn_actions += 1
                if turn_actions >= int(a.max_actions_per_turn):
                    act = A_END
                game.step(seat, act)
            if not acted:
                break
        # 结算
        try:
            pj = json.loads(game.player_json(seat0))
            rank = int(pj.get("placement") or 0)
        except BaseException:  # noqa: BLE001
            rank = 0
        if rank:
            ranks.append(rank)

    print()
    print("=== S0 结果（%d 局，对手=引擎 bot，非实战）===" % int(a.games))
    print("  决策数           : %d" % decisions)
    print("  **非法动作**     : %d   ← 必须为 0" % illegal)
    print("  名次样本         : %d   平均名次 = %s"
          % (len(ranks), ("%.3f" % (sum(ranks) / len(ranks))) if ranks else "n/a"))
    tot = sum(fam_hist.values()) or 1
    print("  动作族分布（占决策）:")
    for k, v in fam_hist.most_common():
        print("    %-12s %7d  %5.1f%%" % (k, v, 100.0 * v / tot))
    end_share = 100.0 * fam_hist.get("end", 0) / tot
    print()
    print("  EndTurn 占比 = %.1f%%  ← >95%% 视为塌陷（输入标定不符）" % end_share)
    print("  非 end 动作族数 = %d  ← <3 视为覆盖不足"
          % sum(1 for k, v in fam_hist.items() if k != "end" and v > 0))
    print()
    ok = (illegal == 0 and end_share < 95.0
          and sum(1 for k, v in fam_hist.items() if k != "end" and v > 0) >= 3)
    print("S0_PASS" if ok else "S0_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
