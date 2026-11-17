"""第三方 PPO 网络的策略适配（只用于评测/对局，不参与我们的训练损失）。

接口对齐 ``PPOGreedyPolicy``（``reset()`` / ``decide(game, seat) -> Decision``），
其余动作规则（max-actions / move-guard / forbid-cancel / freeze-cap / ban-trinkets）
**复用我们自己的实现**，保证与自家模型完全同条件。

与自家模型的三处差异（都由作者 README / lean 实测确定，不是猜）：
1. 网络输入是 ``(features[3227], cards[124], mask[245], z)``，输出 ``(logits[245], value)``；
2. 作者要求 ``no_cancel_choose=1``（我们的 ``forbid-cancel`` 已覆盖 Cancel）与
   ``keep_start_board=1``（回合开始时已在场上的随从不能被卖）——本类自己实现后者；
3. 没有 rank 头 / 卡文本嵌入 / 输入数值压缩，因此**直接喂引擎原始 obs**。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from hsbg.az.train.common import mask_from_bytes, obs_arrays
from hsbg.az.train.pending_guard import without_cancel
from hsbg.az.train.action_ban import ban_choice_options, has_pending_choice
from hsbg.az.train.policies import Decision
from hsbg.env.action_space_std import A_END, A_SELL, N_BOARD

_ROOT = Path(__file__).resolve().parents[3]
_TP = _ROOT / "thirdparty" / "hsbg_ppo"
if str(_TP) not in sys.path:
    sys.path.insert(0, str(_TP))


class ThirdPartyGreedyPolicy:
    """把作者发布的 `tg-ckpt-v1` 网络包装成我们的 Policy。"""

    def __init__(self, ckpt_path: str, *, device: str = "cuda:0", z: int = 45,
                 max_actions_per_turn: int = 80, forbid_cancel: bool = True,
                 keep_start_board: bool = True, ban_trinkets: bool = True,
                 freeze_cap: int = 0, forbid_unfreeze: bool = False,
                 move_cap: int = 5, move_repeat_limit: int = 2) -> None:
        import torch  # 惰性导入（登录节点 import torch 会失败）
        import hsbg_net
        self._torch = torch
        self.net, self.ckpt, _ = hsbg_net.load(str(ckpt_path), device=device)
        self.net.eval()
        self.device = device
        self.z = int(z)
        self.max_actions_per_turn = int(max_actions_per_turn)
        self.forbid_cancel = bool(forbid_cancel)
        self.keep_start_board = bool(keep_start_board)
        self.ban_trinkets = bool(ban_trinkets)
        self.freeze_cap = int(freeze_cap)
        self.forbid_unfreeze = bool(forbid_unfreeze)
        self.move_cap = int(move_cap)
        self.move_repeat_limit = int(move_repeat_limit)
        self._turn_counts: Dict[tuple, int] = {}
        self._freeze_counts: Dict[tuple, int] = {}
        self._start_eids: Dict[tuple, set] = {}
        self.game_id = -1
        self.forced_actions = 0
        self.keep_start_blocked = 0

    # ---- 与 PPOGreedyPolicy 对齐的生命周期 ----
    def reset(self) -> None:
        self._turn_counts = {}
        self._freeze_counts = {}
        self._start_eids = {}
        self.game_id = -1

    def _board_eids(self, game: Any, seat: int):
        try:
            pj = json.loads(game.player_json(int(seat)))
        except BaseException:  # noqa: BLE001
            return []
        out = []
        for m in (pj.get("board") or []):
            if isinstance(m, dict) and isinstance(m.get("eid"), int):
                out.append(int(m["eid"]))
        return out

    def decide(self, game: Any, seat: int) -> Decision:
        seat = int(seat)
        turn = int(getattr(game, "turn", -1))
        gid = int(getattr(game, "game_id", -1))
        if gid != self.game_id:
            self.game_id = gid
            self._start_eids = {}
        key = (gid, seat, turn)
        if key not in self._start_eids:
            self._start_eids[key] = set(self._board_eids(game, seat))

        f, c = obs_arrays(game.observe(seat))
        mask = mask_from_bytes(game.legal_mask(seat))

        if self.forbid_cancel:
            mask = without_cancel(mask)
            if not bool(mask.any()):
                mask = mask_from_bytes(game.legal_mask(seat))
                mask[A_END] = True

        if self.forbid_unfreeze and len(mask) > 2 and bool(mask[2]):
            from hsbg.env.obs_std import GLOBAL_FIELD_INDEX
            if float(f[GLOBAL_FIELD_INDEX["frozen"]]) > 0.5:
                mask = mask.copy()
                mask[2] = False

        if self.freeze_cap > 0 and len(mask) > 2 and bool(mask[2]):
            if int(self._freeze_counts.get(key, 0)) >= self.freeze_cap:
                mask = mask.copy()
                mask[2] = False

        # 作者的推理设置：回合开始时已在场上的随从不能被卖
        if self.keep_start_board:
            start = self._start_eids.get(key) or set()
            if start:
                cur = self._board_eids(game, seat)
                hit = 0
                for i in range(min(N_BOARD, len(cur))):
                    if cur[i] in start:
                        if not mask.flags.writeable:
                            mask = mask.copy()
                        mask[A_SELL + i] = False
                        hit += 1
                self.keep_start_blocked += hit
                if not bool(mask.any()):
                    mask = mask_from_bytes(game.legal_mask(seat))
                    mask[A_END] = True

        # 与本轮训练一致的加重护手封禁
        if self.ban_trinkets and has_pending_choice(mask):
            try:
                pj = json.loads(game.player_json(seat))
            except BaseException:  # noqa: BLE001
                pj = None
            if pj is not None:
                if not mask.flags.writeable:
                    mask = mask.copy()
                ban_choice_options(pj, mask)

        n_act = int(self._turn_counts.get(key, 0))
        if self.max_actions_per_turn > 0 and n_act >= self.max_actions_per_turn:
            self.forced_actions += 1
            return Decision(action=A_END, info={"src": "thirdparty", "forced": "max_actions"})

        torch = self._torch
        with torch.no_grad():
            lg, val = self.net(
                torch.from_numpy(np.asarray(f, dtype=np.float32))[None].to(self.device),
                torch.from_numpy(np.asarray(c, dtype=np.int64))[None].to(self.device),
                torch.from_numpy(np.asarray(mask, dtype=bool))[None].to(self.device),
                z=torch.tensor([self.z], device=self.device),
            )
        act = int(lg[0].argmax().item())
        if not (0 <= act < len(mask) and bool(mask[act])):
            legal = np.flatnonzero(mask)
            act = int(legal[0]) if legal.size else A_END
        self._turn_counts[key] = n_act + 1
        if len(mask) > 2 and act == 2:
            self._freeze_counts[key] = int(self._freeze_counts.get(key, 0)) + 1
        return Decision(action=act, info={"src": "thirdparty",
                                          "value": float(val[0].item()) if hasattr(val, "item") else 0.0})


def is_thirdparty_spec(spec: str) -> bool:
    """``name=thirdparty:/path/model.pt`` 或直接 ``thirdparty:/path``。"""
    return "thirdparty:" in str(spec)


def split_thirdparty(path: str) -> str:
    p = str(path)
    i = p.find("thirdparty:")
    return p[i + len("thirdparty:") :] if i >= 0 else p
