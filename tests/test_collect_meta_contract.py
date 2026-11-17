"""每轮 collect meta 必须**自证动作契约**（2026-11-17 smoke 审计发现）。

以前 `freeze_cap` / `ban_trinkets` / `forbid_unfreeze` / `lobby_exclude` /
`max_actions_per_turn` 只存在于命令行：跑完之后翻 `history.jsonl` 查不到，
等于"无法自证当时用的是哪套限制"。这个测试锁住三处生产路径的 meta。

用真实引擎 + 极小配置（CPU，单局），不依赖 GPU。
"""

from __future__ import annotations

import pytest

from hsbg.az.train.ppo import CollectConfig, collect_ppo_batch
from hsbg.az.train import lobby_filter as LF

pytest.importorskip("hsbg_engine")
pytest.importorskip("torch")

CONTRACT_KEYS = ("max_actions_per_turn", "move_guard", "move_cap",
                 "move_repeat_limit", "freeze_cap", "forbid_cancel",
                 "forbid_unfreeze", "ban_trinkets", "lobby_exclude", "sims")


def _cfg(**kw):
    base = dict(games=1, opponent="bot", behavior="net", seed=424242,
                max_steps=400)
    base.update(kw)
    return CollectConfig(**base)


def _model():
    from hsbg.az.train.model import build_net
    m = build_net(8)          # tiny：只为跑通，不训练
    m.eval()
    return m


def test_serial_collect_meta_records_the_contract():
    cfg = _cfg(freeze_cap=1, ban_trinkets=True, forbid_unfreeze=True,
               lobby_exclude=("MECHANICAL", "DRAGON"))
    _batch, meta = collect_ppo_batch(_model(), cfg, device="cpu", seed=424242,
                                     progress=False)
    for key in CONTRACT_KEYS:
        assert key in meta, f"collect meta 缺 {key}"
    assert meta["max_actions_per_turn"] == 100
    assert meta["move_cap"] == 2 and meta["move_repeat_limit"] == 2
    assert meta["freeze_cap"] == 1
    assert meta["ban_trinkets"] is True and meta["forbid_unfreeze"] is True
    assert meta["lobby_exclude"] == ["MECHANICAL", "DRAGON"]
    assert meta["sims"] == 0


def test_lockstep_worker_meta_records_the_contract():
    """pool/lockstep 路径有**自己**的 meta 组装点，必须同样带契约字段。"""
    from hsbg.az.train.policies import EngineBotPolicy

    class _Eval:
        """把策略当作"总是选最后一个合法动作"的 stub，避免依赖真实网络前向。"""

        def __call__(self, features, cards, mask, seat):
            import numpy as np
            prior = np.zeros(245, dtype=np.float32)
            legal = np.flatnonzero(np.asarray(mask))
            prior[int(legal[-1]) if legal.size else 0] = 1.0
            return prior, 0.0

    from hsbg.az.train.parallel_collect import collect_ppo_lockstep
    cfg = _cfg(freeze_cap=1, ban_trinkets=True, forbid_unfreeze=True,
               lobby_exclude=("MECHANICAL",))
    _records, meta = collect_ppo_lockstep(
        _model(), cfg, eval_fn=_Eval(), device="cpu", seed=424242,
        games=1, gamemode="bot", engine_backend="auto")
    for key in CONTRACT_KEYS:
        assert key in meta, f"lockstep meta 缺 {key}"
    assert meta["freeze_cap"] == 1
    assert meta["lobby_exclude"] == ["MECHANICAL"]
    assert meta["max_actions_per_turn"] == 100
