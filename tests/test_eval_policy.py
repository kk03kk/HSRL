"""``hsbg.az.train.eval_policy`` 的单测：不依赖 ``hsbg_engine`` 与 torch。

覆盖审计 P0-1（评测策略构造对自家 checkpoint 崩溃）、P2-2（训练/评测契约三套并存）、
P0-2（协议指纹缺失）三件事的**根因层**：

* ``make_eval_policy`` 两条分支都必须返回可用策略（回归 ``UnboundLocalError``）；
* 契约必须显式传给 ``PPOGreedyPolicy``，且与训练的冻结契约一致；
* ``protocol_sha256`` 对每个会影响分数的字段都敏感、对分片参数不敏感。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hsbg.az.train import eval_policy as EP


@pytest.mark.parametrize(("profile", "max_actions", "moves"), [
    ("human-turn-3702-standard", 200, 10),
    ("human-turn-3702-economy", 300, 10),
    ("human-turn-3702-economy-move8", 300, 8),
])
def test_bounded_peer_profile_reaches_policy_and_engine(profile, max_actions, moves):
    # Contracts are read at process startup, the same way Slurm workers read them.
    probe = """
import json
from hsbg.az.train.eval_policy import TRAINING_CONTRACT
from hsbg.az.train.lobby_filter import engine_rule_overrides
print(json.dumps([TRAINING_CONTRACT, engine_rule_overrides()]))
"""
    output = subprocess.check_output(
        [sys.executable, "-c", probe], env=os.environ | {"HSAI_ACTION_PROFILE": profile}, text=True)
    policy, engine = json.loads(output)
    assert policy["max_actions_per_turn"] == max_actions
    assert (policy["move_guard"], policy["move_cap"], policy["move_repeat_limit"]) == (True, moves, 0)
    assert policy["freeze_cap"] == engine["freeze_presses_per_turn"] == 3
    assert policy["forbid_cancel"] and not policy["forbid_unfreeze"] and not policy["ban_trinkets"]


# ────────────────────────── 假实现（无引擎/无 torch）──────────────────────────
class _FakePolicy:
    """记录构造参数，供断言契约是否真的传下去了。"""

    def __init__(self, evaluator=None, **kwargs):
        self.evaluator = evaluator
        self.kwargs = dict(kwargs)
        self.name = "fake"

    def reset(self):
        pass


class _FakeThirdPartyPolicy(_FakePolicy):
    def __init__(self, real_path, **kwargs):
        super().__init__(None, **kwargs)
        self.real_path = real_path
        self.net = object()


def _fake_load_checkpoint(path, device="cpu"):
    return ("MODEL:" + str(path), {"arch": "entity", "norm_mean": None,
                                   "norm_std": None})


def _fake_evaluator(model, *, device="cpu", norm_mean=None, norm_std=None):
    return {"model": model, "device": device, "norm_mean": norm_mean}


def _make(spec, **kw):
    params = dict(load_checkpoint=_fake_load_checkpoint,
                  evaluator_factory=_fake_evaluator,
                  greedy_policy_cls=_FakePolicy,
                  thirdparty_policy_cls=_FakeThirdPartyPolicy,
                  is_thirdparty_spec=lambda s: str(s).startswith("thirdparty:"),
                  split_thirdparty=lambda s: str(s).split(":", 1)[1])
    params.update(kw)
    return EP.make_eval_policy(spec, **params)


# ────────────────────────── P0-1：两条分支都要能构造出策略 ──────────────────────────
def test_own_checkpoint_builds_policy_without_unbound_local():
    """回归 ``workflows/eval_human_gap.py`` 的 ``UnboundLocalError``。

    旧代码在非第三方分支只做 ``load_checkpoint``，随后读未绑定的 ``evaluator``。
    """
    proto = EP.EvalProtocol()
    out = _make("artifacts/models/x/model.pt", protocol=proto)
    assert out["kind"] == "own"
    assert isinstance(out["policy"], _FakePolicy)
    assert out["model"].startswith("MODEL:")
    assert out["policy"].evaluator["device"] == "cpu"


def test_thirdparty_spec_builds_adapter_policy():
    proto = EP.EvalProtocol()
    out = _make("thirdparty:data/thirdparty/hsbg-ppo/model.pt", protocol=proto)
    assert out["kind"] == "thirdparty"
    assert isinstance(out["policy"], _FakeThirdPartyPolicy)
    assert out["real_path"] == "data/thirdparty/hsbg-ppo/model.pt"
    assert out["meta"] == {}


# ────────────────────────── P2-2：契约必须真的传下去，且等于训练契约 ──────────────────────────
def test_own_policy_receives_full_training_contract():
    """审计 P2-2：契约必须真的传下去；D-070 之后契约是 100 / cap2 / freeze1。"""
    proto = EP.EvalProtocol()
    out = _make("a/model.pt", protocol=proto)
    kw = out["policy"].kwargs
    assert kw["max_actions_per_turn"] == 100
    assert kw["move_guard"] is True
    assert kw["move_cap"] == 2
    assert kw["move_repeat_limit"] == 2
    assert kw["forbid_cancel"] is True
    assert kw["forbid_unfreeze"] is True
    assert kw["freeze_cap"] == 1          # ← 关键：不是 0（D-070：每回合只能按一次）
    assert kw["ban_trinkets"] is True
    assert kw["no_cancel_choose_on"] is False


def test_thirdparty_policy_receives_contract_flags():
    proto = EP.EvalProtocol(tp_forbid_cancel=False, keep_start_board=True,
                            freeze_cap=1, forbid_unfreeze=True)
    out = _make("thirdparty:x.pt", protocol=proto)
    kw = out["policy"].kwargs
    assert kw["forbid_cancel"] is False   # 作者只要求 no_cancel_choose
    assert kw["keep_start_board"] is True
    assert kw["freeze_cap"] == 1
    assert kw["forbid_unfreeze"] is True


def test_default_protocol_matches_training_contract():
    EP.EvalProtocol().assert_training_contract()   # 不抛即通过


@pytest.mark.parametrize("field,value", [
    ("max_actions_per_turn", 40),
    ("move_guard", False),
    ("move_cap", 9),
    ("move_repeat_limit", 9),
    ("forbid_cancel", False),
    ("forbid_unfreeze", False),
    ("freeze_cap", 0),
    ("ban_trinkets", False),
    ("no_cancel_choose", True),
])
def test_contract_drift_is_rejected(field, value):
    """任何一项与训练不一致都必须抛异常（fail-closed，不允许静默出分）。"""
    proto = EP.EvalProtocol(**{field: value})
    with pytest.raises(ValueError, match=field):
        proto.assert_training_contract()


def test_protocol_from_args_fills_training_defaults():
    args = SimpleNamespace(mirror_snaps="lib/x", mirror_sample="p50band",
                           freeze_cap=1, forbid_unfreeze=1)
    proto = EP.protocol_from_args(args, engine={"game_version": "37.0.2.256589"})
    assert proto.freeze_cap == 1
    assert proto.forbid_unfreeze is True
    assert proto.forbid_cancel is True          # 缺字段 → 用训练契约补
    assert proto.max_actions_per_turn == 100
    assert proto.mirror_snaps == "lib/x"
    proto.assert_training_contract()


def test_protocol_from_args_rejects_deprecated_freeze_cap_zero():
    """老评测入口 ``--freeze-cap 0``（默认值）必须被抓出来。"""
    args = SimpleNamespace(freeze_cap=0)
    proto = EP.protocol_from_args(args)
    with pytest.raises(ValueError, match="freeze_cap"):
        proto.assert_training_contract()


# ────────────────────────── P0-2：协议指纹 ──────────────────────────
def test_sha_is_stable_and_hex():
    a = EP.EvalProtocol(mirror_snaps="lib/x").sha256()
    b = EP.EvalProtocol(mirror_snaps="lib/x").sha256()
    assert a == b and len(a) == 64 and int(a, 16) >= 0


@pytest.mark.parametrize("kwargs", [
    {"mirror_snaps": "lib/y"},
    {"mirror_sample": "p75band"},
    {"mirror_strict": 1},
    {"mirror_seats": 3},
    {"forced_tier_plan": "2:2"},
    {"freeze_cap": 3},
    {"ban_trinkets": False},
    {"engine": {"game_version": "99"}},
    {"engine": {"source_hash": "deadbeef"}},
])
def test_sha_changes_with_every_scoring_relevant_field(kwargs):
    base = EP.EvalProtocol().sha256()
    assert EP.EvalProtocol(**kwargs).sha256() != base


def test_panel_protocol_can_be_re_verified_from_itself():
    """拿面板里存下来的 ``protocol`` 段重算指纹，必须等于它自己记的 sha。

    2026-11-17：以前 ``protocol_sha256_of`` 会把 ``protocol_sha256`` 键也算进去，
    于是"自证"必然 MISMATCH —— 审计者会误判面板被篡改。
    """
    proto = EP.EvalProtocol(mirror_snaps="lib/x", lobby_exclude=("DRAGON",))
    stored = dict(proto.as_dict())
    stored["protocol_sha256"] = proto.sha256()
    assert EP.protocol_sha256_of(stored) == proto.sha256()
    # 去掉那一行无关键后也必须一致
    body = {k: v for k, v in stored.items() if k != "protocol_sha256"}
    assert EP.protocol_sha256_of(body) == proto.sha256()


def test_lobby_exclude_is_part_of_the_protocol():
    """D-071 验证发现的缺陷：``lobby_exclude`` 原先不在 protocol 里 ⇒ 禁/不禁同 sha。

    两个环境不同的评测绝不能共享 ``protocol_sha256``，否则会被当同榜混比。
    """
    off = EP.EvalProtocol()
    on = EP.EvalProtocol(lobby_exclude=("MECHANICAL", "DRAGON"))
    assert off.sha256() != on.sha256()
    assert off.as_dict()["field"]["lobby_exclude"] == []
    assert on.as_dict()["field"]["lobby_exclude"] == ["MECHANICAL", "DRAGON"]
    # 只有空元组与"未设置"等价（回归：默认值必须保持旧 sha 的语义）
    assert EP.EvalProtocol(lobby_exclude=()).sha256() == off.sha256()


def test_protocol_from_args_reads_lobby_exclude():
    a = EP.protocol_from_args(SimpleNamespace(lobby_exclude="MECH,DRAGON"))
    assert a.lobby_exclude == ("MECHANICAL", "DRAGON")
    b = EP.protocol_from_args(SimpleNamespace(lobby_exclude=""))
    assert b.lobby_exclude == ()
    assert a.sha256() != b.sha256()


def test_sha_ignores_per_shard_run_params():
    """分片参数（局数/seed 区间/分片号）**不能**进指纹，否则 8 个分片互不相等。"""
    proto = EP.EvalProtocol(mirror_snaps="lib/x")
    r1 = proto.run_dict(games=128, seed_start=930000, seed_end=930127,
                        shard_index=0, shard_count=8)
    r2 = proto.run_dict(games=128, seed_start=930128, seed_end=930255,
                        shard_index=1, shard_count=8)
    assert r1 != r2
    assert r1["seed_start"] == 930000 and r2["seed_start"] == 930128
    # 指纹只由 protocol 决定
    assert proto.sha256() == proto.sha256()


def test_as_dict_is_json_serialisable_and_ordered():
    d = EP.EvalProtocol(mirror_snaps="lib/x").as_dict()
    json.dumps(d)                       # 必须可序列化
    assert list(d["contract"]) == list(EP.EVAL_CONTRACT_KEYS)
    assert d["schema"] == "hsai.eval_protocol/v1"


# ────────────────────────── assert_same_protocol ──────────────────────────
def test_assert_same_protocol_accepts_identical():
    p = {"protocol": {"protocol_sha256": "a" * 64}}
    assert EP.assert_same_protocol(p, dict(p)) == "a" * 64


def test_assert_same_protocol_rejects_missing():
    with pytest.raises(ValueError, match="没有 protocol_sha256"):
        EP.assert_same_protocol({"protocol": {"protocol_sha256": "a" * 64}},
                                {"models": []})


def test_assert_same_protocol_rejects_mismatch():
    a = {"protocol": {"protocol_sha256": "a" * 64}}
    b = {"protocol": {"protocol_sha256": "b" * 64}}
    with pytest.raises(ValueError, match="不一致"):
        EP.assert_same_protocol(a, b)
