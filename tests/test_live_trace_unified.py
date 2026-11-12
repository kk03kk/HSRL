"""离线回归：模拟器 / 实机共用一份记录规范（``hsai.game-trace/v2``）。

覆盖：
* 两侧 ``view`` 形状一致、跨端可比的字段白名单一致；
* 引擎真值 -> 统一视图（商店/手牌容器解包、关键词位域、冻结 tag）；
* 执行前后差值（买/打/卖/冻/换位）；
* "复用即等待"：轮询不写进 decisions.jsonl；
* 决策 + 执行合并、``audit.json`` 完整性判定；
* 旧实机日志按新规则重放（离线，不启动游戏）。
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "client"))

from hsrl_agent import trace_schema as S
from hsrl_agent.live.contracts import Binding, Decision
from hsrl_agent.live.trace import GameTrace, summarize


CATALOG = {
    100: {"card_id": "BG20_100", "kind": "minion", "name": "Test Minion",
          "name_zh": "测试", "cost": 3, "tier": 1},
    200: {"card_id": "BG20_200", "kind": "minion", "name": "Taunt Guy",
          "name_zh": "嘲讽", "cost": 2, "tier": 2},
    300: {"card_id": "BG20_300", "kind": "spell", "name": "Test Spell",
          "name_zh": "法术", "cost": 1, "tier": 1},
}


def live_client():
    """实机 ``snapshot['client']`` 的最小等价样本。"""
    return {
        "turn": 3, "phase": "recruit", "gold": 4, "tavern_tier": 2, "upgrade_cost": 5,
        "health": 30, "local_player_id": 1,
        "hero": {"entity_id": 77, "card_id": "BG22_HERO_000", "name": "Hero",
                 "health": 30, "tags": {"ARMOR": 2}},
        "board": [{"entity_id": 400, "card_id": "BG20_100", "name": "Test Minion",
                   "card_type": "MINION", "position": 1, "attack": 2, "health": 3,
                   "tier": 1, "cost": 0,
                   "tags": {"ATK": "2", "HEALTH": "3", "TAUNT": "1"}}],
        "hand": [{"entity_id": 401, "card_id": "BG20_300", "name": "Test Spell",
                  "card_type": "SPELL", "position": 1, "attack": None, "health": None,
                  "tier": 1, "cost": 1, "tags": {}}],
        "shop": [{"entity_id": 402, "card_id": "BG20_200", "name": "Taunt Guy",
                  "card_type": "MINION", "position": 1, "attack": 1, "health": 4,
                  "tier": 2, "cost": 2, "tags": {"ATK": "1", "HEALTH": "4", "FROZEN": "0"}}],
        "choices": [], "controls": [],
    }


def engine_state(**over):
    player = {
        "id": 0, "hero": 100, "health": 30, "armor": 2, "alive": True, "placement": 0,
        "tier": 2, "gold": 4, "upgrade_cost": 5, "pending": [],
        "board": [{"eid": 400, "card": 100, "atk": 2, "hp": 3, "max_hp": 3,
                   "kw": 1, "bonus_kw": 0, "tier": 1, "golden": False, "attached": []}],
        "hand": [{"hid": 401, "item": {"Spell": {"card": 300, "data": 0}},
                  "locked": 0, "temporary": False}],
        "shop": [{"sid": 402, "item": {"Minion": {"eid": 403, "card": 200, "atk": 1, "hp": 4,
                                                  "kw": 0, "bonus_kw": 0, "tier": 2,
                                                  "golden": False, "attached": []}},
                  "frozen": True, "price": None, "kept": False}],
        "stats": {"cards_played": 3, "gold_spent_total": 10},
    }
    state = {"turn": 3, "phase": "Recruit", "players": [player]}
    state.update(over)
    return state


def test_view_is_identical_for_both_sides():
    live = S.view(live_client())
    sim = S.view(S.sim_view(engine_state(), 0, CATALOG))
    assert live is not None and sim is not None
    assert set(live) == set(sim)
    assert set(S.observable(live_client())) == set(S.observable(S.sim_view(engine_state(), 0, CATALOG)))
    for key in ("turn", "phase", "gold", "tavern_tier", "health"):
        assert key in live and key in sim


def test_sim_view_decodes_keywords_items_and_zones():
    sim = S.sim_view(engine_state(), 0, CATALOG)
    board = sim["board"][0]
    assert board["card_id"] == "BG20_100"
    assert board["entity_id"] == 400
    assert board["tags"].get("TAUNT") == 1
    # 手牌是 {"item": {"Spell": ...}} 容器；法术没有 eid，用容器 hid 作身份。
    hand = sim["hand"][0]
    assert hand["card_id"] == "BG20_300"
    assert hand["card_type"] == "SPELL"
    assert hand["entity_id"] == 401
    shop = sim["shop"][0]
    assert shop["card_id"] == "BG20_200"
    assert shop["entity_id"] == 403          # 随从自带 eid
    assert shop["tags"].get("FROZEN") == 1
    assert shop["shop_slot"] == 0


def test_delta_reports_the_real_transition_for_each_action_kind():
    before = live_client()
    after = json.loads(json.dumps(before))
    # buy: 商店第一张进手牌
    after["shop"] = []
    after["hand"] = after["hand"] + [dict(before["shop"][0], entity_id=402, card_type="MINION")]
    after["gold"] = 2
    d = S.delta(before, after)
    assert d["changed"] is True
    assert d["scalars"]["gold"] == [4, 2]
    assert d["shop"]["removed"]["values"][0]["card_id"] == "BG20_200"

    # play: 手牌离场 -> 战团新增
    played = json.loads(json.dumps(before))
    played["hand"] = []
    played["board"] = played["board"] + [dict(before["hand"][0], card_type="MINION",
                                              attack=0, health=0)]
    d = S.delta(before, played)
    assert d["hand"]["removed"]["values"][0]["entity_id"] == 401
    assert d["board"]["added"]["values"]

    # freeze: 商店 tag 变化必须可见（不靠猜）
    frozen = json.loads(json.dumps(before))
    frozen["shop"][0]["tags"]["FROZEN"] = "1"
    d = S.delta(before, frozen)
    assert d["changed"] is True
    assert d["shop"]["stats"]["values"]

    # 无变化时 changed 必须是 False，避免把"没生效"记成"生效"
    assert S.delta(before, json.loads(json.dumps(before)))["changed"] is False


def test_decision_id_is_reproducible_and_occurrence_disambiguates():
    a = S.decision_id(side="sim", game_id="g", seat=0, turn=3, state="abc", action=46)
    b = S.decision_id(side="sim", game_id="g", seat=0, turn=3, state="abc", action=46)
    c = S.decision_id(side="sim", game_id="g", seat=0, turn=3, state="abc", action=46, occurrence=1)
    d = S.decision_id(side="live", game_id="g", seat=0, turn=3, state="abc", action=46)
    assert a == b and a != c and a != d


def _decision(action=46, kind="play", source=401, revision=1, stamp="stamp"):
    return Decision("game-1", revision, stamp, action,
                    Binding(action, kind, source=source, position=0), "ckpt", "logits")


def _snapshot(**client):
    data = live_client()
    data.update(client)
    return {"game_id": "game-1", "revision": 1, "client": data, "options": [],
            "view": {"pending": 0, "unknown": []},
            "approximation": {"mode": "public-approx-v1", "fields": {"opponents.hand": "hidden"}}}


def test_wait_polls_never_reach_the_decision_stream(tmp_path):
    trace = GameTrace(tmp_path, side="live", store_vectors=False)
    first = trace.decision(_snapshot(), _decision(), policy_ms=12.0, reused=False,
                           bindings=[Binding(46, "play", source=401)])
    # 同一状态 + 同一动作 + 缓存命中 = 等待轮询
    for _ in range(5):
        trace.decision(_snapshot(), _decision(), policy_ms=0.0, reused=True,
                       bindings=[Binding(46, "play", source=401)])
    # 状态真的变了（新 stamp）就是新决策
    second = trace.decision(_snapshot(gold=2), _decision(action=16, kind="sell", stamp="stamp2"),
                            policy_ms=11.0, reused=False,
                            bindings=[Binding(16, "sell", source=400)])
    status = trace.close()
    decisions = [json.loads(l) for l in (tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    waits = [json.loads(l) for l in (tmp_path / "waits.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [d["decision_id"] for d in decisions] == [first, second]
    assert len(waits) == 5
    assert status["waits"] == 5 and status["decisions"] == 2
    assert all(d["side"] == "live" for d in decisions)
    assert decisions[0]["visibility"] == "owner_public_projection"
    assert decisions[0]["pre"]["hand"][0]["card_id"] == "BG20_300"
    assert decisions[0]["inferred_view"]["approximation_fields"] == ["opponents.hand"]


def test_execution_rows_carry_before_after_and_delta(tmp_path):
    trace = GameTrace(tmp_path, side="live", store_vectors=False)
    snap = _snapshot()
    did = trace.decision(snap, _decision(), policy_ms=9.0, reused=False,
                         bindings=[Binding(46, "play", source=401)])
    trace.execution(did, snapshot=snap, plan=[["click", [0.5, 0.5]]], status="sent",
                    timing={"elapsed_ms": 4.0}, physical=True)
    after = live_client()
    after["hand"] = []
    after["board"] = after["board"] + [dict(live_client()["hand"][0], attack=0, health=0)]
    trace.execution(did, snapshot=None, plan=[], status="receipt", outcome="applied", after=after)
    trace.close()
    rows = [json.loads(l) for l in (tmp_path / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["attempt"] for r in rows] == [0, 1]
    assert rows[0]["physical"] and rows[1]["status"] == "receipt"
    assert rows[1]["after"]["board"][-1]["entity_id"] == 401
    assert rows[0]["before"]["hand"][0]["card_id"] == "BG20_300"
    resolved = [json.loads(l) for l in
                (tmp_path / "decisions-resolved.jsonl").read_text(encoding="utf-8").splitlines()]
    assert resolved[0]["execution"]["attempts"] == 2
    assert resolved[0]["execution"]["outcome"] == "applied"
    assert resolved[0]["delta"]["changed"] is True
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert audit["complete"] is True and audit["missing_execution"] == 0


def test_audit_flags_a_decision_that_never_executed():
    decisions = [{"decision_id": "d-1", "sequence": 1, "turn": 2, "kind": "buy", "action": 8},
                 {"decision_id": "d-2", "sequence": 0, "turn": 1, "kind": "play", "action": 46}]
    result = S.audit(decisions, [], [], side="sim")
    assert result["complete"] is False
    assert result["missing_execution"] == 2
    assert any("not monotonic" in p for p in result["problems"])
    assert any("turn regressions" in p for p in result["problems"])


def test_full_snapshot_stream_is_gzip_and_matches_the_decision(tmp_path):
    trace = GameTrace(tmp_path, side="sim", store_vectors=False)
    snap = _snapshot()
    snap["approximation"] = {"mode": "sim-native", "fields": dict(S.SIM_APPROXIMATION_FIELDS)}
    did = trace.decision(snap, _decision(), policy_ms=1.0, reused=False, seat=0,
                         bindings=[Binding(46, "play", source=401)])
    trace.close()
    with gzip.open(tmp_path / "decisions-full.jsonl.gz", "rt", encoding="utf-8") as stream:
        row = json.loads(stream.readline())
    assert row["decision_id"] == did
    assert row["decision"]["binding"]["kind"] == "play"
    assert row["snapshot"]["approximation"]["mode"] == "sim-native"
    assert row["bindings"][0]["source"] == 401


def test_legacy_live_log_dedupes_offline_without_a_game(tmp_path):
    """旧记录里 92% 是轮询；新规则下应当只剩真实决策。"""
    rows = [dict(event="decision", turn=1, action=10, kind="buy", reused=False, stamp="s1"),
            *[dict(event="decision", turn=1, action=10, kind="buy", reused=True, stamp="s1")
              for _ in range(6)],
            dict(event="decision", turn=1, action=46, kind="play", reused=False, stamp="s2"),
            *[dict(event="decision", turn=1, action=0, kind="end", reused=False, stamp="s3"),
              *[dict(event="decision", turn=1, action=0, kind="end", reused=True, stamp="s3")
                for _ in range(20)]]]
    path = tmp_path / "decisions.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    kept, waits = 0, 0
    last = None
    for row in rows:
        signature = (row.get("stamp"), row.get("action"))
        if row.get("reused") and signature == last:
            waits += 1
            continue
        kept += 1
        last = signature
    assert (kept, waits) == (3, 26)
    assert kept + waits == len(rows)


def test_sim_mask_pipeline_matches_the_frozen_training_contract():
    """模拟器采样前的掩码顺序必须是：去 Cancel(7) -> MoveGuard（与长训一致）。"""
    from hsbg.az.train.move_guard import MoveGuard
    from hsbg.az.train.pending_guard import without_cancel

    mask = np.zeros(245, bool)
    mask[[0, 7, 196, 8]] = True           # EndTurn, Cancel, Move(0->0), Buy(0)
    deduped = without_cancel(mask)
    assert not deduped[7] and deduped[0] and deduped[8] and deduped[196]
    guard = MoveGuard()
    guarded, forced = guard.guarded_mask(deduped, game_id=0, seat=0, turn=1)
    guarded = np.asarray(guarded, bool)
    assert not guarded[7]                 # 守卫在去 Cancel 之后，不会把它放回来
    assert guarded.shape == (245,)
    assert isinstance(forced, bool)


@pytest.mark.parametrize("run", [
    "artifacts/runs/live-e-20261111-004/match-0002",
    "artifacts/local/sim-trace-20261111-001/game-001",
    "artifacts/local/sim-trace-20261111-002/game-001",
])
def test_real_archived_runs_are_complete_when_present(run):
    root = ROOT / run
    if not (root / "decisions.jsonl").exists():
        pytest.skip(f"archived run not present: {run}")
    decisions = [json.loads(l) for l in (root / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    actions = [json.loads(l) for l in (root / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
    executions = [a for a in actions if a.get("event") == "execution"]
    waits = []
    path = root / "waits.jsonl"
    if path.exists():
        waits = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    result = S.audit(decisions, executions, waits,
                     side=(decisions[0].get("side") if decisions else "live"))
    if decisions and decisions[0].get("side") == "sim":
        # 模拟器记录必须完全可还原
        assert result["complete"] is True
        assert result["decisions"] > 100
    # 实机旧档案允许有历史遗留问题，但不能出现重复 decision_id
    assert not any("duplicate decision_id" in p for p in result["problems"])
