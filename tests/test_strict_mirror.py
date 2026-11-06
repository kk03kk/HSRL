"""严格采集拒绝普通对手回退，含真实待选与同库末端储备。"""
import json
import random
from types import SimpleNamespace

import pytest

from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
from hsbg.az.train.mirror_snapshots import MirrorStats
from hsbg.az.train.policies import EngineBotPolicy
from hsbg.az.train.strict_mirror import StrictLibrary, StrictMirror
from tests.test_injection_meta import _synthetic_mirror_pool


def test_strict_native_pending_and_actual_combat_accounting(tmp_path):
    pool = _synthetic_mirror_pool(tmp_path)
    result = evaluate_policy(EngineBotPolicy(), EvalConfig(
        games=8, seed=881200, mirror_pool=pool, mirror_strict=True, collect_exposure=True,
    ))
    mirror, strict = result["mirror"], result["mirror_strict_counts"]
    assert result["truncated_games"] == result["illegal_actions"] == 0
    assert mirror["mirror_attempt"] == mirror["mirror_applied"] == strict["eligible_live_seats"]
    assert mirror["mirror_rejected"] == mirror["mirror_fallback"] == mirror["mirror_errors"] == 0
    assert mirror["mirror_pool_unbalanced"] == mirror["mirror_pool_species_unbalanced"] == 0
    assert strict["pending_choices_resolved"] > 0
    assert strict["tail_reserve"] > 0
    combats = result["combat_exposure"]
    assert strict["live_combats"] + strict.get("ghost_combats", 0) == len(combats)
    assert all(row["ghost"] or row["mirror_applied"] for row in combats)


def test_strict_ghost_must_trace_to_calibrated_history():
    def finished_round(ghost):
        return SimpleNamespace(turn=7, is_over=lambda: False, state_json=lambda: json.dumps({
            "last_combats": [{"turn": 6, "a": 0, "b": None}],
            "ghost_player": ghost,
        }))

    mirror = StrictMirror(None, 0, random.Random(1), MirrorStats())
    mirror.turn, mirror.learner_active = 6, True
    mirror.certified[1] = 4
    mirror.after_learner_step(finished_round(1))
    mirror.after_learner_step(finished_round(1))
    assert mirror.counts["ghost_combats"] == 1
    mirror.audited_turn = 0
    with pytest.raises(RuntimeError, match="幽灵无校准库来源"):
        mirror.after_learner_step(finished_round(2))


def test_strict_library_rejects_missing_interior_turn():
    pool = SimpleNamespace(turns=lambda: [1, 3], has=lambda *_: True)
    with pytest.raises(ValueError, match="缺少整个回合"):
        StrictLibrary(pool)


def test_native_nested_eclipse_discover_reaches_strict_injection(tmp_path):
    import hsbg_engine as E
    from workflows.mirror_hard_gates import _drive_to_turn

    game = _drive_to_turn(E.Game(seed=33), 9)
    player = json.loads(game.player_json(4))
    player['gold'] = 6
    player['pending'] = [
        {'Trinket': {'options': [3052, 3024, 3039, 3027], 'free': False, 'replace': 3035}},
        {'Trinket': {'options': [3051], 'free': False, 'replace': None}},
    ]
    game.set_player_from_json(4, json.dumps(player), True, True)
    game.step(4, 30)
    assert json.loads(game.player_json(4))['gold'] == 2
    assert 'Discover' in json.loads(game.player_json(4))['pending'][-1]
    mirror = StrictMirror(StrictLibrary(_synthetic_mirror_pool(tmp_path)), 0, random.Random(91), MirrorStats())
    mirror.prepare(game)
    assert not json.loads(game.player_json(4))['pending']
    assert mirror.certified[4] == 9
    assert mirror.stats.applied == mirror.counts['eligible_live_seats']
    assert not mirror.stats.fallback


def test_strict_mirror_cancels_empty_native_target_before_injection(tmp_path):
    import hsbg_engine as E
    from workflows.mirror_hard_gates import _drive_to_turn

    game = _drive_to_turn(E.Game(seed=33), 9)
    player = json.loads(game.player_json(4))
    player["pending"] = [{"Target": {"options": [], "cont": {"HeroPower": {"opt": 0, "slot": 0}}}}]
    game.set_player_from_json(4, json.dumps(player), True, True)
    assert game.bot_action(4) == 7
    assert [i for i, legal in enumerate(game.legal_mask(4)) if legal] == [7]
    mirror = StrictMirror(StrictLibrary(_synthetic_mirror_pool(tmp_path)), 0, random.Random(91), MirrorStats())
    class WrongRecommendation:
        def __getattr__(self, name):
            return getattr(game, name)
        def bot_action(self, seat):
            return 30 if seat == 4 else game.bot_action(seat)
    # 原生 Bot 已修复，仍模拟坏推荐以验证 Python 防线未被删除。
    mirror.prepare(WrongRecommendation())
    assert not json.loads(game.player_json(4))["pending"]
    assert mirror.certified[4] == 9
    assert mirror.counts["pending_bot_action_corrected"] >= 1
    assert mirror.stats.applied == mirror.counts["eligible_live_seats"]


def test_strict_mirror_fails_with_diagnostics_for_no_legal_pending_action(tmp_path, monkeypatch):
    import gzip
    import hsbg_engine as E
    from workflows.mirror_hard_gates import _drive_to_turn

    game = _drive_to_turn(E.Game(seed=33), 9)
    player = json.loads(game.player_json(4))
    player["pending"] = [{"Target": {"options": [], "cont": {"Spell": {"card": 1, "data": 0, "opt": 0}}}}]
    game.set_player_from_json(4, json.dumps(player), True, True)
    monkeypatch.setenv("HSRL_FAILURE_DIR", str(tmp_path/"failures"))
    mirror = StrictMirror(StrictLibrary(_synthetic_mirror_pool(tmp_path)), 0, random.Random(91), MirrorStats())
    with pytest.raises(RuntimeError, match=r"legal=\[\].*pending="):
        mirror.prepare(game)
    files = list((tmp_path/"failures").glob("strict-*.json.gz"))
    assert len(files) == 1
    with gzip.open(files[0], "rt", encoding="utf-8") as stream:
        evidence = json.load(stream)
    assert evidence["schema"] == "hsrl.strict_failure/v1"
    replay = E.Game.from_json(json.dumps(evidence["state"]))
    assert replay.state_json() == game.state_json()
    assert not any(replay.legal_mask(4))


class NativePanicForAudit(BaseException):
    pass


@pytest.mark.parametrize('error_type', [NativePanicForAudit, KeyboardInterrupt, SystemExit])
def test_native_panic_keeps_evidence_but_user_interrupt_propagates(tmp_path, monkeypatch, error_type):
    import hsbg_engine as E
    from workflows.mirror_hard_gates import _drive_to_turn
    game = _drive_to_turn(E.Game(seed=33), 1)
    monkeypatch.setenv('HSRL_FAILURE_DIR', str(tmp_path/'states'))
    class InterruptedGame:
        def __getattr__(self, name):
            return getattr(game, name)
        def legal_mask(self, seat):
            raise error_type('native-panic')
    mirror = StrictMirror(None, 0, random.Random(91), MirrorStats())
    expected = RuntimeError if error_type is NativePanicForAudit else error_type
    with pytest.raises(expected, match='native-panic'):
        mirror.prepare(InterruptedGame())
    assert len(list((tmp_path/'states').glob('*.json.gz'))) == int(error_type is NativePanicForAudit)
