import json
import pytest
from hsbg.az.train.economic_tasks import debug
from hsbg.az.train.policies import Decision
from workflows.probe_core_acquisition import FirstBuy, compare


def test_first_buy_intervenes_once_without_recording_a_fake_base_action():
    engine = pytest.importorskip('hsbg_engine')
    game = engine.Game(seed=11809231)
    while game.turn < 1:
        for seat in range(8):
            if game.turn >= 1:
                break
            if game.needs_action(seat):
                game.bot_play(seat)
    while json.loads(game.player_json(0))['pending']:
        game.step(0, game.bot_action(0))
    debug(game, 'clear_zone', zone='shop')
    debug(game, 'add_minion', zone='shop', card='BG_LOE_077')
    debug(game, 'set_player', gold=3)
    class Base:
        calls = 0
        def reset(self):
            self.calls = 0
        def decide(self, game, seat):
            self.calls += 1
            return Decision(game.bot_action(seat))
    base = Base()
    policy = FirstBuy(base, 8, True)
    policy.reset()
    decision = policy.decide(game, 0)
    assert decision.action == 8 and decision.info['forced']
    assert base.calls == 0
    game.step(0, decision.action)
    assert json.loads(game.player_json(0))['gold'] == 0
    policy.decide(game, 0)
    assert base.calls == 1


def test_comparison_clusters_replicates_within_source_case():
    rows = []
    for case, delta in [(0, -2), (1, 2)]:
        for replica in range(8):
            for arm in ('natural', 'buy'):
                rows.append({'model': 'model', 'case': case, 'replica': replica,
                             'core': 'core', 'seed': case*100+replica,
                             'arm': arm, 'rank': 4+(delta if arm == 'buy' else 0)})
    result = compare(rows)['model/core']
    assert result['cases'] == 2 and result['buy_minus_natural_rank'] == 0
    assert result['case_bootstrap_95ci'] == [-2., 2.]
    with pytest.raises(ValueError, match='unpaired'):
        compare(rows[:-1])
