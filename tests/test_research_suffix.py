import json
import pytest
from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
from hsbg.az.train.policies import EngineBotPolicy


def source_state():
    engine = pytest.importorskip('hsbg_engine')
    game = engine.Game(seed=11809001)
    while game.turn < 8 and not game.is_over():
        for seat in range(8):
            if game.turn >= 8:
                break
            if game.needs_action(seat):
                game.bot_play(seat)
    while json.loads(game.player_json(0))['pending']:
        game.step(0, game.bot_action(0))
    return json.loads(game.state_json())


def test_suffix_uses_exact_state_and_does_not_mutate_source():
    state = source_state()
    original = json.dumps(state, sort_keys=True)
    class Recorder:
        first = None
        def begin_game(self, game, game_index, seed, seat):
            self.first = json.loads(game.state_json())
        def before(self, game, seat):
            return None
        def after(self, *args):
            pass
        def end_game(self, *args):
            pass
    recorder = Recorder()
    result = evaluate_policy(EngineBotPolicy(), EvalConfig(
        games=1, seed=state['seed'], research_initial_state=state, decision_recorder=recorder))
    assert recorder.first == state
    assert result['research_suffix'] and result['initial_turn'] == 8
    assert result['games_finished'] == 1 and result['illegal_actions'] == 0
    assert json.dumps(state, sort_keys=True) == original
    with pytest.raises(ValueError, match='one game'):
        evaluate_policy(EngineBotPolicy(), EvalConfig(games=2, seed=state['seed'], research_initial_state=state))
    with pytest.raises(ValueError, match='seed'):
        evaluate_policy(EngineBotPolicy(), EvalConfig(games=1, seed=state['seed']+1, research_initial_state=state))


def test_suffix_preserves_current_turn_opponents(monkeypatch):
    import hsbg.az.train.evaluate as module
    state = source_state()
    seen = []
    def inject(game, pool, turn, **kwargs):
        seen.append(turn)
        assert turn > state['turn']
    monkeypatch.setattr(module, 'inject_mirror_turn', inject)
    result = evaluate_policy(EngineBotPolicy(), EvalConfig(
        games=1, seed=state['seed'], research_initial_state=state, mirror_pool=object()))
    assert result['games_finished'] == 1
    assert seen and min(seen) > state['turn']
