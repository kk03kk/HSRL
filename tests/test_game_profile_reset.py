"""User-approved game profile: original freedoms, two confirmed fixes retained."""
import json
from pathlib import Path

import taverngym as tg
from taverngym.rl.config import Config


def test_source_profiles_restore_original_game_freedoms():
    root = Path(__file__).resolve().parents[1]
    for path in (root/'configs').rglob('*.json'):
        obj = json.loads(path.read_text(encoding='utf-8'))
        if 'rules' not in obj:
            continue
        c = Config.load(path)
        assert c.rules['turn_action_caps'] == [200]
        assert c.rules['max_actions_per_turn'] == 200
        assert c.rules['max_turns'] == 100
        assert c.rules['max_moves_per_turn'] == 0
        assert c.rules['block_move_cycles'] is False
        assert c.rules['lobby_types'] == c.rules['hero_bans'] == []
        assert len(c.rules['dark_gift_pair_bans']) == 3


def test_lobbies_are_random_and_no_naga():
    lobbies = set()
    for seed in range(12):
        env = tg.BatchEnv(1, seed=seed, controlled=range(8), config=Config().rules)
        races = json.loads(env.save(0))['game']['lobby_races']
        assert races.bit_count() == 5 and not races & (1 << 10)
        lobbies.add(races)
    assert len(lobbies) > 1


def test_more_than_sixty_actions_and_eight_repeated_moves_remain_legal():
    # Explicit unbounded reference configuration; training profiles use budget 200.
    g = tg.Game(541, json.dumps({**Config().rules, 'num_players': 2, 'turn_action_caps': []}))
    for p in range(2):
        g.step(p, json.loads(g.view(p))['actions']['CHOOSE'])
    for _ in range(2):
        g.debug(json.dumps({'cmd':'add_minion','p':0,'zone':'board','card':'BG25_008'}))
    state = json.loads(g.state_json())
    state['players'][0].update(actions_this_turn=270, moves_this_turn=12)
    g = tg.Game.from_json(json.dumps(state))
    v = json.loads(g.view(0))
    move = v['actions']['MOVE']
    # MOVE uses a seven-by-seven source/destination grid, including diagonal holes.
    for action in (move + 1, move + 7) * 6:
        assert g.legal_mask(0)[action]
        g.step(0, action)
    assert json.loads(g.state_json())['players'][0]['actions_this_turn'] == 282
