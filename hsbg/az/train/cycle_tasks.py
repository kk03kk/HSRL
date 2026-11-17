"""Native current-patch resource/growth skills with explicitly bounded goals.

These synthetic tasks are NOT full-game expert demonstrations. The solver uses
only a declared action subset and a shortest-plan objective. Every accepted
label must have a native executable continuation; dataset construction also
requires agreement after replacing hidden RNG. No future shop is a target.
"""
import json
import numpy as np
from .economic_tasks import card_ids, own, item_card, debug

BRANN = 'BG_LOE_077'
TEMPEST = 'BGS_123'
ANOMALY = 'BGS_116'
KELP = 'BG36_701'
SELLEMENTAL = 'BGS_115'
DROPLET = 'BGS_115t'
UNBOUND = 'BG36_352'
KINDS = ('amplify_draw', 'buy_draw', 'activate_draw', 'free_refresh', 'elemental_growth')


def make_task(source, kind, variant=0):
    import hsbg_engine as engine
    state = source['state']
    if not int(state['lobby_races']) & (1 << 3):
        raise ValueError('Elemental lobby required')
    if kind == 'activate_draw' and not int(state['lobby_races']) & (1 << 5):
        raise ValueError('Murloc lobby required')
    if kind not in KINDS:
        raise ValueError(kind)
    game = engine.Game.from_json(json.dumps(state))
    p = own(game)
    if p['pending'] or p['ended_turn']:
        raise ValueError('actionable source required')
    edits = []
    def edit(cmd, **kw):
        debug(game, cmd, **kw)
        edits.append({'cmd': cmd, **kw})
    ids = card_ids()
    # Clear potential copies of task cards so accidental triples do not change
    # the intended task. Retain unrelated natural units, never sell them in labels.
    task_names = (BRANN, TEMPEST, ANOMALY, KELP, SELLEMENTAL, DROPLET, UNBOUND,
                  'TB_BaconUps_045', 'TB_BaconUps_157', 'BG36_352_G', 'BG36_701_G')
    remove = {ids[c] for c in task_names if c in ids}
    for i in reversed(range(len(p['board']))):
        if p['board'][i]['card'] in remove:
            edit('remove', zone='board', index=i)
    core = [] if kind == 'amplify_draw' else [BRANN]
    if kind == 'activate_draw':
        core += [KELP, TEMPEST, ANOMALY]
    if kind == 'elemental_growth':
        core = [UNBOUND]
    capacity = 5 if kind == 'elemental_growth' else 4
    while len(own(game)['board']) + len(core) > capacity:
        board = own(game)['board']
        weakest = min(range(len(board)), key=lambda j: board[j]['atk'] + board[j]['hp'])
        edit('remove', zone='board', index=weakest)
    rng = np.random.default_rng(int(source['seed']) + variant * 7919)
    rng.shuffle(core)
    for card in core:
        edit('add_minion', zone='board', card=card)
    edit('clear_zone', zone='hand')
    hand = {'amplify_draw': [BRANN, TEMPEST], 'buy_draw': [],
            'activate_draw': [], 'free_refresh': [ANOMALY],
            'elemental_growth': [SELLEMENTAL, ANOMALY]}[kind]
    rng.shuffle(hand)
    for card in hand:
        edit('add_minion', zone='hand', card=card)
    if kind == 'buy_draw':
        if len(own(game)['shop']) >= 8:
            edit('remove', zone='shop', index=7)
        edit('add_minion', zone='shop', card=TEMPEST)
    if kind == 'elemental_growth':
        # A visible already-buffed shop provides a real payoff for three plays.
        # It is not a future offer or a reward injected after the decision.
        if len(own(game)['shop']) >= 8:
            edit('remove', zone='shop', index=7)
        edit('add_minion', zone='shop', card=SELLEMENTAL, atk=40 + variant % 3 * 10,
             hp=40 + variant % 3 * 10)
    edit('set_player', gold=3 + variant % 5)
    # Free-refresh endowment is an explicit fixture edit (debug lacks this field).
    state = json.loads(game.state_json())
    state['players'][0]['free_refreshes'] = 0
    edits.append({'state_field': 'players[0].free_refreshes', 'value': 0})
    game = engine.Game.from_json(json.dumps(state))
    p = own(game)
    protected = [m['eid'] for m in p['board']]
    goal = {'kind': kind, 'gold': p['gold'], 'refreshes': p['stats']['refreshes'],
            'elementals': p['stats']['elementals_played'], 'protected': protected,
            'max_depth': 6}
    if kind == 'elemental_growth':
        unit = next(m for m in p['board'] if m['card'] == ids[UNBOUND])
        goal.update(core_eid=unit['eid'], core_strength=unit['atk'] + unit['hp'])
    return game, goal, edits


def achieved(game, goal):
    p = own(game)
    if not set(goal['protected']) <= {m['eid'] for m in p['board']}:
        return False
    ids = card_ids()
    kind = goal['kind']
    if kind in ('amplify_draw', 'buy_draw', 'activate_draw'):
        return (len(p['hand']) >= 2
                and any(m['card'] == ids[BRANN] for m in p['board'])
                and any(m['card'] == ids[TEMPEST] for m in p['board']))
    if kind == 'free_refresh':
        return (p['stats']['refreshes'] - goal['refreshes'] == 1
                and p['gold'] >= goal['gold'] and p['free_refreshes'] >= 3)
    if kind == 'elemental_growth':
        unit = next((m for m in p['board'] if m['eid'] == goal['core_eid']), None)
        return (unit is not None and unit['atk'] + unit['hp'] > goal['core_strength']
                and p['stats']['elementals_played'] - goal['elementals'] >= 3)
    raise ValueError(kind)


def candidates(game, goal, *, placements=False):
    p = own(game)
    mask = bytes(game.legal_mask(0))
    if p['pending']:
        return [a for a in range(30, 46) if mask[a]]
    ids = card_ids()
    kind = goal['kind']
    allowed = {ids[c] for c in (BRANN, TEMPEST, ANOMALY, SELLEMENTAL, DROPLET)}
    actions = []
    for i, card in enumerate(p['hand'][:10]):
        if item_card(card) not in allowed:
            continue
        legal = [46 + i * 8 + j for j in range(8) if mask[46 + i * 8 + j]]
        actions.extend(legal if placements else legal[-1:])
    if kind == 'buy_draw':
        actions += [8+i for i, s in enumerate(p['shop'][:8])
                    if item_card(s) == ids[TEMPEST] and mask[8+i]]
    if kind == 'activate_draw':
        actions += [23+i for i, m in enumerate(p['board'])
                    if m['card'] == ids[KELP] and mask[23+i]]
    if kind == 'free_refresh' and mask[1]:
        actions.append(1)
    if kind == 'elemental_growth':
        actions += [16+i for i, m in enumerate(p['board'])
                    if m['eid'] not in goal['protected'] and mask[16+i]]
    return sorted(set(actions))


def solve(game, goal, max_depth=None):
    """Shortest native plans within the declared subset, including tied placements."""
    limit = goal['max_depth'] if max_depth is None else int(max_depth)
    cache = {}
    def distance(g, remaining):
        if achieved(g, goal):
            return 0
        if remaining <= 0:
            return None
        key = (g.state_json(), remaining)
        if key in cache:
            return cache[key]
        best = None
        for action in candidates(g, goal):
            child = g.copy()
            child.step(0, action)
            d = distance(child, remaining - 1)
            if d is not None:
                best = min(best if best is not None else limit + 1, d + 1)
                if best == 1:
                    break
        cache[key] = best
        return best
    if achieved(game, goal):
        return {'distance': 0, 'actions': []}
    distances = {}
    for action in candidates(game, goal, placements=True):
        child = game.copy()
        child.step(0, action)
        d = distance(child, limit - 1)
        if d is not None:
            distances[action] = d + 1
    if not distances:
        raise ValueError('no bounded native plan')
    best = min(distances.values())
    return {'distance': best, 'actions': [a for a, d in distances.items() if d == best]}
