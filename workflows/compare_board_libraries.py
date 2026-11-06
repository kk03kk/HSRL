#!/usr/bin/env python3
"""战团库分回合统计与原生战斗探针；仅分析，不改训练库或模型。"""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_rows(bot, human):
    bots = []
    for path in sorted(Path(bot).glob('snap_*.json.gz')):
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for f in json.load(stream)['frames']:
                bots.append({'turn': f['turn'], 'game_id': str(f['source_game_seed']),
                    'tier': f['player']['tier'], 'board_detail': f['player']['board'],
                    'player': f['player']})
    humans = [json.loads(x) for x in Path(human).read_text(encoding='utf-8').splitlines() if x.strip()]
    return bots, humans


def summarize(rows):
    import numpy as np
    result = {}
    for turn in sorted({int(r['turn']) for r in rows}):
        bucket = [r for r in rows if int(r['turn']) == turn]
        atk, hp, sizes, signatures, cards = [], [], [], Counter(), set()
        for r in bucket:
            board = r['board_detail']
            atk.append(sum(m['atk'] for m in board))
            hp.append(sum(m['hp'] for m in board))
            sizes.append(len(board))
            sig = tuple(sorted(int(m['card']) for m in board))
            signatures[sig] += 1
            cards.update(sig)
        result[str(turn)] = {'n': len(bucket), 'source_games': len({r['game_id'] for r in bucket}),
            'attack': np.percentile(atk, [25, 50, 75, 90]).tolist(),
            'health': np.percentile(hp, [25, 50, 75, 90]).tolist(),
            'combined': np.percentile(np.array(atk)+hp, [25, 50, 75, 90]).tolist(),
            'median_board_size': float(np.median(sizes)), 'unique_cards': len(cards),
            'unique_compositions': len(signatures),
            'composition_collision_probability': sum(n*(n-1) for n in signatures.values()) / max(1, len(bucket)*(len(bucket)-1))}
    return result


def balanced_diversity(bots, humans):
    """统一样本量后报告组成/卡种丰富度，避免用不同总行数直接比较。"""
    rng = random.Random(20261102)
    result = {}
    for turn in sorted({r['turn'] for r in bots} & {r['turn'] for r in humans}):
        buckets = [[r for r in rows if r['turn'] == turn] for rows in (bots, humans)]
        n = min(64, *(len(b) for b in buckets))
        sides = []
        for bucket in buckets:
            compositions, cards = [], []
            for _ in range(100):
                sigs = [tuple(sorted(m['card'] for m in r['board_detail'])) for r in rng.sample(bucket, n)]
                compositions.append(len(set(sigs)))
                cards.append(len({c for s in sigs for c in s}))
            sides.append(dict(mean_unique_compositions=sum(compositions)/100, mean_unique_cards=sum(cards)/100))
        result[str(turn)] = dict(sample_rows=n, repeats=100, bot=sides[0], human=sides[1])
    return result


def install_board(game, seat, row, detail=True, full=False):
    """战斗探针容器：双方均不再招募；可见关键词覆写印刷默认值。"""
    if full:
        rep = game.apply_turn_snapshots([[seat, json.dumps(row['player'])]], False)
        if rep['applied'] != 1 or rep['rejected'] or not rep['pool_balance_ok'] or not rep['pool_species_ok']:
            raise RuntimeError(rep)
        from hsbg.az.train.human_obs_full import PLAYER_COMBAT_FIELDS, PLAYER_ACTIVE_FIELDS
        player = json.loads(game.player_json(seat))
        for key in (*PLAYER_COMBAT_FIELDS, *PLAYER_ACTIVE_FIELDS, 'deity'):
            if key in row['player']:
                player[key] = row['player'][key]
        game.set_player_from_json(seat, json.dumps(player), True, True)
        return
    board = row['board_detail']
    tier = row.get('tier') or 1
    rep = game.inject_board(seat, [[m['card'], m['atk'], m['hp']] for m in board], min(6, max(1, tier)))
    if not rep['applied'] or not rep['pool_balance_ok']:
        raise RuntimeError(rep)
    player = json.loads(game.player_json(seat))
    player['deity'] = None
    if detail:
        for dest, src in zip(player['board'], board):
            for key in ('kw', 'bonus_kw', 'golden', 'shield_max'):
                if key in src:
                    dest[key] = src[key]
            dest['shield_hits'] = max(1, dest['shield_max']) if dest['kw'] & 2 else 0
    game.set_player_from_json(seat, json.dumps(player), True, True)


def combat_probe(bots, humans, pairs, trials, seed):
    import hsbg_engine as E
    from workflows.mirror_hard_gates import _drive_to_turn
    rng = random.Random(seed)
    pools = []
    for rows in (bots, humans):
        buckets = defaultdict(lambda: defaultdict(list))
        for row in rows:
            buckets[int(row['turn'])][row['game_id']].append(row)
        pools.append(buckets)
    output = []
    # 先均匀抽源游戏，再均匀抽该游戏观测，避免长局/可见座位数主导结果。
    for turn in sorted(set(pools[0]) & set(pools[1])):
        picked = []
        for _ in range(pairs):
            picked.append([rng.choice(rng.choice(list(p[turn].values()))) for p in pools])
        for mode in ('printed_only', 'visible_keywords', 'human_visible_vs_bot_full'):
            wins = losses = ties = truncated = violations = 0
            dealt = taken = 0
            for index, (bot, human) in enumerate(picked):
                game = _drive_to_turn(E.Game(seed=seed+turn*10000+index), 1)
                state = json.loads(game.state_json())
                state['turn'] = turn
                for player in state['players']:
                    player['hero'] = None
                    player['deity'] = None
                    for key in ('powers', 'pending', 'trinkets', 'quests', 'rewards', 'secrets', 'effects'):
                        player[key] = []
                game = E.Game.from_json(json.dumps(state))
                # 每个配对轮换座位；winner 按传入 a/b 而非绝对座位解释。
                a, b = (0, 1) if index % 2 == 0 else (1, 0)
                install_board(game, a, human, detail=mode != 'printed_only')
                install_board(game, b, bot, detail=mode != 'printed_only', full=mode == 'human_visible_vs_bot_full')
                violations += len(game.invariant_violations())
                before = game.state_json()
                res = json.loads(game.debug(json.dumps({'cmd': 'combat_preview', 'a': a, 'b': b,
                    'n': trials, 'details': True})))
                if game.state_json() != before:
                    raise RuntimeError('combat_preview 改变原状态')
                for t in res['trials']:
                    wins += t['winner'] == 0
                    losses += t['winner'] == 1
                    ties += t['winner'] == 2
                    truncated += bool(t['truncated'])
                    dealt += t['damage_dealt']
                    taken += t['damage_taken']
            n = wins+losses+ties
            output.append({'turn': turn, 'mode': mode, 'pairs': pairs, 'trials': n,
                'human_win': wins/n, 'tie': ties/n, 'human_loss': losses/n,
                'damage_dealt': dealt/n, 'damage_taken': taken/n,
                'truncated': truncated, 'invariant_violations': violations})
    return output


def validate_humans(rows):
    """逐帧验证可见字段近似注入；不宣称已通过完整八人PPO对局验证。"""
    import hsbg_engine as E
    from workflows.mirror_hard_gates import _drive_to_turn
    state = json.loads(_drive_to_turn(E.Game(seed=730000), 1).state_json())
    for p in state['players']:
        p['pending'] = []
    template = json.dumps(state)
    rejected = []
    for i, row in enumerate(rows):
        game = E.Game.from_json(template)
        try:
            install_board(game, 0, row)
            if game.invariant_violations():
                raise RuntimeError(game.invariant_violations())
            got = json.loads(game.player_json(0))['board']
            for a, b in zip(got, row['board_detail']):
                if any(a[k] != b[k] for k in ('card', 'atk', 'hp', 'kw', 'golden')):
                    raise RuntimeError('可见字段往返不一致')
        except Exception as exc:
            rejected.append(dict(row=i, game_id=row['game_id'], turn=row['turn'], error=str(exc)))
    return dict(checked=len(rows), rejected=rejected, accepted=len(rows)-len(rejected),
        scope='仅独立帧、标准存活接收座位、无后续招募或EOT，不代表完整游戏真实性')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bot', required=True)
    p.add_argument('--human', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--pairs', type=int, default=32)
    p.add_argument('--trials', type=int, default=8)
    p.add_argument('--seed', type=int, default=20261102)
    a = p.parse_args()
    if not 1 <= a.pairs <= 256 or not 1 <= a.trials <= 64:
        p.error('pairs 1..256, trials 1..64')
    out = Path(a.out)
    if out.exists():
        raise FileExistsError('保留已有报告，请选择新输出路径')
    bots, humans = load_rows(a.bot, a.human)
    from workflows.run_learning_experiment import digest
    import hsbg_engine as E
    report = {'schema': 'hsrl.board_comparison/v1', 'bot_path': a.bot, 'human_path': a.human,
        'source_sha256': {**{p.name: digest(p) for p in sorted(Path(a.bot).glob('snap_*.json.gz'))},
            'human_jsonl': digest(a.human), 'analysis_script': digest(__file__)},
        'engine': dict(version=E.version(), game_version=E.game_version()),
        'quantiles': [25, 50, 75, 90], 'seed': a.seed,
        'bot': summarize(bots), 'human': summarize(humans),
        'bot_overlay_omits': {k: sum(bool(r['player'].get(k)) for r in bots) for k in ('owed', 'scheduled')},
        'human_train': summarize([r for r in humans if r.get('split') == 'train']),
        'balanced_diversity': balanced_diversity(bots, humans),
        'human_train_injection': validate_humans([r for r in humans if r.get('split') == 'train']),
        'combat': combat_probe(bots, [r for r in humans if r.get('split') == 'train'], a.pairs, a.trials, a.seed),
        'limitations': ['描述统计按观测帧；战斗先均匀抽游戏。不是模型棋力评测。',
            '真人仅Train参与战斗，不查询模型或封存Test；双方非配对真实对手。',
            '可见关键词有印刷回退，圣盾层数未知按1；缺手牌、玩家计数、饰品、磁力脚本。',
            'printed_only/visible_keywords为中性容器近似；full臂为Bot快照+安全战斗字段overlay，并非完整状态真值。',
            'Bot源Player中的owed/scheduled含EID语义，未overlay；记录非空帧数，不能把第三臂当完全保真。',
            '每个配对多个战斗随机数相关，不能把trials当独立源游戏。']}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
