#!/usr/bin/env python3
"""按真人逐回合标尺和原生条件战斗筛选完整快照；不修改身材/手牌/计数。"""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def signature(row):
    return tuple(sorted(int(m['card']) for m in row['player']['board']))


def mechanics(player):
    """原始关键词位与隐状态覆盖，不把缺字段当零真值。"""
    board = player.get('board') or []
    return dict(keyword_bits={str(bit):sum(bool(int(m.get('kw', 0)) & (1 << bit)) for m in board)
                              for bit in range(16)},
        hand=len(player.get('hand') or []),
        active_fields=[k for k, v in player.items() if v and k not in
                       ('board', 'hand', 'shop', 'pending', 'health', 'gold', 'hero', 'tier')])


def read_sources(sources, split):
    from hsbg.az.train.personal import checked_library
    result, manifests = [], {}
    seen = set()
    for name, path in sources.items():
        manifest = checked_library(path)
        if manifest['split'] != split or not manifest.get('validation', {}).get('accepted'):
            raise ValueError('必须使用逐帧验证过、分割匹配的源库')
        manifests[name] = manifest
        for file in sorted(Path(path).glob('snap_*.json.gz')):
            with gzip.open(file, 'rt', encoding='utf-8') as stream:
                record = json.load(stream)
            for index, frame in enumerate(record['frames']):
                row = dict(frame, source=name, source_file=file.name, source_index=index)
                identity = (name, row['source_game_seed'], row['turn'], row['seat'])
                if identity in seen:
                    raise ValueError(f'重复源帧：{identity}')
                seen.add(identity)
                result.append(row)
    return result, manifests


def sample_candidates(rows, per_source, seed):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for row in rows:
        buckets[(int(row['turn']), int(row['seat']), row['source'])].append(row)
    return [row for key in sorted(buckets) for row in rng.sample(buckets[key], min(per_source, len(buckets[key])))]


def human_references(rows, seed, count=2):
    """每局等权选一个视角，再从中央50%身材分布抽固定参考，精彩局偏差仍存在。"""
    rng = random.Random(seed)
    buckets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get('split') != 'train':
            raise ValueError('校准只允许Firestone Train，不能混入Dev/Test')
        buckets[int(row['turn'])][row['game_id']].append(row)
    result = {}
    for turn, games in sorted(buckets.items()):
        reps = [rng.choice(games[k]) for k in sorted(games)]
        reps.sort(key=lambda r: sum(m['atk']+m['hp'] for m in r['board_detail']))
        central = reps[len(reps)//4:max(len(reps)//4+1, (len(reps)*3)//4)]
        picked = rng.sample(central, min(count*2, len(central)))
        result[turn] = dict(games=len(games), median_strength=sum(m['atk']+m['hp'] for m in reps[len(reps)//2]['board_detail']),
                            anchors=picked[:count], audit_anchors=picked[count:])
    return result


def combat_score(row, anchors, template, trials):
    """复用现有探针的安全字段overlay；不是完整真人战斗真值。"""
    import hsbg_engine as E
    from workflows.compare_board_libraries import install_board
    wins = ties = n = 0
    dealt = taken = 0
    omissions = {k:bool(row['player'].get(k)) for k in ('owed', 'scheduled')}
    for index, anchor in enumerate(anchors):
        game = E.Game.from_json(template)
        a, b = (0, 1) if index % 2 == 0 else (1, 0)
        install_board(game, a, dict(player=row['player']), full=True)
        install_board(game, b, anchor)
        if game.invariant_violations():
            raise RuntimeError(game.invariant_violations())
        before = game.state_json()
        report = json.loads(game.debug(json.dumps(dict(cmd='combat_preview', a=a, b=b, n=trials, details=True))))
        if game.state_json() != before:
            raise RuntimeError('preview改变状态')
        for trial in report['trials']:
            if trial['truncated']:
                raise RuntimeError('战斗异常截断')
            wins += trial['winner'] == 0
            ties += trial['winner'] == 2
            dealt += trial['damage_dealt']
            taken += trial['damage_taken']
            n += 1
    return dict(score=(wins+ties*.5)/n, win=wins/n, tie=ties/n,
                damage_dealt=dealt/n, damage_taken=taken/n, trials=n, overlay_omissions=omissions)


def select_rows(scored, human, keep=4, target=.5):
    """完整帧选择：战力优先、身材辅助、保留不同组成，不造不存在的机制。"""
    buckets, selected = defaultdict(list), []
    for row in scored:
        buckets[(row['turn'], row['seat'])].append(row)
    for key, rows in sorted(buckets.items()):
        h = human.get(key[0])
        def cost(row):
            if not h or 'combat' not in row:
                return 0
            body = sum(m['atk']+m['hp'] for m in row['player']['board'])
            # 身材只作辅助，不覆盖不同机制造成的真实战斗差异。
            return abs(row['combat']['score']-target)+.05*abs(math.log((body+1)/(h['median_strength']+1)))
        ordered = sorted(rows, key=lambda r:(cost(r), r['source'], r['source_game_seed']))
        used, chosen = set(), []
        for row in ordered:
            sig = signature(row)
            if sig not in used:
                chosen.append(row)
                used.add(sig)
            if len(chosen) == keep:
                break
        selected.extend(chosen)
    return selected


def build(sources, human_path, out, split, *, seed=20261103, per_source=3, keep=4, trials=8):
    from hsbg.az.train.personal import write_json, library_digest, sha256, fingerprint
    from hsbg.az.train.mirror_snapshots import write_snapshot_file, engine_stamp, MirrorSnapshotPool
    from workflows.mirror_hard_gates import _drive_to_turn
    from workflows.compare_board_libraries import summarize
    import hsbg_engine as E
    out = Path(out)
    if out.exists():
        raise FileExistsError('禁止覆盖已有库；失败目录也保留')
    rows, manifests = read_sources(sources, split)
    humans = [json.loads(line) for line in Path(human_path).read_text(encoding='utf-8').splitlines() if line.strip()]
    refs = human_references(humans, seed)
    candidates = sample_candidates(rows, per_source, seed)
    state = json.loads(_drive_to_turn(E.Game(seed=seed), 1).state_json())
    for p in state['players']:
        p['hero'] = p['deity'] = None
        for key in ('powers', 'pending', 'trinkets', 'quests', 'rewards', 'secrets', 'effects'):
            p[key] = []
    scored = []
    for i, row in enumerate(candidates):
        turn = int(row['turn'])
        # t1..5保留自然帧；t16+参考覆盖不足，只报告，不拟合为精确真人压力。
        if 6 <= turn <= 15 and turn in refs and refs[turn]['games'] >= 32:
            state['turn'] = turn
            row['combat'] = combat_score(row, refs[turn]['anchors'], json.dumps(state), trials)
        scored.append(row)
        if i % 100 == 0:
            print(f'[calibrate] {split} {i}/{len(candidates)}', flush=True)
    selected = select_rows(scored, refs, keep)
    # 另两局真人参考不参与筛选，检查是否只适应了少数校准对手。
    for row in selected:
        turn = int(row['turn'])
        if 'combat' in row and refs[turn]['audit_anchors']:
            state['turn'] = turn
            row['audit_combat'] = combat_score(row, refs[turn]['audit_anchors'], json.dumps(state), trials)
    grouped = defaultdict(list)
    for row in selected:
        grouped[(row['source'], row['source_game_seed'])].append(row)
    out.mkdir(parents=True)
    for (source, game_seed), frames in sorted(grouped.items()):
        write_snapshot_file(str(out/f'snap_{source}_{game_seed:08d}.json.gz'), frames,
            seed=game_seed, engine=engine_stamp(), meta=dict(source=source, calibration='selection_only'))
    by_turn = {}
    for turn in sorted({r['turn'] for r in selected}):
        bucket = [r for r in selected if r['turn'] == turn]
        combat = [r['combat']['score'] for r in bucket if 'combat' in r]
        available = [r['combat']['score'] for r in scored if r['turn'] == turn and 'combat' in r]
        audit = [r['audit_combat']['score'] for r in bucket if 'audit_combat' in r]
        by_turn[str(turn)] = dict(frames=len(bucket), source_games=len({(r['source'],r['source_game_seed']) for r in bucket}),
            sources=dict(Counter(r['source'] for r in bucket)), unique_compositions=len({signature(r) for r in bucket}),
            selected_combat_score=sum(combat)/len(combat) if combat else None,
            audit_combat_score=sum(audit)/len(audit) if audit else None,
            available_score_range=[min(available), max(available)] if available else None,
            target_reachable=bool(available and min(available) <= .5 <= max(available)),
            human_games=refs.get(turn, {}).get('games', 0),
            reference_games={k:[r['game_id'] for r in refs.get(turn, {}).get(k, [])]
                             for k in ('anchors', 'audit_anchors')},
            mechanics=[mechanics(r['player']) for r in bucket])
    report = dict(schema='hsrl.calibrated_bank/v1', split=split, seed=seed,
        method='完整帧筛选；不缩放身材，不补造隐藏计数；目标条件战斗得分0.5',
        source_digests={k:v['digest'] for k,v in manifests.items()}, human_train_sha256=sha256(human_path),
        candidates=len(scored), selected=len(selected), turns=by_turn,
        source_statistics={name:summarize([dict(turn=r['turn'], game_id=str(r['source_game_seed']), board_detail=r['player']['board'])
            for r in rows if r['source']==name]) for name in sources},
        scored_frames=[dict(source=r['source'], game=r['source_game_seed'], turn=r['turn'], seat=r['seat'],
                            combat=r.get('combat'), audit_combat=r.get('audit_combat')) for r in scored],
        ready_for_long_training=False,
        limitations=['这是候选库，需完整八人硬门、模型压力评测及小规模PPO验证。',
            'Firestone精彩局存在选择偏差；参考仅Train中央50%身材，每回合2局筛选+2局独立审计，仍是有界初筛。',
            '条件战斗是近似标尺，不是真人胜率；probe不overlay含实体引用的owed/scheduled。',
            '完整原始Player快照及引用保持不变；训练注入保真度受原生事务支持字段约束。',
            't16+仅保留自然覆盖；没有复制低回合或制造晚期数据。'])
    write_json(out/'calibration.json', report)
    write_json(out/'manifest.json', dict(schema='hsrl.library/v1', split=split, driver='calibrated_models',
        fingerprint=fingerprint(), digest=library_digest(out), stats=MirrorSnapshotPool(str(out), sample='random').stats(),
        parents=report['source_digests'], calibration_sha256=sha256(out/'calibration.json'),
        missing='engine_bot_fallback', pool_mode='unknown'))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', required=True, help='JSON映射：来源名到已验证快照目录')
    p.add_argument('--human', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--split', choices=['train', 'dev'], required=True)
    p.add_argument('--per-source', type=int, default=3)
    p.add_argument('--trials', type=int, default=8)
    a = p.parse_args()
    if not 1 <= a.per_source <= 8 or not 1 <= a.trials <= 32:
        p.error('per-source 1..8, trials 1..32')
    build(json.loads(Path(a.sources).read_text(encoding='utf-8')), a.human, a.out, a.split,
          per_source=a.per_source, trials=a.trials)
