"""Paired native combats for a public end-of-turn decision on sealed development games."""
import gzip
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.engine_quality import require_clean_engine
from hsbg.runtime.artifacts import atomic_json, file_identity
from tools.compare_research_panels import bootstrap

SOURCE = Path('artifacts/runs/nus-residual-history-suffix-20261118-1')
SOURCE_SHA = 'af2e19b65514d9b0418753b52cfcd52221906e0f7aaf74201ee8fbb9f1c275ad'


def fight(state, rng):
    import hsbg_engine as engine
    game = engine.Game.from_json(json.dumps(dict(state, rng=rng)))
    if not game.needs_action(0) or not game.legal_mask(0)[0]:
        raise ValueError('expected legal EOT at the branch')
    require_clean_engine(game)
    game.step(0, 0)
    for seat in range(1, 8):
        if game.is_over() or game.turn != 12:
            break
        if game.needs_action(seat):
            game.bot_play(seat)
    require_clean_engine(game)
    end = json.loads(game.state_json())
    results = [c for c in end['last_combats'] if c['turn'] == 12 and 0 in (c['a'], c['b'])]
    if len(results) != 1:
        raise ValueError('missing unique native combat')
    combat = results[0]
    side = int(combat['a'] != 0)
    score = 0.5 if combat['winner'] == 2 else float(combat['winner'] == side)
    return score, end['players'][0]['health'], end['players'][0]['alive']


def probe(row):
    import hsbg_engine as engine
    path = SOURCE / 'trials' / row['name'] / f"case-{row['case']:03d}-complete_gold-r{row['replica']:02d}" / 'decisions.jsonl.gz'
    if file_identity(path) != {k: row['trace'][k] for k in ('sha256', 'bytes')}:
        raise ValueError('development trace identity differs')
    result = dict(source=row['source_seed'], case=row['case'], replica=row['replica'])
    trigger = end = None
    count = 0
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        if json.loads(next(stream))['identity'] != row['trace']['identity']:
            raise ValueError('development trace provenance differs')
        for line in stream:
            event = json.loads(line)
            if event['event'] != 'action' or event['turn'] < 12:
                continue
            if event['turn'] > 12:
                break
            if count != event['policy_info']['public_history_actions']:
                raise ValueError('remaining action budget differs from recorded history')
            state = event['state_before']
            if (trigger is None and 95 <= count < 100 and 16 <= event['action'] < 23
                    and len(state['players'][0]['board']) == 7 and event['legal_mask'][0]):
                trigger = state
                result.update(used=count, sale=event['action'], trigger_index=event['action_index'])
            if event['action'] == 0:
                end = state
                break
            count += 1
    if trigger is None:
        return dict(result, triggered=False)
    if end is None:
        raise ValueError('trigger followed by recruit death or incomplete trace: separate endpoint needed')
    paired = []
    for i in range(32):
        seed = 15100000 + row['case'] * 1000 + row['replica'] * 100 + i
        rng = json.loads(engine.Game(seed=seed).state_json())['rng']
        paired.append(dict(seed=seed, close=fight(trigger, rng), continue_cycle=fight(end, rng)))
    return dict(result, triggered=True, recorded_eot_count=count, paired=paired)


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    path = SOURCE / 'complete.json'
    if file_identity(path)['sha256'] != SOURCE_SHA:
        raise ValueError('unrecognized sealed suffix source')
    rows = [r for r in json.loads(path.read_text())['rows'] if r['variant'] == 'complete_gold']
    if len(rows) != 98 or len({(r['source_seed'], r['replica']) for r in rows}) != 98:
        raise ValueError('incomplete source cohort')
    with ProcessPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(probe, rows))
    estimates = {}
    for index, metric in enumerate(('combat_score', 'health_after', 'survival')):
        by_source = defaultdict(list)
        all_sources = defaultdict(list)
        for r in results:
            delta = 0
            if r['triggered']:
                delta = mean(p['close'][index] - p['continue_cycle'][index] for p in r['paired'])
                by_source[r['source']].append(delta)
            all_sources[r['source']].append(delta)
        estimates[metric] = {name: bootstrap([mean(v) for v in group.values()])
                             for name, group in [('triggered_sources', by_source), ('all_sources', all_sources)]}
    atomic_json(out / 'complete.json', dict(source=file_identity(path), results=results, estimates=estimates,
                triggers=sum(r['triggered'] for r in results), games=len(rows), future_samples=32))


if __name__ == '__main__':
    main(Path(os.environ['HSAI_CLOSE_OUT']))
