"""Generate full native bot trajectories, split by source lobby before seat extraction."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import time
import taverngym as tg
from taverngym.rl.config import Config
from taverngym.rl.collect import observation
from taverngym.rl.opponents import profile_action


def generate(config, out, *, games=32, seed=2700000, split='train', batch_size=16, game_seeds=None, allow_empty=False):
    cfg = Config.load(config)
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    if game_seeds is not None:
        games = len(game_seeds)
        if len(set(game_seeds)) != games:
            raise ValueError('source seeds must be unique')
    if split not in ('train', 'validation', 'test') or games < 1:
        raise ValueError('explicit split and positive game count required')
    out.parent.mkdir(parents=True, exist_ok=True)
    traces, rejected, contexts = [], [], Counter()
    rules = {**cfg.rules, 'capture_library': True}
    canonical_rules = None
    card_table = None
    started = time.monotonic()
    for offset in range(0, games, batch_size):
        n = min(batch_size, games-offset)
        env = tg.BatchEnv(n, seed=seed+offset*1009, controlled=range(8), config=rules, threads=cfg.threads)
        env.auto_reset = False
        if game_seeds is not None:
            for g in range(n):
                slot = json.loads(env.save(g))
                slot['game'] = json.loads(tg.Game(game_seeds[offset+g], json.dumps(rules)).state_json())
                env.load(g, json.dumps(slot))
        frames = [{} for _ in range(n)]
        last = [0]*n
        finished = set()
        for tick in range(cfg.max_ticks):
            b = env.observe()
            extras = env.training_extras()
            actions = env.bot_actions()
            for i in b['ready'].nonzero()[0]:
                actions[i] = profile_action(observation(b, extras, int(i)), int(actions[i]), cfg.bot_profile, int(i))
            _, done, info = env.step(actions)
            if info['rejected'].any():
                raise RuntimeError('bot source generation rejected an action')
            after = env.observe()
            for g in range(n):
                turn = round(float(after['features'][g*8, 0])*20)
                if turn != last[g] or done[g*8:g*8+8].any():
                    capture = env.captured_library(g)
                    if capture:
                        frames[g][capture['turn']] = capture
                        card_table = capture['card_table']
                    last[g] = turn
                if all(done[g*8:g*8+8] | ~after['ready'][g*8:g*8+8]) and not after['ready'][g*8:g*8+8].any():
                    finished.add(g)
            if len(finished) == n:
                break
        else:
            raise RuntimeError('source generator tick guard')
        for g in range(n):
            diagnostic = env.diagnostics()[g]
            if any(diagnostic):
                rejected.append({'source_index': offset+g, 'diagnostic': diagnostic,
                                 'source_seed':game_seeds[offset+g] if game_seeds is not None else None})
                continue
            captures = frames[g]
            if not captures or 1 not in captures:
                raise RuntimeError('missing source opening frame')
            first = captures[1]
            context = (first['lobby'], first['deity'], first['paradox'])
            contexts[str(context)] += 1
            source = (f'bot-source-seed-{game_seeds[offset+g]}' if game_seeds is not None
                      else f'bot-seed-{seed}-lobby-{offset+g}')
            for seat in range(8):
                turns = {}
                for t, capture in sorted(captures.items()):
                    if not capture['players'][seat]['alive']: continue
                    frame = copy.deepcopy(capture['players'][seat])
                    # Source scouting/shop histories are not combat templates and are never injected.
                    for field in ('scouted','seen_board','last_board','shop','turn_events','move_orders'):
                        frame[field] = []
                    frame['mishmash_seen'] = None
                    turns[str(t)] = frame
                traces.append({'id': f'{source}-seat-{seat}', 'source_group': source,
                               'lobby': context[0], 'deity': context[1], 'paradox': context[2], 'frames': turns})
            if canonical_rules is None:
                canonical_rules = json.loads(env.save(g))['game']['cfg']
                canonical_rules['capture_library'] = False
        print(json.dumps({'phase': 'bank_generation', 'split': split, 'source_games_processed': offset+n,
                          'traces': len(traces), 'contexts': len(contexts), 'rejected': len(rejected)}), flush=True)
    if not traces and not allow_empty:
        raise RuntimeError('no valid source lobbies')
    bank = {'format': 'taverngym.library.v1', 'split': split, 'rules': canonical_rules,
            'card_table': card_table, 'native_source': tg.build_info()['source_hash'], 'traces': traces,
            'provenance': 'native scripted trajectories; not human-calibrated',
            'pool_policy': 'external-copies-exempt', 'tail_policy': 'hold-final-observed-frame'}
    raw = json.dumps(bank, separators=(',', ':')).encode()
    out.write_bytes(raw)
    report = {'path': str(out), 'sha256': hashlib.sha256(raw).hexdigest(), 'split': split,
              'source_games': games-len(rejected), 'requested_games': games, 'seat_traces': len(traces),
              'contexts': dict(contexts), 'rejected': rejected, 'seconds': time.monotonic()-started,
              'frames': sum(len(t['frames']) for t in traces), 'human_calibrated': False}
    out.with_suffix('.report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
