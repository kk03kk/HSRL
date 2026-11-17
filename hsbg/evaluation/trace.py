"""Lossless, streaming learner decision evidence. Recording never advances the game."""
from __future__ import annotations

import base64
from collections import Counter
import gzip
import json
import os
from pathlib import Path

from hsbg.runtime.artifacts import atomic_json, file_identity


def _json_value(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(f'unsupported trace value: {type(value).__name__}')


class DecisionRecorder:
    """One writer per model/shard, with a terminal manifest only on full completion.

    Learner actions have exact engine pre/post states and can be replayed with
    Game.from_json(). Opponents retain the existing native bot/snapshot behavior;
    their internal bot decisions are not exposed by this evidence schema.
    """

    def __init__(self, path, *, identity):
        self.path = Path(path)
        self.partial = Path(str(path)+'.partial')
        self.manifest_path = Path(str(path)+'.manifest.json')
        if any(p.exists() for p in (self.path, self.partial, self.manifest_path)):
            raise FileExistsError(f'trace output already exists: {path}')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = gzip.open(self.partial, 'xt', encoding='utf-8', compresslevel=1)
        self.identity = identity
        self.counts = Counter()
        self.game = None
        self.seed = None
        self.action_index = 0
        self.seeds = []
        self.write('header', schema='hsai.decision-trace/v1', identity=identity,
                   scope='every learner action; full engine state before and after native step')

    def write(self, event, **fields):
        self.stream.write(json.dumps({'event': event, **fields}, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False,
                                     default=_json_value)+'\n')
        self.counts[event] += 1

    def begin_game(self, game, game_index, seed, seat):
        self.game, self.seed, self.action_index = int(game_index), int(seed), 0
        self.seeds.append(self.seed)
        self.write('game_start', game=self.game, seed=self.seed, seat=seat,
                   state=json.loads(game.state_json()))

    def before(self, game, seat):
        features, cards = game.observe(seat)
        return dict(game=self.game, seed=self.seed, action_index=self.action_index,
                    turn=int(game.turn), seat=int(seat),
                    state_before=json.loads(game.state_json()),
                    observation={'floats_f32le': base64.b64encode(features).decode('ascii'),
                                 'cards_i32le': base64.b64encode(cards).decode('ascii')},
                    legal_mask=list(bytes(game.legal_mask(seat))))

    def after(self, game, before, decision, executed_action, forced_tier=False):
        self.write('action', **before, proposed_action=int(decision.action),
                   action=int(executed_action), policy_distribution=decision.pi,
                   policy_info=decision.info, forced_tier=bool(forced_tier),
                   state_after=json.loads(game.state_json()))
        self.action_index += 1

    def end_game(self, game, rank, quality):
        self.write('game_end', game=self.game, seed=self.seed, rank=int(rank),
                   actions=self.action_index, quality=quality, finished=bool(game.is_over()),
                   placements=list(game.placements()), state=json.loads(game.state_json()))

    def finish(self, result):
        if self.counts['game_end'] != result['games'] or self.counts['action'] != result['decisions']:
            raise RuntimeError('trace coverage does not match evaluation outcome')
        if self.seeds != result['seed_sequence'] or len(set(self.seeds)) != len(self.seeds):
            raise RuntimeError('trace seeds do not uniquely match evaluation outcome')
        self.write('footer', games=result['games'], actions=result['decisions'],
                   seeds=self.seeds, ranks=result['rank_sequence'])
        self.stream.close()
        # Keep failure evidence if another process has claimed the final name.
        os.link(self.partial, self.path)
        self.partial.unlink()
        manifest = {'schema': 'hsai.decision-trace-manifest/v1', 'complete': True,
                    'file': self.path.name, **file_identity(self.path),
                    'counts': dict(self.counts), 'identity': self.identity,
                    'games': result['games'], 'actions': result['decisions'],
                    'seeds': self.seeds}
        atomic_json(self.manifest_path, manifest)
        return manifest

    def abort(self):
        self.stream.close()


def validate_trace(path, *, replay=0):
    """Validate every record and optionally replay bounded native steps on CPU."""
    path = Path(path)
    manifest = json.loads(Path(str(path)+'.manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'hsai.decision-trace-manifest/v1' or manifest.get('complete') is not True:
        raise ValueError('trace manifest is not complete')
    if file_identity(path) != {k: manifest[k] for k in ('bytes', 'sha256')}:
        raise ValueError('trace file identity mismatch')
    counts = Counter()
    current = None
    index = 0
    seeds = []
    ranks = []
    replayed = 0
    ended = False
    game_index = None
    stride = max(1, int(manifest['actions']) // max(1, replay))
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            event = row['event']
            if ended or (not counts and event != 'header'):
                raise ValueError('trace header/footer ordering is invalid')
            counts[event] += 1
            if event == 'header':
                if counts['header'] != 1 or row.get('schema') != 'hsai.decision-trace/v1' or row['identity'] != manifest['identity']:
                    raise ValueError('trace identity differs from manifest')
            elif event == 'game_start':
                if current is not None: raise ValueError('game lacks terminal event')
                current, index = row['seed'], 0
                game_index = row['game']
                if game_index != len(seeds): raise ValueError('noncontiguous game sequence')
                seeds.append(current)
            elif event == 'action':
                if current is None or row['seed'] != current or row['game'] != game_index or row['action_index'] != index:
                    raise ValueError('noncontiguous action sequence')
                if len(row['legal_mask']) != 245 or not 0 <= row['action'] < 245 or not row['legal_mask'][row['action']]:
                    raise ValueError('illegal recorded action')
                obs = row['observation']
                if len(base64.b64decode(obs['floats_f32le'], validate=True)) != 3227*4 or len(base64.b64decode(obs['cards_i32le'], validate=True)) != 124*4:
                    raise ValueError('invalid observation dimensions')
                if replayed < replay and (counts['action']-1) % stride == 0:
                    import hsbg_engine as engine
                    game = engine.Game.from_json(json.dumps(row['state_before']))
                    features, cards = game.observe(row['seat'])
                    if bytes(features) != base64.b64decode(obs['floats_f32le']) or bytes(cards) != base64.b64decode(obs['cards_i32le']):
                        raise ValueError('recorded observation differs from engine state')
                    game.step(row['seat'], row['action'])
                    if json.loads(game.state_json()) != row['state_after']:
                        raise ValueError(f'native step replay diverged at {current}/{index}')
                    replayed += 1
                index += 1
            elif event == 'game_end':
                if current is None or row['seed'] != current or row['game'] != game_index or row['actions'] != index:
                    raise ValueError('terminal event action count mismatch')
                if type(row['rank']) is not int or not 1 <= row['rank'] <= 8 or not row['finished']:
                    raise ValueError('game did not finish with a valid rank')
                ranks.append(row['rank'])
                current = None
            elif event == 'footer':
                if current is not None or row['seeds'] != seeds or row['ranks'] != ranks or row['games'] != len(seeds) or row['actions'] != counts['action']:
                    raise ValueError('footer does not match game records')
                ended = True
            else:
                raise ValueError(f'unknown trace event: {event}')
    if current is not None or counts != Counter(manifest['counts']) or counts['footer'] != 1 or counts['header'] != 1:
        raise ValueError('incomplete trace')
    if seeds != manifest['seeds'] or len(set(seeds)) != len(seeds):
        raise ValueError('manifest seed coverage mismatch')
    if counts['game_end'] != manifest['games'] or counts['action'] != manifest['actions']:
        raise ValueError('manifest coverage mismatch')
    return {'valid': True, 'games': len(seeds), 'actions': counts['action'], 'replayed': replayed,
            'seeds': seeds, 'ranks': ranks, 'identity': manifest['identity']}
