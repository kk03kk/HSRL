"""Evaluate a frozen late-policy handoff on paired natural games."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsbg.az.train.eval_policy import make_eval_policy
from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
from hsbg.az.train.personal import library_digest
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from tools.summarize_peer_finetune import compare, estimate
from workflows.peer_finetune_holdout import INITIAL, INITIAL_SHA, holdout_protocol, read
from workflows.peer_ppo_panel import trial

CANDIDATE = Path('artifacts/runs/nus-peer-d155-pilot-20261119-1/segment-02/fit/train/snapshots/final_000060/model.pt')
CANDIDATE_SHA = 'fc3c564128ebead4e7499867c7970316eef5d6a46c7efe6c91a8096402c557c7'
SEED = 306410000
GAMES = 256
WORKERS = 8


class Handoff:
    def __init__(self, controller, successor):
        self.controller = controller
        self.initial = controller.evaluator
        self.successor = successor

    def reset(self):
        self.controller.evaluator = self.initial
        self.controller.reset()
        self.turn = 0
        self.trigger = None
        self.prefix = sha256()

    def decide(self, game, seat):
        turn = int(game.turn)
        if turn != self.turn and 9 <= turn <= 12 and self.trigger is None:
            player = json.loads(game.player_json(seat))
            hp = player['health'] + player['armor']
            if hp <= 15:
                self.trigger = (turn, hp, self.prefix.hexdigest())
                self.controller.evaluator = self.successor
        self.turn = turn
        choice = self.controller.decide(game, seat)
        if self.trigger is None:
            self.prefix.update(bytes((turn, int(choice.action))))
        return choice


def field():
    proto, _ = holdout_protocol()
    path = Path(proto.mirror_snaps).parent / 'train'
    expected = read(ROOT / 'config/research/human-turn-3702-ppo.json')['library_digests']['train']
    if library_digest(path) != expected:
        raise ValueError('development library changed')
    return replace(proto, mirror_snaps=str(path.as_posix())), expected


def shard(out, index):
    import torch
    torch.set_num_threads(1)
    proto, digest = field()
    plan = read(out / 'plan.json')
    if (proto.sha256(), digest) != (plan['protocol'], plan['library']):
        raise ValueError('field mismatch')
    pool = MirrorSnapshotPool(proto.mirror_snaps, sample=proto.mirror_sample)
    if pool.frames != 40659:
        raise ValueError('development frame count')
    parent, candidate = [make_eval_policy(str(p), protocol=proto, device='cuda:0')['policy']
                         for p in (INITIAL, CANDIDATE)]
    control = Handoff(parent, parent.evaluator)
    switch = Handoff(parent, candidate.evaluator)
    rows, triggers = [], []
    for seed in range(SEED + index, SEED + GAMES, WORKERS):
        left = trial(control, pool, seed, event_turns=True)
        right = trial(switch, pool, seed, event_turns=True)
        if control.trigger != switch.trigger or (switch.trigger is None and
                (control.prefix.digest() != switch.prefix.digest() or left != right)):
            raise ValueError(f'prefix or untouched trajectory differs on {seed}')
        rows.append(dict(parent=left, candidate=right))
        triggers.append(dict(seed=seed, trigger=switch.trigger))
    atomic_json(out / f'panel-{index}.json', rows)
    atomic_json(out / f'triggers-{index}.json', triggers)


def run(out):
    source = source_identity(ROOT)
    if source['tracked_dirty'] or os.environ.get('HSAI_ACTION_PROFILE') != 'human-turn-3702-standard':
        raise ValueError('unexpected release or rules')
    parent, candidate = file_identity(INITIAL), file_identity(CANDIDATE)
    if parent['sha256'] != INITIAL_SHA or candidate['sha256'] != CANDIDATE_SHA:
        raise ValueError('weights changed')
    proto, digest = field()
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'plan.json', dict(source=source, seed=SEED, games=GAMES,
        parent=parent, candidate=candidate, protocol=proto.sha256(), field=proto.as_dict(),
        library=digest, engine_lock=file_identity(ROOT/'config/engine-upstream-lock.json'),
        live_pin=file_identity(ROOT/'config/live/model.json'), job=os.environ['SLURM_JOB_ID'],
        rule='first recruit opening during T9 to T12 with HP plus armor <=15, use candidate thereafter',
        development_gate='full-game mean utility >=0.10, lower CI >0, paired T12 lower CI >=-0.12'))

    def launch(index):
        with (out / f'worker-{index}.log').open('x') as log:
            subprocess.run([sys.executable, __file__, str(out), '--worker', str(index)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    with ThreadPoolExecutor(WORKERS) as executor:
        list(executor.map(launch, range(WORKERS)))
    result = compare(out, SEED, GAMES)
    selected = []
    for i in range(WORKERS):
        for row, mark in zip(read(out/f'panel-{i}.json'), read(out/f'triggers-{i}.json'), strict=True):
            if mark['seed'] != row['parent']['seed']:
                raise ValueError('unmatched trigger')
            if mark['trigger'] is not None:
                selected.append(row['candidate']['metrics']['utility'] - row['parent']['metrics']['utility'])
    utility = result['delta']['utility']
    result['triggered'] = estimate(selected)
    result['development_pass'] = bool(utility['mean'] >= .10 and utility['ci95'][0] > 0
                                      and result['delta']['log12']['ci95'][0] >= -.12)
    atomic_json(out/'complete.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    if args.worker is None:
        run(args.out.resolve())
    else:
        shard(args.out, args.worker)
