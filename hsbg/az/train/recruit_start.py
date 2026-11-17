"""Verified whole-state curriculum starts; never inject into an existing trajectory.

The state is selected before the episode's first learner record. Only the RNG and
episode seed are refreshed. No source-state future is exposed to the policy.
Disabled mode returns the fresh game verbatim without touching the filesystem.
"""
from functools import lru_cache
import gzip
import json
from pathlib import Path
import random

from hsbg.runtime.artifacts import file_identity


def validate_options(bank,probability,*,lockstep=True,rust_collector=False,learner_seat=0):
    probability=float(probability)
    if not 0 <= probability <= 1:raise ValueError('recruit-start-p must be finite and in [0,1]')
    if bool(bank)!=(probability>0):raise ValueError('recruit start requires both bank and positive probability')
    if probability and (not lockstep or rust_collector or learner_seat!=0):
        raise ValueError('recruit starts currently require Python lockstep and learner seat zero')


def validate_state(state):
    p=state['players'][0]
    if not 7<=state['turn']<=10 or p['health']<=0 or p['pending'] or p['ended_turn']:
        raise ValueError('invalid actionable curriculum start')


@lru_cache(maxsize=2)
def load_bank(path):
    from .common import engine_info
    root=Path(path)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest['schema']!='hsai.recruit-start-bank/v1':raise ValueError('unsupported start bank')
    current=engine_info()
    if manifest['engine']!=current:raise ValueError('start-bank engine identity differs')
    if file_identity(root/'states.jsonl.gz')!=manifest['data']:raise ValueError('start-bank checksum differs')
    records=[]
    with gzip.open(root/'states.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line)
            if r['source_seed']%5==0 or r['split']!='train':raise ValueError('development source leaked into start bank')
            validate_state(r['state'])
            records.append(r)
    if not records or len(records)!=manifest['states']:raise ValueError('start-bank count differs')
    return manifest,records,file_identity(root/'manifest.json')


def choose_start(fresh_game,*,bank,probability,seed,engine,expected_sha256=''):
    if not probability:return fresh_game,None
    manifest,records,identity=load_bank(str(bank))
    if identity['sha256']!=expected_sha256:raise ValueError('start-bank identity changed after preflight')
    rng=random.Random(int(seed)^0x48534149)
    if rng.random()>=probability:return fresh_game,None
    index=rng.randrange(len(records));record=records[index]
    # Clone; the cached source and existing recovery evidence remain untouched.
    state=json.loads(json.dumps(record['state']))
    fresh=json.loads(fresh_game.state_json())
    state['rng']=fresh['rng'];state['seed']=fresh['seed']
    restored=engine.Game.from_json(json.dumps(state))
    if not restored.needs_action(0) or restored.is_over():raise ValueError('restored start cannot act')
    if any(restored.diagnostic_counts()):raise ValueError('unclean restored state')
    info={'bank_manifest':identity,'variant':manifest['variant'],'index':index,
          'source_seed':record['source_seed'],'initial_turn':state['turn'],'episode_seed':int(seed)}
    return restored,info
