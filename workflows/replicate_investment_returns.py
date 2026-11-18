"""Measure whether independent future samples reproduce an investment choice at the same state."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.engine_quality import require_clean_engine
from hsbg.az.train.eval_policy import EvalProtocol, make_eval_policy
from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from tools.assess_investment_timing import UTILITY
from tools.compare_research_panels import bootstrap
from workflows.probe_investment_timing import TimingPolicy, MODEL, FIELDS


class Captured(Exception):
    pass


GUARDS = ('_guard','_pending_guard','_turn_counts','_freeze_counts')
STRATA = ('milestone','optional_six','incumbent')
CACHE = {}


class Capture(TimingPolicy):
    def record(self, kind, index, prior):
        super().record(kind,index,prior)
        if kind == self.target:
            require_clean_engine(self.game)
            self.initial = json.loads(self.game.state_json())
            self.guards = {k:deepcopy(getattr(self.policy,k)) for k in GUARDS}
            raise Captured()


class Continue(TimingPolicy):
    def __init__(self, source, arm):
        super().__init__(source.policy,arm,source.late)
        self.source = source

    def reset(self):
        super().reset()
        source = self.source
        for key,value in source.guards.items():
            setattr(self.policy,key,deepcopy(value))
        self.events = {source.target:dict(source.events[source.target],index=0)}
        self.opening = set(source.opening)
        self.turn = source.current

    def record(self, kind, index, prior):
        # Only the originally harvested choice is eligible.
        pass

    def __call__(self, features, cards, mask, seat):
        if not self.actions:
            prior,_ = self.original(features,cards,mask,seat)
            if int(np.argmax(prior)) != self.source.events[self.source.target]['baseline']:
                raise ValueError('original public policy choice differs after controller restoration')
        return super().__call__(features,cards,mask,seat)


def case(task):
    import hsbg_engine as engine
    import torch
    torch.set_num_threads(1)
    group, i, out = task
    field = FIELDS['development']
    if 'policy' not in CACHE:
        CACHE['policy'] = make_eval_policy(MODEL,protocol=EvalProtocol(mirror_snaps=field),device='cpu')['policy']
        CACHE['pool'] = MirrorSnapshotPool(field,sample='p50band')
    source = Capture(CACHE['policy'],'parent',group=='optional_six')
    source.target = 'sell' if group=='incumbent' else 'tier'
    index = STRATA.index(group)*12+i
    for attempt in range(64):
        seed = 77310000 + index*100 + attempt
        try:
            evaluate_policy(source,EvalConfig(games=1,seed=seed,mirror_pool=CACHE['pool']))
        except Captured:
            break
    else:
        raise RuntimeError('source selection exhausted')
    event = source.events[source.target]
    statefile = out/f'{index:02d}-state.json.gz'
    with gzip.open(statefile,'xt',encoding='utf8') as f:
        json.dump(dict(source=seed,state=source.initial,prefix=source.actions,opening=sorted(source.opening),
                       event=event),f,separators=(',',':'))
    initial = source.initial
    arms = ('retain','parent') if group=='incumbent' else ('upgrade','delay')
    futures = []
    for rep in range(8):
        future = 78310000+index*100+rep
        fresh = json.loads(engine.Game(seed=future).state_json())
        state = dict(initial, seed=fresh['seed'], rng=fresh['rng'])
        results = []
        for arm in arms:
            policy = Continue(source,arm)
            r = evaluate_policy(policy,EvalConfig(games=1,seed=future,research_initial_state=state,
                                                  mirror_pool=CACHE['pool']))
            if (r['illegal_actions'] or r['truncated_games'] or r['truncated_combats'] or
                    r['mirror']['mirror_errors'] or r['mirror']['mirror_rejected']):
                raise ValueError('bad paired suffix')
            first = policy.actions[0]
            if (arm=='parent' and first!=event['baseline']) or (arm=='upgrade' and first!=3) or (
                    arm=='delay' and first==3) or (arm=='retain' and first==event['baseline']):
                raise ValueError('source action or intervention did not reproduce')
            results.append(dict(arm=arm,rank=r['rank_sequence'][0],actions=policy.actions,eot=policy.eot))
        futures.append(dict(seed=future,results=results))
    row = dict(group=group, index=i, source=seed, event=event, snapshot=file_identity(statefile),futures=futures)
    atomic_json(out/f'{index:02d}-returns.json',row)
    return row


def summarize(rows):
    report = {}
    for group in STRATA:
        group_rows = [r for r in rows if r['group']==group]
        diffs = np.array([[UTILITY[x['results'][0]['rank']-1]-UTILITY[x['results'][1]['rank']-1]
                           for x in r['futures']] for r in group_rows])
        early,late = diffs[:,:4].mean(1),diffs[:,4:].mean(1)
        # Tie falls back to the observed parent choice.
        defaults = np.array([group!='incumbent' and r['event']['baseline']==3 for r in group_rows])
        picked = np.where(early==0,defaults,early>0)
        per_source = np.where(picked,late,0.)
        report[group] = dict(sources=len(early),invest_selected=int(picked.sum()),
            first_four_mean=bootstrap(early),last_four_mean=bootstrap(late),
            nonzero_both=int(((early!=0)&(late!=0)).sum()),
            signs_agree=int(((early*late)>0).sum()),
            selector_minus_invest=bootstrap(per_source-late),
            selector_minus_alternative=bootstrap(per_source),
            selector_minus_parent_choice=bootstrap(per_source-np.where(defaults,late,0.)))
    return report


if __name__=='__main__':
    from hsbg.az.train.common import engine_info
    out = Path(sys.argv[1])
    out.mkdir(parents=True,exist_ok=False)
    plan = json.loads(Path('config/research/natural12m-20261118.json').read_text())
    paths = [MODEL, str(Path(MODEL).with_name('manifest.json'))]+[
        str(Path(FIELDS['development'])/p) for p in ('manifest.json','snap_cal.json.gz')]
    assets = {p:file_identity(p)['sha256'] for p in paths}
    if any(assets[p]!=plan['asset_sha256'][p] for p in paths) or engine_info()['source_hash']!=plan['engine_fingerprint']:
        raise ValueError('wrong assets or engine')
    src = source_identity(Path.cwd())
    if src['tracked_dirty']:
        raise ValueError('source changed')
    atomic_json(out/'run.json',dict(assets=assets,source=src,engine=engine_info(),sources_per_stratum=12,
              replicas=8,selection='first pre-outcome eligible game in disjoint source blocks',
              protocol='v14 non-strict parent controller, preserved current opponents, research suffix only'))
    with ProcessPoolExecutor(12) as pool:
        rows = list(pool.map(case,[(g,i,out) for g in STRATA for i in range(12)]))
    if len(set(r['source'] for r in rows))!=36:
        raise ValueError('repeated source')
    atomic_json(out/'complete.json',dict(sources=36,paired_futures=288,suffixes=576,
                                         run=file_identity(out/'run.json'),groups=summarize(rows)))
