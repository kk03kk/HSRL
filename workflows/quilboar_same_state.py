"""Diagnostic replay: hand the student's naturally assembled board to either policy."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from hsbg.az.train.evaluate import EvalConfig,evaluate_policy
from hsbg.runtime.artifacts import atomic_json,file_identity
from workflows.d142_crossover import Fork,PrefixComplete,ResumedPolicy,digest
from workflows.quilboar_study import OUT,BASE,PLAN,config,policy,Recorder,read
from tools.assess_quilboar_study import RID,SOURCE,interval,UTILITY
import numpy as np
import json

DIAG=Path(os.environ.get('HSAI_QUILL_DIAG',str(OUT)+'-forks'))


class ForkRecorder(Recorder):
    def __init__(self,expected=None):
        self.expected=expected

    def begin_game(self,game,index,seed,seat):
        super().begin_game(game,index,seed,seat)
        if self.expected and digest(json.loads(game.state_json()))!=self.expected:
            raise ValueError('different native opening')


def opening(g):
    # Select the first eligible turn with an actual first action, irrespective
    # of the subsequent rank, growth or success of any activation.
    for t in range(9,13):
        prev=g['eot'].get(str(t-1))
        if prev:
            b={c.removesuffix('_G') for c in prev[1]}
            if RID in b and SOURCE & b:
                return t


def worker(i):
    torch.set_num_threads(1)
    proto,pool=config('hold')
    student=policy(read(OUT/'selection.json')['path'],proto)
    parent=policy(BASE,proto)
    rows=[]
    for stage in ('natural','special'):
        for source in read(OUT/stage/f'{i}.json'):
            g=source['student']
            t=opening(g)
            if t is None:
                continue
            seed=g['seed']
            f=Fork(stop=True)
            f.turn=t
            cfg=EvalConfig(games=1,seed=seed,mirror_strict=True,mirror_pool=pool,research_capture_strict=f)
            try:
                evaluate_policy(student,cfg)
            except PrefixComplete:
                pass
            if f.state is None:
                rows.append(dict(stage=stage,seed=seed,turn=t,unreachable=True))
                continue
            trial={}
            for tag,p in [('parent',parent),('student',student)]:
                rec=ForkRecorder(f.opening)
                suffix=EvalConfig(games=1,seed=seed,mirror_strict=True,mirror_pool=pool,collect_economy=True,
                    research_initial_state=f.state,research_strict_context=f.strict,decision_recorder=rec)
                r=evaluate_policy(ResumedPolicy(f,p.evaluator),suffix)
                if r['games_finished']!=1 or r['illegal_actions'] or r['truncated_games'] or r['truncated_combats'] or r['mirror']['mirror_fallback']:
                    raise ValueError('invalid continuation')
                strength={str(c['turn']):c['eot_attack']+c['eot_board_hp'] for c in r['end_turn_economy']}
                trial[tag]=dict(rank=r['rank_sequence'][0],strength=strength,events=rec.events)
            if trial['student']!=dict(rank=g['rank'],strength={k:v[0] for k,v in g['eot'].items() if int(k)>=t},
                                       events=[e for e in g['events'] if e[0]>=t]):
                raise ValueError('same policy failed to reproduce natural suffix')
            rows.append(dict(stage=stage,seed=seed,turn=t,opening=f.opening,results=trial))
    atomic_json(DIAG/f'{i}.json',rows)


def run():
    sealed=read(OUT/'complete.json')
    if not sealed['valid']:
        raise ValueError('invalid full evaluation')
    for stage in ('natural','special'):
        done=OUT/stage/'complete.json'
        if file_identity(done)!=sealed['panels'][stage]:
            raise ValueError('invalid source panel')
        for i,f in enumerate(read(done)['source_shards']):
            if file_identity(OUT/stage/f'{i}.json')!=f:
                raise ValueError('source games changed')
    DIAG.mkdir(parents=True,exist_ok=False)
    def launch(i):
        with (DIAG/f'worker-{i}.log').open('x') as log:
            subprocess.run([sys.executable,__file__,str(i)],stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(PLAN['workers']) as executor:
        list(executor.map(launch,range(PLAN['workers'])))
    rows=[r for i in range(PLAN['workers']) for r in read(DIAG/f'{i}.json')]
    eligible=[r for r in rows if 'results' in r]
    diffs={}
    for key in ('rank','utility','repeat','events','reach12','reach14','log12','log14'):
        vals=[]
        for r in eligible:
            cell=[]
            for tag in ('parent','student'):
                s=r['results'][tag]
                value={'rank':s['rank'],'utility':float(UTILITY[s['rank']-1]),
                       'repeat':int(len({e[0] for e in s['events']})>=2),'events':len(s['events']),
                       'reach12':int('12'in s['strength']),'reach14':int('14'in s['strength'])}
                for turn in (12,14):
                    if str(turn) in s['strength']:
                        value['log'+str(turn)]=float(np.log1p(s['strength'][str(turn)]))
                cell.append(value.get(key))
            if None not in cell:
                vals.append(cell)
        diffs[key]=dict(delta=interval([b-a for a,b in vals]),
                        means=np.mean(vals,axis=0).tolist() if vals else [])
    atomic_json(DIAG/'complete.json',dict(valid=True,n=len(eligible),unreachable=len(rows)-len(eligible),
                by_panel={s:sum(r['stage']==s for r in eligible) for s in ('natural','special')},
                comparisons=diffs,source=file_identity(OUT/'complete.json')))


if __name__=='__main__':
    run() if len(sys.argv)==1 else worker(int(sys.argv[1]))
