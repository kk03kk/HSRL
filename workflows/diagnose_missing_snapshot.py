"""Reproduce a sealed suffix exactly while locating snapshot misses.

Diagnostic only: deliberately preserves the old fallback behavior to locate it;
this run is never eligible as a new ability evaluation.
"""
import argparse,gzip,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
from hsbg.az.train.research_economy_policy import EconomicOptionPolicy
from hsbg.az.train.eval_policy import EvalProtocol,make_eval_policy
from hsbg.az.train.evaluate import EvalConfig,evaluate_policy
from hsbg.evaluation.trace import DecisionRecorder,validate_trace
from hsbg.runtime.artifacts import atomic_json,file_identity

class AuditPool(MirrorSnapshotPool):
    def __init__(self,path,out):
        super().__init__(path,sample='p50band');self.game_ref=None;self.misses=[];self.out=out
    def get(self,turn,seat,rng=None):
        frame=super().get(turn,seat,rng=rng)
        if frame is None:
            if self.game_ref is None:raise ValueError('no diagnostic game reference')
            state=json.loads(self.game_ref.state_json());p=state['players'][0]
            path=self.out/f'missing-{len(self.misses):03d}.json.gz'
            with gzip.open(path,'xt',encoding='utf-8') as stream:stream.write(json.dumps(state,separators=(',',':')))
            self.misses.append({'turn':turn,'seat':seat,'library_last_turn':max(self.turns()),
                'learner_alive':p['alive'],'learner_health':p['health'],'placements':list(self.game_ref.placements()),
                'state':file_identity(path)})
        return frame

class ExactRecorder(DecisionRecorder):
    def __init__(self,*args,pool,expected,**kwargs):
        super().__init__(*args,**kwargs);self.pool=pool;self.expected=iter(expected);self.checked=0
    def before(self,game,seat):
        self.pool.game_ref=game;return super().before(game,seat)
    def after(self,game,before,decision,executed_action,forced_tier=False):
        old=next(self.expected)
        if old['action']!=executed_action or old['state_before']!=before['state_before']:
            raise ValueError('original action or input state differs')
        if old['state_after']!=json.loads(game.state_json()):raise ValueError('original native transition differs')
        self.checked+=1;super().after(game,before,decision,executed_action,forced_tier)

def main(args):
    torch.set_num_threads(1);source=Path(args.source);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    original=json.loads((source/'result.json').read_text());summary=original['summary']
    root=json.loads((source.parents[2]/'complete.json').read_text())
    if summary not in root['rows']:raise ValueError('result not in sealed study')
    trace=source/'decisions.jsonl.gz'
    if file_identity(trace)!={k:summary['trace'][k] for k in ('bytes','sha256')}:raise ValueError('trace identity differs')
    initial=None;actions=[];final=None
    with gzip.open(trace,'rt',encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line)
            if r['event']=='game_start':initial=r['state']
            elif r['event']=='action':actions.append(r)
            elif r['event']=='game_end':final=r['state']
    library='artifacts/libraries/arena-cal-v14-snap';checkpoint='artifacts/models/arena-v1.4-r1/model.pt'
    if file_identity(checkpoint)['sha256']!=summary['trace']['identity']['checkpoint_sha256']:raise ValueError('parent changed')
    built=make_eval_policy(checkpoint,protocol=EvalProtocol(mirror_snaps=library),device='cpu')
    policy=EconomicOptionPolicy(built['policy'].evaluator,operator_mode='target_safe')
    pool=AuditPool(library,out)
    recorder=ExactRecorder(out/'decisions.jsonl.gz',identity={'diagnostic_only':True,
        'original_trace':file_identity(trace),'checkpoint_sha256':file_identity(checkpoint)['sha256']},pool=pool,expected=actions)
    try:
        result=evaluate_policy(policy,EvalConfig(games=1,seed=summary['future_seed'],research_initial_state=initial,
            collect_economy=True,decision_recorder=recorder,mirror_pool=pool,
            mirror_strict=False,mirror_seats=7,mirror_p=1.))
        new_trace=recorder.finish(result)
    except BaseException:
        recorder.abort();raise
    if recorder.checked!=len(actions) or result['rank_sequence']!=original['evaluation']['rank_sequence']:
        raise ValueError('original learner outcomes differ')
    if json.loads(pool.game_ref.state_json())!=final:raise ValueError('original final full-game state differs')
    validation=validate_trace(out/'decisions.jsonl.gz',replay=16)
    complete={'diagnostic_only':True,'ability_eligible':False,'original_rollout_exact':True,
        'original_result':file_identity(source/'result.json'),'checked_actions':recorder.checked,
        'misses':pool.misses,'trace':new_trace,'validation':validation}
    atomic_json(out/'complete.json',complete);print(json.dumps({k:v for k,v in complete.items() if k not in ('trace','validation')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True)
    p.add_argument('--out',required=True);main(p.parse_args())
