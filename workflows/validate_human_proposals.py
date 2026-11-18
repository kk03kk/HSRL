"""Value human-proposed trades under the student's own continuation and independent futures."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.common import engine_info
from hsbg.az.train.engine_quality import require_clean_engine
from hsbg.az.train.eval_policy import EvalProtocol, make_eval_policy
from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from tools.assess_investment_timing import UTILITY
from tools.compare_research_panels import bootstrap
from workflows.human_trade_options import Cards,Trade,MODEL,FIELDS,CACHE as TRADE_CACHE
from workflows.probe_investment_timing import TimingPolicy
from workflows.replicate_investment_returns import Captured,GUARDS


FIT = Path('artifacts/runs/nus-human-trade-options-20261118-1/fit.json')
CACHE = {}


class CaptureProposal(Trade):
    def __call__(self,f,c,mask,seat):
        prev=sum(self.adjustments.values())
        recommendation,value = super().__call__(f,c,mask,seat)
        if sum(self.adjustments.values())==prev:
            return recommendation,value
        original,_=self.original(f,c,mask,seat)
        a,b=int(np.argmax(original)),int(np.argmax(recommendation))
        kind='sell' if 16<=a<23 else 'buy_minion'
        if kind!=self.target or self.policy._turn_counts.get((seat,self.current),0)>=90:
            return original,value
        p=self.player
        total=sum(m['atk']+m['hp'] for m in p['board'])
        if kind=='sell':
            ma,mb=p['board'][a-16],p['board'][b-16]
            strength_delta=(mb['atk']+mb['hp']-ma['atk']-ma['hp'])/max(1,total)
            ka,kb=self.cid(ma),self.cid(mb)
        else:
            strength_delta=0.
            ka,kb=self.cid(p['shop'][a-8]),self.cid(p['shop'][b-8])
        board=[self.cid(q) for q in p['board']]
        hand=[self.cid(q) for q in p['hand']]
        diff=np.array(self.cards.describe(kb,board,hand,self.played,kind=='sell',self.current))-np.array(
            self.cards.describe(ka,board,hand,self.played,kind=='sell',self.current))
        require_clean_engine(self.game)
        self.initial=json.loads(self.game.state_json())
        self.guards={k:deepcopy(getattr(self.policy,k)) for k in GUARDS}
        self.event=dict(actions=[a,b],turn=self.current,kind=kind, features=[
            self.current,p['health']+p['armor'],p['gold'],len(hand),np.log1p(total),strength_delta,
            np.log(max(original[a],1e-30))-np.log(max(original[b],1e-30)),
            diff@self.weights[kind]['coef'],diff[3],diff[5],diff[8],int(kind=='sell')])
        raise Captured()


class FirstTrade(TimingPolicy):
    def __init__(self,source,side):
        super().__init__(source.policy,'parent',False)
        self.source,self.side=source,side

    def reset(self):
        super().reset()
        for k,v in self.source.guards.items():
            setattr(self.policy,k,deepcopy(v))

    def __call__(self,f,c,mask,seat):
        prior,value=self.original(f,c,mask,seat)
        if self.actions:
            return prior,value
        a,b=self.source.event['actions']
        if np.argmax(prior)!=a or not (mask[a] and mask[b]):
            raise ValueError('source policy or legal options differ')
        if self.side:
            prior=np.zeros_like(prior)
            prior[b]=1
        return prior,value


def task(args):
    import hsbg_engine as engine
    import torch
    torch.set_num_threads(1)
    idx,out=args
    group='sell' if idx<12 else 'buy_minion'
    field=FIELDS['development']
    if 'native' not in CACHE:
        policy=make_eval_policy(MODEL,protocol=EvalProtocol(mirror_snaps=field),device='cpu')['policy']
        CACHE['native']=policy,MirrorSnapshotPool(field,sample='p50band'),Cards(),json.loads(FIT.read_text())
        TRADE_CACHE['names']={}
    model,pool,cards,weights=CACHE['native']
    for j in range(256):
        seed=79700000+idx*1000+j
        p=CaptureProposal(model,cards,weights,True)
        p.target=group
        try:
            evaluate_policy(p,EvalConfig(games=1,seed=seed,mirror_pool=pool))
        except Captured:
            break
    else:
        raise ValueError('no predeclared source found')
    with gzip.open(out/f'{idx}-source.gz','xt',encoding='utf8') as f:
        json.dump(dict(seed=seed,state=p.initial,prefix=p.actions,event=p.event),f,default=float)
    outcomes=[]
    for rep in range(16):
        future=79800000+idx*100+rep
        fresh=json.loads(engine.Game(seed=future).state_json())
        state=dict(p.initial,seed=fresh['seed'],rng=fresh['rng'])
        pair=[]
        for side in (0,1):
            branch=FirstTrade(p,side)
            r=evaluate_policy(branch,EvalConfig(games=1,seed=future,mirror_pool=pool,research_initial_state=state))
            if branch.actions[0]!=p.event['actions'][side] or any(r[k] for k in (
                    'illegal_actions','truncated_games','truncated_combats')) or (
                    r['mirror']['mirror_errors'] or r['mirror']['mirror_rejected']):
                raise ValueError('invalid causal comparison')
            pair.append(dict(rank=r['rank_sequence'][0],eot=branch.eot,actions=branch.actions))
        outcomes.append(pair)
    event=dict(p.event,features=list(map(float,p.event['features'])))
    row=dict(source=seed,idx=idx,event=event,futures=outcomes)
    atomic_json(out/f'{idx}-result.json',row)
    return row


def assess(rows):
    x=np.array([r['event']['features'] for r in rows])
    ds=np.array([[UTILITY[h['rank']-1]-UTILITY[p['rank']-1] for p,h in r['futures']] for r in rows])
    early,late=ds[:,:8].mean(1),ds[:,8:].mean(1)
    predictions=[]
    for i in range(len(x)):
        use=np.arange(len(x))!=i
        center,scale=x[use].mean(0),x[use].std(0)
        scale[scale<1e-6]=1
        z=(x[use]-center)/scale
        intercept=early[use].mean()
        w=np.linalg.solve(z.T@z+30*np.eye(z.shape[1]),z.T@(early[use]-intercept))
        predictions.append(intercept+((x[i]-center)/scale)@w)
    rec=np.array(predictions)>0
    report={}
    for group in ('all','sell','buy_minion'):
        keep=np.array([group=='all' or r['event']['kind']==group for r in rows])
        e,l=early[keep],late[keep]
        selected=l*rec[keep]
        report[group]=dict(sources=len(l),switched=int(rec[keep].sum()),first8=bootstrap(e),last8=bootstrap(l),
            both_nonzero=int(((e!=0)&(l!=0)).sum()),sign_agreement=int((e*l>0).sum()),
            stored_choice_minus_parent=bootstrap(l*(e>0)),gated_minus_parent=bootstrap(selected),
            gated_minus_always_human=bootstrap(selected-l))
    return report


if __name__=='__main__':
    out=Path(sys.argv[1])
    out.mkdir(parents=True,exist_ok=False)
    inputs=json.loads(Path('config/research/natural12m-20261118.json').read_text())
    src=source_identity(Path.cwd())
    if (src['tracked_dirty'] or file_identity(FIT)['sha256']!=sys.argv[2] or
            engine_info()['source_hash']!=inputs['engine_fingerprint'] or
            any(file_identity(k)['sha256']!=v for k,v in inputs['asset_sha256'].items())):
        raise ValueError('unrecognized input or source')
    atomic_json(out/'run.json',dict(source=src,engine=engine_info(),inputs=inputs['asset_sha256'],
        fit=file_identity(FIT),design='12 new sources per trade kind, pre-outcome first eligible in disjoint blocks. '
        'Only the trade changes. Eight futures for fitting, eight for testing. Leave out the full test source in '
        'ridge fits, penalty 30. Advance only if the source-gated policy beats the parent in both kinds, joint '
        'lower 95% above zero and joint mean above always-human. Report balance and natural exposure separately.'))
    with ProcessPoolExecutor(16) as pool:
        rows=list(pool.map(task,[(i,out) for i in range(24)]))
    groups=assess(rows)
    advance=(all(groups[k]['gated_minus_parent']['mean']>0 for k in ('sell','buy_minion')) and
             groups['all']['gated_minus_parent']['ci95'][0]>0 and groups['all']['gated_minus_always_human']['mean']>0)
    atomic_json(out/'complete.json',dict(sources=24,paired_futures=384,suffixes=768,advance=advance,
                                         groups=groups,run=file_identity(out/'run.json')))
