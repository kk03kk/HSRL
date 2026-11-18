"""Teacher-free paired policy search on closed-loop natural recruit decisions."""
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.economic_tasks import item_card
from hsbg.az.train.evaluate import EvalConfig,evaluate_policy
from hsbg.runtime.artifacts import atomic_json,file_identity
from tools.assess_investment_timing import UTILITY
from tools.compare_research_panels import bootstrap
from workflows.d132_paired_plans import check
from workflows.d132_trade_plan import provenance
from workflows.plan_recruit_openings import native
from workflows.probe_investment_timing import TimingPolicy

ROOT=Path('artifacts/runs/nus-paired-recruit-20261118-1')
SEEDS=(134171,134189)
WIDTH=24


class Residual(TimingPolicy):
    def __init__(self,field,weights):
        policy,self.pool,self.info=native(field)
        super().__init__(policy,'parent',False)
        self.w=np.asarray(weights).reshape(-1)
        if self.w.shape!=(WIDTH,) or not np.isfinite(self.w).all():
            raise ValueError('invalid residual')
        self.enabled=bool(self.w.any())

    def reset(self):
        super().reset()
        self.changed=[]

    def __call__(self,f,c,mask,seat):
        base,value=self.original(f,c,mask,seat)
        if not self.enabled or not mask[0] or self.player['pending']:
            return base,value
        p=self.player
        w=self.w
        hp=p['health']+p['armor']
        risk=np.clip((22-hp)/18,-1,1)
        gold=min(p['gold'],18)/9-1
        late=(self.current-9.5)/2.5
        board=p['board']
        crowded=float(len(board)==7)
        hand=p['hand']
        if not board:
            return base,value
        adj=np.zeros(len(base))
        adj[0]=w[0]+w[1]*gold
        adj[1]=w[2]+w[3]*risk+w[4]*gold
        adj[3]=w[5]+w[6]*risk+w[7]*late+w[8]*min(1,max(0,p['gold']-p['upgrade_cost'])/6)
        adj[4]=w[9]
        total=max(1,sum(m['atk']+m['hp'] for m in board))
        board_info=[self.info[m['card']] for m in board]
        owned=[m['card'] for m in board]+[item_card(q) for q in hand]
        for i,slot in enumerate(p['shop'][:8]):
            if not mask[8+i]:
                continue
            cid=item_card(slot)
            card=self.info[cid]
            races=set(card['races'])-{'ALL'}
            support=sum(bool(races.intersection(m['races'])) for m in board_info)/4
            adj[8+i]=w[10]+w[11]*risk+w[12]*min(support,1)+w[13]*min(2,owned.count(cid))/2
            if card['kind']!='minion':
                adj[8+i]+=w[14]
        for i,m in enumerate(board):
            adj[16+i]=w[15]+w[16]*crowded+w[17]*(m['atk']+m['hp'])/total+w[18]*float(m['eid'] not in self.opening)
        adj[46:126]=w[19]+w[20]*min(len(hand),8)/8
        adj[126:196]=w[21]+w[22]*late
        adj[2]=w[23]
        # Change preferences only among native actions allowed by the controller.
        updated=base*np.exp(np.clip(adj,-3,3))
        chosen=int(updated.argmax())
        parent=int(base.argmax())
        if chosen!=parent:
            self.changed.append([self.current,parent,chosen])
        return updated,value


def run_game(args):
    field,seed,weights=args
    w=Residual(field,weights)
    r=evaluate_policy(w,EvalConfig(games=1,seed=seed,mirror_pool=w.pool))
    check(r)
    return dict(seed=seed,rank=r['rank_sequence'][0],eot=w.eot,changes=len(w.changed),steps=len(w.actions))


def train(arm):
    out=ROOT/f'train-{arm}'
    out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(SEEDS[arm])
    w=np.zeros(WIDTH)
    m,v=np.zeros(WIDTH),np.zeros(WIDTH)
    atomic_json(out/'run.json',dict(**provenance(),seed=SEEDS[arm],plan=file_identity(Path('config/research/paired-recruit-20261118.json'))))
    with ProcessPoolExecutor(int(os.environ.get('SLURM_CPUS_PER_TASK',4))) as pool:
        for step in range(8):
            directions=np.linalg.qr(rng.normal(size=(WIDTH,12)))[0].T*np.sqrt(WIDTH)
            pairs=[]
            for k,d in enumerate(directions):
                for j in range(8):
                    seed=83100000+arm*20000+step*128+8*k+j
                    for sign in (1,-1):
                        pairs.append(('development',seed,w+sign*.6*d))
            outcomes=list(pool.map(run_game,pairs))
            u=np.array([UTILITY[g['rank']-1] for g in outcomes]).reshape(12,8,2)
            advantage=(u[:,:,0]-u[:,:,1]).mean(1)
            gradient=(advantage@directions)/(12*1.2)
            m=.9*m+.1*gradient
            v=.99*v+.01*gradient**2
            w=np.clip(.99*w+.18*(m/(1-.9**(step+1)))/(np.sqrt(v/(1-.99**(step+1)))+1e-6),-1.5,1.5)
            atomic_json(out/f'update-{step}.json',dict(weights=w.tolist(),paired_rank_utility=advantage.tolist(),
                        source_seeds=[g['seed'] for g in outcomes[::2]],ranks=[g['rank'] for g in outcomes],
                        decisions=sum(g['steps'] for g in outcomes),changes=sum(g['changes'] for g in outcomes)))
    atomic_json(out/'endpoint.json',dict(seed=SEEDS[arm],weights=w.tolist(),updates=8,games=1536))


def panel(shard):
    profiles=[np.zeros(WIDTH)]
    for arm in (0,1):
        profiles.append(json.loads((ROOT/f'train-{arm}/endpoint.json').read_text())['weights'])
    out=ROOT/f'eval-{shard}'
    out.mkdir(parents=True,exist_ok=False)
    start=83500000+32*shard
    with ProcessPoolExecutor(int(os.environ.get('SLURM_CPUS_PER_TASK',4))) as pool:
        games=list(pool.map(run_game,[('transfer',s,p) for s in range(start,start+32) for p in profiles]))
    atomic_json(out/'games.json',games)


def assessment():
    rows=[]
    for shard in range(8):
        rows.extend(json.loads((ROOT/f'eval-{shard}/games.json').read_text()))
    atomic_json(ROOT/'complete.json', paired_endpoints(rows, 83500000, 256))


def paired_endpoints(rows, first_seed, n):
    if len(rows)!=3*n:
        raise ValueError('incomplete natural panel')
    seeds=[r['seed'] for r in rows]
    if not all(seeds[3*i:3*i+3]==[first_seed+i]*3 for i in range(n)):
        raise ValueError('mismatched natural seeds')
    data=[]
    for arm in range(3):
        games=rows[arm::3]
        ranks=np.array([r['rank'] for r in games])
        strength=lambda r,t:r['eot'].get(str(t),[np.nan])[0]
        pair=np.array([[strength(r,t) for t in (9,12)] for r in games])
        growth=np.log1p(pair[:,1])-np.log1p(pair[:,0])
        data.append((np.array(UTILITY)[ranks-1],ranks,np.isfinite(pair[:,1]).astype(float),growth))
    report=[]
    diffs=[[],[],[],[]]
    for i,d in enumerate(data):
        summary=dict(rank=float(d[1].mean()),original_utility=float(d[0].mean()),reach12=float(d[2].mean()),
                     winners=int(sum(d[1]==1)),top4=int(sum(d[1]<=4)),
                     changes=sum(r['changes'] for r in rows[i::3]))
        for t in (9,12,14):
            vals=[r['eot'][str(t)][0] for r in rows[i::3] if str(t) in r['eot']]
            summary[f't{t}']=dict(n=len(vals),p50=float(np.median(vals)) if vals else None)
        if i:
            for j,k in enumerate(('utility','rank','reach','growth')):
                delta=d[j]-data[0][j]
                diffs[j].append(delta)
                summary[k+'_diff']=bootstrap(delta[np.isfinite(delta)])
        report.append(summary)
    # Bootstrap the same game seeds jointly and recompute each arm's common-survivor mean.
    rng=np.random.default_rng(118128)
    index=rng.integers(n,size=(10000,n))
    joint={}
    for name,values in zip(('utility','rank','reach','growth'),diffs):
        estimates=np.stack([np.nanmean(v[index],axis=1) for v in values]).mean(0)
        joint[name]=dict(mean=float(np.mean([np.nanmean(v) for v in values])),ci95=np.quantile(estimates,[.025,.975]).tolist())
    gate=(all(r['growth_diff']['mean']>0 and r['utility_diff']['mean']>=0 and r['reach_diff']['mean']>=-.03 for r in report[1:])
          and joint['growth']['ci95'][0]>0 and joint['rank']['ci95'][1]<.20)
    return dict(advance=bool(gate),report=report,joint=joint,paired_seeds=n)


if __name__=='__main__':
    if sys.argv[1]=='train':
        train(int(sys.argv[2]))
    elif sys.argv[1]=='eval':
        panel(int(sys.argv[2]))
    elif sys.argv[1]=='assess':
        assessment()
    else:
        raise ValueError('unsupported stage')
