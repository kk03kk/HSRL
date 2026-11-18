"""Test a public two-turn policy choice on paired natural games, without named-card rules."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.common import engine_info, set_rank_utility_table
from hsbg.az.train.eval_policy import EvalProtocol, make_eval_policy
from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
from hsbg.runtime.artifacts import atomic_json, file_identity
from tools.assess_investment_timing import UTILITY, compare, utility
from workflows.probe_investment_timing import MODEL, FIELDS, TimingPolicy


SPAN = {'development':(79210000,128),'transfer':(79310000,160)}
ALTERNATES = dict(a='7bffc785b804a4fb34aff74c48029f5092df683f267bfb6cd547b293b56bb0a5',
                  b='71fd1287fef2aac1c937386d22c7fd4d581e0a1f7a8280d1960d96f25c2ea13d')
CACHE = {}


def category(action):
    if action in (0,1,3):
        return ('end','refresh','upgrade')[(0,1,3).index(action)]
    for low,high,name in ((8,16,'buy'),(16,23,'sell'),(46,126,'play'),(126,196,'magnet')):
        if low <= action < high:
            return name
    return ''


class Option(TimingPolicy):
    def __init__(self, policy, alternates, enabled, late):
        super().__init__(policy,'parent',late)
        self.alternates = alternates
        self.enabled = enabled

    def __call__(self, f, c, mask, seat):
        base,v = self.original(f,c,mask,seat)
        event = self.events.get('option')
        turn = self.current
        if event and (not self.enabled or turn > event['turn']+1):
            return base,v
        if not event and turn not in ((9,10) if self.late else (7,8)):
            return base,v
        logits = [policy(f,c,mask,seat) for policy in self.alternates]
        other = (logits[0][0]+logits[1][0])*.5
        if not event:
            a,b = int(np.argmax(base)),int(np.argmax(other))
            if a == b or not category(a) or not category(b) or self.player['pending'] or not mask[0]:
                return base,v
            if self.policy._turn_counts.get((seat,turn),0)>=90:
                return base,v
            p = self.player
            total = sum(m['atk']+m['hp'] for m in p['board'])
            # No hidden state enters features. The native hash is only a paired-prefix check.
            shared = [turn,p['tier'],p['health']+p['armor'],p['gold'],len(p['board']),len(p['hand']),
                      float(np.log1p(total)),v,(logits[0][1]+logits[1][1])*.5-v,
                      float(base[a]-base[b]),float(other[b]-other[a]),
                      float(np.argmax(logits[0][0])==np.argmax(logits[1][0]))]
            cats = ('end','refresh','upgrade','buy','sell','play','magnet')
            shared.extend(int(category(b)==k)-int(category(a)==k) for k in cats)
            self.events['option'] = dict(turn=turn,index=len(self.actions),actions=[a,b],features=shared,
                state=hashlib.sha256(self.game.state_json().encode()).hexdigest())
        return (other,v) if self.enabled else (base,v)


def run(task):
    import torch
    torch.set_num_threads(1)
    split,seed,out = task
    field = FIELDS[split]
    if field not in CACHE:
        set_rank_utility_table(UTILITY)
        proto = EvalProtocol(mirror_snaps=field,engine=engine_info())
        paths = [MODEL]+[f'artifacts/runs/nus-practice-mixed-{arm}-20261118-1/train/model.pt' for arm in ALTERNATES]
        policies = [make_eval_policy(p,protocol=proto,device='cpu')['policy'] for p in paths]
        CACHE[field] = policies[0],[p.evaluator for p in policies[1:]],MirrorSnapshotPool(field,sample='p50band')
    parent,alts,pool = CACHE[field]
    rows = []
    for enabled in (False,True):
        policy = Option(parent,alts,enabled,seed%2==1)
        res = evaluate_policy(policy,EvalConfig(games=1,seed=seed,mirror_pool=pool))
        if any(res[k] for k in ('truncated_games','truncated_combats','illegal_actions')) or (
                res['mirror']['mirror_errors'] or res['mirror']['mirror_rejected']):
            raise ValueError('invalid full game')
        rows.append(dict(rank=res['rank_sequence'][0],eot={str(t):v for t,v in policy.eot.items()},
                         event=policy.events.get('option'),actions=policy.actions))
    a,b = rows
    event = a['event']
    if event != b['event']:
        raise ValueError('option branch differs')
    if event:
        i = event['index']
        if a['actions'][:i]!=b['actions'][:i] or [a['actions'][i],b['actions'][i]]!=event['actions']:
            raise ValueError('public option or prefix was not executed')
    elif a!=b:
        raise ValueError('untriggered game differs')
    atomic_json(out/f'{seed}.json',rows)
    return rows


def fit(rows):
    data = [(a['event']['features'], utility(b)-utility(a)) for a,b in rows if a['event']]
    x,y = map(np.array,zip(*data))
    center,scale = x.mean(0),x.std(0)
    scale[scale<1e-6]=1
    z = (x-center)/scale
    weights = np.linalg.solve(z.T@z+40*np.eye(z.shape[1]),z.T@(y-y.mean()))
    return weights,center,scale,y.mean()


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    input_plan = json.loads(Path('config/research/natural12m-20261118.json').read_text())
    paths = dict(input_plan['asset_sha256'])
    for arm,sha in ALTERNATES.items():
        paths[f'artifacts/runs/nus-practice-mixed-{arm}-20261118-1/train/model.pt'] = sha
    if engine_info()['source_hash'] != input_plan['engine_fingerprint'] or any(
            file_identity(p)['sha256']!=sha for p,sha in paths.items()):
        raise ValueError('option source mismatch')
    from hsbg.runtime.provenance import source_identity
    src = source_identity(Path.cwd())
    if src['tracked_dirty']:
        raise ValueError('source changed')
    atomic_json(out/'run.json',dict(assets=paths,engine=engine_info(),source=src,seeds=SPAN,
                primary='positive held-out original utility over both fixed choices, without >3pp T12 reach loss; '
                        'report separate early and late choices and common growth. No source identifiers in fit.'))
    with ProcessPoolExecutor(16) as executor:
        cohorts = {}
        for split,(start,n) in SPAN.items():
            cohorts[split] = list(executor.map(run,[(split,s,out) for s in range(start,start+n)]))
    weights,center,scale,intercept = fit(cohorts['development'])
    stats = {}
    for split,rows in cohorts.items():
        take = [bool(a['event'] and (intercept+((a['event']['features']-center)/scale)@weights)>0)
                for a,b in rows]
        results = {}
        for label,indices in [('all',list(range(len(rows)))),('early',list(range(0,len(rows),2))),
                              ('late',list(range(1,len(rows),2)))]:
            parent = [rows[i][0] for i in indices]
            option = [rows[i][1] for i in indices]
            gated = [rows[i][int(take[i])] for i in indices]
            results[label] = dict(triggered=sum(bool(a['event']) for a in parent),switched=sum(take[i] for i in indices),
                option_minus_parent=compare(option,parent),gated_minus_parent=compare(gated,parent),
                gated_minus_option=compare(gated,option))
        stats[split]=results
    atomic_json(out/'complete.json',dict(fits=dict(weights=weights.tolist(),intercept=float(intercept),
                center=center.tolist(),scale=scale.tolist()),stats=stats,run=file_identity(out/'run.json')))


if __name__=='__main__':
    main(Path(sys.argv[1]))
