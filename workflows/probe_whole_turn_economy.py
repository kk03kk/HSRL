"""Paired whole-turn policy comparison on explicit, held-out-source fixtures."""
import argparse,copy,gzip,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import hsbg_engine as E
from hsbg.az.train.research_economy import Operator,CORES,cid,strength
from hsbg.az.train.economic_tasks import debug,own
from hsbg.az.train.mirror_snapshots import capture_eot
from hsbg.az.train.eval_policy import EvalProtocol,make_eval_policy
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity,environment_identity
from workflows.build_recruit_curriculum import endow
from tools.probe_economic_starts import select_sources,run_turn,canonical_items

VARIANTS=('normal','complete','complete_gold','missing_privateer')


def fixture(source,variant):
    if variant not in VARIANTS:raise ValueError(variant)
    state,edits=endow(source)
    game=E.Game.from_json(json.dumps(state))
    def edit(cmd,**kw):
        debug(game,cmd,**kw);edits.append({'cmd':cmd,**kw})
    if variant!='normal':
        board=json.loads(game.view(0))['me']['board']
        candidates=[i for i,x in enumerate(board) if cid(x) not in CORES]
        if len(board)>=7:edit('remove',zone='board',index=min(candidates,key=lambda i:strength(board[i])))
        edit('add_minion',zone='board',card='BG36_523')
    if variant=='complete_gold':
        for card,golden in [('BG_LOE_077','TB_BaconUps_045'),('BG33_825','BG33_825_G')]:
            board=json.loads(game.view(0))['me']['board']
            edit('remove',zone='board',index=next(i for i,x in enumerate(board) if cid(x)==card))
            edit('add_minion',zone='board',card=golden)
    if variant=='missing_privateer':
        board=json.loads(game.view(0))['me']['board']
        edit('remove',zone='board',index=next(i for i,x in enumerate(board) if cid(x)=='BG33_825'))
        if len(own(game)['shop'])>=8:edit('remove',zone='shop',index=7)
        edit('add_minion',zone='shop',card='BG33_825')
    for key in ('gold','health','tier'):
        if own(game)[key]!=source['players'][0][key]:raise ValueError('unexpected resource intervention')
    if any(game.diagnostic_counts()):raise ValueError('fixture diagnostics')
    return json.loads(game.state_json()),edits


def public_view(game):
    v=json.loads(game.view(0))
    return {k:v[k] for k in ('me','mask','turn','phase','lobby_races')}


def run_operator(state):
    game=E.Game.from_json(json.dumps(state));operator=Operator();steps=[]
    for index in range(128):
        view=public_view(game)
        action,reason=(0,'action_cap') if index>=100 and view['mask'][0] else operator.act(view)
        if not bytes(game.legal_mask(0))[action]:raise ValueError('illegal operator action')
        if action==0:
            end={'action':0,'executed':False,'reason':reason};break
        before=json.loads(game.state_json());game.step(0,action)
        if game.turn!=state['turn'] or any(game.diagnostic_counts()):
            raise ValueError(f'unclean operator transition index={index} action={action} '
                f'turn={state["turn"]}->{game.turn} alive={own(game)["alive"]} '
                f'diagnostics={game.diagnostic_counts()}')
        after=json.loads(game.state_json());operator.note(view,public_view(game),action)
        steps.append({'action':action,'reason':reason,'before':before,'after':after})
    else:raise ValueError('unresolved operator turn budget')
    return {'initial':state,'steps':steps,'final':json.loads(game.state_json()),'end_intent':end}


def metrics(trace):
    initial=trace['initial']['players'][0];final=trace['final']['players'][0]
    game=E.Game.from_json(json.dumps(trace['final']))
    frame=capture_eot(game,0)
    return {'actions':len(trace['steps']),
        'gold_spent':final['stats']['gold_spent_total']-initial['stats']['gold_spent_total'],
        'gold_end':final['gold'],'hand_end':len(final['hand']),
        'initial_body':sum(strength(x) for x in initial['board']),
        'eot_body':sum(strength(x) for x in frame['player']['board']),
        'core_count':len({cid(x) for x in json.loads(game.view(0))['me']['board']} & set(CORES)),
        'stopped_at_cap':trace['end_intent'].get('reason')=='action_cap'}


_POLICY=None
def worker(task):
    global _POLICY
    torch.set_num_threads(1)
    source,variant,checkpoint,out=task
    if _POLICY is None:_POLICY=make_eval_policy(checkpoint,protocol=EvalProtocol(),device='cpu')['policy']
    initial,edits=fixture(source['state'],variant)
    fresh=json.loads(E.Game(seed=source['seed']+40000000).state_json())
    initial['seed']=fresh['seed'];initial['rng']=fresh['rng']
    _,parent=run_turn(_POLICY,initial,canonical_items())
    operator=run_operator(initial)
    trace_path=Path(out)/(str(source['seed'])+'-'+variant+'.jsonl.gz')
    with gzip.open(trace_path,'xt',encoding='utf-8',compresslevel=3) as stream:
        for name,trace in [('parent',parent),('operator',operator)]:
            stream.write(json.dumps({'policy':name,'source_seed':source['seed'],
                'variant':variant,'interventions':edits,**trace},separators=(',',':'))+'\n')
    return {'source_seed':source['seed'],'variant':variant,'parent':metrics(parent),
        'operator':metrics(operator),'trace':file_identity(trace_path)}


def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    sources=select_sources(args.bank)
    if args.limit:sources=sources[:args.limit]
    identity=file_identity(args.checkpoint)
    atomic_json(out/'run.json',{'schema':'hsai.whole-turn-economy-probe/v1','engine':__import__('hsbg.az.train.common',fromlist=['engine_info']).engine_info(),
        'source_manifest':file_identity(Path(args.bank)/'manifest.json'),'source_seeds':[x['seed'] for x in sources],
        'checkpoint':identity,'variants':VARIANTS,'budget':100,
        'source':source_identity(Path(__file__).resolve().parents[1]),'environment':environment_identity(),
        'scope':'paired synthetic whole-turn mechanism diagnosis, not natural full-game strength; public operator has no simulator/RNG access'})
    tasks=[(s,v,args.checkpoint,str(out)) for s in sources for v in VARIANTS]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as executor:
        rows=list(executor.map(worker,tasks))
    if file_identity(args.checkpoint)!=identity:raise ValueError('checkpoint changed')
    summaries={}
    rng=np.random.default_rng(1192901)
    for variant in VARIANTS:
        rr=[r for r in rows if r['variant']==variant]
        diff=np.array([r['operator']['eot_body']-r['parent']['eot_body'] for r in rr])
        boot=diff[rng.integers(0,len(diff),size=(10000,len(diff)))].mean(1)
        summaries[variant]={'sources':len(rr),'paired_eot_delta_mean':float(diff.mean()),
            'paired_eot_delta_ci95':np.quantile(boot,[.025,.975]).tolist(),
            'operator_better':int(sum(diff>0)),
            **{name:{k:float(np.mean([r[name][k] for r in rr])) for k in rr[0][name]} for name in ('parent','operator')}}
    atomic_json(out/'complete.json',{'valid':True,'run':file_identity(out/'run.json'),
        'summary':summaries,'rows':rows})
    print(json.dumps(summaries),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',required=True);p.add_argument('--out',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--limit',type=int,default=0);main(p.parse_args())
