"""Paired first-turn diagnostics on held-out natural and explicitly endowed states.

This is a mechanism probe, never a placement evaluation. Stops BEFORE executing
EndTurn, so combat and next-turn gold reset cannot contaminate economic metrics.
"""
import argparse,gzip,json,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hsbg_engine as engine
import numpy as np
import torch
from hsbg.az.train.common import engine_info
from hsbg.az.train.economic_tasks import own,card_ids
from hsbg.az.train.eval_policy import EvalProtocol,make_eval_policy
from hsbg.runtime.artifacts import atomic_json,file_identity
from workflows.build_recruit_curriculum import endow,CORES


def canonical_items():
    cards=json.loads(Path('engine/crates/hsbg-engine/data/cards.json').read_text(encoding='utf-8'))['cards']
    normal={c['id']:c.get('normal') or c['id'] for c in cards}
    return {i:normal[engine.card_info(i+1)[0]] for i in range(len(cards))}


def item(value,ids):
    for key in ('item','Minion','Spell'):
        if key in value:return item(value[key],ids)
    return ids[value['card']]


def select_sources(bank):
    bank=Path(bank);manifest=json.loads((bank/'manifest.json').read_text())
    if file_identity(bank/'states.jsonl.gz')!=manifest['data']:raise ValueError('source checksum differs')
    groups={}
    with gzip.open(bank/'states.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line);p=r['state']['players'][0]
            if r['seed']%5!=0 or r['seat']!=0 or not r['state']['lobby_races'] & (1<<6):continue
            if not 8<=r['turn']<=9 or p['health']<15 or p['pending'] or p['ended_turn']:continue
            if r['seed'] not in groups or r['turn']<groups[r['seed']]['turn']:groups[r['seed']]=r
    if not groups:raise ValueError('no held-out economic source games')
    return [groups[k] for k in sorted(groups)]


class ProtectCoresView:
    """Explicit diagnostic policy constraint; never changes native game rules."""
    def __init__(self,game,ids):self.game=game;self.ids=ids
    def __getattr__(self,name):return getattr(self.game,name)
    def legal_mask(self,seat):
        mask=bytearray(self.game.legal_mask(seat))
        for i,m in enumerate(json.loads(self.game.player_json(seat))['board']):
            if item(m,self.ids) in CORES:mask[16+i]=0
        return bytes(mask)


def run_turn(policy,state,ids,*,protect_cores=False,allow_recruit_death=False):
    game=engine.Game.from_json(json.dumps(state));policy.reset()
    policy_game=ProtectCoresView(game,ids) if protect_cores else game
    initial=own(game);initial_turn=game.turn;steps=[];counts=Counter();income=0
    for index in range(256):
        before=json.loads(game.state_json());p=before['players'][0]
        legal=list(bytes(game.legal_mask(0)))
        policy_mask=list(bytes(policy_game.legal_mask(0)))
        decision=policy.decide(policy_game,0);action=int(decision.action)
        decision_info=dict(getattr(decision,'info',{}))
        if not 0<=action<len(legal) or not legal[action]:raise ValueError('illegal diagnostic action')
        if not policy_mask[action]:raise ValueError('diagnostic policy constraint failed')
        if action==0:
            end_intent={'action':0,'executed':False,
                        'reason':decision_info.get('forced_reason') or 'probe stops before EndTurn',
                        'policy_info':decision_info,'legal_mask':legal}
            break
        chosen=None;kind=None
        if 8<=action<16:kind='buy';chosen=item(p['shop'][action-8],ids)
        elif 16<=action<23:kind='sell';chosen=item(p['board'][action-16],ids)
        elif 46<=action<126:kind='play';chosen=item(p['hand'][(action-46)//8],ids)
        if kind:counts[kind+':'+chosen]+=1
        game.step(0,action)
        after=json.loads(game.state_json());q=after['players'][0]
        if any(game.diagnostic_counts()):raise ValueError('unclean native diagnostic')
        dead=not q['alive']
        if game.turn!=initial_turn and not (allow_recruit_death and dead):
            raise ValueError('unclean turn boundary')
        income+=max(0,q['gold']-p['gold'])
        steps.append({'action':action,'executed':True,'legal_mask':legal,
                      'policy_constraint_mask':policy_mask,'policy_info':decision_info,
                      'before':before,'after':after})
        if dead and allow_recruit_death:
            end_intent={'action':None,'executed':False,'reason':'recruit_death'}
            break
    else:raise ValueError('policy failed to finish recruit turn')
    final=own(game)
    strength=lambda p:sum(m['atk']+m['hp'] for m in p['board'])
    board=[item(m,ids) for m in final['board']]
    metric={'actions':len(steps),'gold_start':initial['gold'],'gold_end':final['gold'],
        'gross_gold_gains':income,'gold_spent':final['stats']['gold_spent_total']-initial['stats']['gold_spent_total'],
        'strength_start':strength(initial),'strength_end':strength(final),
        'strength_delta':strength(final)-strength(initial),'board_count':len(board),
        'cores_retained':sum(c in board for c in CORES),'all_cores_retained':all(c in board for c in CORES),
        'core_sells':sum(counts['sell:'+c] for c in CORES),
        'rascal_plays':counts['play:BG33_821'],
        'bounty_plays':sum(counts['play:BG33_'+str(i)] for i in range(811,816)),
        'actions_by_card':dict(counts)}
    return metric,{'initial':state,'steps':steps,'end_intent':end_intent,'final':json.loads(game.state_json())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',required=True);p.add_argument('--out',required=True)
    p.add_argument('--checkpoint',action='append',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--protect-cores',action='store_true',help='Diagnostic intervention, not a learned policy')
    args=p.parse_args();torch.set_num_threads(1)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    sources=select_sources(args.bank);tasks=[]
    for source in sources:
        for variant in ('natural','endowed'):
            state,edits=(source['state'],[]) if variant=='natural' else endow(source['state'])
            # Independent future RNG, identical between paired variants/models.
            future=json.loads(engine.Game(seed=source['seed']+40000000).state_json())
            state=json.loads(json.dumps(state));state['rng']=future['rng'];state['seed']=future['seed']
            tasks.append({'source_seed':source['seed'],'variant':variant,'initial':state,'interventions':edits})
    with gzip.open(out/'tasks.jsonl.gz','xt',encoding='utf-8') as f:
        for task in tasks:f.write(json.dumps(task)+'\n')
    ids=canonical_items();results={};proto=EvalProtocol()
    for spec in args.checkpoint:
        name,path=spec.split('=',1)
        if not name.replace('_','').isalnum() or name in results:raise ValueError('invalid duplicate name')
        built=make_eval_policy(path,protocol=proto,device=args.device);rows=[]
        with gzip.open(out/(name+'-trace.jsonl.gz'),'xt',encoding='utf-8',compresslevel=3) as f:
            for task in tasks:
                metrics,trace=run_turn(built['policy'],task['initial'],ids,protect_cores=args.protect_cores)
                row={'source_seed':task['source_seed'],'variant':task['variant'],**metrics};rows.append(row)
                f.write(json.dumps({'result':row,'interventions':task['interventions'],**trace},separators=(',',':'))+'\n')
        summaries={}
        for variant in ('natural','endowed'):
            rr=[r for r in rows if r['variant']==variant]
            summaries[variant]={'games':len(rr),**{k:{'mean':float(np.mean([r[k] for r in rr])),
                'p50':float(np.median([r[k] for r in rr]))} for k in metrics if k!='actions_by_card'}}
        results[name]={'checkpoint':file_identity(path),'summary':summaries,'rows':rows,
                       'trace':file_identity(out/(name+'-trace.jsonl.gz'))}
        atomic_json(out/(name+'.json'),results[name]);del built
        print(json.dumps({'model':name,'summary':summaries}),flush=True)
    atomic_json(out/'complete.json',{'models':results,'engine':engine_info(),'contract':proto.contract_dict(),
        'policy_intervention':'forbid selling the three economic cores' if args.protect_cores else 'none',
        'bank':file_identity(Path(args.bank)/'manifest.json'),'tasks':file_identity(out/'tasks.jsonl.gz'),
        'scope':'Held-out-source first-recruit-turn synthetic diagnostic, not placement performance; no training labels'})


if __name__=='__main__':main()
