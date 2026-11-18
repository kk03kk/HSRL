"""Read sealed trajectories to locate imitation drift and controller memory use.

No new policy actions are executed. A recording evaluator advances the ordinary
guard with the recorded action while an independent teacher supplies diagnostics.
"""
import argparse,copy,gzip,json,multiprocessing,sys
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity
from hsbg.evaluation.quality import suffix_quality


class RecordedEvaluator:
    def __init__(self):self.action=None;self.mask=None
    def __call__(self,features,cards,mask,seat):
        self.mask=np.array(mask,dtype=bool)
        if not self.mask[self.action]:raise ValueError('sealed action violates reconstructed guard')
        pi=np.zeros(len(mask),dtype=np.float32);pi[self.action]=1
        return pi,0.


def scan(spec):
    import hsbg_engine as E
    from hsbg.az.train.ppo import PPOGreedyPolicy
    from hsbg.az.train.research_economy import Operator
    from hsbg.az.train.research_economy_policy import active
    from hsbg.az.train import mechanism_obs as M
    from workflows.build_payoff_dataset import gate
    from hsbg.az.train.conditional_economy import current_gate_spec
    path,expected,is_teacher=spec;path=Path(path)
    if file_identity(path)!={k:expected[k] for k in ('bytes','sha256')}:
        raise ValueError('sealed trace changed')
    evaluator=RecordedEvaluator();guard=PPOGreedyPolicy(evaluator);operator=Operator('growth')
    previous=None;counts=Counter();by_reason=defaultdict(Counter);by_card=defaultdict(Counter)
    deaths=[];examples=[];abstentions=[];disagreement_examples=defaultdict(list)
    previous_gate=None;gate_edges=[];gate_groups=current_gate_spec('gold_engine')['groups']
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['event']!='action':continue
            game=E.Game.from_json(json.dumps(row['state_before']));seat=row['seat']
            full=json.loads(game.view(seat));view={k:full[k] for k in ('me','mask','turn','phase','lobby_races')}
            if previous is not None:
                old,action=previous
                if old['turn']==view['turn']:operator.note(old,view,action)
                else:operator=Operator('growth')
            evaluator.action=int(row['action']);decision=guard.decide(game,seat)
            if decision.action!=row['action'] or bool(decision.info['forced'])!=bool(row['policy_info'].get('forced')):
                raise ValueError('guard replay differs')
            counts['actions']+=1
            f,c=M.observe(game,seat,M.SCHEMA)
            within=gate(np.frombuffer(f,dtype='<f4'),np.frombuffer(c,dtype='<i4'))
            counts['gold_gate_actions']+=int(within)
            if previous_gate is not None and previous_gate['active']!=bool(within):
                counts['observed_gate_entries' if within else 'observed_gate_exits']+=1
                owned={x['id'] for x in view['me']['board']+view['me']['hand']}
                missing=[group[0] for group in gate_groups if not owned.intersection(group)]
                if not within:
                    for card in missing:counts['gate_exit_missing/'+card]+=1
                    if 16<=previous_gate['action']<23:counts['gate_exits_following_sell']+=1
                if len(gate_edges)<8:
                    gate_edges.append({'previous':previous_gate,'now_turn':row['turn'],
                        'now_action_index':row['action_index'],'now_active':bool(within),'missing_groups':missing})
            previous_gate={'active':bool(within),'turn':row['turn'],'action_index':row['action_index'],
                'action':row['action']}
            before_player=row['state_before']['players'][seat];after_player=row['state_after']['players'][seat]
            if before_player['alive'] and not after_player['alive'] and row['action']!=0:
                deaths.append({'turn':row['turn'],'action_index':row['action_index'],'action':row['action'],
                    'before_health':before_player['health'],'after_health':after_player['health'],
                    'pending':view['me']['pending'],'casting':view['me'].get('casting'),
                    'hand':[x.get('id') for x in view['me']['hand']]})
            if active(view):
                guarded={**view,'mask':evaluator.mask.tolist()}
                no_cycle=copy.deepcopy(operator);no_cycle.cycled.clear()
                try:teacher,reason=operator.act(guarded)
                except ValueError as exc:
                    if is_teacher or str(exc)!='destructive teacher target has no safe continuation':raise
                    counts['teacher_unavailable']+=1
                    counts['teacher_unavailable_gold_gate']+=int(within and not decision.info['forced'])
                    if len(abstentions)<5:
                        abstentions.append({'turn':row['turn'],'action_index':row['action_index'],
                            'executed_action':row['action'],'reason':str(exc),'gold_gate':within,
                            'casting':view['me'].get('casting'),'last_play':operator.last_play})
                    previous=(view,int(row['action']))
                    continue
                try:without_cycle,without_reason=no_cycle.act(guarded)
                except ValueError as exc:
                    if str(exc)!='destructive teacher target has no safe continuation':raise
                    without_cycle=None;without_reason=str(exc)
                    counts['memory_ablation_unavailable']+=1
                if not evaluator.mask[teacher]:raise ValueError('teacher suggestion violates shared guarded mask')
                if is_teacher and not decision.info['forced'] and (teacher!=row['action'] or reason!=row['policy_info']['research_operator_reason']):
                    raise ValueError('sealed teacher cannot be reproduced from public history')
                if within and not decision.info['forced']:
                    counts['eligible']+=1;counts['agreement']+=int(teacher==row['action'])
                    by_reason[reason]['rows']+=1;by_reason[reason]['agreement']+=int(teacher==row['action'])
                    from hsbg.az.train.common import group_of
                    from tools.analyze_payoff_labels import action_card
                    cc=np.frombuffer(c,dtype='<i4')
                    wanted=(group_of(teacher),action_card(teacher,cc))
                    actual=(group_of(row['action']),action_card(row['action'],cc))
                    same_card=wanted==actual
                    by_reason[reason]['same_action_group_and_card']+=int(same_card)
                    key='/'.join((reason,*wanted));by_card[key]['rows']+=1
                    by_card[key]['exact']+=int(teacher==row['action'])
                    by_card[key]['same_action_group_and_card']+=int(same_card)
                    if not same_card and len(disagreement_examples[reason])<2:
                        disagreement_examples[reason].append({'turn':row['turn'],'action_index':row['action_index'],
                            'teacher_action':teacher,'teacher':wanted,'executed_action':row['action'],'executed':actual,
                            'tier':view['me']['tier'],'gold':view['me']['gold'],
                            'board':[x['id'] for x in view['me']['board']],
                            'hand':[x['id'] for x in view['me']['hand']]})
                    dependent=without_cycle is not None and teacher!=without_cycle
                    counts['cycle_memory_changes_action']+=int(dependent)
                    by_reason[reason]['cycle_memory_changes_action']+=int(dependent)
                    if dependent and len(examples)<3:
                        examples.append({'turn':row['turn'],'action_index':row['action_index'],
                            'teacher_action':teacher,'teacher_reason':reason,'without_cycle_action':without_cycle,
                            'without_cycle_reason':without_reason,'same_public_view':True})
            previous=(view,int(row['action']))
    if counts['actions']!=expected['actions']:raise ValueError('trace action coverage differs')
    return {'path':str(path),'identity':file_identity(path),'counts':dict(counts),
        'reasons':{k:dict(v) for k,v in by_reason.items()},'recruit_deaths':deaths,'memory_examples':examples,
        'teacher_cards':{k:dict(v) for k,v in by_card.items()},'disagreement_examples':dict(disagreement_examples),
        'teacher_abstention_examples':abstentions,'gate_transition_examples':gate_edges}


def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    rows=[];inputs=[]
    for name,root in zip(args.names,args.runs):
        root=Path(root);complete=json.loads((root/'complete.json').read_text())
        if not complete['valid'] or complete['probe']:raise ValueError('requires sealed complete diagnostic')
        if complete['run']!=file_identity(root/'run.json'):raise ValueError('run changed')
        selected=[r for r in complete['rows'] if r['name']==name and r['variant']=='complete_gold']
        # Replica 0 is the teacher-state development set; use the same futures
        # for student drift diagnostics without treating extra replicas as new sources.
        selected=[r for r in selected if r['replica']==0]
        if len(selected)!=49 or len({r['source_seed'] for r in selected})!=49:
            raise ValueError('requires all 49 fixed development sources')
        specs=[]
        for r in selected:
            trial=root/'trials'/name/f"case-{r['case']:03d}-complete_gold-r00"
            result=json.loads((trial/'result.json').read_text())
            if result['summary']!=r:raise ValueError('trial differs')
            suffix_quality(result['evaluation'])
            specs.append((str(trial/'decisions.jsonl.gz'),r['trace'],name=='growth'))
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            detail=list(pool.map(scan,specs))
        counts=Counter();reasons=defaultdict(Counter);cards=defaultdict(Counter)
        for d in detail:
            counts.update(d['counts'])
            for k,v in d['reasons'].items():reasons[k].update(v)
            for k,v in d['teacher_cards'].items():cards[k].update(v)
        panel={'name':name,'sources':49,'counts':dict(counts),'reasons':{k:dict(v) for k,v in reasons.items()},
            'teacher_cards':{k:dict(v) for k,v in cards.items()},'trials':detail}
        atomic_json(out/(name+'.json'),panel);rows.append(panel)
        inputs.append({'root':str(root),'complete':file_identity(root/'complete.json'),'run':complete['run']})
    result={'inputs':inputs,'rows':rows,'source':source_identity(Path(__file__).resolve().parents[1]),
        'scope':'Read-only shadow teacher on sealed visited states; no new ability samples or labels admitted to training',
        'limitations':['Teacher suggestions are heuristic, not optimal actions',
            'Explicit unsafe-target abstentions remain counted; they are never replaced by invented labels',
            'Same action group and card ignores position, stats and target identity; not an equivalence or optimality test',
            'Gate transitions are between recorded learner decisions; terminal deaths have no subsequent decision',
            'Clearing private cycled memory is a controller ablation, not proof of reachable aliased environment histories',
            'Teacher/student visited-state distributions differ; agreement differences are descriptive']}
    atomic_json(out/'complete.json',result)
    print(json.dumps([{k:r[k] for k in ('name','sources','counts','reasons')} for r in rows]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--names',nargs='+',required=True);p.add_argument('--out',required=True)
    p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    if len(args.runs)!=len(args.names):p.error('one policy name required per run')
    main(args)
