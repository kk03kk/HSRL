"""Locate snapshot misses by replaying sealed actions, never making new decisions."""
import argparse,gzip,json,sys
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.policies import Decision
from hsbg.az.train.evaluate import EvalConfig,evaluate_policy
from hsbg.runtime.artifacts import atomic_json,file_identity
from workflows.diagnose_missing_snapshot import AuditPool

class Replay:
    name='sealed-action-replay-diagnostic'
    def __init__(self,start,actions,final,pool):
        self.start=start;self.actions=actions;self.final=final;self.pool=pool;self.i=0
    def reset(self):self.i=0
    def begin_game(self,game,index,seed,seat):
        self.pool.game_ref=game
        if seed!=self.start['seed'] or json.loads(game.state_json())!=self.start['state']:
            raise ValueError('sealed initial state differs')
    def before(self,game,seat):
        row=self.actions[self.i];self.pool.game_ref=game
        if row['seat']!=seat or json.loads(game.state_json())!=row['state_before']:
            raise ValueError('sealed action input differs')
        return row
    def decide(self,game,seat):
        row=self.actions[self.i]
        return Decision(row['action'],None,row['policy_info'])
    def after(self,game,before,decision,executed_action,forced_tier=False):
        if executed_action!=before['action'] or json.loads(game.state_json())!=before['state_after']:
            raise ValueError('sealed native action transition differs')
        self.i+=1
    def end_game(self,game,rank,quality):
        if self.i!=len(self.actions) or rank!=self.final['rank']:
            raise ValueError('sealed learner result differs')
        if json.loads(game.state_json())!=self.final['state']:
            raise ValueError('sealed terminal world state differs')

def main(args):
    import torch
    torch.set_num_threads(1)
    inventory=json.loads(Path(args.inventory).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    groups=defaultdict(list)
    for case in inventory['late_games']:groups[case['trace']].append(case)
    results=[];counts=Counter()
    for trace,cases in groups.items():
        if file_identity(trace)!=cases[0]['trace_identity']:raise ValueError('original trace changed')
        seeds={c['seed'] for c in cases};records={s:{'actions':[]} for s in seeds}
        with gzip.open(trace,'rt',encoding='utf-8') as stream:
            for line in stream:
                # Seed is near the beginning of each large action row. Parse only
                # requested games, while still binding the complete trace hash.
                import re
                m=re.search(r'"seed":([0-9]+)',line[:200])
                if not m or int(m[1]) not in seeds:continue
                row=json.loads(line);record=records[int(m[1])]
                if row['event']=='action':record['actions'].append(row)
                elif row['event']=='game_start':record['start']=row
                elif row['event']=='game_end':record['end']=row
        for case in cases:
            panel=next(p for p in inventory['panels'] if p['path']==case['panel'])
            field=panel['protocol']['field'];record=records[case['seed']]
            dest=out/f"{case['model']}-{case['seed']}";dest.mkdir()
            pool=AuditPool(field['mirror_snaps'],dest)
            replay=Replay(record['start'],record['actions'],record['end'],pool)
            result=evaluate_policy(replay,EvalConfig(games=1,seed=case['seed'],
                mirror_pool=pool,mirror_strict=False,mirror_seats=field['mirror_seats'],mirror_p=field['mirror_p'],
                decision_recorder=replay,keep_start_board=panel['protocol']['keep_start_board']))
            if result['mirror']['mirror_missing']!=len(pool.misses):raise ValueError('missing counter differs')
            counts[(case['panel'],case['model'])]+=len(pool.misses)
            row={'case':case,'original_all_actions_and_final_state_exact':True,'actions':replay.i,
                'misses':pool.misses,'rank':result['rank_sequence'][0]}
            atomic_json(dest/'complete.json',row);results.append(row)
            print(json.dumps({'model':case['model'],'seed':case['seed'],'actions':replay.i,'misses':pool.misses}),flush=True)
        del records
    for panel in inventory['panels']:
        for model in panel['models']:
            if counts[(panel['path'],model['name'])]!=model['mirror_missing']:
                raise ValueError('late-case replay did not account for every panel missing event')
    atomic_json(out/'complete.json',{'diagnostic_only':True,'ability_eligible':False,
        'inventory':file_identity(args.inventory),'all_panel_missing_events_accounted':True,'cases':results,
        'all_misses_after_learner_death':all(not m['learner_alive'] for r in results for m in r['misses']),
        'scope':'Exact sealed-action/native-state replay, not new policy evaluation or outcome filtering'})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inventory',required=True)
    p.add_argument('--out',required=True);main(p.parse_args())
