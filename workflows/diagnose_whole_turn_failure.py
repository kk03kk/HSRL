"""Preserve and reproduce the first missing complete-turn demonstration."""
import argparse,gzip,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hsbg_engine as E
from workflows.probe_whole_turn_economy import fixture,public_view,VARIANTS
from hsbg.az.train.research_economy import Operator
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity


def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    bank=Path(args.bank);manifest=json.loads((bank/'manifest.json').read_text())
    if file_identity(bank/'states.jsonl.gz')!=manifest['data']:raise ValueError('source changed')
    groups={}
    with gzip.open(bank/'states.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);p=row['state']['players'][0]
            if row['seat']!=0 or not row['state']['lobby_races']&(1<<6):continue
            if not 8<=row['turn']<=9 or p['health']<15 or p['pending'] or p['ended_turn']:continue
            if row['seed'] not in groups or row['turn']<groups[row['seed']]['turn']:groups[row['seed']]=row
    source,variant=next((groups[k],v) for k in sorted(groups) for v in VARIANTS
        if not (Path(args.partial)/'traces'/(str(k)+'-'+v+'.json.gz')).is_file())
    state,edits=fixture(source['state'],variant)
    future=E.Game(seed=source['seed']+(40000000 if source['seed']%5==0 else 70000000))
    fresh=json.loads(future.state_json());state['seed']=fresh['seed'];state['rng']=fresh['rng']
    game=E.Game.from_json(json.dumps(state));operator=Operator();steps=[];failure=None
    for index in range(128):
        view=public_view(game)
        action,reason=(0,'action_cap') if index>=100 and view['mask'][0] else operator.act(view)
        before=json.loads(game.state_json());legal=list(bytes(game.legal_mask(0)))
        if action==0:break
        game.step(0,action);after=json.loads(game.state_json())
        row={'index':index,'action':action,'reason':reason,'legal':bool(legal[action]),
             'diagnostics':list(game.diagnostic_counts()),'before':before,'after':after}
        steps.append(row)
        if game.turn!=state['turn'] or any(row['diagnostics']) or not row['legal']:
            failure={k:row[k] for k in ('index','action','reason','legal','diagnostics')}
            failure.update({'initial_turn':state['turn'],'after_turn':game.turn})
            break
        operator.note(view,public_view(game),action)
    path=out/'reproduction.json.gz'
    with gzip.open(path,'xt',encoding='utf-8',compresslevel=3) as stream:
        stream.write(json.dumps({'source_seed':source['seed'],'variant':variant,'initial':state,
            'interventions':edits,'steps':steps,'failure':failure},separators=(',',':')))
    result={'source_seed':source['seed'],'variant':variant,'failure':failure,
            'source':source_identity(Path(__file__).resolve().parents[1]),
            'source_manifest':file_identity(bank/'manifest.json'),'trace':file_identity(path)}
    atomic_json(out/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',required=True);p.add_argument('--partial',required=True)
    p.add_argument('--out',required=True);main(p.parse_args())
