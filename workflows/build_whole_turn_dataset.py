"""Build complete public-controller trajectories with source-game isolation."""
import argparse,gzip,json,multiprocessing,shutil,sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import hsbg_engine as E
from hsbg.az.train import mechanism_obs as M
from hsbg.az.train.action_ban import ban_choice_options
from hsbg.az.train.common import engine_info
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity
from workflows.build_economic_dataset import observation
from workflows.probe_whole_turn_economy import fixture,run_operator,metrics,VARIANTS


def make_task(spec):
    source,variant,out=spec
    split='dev' if source['seed']%5==0 else 'train'
    state,edits=fixture(source['state'],variant)
    fresh=E.Game(seed=source['seed']+(40000000 if split=='dev' else 70000000))
    random_state=json.loads(fresh.state_json())
    state['seed']=random_state['seed'];state['rng']=random_state['rng']
    trace=run_operator(state);game=E.Game.from_json(json.dumps(state));rows=[]
    sequence=list(trace['steps'])
    # A controller-enforced boundary is not a genuine teacher EndTurn choice.
    if trace['end_intent']['reason']!='action_cap':
        sequence.append({'action':0,'reason':'natural_end','before':trace['final']})
    for step in sequence:
        if json.loads(game.state_json())!=step['before']:raise ValueError('demonstration replay differs')
        f,c,mask=observation(game)
        ban_choice_options(game.player_json(0),mask)
        action=step['action']
        if not mask[action]:raise ValueError('teacher violates policy contract mask')
        target=np.zeros(245,bool);target[action]=True
        rows.append((f,c,mask,target,source['seed'],variant,int(game.turn),step['reason']))
        if action:
            game.step(0,action)
            if json.loads(game.state_json())!=step['after'] or any(game.diagnostic_counts()):
                raise ValueError('native certificate differs')
    name=str(source['seed'])+'-'+variant
    trace_path=Path(out)/'traces'/(name+'.json.gz')
    with gzip.open(trace_path,'xt',encoding='utf-8',compresslevel=3) as stream:
        # Avoid millions of tiny gzip.write calls from json.dump's iterator.
        stream.write(json.dumps({'source_seed':source['seed'],'variant':variant,'split':split,
                                 'interventions':edits,**trace},separators=(',',':')))
    task={'seed':source['seed'],'kind':variant,'split':split,'initial':state,
          'interventions':edits,'teacher':metrics(trace),'trace':file_identity(trace_path),
          'labels':len(rows)}
    return task,rows


def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);(out/'traces').mkdir()
    bank=Path(args.bank);manifest=json.loads((bank/'manifest.json').read_text())
    if file_identity(bank/'states.jsonl.gz')!=manifest['data']:raise ValueError('source bank changed')
    groups={}
    with gzip.open(bank/'states.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            source=json.loads(line);p=source['state']['players'][0]
            if source['seat']!=0 or not source['state']['lobby_races']&(1<<6):continue
            if not 8<=source['turn']<=9 or p['health']<15 or p['pending'] or p['ended_turn']:continue
            if source['seed'] not in groups or source['turn']<groups[source['seed']]['turn']:
                groups[source['seed']]=source
    sources=[groups[k] for k in sorted(groups)]
    if args.limit:sources=sources[:args.limit]
    specs=[(s,v,str(out)) for s in sources for v in VARIANTS]
    all_rows=[];tasks=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as executor:
        for task,rows in executor.map(make_task,specs):tasks.append(task);all_rows.extend(rows)
    f,c,m,t,seeds,kinds,turns,reasons=zip(*all_rows)
    np.savez_compressed(out/'labels.npz',features=f,cards=c,masks=m,targets=t,seeds=seeds,
        dev=np.asarray(seeds)%5==0,kinds=kinds,turns=turns,reasons=reasons)
    with gzip.open(out/'tasks.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as stream:
        for task in tasks:stream.write(json.dumps(task,separators=(',',':'))+'\n')
    anchor_root=Path(args.anchors);anchor_manifest=json.loads((anchor_root/'manifest.json').read_text())
    if file_identity(anchor_root/'anchors.npz')!=anchor_manifest['files']['anchors.npz']:
        raise ValueError('natural anchor identity differs')
    shutil.copyfile(anchor_root/'anchors.npz',out/'anchors.npz')
    atomic_json(out/'manifest.json',{'schema':'hsai.whole-turn-process-dataset/v1','engine':engine_info(),
        'mechanism_schema_sha256':M.SCHEMA_SHA256,'source':source_identity(Path(__file__).resolve().parents[1]),
        'source_manifest':file_identity(bank/'manifest.json'),'anchor_source':file_identity(anchor_root/'manifest.json'),
        'source_seeds':[s['seed'] for s in sources],'source_games':len(sources),'tasks':len(tasks),'labels':len(all_rows),
        'training_sources':sum(s['seed']%5!=0 for s in sources),'development_sources':sum(s['seed']%5==0 for s in sources),
        'variants':VARIANTS,'action_budget':100,'source_split':'seed % 5 == 0 dev; no variants cross split',
        'scope':'complete heuristic behavior trajectories, native replay verified; NOT optimal-action certificates',
        'limitations':['Explicitly endowed components, not natural acquisition evidence',
            'Teacher remembers its own newly played minions; student is feedforward',
            'Action-cap EndTurn is excluded from imitation labels',
            'Single public controller trajectory per source/variant, no per-case outcome filtering'],
        'files':{n:file_identity(out/n) for n in ('labels.npz','anchors.npz','tasks.jsonl.gz')}})
    print(json.dumps({'sources':len(sources),'tasks':len(tasks),'labels':len(all_rows)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bank',required=True)
    p.add_argument('--anchors',required=True);p.add_argument('--out',required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int,default=0)
    main(p.parse_args())
