"""Build verified local economic-process labels from independent recruit states."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import hsbg_engine as engine
from hsbg.az.train import mechanism_obs as M
from hsbg.az.train.economic_tasks import fixture,solve,own,wealth_remaining,score
from hsbg.runtime.artifacts import atomic_json,file_identity


def observation(game,seat=0):
    f,c=M.observe(game,seat,M.SCHEMA)
    features=np.frombuffer(f,dtype='<f4').copy()
    mask=np.frombuffer(game.legal_mask(seat),dtype=np.uint8).astype(bool)
    mask[7]=False
    from hsbg.env.obs_std import GLOBAL_FIELD_INDEX
    if features[GLOBAL_FIELD_INDEX['frozen']]>.5:mask[2]=False
    return features,np.frombuffer(c,dtype='<i4').astype(np.int16),mask


def build_task(arg):
    source,kind,bounties,variant=arg
    seat=source['seat'];seed=source['seed']
    game,edits=fixture(source['state'],kind=kind,bounties=bounties,
                       gold=3+variant%6,seed=seed+variant,seat=seat)
    initial=json.loads(game.state_json())
    original_eids={m['eid'] for m in own(game,seat)['board']}
    rows=[];certificate=[]
    try:
        while wealth_remaining(game,seat):
            depth=bounties+2-len(rows)
            plan=solve(game,seat=seat,max_depth=depth,original_eids=original_eids)
            # A label must be robust to unseen RNG, without changing observation.
            accepted=set(plan['optimal_actions']);random_checks=[]
            for salt in (17,91):
                changed=json.loads(game.state_json())
                changed['rng']=json.loads(engine.Game(seed=seed+salt).state_json())['rng']
                alternate=engine.Game.from_json(json.dumps(changed))
                assert M.observe(alternate,seat,M.SCHEMA)==M.observe(game,seat,M.SCHEMA)
                other=solve(alternate,seat=seat,max_depth=depth,original_eids=original_eids)
                accepted.intersection_update(other['optimal_actions'])
                random_checks.append(other['value'])
            if not accepted:raise ValueError('RNG-dependent oracle label')
            f,c,mask=observation(game,seat)
            targets=np.zeros(245,bool);targets[list(accepted)]=True
            assert (targets<=mask).all()
            action=min(accepted)
            rows.append((f,c,mask,targets,seed,kind))
            before=own(game,seat)
            game.step(seat,action)
            certificate.append({'action':action,'optimal_actions':sorted(accepted),
                'objective':plan['value'],'alternate_rng_objectives':random_checks,
                'gold_before':before['gold'],'gold_after':own(game,seat)['gold']})
            assert original_eids<={m['eid'] for m in own(game,seat)['board']}
        actual=score(game,original_eids,seat)
        assert abs(actual-certificate[-1]['objective'])<1e-6
        assert not any(game.diagnostic_counts())
    except (ValueError,AssertionError,RuntimeError) as error:
        return {'accepted':False,'seed':seed,'kind':kind,'reason':str(error)}
    return {'accepted':True,'seed':seed,'turn':source['turn'],'kind':kind,'bounties':bounties,
            'source_sha256':hashlib.sha256(json.dumps(source['state'],sort_keys=True).encode()).hexdigest(),
            'initial':initial,'interventions':edits,'certificate':certificate,
            'final_gold':own(game,seat)['gold'],'objective':actual,'rows':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--max-states',type=int,default=96)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    bank=Path(args.bank)
    manifest=json.loads((bank/'manifest.json').read_text())
    assert file_identity(bank/'states.jsonl.gz')==manifest['data']
    with gzip.open(bank/'states.jsonl.gz','rt',encoding='utf-8') as f:sources=[json.loads(x) for x in f]
    # One representative turn per source game; partition all variants by source seed.
    groups={}
    for row in sources:
        if not int(row['state']['lobby_races']) & (1 << 6):continue
        seed=row['seed']
        if seed not in groups or abs(row['turn']-9)<abs(groups[seed]['turn']-9):groups[seed]=row
    chosen=[groups[k] for k in sorted(groups)][:args.max_states]
    kinds=('order_privateer','buy_privateer','redundant_privateer')
    tasks=[(s,kind,n,i) for i,s in enumerate(chosen) for kind in kinds for n in (2,3)]
    allrows=[];rejected=[];accepted=0
    with gzip.open(out/'tasks.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as stream:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for task in pool.map(build_task,tasks):
                if not task['accepted']:
                    rejected.append(task);continue
                allrows.extend(task.pop('rows'));accepted+=1
                task['split']='dev' if task['seed']%5==0 else 'train'
                stream.write(json.dumps(task,separators=(',',':'))+'\n')
    if accepted<6 or not allrows:raise RuntimeError('insufficient verified teaching tasks')
    f,c,m,t,seeds,names=zip(*allrows)
    np.savez_compressed(out/'labels.npz',features=np.stack(f),cards=np.stack(c),
        masks=np.stack(m),targets=np.stack(t),seeds=np.array(seeds,np.int64),
        dev=np.array([s%5==0 for s in seeds],bool),kinds=np.array(names))
    # Natural, unedited states anchor the student's original policy. Exclude dev
    # source games from training anchors as well as supervised labels.
    anchors=[observation(engine.Game.from_json(json.dumps(s['state'])),s['seat'])
             for s in sources if s['seed']%5!=0]
    # Add early-game states from an independent bot rollout distribution; these
    # are KL anchors evaluated by the parent policy, never bot action labels.
    for seed in range(11820000,11820016):
        game=engine.Game(seed=seed);seen=set()
        while game.turn<=6 and not game.is_over():
            for seat in range(game.num_players):
                if not game.needs_action(seat):continue
                if seat==0 and game.turn not in seen:
                    anchors.append(observation(game,0));seen.add(game.turn)
                game.step(seat,game.bot_action(seat))
    af,ac,am=zip(*anchors)
    np.savez_compressed(out/'anchors.npz',features=np.stack(af),cards=np.stack(ac),masks=np.stack(am))
    atomic_json(out/'manifest.json',{'schema':'hsai.economic-process-dataset/v1',
        'bank_manifest':file_identity(bank/'manifest.json'),'mechanism_schema':M.SCHEMA,
        'mechanism_schema_sha256':M.SCHEMA_SHA256,'args':vars(args),'tasks':accepted,
        'rows':len(allrows),'source_games':len(chosen),'anchors':len(anchors),
        'rejected':rejected,'split_rule':'seed % 5 == 0 is dev; source games never cross splits',
        'scope':'Synthetic privateer/wealthy-bounty local tasks, not full-game optimal actions',
        'files':{name:file_identity(out/name) for name in ('tasks.jsonl.gz','labels.npz','anchors.npz')}})
    print(json.dumps({'tasks':accepted,'rows':len(allrows),'rejected':len(rejected),'anchors':len(anchors)}))


if __name__=='__main__':main()
