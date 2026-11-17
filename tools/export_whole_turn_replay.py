"""Export training-only complete-turn labels through the existing PPO auxiliary CE interface."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from hsbg.az.train.board_problems import encode_obs,load_problem_pool
from hsbg.runtime.artifacts import atomic_json,file_identity


def export(dataset,out):
    root=Path(dataset);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['schema']!='hsai.whole-turn-process-dataset/v1':raise ValueError('not a complete-turn dataset')
    if file_identity(root/'labels.npz')!=manifest['files']['labels.npz']:raise ValueError('label identity differs')
    data=np.load(root/'labels.npz',allow_pickle=False)
    if not np.array_equal(data['dev'],data['seeds']%5==0):raise ValueError('source split is inconsistent')
    indices=np.flatnonzero(~data['dev'])
    if not len(indices):raise ValueError('empty training split')
    target=data['targets'][indices]
    if not (target.sum(1)==1).all() or not (target<=data['masks'][indices]).all():
        raise ValueError('one legal teacher action per row required')
    path=out/'process.jsonl'
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        for i in indices:
            action=int(data['targets'][i].argmax())
            row={'schema':'hsai.whole-turn-aux-row/v1','source_seed':int(data['seeds'][i]),
                'turn':int(data['turns'][i]),'tags':['synthetic_public_controller',str(data['kinds'][i])],
                'obs':encode_obs(data['features'][i,:3227],data['cards'][i],data['masks'][i]),
                'policy_action':action,'ref':{'actions':[action],'weights':[1.]},
                'label':{'available':False,'survived':False,
                         'note':'Imitation action only; no t+3 or strength outcome label'}}
            stream.write(json.dumps(row,separators=(',',':'))+'\n')
    pool=load_problem_pool(str(path))
    if len(pool)!=len(indices) or not pool.bc_valid.all() or pool.aux_valid.any():
        raise ValueError('existing PPO loader lost labels or fabricated auxiliary outcomes')
    if not np.array_equal(pool.ref_actions[:,0],target.argmax(1)):
        raise ValueError('PPO reference actions differ')
    result={'schema':'hsai.whole-turn-replay/v1','valid':True,'rows':len(indices),
        'source_manifest':file_identity(root/'manifest.json'),'data':file_identity(path),
        'source_seeds':sorted(set(map(int,data['seeds'][indices]))),'observation':'legacy3227/124',
        'scope':'training-only off-policy behavior CE, separate from on-policy PPO surrogate',
        'outcome_auxiliary_labels':False}
    atomic_json(out/'manifest.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True)
    p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(export(a.dataset,a.out)))
