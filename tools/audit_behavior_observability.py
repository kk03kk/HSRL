"""Measure exact observation aliasing in behavior labels, without model updates."""
import argparse,hashlib,json,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from hsbg.runtime.artifacts import atomic_json,file_identity


def audit(data,width):
    groups={}
    for i in range(len(data['seeds'])):
        digest=hashlib.sha256()
        for value,dtype in ((data['features'][i,:width],'<f4'),(data['cards'][i],'<i8'),
                            (data['masks'][i],np.bool_)):
            digest.update(np.asarray(value,dtype=dtype).tobytes())
        groups.setdefault(digest.digest(),[]).append(i)
    result={}
    for split in ('train','dev','all'):
        repeated=conflicts=minimum_errors=rows=0;examples=[]
        for ix in groups.values():
            ids=[i for i in ix if split=='all' or bool(data['dev'][i])==(split=='dev')]
            if not ids:continue
            rows+=len(ids)
            if len(ids)<2:continue
            repeated+=1;actions=Counter(int(data['targets'][i].argmax()) for i in ids)
            if len(actions)>1:
                conflicts+=1;minimum_errors+=len(ids)-max(actions.values())
                if len(examples)<5:
                    examples.append([{'source_seed':int(data['seeds'][i]),'kind':str(data['kinds'][i]),
                        'action':int(data['targets'][i].argmax()),
                        'reason':str(data['reasons'][i]) if 'reasons' in data else None} for i in ids[:12]])
        result[split]={'rows':rows,'repeated_observation_groups':repeated,
            'conflicting_action_groups':conflicts,'minimum_deterministic_label_errors':minimum_errors,
            'minimum_error_fraction':minimum_errors/max(1,rows),'examples':examples}
    result['train_dev_identical_observation_groups']=sum(any(data['dev'][i] for i in ix)
        and any(not data['dev'][i] for i in ix) for ix in groups.values())
    return result


def run(dataset,out):
    root=Path(dataset);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('schema')!='hsai.whole-turn-process-dataset/v1':raise ValueError('wrong dataset schema')
    if file_identity(root/'labels.npz')!=manifest['files']['labels.npz']:raise ValueError('labels changed')
    with np.load(root/'labels.npz',allow_pickle=False) as archive:
        data={k:archive[k] for k in archive.files}
    if data['features'].shape[1]!=3282 or data['cards'].shape[1]!=124 or data['masks'].shape[1]!=245:
        raise ValueError('observation or action layout differs')
    if not np.isfinite(data['features']).all():raise ValueError('nonfinite observation')
    if not (data['targets'].sum(1)==1).all():raise ValueError('one behavior action required')
    if not (data['targets']<=data['masks']).all():raise ValueError('illegal label')
    if not np.array_equal(data['dev'],data['seeds']%5==0):raise ValueError('source split changed')
    result={'schema':'hsai.behavior-observability/v1','source_manifest':file_identity(root/'manifest.json'),
        'legacy3227':audit(data,3227),'mechanism3282':audit(data,3282),
        'scope':'Exact float32 observation+card+mask conflicts only; no-conflict does not prove sufficiency or learnability',
        'reason_counts':dict(Counter(map(str,data.get('reasons',[]))))}
    atomic_json(out/'result.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True)
    p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(run(a.dataset,a.out)))
