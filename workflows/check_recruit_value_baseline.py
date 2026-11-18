"""Cross-fit a pre-action control variate for the existing matched terminal policy gradients."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.model import load_checkpoint
from hsbg.runtime.artifacts import atomic_json,file_identity
from tools.assess_investment_timing import UTILITY
from workflows.group_relative_recruit import checkpoint,output,TEMPERATURE
from workflows.probe_investment_timing import MODEL


ROOT=Path('artifacts/runs/nus-recruit-control-variate-20261118-1')


def grab(arm,it):
    features=[];cards=[];masks=[];labels=[]
    for shard in range(8):
        p=output(arm,it)/f'rollout-{shard:02}'
        for row in json.loads((p/'groups.json').read_text()):
            lengths=row['actions']
            if not sum(lengths):
                continue
            with np.load(p/f"group-{row['i']}.npz") as batch:
                f,c,m,a,adv,lp=[batch[k] for k in ('f','c','m','a','adv','logp')]
            offset=0
            for replica,n in enumerate(lengths):
                if n:
                    for ix in sorted({0,(n-1)//3,2*(n-1)//3,n-1}):
                        k=ix+offset
                        features.append(f[k]);cards.append(c[k]);masks.append(m[k])
                        labels.append((row['seed'],replica,ix,int(a[k]),float(adv[k]),float(lp[k])))
                offset+=n
            if offset!=len(f):
                raise ValueError('archive choice offsets differ')
    return np.array(features),np.array(cards),np.array(masks),np.array(labels)


def extract(arm,it):
    d=ROOT/f'part-{arm}-{it}';d.mkdir(parents=True,exist_ok=False)
    f,c,m,labels=grab(arm,it)
    torch.set_num_threads(4)
    parent,meta=load_checkpoint(MODEL,'cpu');parent.eval()
    behavior=parent if not it else load_checkpoint(checkpoint(arm,it),'cpu')[0].eval()
    moments=[]
    with torch.inference_mode():
        for k in range(0,len(f),128):
            x=torch.from_numpy(f[k:k+128]);ci=torch.from_numpy(c[k:k+128].astype(np.int64))
            mask=torch.from_numpy(np.unpackbits(m[k:k+128],axis=1)[:,:245].astype(bool))
            if meta.get('norm_mean') is not None:
                x=(x-torch.as_tensor(meta['norm_mean']))/torch.as_tensor(meta['norm_std']).clamp_min(1e-6)
            logits,rank=parent(x,ci,mask)[:2]
            if it:
                logits=behavior(x,ci,mask)[0]
            prob=torch.softmax(logits/TEMPERATURE,1)
            value=torch.softmax(rank,1)@torch.as_tensor(UTILITY,dtype=rank.dtype)
            acts=labels[k:k+len(x),3].astype(int)
            pa=prob[np.arange(len(x)),acts]
            if np.max(np.abs(pa.log().numpy()-labels[k:k+len(x),5]))>.004:
                raise ValueError('behavior does not match rollout')
            w=1-2*pa+prob.square().sum(1)
            moments.extend(zip(value.tolist(),w.tolist()))
    rows=[];root=None
    for (seed,rep,depth,_,adv,_),(v,w) in zip(labels,moments):
        if depth==0:
            root=v
        rows.append((int(seed),int(rep),int(depth),adv,v-root,max(0.,w)))
    atomic_json(d/'moments.json',dict(behavior=file_identity(checkpoint(arm,it)),rows=rows))


def assess():
    arms=[np.array([r for it in range(4) for r in json.loads((ROOT/f'part-{a}-{it}'/'moments.json').read_text())['rows']])
          for a in (0,1)]
    result=[]
    for a,test in enumerate(arms):
        train=arms[1-a]
        y,z,w=train[:,3:].T
        beta=float((w*y*z).sum()/max(1e-8,(w*z*z).sum()))
        y,z,w=test[:,3:].T
        seeds=np.unique(test[:,0])
        per=np.array([[v[test[:,0]==s].sum() for v in (w*(y-beta*z)**2,w*y*y)] for s in seeds])
        draw=np.random.default_rng(243642).integers(len(seeds),size=(5000,len(seeds)))
        samples=per[draw].sum(1)
        ratio=per.sum(0)
        result.append(dict(arm=a,beta=beta,choices_sampled=len(test),sources=len(seeds),
                           ratio=ratio[0]/ratio[1],ci95=np.quantile(samples[:,0]/samples[:,1],[.025,.975]).tolist()))
    atomic_json(ROOT/'assessment.json',dict(arms=result,advance=all(r['ci95'][1]<.90 for r in result)))


if __name__=='__main__':
    if sys.argv[1]=='extract':
        i=int(sys.argv[2]);extract(i//4,i%4)
    elif sys.argv[1]=='assess':
        assess()
    else:
        raise ValueError('unknown stage')
