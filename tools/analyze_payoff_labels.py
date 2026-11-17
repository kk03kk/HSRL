"""Describe fixed-endpoint imitation errors; no updates or optimality claims."""
import argparse,json,sys
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import hsbg_engine as E
from hsbg.env import obs_std as O
from hsbg.az.train.common import group_of
from hsbg.az.train.model import load_checkpoint
from hsbg.az.train.process_learning import log_policy
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity,environment_identity
from workflows.learn_economic_process import tensors,forward

def action_card(action,cards):
    if 8<=action<16:index=O.SLICE_CARD_SHOP.start+action-8
    elif 16<=action<23:index=O.SLICE_CARD_BOARD.start+action-16
    elif 30<=action<46:index=O.SLICE_CARD_CHOICES.start+action-30
    elif 46<=action<126:index=O.SLICE_CARD_HAND.start+(action-46)//8
    else:return ''
    return E.card_info(int(cards[index]))[0] if cards[index] else ''

@torch.no_grad()
def main(args):
    torch.set_num_threads(1);root=Path(args.dataset);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    plan=json.loads(Path(args.plan).read_text())
    identity=file_identity(args.checkpoint)
    if identity['sha256']!=plan['models'][args.policy]['sha256']:raise ValueError('fixed endpoint changed')
    manifest=json.loads((root/'manifest.json').read_text())
    if file_identity(root/'labels.npz')!=manifest['files']['labels.npz']:raise ValueError('labels changed')
    with np.load(root/'labels.npz',allow_pickle=False) as raw:
        ix=np.flatnonzero(raw['dev']);data={k:raw[k][ix] for k in raw.files}
    model,meta=load_checkpoint(args.checkpoint,device=args.device)
    batch=tensors(data,model,args.device);pred=[]
    for offset in range(0,len(ix),128):
        rows=np.arange(offset,min(offset+128,len(ix)))
        logits,_=forward(model,batch,rows)
        pred.extend(log_policy(logits,batch['masks'][rows]).argmax(-1).cpu().tolist())
    pred=np.array(pred);target=data['targets'].argmax(-1)
    if not (data['targets'].sum(-1)==1).all():raise ValueError('expected single demonstrator action')
    grouped=defaultdict(list);confusion=Counter()
    for i,(a,b) in enumerate(zip(target,pred)):
        reason=str(data['reasons'][i]);source=int(data['cards'][i,4])
        pending=E.card_info(source)[0] if source else 'none'
        for key in ('all','reason/'+reason,'phase/'+str(data['kinds'][i])):
            grouped[key].append(i)
        if reason=='pending':grouped['pending_source/'+pending].append(i)
        if a!=b:
            confusion[(group_of(int(a)),action_card(int(a),data['cards'][i]),
                       group_of(int(b)),action_card(int(b),data['cards'][i]))]+=1
    summaries={}
    for key,rows in grouped.items():
        r=np.array(rows);summaries[key]={'rows':len(r),'sources':len(set(data['seeds'][r].tolist())),
            'agreement':float(np.mean(target[r]==pred[r])),'disagreements':int(np.sum(target[r]!=pred[r]))}
    np.savez_compressed(out/'predictions.npz',original_row=ix,prediction=pred,target=target,seeds=data['seeds'])
    result={'model':identity,'dataset_manifest':file_identity(root/'manifest.json'),
        'source':source_identity(Path(__file__).resolve().parents[1]),'environment':environment_identity(),
        'groups':summaries,'confusions':[{'teacher_group':k[0],'teacher_card':k[1],
            'student_group':k[2],'student_card':k[3],'count':n} for k,n in confusion.most_common()],
        'predictions':file_identity(out/'predictions.npz'),
        'scope':'Fixed-model disagreement on sealed teacher-visited development states; not student rollout errors or optimal-action labels',
        'limitations':['Correlated actions within 49 sources','Card keys omit position and stats; identical keys can still have different consequences']}
    atomic_json(out/'complete.json',result)
    print(json.dumps({'all':summaries['all'],'reasons':{k:v for k,v in summaries.items() if k.startswith('reason/')}}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--plan',default='config/research/payoff-student-suffix-20261118.json')
    p.add_argument('--policy',default='payoff_student')
    main(p.parse_args())
