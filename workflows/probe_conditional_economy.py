"""Actual-weight CPU preflight for conditional policy routing and PPO recovery."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from hsbg.az.train.conditional_economy import fork_conditional
from hsbg.az.train.model import load_checkpoint,save_checkpoint
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity,environment_identity
from workflows.learn_economic_process import tensors,forward


def main(args):
    torch.set_num_threads(1)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);root=Path(args.dataset)
    manifest=json.loads((root/'manifest.json').read_text())
    for name in ('labels.npz','anchors.npz'):
        if file_identity(root/name)!=manifest['files'][name]:raise ValueError('dataset identity differs')
    parent,meta=load_checkpoint(args.checkpoint);model=fork_conditional(parent,meta,mechanisms=args.mechanisms).eval()
    with np.load(root/'labels.npz',allow_pickle=False) as raw:labels=tensors(raw,model,'cpu')
    with np.load(root/'anchors.npz',allow_pickle=False) as raw:anchors=tensors(raw,model,'cpu')
    with torch.no_grad():
        active=model.gate(labels['features'],labels['cards'])
        anchor_active=model.gate(anchors['features'],anchors['cards'])
    if not bool(active.all()):raise ValueError('demonstration outside conditional gate')
    mix={k:torch.cat((labels[k][:16],anchors[k][:16])) for k in ('features','cards','masks')}
    parent_mix={**mix,'features':mix['features'][:,:getattr(parent,'input_floats',3227)]}
    with torch.no_grad():
        expected=forward(parent,parent_mix,np.arange(32));actual=forward(model,mix,np.arange(32))
    if not all(torch.equal(a,b) for a,b in zip(expected,actual)):raise ValueError('neutral fork differs')
    save_checkpoint(model,str(out/'neutral-model.pt'),meta={**meta,'research_probe_only':True})
    restored,_=load_checkpoint(str(out/'neutral-model.pt'))
    with torch.no_grad():roundtrip=forward(restored,mix,np.arange(32))
    if not all(torch.equal(a,b) for a,b in zip(actual,roundtrip)):raise ValueError('roundtrip differs')
    summary={'source':source_identity(Path(__file__).resolve().parents[1]),'environment':environment_identity(),
        'parent':file_identity(args.checkpoint),'dataset_manifest':file_identity(root/'manifest.json'),
        'gate':model.gate_spec,'label_rows':len(active),'active_label_rows':int(active.sum()),
        'anchor_rows':len(anchor_active),'active_anchor_rows':int(anchor_active.sum()),
        'neutral_exact':True,'roundtrip_exact':True,'checkpoint':file_identity(out/'neutral-model.pt')}
    summary['input_floats']=model.input_floats;summary['mechanism_schema']=model.mechanism_schema
    atomic_json(out/'preflight.json',summary)
    if args.ppo_probe:
        command=[sys.executable,'workflows/probe_recruit_resume.py','--out',str(out/'ppo'),
            '--bank','artifacts/runs/nus-recruit-curriculum-20261118-2/endowed',
            '--checkpoint',str(out/'neutral-model.pt')]
        atomic_json(out/'ppo-command.json',command)
        with (out/'ppo.log').open('x',encoding='utf-8') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        summary['ppo_complete']=file_identity(out/'ppo/complete.json')
    summary['passed']=True;atomic_json(out/'complete.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--out',required=True)
    p.add_argument('--ppo-probe',action='store_true');p.add_argument('--mechanisms',action='store_true')
    main(p.parse_args())
