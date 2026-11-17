"""Diagnostic only: test learned economic ordering when disposable slots must be freed."""
import argparse,copy,gzip,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import hsbg_engine as engine
from hsbg.az.train.economic_tasks import own,debug,card_ids
from hsbg.az.train.model import load_checkpoint
from hsbg.runtime.artifacts import atomic_json,file_identity
from workflows.learn_economic_process import closed_loop


def fill_board(task):
    result=copy.deepcopy(task)
    game=engine.Game.from_json(json.dumps(task['initial']))
    before=own(game)
    protected={m['eid'] for m in before['board']}
    existing={m['card'] for m in before['board']}
    edits=[]
    # Vanilla bodies have no recruit effect. Tokens are explicit diagnostic
    # interventions, not claims about natural availability or training labels.
    for card in ('BGS_061t','BG29_864t','BG_CS2_200','BG30_MagicItem_442t',
                 'BG_CFM_315t','BG_BRM_006t','BGS_115t','BG_EX1_556t'):
        if len(own(game)['board'])==7:break
        if card_ids()[card] in existing:continue
        debug(game,'add_minion',zone='board',card=card,atk=1,hp=1)
        edits.append({'cmd':'add_minion','zone':'board','card':card,'atk':1,'hp':1})
    after=own(game)
    if len(after['board'])!=7:raise RuntimeError('could not create declared full-board diagnostic')
    liquid={m['eid'] for m in after['board']}-protected
    result.update(initial=json.loads(game.state_json()),protected_eids=sorted(protected),
                  liquid_eids=sorted(liquid),objective=task['objective']+len(liquid),
                  diagnostic_interventions=edits)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--checkpoint',action='append',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();torch.set_num_threads(1)
    root=Path(args.dataset);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((root/'manifest.json').read_text())
    if file_identity(root/'tasks.jsonl.gz')!=manifest['files']['tasks.jsonl.gz']:raise ValueError('task identity mismatch')
    with gzip.open(root/'tasks.jsonl.gz','rt',encoding='utf-8') as stream:
        tasks=[fill_board(t) for line in stream if (t:=json.loads(line))['split']=='dev']
    with gzip.open(out/'tasks.jsonl.gz','xt',encoding='utf-8') as stream:
        for task in tasks:stream.write(json.dumps(task)+'\n')
    results={}
    for spec in args.checkpoint:
        name,path=spec.split('=',1)
        if not name.replace('_','').isalnum() or name in results:raise ValueError('invalid/duplicate model name')
        model,_=load_checkpoint(path,device=args.device)
        result=closed_loop(model,tasks,args.device,out/(name+'.jsonl.gz'))
        results[name]={'checkpoint':file_identity(path),'metrics':result}
        atomic_json(out/(name+'.json'),results[name])
        print(json.dumps({'name':name,'summary':result['all']}),flush=True)
        del model
    atomic_json(out/'complete.json',{'models':results,'dataset':file_identity(root/'manifest.json'),
        'task_file':file_identity(out/'tasks.jsonl.gz'),'scope':'Synthetic full-board diagnostic; not formal games or BC labels'})


if __name__=='__main__':main()
