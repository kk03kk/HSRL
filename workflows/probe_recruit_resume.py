"""CPU compute-node check of real multiworker curriculum PPO and state restore."""
import argparse,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from hsbg.runtime.artifacts import atomic_json,file_identity
from workflows.audit_full_trainer import audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--bank',required=True)
    p.add_argument('--checkpoint',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    common=[sys.executable,'workflows/train_entity_sync.py','--device','cpu',
        '--opponent','bot','--behavior','net','--pipeline','1','--rust-collector','0','--lockstep','1',
        '--mirror-snaps','artifacts/libraries/arena-cal-v14-snap','--mirror-seats','7',
        '--mirror-p','1','--mirror-sample','p50band','--mirror-strict','0',
        '--recruit-start-bank',args.bank,'--recruit-start-p','1',
        '--games','8','--iterations','1','--minutes','10','--epochs','1',
        '--minibatch','128','--grad-accum','1','--lr','6e-6','--max-updates','300',
        '--gamma','1','--lam','.995','--entropy-coef','.002','--anchor-beta','.1',
        '--reset-anchor','0','--seed','4181901','--sims','0','--processes','2',
        '--gpus','0','--gpu-index-space','visible','--threads','1','--amp','off',
        '--leaf-amp','off','--snapshot-every','1','--keep-snapshots','3',
        '--max-actions-per-turn','100','--move-guard','1','--move-cap','2',
        '--move-repeat-limit','2','--forbid-cancel','1','--forbid-unfreeze','1',
        '--freeze-cap','1','--ban-trinkets','1']
    runs=[]
    for name,extra in [('first',['--checkpoint',args.checkpoint]),
                       ('resumed',['--checkpoint',str(out/'first/model.pt'),
                                   '--trainer-state',str(out/'first/trainer.pt')])]:
        dest=out/name;cmd=common+['--out',str(dest)]+extra
        atomic_json(out/(name+'-command.json'),cmd)
        with (out/(name+'.log')).open('x',encoding='utf-8') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=audit(dest,expected_version=len(runs)+1)
        atomic_json(out/(name+'-audit.json'),result);runs.append(result)
    first=torch.load(out/'first/trainer.pt',map_location='cpu',weights_only=False)
    resumed=torch.load(out/'resumed/trainer.pt',map_location='cpu',weights_only=False)
    if resumed['extra']['cum_decisions']<=first['extra']['cum_decisions']:
        raise ValueError('resumed decision count did not advance')
    if any(not torch.equal(v,resumed['anchor_model'][k]) for k,v in first['anchor_model'].items()):
        raise ValueError('resume changed the original anchor')
    if runs[1]['adam_step_min']<=runs[0]['adam_step_min']:
        raise ValueError('optimizer did not continue')
    atomic_json(out/'complete.json',{'passed':True,'runs':runs,
        'bank':file_identity(Path(args.bank)/'manifest.json'),
        'scope':'Real CPU multiworker PPO, full Trainer load, further update and preserved anchor; not uninterrupted-pipeline bit equivalence'})
    print(json.dumps({'passed':True,'decisions':[r['cum_decisions'] for r in runs]}))


if __name__=='__main__':main()
