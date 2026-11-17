"""Teach a conditional economic policy while freezing fallback and shared critic."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from workflows.learn_cycle_process import run
from workflows.learn_whole_turn_process import make_closure
from hsbg.az.train.conditional_economy import fork_conditional


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--baseline',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--mode',default='all',choices=('all',))
    p.add_argument('--updates',type=int,default=1000);p.add_argument('--lr',type=float,default=3e-6)
    p.add_argument('--anchor-kl',type=float,default=10.);p.add_argument('--per-kind-batch',type=int,default=32)
    p.add_argument('--mechanisms',action='store_true')
    p.add_argument('--seed',type=int,default=1193801);args=p.parse_args()
    run(args,expected_schema='hsai.whole-turn-process-dataset/v1',closed_loop_fn=make_closure(args),
        learning_tag='conditional_economy_learning',
        model_transform=lambda model,meta:fork_conditional(model,meta,mechanisms=args.mechanisms))
