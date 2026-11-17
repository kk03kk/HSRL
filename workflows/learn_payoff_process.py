"""Learn multi-turn payoff operation inside a declared golden-engine context."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.conditional_economy import fork_conditional
from workflows.learn_cycle_process import run

def separate_assessment(model,tasks,device,path):
    return {'kinds':{},'tasks':len(tasks),'status':'not run here',
        'scope':'Closed-loop ability must be assessed by a separate sealed full-suffix job; label agreement is insufficient'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--mode',default='all',choices=('all',))
    p.add_argument('--updates',type=int,default=2000);p.add_argument('--lr',type=float,default=3e-6)
    p.add_argument('--anchor-kl',type=float,default=10.);p.add_argument('--per-kind-batch',type=int,default=32)
    p.add_argument('--seed',type=int,default=1203801);args=p.parse_args()
    run(args,expected_schema='hsai.payoff-process-dataset/v1',closed_loop_fn=separate_assessment,
        learning_tag='gold_payoff_learning',
        model_transform=lambda model,meta:fork_conditional(model,meta,mechanisms=True,gate_mode='gold_engine'))
