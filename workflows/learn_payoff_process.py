"""Learn multi-turn payoff operation inside a declared golden-engine context."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.conditional_economy import fork_conditional
from workflows.learn_cycle_process import run
from hsbg.runtime.artifacts import file_identity

def validate_dataset(root,checkpoint):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('probe') or (manifest['source_games'],manifest['training_sources'],manifest['development_sources'])!=(256,207,49):
        raise ValueError('requires the complete predeclared source split, not a probe/subset')
    if manifest['run']!=file_identity(root/'run.json'):raise ValueError('dataset run changed')
    identity=json.loads((root/'run.json').read_text())
    if identity['parent']!=file_identity(checkpoint):raise ValueError('demonstrator fallback parent differs')
    from hsbg.az.train.conditional_economy import current_gate_spec
    from hsbg.az.train.common import engine_info
    from hsbg.az.train.eval_policy import TRAINING_CONTRACT
    if identity['gate']!=current_gate_spec('gold_engine') or identity['engine']!=engine_info():
        raise ValueError('dataset gate or engine differs')
    if identity['action_contract']!=dict(TRAINING_CONTRACT):raise ValueError('dataset action contract differs')

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
    validate_dataset(args.dataset,args.checkpoint)
    run(args,expected_schema='hsai.payoff-process-dataset/v1',closed_loop_fn=separate_assessment,
        learning_tag='gold_payoff_learning',
        model_transform=lambda model,meta:fork_conditional(model,meta,mechanisms=True,gate_mode='gold_engine'))
