"""Real-game zero-adapter equivalence, complete trace replay and observation cost."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import hsbg_engine as engine
from hsbg.az.train import mechanism_obs as M
from hsbg.az.train.eval_policy import EvalProtocol, make_eval_policy
from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
from hsbg.evaluation.trace import DecisionRecorder, validate_trace
from hsbg.runtime.artifacts import atomic_json, file_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--games',type=int,default=2)
    args=parser.parse_args()
    torch.set_num_threads(1)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    results={};validation={}
    for name,path in [('baseline',args.checkpoint),('expanded',args.candidate)]:
        built=make_eval_policy(path,protocol=EvalProtocol(),device='cpu')
        schema=M.schema_of(built['model'])
        recorder=DecisionRecorder(out/(name+'.jsonl.gz'),identity={
            'name':name,'checkpoint':file_identity(path),'mechanism_schema':schema,
            'mechanism_schema_sha256':M.SCHEMA_SHA256 if schema else ''})
        cfg=EvalConfig(games=args.games,seed=1118000,collect_economy=True,
                       collect_action_hist=True,decision_recorder=recorder)
        try:
            results[name]=evaluate_policy(built['policy'],cfg)
            recorder.finish(results[name])
        except BaseException:
            recorder.abort();raise
        validation[name]=validate_trace(recorder.path,replay=1000000)
    for key in ('rank_sequence','seed_sequence','action_hist','end_turn_economy','decisions'):
        assert results['baseline'][key]==results['expanded'][key],key
    game=engine.Game(seed=1118009)
    costs={}
    for schema in ('',M.SCHEMA):
        M.observe(game,0,schema)
        started=time.perf_counter()
        for _ in range(1000):M.observe(game,0,schema)
        costs[schema or 'legacy']=1e6*(time.perf_counter()-started)/1000
    atomic_json(out/'validation.json',{'valid':True,'games_per_model':args.games,
        'real_trajectory_exact':True,'trace_validation':validation,
        'observation_microseconds_initial_state':costs,
        'note':'CPU engineering equivalence, not a strength evaluation'})
    print((out/'validation.json').read_text())


if __name__=='__main__':main()
