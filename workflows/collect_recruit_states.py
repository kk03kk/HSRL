"""Collect fresh natural own-policy recruit states for development and teaching.

No examination traces are consumed. The source is a declared checkpoint and
training snapshot field; each state is captured once per turn before deciding.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from hsbg.az.train.eval_policy import EvalProtocol, make_eval_policy
from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
from hsbg.runtime.artifacts import atomic_json, file_identity


class StateBank:
    def __init__(self,path):
        self.stream=gzip.open(path,'xt',encoding='utf-8',compresslevel=3)
        self.n=0;self.seen=set();self.seed=None
    def begin_game(self,game,game_index,seed,seat):self.seed=int(seed)
    def before(self,game,seat):
        turn=int(game.turn);key=(self.seed,turn)
        if not 7<=turn<=12 or key in self.seen:return None
        own=json.loads(game.player_json(seat))
        if own['pending'] or own['ended_turn']:return None
        self.seen.add(key)
        self.stream.write(json.dumps({'seed':self.seed,'turn':turn,'seat':seat,
            'state':json.loads(game.state_json())},separators=(',',':'))+'\n')
        self.n+=1
        return None
    def after(self,*args,**kwargs):pass
    def end_game(self,*args,**kwargs):pass


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--games',type=int,default=64)
    p.add_argument('--seed',type=int,default=11800000)
    p.add_argument('--device',default='cuda')
    p.add_argument('--mirror-snaps',default='artifacts/libraries/arena-cal-v14-snap')
    args=p.parse_args()
    if not (args.seed>=11800000 and args.seed+args.games<=11810000):
        raise ValueError('use the separate training-source seed namespace')
    torch.set_num_threads(1)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    proto=EvalProtocol(mirror_snaps=args.mirror_snaps)
    built=make_eval_policy(args.checkpoint,protocol=proto,device=args.device)
    bank=StateBank(out/'states.jsonl.gz')
    try:
        result=evaluate_policy(built['policy'],EvalConfig(games=args.games,seed=args.seed,
            collect_economy=True,collect_action_hist=True,decision_recorder=bank,
            mirror_pool=MirrorSnapshotPool(args.mirror_snaps,sample='p50band'),
            mirror_strict=False,mirror_seats=7,mirror_p=1.0))
    finally:bank.stream.close()
    if result['illegal_actions'] or result['truncated_games']:
        raise RuntimeError('state-bank rollout failed game quality gates')
    atomic_json(out/'manifest.json',{'schema':'hsai.recruit-state-bank/v1',
        'checkpoint':file_identity(args.checkpoint),'checkpoint_path':args.checkpoint,
        'args':vars(args),'states':bank.n,'data':file_identity(out/'states.jsonl.gz'),
        'seeds':result['seed_sequence'],'ranks':result['rank_sequence'],
        'purpose':'training/development source, never a formal evaluation'})
    print(json.dumps({'states':bank.n,'games':args.games,'mean_rank':result['mean_rank']}))


if __name__=='__main__':main()
