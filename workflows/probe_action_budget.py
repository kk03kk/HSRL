#!/usr/bin/env python3
"""固定模型/Train库比较40与80动作预算；保留逐回合资源、动作和成长，不更新权重。"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def summarize_records(records,mean=None,std=None):
    import numpy as np
    from hsbg.env import obs_std as O
    from hsbg.env.action_space_std import segment_of
    turns=defaultdict(list)
    for r in records:turns[(int(r['game_id']),int(r['turn']))].append(r)
    grouped=defaultdict(list)
    for (_,turn),rows in turns.items():
        f=np.asarray(rows[-1]['features'])
        if mean is not None:f=f*(std if std is not None else 1)+mean
        gold=float(f[O.SLICE_GLOBAL.start+O.GLOBAL_FIELD_INDEX['gold']])*10
        board=f[O.SLICE_BOARD].reshape(O.N_BOARD,O.MINION_F)
        grouped[turn].append(dict(decisions=len(rows),forced=any(r.get('forced') for r in rows),
            gold=gold,tier=float(f[O.SLICE_GLOBAL.start+O.GLOBAL_FIELD_INDEX['tier']])*6,
            board_strength=float((board[:,O.MINION_ATK]+board[:,O.MINION_HP]).sum())*10,
            actions=Counter(str(segment_of(int(r['action']))) for r in rows if not r.get('forced'))))
    result={}
    for turn,rows in sorted(grouped.items()):
        forced=[r for r in rows if r['forced']]
        actions=Counter()
        for r in rows:actions.update(r['actions'])
        result[str(turn)]=dict(learner_turns=len(rows),forced_turns=len(forced),
            forced_turn_fraction=len(forced)/len(rows),actions=dict(actions),
            decisions_p50=float(np.median([r['decisions'] for r in rows])),
            end_gold_p50=float(np.median([r['gold'] for r in rows])),
            forced_gold_p50=float(np.median([r['gold'] for r in forced])) if forced else None,
            tier_p50=float(np.median([r['tier'] for r in rows])),
            board_strength_p50=float(np.median([r['board_strength'] for r in rows])))
    return result


def run(checkpoint,library,out,device='cuda:1',games=64):
    from hsbg.az.train.model import load_checkpoint
    from hsbg.az.train.batch_eval import BatchLeafServer,BatchPPOEvaluator
    from hsbg.az.train.ppo import CollectConfig
    from hsbg.az.train.parallel_collect import collect_ppo_lockstep
    from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
    from hsbg.az.train.collect_diagnostics import require_valid_mirror_collect
    from hsbg.az.train.personal import checked_library
    from workflows.run_learning_experiment import write,digest
    import numpy as np
    manifest=checked_library(library)
    if manifest['split']!='train':raise ValueError('动作诊断只使用Train')
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    model,meta=load_checkpoint(checkpoint,device=device)
    server=BatchLeafServer(model,device=device,max_wait_ms=0,amp='bf16',fused_act=True,
        norm_mean=meta.get('norm_mean'),norm_std=meta.get('norm_std'))
    results={}
    try:
        for cap in (40,80):
            cfg=CollectConfig(games=games,seed=62000000,max_actions_per_turn=cap,opponent='bot',behavior='net',lockstep=True)
            start=time.monotonic()
            records,collect=collect_ppo_lockstep(model,cfg,eval_fn=BatchPPOEvaluator(server),device=device,
                seed=62000000,games=games,gamemode='bot',norm_mean=meta.get('norm_mean'),norm_std=meta.get('norm_std'),
                mirror_pool=MirrorSnapshotPool(str(library),sample='random'),mirror_seats_count=7,mirror_p=1,
                mirror_seat_random=True)
            require_valid_mirror_collect(collect,str(out),iteration=cap,config=vars(cfg))
            results[str(cap)]=dict(seconds=time.monotonic()-start,decisions=len(records),
                mean_rank=float(np.mean([r['rank'] for r in collect['games_meta']])),
                collect={k:collect.get(k) for k in ('illegal_actions','truncated_games','mirror_errors','mirror_rejected_by_reason','forced_end_turns')},
                turns=summarize_records(records,meta.get('norm_mean'),meta.get('norm_std')))
            write(out/'progress.json',results)
    finally:server.close()
    report=dict(schema='hsrl.action_budget/v1',checkpoint=digest(checkpoint),library=manifest['digest'],
        games=games,seed=62000000,results=results,ppo_updates=0,
        note='相同权重/初始种子，动作后续分支不相同；40与80名次不能冒充模型进步。末条动作前观测用于资源与场面统计，强制结束不算策略采样。')
    write(out/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--library',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda:1');p.add_argument('--games',type=int,default=64)
    a=p.parse_args();run(a.checkpoint,a.library,a.out,a.device,a.games)
