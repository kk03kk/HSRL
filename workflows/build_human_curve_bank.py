#!/usr/bin/env python3
"""以Firestone Train逐回合成对攻血分布校准完整模拟器快照；不导入真人隐藏状态。"""
from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def coefficient(turn):
    return 1.0 if turn<=7 else .9 if turn<=10 else .85 if turn==11 else .8 if turn==12 else .75 if turn==13 else .7


def references(humans,seed=20261103):
    """源游戏等权；固定1/5游戏只作战斗审计，不参与拟合。"""
    fit=defaultdict(lambda:defaultdict(list));audit=[]
    for row in humans:
        if row.get('split')!='train':raise ValueError('仅允许Firestone Train')
        if not row.get('board_detail') or row.get('is_ghost'):continue
        if row.get('stage') not in ('combat_start','recruitment_end_at_phase_switch'):continue
        key=str(row['game_id'])
        if int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest(),16)%5==0:
            audit.append(row)
        else:fit[int(row['turn'])][key].append(row)
    rng=random.Random(seed)
    result={}
    for turn,games in sorted(fit.items()):
        pairs=[]
        for game in sorted(games):
            row=rng.choice(games[game]);board=row['board_detail']
            a=sum(int(m['atk']) for m in board);h=sum(int(m['hp']) for m in board)
            if a<0 or h<=0:raise ValueError('真人攻血非法')
            pairs.append((a,h))
        result[turn]=sorted(pairs,key=lambda v:(sum(v),v))
    return result,audit


def allocate(values,target,minimum):
    """最大余数分配，保留总量及原相对比例，生命每个至少1。"""
    n=len(values)
    if not n:return []
    target=max(n*minimum,int(target));remaining=target-n*minimum
    weights=[max(0,float(v)-minimum) for v in values]
    if not sum(weights):weights=[1.0]*n
    exact=[remaining*w/sum(weights) for w in weights]
    result=[minimum+math.floor(v) for v in exact]
    for i in sorted(range(n),key=lambda i:(-(exact[i]-math.floor(exact[i])),i))[:target-sum(result)]:
        result[i]+=1
    if max(result)>2**31-1:raise ValueError('校准数值超出引擎i32范围')
    return result


def transform(frame,target,reference_turn,q):
    row=copy.deepcopy(frame);board=row['player']['board'];turn=int(row['turn'])
    original=(sum(m['atk'] for m in board),sum(m['hp'] for m in board))
    if not board:return row
    a,h=(round(v*coefficient(turn)) for v in target)
    attacks=allocate([m['atk'] for m in board],a,0)
    health=allocate([m['hp'] for m in board],h,1)
    ratio=sum(health)/max(1,original[1])
    for m,atk,hp in zip(board,attacks,health):
        old_max=m.get('max_hp',m['hp']);m['atk']=atk;m['hp']=hp
        if 'max_hp' in m:
            m['max_hp']=max(hp,round(old_max*ratio))
            if m['max_hp']>2**31-1:raise ValueError('max_hp溢出')
    row['calibration']=dict(reference_turn=reference_turn,quantile=q,coefficient=coefficient(turn),
        original_attack=original[0],original_health=original[1],target_attack=a,target_health=h)
    row['frame_sha256']=hashlib.sha256(json.dumps(row['player'],ensure_ascii=False).encode()).hexdigest()
    return row


def build(sources,human_path,out,split,*,seed=20261103,quota=32,min_games=8):
    from workflows.calibrate_opponent_bank import read_sources
    from workflows.compare_board_libraries import summarize
    from hsbg.az.train.personal import write_json,library_digest,fingerprint,sha256
    from hsbg.az.train.mirror_snapshots import write_snapshot_file,engine_stamp,MirrorSnapshotPool
    if quota<1 or min_games<2:raise ValueError('quota/min_games非法')
    out=Path(out)
    if out.exists():raise FileExistsError('保留已有库及失败证据，使用新目录')
    rows,parents=read_sources(sources,split)
    humans=[json.loads(line) for line in Path(human_path).read_text(encoding='utf-8').splitlines() if line.strip()]
    refs,audit=references(humans)
    reliable={t:v for t,v in refs.items() if len(v)>=min_games}
    if not reliable:raise ValueError('无足够真人Train源游戏')
    buckets=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for row in rows:
        buckets[(int(row['turn']),int(row['seat']))][row['source']][row['source_game_seed']].append(row)
    rng=random.Random(seed);frames=[];missing={};fallback={};empty=0
    for (turn,seat),at_key in sorted(buckets.items()):
        available=[t for t in reliable if t<=turn]
        rt=max(available) if available else None
        if rt!=turn:fallback[str(turn)]=rt
        if set(at_key)!=set(sources):missing[str((turn,seat))]=sorted(set(sources)-set(at_key))
        for source,games in sorted(at_key.items()):
            order=sorted(games);rng.shuffle(order)
            for j in range(quota):
                base=rng.choice(games[order[j%len(order)]])
                q=.05+.9*(j+.5)/quota
                if rt is not None and base['player']['board']:
                    target=reliable[rt][min(len(reliable[rt])-1,int(q*len(reliable[rt])))]
                    frame=transform(base,target,rt,q)
                else:
                    frame=copy.deepcopy(base);empty+=not bool(base['player']['board'])
                    frame['calibration']=dict(unchanged_reason='empty_board' if not base['player']['board'] else 'no_prior_reference')
                frames.append(frame)
    out.mkdir(parents=True)
    write_snapshot_file(str(out/'snap_human_curve.json.gz'),frames,seed=seed,engine=engine_stamp(),
        meta=dict(split=split,calibration='paired_attack_health_train_curve',quota_per_source=quota))
    summary=summarize([dict(turn=f['turn'],game_id=f"{f['source']}:{f['source_game_seed']}",board_detail=f['player']['board']) for f in frames])
    import numpy as np
    reference_stats={str(t):dict(games=len(pairs),reliable=len(pairs)>=min_games,
        attack=np.percentile([a for a,h in pairs],[25,50,75,90]).tolist(),
        health=np.percentile([h for a,h in pairs],[25,50,75,90]).tolist(),
        combined=np.percentile([a+h for a,h in pairs],[25,50,75,90]).tolist()) for t,pairs in refs.items()}
    report=dict(schema='hsrl.human_curve_bank/v1',split=split,seed=seed,frames=len(frames),
        human_train_sha256=sha256(human_path),fit_source_games={str(t):len(v) for t,v in refs.items()},
        human_reference_stats=reference_stats,percentiles=['P25','P50','P75','P90'],reference_quantile_range=[.05,.95],
        audit_source_games=len({r['game_id'] for r in audit}),quota_per_source_per_turn_seat=quota,
        source_digests={k:m['digest'] for k,m in parents.items()},
        source_games={s:len({r['source_game_seed'] for r in rows if r['source']==s}) for s in sources},
        target_coefficients={str(t):coefficient(t) for t in sorted({f['turn'] for f in frames})},
        missing_sources=missing,reference_turn_fallback=fallback,empty_board_frames=empty,turns=summary,
        limitations=['仅调整board atk/hp/max_hp，其余完整源状态及实体引用不变；这是合成对手，不是完整真人复现。',
            '成对攻血取真人Train游戏等权的P05-P95；按回合适度缓和后期，未严格复制全部真人分布。',
            '同一源帧可生成不同目标身材，衍生条目不等于新增独立源游戏；缺失来源与回合回退明确报告。',
            '空场保持空场；晚期参考不足沿用最近可靠回合，不复制低回合玩家状态；战斗压力另行审计。',
            '未同步改写附魔/光环/成长计数，后续触发可能改变牌面；不以身材匹配冒充实际战斗等效。'])
    write_json(out/'calibration.json',report)
    write_json(out/'manifest.json',dict(schema='hsrl.library/v1',split=split,driver='human_curve_full_snapshots',
        fingerprint=fingerprint(),digest=library_digest(out),parents=report['source_digests'],
        stats=MirrorSnapshotPool(str(out),sample='random').stats(),missing='engine_bot_fallback'))
    return report


def pressure_audit(sources,bank,human_path,out,split,*,seed=20261103,trials=16):
    """固定四个分位的源/校准配对探针；审计真人游戏未参与曲线拟合。"""
    import gzip
    import hsbg_engine as E
    from workflows.calibrate_opponent_bank import read_sources,human_references,combat_score
    from workflows.mirror_hard_gates import _drive_to_turn
    from workflows.run_learning_experiment import write
    rows,_=read_sources(sources,split)
    original={(r['source'],r['source_game_seed'],r['turn'],r['seat']):r for r in rows}
    humans=[json.loads(line) for line in Path(human_path).read_text(encoding='utf-8').splitlines() if line.strip()]
    _,audit=references(humans);refs=human_references(audit,seed)
    grouped=defaultdict(list)
    for path in Path(bank).glob('snap_*.json.gz'):
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            for row in json.load(stream)['frames']:
                if 6<=row['turn']<=16 and 'quantile' in row.get('calibration',{}):grouped[(row['turn'],row['source'])].append(row)
    state=json.loads(_drive_to_turn(E.Game(seed=63000000),1).state_json())
    for p in state['players']:
        p['hero']=p['deity']=None
        for k in ('powers','pending','trinkets','quests','rewards','secrets','effects'):p[k]=[]
    result=[]
    for (turn,source),bucket in sorted(grouped.items()):
        anchors=refs.get(turn,{}).get('anchors',[])
        if not anchors:continue
        state['turn']=turn
        selected=[]
        for q in (.125,.375,.625,.875):
            row=min(bucket,key=lambda r:(abs(r['calibration']['quantile']-q),r['source_game_seed'],r['seat']))
            before=original[(source,row['source_game_seed'],turn,row['seat'])]
            selected.append(dict(game=row['source_game_seed'],seat=row['seat'],quantile=row['calibration']['quantile'],
                original=combat_score(before,anchors,json.dumps(state),trials),
                calibrated=combat_score(row,anchors,json.dumps(state),trials)))
        result.append(dict(turn=turn,source=source,human_games=refs[turn]['games'],
            audit_game_ids=[r['game_id'] for r in anchors],pairs=selected))
    report=dict(schema='hsrl.human_curve_pressure/v1',results=result,
        note='源与校准使用相同参考及RNG；多次战斗不等于独立真人游戏；probe隐藏状态overlay限制沿用成熟探针，不宣称真人等效。')
    if not result:raise ValueError('没有可用的独立真人战斗审计参考')
    write(out,report)
    return report
