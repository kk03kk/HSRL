"""新版完整Bot EOT按总攻血P50校准；复用原始数据，拒绝旧指纹和混合种子。"""
import argparse, copy, gzip, hashlib, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median, mean
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from workflows.build_human_curve_bank import allocate
from workflows.build_3702_bot_bank import read_frames, capture, totals
from workflows.report_firestone_learning import strength_quantiles

DRIVER='bot_firestone_p50_3702'

def scale_frame(frame, target_total, factor):
    row=copy.deepcopy(frame); board=row['player']['board']; old=sum(totals(row))
    if old<=0 or not board: raise ValueError('空或无生命战团不能校准')
    # 各帧保留相对强弱，限制极端尾部；不是给每帧同样总量。
    total=max(len(board),round(min(target_total*3,max(target_total*.25,old*factor))))
    attack=sum(m['atk'] for m in board); hp=sum(m['hp'] for m in board)
    wanted_attack=min(total-len(board),round(total*attack/(attack+hp)))
    aa=allocate([m['atk'] for m in board],wanted_attack,0)
    hh=allocate([m['hp'] for m in board],total-wanted_attack,1)
    for m,a,h in zip(board,aa,hh):
        if m['hp']<=0:raise ValueError('死亡随从进入战团')
        fa=a/m['atk'] if m['atk'] else 0; fh=h/m['hp']
        m['max_hp']=max(h,round(m.get('max_hp',m['hp'])*fh))
        # 历史峰值不低于当前牌面，避免缩放留下不一致的生命峰值。
        if 'peak_atk' in m:m['peak_atk']=max(a,round(m['peak_atk']*fa))
        if 'peak_hp' in m:m['peak_hp']=max(h,round(m['peak_hp']*fh))
        m['atk'],m['hp']=a,h
        if max(a,h,m['max_hp'])>2**31-1:raise ValueError('i32溢出')
    return row

def calibrated(rows,target):
    if not math.isfinite(target) or target<=0:raise ValueError('P50目标必须有限正数')
    base=median(sum(totals(f)) for f in rows)
    if base<=0:raise ValueError('源P50必须为正数')
    result=[scale_frame(f,target,target/base) for f in rows]
    observed=median(sum(totals(f)) for f in result)
    if abs(observed-target)>max(1,.02*target):raise ValueError('实际抽样P50校准超差')
    return result

def select_rows(paths, engine, threshold, minimum_turns=15, limit=1024, expected_rules=None, expected_source_hash=None):
    candidates=defaultdict(dict);dropped=Counter()
    for frame in read_frames(paths,engine,None,expected_rules,expected_source_hash):
        from hsbg.az.train.mirror_snapshots import STAGE_EOT
        if frame.get('stage')!=STAGE_EOT or not frame['player'].get('ended_turn'):
            raise ValueError('源帧不是已结算回合末、开战前状态')
        turn=int(frame['turn']); board=frame['player']['board']
        if not 1<=turn<=19:continue
        if len(board)<(5 if turn>=10 else 1):dropped[turn]+=1;continue
        seed=frame['source_game_seed'];seat=frame['seat']
        rank=hashlib.sha256(f'{turn}:{seed}:{seat}'.encode()).hexdigest()
        previous=candidates[turn].get(seed)
        if previous is None or rank<previous[0]:candidates[turn][seed]=(rank,frame)
    last=0
    while last<19 and len(candidates[last+1])>=threshold:last+=1
    if last<minimum_turns:raise ValueError(f'源对局覆盖不足:连续{last}回合，需要至少{minimum_turns}')
    selected={t:[x[1] for x in sorted(candidates[t].values(),key=lambda v:v[0])[:limit]] for t in range(1,last+1)}
    return selected,last,dict(dropped)

def stage_spec(spec, name, split):
    """返回 ``(wanted, explicit)``：优先用曲线里显式的 ``stage_targets[stage]``。

    ``explicit=True`` 时**不做插值**——"稍弱于模型"的阶梯每档都是算好的，
    从源库实测 P50 插值会让低目标档反而比高档强（阶梯反向）。
    旧格式（无 ``stage_targets``）返回 ``explicit=False``，保持逐位等价的历史行为。
    """
    stage_targets = spec.get('stage_targets') or None
    if stage_targets is not None:
        missing = [s for s in ('train-start', 'train-ramp1', 'train-ramp2',
                               'train', 'dev', 'test') if s not in stage_targets]
        if missing:
            raise ValueError('曲线 stage_targets 缺少档位:' + ','.join(missing))
        if name in stage_targets:
            return stage_targets[name], True
    return spec['train_targets' if split == 'train' else 'dev_test_targets'], False


def target_for(wanted, *, split, turn, progress, start_p50, explicit):
    """单回合目标。``explicit`` 档位直接取曲线值；旧格式在 train/T9+ 从源库水平插值。"""
    target = float(wanted[str(int(turn))])
    if not explicit and split == 'train' and int(turn) >= 9:
        previous = start_p50[min(int(turn), max(start_p50))]
        target = previous + (target - previous) * float(progress)
    return target


def build(source,out,curve,workers=8,smoke=False):
    import hsbg_engine as E
    from hsbg.az.train.mirror_snapshots import engine_stamp,write_snapshot_file,MirrorSnapshotPool
    from hsbg.az.train.personal import fingerprint,library_digest,write_json,checked_library
    from workflows.mirror_hard_gates import _drive_to_turn
    from hsbg.az.train.evaluate import evaluate_policy,EvalConfig
    from hsbg.az.train.policies import EngineBotPolicy
    source,out,curve=map(Path,(source,out,curve))
    if out.exists():raise FileExistsError('新库必须使用独立不存在目录；保留失败证据')
    spec=json.loads(curve.read_text(encoding='utf8'));fp=fingerprint();stamp=engine_stamp()
    if stamp['game_version']!='37.0.2.256589':raise ValueError('引擎版本不符')
    old={s:checked_library(source/s) for s in ('train','dev','test')}
    if any(m['fingerprint']!=fp or m['split']!=s for s,m in old.items()):raise ValueError('原库指纹/分割不符')
    # 当前库单桶随机等权，按真实帧重新计算过渡起点，不信任旧均值配置。
    pool=MirrorSnapshotPool(str(source/'train'),sample='random')
    if pool.bad_files or any(seat!=0 for turn,seat in pool.index):raise ValueError('源Train必须是单桶随机等权库')
    start_p50={t:median(f.strength for f in pool.index[(t,0)]) for t in pool.turns()}
    seed_sets={};all_rows={};source_last={};drop_reports={}
    out.mkdir(parents=True)
    for index,split in enumerate(('train','dev','test')):
        paths=sorted((source/'raw'/split/'frames').glob('snap_*.json.gz'))
        if not paths:raise ValueError('缺少原始Bot快照:'+split+'；不能把已校准帧冒充原始数据')
        threshold=1 if smoke else 128 if split=='train' else 32
        try:selected,last,dropped=select_rows(paths,stamp,threshold,7 if smoke else 15,1024 if split=='train' else 256,expected_rules=json.loads(E.Game(seed=0).state_json())['cfg'],expected_source_hash=fp['engine']['source_hash'])
        except ValueError:
            # 新种子互斥段补足，最多8192/1024；旧源从不修改。
            selected=None
            for offset in range(0,8192 if split=='train' else 1024,256):
                capture(out/'supplement'/split,20000000+index*1000000+offset,256,workers)
                extra=sorted((out/'supplement'/split/'frames').glob('snap_*.json.gz'))
                try:selected,last,dropped=select_rows(paths+extra,stamp,threshold,7 if smoke else 15,1024 if split=='train' else 256,expected_rules=json.loads(E.Game(seed=0).state_json())['cfg'],expected_source_hash=fp['engine']['source_hash']);break
                except ValueError:pass
            if selected is None:raise ValueError('达到补采上限，仍缺连续覆盖')
        seed_sets[split]={f['source_game_seed'] for rows in selected.values() for f in rows}
        all_rows[split]=selected;source_last[split]=last;drop_reports[split]=dropped
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        if seed_sets[a]&seed_sets[b]:raise ValueError('分割源对局重复')
    template=_drive_to_turn(E.Game(seed=730000),1).state_json()
    manifests={};audits={}
    # 两种曲线格式：
    # - ``stage_targets``（``hsai.opponent_curve/v2``）：每档显式目标，**不做插值**；
    #   这是"稍弱于模型"梯队的唯一正确形式——旧实现从源库实测 P50 插值，一旦新目标低于
    #   源库水平，warm0 就会比 train 还强（阶梯反向）。
    # - 旧格式（无 ``stage_targets``）：保持逐位等价的历史行为，便于复现旧库。
    stage_targets=spec.get('stage_targets') or None
    if stage_targets is not None:
        missing=[s for s in ('train-start','train-ramp1','train-ramp2','train','dev','test')
                 if s not in stage_targets]
        if missing:raise ValueError('曲线 stage_targets 缺少档位:'+','.join(missing))
    curve_profile=str(spec.get('calibration_profile') or 'p50-v1')
    model_ref=((spec.get('reference') or {}).get('model_p50') or None)
    profiles=[('train-start','train',0.),('train-ramp1','train',1/3),('train-ramp2','train',2/3),('train','train',1.),('dev','dev',1.),('test','test',1.)]
    for name,split,progress in profiles:
        dest=out/name;dest.mkdir();observed={};shard=[];count=0;last=source_last[split]
        explicit=bool(stage_targets and (name in stage_targets))
        wanted=(stage_targets[name] if explicit
                else spec['train_targets' if split=='train' else 'dev_test_targets'])
        for turn in range(1,20):
            origin=min(turn,last);target=float(wanted[str(turn)])
            if not explicit and split=='train' and turn>=9:
                previous=start_p50[min(turn,max(start_p50))]
                target=previous+(target-previous)*progress
            rows=calibrated(all_rows[split][origin],target)
            for row in rows:
                row['source_turn']=origin;row['source_seat']=row['seat'];row['turn']=turn;row['seat']=0;row['source']='bot_p50_'+split
                game=E.Game.from_json(template);result=game.apply_turn_snapshots([[0,json.dumps(row['player'])]],False)
                if result['applied']!=1 or result['rejected'] or not result['pool_balance_ok'] or not result['pool_species_ok'] or game.invariant_violations():raise RuntimeError('逐帧注入/卡池门失败:'+str(result))
                row['frame_sha256']=hashlib.sha256(json.dumps(row['player'],sort_keys=True).encode()).hexdigest();shard.append(row);count+=1
                if len(shard)>=512:write_snapshot_file(str(dest/f'snap_{count:08d}.json.gz'),shard,engine=stamp,meta={'driver':DRIVER,'split':split});shard=[]
            bodies=[sum(totals(f)) for f in rows]
            shares=[max(m['atk']+m['hp'] for m in f['player']['board'])/sum(totals(f)) for f in rows]
            observed[turn]=dict(target_p50=target,target_over_model=((target/float(model_ref[str(turn)])) if model_ref and str(turn) in model_ref else None),**strength_quantiles(bodies),mean_strength=mean(bodies),frames=len(rows),source_games=len({f['source_game_seed'] for f in rows}),source_turn=origin,tail_reserve=origin!=turn,p50_board_count=median(len(f['player']['board']) for f in rows),max_body_share=max(shares),p50_body_share=median(shares),p90_body_share=sorted(shares)[int(.9*(len(shares)-1))],minimum_strength=min(bodies),maximum_strength=max(bodies))
        if shard:write_snapshot_file(str(dest/f'snap_{count:08d}.json.gz'),shard,engine=stamp,meta={'driver':DRIVER,'split':split})
        write_json(dest/'calibration.json',observed)
        manifests[name]=dict(schema='hsrl.library/v1',split=split,driver=DRIVER,digest=library_digest(dest),fingerprint=fp,last_turn=19,source_last_turn=last,smoke=smoke,calibration_profile=curve_profile,curriculum_progress=progress,source_digest=old[split]['digest'],curve_sha256=hashlib.sha256(curve.read_bytes().replace(b'\r\n',b'\n')).hexdigest(),model_reference=((spec.get('reference') or {}).get('model')),source_games=len(seed_sets[split]),validation=dict(checked=count,accepted=count,quarantined=0),sampling='uniform_frames_single_seat_bucket',dropped_small_boards=drop_reports[split])
        write_json(dest/'manifest.json',manifests[name])
        pool=MirrorSnapshotPool(str(dest),sample='random')
        result=evaluate_policy(EngineBotPolicy(),EvalConfig(games=1 if smoke else 8,seed=9400000+profiles.index((name,split,progress))*1000,mirror_pool=pool,mirror_strict=True,collect_exposure=True))
        m,c=result['mirror'],result['mirror_strict_counts']
        if (result['illegal_actions'] or result['truncated_games'] or result['truncated_combats'] or any(m.get(k,0) for k in ('mirror_errors','mirror_missing','mirror_rejected','mirror_fallback','mirror_pool_unbalanced','mirror_pool_species_unbalanced')) or not m['mirror_attempt']==m['mirror_applied']==c['eligible_live_seats']):raise RuntimeError('完整对局严格镜像验收失败')
        audits[name]=dict(mirror=m,strict=c)
        write_json(out/'strict-audit.json',audits)
    write_json(out/'report.json',dict(complete=True,smoke=smoke,fingerprint=fp,splits=manifests,strict_audit=audits,source=str(source.resolve()),source_last_turns=source_last,primary_statistic='P50(sum_attack+sum_health)',start_p50=start_p50,source_games_disjoint=True,calibration_profile=curve_profile,curve=str(curve),curve_sha256=hashlib.sha256(curve.read_bytes().replace(b'\r\n',b'\n')).hexdigest(),model_reference=((spec.get('reference') or {}).get('model')),stage_targets=(spec.get('stage_targets') or None),stage_ratios=(spec.get('stage_ratios') or None)))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--curve',type=Path,default=ROOT/'config/opponent-p50-next.planned.json');p.add_argument('--workers',type=int,default=8);p.add_argument('--smoke',action='store_true');a=p.parse_args();build(a.source,a.out,a.curve,a.workers,a.smoke)
