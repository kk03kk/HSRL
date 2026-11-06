"""Firestone稳健战力曲线：回放去重、回合等权、保留空板/极端值，显式报告筛选。"""
from pathlib import Path
import argparse,json,hashlib,collections,gzip,math,datetime,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def split(gid):
 h=int(hashlib.sha256(gid.encode()).hexdigest()[:8],16)%10
 return 'test' if h==0 else 'dev' if h==1 else 'train'
def quantile(x,w,q):
 x=np.asarray(x,float);w=np.asarray(w,float);keep=w>0;x=x[keep];w=w[keep]
 order=np.argsort(x,kind='stable');x=x[order];w=w[order]
 cumulative=np.cumsum(w);target=q*w.sum();j=min(int(np.searchsorted(cumulative,target,side='left')),len(x)-1)
 # 中位数恰落两点之间时取区间中点，与普通偶数样本median一致。
 if q==.5 and j+1<len(x) and np.isclose(cumulative[j],target,rtol=0,atol=1e-10):return float((x[j]+x[j+1])/2)
 return float(x[j])
def trimmed(x,w,alpha=.1):
 order=np.argsort(x);x=np.asarray(x)[order];w=np.asarray(w)[order];end=np.cumsum(w);start=end-w
 inside=np.maximum(0,np.minimum(end,(1-alpha)*w.sum())-np.maximum(start,alpha*w.sum()))
 return float(np.dot(x,inside)/inside.sum())
def aggregate(rows,bootstrap=500):
 groups=collections.Counter(x['game'] for x in rows);ids={g:i for i,g in enumerate(sorted(groups))}
 x=np.array([r['strength'] for r in rows],float);w=np.array([1/groups[r['game']] for r in rows]);n=len(groups)
 out=dict(games=n,boards=len(rows),self_boards=sum(r['side']=='self' for r in rows),opponent_boards=sum(r['side']=='opponent' for r in rows),
  empty_board_fraction=float(np.average([r['count']==0 for r in rows],weights=w)),
  p25=quantile(x,w,.25),p50=quantile(x,w,.5),p75=quantile(x,w,.75),p90=quantile(x,w,.9),
  weighted_mean=float(np.average(x,weights=w)),unweighted_mean=float(x.mean()),trimmed_mean_10pct=trimmed(x,w),maximum=float(x.max()),
  median_attack=quantile([r['attack'] for r in rows],w,.5),median_health=quantile([r['health'] for r in rows],w,.5),
  balanced_body_p50=quantile([r['balanced_body'] for r in rows],w,.5),support='very_low' if n<30 else 'low' if n<100 else 'covered')
 for field in ('attack','health'):
  for name,q in (('p25',.25),('p50',.5),('p75',.75),('p90',.9)):
   out[field+'_'+name]=quantile([r[field] for r in rows],w,q)
 for flag in ('shield','reborn','venomous','deathrattle_printed'):
  out[flag+'_board_fraction']=float(np.average([r[flag] for r in rows],weights=w))
 for side in ('self','opponent'):
  selected=[r['strength'] for r in rows if r['side']==side];out[side+'_p50']=float(np.median(selected)) if selected else None
 rng=np.random.default_rng(20261106);gi=np.array([ids[r['game']] for r in rows]);medians=[]
 for _ in range(bootstrap):
  counts=np.bincount(rng.integers(0,n,n),minlength=n);medians.append(quantile(x,w*counts[gi],.5))
 out['p50_ci95_replay_cluster']=np.quantile(medians,[.025,.975]).tolist()
 # 代表帧只用于展示联合攻血，不把两个边际中位数相加冒充S中位数。
 nearest=min(rows,key=lambda r:abs(r['strength']-out['p50']))
 out['representative']=dict(game=nearest['game'],side=nearest['side'],attack=nearest['attack'],health=nearest['health'],strength=nearest['strength'])
 return out
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--build',type=int);p.add_argument('--split',choices=('train','dev','test'),default='train');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 feed=json.loads((a.root/'feed-fresh.json').read_text(encoding='utf8'));index=json.loads((a.root/'index.json').read_text(encoding='utf8'))
 build=a.build or max(g['buildNumber'] for g in index);selected=[g for g in index if g['buildNumber']==build and split(g.get('originalReviewId') or g['reviewId'])==a.split]
 cards=json.loads((ROOT/'engine/crates/hsbg-engine/data/cards.json').read_text(encoding='utf8'))['cards'];cardmap={c['id']:c for c in cards}
 reasons=collections.Counter();rows=[];outer=collections.Counter();bad_cases=[]
 for g in selected:
  gid=g.get('originalReviewId') or g['reviewId'];data=(a.root/'pms'/(gid+'.json')).read_bytes();d=json.loads(gzip.decompress(data) if data[:2]==b'\x1f\x8b' else data)
  slots={};conflicts=set()
  for battle in d.get('battleResultHistory',[]):
   info=battle.get('battleInfo')
   if not info or battle.get('actualResult') is None:reasons['no_completed_battle']+=1;continue
   if info.get('playerTeammateBoard') is not None or info.get('opponentTeammateBoard') is not None:reasons['duos']+=1;continue
   turn=(info.get('gameState') or {}).get('currentTurn')
   if type(turn)!=int or not 1<=turn<=100:reasons['invalid_exact_turn']+=1;continue
   if isinstance(battle.get('turn'),int):outer[str(battle['turn']-turn)]+=1
   for key,side in (('playerBoard','self'),('opponentBoard','opponent')):
    raw=info.get(key) or {};player=raw.get('player') or {};hp=player.get('hpLeft');board=raw.get('board')
    if isinstance(hp,(int,float)) and hp<=0:reasons['nonalive_'+side]+=1;continue
    if not isinstance(board,list) or len(board)>7:reasons['missing_or_invalid_board']+=1;continue
    if any(not isinstance(m.get('attack'),int) or isinstance(m.get('attack'),bool) or not isinstance(m.get('health'),int) or isinstance(m.get('health'),bool) or m['attack']<0 or m['health']<=0 or m.get('definitelyDead') for m in board):reasons['invalid_minion_stats']+=1;continue
    atk=sum(m['attack'] for m in board);health=sum(m['health'] for m in board)
    frame=dict(game=gid,review_id=g['reviewId'],build=build,side=side,turn=turn,attack=atk,health=health,strength=atk+health,count=len(board),tier=player.get('tavernTier'),
      balanced_body=2*sum(math.sqrt(m['attack']*m['health']) for m in board),shield=any(m.get('divineShield') for m in board),reborn=any(m.get('reborn') for m in board),
      venomous=any(m.get('venomous') or m.get('poisonous') for m in board),deathrattle_printed=any('DEATHRATTLE' in cardmap.get(m.get('cardId'),{}).get('keywords',[]) for m in board),
      unknown_cards=[m.get('cardId') for m in board if m.get('cardId') not in cardmap],board=board)
    slot=(turn,side);sig=json.dumps(board,sort_keys=True)
    if slot in slots:
     if slots[slot][0]!=sig:conflicts.add(slot);reasons['conflicting_duplicate']+=1
     else:reasons['identical_duplicate']+=1
    else:slots[slot]=(sig,frame)
  for key,(_,frame) in slots.items():
   if key not in conflicts:rows.append(frame)
   else:
    reasons['conflict_frames_quarantined']+=1
    if len(bad_cases)<20:bad_cases.append(dict(game=gid,turn=key[0],side=key[1]))
 turns={str(t):aggregate([r for r in rows if r['turn']==t]) for t in sorted({r['turn'] for r in rows})}
 a.out.mkdir(parents=True,exist_ok=True)
 with gzip.open(a.out/'boards.jsonl.gz','wt',encoding='utf8') as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 result=dict(schema='hsai.firestone-strength/v1',feed_url='https://static.zerotoheroes.com/api/bgs/bgs-perfect-games.json',feed_sha256=hashlib.sha256((a.root/'feed-fresh.json').read_bytes()).hexdigest(),feed_entries=len(feed),unique_replay_records=len(index),feed_build_counts=dict(collections.Counter(str(x['buildNumber']) for x in feed)),
  feed_creation_range_utc=[min(x['creationDate'] for x in feed),max(x['creationDate'] for x in feed)],build=build,split=a.split,selected_games=len(selected),valid_games=len({r['game'] for r in rows}),boards=len(rows),turns=turns,
  dropped=dict(reasons),outer_turn_minus_explicit_turn=dict(outer),conflict_examples=bad_cases,empty_boards_kept=sum(r['count']==0 for r in rows),unknown_card_frames_kept=sum(bool(r['unknown_cards']) for r in rows),
  notes=['primary S=sum(attack)+sum(current health); not combat winrate','each replay-turn has total weight 1, divided among eligible sides','P50 and P25/P75 primary; 10% weighted trimmed mean supplementary','no engine mapping or large-stat cap filter','primary uses deterministic Train only; Dev/Test PMS not analyzed','replay-cluster CI does not remove highlight-selection bias or unknown same-lobby duplicates','PMS prebattle supplied boards; exact EOT/start-of-combat phase cannot be newly guaranteed for all replays','currentTurn from battleInfo.gameState; outer turn not used','zero/negative hero hp excluded as nonalive; this depends on supplied snapshot semantics'])
 (a.out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 columns=['turn','games','boards','p25','p50','p75','p90','weighted_mean','trimmed_mean_10pct','maximum','self_p50','opponent_p50','empty_board_fraction','balanced_body_p50']+[field+'_'+q for field in ('attack','health') for q in ('p25','p50','p75','p90')]
 with (a.out/'turn-strength.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
  for t,row in turns.items():w.writerow({k:(int(t) if k=='turn' else row[k]) for k in columns})
 print('build',build,'split',a.split,'games',len(selected),'boards',len(rows),'dropped',dict(reasons),flush=True)
 for t,r in turns.items():print(t,r['games'],r['boards'],*[round(r[k],1) for k in ('p25','p50','p75','p90','weighted_mean','trimmed_mean_10pct','maximum')],flush=True)
if __name__=='__main__':main()
