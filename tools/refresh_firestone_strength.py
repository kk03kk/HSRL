"""新feed只读抓取PMS；原始文件留档，不用旧引擎映射/身材上限截掉新卡和大身材。"""
from pathlib import Path
import json,urllib.request,concurrent.futures,hashlib,time,gzip,collections,datetime,zipfile,re,argparse
ROOT=Path(__file__).resolve().parents[1]
DEST=None
HEADERS={'User-Agent':'Mozilla/5.0','Referer':'https://replays.firestoneapp.com/','Accept':'*/*'}
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8');q.replace(p)
def get(url):
 for attempt in range(3):
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers=HEADERS),timeout=45) as r:return r.read()
  except Exception:
   if attempt==2:raise
   time.sleep(1+2*attempt)
def main():
 global DEST
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path);parser.add_argument('--reuse-feed',action='store_true');args=parser.parse_args()
 DEST=args.out or ROOT/'artifacts/datasets/firestone'/('refresh-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
 if args.reuse_feed and not (DEST/'feed-fresh.json').exists():parser.error('--reuse-feed需要已有feed-fresh.json')
 if not args.reuse_feed:
  if DEST.exists() and any(DEST.iterdir()):parser.error('新抓取需要新目录；复验已有快照请显式--reuse-feed')
  DEST.mkdir(parents=True,exist_ok=True)
  url='https://static.zerotoheroes.com/api/bgs/bgs-perfect-games.json?hsai='+str(int(time.time()))
  with urllib.request.urlopen(urllib.request.Request(url,headers={**HEADERS,'Cache-Control':'no-cache'}),timeout=60) as response:body=response.read();headers=dict(response.headers)
  (DEST/'feed-fresh.json').write_bytes(body)
  write(DEST/'fetch-metadata.json',dict(url=url,observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),headers=headers,sha256=hashlib.sha256(body).hexdigest()))
 feed=json.loads((DEST/'feed-fresh.json').read_text(encoding='utf8'));games={}
 for g in sorted(feed,key=lambda x:x['creationTimestamp'],reverse=True):games.setdefault(g.get('originalReviewId') or g['reviewId'],g)
 write(DEST/'index.json',list(games.values()))
 cache={}
 for p in (ROOT/'artifacts/datasets/firestone').glob('*/pms/*.json'):
  if p.parent.parent!=DEST:cache.setdefault(p.stem,p)
 (DEST/'pms').mkdir(exist_ok=True);done=[]
 def job(item):
  original,g=item;p=DEST/'pms'/(original+'.json');source=None
  try:
   if not p.exists():
    source=cache.get(original) or cache.get(g['reviewId'])
    if source:
     # 原始资料不可变，在本工作区内复用物理字节；后续只读。
     import os;os.link(source,p)
    else:
     data=get('https://bgs-post-match-stats.firestoneapp.com/'+g['reviewId']+'.gz.json');tmp=p.with_suffix('.part');tmp.write_bytes(data);tmp.replace(p);time.sleep(.15)
   raw=p.read_bytes();d=json.loads(gzip.decompress(raw) if raw[:2]==b'\x1f\x8b' else raw)
   if not isinstance(d.get('battleResultHistory'),list):raise ValueError('missing battleResultHistory')
   return dict(original_id=original,review_id=g['reviewId'],build=g.get('buildNumber'),path=p.relative_to(ROOT).as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),cache_source=str(source) if source else None,battles=len(d['battleResultHistory']),ok=True)
  except Exception as exc:return dict(original_id=original,review_id=g['reviewId'],build=g.get('buildNumber'),ok=False,error=repr(exc))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
  for f in concurrent.futures.as_completed([ex.submit(job,i) for i in games.items()]):
   done.append(f.result())
   if len(done)%50==0:
    print('PMS',len(done),'/',len(games),'fail',sum(not d['ok'] for d in done),flush=True);write(DEST/'download-progress.json',done)
 write(DEST/'download-summary.json',done)
 # 抽查每版本最新两个回放XML头，不将抽查冒充逐局版本验证。
 samples=[]
 for build in sorted({g.get('buildNumber') for g in games.values()}):
  for g in [x for x in games.values() if x.get('buildNumber')==build][:2]:
   try:
    raw=get('https://xml.firestoneapp.com/'+g['replayKey']);p=DEST/'version-checks'/(g['reviewId']+'.xml.zip');p.parent.mkdir(exist_ok=True);p.write_bytes(raw)
    with zipfile.ZipFile(p) as z:head=z.open(next(n for n in z.namelist() if n.endswith('.xml'))).read(4096).decode('utf8','replace')
    m=re.search(r'<Game\b[^>]*buildNumber="(\d+)"',head);actual=int(m[1]) if m else None
    samples.append(dict(review_id=g['reviewId'],feed_build=build,xml_build=actual,match=actual==build,sha256=hashlib.sha256(raw).hexdigest()))
   except Exception as e:samples.append(dict(review_id=g['reviewId'],feed_build=build,error=str(e)))
 write(DEST/'version-checks.json',samples)
 print('DONE',len(done),'ok',sum(x['ok'] for x in done),'xml',samples,flush=True)
if __name__=='__main__':main()
