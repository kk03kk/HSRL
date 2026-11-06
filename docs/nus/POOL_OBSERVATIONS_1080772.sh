cd ~/projects/HSRL || exit
/usr/bin/python3 -S - <<'PY'
import json
from pathlib import Path
p=Path('personal/runs/nus-4159-economy-long-002/train/history.jsonl')
limit=min(p.stat().st_size,64*1024*1024)
counts=0;examples=[];complete=0
with p.open('rb') as f:
    while f.tell()<limit:
        raw=f.readline(min(256*1024,limit-f.tell()))
        if not raw or not raw.endswith(b'\n'):break
        try:r=json.loads(raw)
        except ValueError:continue
        if r.get('iter') is None or r['iter']>=6141:continue
        complete+=1;c=r.get('collect') or {}
        n=c.get('mirror_pool_species_preexisting',0);counts+=n
        if n and len(examples)<5:
            examples.append(dict(iter=r['iter'],count=n,observations=(c.get('mirror_pool_observations') or [])[:2]))
print(json.dumps(dict(scope='提交版本6141之前，与回传缓存汇总范围对应',read_limit_bytes=limit,complete_rows=complete,count=counts,examples=examples),ensure_ascii=False,indent=2))
PY
