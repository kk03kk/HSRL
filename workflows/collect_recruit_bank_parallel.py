"""Collect an independent natural-state bank in bounded CPU subprocess shards."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json,file_identity
from hsbg.runtime.provenance import source_identity,environment_identity
from hsbg.az.train.common import engine_info
from workflows.collect_recruit_states import validate_source_namespace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--games',type=int,default=512)
    p.add_argument('--shards',type=int,default=4);p.add_argument('--seed',type=int,default=11802000)
    p.add_argument('--assessment-only',action='store_true',help='Seal all collected sources against any training use')
    args=p.parse_args()
    if args.shards<1 or args.games<1 or args.games%args.shards:raise ValueError('require equal nonempty shards')
    validate_source_namespace(args.seed,args.games,args.assessment_only)
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    checkpoint='artifacts/models/arena-v1.4-r1/model.pt'
    source=source_identity(Path.cwd());environment=environment_identity()
    if source['tracked_dirty']:raise RuntimeError('committed source required')
    count=args.games//args.shards
    def worker(index):
        dest=out/f'shard{index}'
        cmd=[sys.executable,'workflows/collect_recruit_states.py','--checkpoint',checkpoint,
             '--out',str(dest),'--games',str(count),'--seed',str(args.seed+count*index),'--device','cpu']
        if args.assessment_only:cmd.append('--assessment-only')
        atomic_json(out/f'command{index}.json',cmd)
        with (out/f'shard{index}.log').open('x',encoding='utf-8') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,
                           env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        return dest
    with ThreadPoolExecutor(max_workers=args.shards) as pool:parts=list(pool.map(worker,range(args.shards)))
    states=0;seeds=[];ranks=[];identities={}
    parent=file_identity(checkpoint)
    with gzip.open(out/'states.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as stream:
        for part in parts:
            m=json.loads((part/'manifest.json').read_text())
            if m['checkpoint']!=parent or file_identity(part/'states.jsonl.gz')!=m['data']:
                raise ValueError('shard identity mismatch')
            if bool(m.get('assessment_only'))!=args.assessment_only:
                raise ValueError('shard assessment role differs')
            n=0
            with gzip.open(part/'states.jsonl.gz','rt',encoding='utf-8') as f:
                for line in f:stream.write(line);n+=1
            if n!=m['states']:raise ValueError('state count mismatch')
            states+=n;seeds+=m['seeds'];ranks+=m['ranks']
            identities[part.name]=file_identity(part/'manifest.json')
    if seeds!=list(range(args.seed,args.seed+args.games)):raise ValueError('source game seed coverage mismatch')
    if environment_identity()!=environment or source_identity(Path.cwd())!=source:
        raise ValueError('source/environment changed during collection')
    manifest={'schema':'hsai.recruit-state-bank/v1','checkpoint':parent,
        'checkpoint_path':checkpoint,'args':vars(args),'states':states,'seeds':seeds,'ranks':ranks,
        'data':file_identity(out/'states.jsonl.gz'),'shards':identities,'source':source,
        'environment':environment,'engine':engine_info(),'purpose':'Independent natural training/development source, never formal assessment'}
    if args.assessment_only:
        manifest.update(assessment_only=True,purpose='Independent conditional assessment-only source; never training or natural-game promotion')
    atomic_json(out/'manifest.json',manifest)
    print(json.dumps({'games':args.games,'states':states,'out':str(out)}))


if __name__=='__main__':main()
