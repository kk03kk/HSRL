"""Source-paired actual whole-turn outcomes, separate from imitation accuracy."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from hsbg.runtime.artifacts import atomic_json,file_identity

def compare(baseline,result):
    before={(r['seed'],r['kind']):r for r in baseline['closed_loop']['rows']}
    after={(r['seed'],r['kind']):r for r in result['closed_loop']['rows']}
    if len(before)!=len(baseline['closed_loop']['rows']) or len(after)!=len(result['closed_loop']['rows']):
        raise ValueError('duplicate source outcome')
    if set(before)!=set(after):raise ValueError('source pairing differs')
    rng=np.random.default_rng(1196801);out={}
    for kind in sorted({k for s,k in before}):
        keys=sorted(k for k in before if k[1]==kind)
        if any(before[k]['teacher']!=after[k]['teacher'] for k in keys):raise ValueError('teacher differs')
        arms={'parent':[before[k]['student'] for k in keys],
              'student':[after[k]['student'] for k in keys],
              'teacher':[after[k]['teacher'] for k in keys]}
        groups={}
        for name,rows in arms.items():
            groups[name]={'n':len(rows),'eot_body_median':float(np.median([r['eot_body'] for r in rows])),
                'eot_body_mean':float(np.mean([r['eot_body'] for r in rows])),
                'gold_spent_mean':float(np.mean([r['gold_spent'] for r in rows])),
                'core_count_mean':float(np.mean([r['core_count'] for r in rows])),
                'recruit_deaths':sum(not r.get('alive',True) for r in rows),
                # The reused older parent trace did not observe forced reasons.
                'action_caps':None if name=='parent' else sum(r['stopped_at_cap'] for r in rows)}
        differences={}
        for reference in ('parent','teacher'):
            x=np.asarray([a['eot_body']-b['eot_body'] for a,b in zip(arms['student'],arms[reference])])
            boot=np.mean(rng.choice(x,size=(5000,len(x)),replace=True),axis=1)
            differences['student_minus_'+reference]={'mean':float(x.mean()),
                'source_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),
                'better':int((x>0).sum()),'equal':int((x==0).sum()),'worse':int((x<0).sum())}
        out[kind]={'sources':len(keys),'arms':groups,'differences':differences}
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',required=True)
    p.add_argument('--result',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    baseline=json.loads(Path(args.baseline).read_text());result=json.loads(Path(args.result).read_text())
    output={'inputs':{n:file_identity(v) for n,v in [('baseline',args.baseline),('result',args.result)]},
        'comparison':compare(baseline,result),'labels':result['labels'],'anchor':result['anchor'],
        'model':result['model'],'scope':'Same explicitly endowed source starts; not natural ability. Source-game bootstrap, standard median; original parent cap count unobserved.'}
    atomic_json(args.out,output);print(json.dumps(output['comparison']))

if __name__=='__main__':main()
