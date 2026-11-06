"""Paired, no-input offline CPU chains and distinct real-frame scene checks.

Authored effects exercise production geometry/planning/checkpoint/verification;
they do not simulate Hearthstone or prove a native click. Pixel comparison is
reported separately and never used to authorize these authored actions.
"""
import argparse,hashlib,json,statistics,time,tempfile
from dataclasses import asdict,replace
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from .model import Observation,Decision,Kind
from .checkpoint import ActionCheckpoint
from .chain_metrics import ChainMetrics
from .geometry_benchmark import changing_chain
from .layout_resolver import LayoutCatalog,LayoutResolver,Viewport
from .local_layout import LocalLayoutProfile
from .planning import ShadowPlanner
from .client_readiness import logical_reasons
from .verification import verify,Status
from .recording import compact_observation


def frozen(root,name):
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    path=root/(name+'.py.txt');data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=manifest['files'][path.name]:raise ValueError('baseline_hash_mismatch')
    ns={'__name__':'bg_agent.frozen_baseline','__package__':'bg_agent'}
    exec(compile(data,str(path),'exec'),ns)
    return ns


def authored(length):
    history=[];rows=[]
    for before,action,after in changing_chain(length):
        # Match the causal snapshot structure of the real log projector: no
        # recursive receipt history inside a root's own post-state.
        proof=replace(after,resolutions=(),actions=()).to_dict()
        receipt=replace(after.resolutions[-1],post_state=proof)
        before=replace(before,source='sandbox',resolutions=tuple(history))
        history.append(receipt);after=replace(after,source='sandbox',resolutions=tuple(history))
        rows.append((before,action,after))
    return rows


def run_chain(rows,mode,profile,old_checkpoint,out):
    resolver=LayoutResolver(LayoutCatalog(profile.entries));metrics=ChainMetrics();failures=[];delivered=[]
    path=out/(mode+'-checkpoint.json');ledger=(old_checkpoint if mode=='baseline' else ActionCheckpoint)(path,None,'authored')
    cpu=time.process_time();wall=time.perf_counter()
    serializer=asdict if mode=='baseline' else Observation.to_dict
    for step,(published,action,after) in enumerate(rows):
        with metrics.measure('chain'):
            with metrics.measure('snapshot_and_projection'):
                if mode=='baseline':
                    payload=serializer(published);before=Observation.from_dict(payload)
                else:
                    payload=compact_observation(published).to_dict();before=published
            with metrics.measure('decision'):
                decision=Decision(action,before.revision,'authored model output; policy unchanged')
            with metrics.measure('geometry'):
                projection=resolver.resolve(before,Viewport(2560,1600),recovery=True)
            row=dict(id=str(step),session='authored',window={'client':dict(x=0,y=0,width=2560,height=1600),'token':'authored'},
                observation=payload,capture={'end':time.time()},blockers=[],log_snapshot_unchanged=True)
            with metrics.measure('plan_and_semantic_readiness'):
                plan=ShadowPlanner(row,projection.mappings,compact=True).build(before,decision)
                blocked=logical_reasons(row,action)
                if plan['rejection_reasons'] or blocked:failures.append(dict(step=step,plan=plan['rejection_reasons'],logical=blocked))
            with metrics.measure('durable_prepare'):
                ledger.begin(action,before,str(step));ledger.stage('initial','prepared')
            with metrics.measure('memory_input'):
                delivered.append((action.key,tuple((s.get('kind'),s.get('point_client')) for s in plan['steps'])))
            with metrics.measure('durable_sent'):
                ledger.stage('initial','sent');ledger.delivery('accepted')
            with metrics.measure('verification'):
                outcome=verify(action,before,after)
                if outcome.status!=Status.CONFIRMED:failures.append(dict(step=step,status=str(outcome.status),reason=outcome.reason))
            with metrics.measure('durable_commit'):
                ledger.commit()
    return dict(steps=len(rows),verified=ledger.state['verified'],cpu_ms=(time.process_time()-cpu)*1000,
        wall_ms=(time.perf_counter()-wall)*1000,stages=metrics.report(),failures=failures,
        geometry_p95_ms=metrics.report()['latency_ms']['geometry']['p95'],
        final_revision=rows[-1][2].revision,final_gold=rows[-1][2].gold,final_request=rows[-1][2].interaction,
        input_fingerprint=hashlib.sha256(repr(delivered).encode()).hexdigest())


def real_scenes(directory,baseline,limit=116):
    from .geometry_mapper import GeometryMapper
    results=[];seen=set();all_timings={'baseline':[],'current':[]}
    for path in sorted(directory.glob('calibration-*.png'))[:limit]:
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:continue
        seen.add(digest);image=Image.open(path).convert('RGB');view=Viewport(*image.size);found={};timings={}
        for name,cls in [('baseline',baseline),('current',GeometryMapper)]:
            start=time.perf_counter();state=cls._small_scene(None,image,view);elapsed=(time.perf_counter()-start)*1000
            found[name]=state['mode'];timings[name]=elapsed;all_timings[name].append(elapsed)
        results.append(dict(file=path.name,sha256=digest,modes=found,timings_ms=timings))
    return dict(distinct_real_captures=len(results),rows=results,
        mode_differences=[r for r in results if r['modes']['baseline']!=r['modes']['current']],
        p50_ms={k:statistics.median(v) for k,v in all_timings.items()},
        total_ms={k:sum(v) for k,v in all_timings.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--frames',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    profile=LocalLayoutProfile.load();old=frozen(a.baseline,'checkpoint')['ActionCheckpoint'];chains=[]
    for count in (20,50,100):
        rows=authored(count);runs={'baseline':[],'current':[]}
        # Alternate order; no duplicate real-frame cache is timed as capture.
        for repeat in range(3):
            for mode in (('baseline','current') if repeat%2==0 else ('current','baseline')):
                if mode=='baseline':
                    with patch.object(Observation,'to_dict',asdict):r=run_chain(rows,mode,profile,old,a.out)
                else:r=run_chain(rows,mode,profile,old,a.out)
                runs[mode].append(r)
        med={k:statistics.median(r['cpu_ms'] for r in v) for k,v in runs.items()}
        chains.append(dict(steps=count,runs=runs,cpu_median_ms=med,cpu_reduction=1-med['current']/med['baseline'],
            fingerprints_match=len({r['input_fingerprint'] for v in runs.values() for r in v})==1))
    report=dict(schema=1,baseline_commit='0ba2695',scope='authored memory effects; production planning, semantic checks, checkpoint IO and verifier; no native client readiness or native input',
        excluded=dict(game_wait='not measured offline',mouse='safe timings unchanged; no Windows input',capture='no live capture in this benchmark',calibration='not included'),
        chains=chains,live_game_accepted=False)
    if a.frames:report['real_scene_comparison']=real_scenes(a.frames,frozen(a.baseline,'geometry_mapper')['GeometryMapper'])
    report['passed']=all((c['steps']!=100 or c['cpu_reduction']>=.30) and c['fingerprints_match'] and all(not r['failures'] and r['geometry_p95_ms']<=1 for v in c['runs'].values() for r in v) for c in chains)
    if a.frames:report['passed']=report['passed'] and not report['real_scene_comparison']['mode_differences']
    (a.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(passed=report['passed'],chains=[dict(steps=c['steps'],reduction=c['cpu_reduction'],failures=c['runs']['current'][0]['failures'][:3]) for c in chains],scene_differences=report.get('real_scene_comparison',{}).get('mode_differences',[])),ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
