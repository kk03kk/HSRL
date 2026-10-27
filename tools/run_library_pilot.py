"""One large-model server qualification: fresh BC, gated library PPO, independent test."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import signal
import time
try:
    from tools.local_offline import prepare
except ModuleNotFoundError:
    from local_offline import prepare


def run(config, out, budget=50000, bc_waves=16):
    import torch
    import taverngym as tg
    from taverngym.rl.config import Config, identity
    from taverngym.rl.train import train, evaluate, load_policy
    from taverngym.research.bootstrap import bootstrap
    cfg = Config.load(config)
    if cfg.policy_kind != 'entity_ff' or cfg.opponent_mode != 'library':
        raise ValueError('pilot requires entity_ff and prepared library config')
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('pilot requires a new output directory')
    out.mkdir(parents=True)
    started = time.monotonic()
    stopped = [False]
    def stop(signum, frame): stopped[0] = True
    for sig in (signal.SIGTERM,signal.SIGINT,*([signal.SIGUSR1] if hasattr(signal,'SIGUSR1') else [])):
        signal.signal(sig,stop)
    report = {'format':'taverngym.entity-library-pilot.v1','status':'started',
              'native':tg.build_info(),'identity':identity(cfg),'config':cfg.to_dict(),
              'torch':str(torch.__version__),'device':cfg.device,
              'gpu':torch.cuda.get_device_name() if cfg.device.startswith('cuda') else None,
              'notice':'Qualification, not full paper replication or human-level qualification.'}
    def save():
        report['seconds'] = time.monotonic()-started
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    if cfg.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    try:
        from taverngym.research.resources import qualify_resources
        cfg,report['resource_preflight'] = qualify_resources(cfg)
        report['parameters'] = report['resource_preflight']['successful']['parameters']
        report['config'] = cfg.to_dict()
        effective_config = out/'effective-config.json'
        effective_config.write_text(json.dumps(cfg.to_dict(),indent=2),encoding='utf-8')
        save()
        # New architecture starts fresh: never load a differently shaped small checkpoint.
        gate = bootstrap(effective_config,out/'bootstrap',init_from=None,waves=bc_waves,
                         epochs=4,bc_lr=3e-4,batch_chunks=min(4,cfg.chunks_per_batch),gate_games=8,stop=lambda:stopped[0])
        report['parameters'] = gate['source_initialization'].get('parameters')
        report['gate'] = {k:gate[k] for k in ('ready_for_ppo','checks','agreement')}
        if not gate['ready_for_ppo']:
            report['status']='bc_not_ready';save();return report
        if stopped[0]:
            report['status']='interrupted';save();return report
        cfg = replace(cfg,decision_budget=budget)
        (out/'pilot-config.json').write_text(json.dumps(cfg.to_dict(),indent=2),encoding='utf-8')
        train(cfg,out/'ppo',init_from=out/'bootstrap/bc-ready.pt')
        rows=[json.loads(x) for x in (out/'ppo/metrics.jsonl').read_text().splitlines()]
        complete=next((r for r in reversed(rows) if r.get('phase')=='completed'),None)
        if complete is None:
            report['status']='interrupted';save();return report
        ppo=[r for r in rows if r.get('phase')=='ppo']
        elapsed=sum(r['collection']['seconds']+r['training']['seconds'] for r in ppo)
        report['ppo']={'completion':complete,'effective_decisions_per_second':sum(r['collection']['samples'] for r in ppo)/elapsed,
                       'collection_seconds':sum(r['collection']['seconds'] for r in ppo),
                       'optimization_seconds':sum(r['training']['seconds'] for r in ppo),
                       'optimizer_updates':sum(r['training']['updates'] for r in ppo)}
        # Test config points only the evaluation side at a source-disjoint held-out bank.
        test_cfg=Config.load(Path(config).parent/'test-config.json')
        test_cfg=replace(test_cfg,device=cfg.device,eval_games=8,eval_seeds=[1957100000,1957200000])
        for name,path in [('bc',out/'bootstrap/bc-ready.pt'),('latest',out/'ppo/latest.pt'),('best',out/'ppo/best.pt')]:
            if stopped[0]:
                report['status']='interrupted';save();return report
            model=load_policy(path,test_cfg,identity(test_cfg))
            result=evaluate(test_cfg,model,stop=lambda:stopped[0],artifact_dir=out/'test-failures'/name)
            (out/f'independent-{name}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        report['status']='completed'
    except Exception as exc:
        report['status']='interrupted' if stopped[0] else 'failed'
        report['error']=type(exc).__name__+': '+str(exc)
        save()
        if not stopped[0]: raise
    finally:
        if cfg.device.startswith('cuda'):
            report['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
            report['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
        save()
    return report


if __name__=='__main__':
    handles=prepare()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--budget',type=int,default=50000);p.add_argument('--bc-waves',type=int,default=16)
    a=p.parse_args()
    if min(a.budget,a.bc_waves)<1:p.error('positive budgets required')
    print(json.dumps(run(a.config,a.out,a.budget,a.bc_waves)))
