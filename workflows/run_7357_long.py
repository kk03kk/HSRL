"""7357完整状态分叉与P50固定课程；来源预检可在登录节点纯标准库运行。"""
import argparse, json, os, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
MODEL_SHA='60594db00a627884d47afd4a326639ab9b1fa7c48eb88618399051d971dcff82'
TRAINER_SHA='13ef5b03338a7fd31a96a7e024f4431e2dd41183aa49f590d026299a41aca2ab'
SOURCE_CUM=558873732
SCHEDULE=[[0,'warm0'],[10000000,'warm1'],[20000000,'warm2'],[30000000,'train']]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def defaults(p):
    p.add_argument('--source-7357',type=Path,default=os.environ.get('HSAI_SOURCE_7357','artifacts/runs/nus-4159-economy-long-002/milestones/decisions-558819917'))
    p.add_argument('--source-run',type=Path,default=os.environ.get('HSAI_SOURCE_RUN','artifacts/runs/nus-4159-economy-long-002'))
    p.add_argument('--bot-3702',type=Path,default=os.environ.get('HSAI_P50_BANK','artifacts/libraries/bot-p50-3702-v1'))
    p.add_argument('--gpu-audit',type=Path,default=os.environ.get('HSAI_P50_AUDIT','artifacts/runs/p50-7357-preflight'))
    return p

def check_inputs(args,require_gpu=True):
    source=Path(args.source_7357);old=Path(args.source_run);bank=Path(args.bot_3702)
    for p in [source/n for n in ('model.pt','trainer.pt','checkpoint.json')]+[old/n for n in ('plan.json','report.json','train.config.json')]+[bank/'report.json']:
        if not p.is_file() or not p.stat().st_size:raise ValueError('缺少真实输入:'+str(p))
    cp=read(source/'checkpoint.json')
    if cp['version']!=7357 or cp['actual_decisions']!=SOURCE_CUM or cp['sha256']!={'model.pt':MODEL_SHA,'trainer.pt':TRAINER_SHA}:raise ValueError('7357来源元数据不符')
    report=read(bank/'report.json')
    import hashlib
    # 曲线可以换（本轮是 opponent-p50-slightly-weaker.json）：写死旧路径会让新库
    # 在这个检查里被误判为"库清单不符"（1093247）。默认仍是旧曲线，保持旧入口行为。
    curve_path=Path(os.environ.get('HSAI_P50_CURVE',ROOT/'config/opponent-p50-next.planned.json'))
    curve_digest=hashlib.sha256(curve_path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    if not report.get('complete') or report.get('smoke') or report.get('primary_statistic')!='P50(sum_attack+sum_health)':raise ValueError('需要正式P50库验收报告')
    for key in ('train-start','train-ramp1','train-ramp2','train','dev','test'):
        m=read(bank/key/'manifest.json')
        if m!=report['splits'][key] or m['driver']!='bot_firestone_p50_3702' or m.get('curve_sha256')!=curve_digest:raise ValueError('库清单不符:'+key)
        if not next((bank/key).glob('snap_*.json.gz'),None):raise ValueError('只有库元数据，没有快照:'+key)
    if require_gpu:
        gpu=read(Path(args.gpu_audit)/'report.json')
        if not gpu.get('passed') or gpu['source_sha256']!=cp['sha256'] or gpu['libraries']!={k:v['digest'] for k,v in report['splits'].items()}:raise ValueError('GPU验收来源/库不一致')
    return dict(passed=True,version=7357,scope='metadata_file_precheck_only',bank=str(bank),gpu_required=require_gpu)

def inputs(args):
    from hsbg.az.train.personal import checked_library,fingerprint,sha256,write_json,validate_config,training_source_digest
    from workflows.audit_full_trainer import audit
    check_inputs(args,require_gpu=not getattr(args,'preflight_mode',False))
    bank=Path(args.bot_3702).resolve();old=Path(args.source_run).resolve();source=Path(args.source_7357).resolve();out=Path(args.out).resolve()
    keys={'train':'train','hard':'dev','warm0':'train-start','warm1':'train-ramp1','warm2':'train-ramp2'}
    paths={k:bank/v for k,v in keys.items()};manifests={k:checked_library(p) for k,p in paths.items()}
    actual=fingerprint();report=read(bank/'report.json')
    if read(old/'plan.json')['config']!=read(old/'train.config.json'):raise ValueError('原实验配置证据不同')
    config=validate_config(read(ROOT/'config/p50-7357-train.json'))
    if config!=read(old/'train.config.json'):raise ValueError('本轮保留原PPO配方，发生意外参数变化')
    proof=audit(source,7357,{'model.pt':MODEL_SHA,'trainer.pt':TRAINER_SHA},device='cpu')
    if not getattr(args,'preflight_mode',False):
        gpu=read(Path(args.gpu_audit)/'report.json')
        if gpu['fingerprint']!=actual or gpu['training_source_sha256']!=training_source_digest():raise ValueError('GPU验收后代码或引擎改变，必须重验')
    frozen=out/'initial-state';frozen.mkdir(parents=True,exist_ok=True)
    for name,digest in proof['sha256'].items():
        target=frozen/name
        if target.exists():
            if sha256(target)!=digest:raise ValueError('冻结源不同，保留并停止')
        else:
            tmp=target.with_suffix('.pt.tmp');shutil.copyfile(source/name,tmp);os.replace(tmp,target)
    inherited=read(source/'checkpoint.json');parent=out/'provenance-source';parent.mkdir(exist_ok=True)
    # 真实来源报告和原计划摘要，不伪造上游plan.json或complete=True。
    write_json(parent/'report.json',read(old/'report.json'))
    write_json(parent/'source-fork.json',dict(source=inherited,source_run=str(old),source_plan_sha256=sha256(old/'plan.json'),source_report_sha256=sha256(old/'report.json'),restore_audit=proof,config=config))
    args.reference_checkpoint=str(frozen/'model.pt');args.train_schedule=SCHEDULE
    return parent,frozen,inherited,'p50-fixed-curriculum-v1',paths,manifests,config,{}

def main():
    p=defaults(argparse.ArgumentParser(description=__doc__))
    p.add_argument('--out',type=Path,default=os.environ.get('HSAI_LONG_OUT','artifacts/runs/p50-7357-long-001'))
    p.add_argument('--resume',action='store_true',default=os.environ.get('HSAI_RESUME')=='1');p.add_argument('--check-inputs-only',action='store_true');a=p.parse_args()
    if a.check_inputs_only:print(json.dumps(check_inputs(a),ensure_ascii=False,indent=2));return
    if not os.environ.get('SLURM_JOB_ID'):p.error('只能在Slurm计算节点运行')
    out=a.out
    if out.exists() and any(not x.name.startswith('visibility-') for x in out.iterdir()) and not a.resume:p.error('输出非空，必须显式resume或新目录')
    if a.resume and (out/'STOP_REQUESTED').exists():p.error('先重命名STOP_REQUESTED保留暂停记录')
    if (out/'report.json').exists() and read(out/'report.json').get('complete'):print('本轮已完成，不重复训练');return
    a.wall_minutes=1410;a.reserve_minutes=45;a.budget_decisions=1000000000;a.monitor_every=30000000;a.monitor_games=12;a.eval_games=64
    a.strict_mirror=True;a.curriculum_parent=None;a.continue_from=None;a.device='cuda:0';a._writer_acquired=False
    os.environ['HSRL_FAILURE_DIR']=str(out/'failure-states');os.environ['HSBG_ENGINE_GUARD_DIR']=str(out)
    from workflows.run_sustained_learning import run
    from workflows.run_curriculum_long import record_failure
    try:run(a)
    except Exception as exc:
        if a._writer_acquired:record_failure(out,exc)
        raise

if __name__=='__main__':main()
