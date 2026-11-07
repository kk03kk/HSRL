"""登录节点纯标准库预检；检查完成后才调用sbatch，避免排队数小时后路径秒失败。"""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def _stdlib_module(relpath,root=ROOT):
    """按文件路径加载一个纯标准库模块，**不 import 包**。

    登录节点与 ``python -S`` 都没有 numpy/torch，而 ``import hsbg.az…`` 会经过
    ``hsbg/az/__init__.py`` → ``belief.py`` → numpy。预检只需要 ``forced_tier``
    的解析器，所以直接按文件加载（该模块只依赖 re/dataclasses/typing）。

    注意：必须在 ``exec_module`` **之前**把模块登记进 ``sys.modules``。
    ``@dataclass`` 会通过 ``sys.modules[cls.__module__].__dict__`` 反查注解类型，
    未登记时在部分 Python 版本（如 3.12）会抛
    ``AttributeError: 'NoneType' object has no attribute '__dict__'``。
    """
    import importlib.util
    path=(Path(root)/relpath).resolve()
    if not path.is_file():raise ValueError('缺少预检模块:'+str(path))
    name='_hsai_stdlib_'+path.stem
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name,None)
        raise
    return module

def precheck(phase,root=ROOT):
    from workflows.run_7357_long import defaults,check_inputs,MODEL_SHA,TRAINER_SHA,SOURCE_CUM
    p=defaults(argparse.ArgumentParser());args=p.parse_args([])
    if phase in ('forced','forced-gpu'):
        # 强制升本长训：来源/父实验/现成库（不新建库）；配方由新 runner 在计算节点校验。
        needs_gpu_audit=phase=='forced'
        source=Path(args.source_7357)
        for name in ('checkpoint.json','model.pt','trainer.pt'):
            if not (source/name).is_file():raise ValueError('缺少7357输入:'+str(source/name))
        cp=read(source/'checkpoint.json')
        if cp['version']!=7357 or cp['actual_decisions']!=SOURCE_CUM or cp['sha256']!={'model.pt':MODEL_SHA,'trainer.pt':TRAINER_SHA}:raise ValueError('7357来源元数据不符')
        for name,size in [('model.pt',67864188),('trainer.pt',254312873)]:
            if (source/name).stat().st_size!=size:raise ValueError('7357文件长度不符:'+name)
        for name in ('plan.json','report.json','train.config.json'):
            if not (Path(args.source_run)/name).is_file():raise ValueError('缺少真实父实验文件:'+name)
        bank=Path(args.bot_3702)
        if not (bank/'report.json').is_file():raise ValueError('缺少P50库正式报告:'+str(bank/'report.json'))
        for key in ('train-start','train-ramp1','train-ramp2','train','dev','test'):
            if not (bank/key/'manifest.json').is_file():raise ValueError('缺少库清单:'+key)
            if not next((bank/key).glob('snap_*.json.gz'),None):raise ValueError('只有库元数据，没有快照:'+key)
        cfg=read(root/'config/p50-forced-tier.json')
        _ft=_stdlib_module('hsbg/az/train/forced_tier.py',root)
        if not _ft.parse_forced_tier_plan(str(cfg.get('forced-tier-plan',''))):raise ValueError('配置缺少强制升本计划')
        _ft.parse_prob_schedule(str(cfg.get('forced-tier-schedule','')))
        if int(cfg.get('reset-anchor',0))!=1:raise ValueError('配置必须 reset-anchor=1（修正继承的旧 anchor）')
        if float(cfg.get('anchor-beta',0.0))>0.05:raise ValueError('anchor-beta 过大')
        cpu_path=root/'artifacts/local/p50-cpu-validation.json'
        if not cpu_path.is_file():raise ValueError('缺少CPU门禁结果:'+str(cpu_path)+'（先跑 --phase cpu）')
        if not read(cpu_path).get('passed'):raise ValueError('CPU完整门禁没有通过')
        if needs_gpu_audit:
            gpu=read(Path(args.gpu_audit)/'report.json')
            paths=sorted((root/'hsbg/az/train').glob('*.py'))+[root/'workflows/train_entity_sync.py'];h=hashlib.sha256()
            for path in paths:h.update(path.relative_to(root).as_posix().encode());h.update(path.read_bytes().replace(b'\r\n',b'\n'))
            if gpu['training_source_sha256']!=h.hexdigest():raise ValueError('GPU验收后训练源码变化，不能提交长训')
            if gpu.get('runner_sha256')!=hashlib.sha256((root/'workflows/run_forced_tier_long.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest():raise ValueError('长训入口改变，需要重验')
        result=dict(passed=True,phase=phase,scope='metadata_file_precheck_only',
                    config=str(root/'config/p50-forced-tier.json'),needs_gpu_audit=needs_gpu_audit)
    elif phase=='cpu':
        source=Path(args.source_7357)
        for name in ('checkpoint.json','model.pt','trainer.pt'):
            if not (source/name).is_file():raise ValueError('缺少7357输入:'+str(source/name))
        cp=read(source/'checkpoint.json')
        if cp['version']!=7357 or cp['actual_decisions']!=SOURCE_CUM or cp['sha256']!={'model.pt':MODEL_SHA,'trainer.pt':TRAINER_SHA}:raise ValueError('7357来源元数据不符')
        for name,size in [('model.pt',67864188),('trainer.pt',254312873)]:
            if (source/name).stat().st_size!=size:raise ValueError('7357文件长度不符:'+name)
        for name in ('plan.json','report.json','train.config.json'):
            if not (Path(args.source_run)/name).is_file():raise ValueError('缺少真实父实验文件:'+name)
        if read(Path(args.source_run)/'train.config.json')!=read(root/'config/p50-7357-train.json'):raise ValueError('原配方与新配置意外不同')
        old=Path(os.environ.get('HSAI_OLD_BANK','../HSRL/personal/artifacts/libraries/bot-firestone-3702-economy-v2-live-v1'))
        for split in ('train','dev','test'):
            for file in (old/split/'manifest.json',old/'raw'/split/'frames'):
                if not file.exists():raise ValueError('缺少原始Bot源或库清单:'+str(file))
            if not next((old/'raw'/split/'frames').glob('snap_*.json.gz'),None):raise ValueError('原始Bot快照为空:'+split)
        if Path(args.bot_3702).exists():raise ValueError('新库输出已存在；保留证据，设置HSAI_P50_BANK为新路径')
    else:
        result=check_inputs(args,require_gpu=phase=='long')
        cpu=read(root/'artifacts/local/p50-cpu-validation.json')
        if not cpu.get('passed'):raise ValueError('CPU完整门禁没有通过')
        if phase=='long':
            from workflows.run_7357_long import read as same_read
            gpu=read(Path(args.gpu_audit)/'report.json')
            paths=sorted((root/'hsbg/az/train').glob('*.py'))+[root/'workflows/train_entity_sync.py'];h=hashlib.sha256()
            for path in paths:h.update(path.relative_to(root).as_posix().encode());h.update(path.read_bytes().replace(b'\r\n',b'\n'))
            if gpu['training_source_sha256']!=h.hexdigest():raise ValueError('GPU验收后训练源码变化，不能提交长训')
            if gpu.get('runner_sha256')!=hashlib.sha256((root/'workflows/run_7357_long.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest():raise ValueError('长训入口改变，需要重验')
    gpu_phase = phase in ('gpu','forced-gpu')
    out=Path(os.environ.get('HSAI_P50_AUDIT' if gpu_phase else 'HSAI_LONG_OUT',
                            'artifacts/runs/p50-forced-tier-preflight' if phase=='forced-gpu'
                            else 'artifacts/runs/p50-7357-preflight' if phase=='gpu'
                            else ('artifacts/runs/p50-forced-tier-long-001' if phase=='forced'
                                  else 'artifacts/runs/p50-7357-long-001')))
    if phase!='cpu' and out.exists():
        non_visibility=[p for p in out.iterdir() if not p.name.startswith('visibility-') and p.name!='pool-evidence']
        if non_visibility and not (phase in ('long','forced') and os.environ.get('HSAI_RESUME')=='1'):raise ValueError('输出目录非空，保留失败证据；使用新目录或显式恢复:'+str(out))
    if phase in ('long','forced') and (out/'STOP_REQUESTED').exists():raise ValueError('恢复前先重命名STOP_REQUESTED并保留记录')
    script=root/'deploy/slurm'/('p50_cpu.sbatch' if phase=='cpu'
                                else 'p50_preflight.sbatch' if phase=='gpu'
                                else 'p50_forced_preflight.sbatch' if phase=='forced-gpu'
                                else 'p50_forced.sbatch' if phase=='forced'
                                else 'p50_long.sbatch')
    for required in (script,root/'deploy/slurm/common.sh',root/'workflows/launch_nus_refinement.py'):
        if not required.is_file():raise ValueError('部署文件缺失:'+str(required))
    return dict(passed=True,phase=phase,script=str(script),scope='bounded_standard_library_metadata_no_model_loading')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['cpu','gpu','long','forced','forced-gpu'],required=True);g=p.add_mutually_exclusive_group();g.add_argument('--submit',action='store_true');g.add_argument('--check-only',action='store_true');a=p.parse_args()
    os.chdir(ROOT)
    try:result=precheck(a.phase)
    except (OSError,ValueError,KeyError) as exc:p.exit(2,"PRECHECK FAILED: "+str(exc)+"\n")
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if a.submit:
        (ROOT/'artifacts/runs/slurm').mkdir(parents=True,exist_ok=True)
        subprocess.run(['bash','-n',result['script']],check=True)
        subprocess.run(['sbatch','--test-only',result['script']],check=True)
        subprocess.run(['sbatch','--parsable',result['script']],check=True)
if __name__=='__main__':main()
