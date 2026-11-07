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

def _sha_file(path):
    """原始字节 SHA256（权重文件；不做 CRLF 归一化）。流式读取，登录节点内存安全。"""
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _sha_lf(path):
    """按 LF 归一化后的 SHA256（与服务端 ``training_source_digest`` 同口径）。"""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()

def _training_source_sha(root=ROOT):
    """训练源码指纹：``hsbg/az/train/*.py`` + ``workflows/train_entity_sync.py``。"""
    paths=sorted((Path(root)/'hsbg/az/train').glob('*.py'))
    paths.append(Path(root)/'workflows/train_entity_sync.py')
    h=hashlib.sha256()
    for path in paths:
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes().replace(b'\r\n',b'\n'))
    return h.hexdigest()

def audit_problem(audit_path, *, root=ROOT, runner='workflows/run_forced_tier_long.py',
                  expected_libraries=None, expect_profile='forced'):
    """校验 GPU 验收报告；返回 ``''`` 表示通过，否则返回可操作的原因。

    纯函数（只读文件），登录节点可用；这样"报告缺失/用错 profile/源码变了/库变了"
    四类错误都能被单测覆盖，而不是等排队后才发现。
    """
    p=Path(audit_path)
    if not p.is_file():
        return (f'缺少GPU验收报告: {p}。本轮应先跑 `--phase forced-gpu --submit`，'
                '并用 HSAI_P50_AUDIT 指向同一个目录')
    try:
        gpu=read(p)
    except (OSError,ValueError) as exc:
        return f'GPU验收报告不可读: {p} ({exc})'
    if str(gpu.get('profile') or '')!=expect_profile:
        return (f'这份GPU验收报告不是 {expect_profile} 预检（profile={gpu.get("profile")}）：'
                '请指向 --phase forced-gpu 的输出目录，或重跑 --phase forced-gpu --submit')
    if gpu.get('training_source_sha256')!=_training_source_sha(root):
        return 'GPU验收后训练源码变化，不能提交长训：请重跑 --phase forced-gpu --submit'
    if gpu.get('runner_sha256')!=_sha_lf(Path(root)/runner):
        return '长训入口改变，需要重验：请重跑 --phase forced-gpu --submit'
    if expected_libraries is not None and gpu.get('libraries')!=expected_libraries:
        return 'GPU验收时的战团库与当前不一致（库被重建或替换过）：请重跑 --phase forced-gpu --submit'
    return ''

def eval_precheck(root=ROOT):
    """``--phase eval``：2 模型 + 6 bot 的 128 局配对评测的登录节点预检。

    只做**元数据/摘要**检查（不 import torch、不 import hsbg、不解析大 JSON）：

    - 协议配置存在且仍是用户批准的规则（128 局 / 80 动作 / 5 换位 / 去抖 2）；
    - A/B 两份权重存在、字节数与 SHA256 与配置一致、且不是同一份权重；
    - 输出目录不存在（保留失败证据，禁止覆盖旧基线）；
    - sbatch / common.sh / 启动器 / 评测入口都在；
    - 其它路径假设一律显式失败，不猜目录。
    """
    cfg_path = root / os.environ.get('HSAI_EVAL_CONFIG', 'config/eval-duel-7357-vs-10004.json')
    if not cfg_path.is_file():
        raise ValueError('缺少评测协议配置:' + str(cfg_path))
    cfg = read(cfg_path)
    approved = {'games': 128, 'max_actions_per_turn': 80, 'move_cap': 5,
                'move_repeat_limit': 2}
    for key, expected in approved.items():
        if int(cfg.get(key, -1)) != expected:
            raise ValueError(f'评测协议被改动：{key}={cfg.get(key)!r}，本轮批准值是 {expected}')
    if str(cfg.get('decode', 'greedy')) != 'greedy':
        raise ValueError('评测协议被改动：decode 必须是 greedy（采样会改变名次口径）')
    models = cfg.get('models') or {}
    shas = {}
    for label, env in (('A', 'HSAI_EVAL_MODEL_A'), ('B', 'HSAI_EVAL_MODEL_B')):
        entry = models.get(label) or {}
        path = Path(os.environ.get(env) or entry.get('path') or '')
        if str(path) and not path.is_absolute():
            # 相对路径按仓库根解析：登录节点与作业内 cwd 都是仓库根，单测也能指向临时树
            path = Path(root) / path
        if not str(path) or not path.is_file():
            raise ValueError(f'缺少模型 {label} 权重:{path}（先用 tools/stage_duel_models.py '
                             '准备并上传，或用 ' + env + ' 指定）')
        expected_bytes = int(entry.get('bytes') or 0)
        if expected_bytes and path.stat().st_size != expected_bytes:
            raise ValueError(f'模型 {label} 字节数不符: {path.stat().st_size} != {expected_bytes}')
        expected_sha = str(entry.get('file_sha256') or '')
        if not expected_sha:
            raise ValueError(f'协议配置缺少 models.{label}.file_sha256')
        actual = _sha_file(path)
        if actual != expected_sha:
            raise ValueError(f'模型 {label} SHA256 不符:{actual} != {expected_sha}')
        shas[label] = actual
    if shas['A'] == shas['B']:
        raise ValueError('A/B 两个模型是同一份权重，不能当作两模型评测')
    out = Path(os.environ.get('HSAI_EVAL_OUT', 'artifacts/runs/duel-eval-001'))
    if not out.is_absolute():
        out = Path(root) / out
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('评测输出目录已存在且非空，请改用新目录:' + str(out))
    script = root / 'deploy/slurm/p50_eval.sbatch'
    for required in (script, root / 'deploy/slurm/common.sh',
                     root / 'workflows/launch_nus_refinement.py',
                     root / 'workflows/eval_two_models_six_bots.py',
                     root / 'tools/stage_duel_models.py'):
        if not required.is_file():
            raise ValueError('部署文件缺失:' + str(required))
    return dict(passed=True, phase='eval', script=str(script),
                scope='metadata_file_precheck_only',
                config=str(cfg_path), games=int(cfg['games']),
                seed=int(cfg.get('seed', 0)), out=str(out),
                models={label: {'path': str(models[label]['path']), 'file_sha256': sha}
                        for label, sha in shas.items()})


def precheck(phase,root=ROOT):
    if phase=='eval':
        return eval_precheck(root)
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
            _libs={k:v['digest'] for k,v in read(bank/'report.json')['splits'].items()}
            _why=audit_problem(Path(args.gpu_audit)/'report.json',root=root,
                               expected_libraries=_libs)
            if _why:raise ValueError(_why+f'（验收目录={args.gpu_audit}）')
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['cpu','gpu','long','forced','forced-gpu','eval'],required=True);g=p.add_mutually_exclusive_group();g.add_argument('--submit',action='store_true');g.add_argument('--check-only',action='store_true');a=p.parse_args()
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
