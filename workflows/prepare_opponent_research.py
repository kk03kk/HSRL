#!/usr/bin/env python3
"""有界准备实验：冻结三代模型、生成完整快照，再校准/评测/验证PPO；不启动长训。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows.run_learning_experiment import read, write, digest, writer_lock, training_report
from workflows.run_overnight_learning import check_eval

MODELS = {
    'best': ('artifacts/models/s14_best_main_model.pt',
             'fffc395c4a24207acd2d297f0b4ee25e7aabde66e7249dd7a71cd388b58f500c'),
    'b100k': ('artifacts/runs/nus-learning-001/experiment/target-000100000/seed-200000000/model.pt',
              '0858209b4edba906016adcd7584733f4891539372540bcc8562571eb47d64ba0'),
    'b10m': ('artifacts/runs/nus-sustained-001/last-committed/step_002328/model.pt',
             '5e08ae694583491f2b3cc8a0efd9852d18d66f352981a4fd145b111b94aea987'),
}
TRAINER_SHA = '9f897ce9886291d3b1c61d11e1dfd67e01f32090b1ec1fe3cc7aa5540fa59f6c'
HUMAN_SHA = 'b142d7d6e99ad49c310fccb542497f01649cb12a654f16ba907db6e841710330'
ARCHIVE_SHA = 'acb5420c4ef32390dacb58b5392e33c4535f465219411794cde221149f149b2a'


def ensure_human_train(path):
    """计算节点上按摘要恢复Train单文件；不展开Test或信任归档内的路径。"""
    import os
    import shutil
    import tarfile
    path = Path(path)
    if path.exists():
        return
    archive = ROOT/'artifacts/exports/firestone-20261102-v1.tar.gz'
    if not archive.exists() or digest(archive) != ARCHIVE_SHA:
        raise FileNotFoundError('缺Firestone Train；请上传已冻结15MB归档到artifacts/exports，不在登录节点解析')
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as stream:
        members = [m for m in stream.getmembers() if Path(m.name).name == 'train.jsonl' and m.isfile()]
        if len(members) != 1:
            raise ValueError('归档Train文件不唯一')
        temp = path.with_suffix('.extract.tmp')
        with stream.extractfile(members[0]) as source, temp.open('wb') as target:
            shutil.copyfileobj(source, target)
        if digest(temp) != HUMAN_SHA:
            raise ValueError('归档Train摘要不符，临时文件保留')
        os.replace(temp, path)


def run(args):
    from hsbg.az.train.personal import checked_library, training_source_digest
    from workflows.prepare_validated_library import prepare
    out = Path(args.out).resolve()
    deadline = time.monotonic()+args.minutes*60
    def command(*argv):
        remaining = deadline-time.monotonic()
        if remaining < 60:
            raise TimeoutError('准备实验时间预算耗尽；保留已完成阶段，不启动长训')
        cmd = [sys.executable, *map(str, argv)]
        print('[run] '+' '.join(cmd), flush=True)
        from workflows.run_overnight_learning import run_process
        # 超时回收整组子进程，避免模型采集器/硬门worker在父进程结束后继续占卡。
        run_process(cmd, deadline)
    human = Path(args.human).resolve()
    ensure_human_train(human)
    if digest(human) != HUMAN_SHA:
        raise ValueError('需要冻结Firestone v1 Train，不能替换为Dev/Test或新版')
    models = {name:ROOT/path for name, (path, _) in MODELS.items()}
    for name, path in models.items():
        if digest(path) != MODELS[name][1]:
            raise ValueError(f'{name}源模型摘要不符')
    trainer = models['b10m'].with_name('trainer.pt')
    if digest(trainer) != TRAINER_SHA:
        raise ValueError('B10M完整trainer摘要不符')
    with writer_lock(out):
        import torch
        import platform
        plan = dict(schema='hsrl.opponent_research_plan/v1', source=training_source_digest(),
            scripts={name:digest(ROOT/'workflows'/name) for name in
                     ('prepare_opponent_research.py', 'calibrate_opponent_bank.py', 'mix_opponent_banks.py',
                      'capture_trajectory.py', 'benchmark_checkpoint_publish.py', 'summarize_opponent_research.py',
                      'prepare_validated_library.py', 'mirror_hard_gates.py', 'compare_board_libraries.py')},
            models={k:digest(v) for k,v in models.items()}, trainer=digest(trainer), human=digest(human),
            train_games=args.train_games, dev_games=args.dev_games, device=args.device,
            eval_games=args.eval_games, per_source=args.per_source, trials=args.trials,
            training_config=digest(args.training_config),
            runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,os=platform.system()))
        if (out/'plan.json').exists() and read(out/'plan.json') != plan:
            raise ValueError('准备计划漂移，请使用新目录，不覆盖旧证据')
        write(out/'plan.json', plan)
        if args.phase == 'capture':
            for name in (*models, 'bot'):
                for split, games in [('train', args.train_games), ('dev', args.dev_games)]:
                    target = out/'sources'/name/split
                    if not (target/'manifest.json').exists():
                        root = out/'raw'/name/split
                        if root.exists():
                            raise ValueError(f'未完成源采集需保留并另开输出目录：{root}')
                        command('-m', 'hsbg.az.train.personal', 'build-library', '--driver', 'bot' if name == 'bot' else 'model',
                            '--checkpoint', models[name] if name in models else '', '--device', args.device,
                            '--games', games, '--seed-base', 20000000, '--splits', split, '--out', root)
                        trajectories = [read_trajectory(p) for p in (root/'trajectories'/split).glob('traj_*.json.gz')]
                        if len(trajectories) != games or any(r['meta']['truncated'] for r in trajectories):
                            raise RuntimeError('源游戏不完整，不允许进入建库')
                        prepare(root/split, target, derived_dev=split == 'dev')
                    manifest = checked_library(target)
                    expected_model = MODELS[name][1] if name in models else None
                    if manifest['source_checkpoint'] != expected_model or manifest['games'] != games:
                        raise ValueError('已完成源库与冻结模型/游戏数不同')
            write(out/'capture-report.json', dict(complete=True, models=plan['models'],
                games_per_source=dict(train=args.train_games, dev=args.dev_games), test_generated=False,
                libraries={name:{split:checked_library(out/'sources'/name/split)['digest']
                                  for split in ('train','dev')} for name in (*models,'bot')}))
            return
        captured = read(out/'capture-report.json')
        if not captured['complete']:
            raise ValueError('需要先完成capture阶段')
        for name, splits in captured['libraries'].items():
            for split, expected in splits.items():
                if checked_library(out/'sources'/name/split)['digest'] != expected:
                    raise ValueError('采集后源库漂移，禁止复用旧校准')
        if not (out/'checkpoint-benchmark.json').exists():
            command('workflows/benchmark_checkpoint_publish.py', '--model', models['b10m'], '--trainer', trainer,
                    '--device', args.device, '--out', out/'checkpoint-benchmark.json')
        for split in ('train', 'dev'):
            mapping = {name:str(out/'sources'/name/split) for name in (*models, 'bot')}
            write(out/f'{split}-sources.json', mapping)
            cal = out/'calibrated'/split
            if not (cal/'manifest.json').exists():
                command('workflows/calibrate_opponent_bank.py', '--sources', out/f'{split}-sources.json',
                        '--human', human, '--out', cal, '--split', split,
                        '--per-source', args.per_source, '--trials', args.trials)
            validated = out/'validated-calibrated'/split
            if not (validated/'manifest.json').exists():
                prepare(cal, validated, derived_dev=split == 'dev')
            from workflows.mix_opponent_banks import mix
            for name, quotas in [('adapt', dict(control=1, calibrated=1)), ('main', dict(control=1, calibrated=3))]:
                mixture = out/'mixtures'/name/split
                if not (mixture/'manifest.json').exists():
                    mix(dict(control=out/'sources/bot'/split, calibrated=validated), quotas, mixture, split)
                final = out/'validated-mixtures'/name/split
                if not (final/'manifest.json').exists():
                    prepare(mixture, final, derived_dev=split == 'dev')
        # 原有成熟硬门：不重复全量Rust/pytest；只验本次新训练库。
        for name in ('adapt', 'main'):
            audit = out/f'{name}-mirror-audit.json'
            if not audit.exists():
                command('workflows/mirror_hard_gates.py', '--snaps', out/'validated-mixtures'/name/'train',
                    '--sample', 'random', '--seeds', 16, '--min-injections', 1, '--p-values', 1,
                    '--seat-values', 7, '--baseline-games', 16, '--workers', 4, '--out', audit)
            if not read(audit)['zero_violations']:
                raise RuntimeError('新库完整对局硬门失败')
        panels = {'natural-b10m':out/'sources/b10m/dev', 'calibrated':out/'validated-calibrated/dev',
                  'mixture':out/'validated-mixtures/main/dev'}
        evaluated = {}
        # 三代都在同条件新Dev比较；每面板64局是压力定位，不作为晋级资格。
        for panel, library in panels.items():
            evaluated[panel] = {}
            for name, model in models.items():
                path = out/'evaluations'/f'{panel}-{name}.json'
                if not path.exists():
                    command('-m', 'hsbg.az.train.personal', 'eval', '--checkpoint', model, '--library', library,
                            '--games', args.eval_games, '--seed', 40000000, '--device', args.device, '--exposure', 1, '--out', path)
                value = read(path)
                check_eval(value, digest(model))
                protocol = value['protocol']
                if protocol['library_digest'] != checked_library(library)['digest'] or \
                        protocol['training_source_sha256'] != training_source_digest() or \
                        protocol['seed'] != 40000000 or protocol['games_per_seat'] != args.eval_games:
                    raise ValueError('压力评测缓存协议不匹配')
                evaluated[panel][name] = {k:value[k] for k in ('mean_rank', 'win_rate', 'top4_rate')}
        # 仅2轮完整状态分叉验证；不以短期名次/CI阻止后续长训。
        cfg = read(args.training_config)
        # 先物化下一轮可用的原生训练配置；这里不启动其长期预算。
        next_cfg = dict(cfg, iterations=0, minutes=30)
        write(out/'next-training.config.json', next_cfg)
        cfg.update(iterations=2, minutes=10)
        write(out/'ppo-check.config.json', cfg)
        train = out/'ppo-check'
        if not (train/'summary.json').exists():
            state_args = ['--resume'] if (train/'latest.json').exists() else [
                '--checkpoint', models['b10m'], '--initial-trainer-state', trainer]
            command('-m', 'hsbg.az.train.personal', 'train', '--config', out/'ppo-check.config.json',
                *state_args,
                '--library', out/'validated-mixtures/main/train', '--device', args.device, '--out', train)
        quality = training_report(train, out/'ppo-check-report', read(train/'latest.json')['version'])
        if quality.get('unexpected_mirror_rejections'):
            raise RuntimeError('小规模PPO出现非预期拒绝')
        from workflows.summarize_opponent_research import summarize
        report = summarize(out)
        report.update(complete=True, evaluations=evaluated, long_training_submitted=False)
        write(out/'next-training-proposal.json', dict(report['next_training_proposal'],
            model_sha256=plan['models']['b10m'], trainer_sha256=plan['trainer'],
            config_sha256=digest(out/'next-training.config.json'),
            library_digests={name:checked_library(out/'validated-mixtures'/name/'train')['digest']
                             for name in ('adapt','main')}))
        write(out/'report.json', report)


def read_trajectory(path):
    import gzip
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['capture', 'analyze'], required=True)
    p.add_argument('--out', default='artifacts/runs/opponent-research-001')
    p.add_argument('--human', default='artifacts/libraries/firestone-20261102-v1/train.jsonl')
    p.add_argument('--device', default='cuda')
    p.add_argument('--train-games', type=int, default=32)
    p.add_argument('--dev-games', type=int, default=8)
    p.add_argument('--minutes', type=int, default=240)
    p.add_argument('--training-config', default='artifacts/runs/nus-sustained-001/train.config.json')
    p.add_argument('--eval-games', type=int, default=8)
    p.add_argument('--per-source', type=int, default=3)
    p.add_argument('--trials', type=int, default=8)
    a = p.parse_args()
    if not 1 <= a.train_games <= 256 or not 1 <= a.dev_games <= 64 or not 1 <= a.minutes <= 300:
        p.error('建库train 1..256、dev 1..64、时间1..300分钟')
    if not 1 <= a.eval_games <= 32 or not 1 <= a.per_source <= 8 or not 1 <= a.trials <= 32:
        p.error('评测games 1..32，候选per-source 1..8，战斗trials 1..32')
    run(a)
