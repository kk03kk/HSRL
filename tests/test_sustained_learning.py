"""关键回归：两个面板都显著下降也继续预算；工程失败仍停止。"""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from workflows.run_learning_experiment import read, write, digest
from tests.test_learning_experiment import panel


def setup(tmp_path):
    from hsbg.az.train.personal import fingerprint, library_digest
    parent = tmp_path/'parent'
    for name, split in [('train-x1.5', 'train'), ('dev-control', 'dev'), ('dev-x1.5', 'dev')]:
        lib = parent/'libraries'/name
        lib.mkdir(parents=True)
        (lib/'snap_0001.json.gz').write_bytes(b'immutable')
        write(lib/'manifest.json', dict(split=split, digest=library_digest(lib), fingerprint=fingerprint(), validation=dict(accepted=1)))
    source = parent/'last-committed/step_000005'
    source.mkdir(parents=True)
    (source/'model.pt').write_bytes(b'parent-model')
    (source/'trainer.pt').write_bytes(b'parent-adam-rng-anchor')
    record = dict(version=5, actual_decisions=1000, target_decisions=1000,
        sha256={n: digest(source/n) for n in ('model.pt', 'trainer.pt')})
    write(source/'checkpoint.json', record)
    write(parent/'report.json', dict(complete=True, stopped=True, selected=dict(factor=1.5),
        latest=dict(version=5, cum_decisions=1000)))
    write(parent/'training-report.json', dict(totals=dict(illegal_actions=0, truncated_games=0, mirror_errors=0), unexpected_mirror_rejections={}))
    write(parent/'train.config.json', dict(games=16, minutes=30, iterations=0, **{'snapshot-every':1}))
    reference = tmp_path/'reference.pt'
    reference.write_bytes(b'B100k-reference')
    write(parent/'plan.json', dict(checkpoint=digest(reference)))
    return SimpleNamespace(parent=str(parent), out=str(tmp_path/'sustained'), reference_checkpoint=str(reference),
        budget_decisions=60, monitor_every=20, monitor_games=1, eval_games=1, wall_minutes=540, reserve_minutes=75, device='cpu')


class Commands:
    def __init__(self, bad_quality=False, fail_eval=False):
        self.calls = []
        self.bad_quality, self.fail_eval = bad_quality, fail_eval
    def __call__(self, argv):
        from hsbg.az.train.personal import checked_library, training_source_digest
        self.calls.append(argv)
        def arg(name): return argv[argv.index(name)+1]
        out = Path(arg('--out'))
        if argv[3] == 'eval':
            if self.fail_eval and 'milestones' in str(out):
                self.fail_eval = False
                raise RuntimeError('模拟评测中断')
            model = Path(arg('--checkpoint'))
            value = panel(digest(model), 2 if model.name == 'reference.pt' else 7)
            library = checked_library(arg('--library'))
            value['protocol'].update(library_digest=library['digest'], fingerprint=library['fingerprint'],
                training_source_sha256=training_source_digest(), seed=int(arg('--seed')))
            for row in value['per_seat']:
                row['seed_start'] = row['seed_end'] = int(arg('--seed'))
            write(out, value)
            return
        current = read(out/'latest.json') if (out/'latest.json').exists() else dict(version=5, cum_decisions=1000)
        version, count = current['version'], current['cum_decisions']
        while count < int(arg('--stop-at-decisions')):
            count += 10
            version += 1
            snap = out/'snapshots'/f'step_{version:06d}'
            snap.mkdir(parents=True)
            (snap/'model.pt').write_bytes(f'weights-{version}'.encode())
            (snap/'trainer.pt').write_bytes(f'full-state-{version}'.encode())
            with (out/'history.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(iter=version-1, collect=dict(decisions=10, illegal_actions=int(self.bad_quality)),
                    ppo=dict(processed_samples=10, updates=1, epochs=[dict(total=1., entropy=.5)]),
                    profile=dict(collect_s=1., ppo_s=1.)))+'\n')
            write(out/'latest.json', dict(version=version, cum_decisions=count))


def test_bad_performance_in_both_panels_uses_full_budget_and_evaluates_latest(tmp_path):
    from workflows.run_sustained_learning import run
    args = setup(tmp_path)
    commands = Commands()
    result = run(args, commands)
    assert result['budget_complete'] and not result['stopped_for_performance']
    assert result['latest']['cum_decisions'] == 1060
    assert len(result['stages']) == 3
    assert all(s['paired']['bot']['ci_low'] > 0 and s['paired']['hard']['ci_low'] > 0 for s in result['stages'])
    assert result['final_evaluations']['latest']['hard']['mean_rank_after'] == 7
    trains = [a for a in commands.calls if a[3] == 'train']
    assert '--initial-trainer-state' in trains[0] and all('--resume' in a for a in trains[1:])
    count = len(commands.calls)
    run(args, commands)
    assert len(commands.calls) == count


def test_engineering_corruption_still_stops(tmp_path):
    from workflows.run_sustained_learning import run
    args = setup(tmp_path)
    commands = Commands(bad_quality=True)
    with pytest.raises(RuntimeError, match='质量门'):
        run(args, commands)
    assert len([a for a in commands.calls if a[3] == 'train']) == 1


def test_user_stop_preserves_checkpoint_and_resumes_when_request_removed(tmp_path):
    from workflows.run_sustained_learning import run
    args=setup(tmp_path)
    commands=Commands()
    def stop_after_first_train(argv):
        commands(argv)
        if argv[3]=='train':
            (Path(args.out)/'STOP_REQUESTED').touch()
    result=run(args,stop_after_first_train)
    assert result['state']=='paused' and not result['complete']
    assert result['latest']['cum_decisions']==1020
    assert len([a for a in commands.calls if a[3]=='train'])==1
    version=result['latest']['version']
    assert (Path(args.out)/'last-committed'/f'step_{version:06d}'/'trainer.pt').exists()
    count=len(commands.calls)
    run(args,commands)
    assert len(commands.calls)==count
    (Path(args.out)/'STOP_REQUESTED').unlink()
    resumed=run(args,commands)
    assert resumed['budget_complete'] and resumed['latest']['cum_decisions']==1060


def test_interrupted_evaluation_resumes_without_retraining_or_touching_parent(tmp_path):
    from workflows.run_sustained_learning import run
    args = setup(tmp_path)
    originals = {p: digest(p) for p in Path(args.parent).rglob('*') if p.is_file()}
    commands = Commands(fail_eval=True)
    with pytest.raises(RuntimeError, match='评测中断'):
        run(args, commands)
    previous = len([a for a in commands.calls if a[3] == 'train'])
    run(args, commands)
    assert len([a for a in commands.calls if a[3] == 'train']) == previous+2
    assert all(digest(p) == sha for p, sha in originals.items())


def test_anchor_schedule_changes_only_in_new_fork(tmp_path):
    from workflows.run_sustained_learning import run
    args = setup(tmp_path)
    original = read(Path(args.parent)/'train.config.json')
    args.anchor_iterations, args.anchor_beta_tail = 250, .02
    run(args, Commands())
    assert read(Path(args.parent)/'train.config.json') == original
    cfg = read(Path(args.out)/'train.config.json')
    assert cfg['anchor-iterations'] == 250 and cfg['anchor-beta-tail'] == .02
