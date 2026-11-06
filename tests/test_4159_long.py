"""正式入口必须取4159完整源，不能误取4162探针；详细评测不改变对局。"""
from pathlib import Path
from types import SimpleNamespace

import pytest


def prepared(tmp_path, monkeypatch):
    from hsbg.az.train import personal
    from workflows import run_3702_preflight as preflight
    from workflows.run_learning_experiment import write, digest
    audit, bank = tmp_path/'audit',tmp_path/'bank'
    config={**preflight.ACTION_PROFILE,'games':512,'minutes':30,'processes':8,'pipeline':1,
            'mirror-strict':1,'mirror-sample':'random','epochs':1,'minibatch':1024,
            'grad-accum':1,'snapshot-every':1,'lr':2e-6,
            'gpu-index-space':'visible','gpus':'1'}
    fp={'engine':'audited-native'}
    manifests={split:dict(fingerprint=fp,digest=split) for split in ('train','dev')}
    bank_report=dict(complete=True,smoke=False,fresh_generation=True,splits=manifests)
    source=dict(version=4159,actual_decisions=198819917,sha256={'model.pt':preflight.COMPARISON_SHA,'trainer.pt':'original-adam'})
    outputs={}
    for name in ('model.pt','trainer.pt'):
        write(audit/'initial-state'/name, {'name':name,'version':4159})
        outputs[name]=digest(audit/'initial-state'/name)
    # Probe权重刻意不同：正式入口绝不能从train/latest或4162目录拷贝。
    write(audit/'train/snapshots/step_004162/model.pt', {'version':4162})
    migration=dict(inputs=source['sha256'],outputs=outputs,fingerprint=fp)
    write(audit/'initial-state/migration.json',migration)
    write(audit/'source-fork.json',dict(source=source,migration=migration,config=config))
    write(audit/'train.config.json',config)
    evidence=dict(passed=True,source=source,latest={'version':4162},fingerprint=fp,bank=bank_report,
                  training=dict(totals=dict(processed_samples=324117,optimizer_steps=318,
                    engine_quality_checked_games=1536,mirror_attempt=129730,mirror_applied=129730),
                    mirror_strict_counts={'eligible_live_seats':129730},unexpected_mirror_rejections={}))
    write(audit/'report.json',evidence)
    write(bank/'report.json',bank_report)
    for split,manifest in manifests.items():write(bank/split/'manifest.json',manifest)
    monkeypatch.setattr(personal,'fingerprint',lambda:fp)
    monkeypatch.setattr(personal,'checked_library',lambda p:manifests[p.name])
    return SimpleNamespace(source_4159_audit=audit,bot_3702=bank,out=tmp_path/'long'),evidence


def test_start_and_resume_keep_verified_initial_trainer(tmp_path,monkeypatch):
    from workflows.run_4159_long import inputs
    from workflows.run_learning_experiment import digest,read
    args,_=prepared(tmp_path,monkeypatch)
    result=inputs(args)
    assert result[2]['version']==4159
    assert result[2]['actual_decisions']==198819917
    assert result[6]['move-cap']==5 and result[6]['lr']==2e-6
    assert read(result[1]/'model.pt')['version']==4159
    original=digest(result[1]/'trainer.pt')
    inputs(args)
    assert digest(result[1]/'trainer.pt')==original
    assert args.reference_checkpoint==str(result[1]/'model.pt')


@pytest.mark.parametrize('corrupt',['source','engine','injection','quality','config','weights','bank'])
def test_refuse_changed_source_or_failed_engineering(tmp_path,monkeypatch,corrupt):
    from workflows.run_4159_long import inputs
    from workflows.run_learning_experiment import write,read
    args,evidence=prepared(tmp_path,monkeypatch)
    if corrupt=='source':evidence['source']['version']=4162
    if corrupt=='engine':evidence['fingerprint']={'engine':'old'}
    if corrupt=='injection':evidence['training']['totals']['mirror_applied']-=1
    if corrupt=='quality':evidence['training']['totals']['truncated_combats']=1
    if corrupt=='config':
        config=read(args.source_4159_audit/'train.config.json');config['move-cap']=14
        write(args.source_4159_audit/'train.config.json',config)
    if corrupt=='weights':write(args.source_4159_audit/'initial-state/trainer.pt',{'reset':True})
    if corrupt=='bank':write(args.bot_3702/'report.json',{'changed':True})
    write(args.source_4159_audit/'report.json',evidence)
    with pytest.raises(ValueError):inputs(args)


def test_detailed_native_eval_keeps_game_results_and_complete_boards(tmp_path):
    from hsbg.az.train.evaluate import evaluate_policy,EvalConfig
    from hsbg.az.train.policies import EngineBotPolicy
    from tests.test_injection_meta import _synthetic_mirror_pool
    from workflows.report_firestone_learning import by_turn
    pool=_synthetic_mirror_pool(tmp_path)
    base=dict(games=4,seed=881200,mirror_pool=pool,mirror_strict=True)
    plain=evaluate_policy(EngineBotPolicy(),EvalConfig(**base))
    detail=evaluate_policy(EngineBotPolicy(),EvalConfig(**base,collect_exposure=True,collect_economy=True))
    assert plain['rank_sequence']==detail['rank_sequence']
    assert plain['decisions']==detail['decisions']
    rows=detail['end_turn_economy']
    assert rows and any(r['gold_events'] for r in rows)
    for r in rows:
        assert len(r['board'])<=7
        assert r['eot_available']
        assert r['eot_board_stage']=='recruitment_end_before_combat'
        assert sum(m['atk']+m['hp'] for m in r['eot_board'])==r['eot_attack']+r['eot_board_hp']
        assert sum(m['atk'] for m in r['board'])==r['attack']
        assert sum(m['hp'] for m in r['board'])==r['board_hp']
        assert all('name' in m and 'card_id' in m for m in r['board'])
        assert all(e['after']-e['before']==e['net_change'] for e in r['gold_events'])
    assert all(r['ghost'] or r['opponent_board'] is not None for r in detail['combat_exposure'])
    report=dict(valid=True,per_seat=[detail])
    assert by_turn(report)[2]['mean_board_count']>=0


def test_long_cli_budget_and_resume_switch(tmp_path,monkeypatch):
    from workflows import run_4159_long as entry,run_sustained_learning as runner
    monkeypatch.setenv('SLURM_JOB_ID','local-test')
    monkeypatch.setattr(entry.sys,'argv',['runner','--out',str(tmp_path/'new')])
    calls=[]
    monkeypatch.setattr(runner,'run',lambda a:calls.append(a))
    entry.main()
    args=calls[0]
    assert args.wall_minutes==1410 and args.reserve_minutes==45
    assert args.monitor_every==30000000 and args.monitor_games*8==96 and args.eval_games*8==512


@pytest.mark.parametrize('pause', [True,False])
def test_real_entry_and_real_provider_reach_training_then_pause_and_resume_without_audit_plan(tmp_path,monkeypatch,pause):
    """只替代外部GPU子进程；真实main/provider/controller/文件冻结完整连通。"""
    from workflows import run_4159_long as entry,run_sustained_learning as runner
    from workflows.run_learning_experiment import read,write,digest
    from hsbg.az.train import runlog,personal
    from workflows.run_3702_preflight import ACTION_PROFILE
    args,evidence=prepared(tmp_path,monkeypatch)
    audit,bank,out=Path(args.source_4159_audit),Path(args.bot_3702),Path(args.out)
    assert not (audit/'plan.json').exists()
    originals={str(p.relative_to(audit)):digest(p) for p in audit.rglob('*') if p.is_file()}
    monkeypatch.setenv('SLURM_JOB_ID','local-controller-test')
    argv=['runner','--out',str(out),'--source-4159-audit',str(audit),'--bot-3702',str(bank)]
    monkeypatch.setattr(entry.sys,'argv',argv)
    original_run=runner.run
    def bounded_run(a):
        a.budget_decisions=60000000
        return original_run(a)
    monkeypatch.setattr(runner,'run',bounded_run)
    monkeypatch.setattr(runlog,'recover_committed_record',lambda *_:None)
    monkeypatch.setattr(runner,'learning_activity',lambda _:dict(optimizer_steps=1,trailing_zero_update_iterations=0))
    monkeypatch.setattr(runner,'training_report',lambda *_:dict(
        totals=dict(decisions=read(out/'train/latest.json')['cum_decisions']-198819917,
                    mirror_fallback=0,mirror_applied=10),
        mirror_strict_counts=dict(eligible_live_seats=10,live_combats=1)))
    calls=[]
    def execute(command, deadline):
        def value(flag):return command[command.index(flag)+1]
        mode=command[3];calls.append(mode)
        if mode=='eval':
            protocol={k.replace('-','_'):v for k,v in ACTION_PROFILE.items()}
            games=int(value('--games'));seed=int(value('--seed'))
            protocol.update(library_digest='dev',fingerprint=evidence['fingerprint'],
                training_source_sha256=personal.training_source_digest(),seed=seed,games_per_seat=games,
                seats=list(range(8)),mirror_strict=True)
            write(value('--out'),dict(valid=True,checkpoint={'sha256':digest(value('--checkpoint'))},
                protocol=protocol,protocol_sha256='fixed',mean_rank=4,top4_rate=.5,win_rate=.1,
                per_seat=[dict(policy_seat=s,rank_sequence=[4]*games,end_turn_economy=[],combat_exposure=[],
                              seed_start=seed,seed_end=seed+games-1) for s in range(8)]))
            return
        assert mode=='train' and (out/'baselines/diagnostics.json').exists()
        latest=out/'train/latest.json'
        if latest.exists():
            assert '--resume' in command
            previous=read(latest)
        else:
            assert Path(value('--initial-trainer-state'))==out.resolve()/'initial-state/trainer.pt'
            previous=dict(version=4159,cum_decisions=198819917)
        version=previous['version']+1
        snap=out/'train/snapshots'/f'step_{version:06d}'
        for name in ('model.pt','trainer.pt'):write(snap/name,dict(version=version,name=name))
        count=previous['cum_decisions']+108039 if pause else int(value('--stop-at-decisions'))
        write(latest,dict(version=version,cum_decisions=count))
        if pause:(out/'STOP_REQUESTED').touch()
    monkeypatch.setattr(runner,'run_process',execute)
    entry.main()
    assert read(out/'plan.json')['parent_plan_file']=='source-fork.json'
    assert read(out/'plan.json')['parent_plan_sha256']==digest(audit/'source-fork.json')
    if pause:
        assert read(out/'status.json')['state']=='paused'
        assert read(out/'status.json')['latest']['version']==4160
        (out/'STOP_REQUESTED').unlink()
        monkeypatch.setattr(entry.sys,'argv',argv+['--resume'])
        entry.main()
        assert read(out/'status.json')['latest']['version']==4161
        assert calls==['eval','train','train']
    else:
        result=read(out/'report.json')
        assert result['complete'] and result['latest']['version']==4161
        assert calls==['eval','train','eval','train','eval','eval','eval']
        assert len(result['stages'])==2
        assert set(result['final_evaluations'])=={'latest'}
    assert {str(p.relative_to(audit)):digest(p) for p in audit.rglob('*') if p.is_file()}==originals


def test_login_source_precheck_uses_stdlib_without_slurm_or_audit_plan(tmp_path,monkeypatch):
    import subprocess,sys,json
    args,_=prepared(tmp_path,monkeypatch)
    command=[sys.executable,'-S',str(Path(__file__).resolve().parents[1]/'workflows/run_4159_long.py'),
             '--check-inputs-only','--source-4159-audit',str(args.source_4159_audit),'--bot-3702',str(args.bot_3702)]
    result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['parent_plan_file']=='source-fork.json'
    (args.source_4159_audit/'source-fork.json').unlink()
    result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8')
    assert result.returncode!=0 and 'source-fork.json' in result.stderr
