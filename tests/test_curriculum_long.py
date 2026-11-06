"""换位守卫评测、流水线零更新退回与完整Adam学习率衔接。"""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
import pytest


def test_failure_status_retains_committed_checkpoint(tmp_path):
    from workflows.run_curriculum_long import record_failure
    from workflows.run_learning_experiment import read, write
    committed = dict(version=6160, cum_decisions=382004532)
    write(tmp_path/'train/latest.json', committed)
    write(tmp_path/'status.json', dict(state='training', latest=dict(version=6159)))
    record_failure(tmp_path, RuntimeError('pending has no legal action'))
    status = read(tmp_path/'status.json')
    assert status['state'] == 'failed' and status['latest'] == committed
    assert not status['stopped_for_performance']
    assert read(tmp_path/'FAILED.json') == status['failure']
    assert len(list((tmp_path/'failures').glob('*.json'))) == 1


def test_eval_cache_cannot_reuse_identical_boards_from_an_old_engine():
    from workflows.run_sustained_learning import validate_eval_cache
    library = dict(digest='unchanged-frame-files', fingerprint=dict(engine='new-rules'))
    protocol = dict(library_digest=library['digest'], fingerprint=dict(engine='old-rules'),
                    seed=11000000, games_per_seat=12, training_source_sha256='same-python')
    with pytest.raises(ValueError, match='引擎指纹'):
        validate_eval_cache(protocol, library, 11000000, 12, 'same-python')
    protocol['fingerprint'] = library['fingerprint']
    validate_eval_cache(protocol, library, 11000000, 12, 'same-python')


def test_zero_step_fallback_keeps_prefetched_future():
    from hsbg.az.train.pipeline_safety import after_update
    pool=ThreadPoolExecutor(max_workers=1);future=pool.submit(lambda:'valid-current-policy-batch')
    enabled,executor,event=after_update(True,0,1,pool,True)
    assert not enabled and executor is None and event
    assert future.result()=='valid-current-policy-batch'


@pytest.mark.parametrize('steps,lag,allow',[(1,1,True),(0,0,True),(0,1,False)])
def test_pipeline_not_disabled_for_success_or_default(steps,lag,allow):
    from hsbg.az.train.pipeline_safety import after_update
    with ThreadPoolExecutor(max_workers=1) as pool:
        enabled,executor,event=after_update(True,steps,lag,pool,allow)
        assert enabled and executor is pool and not event


def test_greedy_guard_preserves_buy_and_resets_per_game(monkeypatch):
    from hsbg.az.train import ppo
    monkeypatch.setattr(ppo,'obs_arrays',lambda _: (np.zeros(3227),np.zeros(124)))
    monkeypatch.setattr(ppo,'mask_from_bytes',lambda x:x)
    mask=np.zeros(245,dtype=bool);mask[[0,8,197,203]]=True
    game=SimpleNamespace(turn=12,observe=lambda seat:None,legal_mask=lambda seat:mask)
    def evaluator(f,c,m,seat):
        prior=np.zeros(245);prior[197]=10;prior[203]=9;prior[8]=1
        return prior,0
    guarded=ppo.PPOGreedyPolicy(evaluator,max_actions_per_turn=80,move_guard=True,move_cap=1)
    assert guarded.decide(game,0).action==197
    assert guarded.decide(game,0).action==8
    guarded.reset();assert guarded.decide(game,0).action==197
    plain=ppo.PPOGreedyPolicy(evaluator,max_actions_per_turn=80)
    assert [plain.decide(game,0).action for _ in range(3)]==[197]*3


def test_lr_fork_changes_schedule_but_preserves_adam_and_rng():
    import torch
    from hsbg.az.train.ppo import PPOConfig
    from hsbg.az.train.model import build_net
    from hsbg.az.train.ppo_trainer import PPOTrainer
    from tests.test_az_ppo import _batch_from_model
    from workflows.fork_trainer_lr import rescale
    model=build_net(16);cfg=PPOConfig(epochs=1,minibatch=64,lr=6e-6,device='cpu')
    trainer=PPOTrainer(model,cfg);trainer.update(_batch_from_model(model,n_games=1,decisions_per_seat=1))
    state=trainer.state_dict();fork=rescale(state,2e-6)
    assert fork['updates']==state['updates']
    assert fork['scheduler']['base_lrs'][0]==pytest.approx(2e-6)
    for k,v in state['optimizer']['state'].items():
        for name,tensor in v.items():
            if torch.is_tensor(tensor):assert torch.equal(tensor,fork['optimizer']['state'][k][name])
    assert torch.equal(fork['torch_rng'],state['torch_rng'])
    loaded=PPOTrainer(build_net(16),PPOConfig(lr=2e-6,device='cpu'),state=fork)
    assert loaded.optimizer.param_groups[0]['lr']==pytest.approx(state['scheduler']['_last_lr'][0]/3)
    assert state['scheduler']['base_lrs'][0]==6e-6


def test_learning_activity_uses_optimizer_steps_not_trainer_counter(tmp_path):
    import json
    from workflows.run_sustained_learning import learning_activity
    rows=[dict(iter=i,ppo=dict(updates=n,trainer_updates=10+i,samples=100,processed_samples=20*n))
          for i,n in enumerate((2,0,0))]
    (tmp_path/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    result=learning_activity(tmp_path)
    assert result['optimizer_steps']==2 and result['trailing_zero_update_iterations']==2
    assert result['processed_fraction']==pytest.approx(40/300)


@pytest.mark.parametrize('new_4159', [False,True])
def test_3702_baseline_precedes_training_and_milestones_publish_diagnostics(tmp_path, monkeypatch,new_4159):
    from pathlib import Path

    from hsbg.az.train import personal
    from hsbg.az.train import runlog
    from workflows import run_sustained_learning as runner
    from workflows.run_3702_preflight import ACTION_PROFILE

    parent, source, out = (tmp_path / name for name in ("parent", "source", "run"))
    baseline = source / "model.pt"
    runner.write(baseline, {"source": True})
    for name in ("report.json", "source-fork.json" if new_4159 else "plan.json"):
        runner.write(parent / name, {})
    args = SimpleNamespace(
        out=out, bot_3702=True, curriculum_parent=parent, strict_mirror=True,
        reference_checkpoint=baseline, wall_minutes=1410, reserve_minutes=45,
        device="cuda:0", monitor_every=30000000, monitor_games=12, eval_games=64,
        budget_decisions=60000000,
    )
    if new_4159:
        args.curriculum_parent=None
        args.source_4159_audit=parent
    manifests = {key: {"digest": key, "fingerprint": {"engine": "current"}} for key in ("train", "hard")}
    monkeypatch.setattr(runner, "parent_inputs", lambda _: (
        parent, source, {"actual_decisions": 100}, "economy", {"train": "train", "hard": "hard"},
        manifests, {"minutes": 30}, {}))
    monkeypatch.setattr(personal, "checked_library", manifests.__getitem__)
    monkeypatch.setattr(personal, "training_source_digest", lambda: "code")
    monkeypatch.setattr(runlog, "recover_committed_record", lambda *_: None)
    monkeypatch.setattr(runner, "learning_activity", lambda _: {
        "optimizer_steps": 1, "trailing_zero_update_iterations": 0})
    monkeypatch.setattr(runner, "training_report", lambda *_: {
        "totals": {"decisions": runner.read(out/"train/latest.json")["cum_decisions"] - 100,
                   "mirror_fallback": 0, "mirror_applied": 1},
        "mirror_strict_counts": {"eligible_live_seats": 1, "live_combats": 1},
    })

    def freeze(train, target, decisions):
        runner.write(target/"model.pt", {"decisions": decisions})
        record = {"version": runner.read(train/"latest.json")["version"],
                  "sha256": {"model.pt": runner.digest(target/"model.pt")}}
        runner.write(target/"checkpoint.json", record)
        return record

    monkeypatch.setattr(runner, "freeze_checkpoint", freeze)
    calls = []

    def execute(argv):
        def value(flag):
            return argv[argv.index(flag)+1]

        calls.append(argv[3])
        if argv[3] == "train":
            assert (out/"baselines/diagnostics.json").is_file()
            latest = out/"train/latest.json"
            version = runner.read(latest)["version"] + 1 if latest.exists() else 6158
            runner.write(latest, dict(version=version, cum_decisions=int(value("--stop-at-decisions"))))
            return
        games, seed = int(value("--games")), int(value("--seed"))
        rank = 4 if Path(value("--checkpoint")) == baseline else 5
        protocol = {key.replace("-", "_"): value for key, value in ACTION_PROFILE.items()}
        protocol.update(library_digest="hard", training_source_sha256="code", seed=seed,
                        games_per_seat=games, seats=list(range(8)), mirror_strict=True,
                        fingerprint={"engine": "current"})
        rows = [dict(policy_seat=seat, seed_start=seed, seed_end=seed+games-1,
                     rank_sequence=[rank]*games, end_turn_economy=[], combat_exposure=[])
                for seat in range(8)]
        runner.write(value("--out"), dict(valid=True, checkpoint={"sha256": runner.digest(value("--checkpoint"))},
                     protocol=protocol, protocol_sha256=f"protocol-{seed}", per_seat=rows,
                     mean_rank=rank, top4_rate=float(rank <= 4), win_rate=0))

    report = runner.run(args, execute=execute)
    assert calls == ["eval", "train", "eval", "train", "eval", "eval", "eval"]
    assert [s["target"] for s in report["stages"]] == [30000100, 60000100]
    for stage in report["stages"]:
        diagnostic = runner.read(Path(stage["diagnostics"]))
        assert diagnostic["paired"]["delta"] == 1
        assert diagnostic["reference"]["mean_rank"] == 4
        assert diagnostic["candidate"]["mean_rank"] == 5
        assert diagnostic["candidate"]["turns"]["19"]["games"] == 96
    assert report["budget_complete"] and not report["stopped_for_performance"]


def test_failed_run_recovery_keeps_committed_progress_and_monitor_cadence(tmp_path, monkeypatch):
    from workflows import run_3702_preflight as repair
    from workflows.run_learning_experiment import write, read
    from hsbg.az.train import personal

    failed, out = tmp_path/"failed", tmp_path/"new"
    inherited = {"actual_decisions": 381659029, "original_sha256": {"model.pt": "original", "trainer.pt": "original-trainer"}}
    config = {"move-cap": 5}
    fp = {"engine": "3702"}
    manifests = {"train": {"fingerprint": fp, "digest": "train"}}
    expected = {"source": inherited["original_sha256"], "libraries": {"train": "train", "hard": "dev"},
                "fingerprint": fp, "config": config, "migration_sha256": "migration",
                "training_source_sha256": "fixed-source"}
    evidence = {"passed": True, "provenance": {**expected, "training_source_sha256": "old-source"}}
    write(failed/"plan.json", {"source_sha256": "old-source", "config": config,
                              "libraries": expected["libraries"], "inherited": inherited})
    write(failed/"train/provenance.json", {"training_source_sha256": "old-source", "config": config,
                                         "fingerprint": fp, "library_digest": "train"})
    write(failed/"train/latest.json", {"version": 6160, "cum_decisions": 382004532})
    write(failed/"FAILED.json", {"error": "pending"})
    restored = {"version": 6160, "actual_decisions": 382004532,
                "sha256": {"model.pt": "recovered", "trainer.pt": "recovered-trainer"}}
    monkeypatch.setattr(repair, "freeze_checkpoint", lambda *args: restored)
    monkeypatch.setattr(personal, "inspect_checkpoint", lambda *args: {"forward_ok": True})
    args = SimpleNamespace(recover_from=failed, monitor_every=30000000)
    source, result = repair.recover(args, out, inherited, manifests, config, evidence, expected)
    assert source == out/"recovered-state"
    assert result["version"] == 6160 and result["sha256"] == restored["sha256"]
    assert result["actual_decisions"] + args.monitor_first == 411659029
    assert read(out/"recovery.json")["old_training_source"] == "old-source"
    assert read(failed/"train/latest.json")["version"] == 6160
    expected["config"] = {"move-cap": 14}
    with pytest.raises(ValueError, match="不匹配"):
        repair.recover(args, tmp_path/"bad", inherited, manifests, config, evidence, expected)
