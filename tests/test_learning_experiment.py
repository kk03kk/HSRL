"""分段决策预算、阶段checkpoint、止损与失败续跑，不另造PPO实现。"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from workflows.run_learning_experiment import read, write, digest, run, writer_lock, eval_stop


def panel(sha, rank):
    return dict(valid=True, protocol_sha256="same", checkpoint=dict(sha256=sha),
        protocol=dict(seed=8000000, games_per_seat=1, seats=list(range(8))),
        per_seat=[dict(policy_seat=s, seed_start=8000000, seed_end=8000000,
                       rank_sequence=[rank]) for s in range(8)])


def setup_experiment(tmp_path):
    from hsbg.az.train.personal import fingerprint, library_digest
    for split in ("train", "dev"):
        root = tmp_path / split
        root.mkdir()
        (root / "snap_0001.json.gz").write_bytes(b"immutable-test-library")
        write(root / "manifest.json", dict(split=split, fingerprint=fingerprint(),
              digest=library_digest(root), validation=dict(accepted=1)))
    cfg = tmp_path / "config.json"
    write(cfg, {"games": 16, "iterations": 0, "minutes": 25, "snapshot-every": 1})
    model = tmp_path / "best.pt"
    model.write_bytes(b"original-best")
    args = SimpleNamespace(config=str(cfg), checkpoint=str(model), train_library=str(tmp_path / "train"),
        eval_library=str(tmp_path / "dev"), out=str(tmp_path / "experiment"), target_decisions=10,
        seeds=[100000000, 200000000], eval_games=1, device="cpu")
    return args


class FakeCommands:
    def __init__(self, fail_candidate=False):
        self.calls = []
        self.fail_candidate = fail_candidate

    def __call__(self, argv):
        self.calls.append(argv)
        def arg(name):
            return argv[argv.index(name) + 1]
        out = Path(arg("--out"))
        if argv[3] == "eval":
            if self.fail_candidate and out.name == "evaluation.json":
                self.fail_candidate = False
                raise RuntimeError("模拟评测中断")
            model = Path(arg("--checkpoint"))
            write(out, panel(digest(model), 2 if model.name == "best.pt" else 1))
        else:
            target = int(arg("--stop-at-decisions"))
            latest = read(out / "latest.json") if (out / "latest.json").exists() else dict(version=0, cum_decisions=0)
            version, decisions = latest["version"], latest["cum_decisions"]
            while decisions < target:
                decisions += 5
                version += 1
                snap = out / "snapshots" / f"step_{version:06d}"
                snap.mkdir(parents=True)
                (snap / "model.pt").write_bytes(f"weights-{version}".encode())
                (snap / "trainer.pt").write_bytes(f"adam-rng-anchor-{version}".encode())
                with (out / "history.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(iter=version-1, collect=dict(decisions=5),
                        ppo=dict(processed_samples=5, updates=1),
                        profile=dict(collect_s=1, ppo_s=.5))) + "\n")
                write(out / "latest.json", dict(version=version, cum_decisions=decisions))


def test_two_seed_stages_resume_and_preserve_frozen_checkpoints(tmp_path):
    args = setup_experiment(tmp_path)
    commands = FakeCommands()
    rows = run(args, commands)
    assert len(rows) == 2 and all(r["actual_decisions"] == 10 for r in rows)
    trains = [c for c in commands.calls if c[3] == "train"]
    assert len(trains) == 4
    assert sum("--resume" in c for c in trains) == 2
    stage = Path(args.out) / "target-000000010/seed-100000000"
    original = (stage / "model.pt").read_bytes()
    args.target_decisions = 20
    run(args, commands)
    assert (stage / "model.pt").read_bytes() == original
    count = len(commands.calls)
    run(args, commands)
    assert len(commands.calls) == count  # 已完成阶段不重复训练/评测
    args.target_decisions = 10
    run(args, commands)
    assert len(commands.calls) == count


def test_failed_evaluation_resumes_frozen_model_without_retraining(tmp_path):
    args = setup_experiment(tmp_path)
    commands = FakeCommands(fail_candidate=True)
    with pytest.raises(RuntimeError, match="评测中断"):
        run(args, commands)
    first_trains = sum(c[3] == "train" for c in commands.calls)
    run(args, commands)
    assert first_trains == 2
    assert sum(c[3] == "train" for c in commands.calls) == 4  # 仅新增第二枚seed的训练


def test_experiment_refuses_changed_config_and_tampered_milestone(tmp_path):
    args = setup_experiment(tmp_path)
    commands = FakeCommands()
    run(args, commands)
    path = Path(args.out) / "target-000000010/seed-100000000/model.pt"
    original = path.read_bytes()
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="摘要"):
        run(args, commands)
    path.write_bytes(original)
    cfg = read(args.config)
    write(args.config, {**cfg, "lr": .001})
    with pytest.raises(ValueError, match="已改变"):
        run(args, commands)


def test_only_one_writer_and_os_releases_lock(tmp_path):
    with writer_lock(tmp_path):
        with pytest.raises(RuntimeError, match="写入者"):
            with writer_lock(tmp_path):
                pass
    with writer_lock(tmp_path):
        pass


def test_stop_rule_distinguishes_confirmed_harm_from_inconclusive_results():
    pair = dict(delta=.2, ci_low=-.1, top4_delta=0, top4_delta_ci95=[-.1, .1])
    assert not eval_stop(pair)
    assert eval_stop({**pair, "ci_low": .01})
    assert eval_stop({**pair, "top4_delta": -.06, "top4_delta_ci95": [-.1, -.01]})


def test_published_snapshot_recovers_missing_history_exactly_once(tmp_path):
    from hsbg.az.train.runlog import recover_committed_record
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    write(snapshot / "record.json", dict(iter=2, collect=dict(decisions=100)))
    history = tmp_path / "history.jsonl"
    assert recover_committed_record(history, snapshot, 3)
    assert not recover_committed_record(history, snapshot, 3)
    assert len(history.read_text().splitlines()) == 1
    with pytest.raises(ValueError, match="版本"):
        recover_committed_record(history, snapshot, 4)
