import pytest
from ops import peer_round_series as series
from hsbg.runtime.artifacts import atomic_json, file_identity

from ops.peer_round_series import PANELS, job_args, round_one_ready, settings, stages


@pytest.mark.parametrize("smoke", (True, False))
def test_train_and_score_before_every_continuation(smoke):
    cfg = settings(smoke)
    steps = list(stages(cfg["rounds"]))
    assert steps[:-2] == [(i, stage) for i in range(1, cfg["rounds"] + 1) for stage in ("train", "monitor")]
    assert steps[-2:] == [(cfg["rounds"], "blind"), (cfg["rounds"], "adaptive")]
    for _, stage in steps:
        args = job_args(stage, smoke)
        assert args[args.index("-p") + 1] == ("normal" if stage == "adaptive" else "gpu")
        if stage == "train" and not smoke:
            assert args[args.index("-t") + 1] == "03:00:00"


def test_dress_rehearsal_uses_separate_games_and_seed():
    regular, smoke = settings(False), settings(True)
    def seeds(cfg):
        return set().union(*(range(cfg[k]["seed"], cfg[k]["seed"] + cfg[k]["games"]) for k in PANELS))
    assert not seeds(regular) & seeds(smoke)
    assert regular["train_seed"] != smoke["train_seed"]
    assert regular["environment"] == smoke["environment"]
    assert regular["protocol_sha256"] == smoke["protocol_sha256"]


def test_budget_decision_requires_rank_growth_and_survival():
    gate = settings(False)["round_one_budget_gate"]
    d = dict(rank=dict(mean=-0.1, ci95=[-0.3, 0.08]), growth9to12=dict(mean=0.04, ci95=[-0.06, 0.11]),
             utility=dict(mean=0.02), reach12=dict(mean=-0.01))
    assert round_one_ready({"delta": d}, gate)
    for key, value in (("rank", dict(mean=-0.1, ci95=[-0.3, 0.2])),
                       ("growth9to12", dict(mean=-0.01, ci95=[-0.05, 0.09])),
                       ("utility", dict(mean=-0.02)), ("reach12", dict(mean=-0.03))):
        assert not round_one_ready({"delta": d | {key: value}}, gate)


def test_failed_budget_monitor_cannot_submit_remaining_jobs(tmp_path, monkeypatch):
    cfg = settings(False)
    source = {"commit": "fixture", "tracked_dirty": False}
    release = tmp_path / "releases" / "fixture"
    release.mkdir(parents=True)
    monkeypatch.setattr(series, "ROOT", release)
    monkeypatch.setattr(series, "source_identity", lambda root: source)
    out = tmp_path / "series"
    atomic_json(out / "series.json", dict(source=source, recipe=file_identity(series.RECIPE), smoke=False))
    folder = out / "round-01"
    report = folder / "test-scores.json"
    atomic_json(report, dict(delta=dict(rank=dict(mean=0.1))))
    atomic_json(folder / "train.json", dict(checkpoint=dict(files={"model.pt": "test-weight"})))
    atomic_json(folder / "monitor.json", dict(checkpoint="test-weight", seed=cfg["monitor"]["seed"], n=512,
                protocol=cfg["protocol_sha256"], report=str(report), report_identity=file_identity(report), job="123"))
    def scheduler(command, **kwargs):
        assert command[0] == "sacct"
        return "123|COMPLETED\n"
    monkeypatch.setattr(series.subprocess, "check_output", scheduler)
    series.submit(out, cfg, smoke=False, dry_run=False, continuation=True)
    assert (out / "budget-not-released.json").is_file()
    assert not (out / "continuation.json").exists()


def test_day_review_and_explicit_midround_night_handoff(tmp_path, monkeypatch):
    cfg = settings(False)
    source = {"commit": "fixture", "tracked_dirty": False}
    release = tmp_path / "releases" / "fixture"
    release.mkdir(parents=True)
    monkeypatch.setattr(series, "ROOT", release)
    monkeypatch.setattr(series, "source_identity", lambda root: source)
    calls = []
    def scheduler(command, **kwargs):
        if command[0] == "sacct":
            return "102|PENDING\n"
        assert command[0] == "sbatch"
        calls.append(command)
        return str(100 + len(calls))
    monkeypatch.setattr(series.subprocess, "check_output", scheduler)
    out = tmp_path / "run"
    series.submit(out, cfg, False, False)
    assert len(calls) == 2 and "--dependency=afterok:101" in calls[1]
    with pytest.raises(ValueError, match="not ready"):
        series.submit(out, cfg, False, False, continuation=True)
    series.submit(out, cfg, False, False, continuation=True, mode="night")
    assert len(calls) == 10 and "--dependency=afterok:102" in calls[2]
    assert all("--dependency=afterok:" + str(100 + i) in calls[i] for i in range(1, len(calls)))
    with pytest.raises(ValueError, match="all planned rounds"):
        series.submit(out, cfg, False, False, continuation=True, mode="night")
