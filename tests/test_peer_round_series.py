import pytest

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
