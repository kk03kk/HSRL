import pytest

from ops.peer_round_series import PANELS, job_args, settings, stages


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
