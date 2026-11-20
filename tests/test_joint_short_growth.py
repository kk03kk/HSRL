import pytest

pytest.importorskip("hsbg_engine")
from workflows import azerite_course as course
from workflows import joint_short_assessment as assessment
from tools.probe_four_growth_curves import HOPE, POET


def test_four_turn_fork_keeps_native_clock_and_same_future():
    class EndTurn:
        def reset(self):
            pass

        def decide(self,game,seat):
            from types import SimpleNamespace
            return SimpleNamespace(action=0)

    initial=course.opening(740001,"elem_budget")
    policy=EndTurn()
    native=course.probe.play(740001,"elem_budget",policy,opening=initial,normal_health=True,last_turn=11)
    assert [t["turn"] for t in native]==[9,10,11]
    assert initial.turn==12
    forks=[course.hsbg_engine.Game.from_json(initial.state_json()) for _ in range(2)]
    outcome=[course.probe.play(740001,"elem_budget",policy,opening=g,normal_health=True,last_turn=15) for g in forks]
    assert outcome[0]==outcome[1]
    assert [t["turn"] for t in outcome[0]]==[12,13,14,15]


def test_mechanism_rates_require_material_payout_and_prior_actions():
    arm=dict(rank=2,eot={"9":[350,[HOPE,POET]],"10":[420,[HOPE,POET]]},
             absorptions=[[8,100,0],[9,20,3],[9,20,6],[10,0,9]],gems=[],
             hope_growth=[],offers=["element"],buys=[],magnet=[])
    stats=assessment.stats(arm)
    assert stats["repeat_element"]==stats["dragon"]==stats["element_bought"]==0
    assert stats["elemental_return"]==40 and stats["element_offered"]==1
    arm["absorptions"].append([11,100,12])
    arm["hope_growth"]=[9]
    assert assessment.stats(arm)["repeat_element"]==assessment.stats(arm)["dragon"]==1


def test_pin_accepts_published_rolling_receipt_without_final_files(tmp_path,monkeypatch):
    import json
    import os
    from hsbg.az.train import personal
    monkeypatch.setattr(assessment,"SEG",tmp_path)
    generation=tmp_path/"train/snapshots/step_000005"
    generation.mkdir(parents=True)
    ckpt=generation/"model.pt"
    ckpt.write_bytes(b"fixture")
    (generation/"record.json").write_text(json.dumps(dict(ppo=dict(trainer_updates=5,updates=12))))
    receipt=dict(version=5,cum_decisions=490000,seed=assessment.J.PPO_SEED,
                 model=str(ckpt),trainer=str(generation/"trainer.pt"))
    (tmp_path/"train/latest.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(personal,"inspect_checkpoint",lambda p,device:dict(
        forward_ok=True,sha256=assessment.file_identity(p)["sha256"]))
    out=tmp_path/"evaluation"
    out.mkdir()
    # This is an archived, engine-pinned monitor; newer wheels must reject it.
    if assessment.C.quill.engine_info()["source_hash"] != assessment.C.quill.PLAN["engine"]:
        import pytest
        with pytest.raises(ValueError, match="monitor engine changed"):
            assessment.pin_middle(out)
        assert not (out/"model.pt").exists()
        monkeypatch.setitem(assessment.C.quill.PLAN, "engine", assessment.C.quill.engine_info()["source_hash"])
    model,actual=assessment.pin_middle(out)
    assert os.path.samefile(model,ckpt) and actual==receipt
