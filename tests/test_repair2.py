"""Regression cases found in the first NUS long run and the subsequent audit."""
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
torch = pytest.importorskip("torch")
import taverngym as tg
from taverngym.rl.config import Config, identity
from taverngym.rl.collect import collect, observation, gae
from taverngym.rl.model import Policy, pack
from taverngym.rl.train import (update, evaluate, prefix_hidden, atomic_save,
                                initialize_weights, rng_state)


def config():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    return Config(device="cpu",n_games=1,learner_seats=1,threads=1,torch_threads=1,
                  dim=32,layers=1,heads=2,hidden=32,seq_len=4,memory_horizon=8,
                  chunks_per_batch=2,epochs=1,bc_epochs=1,eval_seeds=[900000000])


def test_cross_seat_choices_defer_stale_legal_actions():
    f = json.loads((Path(__file__).parent/"fixtures/cross_seat_choice.json").read_text())
    e=tg.BatchEnv(1,controlled=range(8)); e.auto_reset=False
    e.load(0,json.dumps(f["episode"]))
    assert e.observe()["action_mask"][7,f["actions"][7]]
    g=e.game(0)
    g.step(0,f["actions"][0])
    with pytest.raises(ValueError,match="choice is pending"):
        g.step(7,f["actions"][7])
    _,_,info=e.step(f["actions"])
    assert not info["rejected"].any() and not info["applied"][7]
    assert info["applied"][0]
    assert e.observe()["ready"][7]


def test_kl_guard_skips_optimizer_when_already_over_limit():
    c=config(); m=Policy(c).eval(); e=tg.BatchEnv(1,controlled=range(8))
    o=observation(e.observe(),e.training_extras(),0)
    samples=[]; h=torch.zeros(1,c.hidden)
    with torch.no_grad():
        for _ in range(3):
            logits,val,_,h=m(pack([o],"cpu"),h)
            d=torch.distributions.Categorical(logits=logits); a=d.sample()
            samples.append(dict(obs=o,action=a.item(),value=val.item(),logp=d.log_prob(a).item()+1))
    before={k:v.clone() for k,v in m.state_dict().items()}
    opt=torch.optim.Adam(m.parameters(),lr=.001)
    r=update(m,opt,[dict(samples=samples,rank=1)],c)
    assert r["kl_early_stop"] and r["updates"]==0 and r["kl_trigger"]>c.target_kl
    assert not opt.state
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())


def test_periodic_memory_replay_and_bounded_prefix():
    c=config(); m=Policy(c).eval()
    tr,stats=collect(c,m,321)
    samples=tr[0]["samples"]
    assert len(samples)>c.memory_horizon
    with torch.no_grad():
        for start in range(0,len(samples),c.seq_len):
            ss=samples[start:start+c.seq_len]
            h=prefix_hidden(m,[(tr[0],start)],"cpu",c.memory_horizon)
            ob=pack([s["obs"] for s in ss],"cpu")
            z,entities=m.encode(ob); states,_=m.memory(z[None],h)
            logits,v,_=m.heads(states[0],entities,ob["mask"])
            lp=torch.distributions.Categorical(logits=logits).log_prob(torch.tensor([s["action"] for s in ss]))
            torch.testing.assert_close(lp,torch.tensor([s["logp"] for s in ss]),atol=2e-5,rtol=2e-5)
    calls=[]
    def encode(obs):
        calls.append(len(obs["features"]))
        return torch.zeros(len(obs["features"]),c.dim),None
    long={"samples":[samples[0]]*512}
    with patch.object(m,"encode",encode):
        prefix_hidden(m,[(long,508)],"cpu",8)
    assert sum(calls)==4
    assert stats["samples"]==sum(len(t["samples"]) for t in tr)


def test_evaluation_protocol_covers_modes_and_preserves_sampling_rng():
    c=config(); m=Policy(c); before=rng_state(); calls=[]
    def fake_collect(cfg,model,seed,opponents=None,**kwargs):
        calls.append((kwargs["greedy"],kwargs["opponent_greedy"],list(opponents or {})))
        torch.rand(5)
        return [],dict(ranks=[4,5],seconds=.1)
    with patch("taverngym.rl.train.collect",fake_collect):
        result=evaluate(c,m,anchor=m)
    assert set(result)=={"bots_greedy","bots_sample","anchor_mix_greedy","anchor_mix_sample"}
    assert torch.equal(before["torch"],rng_state()["torch"])
    assert {a for a,b,k in calls}=={True,False} and all(b for a,b,k in calls)


def test_explicit_weight_initialization_resets_experiment_identity(tmp_path):
    c=config(); m=Policy(c); signature=identity(c)
    old={**signature,"digest":"old-run","engine_source":"old-source","trainer_source":"old-trainer"}
    path=tmp_path/"old.pt"
    atomic_save(dict(model=m.state_dict(),identity=old,progress={"iteration":25}),path)
    other=Policy(c); meta=initialize_weights(other,path,c,signature)
    assert meta["source_progress"]=={"iteration":25} and "optimizer" in meta["reset"]
    assert all(torch.equal(v,other.state_dict()[k]) for k,v in m.state_dict().items())
    bad={**old,"vocabulary":["wrong"]}
    atomic_save(dict(model=m.state_dict(),identity=bad),path)
    with pytest.raises(ValueError,match="vocabulary"):
        initialize_weights(other,path,c,signature)


def test_memory_boundaries_are_explicit_and_aligned():
    with pytest.raises(ValueError,match="multiple"):
        replace(config(),memory_horizon=7).validate()
    c=config()
    assert identity(c)["digest"] != identity(replace(c,memory_horizon=16))["digest"]


def test_full_episode_returns_do_not_attenuate_early_terminal_credit():
    c=config()
    samples=[{"value":(-1)**i*.2} for i in range(1000)]
    _,returns=gae(samples,6,c.gamma,c.gae_lambda)
    np.testing.assert_allclose(returns,np.full(1000,-3/7),atol=1e-6)


def test_failure_artifact_uses_actual_failed_lobby(tmp_path,monkeypatch):
    real=tg.BatchEnv
    def injected(*args,**kwargs):
        env=real(*args,**kwargs)
        step=env.step
        def fail(actions):
            rewards,done,info=step(actions)
            info["rejected"][8]=True
            return rewards,done,info
        env.step=fail
        return env
    monkeypatch.setattr(tg,"BatchEnv",injected)
    c=replace(config(),n_games=2)
    with pytest.raises(RuntimeError,match="collection failure"):
        collect(c,None,123,teacher=True,artifact_dir=tmp_path)
    files=list(tmp_path.glob("failure-*.json"))
    assert len(files)==1
    payload=json.loads(files[0].read_text())
    assert payload["game_index"]==1 and payload["rejected"][0]
    assert payload["initial"]["game"]["seed"]==payload["post"]["game"]["seed"]
    assert len(payload["actions"])==1
