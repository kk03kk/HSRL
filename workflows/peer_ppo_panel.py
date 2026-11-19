"""Paired research panel for PrentendYan's public PPO release."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from hsbg.az.train.eval_policy import make_eval_policy
from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
from hsbg.az.train.ppo import PPOGreedyPolicy
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.references import reference_directory
from workflows import quilboar_study as Q
from workflows.joint_short_assessment import Mechanisms, stats

WEIGHT = Path("data/thirdparty/hsbg-ppo/model.pt")
PEER_SHA = "0454a99dda1d95b03922aa5015c827b1d499c8e952893284a3b3a0e87bf0352b"
SEED = 82370000
N = 128
SHARDS = 8


class Scores:
    def __init__(self, net, device):
        self.net = net
        self.device = device

    @torch.inference_mode()
    def __call__(self, f, c, mask, seat):
        legal = mask.copy()
        legal[self.blocked] = False
        if not legal.any():
            raise RuntimeError("start-board sell restriction emptied the legal mask")
        inp = [torch.as_tensor(x, device=self.device)[None] for x in (f, c, legal)]
        logits, value = self.net(*inp)
        if not torch.isfinite(logits[0]).all() or not torch.isfinite(value).all():
            raise RuntimeError("nonfinite external inference")
        chosen = int(logits[0].argmax())
        if not legal[chosen]:
            raise ValueError("illegal network argmax")
        pi = np.zeros_like(f[:245])
        pi[chosen] = 1
        return pi, float(value[0])


class Peer(PPOGreedyPolicy):
    def __init__(self, scores, protocol, keep):
        super().__init__(scores, **{k:getattr(protocol,k) for k in (
            "move_guard","move_cap","move_repeat_limit","max_actions_per_turn",
            "forbid_cancel","freeze_cap","forbid_unfreeze","ban_trinkets")}, no_cancel_choose_on=True)
        self.keep = keep
        self.reset()

    def reset(self):
        super().reset()
        self.turn = None

    def decide(self, game, seat):
        self.evaluator.blocked = []
        if self.keep:
            board = json.loads(game.player_json(seat))["board"]
            ids = [m["eid"] for m in board]
            if self.turn != (seat, game.turn):
                self.turn = (seat, game.turn)
                self.start = set(ids)
            self.evaluator.blocked = [16+i for i, eid in enumerate(ids) if eid in self.start]
        return super().decide(game, seat)


def trial(policy, pool, seed):
    recorder = Mechanisms()
    r = evaluate_policy(policy, EvalConfig(games=1, seed=seed, mirror_pool=pool,
        mirror_strict=True, collect_economy=True, decision_recorder=recorder))
    mirror = r["mirror"]
    if (r["games_finished"] != 1 or r["seed_sequence"] != [seed]
        or any(r[k] for k in ("illegal_actions","truncated_games","truncated_combats"))
        or any(mirror["mirror_"+k] for k in ("errors","rejected","missing","fallback","pool_unbalanced","pool_species_unbalanced"))
        or not mirror["mirror_applied"] or mirror["mirror_attempt"] != mirror["mirror_applied"]):
        raise ValueError(f"invalid game {seed}")
    turns = {t["turn"]:dict(atk=t["eot_attack"],hp=t["eot_board_hp"],tier=t["tier"],
             cards=[m["card_id"] for m in t["eot_board"]],actions=t["action_counts"])
             for t in r["end_turn_economy"]}
    arm = dict(rank=r["rank_sequence"][0], eot={t:[v["atk"]+v["hp"],v["cards"]] for t,v in turns.items()},
         absorptions=recorder.absorptions,gems=recorder.measured,magnet=recorder.mechanics,
         hope_growth=recorder.hope_growth,offers=recorder.offers,buys=recorder.buys)
    return dict(seed=seed, lobby=recorder.lobby, turns=turns, metrics=stats(arm),
                decisions=r["decisions"], injections=mirror["mirror_applied"])


def crossover(pool, base, peer, dest, index):
    from workflows.d142_crossover import Fork, play
    for seed in range(179240000+index,179240128,SHARDS):
        fork = Fork()
        ordinary = play(pool,base,seed,capture=fork)
        legs = {"base":ordinary}
        if fork.state is not None:
            same = play(pool,base,seed,resume=fork)
            if (same["terminal"] != ordinary["terminal"] or same["rank"] != ordinary["rank"]
                    or same["actions"] != [a for a in ordinary["actions"] if a[0]>=9]):
                raise ValueError(f"same-policy replay mismatch {seed}")
            legs["peer_free"] = play(pool,peer,seed,resume=fork)
        else:
            legs["peer_free"] = ordinary
        atomic_json(dest/f"{seed}.json",dict(seed=seed, opening=fork.opening if fork.state else None,
                   legs={k:dict(rank=v["rank"],strength=v["strength"],terminal=v["terminal"])
                         for k,v in legs.items()}))


def main():
    torch.set_num_threads(1)
    out = Path(os.environ["HSAI_PEER_OUT"])
    index = int(sys.argv[1])
    device = os.environ.get("HSAI_PEER_DEVICE","cuda:0")
    proto,pool = Q.config("hold")
    if file_identity(WEIGHT)["sha256"] != PEER_SHA or file_identity(Q.BASE)["sha256"] != Q.PLAN["base_sha"]:
        raise ValueError("incorrect weights")
    sys.path.insert(0,str(reference_directory("hsbg-ppo")))
    import hsbg_net
    net,ck,result = hsbg_net.load(WEIGHT,device=device,strict=True)
    if result.missing_keys or result.unexpected_keys or ck["net_config"]["z_vocab"][45] != "none":
        raise ValueError("wrong model or conditioning")
    net.eval()
    policies = dict(base=make_eval_policy(str(Q.BASE),protocol=proto,device=device)["policy"],
        peer=Peer(Scores(net,device),proto,True),peer_free=Peer(Scores(net,device),proto,False))
    assert 0 <= index < SHARDS
    dest=out/f"shard-{index}"
    dest.mkdir(exist_ok=False)
    atomic_json(dest/"identity.json",dict(model=PEER_SHA,base=Q.PLAN["base_sha"],protocol=proto.sha256(),
        upstream_engine=ck["meta"].get("engine_source_hash"),device=device,z=45))
    if sys.argv[2:]==["cross"]:
        crossover(pool,policies["base"],policies["peer_free"],dest,index)
        return
    for seed in range(SEED+index,SEED+N,SHARDS):
        rows = {k:trial(p,pool,seed) for k,p in policies.items()}
        if len({r["lobby"] for r in rows.values()}) != 1:
            raise ValueError("unpaired lobby")
        atomic_json(dest/f"{seed}.json",rows)
    print(f"shard {index} complete",flush=True)


if __name__=="__main__":
    main()
