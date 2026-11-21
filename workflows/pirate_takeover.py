"""Frozen CPU-only opportunity census and paired, recurrent policy handoff."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import numpy as np
import torch
from hsbg.runtime.artifacts import atomic_json, file_identity
from workflows import asd0750_paired_eval as ref
from replay_asd0750_mechanisms import Replay, Witness, CORE
from summarize_asd0750_paired import normal, sustained

PIRATES = set(CORE["pirate"])
TECH = {"BG23_318", "BG26_888", "BG27_013", "BGS_131"}
PAIRS = dict(CORE, boar=("BG31_323", "BG31_326"), element=("BG28_707", "BG35_881"))
SEEDS = {"hold": (691210000, 160), "human": (692210000, 160)}
SOURCE = Path("artifacts/runs/nus-asd0750-paired-eval-20261121-01")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def options(player, mask, tracked):
    """Inspect only current own shop, visible Discover offers and playable hand slots."""
    result = {}
    for slot, entry in enumerate(player["shop"][:8]):
        if mask[8 + slot]:
            card = normal(ref.item_card(entry))
            if card in tracked:
                result[8 + slot] = card
    if player["pending"] and "Discover" in player["pending"][-1]:
        for i in range(16):
            if mask[30 + i]:
                name = ref.chosen_card(player, i)
                if name and normal(name) in tracked:
                    result[30 + i] = normal(name)
    for i, entry in enumerate(player["hand"][:10]):
        card = normal(ref.item_card(entry))
        if card in tracked:
            for a in (*range(46 + 8*i, 54 + 8*i), *range(126 + 7*i, 133 + 7*i)):
                if mask[a]:
                    result[a] = card
    return result


class Census(Witness):
    def begin_game(self, game, index, seed, seat):
        super().begin_game(game, index, seed, seat)
        self.first = self.low = None
        self.target_rows = self.target_actions = self.tech_offers = self.tech_buys = 0
        self.invested = False
        self.alive_turn = -1

    def before_with_player(self, game, seat, player):
        before = super().before_with_player(game, seat, player)
        turn = int(game.turn)
        mask = game.legal_mask(seat)
        if turn != self.alive_turn:
            self.alive_turn = turn
            self.alive = sum(bool(x["alive"]) for x in json.loads(game.state_json())["players"])
        self.offers = options(player, mask, PIRATES | TECH)
        hp = int(player["health"] + player["armor"])
        if 0 < hp <= 15 and self.low is None:
            self.low = turn
        if self.first is None and any(c in PIRATES for c in self.offers.values()):
            self.first = dict(turn=turn, hp=hp, tier=player["tier"], gold=player["gold"])
        self.target_rows += int(any(mask[126:196]))
        self.tech_offers += int(self.alive == 2 and any(a < 46 and c in TECH for a,c in self.offers.items()))
        return before

    def after(self, game, before, decision, action, forced):
        super().after(game, before, decision, action, forced)
        self.target_actions += int(126 <= action < 196)
        self.tech_buys += int(self.alive == 2 and action < 46 and self.offers.get(action) in TECH)
        self.invested |= self.offers.get(action) in PIRATES


def census(job):
    row, model = job
    old = row["arms"][model]
    rec = Census()
    result = ref.play(Replay(old, rec), ref.PANELS[row["panel"]]["bank"], row["seed"], row["seat"],
                      196 if model == "0750" else 245, rec)
    result["turns"] = {str(k):v for k,v in result["turns"].items()}
    if result != old:
        raise ValueError(("archive replay differs", row["seed"], model))
    combos = sustained(result)
    return dict(seed=row["seed"], panel=row["panel"], seat=row["seat"], model=model, rank=old["rank"],
                lobby=old["lobby"], first=rec.first, low_turn=rec.low, invested=rec.invested,
                target_rows=rec.target_rows, target_actions=rec.target_actions, final_tech_rows=rec.tech_offers,
                final_tech_choices=rec.tech_buys, formed={k:tuple(sorted(v)) in combos for k,v in PAIRS.items()},
                events=dict(Counter(e[1] for e in rec.witnesses)), strength={t:v["strength"] for t,v in old["turns"].items()})


def copy_guard(policy):
    return deepcopy(policy, {id(policy.evaluator):policy.evaluator})


class Prefix:
    turn = 1

    def __init__(self, student, teacher):
        self.student = student
        self.teacher = teacher

    def __call__(self, game, policy, strict):
        self.context = strict

    def reset(self):
        self.student.reset()
        self.fork = None
        self.tape = []
        self.actions = []

    def decide(self, game, seat):
        mask = bytes(game.legal_mask(seat))
        if self.fork is None:
            player = json.loads(game.player_json(seat))
            offered = options(player, mask, PIRATES)
            if offered:
                strict = self.context
                if strict.turn != game.turn:
                    raise RuntimeError("unprepared opponent at opportunity")
                self.fork = dict(state=json.loads(game.state_json()), guard=copy_guard(self.student),
                                 strict=deepcopy(strict, {id(strict.library):strict.library}), offered=offered,
                                 index=len(self.actions), hp=player["health"]+player["armor"],
                                 tier=player["tier"], gold=player["gold"])
        decision = self.student.decide(game, seat)
        action = int(decision.action)
        if self.fork is None and action < 196:
            floats, ids = self.teacher.encode(game.state_json(), seat)
            self.tape.append((floats, ids, mask, action))
        self.actions.append(action)
        return decision


@torch.no_grad()
def conditioned(model, hidden, floats, cards, mask, action):
    features, ids, legal = model._decode_inputs(floats, cards, mask)
    if not legal[action]:
        raise ValueError("recorded history action not legal")
    args = (torch.from_numpy(features), torch.from_numpy(ids), torch.from_numpy(legal))
    chosen, _, next_state = model.net.step(*args, hidden, forced=action)
    if chosen != action:
        raise ValueError("source-conditioned history mismatch")
    return next_state


class Takeover:
    def __init__(self, prefix, hidden, duration):
        self.prefix = prefix
        self.hidden = hidden
        self.duration = duration

    def reset(self):
        self.guard = copy_guard(self.prefix.fork["guard"])
        self.prefix.teacher.model_policy.hidden = self.hidden
        self.actions = []
        self.teacher_steps = self.return_steps = 0
        self.first_action = None

    def decide(self, game, seat):
        teacher = self.prefix.teacher
        if int(game.turn) < self.prefix.fork["state"]["turn"] + self.duration:
            self.teacher_steps += 1
            original = self.guard.evaluator
            self.guard.evaluator = teacher
            teacher.game = game
            before_hidden = teacher.model_policy.hidden
            try:
                decision = self.guard.decide(ref.SharedActionsGame(game), seat)
                if decision.info.get("forced"):
                    floats, ids = teacher.encode(game.state_json(), seat)
                    teacher.model_policy.hidden = conditioned(teacher.model_policy, before_hidden,
                        floats, ids, bytes(game.legal_mask(seat)), int(decision.action))
            finally:
                self.guard.evaluator = original
                teacher.game = None
        else:
            self.return_steps += 1
            decision = self.guard.decide(game, seat)
        self.actions.append(int(decision.action))
        return decision


class Payout(Witness):
    def before_with_player(self, game, seat, player):
        return int(game.turn), player


def evaluate(policy, bank, seed, seat, **config):
    from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
    rec = Payout()
    result = evaluate_policy(policy, EvalConfig(games=1, seed=seed, policy_seat=seat,
        mirror_pool=bank, mirror_strict=True, collect_economy=True, decision_recorder=rec, **config))
    mirror = result["mirror"]
    if (result["games_finished"] != 1 or result["seed_sequence"] != [seed]
        or any(result[k] for k in ("illegal_actions","truncated_games","truncated_combats"))
        or mirror["mirror_attempt"] != mirror["mirror_applied"] or not mirror["mirror_applied"]
        or any(mirror["mirror_"+k] for k in ("errors","rejected","missing","fallback","pool_unbalanced","pool_species_unbalanced"))):
        raise ValueError(("invalid game",seed,mirror))
    board_turns = [x for x in result["end_turn_economy"] if x["eot_available"]]
    present = {x["turn"] for x in board_turns if PIRATES <= {normal(m["card_id"]) for m in x["eot_board"]}}
    return dict(rank=result["rank_sequence"][0], tail=result.get("mirror_strict_counts",{}).get("human_last_observed_live_seats",0),
                events=[e for e in rec.witnesses if e[1]=="castaway_pick_with_hooktusk"],
                formed=any(t+1 in present for t in present), last=max((x["turn"] for x in board_turns),default=0),
                first=rec.events[0][1] if rec.events else None,
                acquisition=any(8<=a<16 or 30<=a<46 for _,a,c in rec.events if c and normal(c) in PIRATES),
                actions=digest(policy.actions), decisions=len(policy.actions))


def trial(job):
    import hsbg_engine
    from hsbg.az.train.evaluate import _mix_seed
    panel, seed, seat = job
    actors = ref.ACTORS[panel]
    bank = ref.PANELS[panel]["bank"]
    prefix = Prefix(actors["3286"], actors["0750"])
    natural = evaluate(prefix, bank, seed, seat, research_capture_strict=prefix)
    row = dict(panel=panel, seed=seed, seat=seat, natural=natural, opportunity=None, futures=[])
    fork = prefix.fork
    if fork is None:
        return row
    opening = fork["state"]
    model = prefix.teacher.model_policy
    hidden = model.net.initial_state()
    for floats, ids, mask, act in prefix.tape:
        hidden = conditioned(model, hidden, floats, ids, mask, act)
    row["opportunity"] = dict(turn=opening["turn"], hp=fork["hp"], tier=fork["tier"], gold=fork["gold"],
        offered=fork["offered"], opening_sha256=digest(opening), prefix_steps=fork["index"],
        skipped_moves=fork["index"]-len(prefix.tape), conditioned_steps=len(prefix.tape))
    for j in range(4):
        state = opening
        strict = deepcopy(fork["strict"], {id(fork["strict"].library):fork["strict"].library})
        if j:
            donor = seed + j*10000000
            rng = json.loads(hsbg_engine.Game(seed=donor).state_json())["rng"]
            state = dict(opening, rng=rng)
            strict.frame_rng = lambda turn, target: random.Random(_mix_seed(donor, turn, target))
        outcomes = {}
        for duration in (0, 1, 2, 100):
            runner = Takeover(prefix, hidden, duration)
            result = evaluate(runner, bank, seed, seat, research_initial_state=state, research_strict_context=strict)
            if j==0 and duration==0 and (result["rank"] != natural["rank"] or runner.actions != prefix.actions[fork["index"]:]):
                raise ValueError(("same-policy restoration failed",seed))
            result.update(teacher_steps=runner.teacher_steps, return_steps=runner.return_steps)
            outcomes[str(duration)] = result
        row["futures"].append(outcomes)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage",choices=("baseline","hold","human"))
    p.add_argument("out",type=Path)
    p.add_argument("--workers",type=int,required=True)
    p.add_argument("--start",type=int,default=0)
    p.add_argument("--count",type=int,default=160)
    args = p.parse_args()
    if os.environ.get("SLURM_JOB_PARTITION")!="normal":
        raise RuntimeError("requires a normal CPU allocation")
    args.out.mkdir(exist_ok=False,parents=True)
    os.environ.update(HSAI_ACTION_PROFILE="human-turn-3703-player150",HSAI_HUMAN_TAIL="last-observed-v1",
                      HSRL_FAILURE_DIR=str(args.out/"failure"))
    args.panels=["hold","human"]
    args.seeds=96
    modeldir=Path("artifacts/models/asd-policy-0750-public-20261121")
    args.asd=modeldir/"policy-0750.pt"
    args.static=modeldir/"author-card-static.f32"
    args.final=Path("artifacts/runs/nus-player150-five-20261120-01/round-05/train/fit/train/snapshots/final_003286/model.pt")
    args.r1=Path("artifacts/runs/nus-peer-rounds-production-20261120-1/round-01/train-1170497/fit/train/snapshots/final_000647/model.pt")
    identity=ref.setup(args)
    if identity["source"]["tracked_dirty"] or identity["native_obs2"]["sha256"]!="a2fddbca31b9a2b0ac6e0e6890d681c2d0a00192e0876cf91359282792f9014b":
        raise ValueError("unsealed source or observer")
    protocol=dict(source=identity["source"]["commit"],engine=ref.ENGINE,weights=identity["weights"],fields=ref.EXPECTED_FIELDS,
       trigger="first legal buy, Discover or hand play of BG36_342 or BG36_344",durations=[0,1,2,100],
       seeds=SEEDS,futures=4,low_hp=15,primary_duration=2,utility=[1,.35,-1,-1.15,-1.30,-1.45,-1.60,-1.75],
       gate="blind source-clustered 95% lower utility above zero and positive mechanism evidence; report both windows, all sources",
       hidden="force executed supported sources; omit unsupported Move tokens; retain post-move observation")
    atomic_json(args.out/"plan.json",dict(protocol=protocol,protocol_sha256=digest(protocol),identity=identity,
                                          stage=args.stage,start=args.start,count=args.count))
    if args.stage=="baseline":
        path=SOURCE/"native.jsonl.gz"
        sealed=json.loads((SOURCE/"complete.json").read_bytes())
        if file_identity(path)!=sealed["evidence"][path.name]:
            raise ValueError("unsealed historical replay")
        with gzip.open(path,"rt") as stream:
            jobs=[(row,m) for row in map(json.loads,stream) for m in ref.SHAS]
        worker=census
    else:
        start,n=SEEDS[args.stage]
        if args.start+args.count>n:
            raise ValueError("invalid block")
        jobs=[(args.stage,start+i,i%8) for i in range(args.start,args.start+args.count)]
        worker=trial
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context("fork")) as pool:
        with (args.out/"rows.jsonl").open("x") as out:
            for i,row in enumerate(pool.map(worker,jobs),1):
                out.write(json.dumps(row,separators=(",",":"))+"\n")
                out.flush()
                if i%8==0: print(args.stage,i,len(jobs),flush=True)
    atomic_json(args.out/"complete.json",dict(rows=len(jobs),evidence=file_identity(args.out/"rows.jsonl")))


if __name__=="__main__":
    main()
