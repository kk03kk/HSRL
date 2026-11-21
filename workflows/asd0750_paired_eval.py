"""Paired 0750, raw R1 and 3286 study on the pinned current-rule engine."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity

ENGINE = "8bd1f4785018c474"
SHAS = {
    "0750": "eb2cb3de029705beaa278f079a676999e17b43a2c789fc36da7a2b2547b844b8",
    "r1": "1bcb12c6038f0d07223d925893c58a4be244531769f32e9820eeb2e9f10e134a",
    "3286": "778cc6f4ca9ba6175ec6f54c2241a030a70ce4e35b7ac7120492666360ba86df",
}
PANEL_SEEDS = {"hold": 627410000, "human": 628410000, "bot": 629410000}
EXPECTED_FIELDS = {
    "hold": "5361375f96441316237a792205b5ff76587bff27a9fd80d7cad25307075927d9",
    "human": "24d387ec449479425bd356b61c639b35f6095b4a32cab700519a0a62a04581f4",
}
ACTORS = {}
PANELS = {}


class SharedActionsGame:
    def __init__(self, game):
        self.game = game

    def __getattr__(self, name):
        return getattr(self.game, name)

    def legal_mask(self, seat):
        legal = bytes(self.game.legal_mask(seat))
        return legal[:196] + bytes(49)


class PolicyArm:
    def __init__(self, policy, shared):
        self.policy = policy
        self.shared = shared

    def reset(self):
        self.policy.reset()

    def decide(self, game, seat):
        return self.policy.decide(SharedActionsGame(game) if self.shared else game, seat)


class ASDActor:
    """Apply the same current HSAI policy guards to the recurrent 196-action model."""

    def __init__(self, path, protocol):
        import _asd_obs2
        from workflows.asd0750 import Obs2Policy
        from hsbg.az.train.ppo import PPOGreedyPolicy

        self.encode = _asd_obs2.observe
        self.model_policy = Obs2Policy(path, device="cpu")
        if (self.model_policy.net.layout.floats, self.model_policy.net.layout.cards,
                self.model_policy.net.layout.actions) != (4017, 447, 196):
            raise ValueError("wrong 0750 policy dimensions")
        self.policy = PPOGreedyPolicy(
            self, max_actions_per_turn=protocol.max_actions_per_turn,
            move_guard=protocol.move_guard, move_cap=protocol.move_cap,
            move_repeat_limit=protocol.move_repeat_limit, forbid_cancel=protocol.forbid_cancel,
            freeze_cap=protocol.freeze_cap, forbid_unfreeze=protocol.forbid_unfreeze,
            ban_trinkets=protocol.ban_trinkets, no_cancel_choose_on=protocol.no_cancel_choose,
        )

    def reset(self):
        self.model_policy.new_game()
        self.policy.reset()

    def decide(self, game, seat):
        self.game = game
        try:
            return self.policy.decide(game, seat)
        finally:
            self.game = None

    def __call__(self, features, cards, mask, seat):
        from hsbg.az.train.common import one_hot_policy

        if mask[196:].any() or not mask[:196].any():
            raise ValueError("0750 has invalid or empty supported action mask")
        floats, ids = self.encode(self.game.state_json(), seat)
        if (not np.array_equal(np.frombuffer(floats, "<f4")[:3227], features)
                or not np.array_equal(np.frombuffer(ids, "<i4")[:124], cards)):
            raise ValueError("observe2 changed the original observation prefix")
        action = self.model_policy.act(floats, ids, mask)
        if not 0 <= action < 196 or not mask[action]:
            raise ValueError("0750 produced a masked action")
        return one_hot_policy(action, mask), 0.0


@lru_cache(maxsize=None)
def card_name(card):
    import hsbg_engine
    return hsbg_engine.card_info((card & 0x0FFF) + 1)[0]


def item_card(entry):
    value, = entry["item"].values()
    return card_name(int(value["card"]))


def chosen_card(player, index):
    pending = player["pending"]
    if not pending:
        return None
    (kind, data), = pending[-1].items()
    options = data.get("options", [])
    if index >= len(options):
        return None
    option = options[index]
    if kind in ("HeroSelect", "HeroPowerChoice", "Trinket"):
        return card_name(int(option))
    if kind == "Discover":
        return item_card({"item": option})
    if kind == "Quest":
        return card_name(int(option[1]))
    if kind == "Timewarp":
        return item_card({"item": option[0]})
    return None


class Recorder:
    def begin_game(self, game, index, seed, seat):
        info = json.loads(game.state_json())
        self.lobby = int(info["lobby_races"])
        self.hero_offers = info["players"][seat]["pending"]
        self.events = []

    def before_with_player(self, game, seat, player):
        return game.turn, player

    def after(self, game, before, decision, action, forced):
        turn, player = before
        card = None
        if 8 <= action < 16:
            card = item_card(player["shop"][action - 8])
        elif 16 <= action < 23:
            card = card_name(int(player["board"][action - 16]["card"]))
        elif 30 <= action < 46:
            card = chosen_card(player, action - 30)
        elif 46 <= action < 196:
            width, first = (8, 46) if action < 126 else (7, 126)
            card = item_card(player["hand"][(action - first) // width])
        self.events.append([turn, action, card])

    def end_game(self, game, rank, quality):
        if any(quality.values()):
            raise ValueError(("invalid native game", rank, quality))


def play(policy, bank, seed, seat, max_action):
    from hsbg.az.train.evaluate import EvalConfig, evaluate_policy

    recorder = Recorder()
    result = evaluate_policy(policy, EvalConfig(
        games=1, seed=seed, policy_seat=seat, mirror_pool=bank, mirror_strict=bank is not None,
        collect_economy=True, decision_recorder=recorder))
    mirror = result["mirror"]
    if (result["games_finished"] != 1 or result["seed_sequence"] != [seed]
            or any(result[key] for key in ("illegal_actions", "truncated_games", "truncated_combats"))
            or bank is not None and (
                not mirror["mirror_applied"] or mirror["mirror_applied"] != mirror["mirror_attempt"]
                or any(mirror["mirror_" + key] for key in
                       ("errors", "rejected", "missing", "fallback", "pool_unbalanced", "pool_species_unbalanced")))):
        raise ValueError(("invalid paired match", seed, seat, result["rank_sequence"], mirror))
    if any(not 0 <= action < max_action for _, action, _ in recorder.events):
        raise ValueError(("action range exceeded", seed, seat, max_action))
    if len(recorder.events) != result["decisions"]:
        raise ValueError("incomplete action evidence")
    turns = {}
    for stage in result["end_turn_economy"]:
        turn = int(stage["turn"])
        if turn in turns or not stage["eot_available"]:
            raise ValueError(("missing recruit-end board", seed, seat, turn))
        turns[turn] = dict(
            tier=stage["tier"], gold=stage["gold"], health=stage["health"],
            actions=stage["actions"], counts=stage["action_counts"], hero=stage["hero"],
            trinkets=stage["trinkets"], hand_n=stage["hand_n"], forced=stage["forced"],
            board=[[x["card_id"], x["atk"], x["hp"], x["eid"], x.get("kw", 0)]
                   for x in stage["eot_board"]],
            strength=stage["eot_attack"] + stage["eot_board_hp"],
        )
    return dict(rank=result["rank_sequence"][0], lobby=recorder.lobby, hero_offers=recorder.hero_offers,
                turns=turns, decisions=result["decisions"], actions=recorder.events,
                injections=mirror["mirror_applied"],
                tail=result.get("mirror_strict_counts", {}).get("human_last_observed_live_seats", 0))


def matches(task):
    mode, panel, seed, seat, carried = task
    bank = PANELS[panel]["bank"]
    if mode == "native" and carried is not None:
        arms = {"0750": carried}
    else:
        arms = {"0750": play(PolicyArm(ACTORS[panel]["0750"], True), bank, seed, seat, 196)}
    for name in ("r1", "3286"):
        shared = mode == "shared"
        arms[name] = play(PolicyArm(ACTORS[panel][name], shared), bank, seed, seat, 196 if shared else 245)
    if any((a["lobby"], a["hero_offers"]) != (arms["0750"]["lobby"], arms["0750"]["hero_offers"])
           for a in arms.values()):
        raise ValueError(("paired lobby or hero offers differ", panel, seed, seat))
    return dict(mode=mode, panel=panel, seed=seed, seat=seat, arms=arms)


def setup(args):
    import _asd_obs2
    import hsbg_engine
    import torch
    from hsbg.az.train.eval_policy import EvalProtocol, make_eval_policy, protocol_sha256_of
    from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
    from hsbg.az.train.player_observation import SCHEMA_SHA
    from workflows.asd0750.runtime import use_author_static
    from workflows.peer_3703_round import field

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if hsbg_engine.build_info()["source_hash"] != ENGINE or not hsbg_engine.build_is_current(ROOT / "engine"):
        raise ValueError("current engine does not match the frozen Player150 source")
    use_author_static(args.static)
    paths = dict(zip(SHAS, (args.asd, args.r1, args.final), strict=True))
    ids = {name: file_identity(path) for name, path in paths.items()}
    for name, sha in SHAS.items():
        if ids[name]["sha256"] != sha:
            raise ValueError(("wrong checkpoint", name))
    order = "\n".join(hsbg_engine.card_info(i)[0] for i in range(1, hsbg_engine.N_CARDS))
    if hashlib.sha256(order.encode()).hexdigest() != "3b6fa2aeebe0955cb57438a9d8d5b487d61d8beb614f0c94c360dea3e9232270":
        raise ValueError("0750 embedding vocabulary differs")
    for panel in args.panels:
        if panel == "bot":
            protocol = EvalProtocol()
            manifest, spec, digest, bank = None, {"opponents": "current native bots"}, None, None
        else:
            protocol, manifest, spec, digest = field(panel)
            if digest != EXPECTED_FIELDS[panel]:
                raise ValueError(("reference panel drift", panel, digest))
            bank = MirrorSnapshotPool(protocol.mirror_snaps, sample="random")
            if bank.frames != manifest["frames"] or bank.bad_files:
                raise ValueError(("invalid opponent bank", panel, bank.frames, bank.bad_files))
        protocol.assert_training_contract()
        PANELS[panel] = dict(bank=bank, metadata=dict(source_panel_sha256=digest, field=spec,
                                                      library_sha256=manifest["digest"] if manifest else None))
        ACTORS[panel] = {name: make_eval_policy(str(paths[name]), protocol=protocol, device="cpu")["policy"]
                         for name in ("r1", "3286")}
        ACTORS[panel]["0750"] = ASDActor(paths["0750"], protocol)
    receipt = dict(source=source_identity(ROOT), native_engine=ENGINE,
                   native_obs2=file_identity(_asd_obs2.__file__), native_lock=file_identity(ROOT / "config/engine-upstream-lock.json"),
                   weights=ids, current_observation_sha256=SCHEMA_SHA, author_static=file_identity(args.static),
                   job=os.environ.get("SLURM_JOB_ID"), share_rng=True, seat_rotation="index modulo 8",
                   panels={panel: PANELS[panel]["metadata"] | {"start": PANEL_SEEDS[panel], "triplets": args.seeds}
                           for panel in args.panels},
                   engine_rules=protocol.engine_rules,
                   shared_action_guards=protocol.contract_dict(),
                   sample=False, phases={"shared": [196, 196, 196], "native": [196, 245, 245]})
    receipt["protocol_sha256"] = {
        mode: protocol_sha256_of(dict(engine=ENGINE, field=receipt["panels"], profile=receipt["engine_rules"],
                                      masks=widths, guards=receipt["shared_action_guards"], sample=False,
                                      observation=SCHEMA_SHA, policy_0750="observe2 with authored card static"))
        for mode, widths in receipt["phases"].items()
    }
    return receipt


def run_phase(out, mode, tasks, pool):
    results = pool.map(matches, tasks) if pool else map(matches, tasks)
    rows = []
    temp = out / f"{mode}.jsonl.gz.partial"
    with gzip.open(temp, "wt", encoding="utf-8", compresslevel=3) as stream:
        for i, row in enumerate(results, 1):
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            rows.append(row)
            if i % 16 == 0 or i == len(tasks):
                print(mode, i, "/", len(tasks), flush=True)
    temp.rename(out / f"{mode}.jsonl.gz")
    return rows


def main(args):
    for key in tuple(os.environ):
        if key.startswith(("HSAI_ARENA_", "HSAI_RECRUIT_START_", "HSAI_QUILL_")):
            del os.environ[key]
    os.environ.update(HSAI_ACTION_PROFILE="human-turn-3703-player150", HSAI_HUMAN_TAIL="last-observed-v1")
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["HSRL_FAILURE_DIR"] = str(args.out / "failure")
    receipt = setup(args)
    if args.workers > 1 and (os.environ.get("SLURM_JOB_PARTITION") != "normal" or receipt["source"]["tracked_dirty"]):
        raise ValueError("parallel experiment needs a frozen source and normal CPU allocation")
    atomic_json(args.out / "plan.json", receipt)
    pairs = [(panel, PANEL_SEEDS[panel] + i, i % 8) for i in range(args.seeds) for panel in args.panels]
    shared_tasks = [("shared", *pair, None) for pair in pairs]

    def run(pool):
        shared = run_phase(args.out, "shared", shared_tasks, pool)
        tasks = [("native", *pair, None if args.recheck_asd else row["arms"]["0750"])
                 for pair, row in zip(pairs, shared, strict=True)]
        native = run_phase(args.out, "native", tasks, pool)
        if any(a["arms"]["0750"] != b["arms"]["0750"] for a, b in zip(shared, native, strict=True)):
            raise ValueError("0750 should be invariant because its available actions do not change")
        atomic_json(args.out / "complete.json", dict(
            triplets_per_phase=len(pairs), independent_games=len(pairs) * (6 if args.recheck_asd else 5),
            asd_trajectory_reused=not args.recheck_asd,
            protocol_sha256=receipt["protocol_sha256"],
            evidence={name: file_identity(args.out / name) for name in ("plan.json", "shared.jsonl.gz", "native.jsonl.gz")}))

    if args.workers > 1:
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
            run(pool)
    else:
        run(None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("out", "asd", "static", "r1", "final"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--panels", nargs="+", choices=tuple(PANEL_SEEDS), default=["hold", "human"])
    parser.add_argument("--seeds", type=int, default=96)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--recheck-asd", action="store_true")
    main(parser.parse_args())
