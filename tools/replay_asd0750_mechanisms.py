"""Replay frozen 0750 actions and verify observable mechanics in early eligible examples."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsbg.runtime.artifacts import atomic_json, file_identity
from summarize_asd0750_paired import CARDS, normal, sustained
from workflows.asd0750_paired_eval import ENGINE, EXPECTED_FIELDS, Recorder, card_name, play


CORE = {
    "pirate": tuple(sorted(("BG36_342", "BG36_344"))),
    "dragon": tuple(sorted(("BG29_813", "BG36_364"))),
    "mech": tuple(sorted(("BG36_367", "BG36_851"))),
}
BANKS = {}


def index_board(player):
    return {card["eid"]: card for card in player["board"]}


def card_is(minion, cid):
    return normal(card_name(minion["card"])) == cid


def strength(unit):
    return int(unit["atk"]) + int(unit["hp"])


class Witness(Recorder):
    def begin_game(self, game, index, seed, seat):
        super().begin_game(game, index, seed, seat)
        self.seat = seat
        self.last_turn = -1
        self.end = None
        self.witnesses = []

    def before_with_player(self, game, seat, player):
        turn = int(game.turn)
        if turn != self.last_turn and self.end is not None:
            after = index_board(player)
            old = index_board(self.end)
            for eid in old.keys() & after.keys():
                if card_is(old[eid], "BG36_364"):
                    gain = int(after[eid]["counter"]) - int(old[eid]["counter"])
                    if gain > 0:
                        self.witnesses.append([turn - 1, "hope_shields", gain])
            board = self.end["board"]
            for i, unit in enumerate(board):
                if not card_is(unit, "BG29_813"):
                    continue
                for adj in board[max(0, i - 1):i] + board[i + 1:i + 2]:
                    eid = adj["eid"]
                    if eid in after and "DRAGON" in CARDS[normal(card_name(adj["card"]))].get("races", ()):
                        gain = strength(after[eid]) - strength(adj)
                        if gain > 0:
                            self.witnesses.append([turn - 1, "adjacent_dragon_retained_stats", gain])
        self.last_turn = turn
        self.end_candidate = any(card_is(c, "BG29_813") or card_is(c, "BG36_364") for c in player["board"])
        return turn, player

    def before_end(self, game):
        from hsbg.az.train.mirror_snapshots import capture_eot

        result = capture_eot(game, self.seat) if self.end_candidate else None
        self.end = result["player"] if result else None

    def after(self, game, before, decision, action, forced):
        super().after(game, before, decision, action, forced)
        turn, player = before
        if not (8 <= action < 16 or 30 <= action < 46 or 46 <= action < 196):
            return
        old = index_board(player)
        new = index_board(json.loads(game.player_json(self.seat)))
        survivors = old.keys() & new.keys()
        if 30 <= action < 46 and player["pending"]:
            payload = player["pending"][-1].get("Discover")
            if (payload is not None and normal(card_name(payload["source"])) == "BG36_342"
                    and any(card_is(c, "BG36_344") for c in old.values())):
                gains = [strength(new[eid]) - strength(old[eid]) for eid in survivors
                         if "PIRATE" in CARDS[normal(card_name(old[eid]["card"]))].get("races", ())]
                if sum(gains) > 0:
                    self.witnesses.append([turn, "castaway_pick_with_hooktusk", sum(gains)])
        for eid in survivors:
            gain = len(new[eid]["attached"]) - len(old[eid]["attached"])
            if gain <= 0:
                continue
            if 8 <= action < 16 and card_is(old[eid], "BG36_367"):
                self.witnesses.append([turn, "auto_reveille_buy_attachment", gain])
            if 46 <= action < 126 and any(card_is(unit, "BG36_851") for unit in old.values()):
                self.witnesses.append([turn, "spark_snapper_play_attachment", gain])
            if 126 <= action < 196:
                self.witnesses.append([turn, "direct_target_attachment", gain])


class Replay:
    def __init__(self, expected, recorder):
        self.actions = expected["actions"]
        self.turns = expected["turns"]
        self.recorder = recorder

    def reset(self):
        self.index = 0

    def decide(self, game, seat):
        from hsbg.az.train.policies import Decision

        turn, action, _ = self.actions[self.index]
        self.index += 1
        if turn != game.turn or not game.legal_mask(seat)[action]:
            raise ValueError(("recorded action no longer applies", self.index, turn, action))
        if action == 0:
            self.recorder.before_end(game)
        forced = action == 0 and self.turns[str(turn)]["forced"]
        return Decision(action, info={"forced": forced})


def replay(job):
    kind, row = job
    panel, seed, seat = row["panel"], row["seed"], row["seat"]
    expected = row["arms"]["0750"]
    recorder = Witness()
    actor = Replay(expected, recorder)
    observed = play(actor, BANKS[panel], seed, seat, 196, recorder)
    observed["turns"] = {str(t): v for t, v in observed["turns"].items()}
    if observed != expected:
        raise ValueError(("replay drift", panel, seed, [k for k in observed if observed[k] != expected[k]]))
    return dict(kind=kind, panel=panel, seed=seed, seat=seat, rank=observed["rank"], events=recorder.witnesses)


def main(source, out, workers):
    os.environ.update(HSAI_ACTION_PROFILE="human-turn-3703-player150", HSAI_HUMAN_TAIL="last-observed-v1")
    import hsbg_engine as engine
    from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
    from workflows.peer_3703_round import field

    if os.environ.get("SLURM_JOB_PARTITION") != "normal" or engine.build_info()["source_hash"] != ENGINE:
        raise ValueError("mechanism replay requires the current pinned engine on a CPU worker")
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    jobs = []
    for panel, expected_digest in EXPECTED_FIELDS.items():
        proto, manifest, _, digest = field(panel)
        if digest != expected_digest:
            raise ValueError("opponent field changed")
        pool = MirrorSnapshotPool(proto.mirror_snaps, sample="random")
        if pool.frames != manifest["frames"] or pool.bad_files:
            raise ValueError("opponent field differs from manifest")
        BANKS[panel] = pool
        for label, pair in CORE.items():
            eligible = [r for r in rows if r["panel"] == panel and pair in sustained(r["arms"]["0750"])]
            jobs.extend((label, row) for row in eligible[:4])
    out.mkdir(parents=True, exist_ok=False)
    os.environ["HSRL_FAILURE_DIR"] = str(out / "failure")
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
        events = list(pool.map(replay, jobs))
    atomic_json(out / "complete.json", dict(source=file_identity(source), cases=events, core=CORE))
    print("exact semantic replays:", len(events), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    main(args.source, args.out, args.workers)
