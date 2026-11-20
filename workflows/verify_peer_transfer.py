"""Offline peer-checkpoint transfer check with native lobbies and archived public frames."""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "client")]

import torch
import hsbg_engine as engine
from hsbg.az.train.eval_policy import EvalProtocol, make_eval_policy
from hsbg.az.train.strict_mirror import ECONOMY_ACTION_PROFILE
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsrl_agent.live.actions import bind_actions
from hsrl_agent.live.pipeline import bind_unknown_cards
from hsrl_agent.live.policy import ModelPolicy


def archived(policy, folder):
    from PIL import Image
    from hsrl_agent.live.vision import OCR, races_from_hero_banner
    from hsrl_agent.live.encoding import RACES
    image_file = folder / "visual-evidence/hero-wait-last.jpg"
    metadata = json.loads(image_file.with_suffix(".json").read_bytes())
    with Image.open(image_file) as image:
        races = races_from_hero_banner(OCR(max_width=1280).locate(image))
    if not races:
        raise ValueError("no witnessed five-type subtitle in this archive")
    changed = restored = 0
    hand_frames = undead_frames = positional_frames = 0
    with (folder / "decisions.jsonl").open(encoding="utf8") as f:
        ids = {r["decision_id"] for line in f for r in [json.loads(line)]
               if r["phase"] == "recruit" and r["role"] == "model" and not r["reused"]}
    seen = set()
    unknown_lobby = 0
    kinds = {}
    with gzip.open(folder / "decisions-full.jsonl.gz", "rt", encoding="utf8") as f:
        for line in f:
            record = json.loads(line)
            key = record.get("decision_id")
            if key not in ids or "snapshot" not in record:
                continue
            assert key not in seen
            seen.add(key)
            s = record["snapshot"]
            s["action_profile"] = "peer-r1"
            bind_unknown_cards(s["view"],policy,s["approximation"])
            unknown_lobby += not any(s["view"]["lobby_races"])
            bindings = bind_actions(s)
            sources = {card['entity_id'] for card in s['client']['hand']
                       if card['card_id'] in ('BG_DEEP_015','BG_DEEP_015_G')}
            if sources:
                hand_frames += 1
                magnetic = [b for b in bindings if b.kind == 'magnetize' and b.source in sources]
                positional_frames += any(b.position is not None for b in magnetic)
                undead = {card['entity_id'] for card in s['client']['board']
                          if str(card['tags'].get('CARDRACE')) in ('11', 'UNDEAD')}
                undead_frames += any(b.target in undead for b in magnetic)
            d = policy.decide(s, bindings)
            assert d.binding in bindings
            assert s["game_id"] == metadata["game_id"]
            evidence = [int(race in races) for race in RACES]
            old = s["view"]["lobby_races"]
            if any(old) and old != evidence:
                raise ValueError("contradictory lobby evidence")
            if not any(old):
                repaired = {**s, "view":{**s["view"], "lobby_races":evidence}}
                assert bind_actions(repaired) == bindings
                other = policy.decide(repaired, bindings)
                assert other.binding in bindings
                restored += 1
                changed += other.action != d.action
            kinds[d.binding.kind] = kinds.get(d.binding.kind, 0) + 1
    assert seen == ids and ids
    return dict(frames=len(seen), missing_lobby=unknown_lobby, action_families=kinds,
                verified_races=races, banner=file_identity(image_file), restored=restored, changed_argmax=changed,
                prosthetic_hand_frames=hand_frames, hand_with_undead_target=undead_frames,
                hand_with_positional_magnetic=positional_frames,
                interpretation="Single archived decisions; alternate actions were never dispatched.")


def selfplay(checkpoint, n, seed):
    protocol = EvalProtocol()
    assert (protocol.max_actions_per_turn, protocol.move_cap, protocol.freeze_cap,
            protocol.forbid_unfreeze, protocol.move_repeat_limit) == (300, 10, 3, False, 0)
    actor = make_eval_policy(str(checkpoint), protocol=protocol, device="cpu")["policy"]
    rows = []
    for match_seed in range(seed, seed+n):
        actor.reset()
        g = engine.Game(seed=match_seed)
        lobby = json.loads(g.view(0))["lobby_races"]
        assert len(set(lobby)) == 5
        actions = [0] * 8
        counts_by_turn = {}
        while not g.is_over():
            before = sum(actions)
            for seat in range(8):
                if not g.needs_action(seat):
                    continue
                decision = actor.decide(g, seat)
                assert bytes(g.legal_mask(seat))[decision.action]
                counts = counts_by_turn.setdefault(int(g.turn), [0] * 8)
                counts[seat] += 1
                g.step(seat, decision.action)
                actions[seat] += 1
            if before == sum(actions) or sum(actions) > 60000:
                raise RuntimeError(f"stalled lobby {match_seed}")
        places = list(g.placements())
        assert sorted(places) == list(range(1, 9))
        rows.append(dict(seed=match_seed, lobby=lobby, decisions=actions, turn=g.turn, places=places, counts_by_turn=counts_by_turn))
        print(f"verified full lobby {match_seed}, {sum(actions)} decisions", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--games", default=8, type=int)
    parser.add_argument("--seed", default=37032001, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists() or args.games < 0 or (not args.games and not args.archive):
        raise ValueError("expected a new output and a game or an archived replay")
    if os.environ.get("HSAI_ACTION_PROFILE") != ECONOMY_ACTION_PROFILE or not engine.build_is_current(ROOT/"engine"):
        raise ValueError("peer action profile and matching compiled engine are required")
    data = engine.data_info()
    if data["server_rules_version"] != "37.0.3":
        raise ValueError("37.0.3 engine required")
    torch.set_num_threads(1)
    policy = ModelPolicy(args.checkpoint)
    if not policy.peer_round:
        raise ValueError("peer round checkpoint required")
    rows = selfplay(args.checkpoint, args.games, args.seed) if args.games else []
    frames = archived(policy, args.archive) if args.archive else None
    atomic_json(args.out, dict(model=policy.sha, engine=engine.build_info()["source_hash"], data=data,
                card_table=file_identity(ROOT/"engine/crates/hsbg-engine/data/cards.json"),
                profile=ECONOMY_ACTION_PROFILE, full_native_games=rows, archived_frames=frames))


if __name__ == "__main__":
    main()
