"""Evaluate the original peer on the confirmed action contract with full traces."""

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(HSAI_ACTION_PROFILE="human-turn-3702-standard", HSAI_HUMAN_TAIL="last-observed-v1")

from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
from hsbg.az.train.move_guard import decode_move, is_move
from hsbg.az.train.personal import library_digest
from hsbg.env.obs_std import GLOBAL_FIELD_INDEX
from hsbg.evaluation.trace import DecisionRecorder, validate_trace
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity
from tools.summarize_peer_finetune import UTILITY, utility_limits
from workflows.peer_exploration import load_peer
from workflows.peer_finetune_holdout import holdout_protocol, read
from workflows.peer_ppo_panel import PEER_SHA, WEIGHT, Peer, Scores

FIRST_SEED = 258910000
GAMES = 1024
SHARDS = 16
SMOKE_GAMES = 8
SMOKE_SEED = 258909800


class AuditedRecorder(DecisionRecorder):
    def begin_game(self, game, game_index, seed, seat):
        super().begin_game(game, len(self.seeds), seed, seat)
        self.turn = None
        self.turns = {}

    def after(self, game, before, decision, executed_action, forced_tier=False):
        turn = before["turn"]
        if turn != self.turn:
            self.turn = turn
            self.previous = -1
            self.turns[turn] = Counter()
        counts = self.turns[turn]
        action = int(executed_action)
        legal = before["legal_mask"]
        if not legal[action] or action == 7 or action != decision.action or forced_tier:
            raise RuntimeError(f"unexpected action {action} at {before['seed']}/{turn}")
        if counts["actions"] >= 200:
            if legal[0]:
                if action != 0 or decision.info.get("forced_reason") != "action_cap":
                    raise RuntimeError("eligible EndTurn after 200 was not forced")
                counts["forced_end_cap"] += 1
            else:
                if not before["state_before"]["players"][before["seat"]]["pending"]:
                    raise RuntimeError("over-budget action without pending choice")
                counts["pending_after_cap"] += 1
        counts["actions"] += 1
        if is_move(action):
            counts["move"] += 1
            if counts["move"] > 10 or (
                    is_move(self.previous) and decode_move(action) == decode_move(self.previous)[::-1]):
                raise RuntimeError("move guard violation")
        if action == 2:
            binary = base64.b64decode(before["observation"]["floats_f32le"])
            frozen = struct.unpack_from("<f", binary, GLOBAL_FIELD_INDEX["frozen"] * 4)[0]
            counts["unfreeze" if frozen > 0.5 else "freeze"] += 1
            if counts["freeze"] + counts["unfreeze"] > 3:
                raise RuntimeError("freeze cap violation")
        self.previous = action
        super().after(game, before, decision, action, forced_tier)


def run_shard(out, start, count, index, *, smoke=False):
    import torch
    torch.set_num_threads(1)
    protocol, pool = holdout_protocol()
    plan = read(out / "plan.json")
    if protocol.sha256() != plan["protocol_sha256"]:
        raise ValueError("worker evaluation protocol mismatch")
    actor, _ = load_peer()
    policy = Peer(Scores(actor, "cuda:0"), protocol, keep=False)
    identity = dict(model=PEER_SHA, protocol=plan["protocol"], protocol_sha256=protocol.sha256())
    prefix = out / f"shard-{index:02d}"
    trace = AuditedRecorder(prefix.with_suffix(".decisions.jsonl.gz"), identity=identity)
    detail = prefix.with_suffix(".games.jsonl.gz")
    pending = Path(str(detail) + ".partial")
    compact = []
    totals = Counter()
    try:
        with gzip.open(pending, "xt", encoding="utf-8", compresslevel=1) as stream:
            for seed in range(start, start + count):
                result = evaluate_policy(policy, EvalConfig(
                    games=1, seed=seed, mirror_pool=pool, mirror_strict=True,
                    collect_economy=True, collect_exposure=True, decision_recorder=trace))
                mirror = result["mirror"]
                if (result["games_finished"] != 1 or result["seed_sequence"] != [seed]
                        or any(result[k] for k in ("illegal_actions", "truncated_games", "truncated_combats"))
                        or not mirror["mirror_applied"] or mirror["mirror_applied"] != mirror["mirror_attempt"]
                        or any(mirror["mirror_" + k] for k in ("rejected", "missing", "errors", "fallback",
                                                               "pool_unbalanced", "pool_species_unbalanced"))):
                    raise ValueError(f"invalid game: {seed}")
                totals.update(result["mirror_strict_counts"])
                totals["injections"] += mirror["mirror_applied"]
                economy = result["end_turn_economy"]
                if not economy or any(not row["eot_available"] for row in economy):
                    raise ValueError(f"missing precombat board: {seed}")
                row = dict(seed=seed, rank=result["rank_sequence"][0], decisions=result["decisions"],
                           turns={item["turn"]: {
                               "attack": item["eot_attack"], "health": item["eot_board_hp"],
                               **trace.turns[item["turn"]],
                           } for item in economy},
                           tail_rank_upper=result["mirror_strict_counts"].get("human_tail_rank_upper", 0),
                           tail_seats=result["mirror_strict_counts"].get("human_last_observed_live_seats", 0))
                row["final_board"] = [{k: card[k] for k in ("card_id", "name", "position", "atk", "hp")}
                                      for card in economy[-1]["eot_board"]]
                combat = result["combat_exposure"]
                row["combat"] = dict(total=len(combat), wins=sum(f["win"] for f in combat),
                                     ties=sum(f["tie"] for f in combat))
                if not combat:
                    raise ValueError(f"missing combat evidence: {seed}")
                stream.write(json.dumps({**row, "economy": economy, "exposure": combat,
                                         "mirror": mirror}, ensure_ascii=False, separators=(",", ":")) + "\n")
                compact.append(row)
                if len(compact) % 8 == 0:
                    print(f"shard {index}: {len(compact)}/{count}", flush=True)
        manifest = trace.finish(dict(games=count, decisions=sum(r["decisions"] for r in compact),
                                     seed_sequence=[r["seed"] for r in compact],
                                     rank_sequence=[r["rank"] for r in compact]))
    except BaseException:
        trace.abort()
        raise
    os.link(pending, detail)
    pending.unlink()
    replay = validate_trace(trace.path, replay=24) if smoke else None
    atomic_json(prefix.with_suffix(".json"), dict(
        rows=compact, totals=dict(totals), trace=manifest, detail=file_identity(detail), replay=replay))


def summarize(out, count, start, shards):
    parts = [read(out / f"shard-{i:02d}.json") for i in range(shards)]
    rows = [row for part in parts for row in part["rows"]]
    if [row["seed"] for row in rows] != list(range(start, start + count)):
        raise ValueError("seed coverage mismatch")
    for part in parts:
        if not part["trace"]["complete"] or part["trace"]["actions"] != sum(r["decisions"] for r in part["rows"]):
            raise ValueError("trace coverage mismatch")
    ranks = Counter(row["rank"] for row in rows)
    limits = [utility_limits(dict(metrics={"rank": row["rank"]}, **{
        k: row[k] for k in ("tail_rank_upper", "tail_seats")})) for row in rows]
    turns = {}
    for turn in sorted({int(t) for row in rows for t in row["turns"]}):
        samples = [v for row in rows if (v := row["turns"].get(str(turn))) is not None]
        bodies = sorted(s["attack"] + s["health"] for s in samples)
        turns[turn] = dict(n=len(bodies), p50=bodies[len(bodies)//2], p90=bodies[int(.90*len(bodies))],
                           move10=sum(s.get("move", 0) == 10 for s in samples),
                           freeze3=sum(s.get("freeze", 0) + s.get("unfreeze", 0) == 3 for s in samples),
                           forced_end=sum(s.get("forced_end_cap", 0) for s in samples),
                           pending_after_cap=sum(s.get("pending_after_cap", 0) for s in samples))
    top = sorted(rows, key=lambda r: sum(m["atk"] + m["hp"] for m in r["final_board"]), reverse=True)[:10]
    totals = sum((Counter(part["totals"]) for part in parts), Counter())
    combats = sum((Counter(row["combat"]) for row in rows), Counter())
    atomic_json(out / "complete.json", dict(
        n=count, start=start, ranks=dict(sorted(ranks.items())), rank_mean=sum(r*c for r,c in ranks.items())/count,
        top4=sum(c for r,c in ranks.items() if r<=4), top2=ranks[1]+ranks[2], wins=ranks[1],
        mean_utility=sum(UTILITY[r-1]*c for r,c in ranks.items())/count,
        utility_bounds=[sum(x[i] for x in limits)/count for i in range(2)],
        combat=dict(combats), turns=turns, totals=dict(totals), top=top,
        tail_games=[{k: row[k] for k in ("seed", "rank", "tail_rank_upper", "tail_seats")}
                    for row in rows if row["tail_seats"]],
        traces=len(parts), trace_bytes=sum(part["trace"]["bytes"] for part in parts),
        actions=sum(part["trace"]["actions"] for part in parts),
        protocol=read(out / "plan.json")["protocol_sha256"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--require-smoke", type=Path)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    start, count, shards = (SMOKE_SEED, SMOKE_GAMES, 2) if args.smoke else (FIRST_SEED, GAMES, SHARDS)
    size = count // shards
    if args.worker is not None:
        run_shard(args.out, start + args.worker * size, size, args.worker, smoke=args.smoke)
        return
    source = source_identity(ROOT)
    protocol, _ = holdout_protocol()
    recipe = read(ROOT / "config/research/human-turn-3702-ppo.json")
    library = library_digest(protocol.mirror_snaps)
    if (source["tracked_dirty"] or library != recipe["library_digests"]["hold"]
            or protocol.engine["source_hash"] != recipe["engine_source_hash"]
            or file_identity(WEIGHT)["sha256"] != PEER_SHA):
        raise ValueError("source, model, engine, or holdout identity mismatch")
    if not args.smoke:
        smoke = read(args.require_smoke / "complete.json")
        if smoke["protocol"] != protocol.sha256() or smoke["n"] != SMOKE_GAMES:
            raise ValueError("smoke coverage mismatch")
    args.out.mkdir(parents=True, exist_ok=False)
    atomic_json(args.out / "plan.json", dict(
        source=source, job=os.environ["SLURM_JOB_ID"], model=file_identity(WEIGHT),
        protocol=protocol.as_dict(), protocol_sha256=protocol.sha256(), library=library,
        engine_lock=file_identity(ROOT / "config/engine-upstream-lock.json"),
        live_pin=file_identity(ROOT / "config/live/model.json"), z="none", first_seed=start, n=count, shards=shards))

    def launch(index):
        with (args.out / f"worker-{index:02d}.log").open("x") as log:
            subprocess.run([sys.executable, __file__, *sys.argv[1:], "--worker", str(index)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)

    with ThreadPoolExecutor(shards) as pool:
        list(pool.map(launch, range(shards)))
    summarize(args.out, count, start, shards)


if __name__ == "__main__":
    main()
