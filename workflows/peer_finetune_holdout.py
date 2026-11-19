"""Fresh, paired full games with same-state protective checks for peer PPO."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.runtime.artifacts import atomic_json, file_identity
from tools.summarize_peer_finetune import UTILITY, compare, estimate
from workflows import peer_ppo_panel as panel
from workflows.human_turn_preflight import verified_generation
from workflows.d142_crossover import Fork, PrefixComplete, play, ResumedPolicy
from hsbg.az.train.strict_mirror import OPEN_ACTION_PROFILE, STANDARD_ACTION_PROFILE

PROTOCOL = "ec900f2c14cd9922d2b2e9b2f68835b703d57090c9e68d666f4620930351bcd4"
OPEN_PROTOCOL = "6a639e320632892fbdf3b632935e63df3948e0f21154e00aa52426474250aeb2"
STANDARD_PROTOCOL = "1d45f262cdb8323820beb2f258e62cdc71d6201f097ed5d175899fbad215a350"
INITIAL = Path("artifacts/runs/nus-peer-heads-20261119-1/initial/model.pt")
INITIAL_SHA = "55f6665c0b357860eefd6b6916620a0118e1dc2c5d5e77bbea9136f64f48c5a9"


def read(path):
    return json.loads(Path(path).read_bytes())


def holdout_protocol():
    from hsbg.az.train.personal import library_digest
    profile = os.environ.get("HSAI_ACTION_PROFILE")
    if profile not in (OPEN_ACTION_PROFILE, STANDARD_ACTION_PROFILE):
        proto, pool = panel.Q.config("hold", allow_tail_variant=True)
        expected = PROTOCOL
    else:
        from hsbg.az.train.common import engine_info
        from hsbg.az.train.eval_policy import TRAINING_CONTRACT, protocol_from_args
        from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool

        opts = read(ROOT / "config/evaluation/human-turn-3702-parent.json")["evaluation"]
        fields = {key.replace("-", "_"): value for key, value in opts.items()}
        if profile == STANDARD_ACTION_PROFILE:
            fields.update(keep_start_board=False, tp_forbid_cancel=True)
        proto = protocol_from_args(argparse.Namespace(**(fields | TRAINING_CONTRACT)), engine=engine_info())
        proto.assert_training_contract()
        pool = MirrorSnapshotPool(proto.mirror_snaps, sample=proto.mirror_sample)
        expected = STANDARD_PROTOCOL if profile == STANDARD_ACTION_PROFILE else OPEN_PROTOCOL
    if proto.sha256() != expected or pool.frames != 5477:
        raise ValueError("hold protocol or library size mismatch")
    expected_library = read(ROOT / "config/research/human-turn-3702-ppo.json")["library_digests"]["hold"]
    if library_digest(proto.mirror_snaps) != expected_library:
        raise ValueError("hold library content mismatch")
    return proto, pool


def check_states(j, seed, parent, players, pool, rows):
    import hsbg_engine
    from hsbg.az.train.economic_tasks import card_ids
    fork = Fork(stop=True)
    fork.turn = 9 + j % 4
    try:
        play(pool, parent, seed, capture=fork)
    except PrefixComplete:
        pass
    if fork.state is None:
        return None
    game = hsbg_engine.Game.from_json(json.dumps(fork.state))
    seat = json.loads(game.player_json(0))
    hp = seat["health"] + seat["armor"]
    cards = {m["card"] for m in seat["board"]}
    ids = card_ids()
    groups = ["low"] if hp <= 15 else []
    for name, keys in (("mech", ("BG36_366","BG26_152")), ("dragon", ("BG36_364","BG29_813"))):
        if any(ids.get(k) in cards for raw in keys for k in (raw, raw+"_G")):
            groups.append(name)
    if not groups:
        return None
    ranks = {k: play(pool, ResumedPolicy(fork, p.evaluator), seed, resume=fork)["rank"]
             for k,p in players.items()}
    if ranks["parent"] != rows["parent"]["metrics"]["rank"]:
        raise ValueError("unchanged parent did not reproduce the natural rank")
    return dict(seed=seed, turn=fork.turn, hp=hp, opening=fork.opening, groups=groups, ranks=ranks)


def selected_field(split):
    if split == "hold":
        return holdout_protocol()
    from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
    from workflows.peer_health_switch import field
    proto, _ = field()
    return proto, MirrorSnapshotPool(proto.mirror_snaps, sample=proto.mirror_sample)


def shard(args):
    import torch
    torch.set_num_threads(1)
    plan = read(args.out / "plan.json")
    proto, pool = selected_field(args.split)
    if proto.sha256() != plan["protocol"]:
        raise ValueError("hold protocol differs from the assessment plan")
    players = {"candidate": panel.Q.policy(Path(plan["models"]["candidate"]), proto)}
    if plan.get("raw_parent"):
        from workflows.peer_exploration import load_peer

        raw, _ = load_peer()
        players["parent"] = panel.Peer(panel.Scores(raw, "cuda:0"), proto, keep=False)
        wrapped = panel.Q.policy(INITIAL, proto)
        seed = args.seed + args.worker
        if panel.trial(players["parent"], pool, seed, event_turns=True) != panel.trial(
                wrapped, pool, seed, event_turns=True):
            raise ValueError(f"wrapped starting policy differs from raw actor on seed {seed}")
    else:
        players["parent"] = panel.Q.policy(Path(plan["models"]["parent"]), proto)
    records, matched = [], []
    for j in range(args.worker, args.games, 8):
        seed = args.seed + j
        row = {k: panel.trial(p, pool, seed, event_turns=True) for k,p in players.items()}
        records.append(row)
        if j < args.protected:
            item = check_states(j, seed, players["parent"], players, pool, row)
            if item:
                matched.append(item)
    atomic_json(args.out / f"panel-{args.worker}.json", records)
    atomic_json(args.out / f"matched-{args.worker}.json", matched)


def run(args):
    from hsbg.runtime.provenance import source_identity
    source = source_identity(ROOT)
    seal = verified_generation(args.segment, args.train_seed)
    if source["tracked_dirty"] or file_identity(INITIAL)["sha256"] != INITIAL_SHA:
        raise ValueError("unexpected release or parent")
    if not (0 <= args.protected <= args.games and args.games > 0):
        raise ValueError("invalid cohort sizes")
    protocol, _ = selected_field(args.split)
    if args.raw_parent and protocol.sha256() != STANDARD_PROTOCOL:
        raise ValueError("raw parent assessment requires the confirmed action protocol")
    parent = str((panel.WEIGHT if args.raw_parent else INITIAL).resolve())
    parent_sha = panel.PEER_SHA if args.raw_parent else INITIAL_SHA
    if args.parent_model or args.parent_sha:
        if args.raw_parent or not args.parent_model or file_identity(args.parent_model)["sha256"] != args.parent_sha:
            raise ValueError("invalid comparison policy")
        parent, parent_sha = str(args.parent_model.resolve()), args.parent_sha
    args.out.mkdir(parents=True, exist_ok=False)
    plan = dict(source=source, seed=args.seed, games=args.games,
        matched_prefixes=args.protected, protocol=protocol.sha256(), candidate=seal["files"]["model.pt"],
        models=dict(parent=parent, candidate=seal["model"]), initial_sha=INITIAL_SHA,
        raw_parent=args.raw_parent, parent_sha=parent_sha, split=args.split)
    workers = list(range(8))
    if args.reuse_complete:
        old = read(args.reuse_complete / "plan.json")
        required = ("seed", "games", "matched_prefixes", "protocol", "candidate", "initial_sha", "raw_parent", "parent_sha")
        if any(plan[key] != old[key] for key in required):
            raise ValueError("recovered assessment plan differs")
        reused = {}
        for worker in workers:
            p = args.reuse_complete / f"panel-{worker}.json"
            m = args.reuse_complete / f"matched-{worker}.json"
            if not p.is_file() or not m.is_file():
                continue
            rows, states = read(p), read(m)
            seeds = list(range(args.seed + worker, args.seed + args.games, 8))
            if ([row["parent"]["seed"] for row in rows] != seeds
                    or any(set(row) != {"parent", "candidate"} or
                           (row["parent"]["seed"], row["parent"]["lobby"]) !=
                           (row["candidate"]["seed"], row["candidate"]["lobby"]) for row in rows)
                    or any(item["seed"] not in seeds or item["seed"] >= args.seed + args.protected for item in states)):
                raise ValueError(f"invalid recovered worker {worker}")
            atomic_json(args.out / p.name, rows)
            atomic_json(args.out / m.name, states)
            reused[str(worker)] = {f.name: file_identity(f) for f in (p, m)}
        workers = [worker for worker in workers if str(worker) not in reused]
        plan["reused"] = dict(origin=str(args.reuse_complete.resolve()), files=reused)
    atomic_json(args.out / "plan.json", plan)
    def launch(i):
        with (args.out / f"worker-{i}.log").open("x") as log:
            subprocess.run([sys.executable, __file__, *sys.argv[1:], "--worker", str(i)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(launch, workers))
    natural = compare(args.out, args.seed, args.games)
    matched = [v for i in range(8) for v in read(args.out / f"matched-{i}.json")]
    if len({v["seed"] for v in matched}) != len(matched):
        raise ValueError("duplicate matched seed")
    natural["same_state"] = {}
    for group in ("low", "mech", "dragon"):
        ranks = [v["ranks"] for v in matched if group in v["groups"]]
        natural["same_state"][group] = dict(n=len(ranks),
            top4={k: sum(r[k]<=4 for r in ranks) for k in ("parent","candidate")},
            utility=estimate([UTILITY[r["candidate"]-1] - UTILITY[r["parent"]-1] for r in ranks]))
    atomic_json(args.out / "complete.json", natural)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-seed", type=int, default=232700000)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--protected", type=int, default=256)
    parser.add_argument("--raw-parent", action="store_true")
    parser.add_argument("--parent-model", type=Path)
    parser.add_argument("--parent-sha")
    parser.add_argument("--split", choices=("train", "hold"), default="hold")
    parser.add_argument("--reuse-complete", type=Path)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    run(args) if args.worker is None else shard(args)
