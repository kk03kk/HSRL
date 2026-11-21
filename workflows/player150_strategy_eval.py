"""Run a CPU-only paired operating-style assessment of R1 and Player150 3286."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity

PANELS = {"hold": (582410000, 552410000, "monitor"),
          "human": (585410000, 555410000, "external")}
EXPECTED = {"hold": "5361375f96441316237a792205b5ff76587bff27a9fd80d7cad25307075927d9",
            "human": "24d387ec449479425bd356b61c639b35f6095b4a32cab700519a0a62a04581f4"}
R5_SHA = "778cc6f4ca9ba6175ec6f54c2241a030a70ce4e35b7ac7120492666360ba86df"
R1_SHA = "1bcb12c6038f0d07223d925893c58a4be244531769f32e9820eeb2e9f10e134a"
ENGINE = "8bd1f4785018c474"
END = Path("artifacts/runs/nus-player150-five-20261120-01/round-05")
ACTORS = {}
BANKS = {}


def read(path):
    return json.loads(path.read_bytes())


def game(policy, pool, seed):
    from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
    from workflows.joint_short_assessment import Mechanisms

    recorder = Mechanisms()
    result = evaluate_policy(policy, EvalConfig(
        games=1, seed=seed, mirror_pool=pool, mirror_strict=True,
        collect_economy=True, decision_recorder=recorder))
    mirror = result["mirror"]
    if (result["games_finished"] != 1 or result["seed_sequence"] != [seed]
            or any(result[k] for k in ("illegal_actions", "truncated_games", "truncated_combats"))
            or not mirror["mirror_applied"] or mirror["mirror_applied"] != mirror["mirror_attempt"]
            or any(mirror["mirror_"+k] for k in
                   ("errors", "rejected", "missing", "fallback", "pool_unbalanced", "pool_species_unbalanced"))):
        raise ValueError(("invalid paired match", seed, mirror))
    turns = {}
    for stage in result["end_turn_economy"]:
        turn = int(stage["turn"])
        if turn in turns or not stage["eot_available"]:
            raise ValueError(("invalid turn checkpoint", seed, turn))
        turns[turn] = dict(
            tier=stage["tier"], gold=stage["gold"], health=stage["health"], hand_n=stage["hand_n"],
            actions=stage["actions"], counts=stage["action_counts"], hero=stage["hero"],
            trinkets=stage["trinkets"], rewards=stage["rewards"],
            board=[[x["card_id"], x["atk"], x["hp"], x["eid"], x.get("kw", 0)]
                   for x in stage["eot_board"]],
            strength=stage["eot_attack"] + stage["eot_board_hp"],
        )
    return dict(rank=result["rank_sequence"][0], lobby=recorder.lobby, turns=turns,
                decisions=result["decisions"], injections=mirror["mirror_applied"],
                tail=result["mirror_strict_counts"].get("human_last_observed_live_seats", 0),
                absorptions=recorder.absorptions, unmeasured_absorptions=recorder.unmeasured_absorptions,
                gems=recorder.measured, magnet=recorder.mechanics, hope_growth=sorted(recorder.hope_growth),
                offers=sorted(recorder.offers), buys=sorted(recorder.buys))


def pair(task):
    panel, seed = task
    arms = [game(policy, BANKS[panel], seed) for policy in ACTORS[panel]]
    if arms[0]["lobby"] != arms[1]["lobby"]:
        raise ValueError(("paired lobby differs", seed))
    return dict(panel=panel, seed=seed, r1=arms[0], final=arms[1])


def compare_reference(row, expected):
    for name, arm in (("r1", "base"), ("final", "candidate")):
        actual = row[name]
        comparable = dict(rank=actual["rank"], injections=actual["injections"], tail=actual["tail"],
                          boards={str(t): data["strength"] for t, data in actual["turns"].items()})
        if comparable != expected[arm]:
            raise ValueError(("formal replay mismatch", row["panel"], row["seed"], name, comparable, expected[arm]))


def run(out, n, anchors):
    if os.environ.get("SLURM_JOB_PARTITION") != "normal" or not 1 <= n <= 192 or not 1 <= anchors <= 8:
        raise ValueError("requires a bounded normal CPU job")
    for key in tuple(os.environ):
        if key.startswith(("HSAI_ARENA_", "HSAI_RECRUIT_START_", "HSAI_QUILL_")):
            del os.environ[key]
    os.environ.update(HSAI_ACTION_PROFILE="human-turn-3703-player150", HSAI_HUMAN_TAIL="last-observed-v1")
    import hsbg_engine as engine
    import torch
    from hsbg.az.train.eval_policy import make_eval_policy
    from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
    from hsbg.az.train.player_observation import SCHEMA_SHA
    from workflows.peer_3703_field import R1
    from workflows.peer_3703_round import field

    torch.set_num_threads(1)
    source = source_identity(ROOT)
    if (source["tracked_dirty"] or not engine.build_is_current(ROOT/"engine")
            or engine.build_info()["source_hash"] != ENGINE):
        raise ValueError("unexpected native engine or dirty code")
    seal = read(END/"train/complete.json")["checkpoint"]
    candidate = Path(seal["model"])
    models = dict(r1=file_identity(R1), final=file_identity(candidate))
    if (models["r1"]["sha256"] != R1_SHA or models["final"]["sha256"] != R5_SHA
            or not seal["final"] or seal["version"] != 3286 or models["final"] != seal["files"]["model.pt"]):
        raise ValueError("wrong R1 or final checkpoint")
    protocols, references, tests = {}, {}, []
    for panel, (start, reference_start, formal) in PANELS.items():
        protocol, manifest, spec, digest = field(panel)
        if digest != EXPECTED[panel]:
            raise ValueError("frozen field protocol drift")
        bank = MirrorSnapshotPool(protocol.mirror_snaps, sample="random")
        if bank.frames != manifest["frames"] or bank.bad_files:
            raise ValueError("invalid opponent bank")
        BANKS[panel] = bank
        ACTORS[panel] = [make_eval_policy(str(path), protocol=protocol, device="cpu")["policy"]
                         for path in (R1, candidate)]
        published = read(END/formal/"complete.json")
        if published["protocol_sha256"] != digest or published["start"] != reference_start:
            raise ValueError("incorrect reference evaluation")
        references[panel] = {r["seed"]: r for r in read(END/formal/"pairs.json")[:anchors]}
        tests.extend((panel, seed) for seed in references[panel])
        protocols[panel] = dict(protocol_sha256=digest, spec=spec, start=start, pairs=n,
                                reference_start=reference_start, anchors=anchors)
    out.mkdir(parents=True, exist_ok=False)
    plan = dict(source=source, engine=ENGINE, observation_sha256=SCHEMA_SHA, models=models, panels=protocols,
                engine_lock=file_identity(ROOT/"config/engine-upstream-lock.json"), job=os.environ["SLURM_JOB_ID"])
    atomic_json(out/"plan.json", plan)
    workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    with ProcessPoolExecutor(min(workers, 2*n), mp_context=multiprocessing.get_context("fork")) as pool:
        for row in pool.map(pair, tests):
            compare_reference(row, references[row["panel"]][row["seed"]])
        print(f"reproduced {len(tests)} archived pairs using original R1", flush=True)
        tasks = [(panel, seed) for panel, (start, _, _) in PANELS.items() for seed in range(start, start+n)]
        results = []
        for future in as_completed([pool.submit(pair, task) for task in tasks]):
            results.append(future.result())
            if len(results) % 24 == 0:
                print(f"paired games: {len(results)}/{len(tasks)}", flush=True)
    results.sort(key=lambda row: (row["panel"], row["seed"]))
    if [(row["panel"],row["seed"]) for row in results] != sorted(tasks):
        raise ValueError("missing or duplicate seeds")
    temp = out/"pairs.jsonl.gz.partial"
    with gzip.open(temp, "wt", encoding="utf8", compresslevel=3) as stream:
        for row in results:
            stream.write(json.dumps(row, separators=(",", ":"))+"\n")
    temp.rename(out/"pairs.jsonl.gz")
    atomic_json(out/"complete.json", dict(plan=plan, anchors=len(tests), pairs=len(results),
                                           evidence=file_identity(out/"pairs.jsonl.gz")))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=192)
    parser.add_argument("--anchors", type=int, default=8)
    args = parser.parse_args()
    os.chdir(ROOT)
    run(args.out, args.pairs, args.anchors)
