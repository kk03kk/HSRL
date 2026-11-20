"""Prepare and run the independently queued rounds of the frozen peer training series."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
from statistics import mean, median
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.runtime.artifacts import atomic_json, file_identity
from hsbg.runtime.provenance import source_identity

RECIPE = ROOT / "config/research/peer-gpu-rounds.json"
PANELS = ("monitor", "blind", "adaptive")


def read(path):
    return json.loads(Path(path).read_bytes())


def settings(smoke):
    cfg = read(RECIPE)
    if smoke:
        trial = cfg["smoke"]
        cfg.update(rounds=trial["rounds"], train_seed=trial["train_seed"])
        for name in PANELS:
            cfg[name] = dict(seed=trial[name + "_seed"], games=trial["games"])
    panels = [range(cfg[k]["seed"], cfg[k]["seed"] + cfg[k]["games"]) for k in PANELS]
    if max(p.stop for p in panels) >= cfg["train_seed"] or len(set().union(*panels)) != sum(map(len, panels)):
        raise ValueError("overlapping panel or training seeds")
    return cfg


def stages(rounds):
    for index in range(1, rounds + 1):
        yield index, "train"
        yield index, "monitor"
    yield rounds, "blind"
    yield rounds, "adaptive"


def job_args(stage, smoke):
    exclusions = ["--exclude=xgpi11,xgpi15,xgpi17"]
    if stage == "adaptive":
        return ["-p", "normal", "-c", "32", "--mem=48G", "-t", "01:50:00"]
    if stage == "train":
        return ["-p", "gpu", "--gres=gpu:h100-47:2", "-c", "32", "--mem=96G",
                "-t", "00:15:00" if smoke else "03:00:00", *exclusions]
    return ["-p", "gpu", "--gres=gpu:h100-47:1", "-c", "16", "--mem=48G",
            "-t", "00:15:00" if smoke else "00:55:00", *exclusions]


def submit(out, cfg, smoke, dry_run):
    source = source_identity(ROOT)
    if source["tracked_dirty"]:
        raise ValueError("use a clean, frozen release")
    if not dry_run:
        if ROOT.parent.name != "releases":
            raise ValueError("submit from an immutable server release")
        out.mkdir(parents=True, exist_ok=False)
        atomic_json(out / "series.json", dict(source=source, recipe=file_identity(RECIPE), smoke=smoke))
    parent = None
    for index, stage in stages(cfg["rounds"]):
        cmd = ["sbatch", "--parsable", "-A", "allusers", "-q", "normal", "--propagate=NONE",
               "--kill-on-invalid-dep=yes", "-J", f"peer_r{index}_{stage}",
               "--output=artifacts/runs/slurm/%x-%j.log", *job_args(stage, smoke)]
        if parent:
            cmd.append("--dependency=afterok:" + parent)
        cmd.extend(["deploy/slurm/peer_round_stage.sh", stage, str(out), str(index)])
        if smoke:
            cmd.append("--smoke")
        if dry_run:
            print(" ".join(cmd))
            parent = f"<{stage}-{index}>"
        else:
            parent = subprocess.check_output(cmd, text=True).strip().split(";")[0]
            if not parent.isdigit():
                raise ValueError("unexpected scheduler response")
            atomic_json(out / f"submit-{index}-{stage}.json", dict(job=parent, command=cmd))
            print(index, stage, parent, flush=True)


def adaptive_pair(index):
    from workflows.peer_adaptive_lobby import play
    return {arm: play(OPPONENTS, index, False, START, focal_policy=FOCUS[arm])
            for arm in ("parent", "candidate")}


def adaptive_panel(out, cfg, checkpoint, protocol):
    global OPPONENTS, FOCUS, START
    import torch
    from hsbg.az.train.eval_policy import make_eval_policy
    from tools.summarize_peer_finetune import estimate, UTILITY
    from workflows import quilboar_study as native
    from workflows.peer_finetune_holdout import INITIAL
    torch.set_num_threads(1)
    if file_identity(native.BASE)["sha256"] != native.PLAN["base_sha"]:
        raise ValueError("native opponent changed")
    def actor(path):
        return make_eval_policy(str(path), protocol=protocol, device="cpu")["policy"]
    OPPONENTS = dict(peer=actor(INITIAL), native=actor(native.BASE))
    FOCUS = dict(parent=actor(INITIAL), candidate=actor(checkpoint["model"]))
    START = cfg["adaptive"]["seed"]
    games = cfg["adaptive"]["games"]
    with ProcessPoolExecutor(min(games, int(os.environ["SLURM_CPUS_PER_TASK"])),
                             mp_context=multiprocessing.get_context("fork")) as pool:
        rows = list(pool.map(adaptive_pair, range(games)))
    if any((g["seed"], g["focus"]) != (START + i, i % 8) for i, row in enumerate(rows) for g in row.values()):
        raise ValueError("adaptive panel coverage mismatch")
    atomic_json(out / "lobbies.json", rows)
    common = [row for row in rows if all(9 in g["boards"] and 12 in g["boards"] for g in row.values())]
    def growth(g):
        return math.log1p(g["boards"][12]) - math.log1p(g["boards"][9])
    arms = {}
    for arm in FOCUS:
        ranks = [row[arm]["rank"] for row in rows]
        turns = {}
        for turn in range(1, max(row[arm]["turn"] for row in rows) + 1):
            bodies = [r[arm]["boards"][turn] for r in rows if turn in r[arm]["boards"]]
            turns[turn] = dict(n=len(bodies), p50=median(bodies) if bodies else None)
        arms[arm] = dict(rank=mean(ranks), wins=ranks.count(1), top4=sum(r <= 4 for r in ranks), turns=turns)
    field = dict(schema="hsai.eight-seat-peer-panel/v1", engine=protocol.engine, contract=protocol.contract_dict(),
                 engine_rules=protocol.as_dict()["engine_rules"], native_at_relative_seats=[2, 4, 6],
                 native_sha256=native.PLAN["base_sha"], peer_sha256=file_identity(INITIAL)["sha256"],
                 seat_schedule="seed*1000+turn shuffle", focal_seat="index modulo 8", snapshot_seats=0)
    field_sha = hashlib.sha256(json.dumps(field, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(n=games, models=arms, field=field, protocol_sha256=field_sha, delta=dict(
        rank=estimate([r["candidate"]["rank"] - r["parent"]["rank"] for r in rows]),
        utility=estimate([UTILITY[r["candidate"]["rank"]-1] - UTILITY[r["parent"]["rank"]-1] for r in rows]),
        growth9to12=estimate([growth(r["candidate"]) - growth(r["parent"]) for r in common])))


def training_round(out, attempt, index, cfg, smoke, protocol, parent_recipe):
    from workflows.human_turn_preflight import allocation_end, verified_generation
    from workflows.peer_open_ppo import inspect_trainer
    from workflows.peer_finetune_holdout import INITIAL, INITIAL_SHA
    from workflows.peer_ppo_panel import PEER_SHA, WEIGHT
    from hsbg.az.train.personal import checked_library
    source_cfg = read(ROOT / "config/research/human-turn-3702-ppo.json")
    libs = {}
    for split, path in (("train", cfg["environment"]["HSAI_ARENA_SNAPS"]), ("hold", protocol.mirror_snaps)):
        lib = checked_library(path)
        if lib["split"] != split or lib["digest"] != source_cfg["library_digests"][split]:
            raise ValueError("training library contents changed")
        libs[split] = set(lib["games"])
    if libs["hold"] & libs["train"]:
        raise ValueError("leaked opponent source game")
    if (file_identity(INITIAL)["sha256"] != INITIAL_SHA or file_identity(WEIGHT)["sha256"] != PEER_SHA
            or file_identity(INITIAL.with_name("manifest.json"))["sha256"] != parent_recipe["initial_manifest_sha256"]):
        raise ValueError("unverified initial model")
    seed = cfg["train_seed"]
    parent = None
    if index > 1:
        previous = out / f"round-{index-1:02d}"
        previous_train = read(previous / "train.json")
        panel = read(previous / "monitor.json")
        parent = verified_generation(Path(previous_train["fit"]), seed)
        if (parent != previous_train["checkpoint"] or panel["checkpoint"] != parent["files"]["model.pt"]
                or panel["protocol"] != protocol.sha256() or panel["n"] != cfg["monitor"]["games"]
                or file_identity(panel["report"]) != panel["report_identity"]):
            raise ValueError("previous training or monitoring seal mismatch")
    fit = attempt / "fit"
    stop = allocation_end() - cfg["save_margin_seconds"]
    if stop - time.time() < 120:
        raise ValueError("insufficient allocation for rollout and save")
    env = dict(cfg["environment"], HSAI_ACTION_PROFILE=cfg["action_profile"], HSRL_ROOT=str(ROOT),
               HSAI_ARENA_OUT=str(fit), HSAI_ARENA_SEED=str(seed),
               HSAI_ARENA_CKPT=parent["model"] if parent else str(INITIAL.resolve()),
               HSAI_ARENA_MINUTES=str(cfg["minutes_per_round"]), HSAI_ARENA_STOP_EPOCH=str(stop),
               HSAI_ARENA_ITERATIONS=str(cfg["smoke"]["batches"] if smoke else 0))
    if parent:
        env["HSAI_ARENA_PARENT_TRAINER"] = parent["trainer"]
    atomic_json(attempt / "plan.json", dict(environment=env, parent=parent, protocol=protocol.as_dict(),
                libraries=source_cfg["library_digests"], engine_lock=file_identity(ROOT/"config/engine-upstream-lock.json"),
                live_pin=file_identity(ROOT/"config/live/model.json"), initial_manifest=parent_recipe["initial_manifest_sha256"]))
    subprocess.run(["bash", "deploy/slurm/stage_arena.sbatch"], env=os.environ | env, check=True)
    current = verified_generation(fit, seed)
    version = parent["version"] if parent else 0
    if (current["version"] <= version or current["cum_decisions"] <= (parent["cum_decisions"] if parent else 0)
            or (smoke and current["version"] != version + cfg["smoke"]["batches"])):
        raise ValueError("invalid finalized checkpoint")
    log = (fit / "train.log").read_text(encoding="utf8")
    if f"[resume] start_iter={version} updates={version}" not in log:
        raise ValueError("trainer started at an unexpected version")
    if parent and f"[trainer] loaded updates={version}" not in log:
        raise ValueError("full Trainer restore missing")
    return dict(checkpoint=current, trainer=inspect_trainer(current), fit=str(fit),
                parent_version=version, protocol=protocol.sha256())


def run(out, cfg, index, stage, smoke):
    from workflows.human_turn_preflight import verified_generation
    from workflows.peer_finetune_holdout import holdout_protocol
    series = read(out / "series.json")
    if (series["recipe"] != file_identity(RECIPE) or series["smoke"] != smoke
            or series["source"]["commit"] != source_identity(ROOT)["commit"] or source_identity(ROOT)["tracked_dirty"]
            or (index, stage) not in stages(cfg["rounds"])):
        raise ValueError("unexpected release, stage or recipe")
    if os.environ["SLURM_JOB_PARTITION"] != ("normal" if stage == "adaptive" else "gpu"):
        raise ValueError("unexpected partition")
    protocol, _ = holdout_protocol()
    short = read(ROOT / cfg["parent_recipe"])
    if protocol.sha256() != cfg["protocol_sha256"] or short["protocol_sha256"] != protocol.sha256():
        raise ValueError("action protocol drift")
    directory = out / f"round-{index:02d}"
    seal = directory / f"{stage}.json"
    if seal.exists():
        raise FileExistsError(seal)
    attempt = directory / f"{stage}-{os.environ['SLURM_JOB_ID']}"
    attempt.mkdir(parents=True, exist_ok=False)
    if stage == "train":
        result = training_round(out, attempt, index, cfg, smoke, protocol, short)
    else:
        training = read(directory / "train.json")
        checkpoint = verified_generation(Path(training["fit"]), cfg["train_seed"])
        if checkpoint != training["checkpoint"]:
            raise ValueError("checkpoint differs from trained endpoint")
        if stage != "monitor":
            prev = read(directory / ("monitor.json" if stage == "blind" else "blind.json"))
            if prev["checkpoint"] != checkpoint["files"]["model.pt"] or file_identity(prev["report"]) != prev["report_identity"]:
                raise ValueError("previous assessment incomplete")
        if stage == "adaptive":
            report = attempt / "complete.json"
            atomic_json(report, adaptive_panel(attempt, cfg, checkpoint, protocol))
        else:
            target = attempt / "paired"
            subprocess.run([sys.executable, "workflows/peer_finetune_holdout.py", "--segment", training["fit"],
                            "--out", str(target), "--train-seed", str(cfg["train_seed"]),
                            "--seed", str(cfg[stage]["seed"]), "--games", str(cfg[stage]["games"]),
                            "--protected", "0", "--raw-parent"], check=True)
            report = target / "complete.json"
        scores = read(report)
        if scores["n"] != cfg[stage]["games"]:
            raise ValueError("missing evaluation games")
        result = dict(n=scores["n"], seed=cfg[stage]["seed"], report=str(report), report_identity=file_identity(report),
                      field=stage, checkpoint=checkpoint["files"]["model.pt"],
                      protocol=scores["protocol_sha256"] if stage == "adaptive" else protocol.sha256(),
                      cumulative_decisions=checkpoint["cum_decisions"])
        if stage == "adaptive":
            curve = []
            for i in range(1, index + 1):
                monitor = read(out / f"round-{i:02d}/monitor.json")
                if file_identity(monitor["report"]) != monitor["report_identity"]:
                    raise ValueError("monitor evidence changed")
                paired = read(monitor["report"])
                curve.append(dict(round=i, decisions=monitor["cumulative_decisions"], models=paired["models"],
                                  delta=paired["delta"], report_identity=monitor["report_identity"]))
            blind = read(prev["report"])
            promotion = None
            if not smoke:
                delta = blind["delta"]
                promotion = (delta["rank"]["ci95"][1] < 0 and delta["growth9to12"]["ci95"][0] > 0
                             and delta["utility"]["ci95"][0] > 0 and delta["reach12"]["mean"] >= -0.02
                             and scores["delta"]["rank"]["mean"] <= 0 and scores["delta"]["growth9to12"]["mean"] >= 0)
            atomic_json(attempt / "series-summary.json", dict(curve=curve, blind=blind, adaptive=scores,
                        smoke=smoke, promotion_evidence=promotion))
            result["summary"] = str(attempt / "series-summary.json")
    atomic_json(seal, dict(result, job=os.environ["SLURM_JOB_ID"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preview", "submit", "train", *PANELS))
    parser.add_argument("out", type=Path)
    parser.add_argument("index", nargs="?", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    cfg = settings(args.smoke)
    if args.stage in ("preview", "submit"):
        submit(args.out.resolve(), cfg, args.smoke, args.stage == "preview")
    else:
        for key in list(os.environ):
            if key.startswith(("HSAI_ARENA_", "HSAI_RECRUIT_START_", "HSAI_QUILL_")):
                del os.environ[key]
        os.environ.update(HSAI_ACTION_PROFILE=cfg["action_profile"], HSAI_HUMAN_TAIL=cfg["environment"]["HSAI_HUMAN_TAIL"])
        run(args.out.resolve(), cfg, args.index, args.stage, args.smoke)
