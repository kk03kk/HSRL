#!/usr/bin/env python3
"""有限预算、多训练seed的个人实验调度；复用现有train/eval/配对工具。"""
from contextlib import contextmanager
import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


@contextmanager
def writer_lock(out):
    """进程退出时由OS释放锁；保留锁文件，不用删除所谓过期锁。"""
    out.mkdir(parents=True, exist_ok=True)
    with (out / "writer.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("实验目录已有写入者，禁止并发训练") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def freeze_checkpoint(train, milestone, target):
    """保留阶段权重和完整trainer，独立于训练目录的轮转快照。"""
    done = milestone / "checkpoint.json"
    if done.exists():
        record = read(done)
        if record["target_decisions"] != target:
            raise ValueError("阶段checkpoint预算标签不一致")
        for name in ("model.pt", "trainer.pt"):
            if digest(milestone / name) != record["sha256"][name]:
                raise ValueError("阶段checkpoint摘要不符")
        return record
    latest = read(train / "latest.json")
    actual = int(latest["cum_decisions"])
    if actual < target:
        raise RuntimeError(f"预算尚未完成：{actual} < {target}；保留快照后重跑同阶段")
    milestone.mkdir(parents=True, exist_ok=True)
    snapshot = train / "snapshots" / f"step_{int(latest['version']):06d}"
    hashes = {}
    for name in ("model.pt", "trainer.pt"):
        src, dest = snapshot / name, milestone / name
        sha = digest(src)
        if dest.exists() and digest(dest) != sha:
            raise ValueError("未完成的阶段副本与当前快照不同，保留现场并停止")
        if not dest.exists():
            tmp = dest.with_suffix(".pt.tmp")
            shutil.copyfile(src, tmp)
            os.replace(tmp, dest)
        hashes[name] = sha
    record = {"target_decisions": target, "actual_decisions": actual,
              "version": latest["version"], "sha256": hashes}
    write(done, record)
    return record


def eval_stop(pair):
    """预先约定的退化止损；CI未显著不等于已经证明非劣。"""
    rank_harm = pair["delta"] >= .15 and pair["ci_low"] > 0
    top4_harm = pair["top4_delta"] <= -.05 and pair["top4_delta_ci95"][1] < 0
    return bool(rank_harm or top4_harm)


def training_report(train, milestone, version):
    from workflows.report_personal_run import summarize
    records = []
    for line in (train / "history.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("iter") is None or int(row["iter"]) >= version:
            continue
        for epoch in row.get("ppo", {}).get("epochs", []):
            for key in ("total", "policy_loss", "value_loss", "approx_kl", "entropy"):
                if key in epoch and not math.isfinite(float(epoch[key])):
                    raise RuntimeError("PPO出现非有限指标，禁止继续此配方")
        collect = row["collect"]
        if any(collect.get(k, 0) for k in ("illegal_actions", "truncated_games", "mirror_errors",
                                          "mirror_pool_unbalanced", "mirror_pool_species_unbalanced")):
            raise RuntimeError("训练采集质量门未通过")
        records.append(row)
    report = summarize(records)
    write(milestone / "training-report.json", report)
    return report


def run(args, execute=None):
    import torch
    from hsbg.az.train.personal import checked_library, training_source_digest, validate_config
    from workflows.paired_panel_delta import compare_personal
    def command(*argv):
        full = [sys.executable, *map(str, argv)]
        if execute:
            execute(full)
        else:
            subprocess.run(full, cwd=ROOT, check=True)
    out = Path(args.out).resolve()
    config = validate_config(read(args.config))
    train_library, eval_library = checked_library(args.train_library), checked_library(args.eval_library)
    if train_library["split"] != "train" or eval_library["split"] != "dev":
        raise ValueError("学习实验只使用Train与Dev，禁止查询封存Test")
    if not train_library.get("validation") or train_library["validation"]["accepted"] <= 0:
        raise ValueError("必须使用逐帧验证后的Train库")
    if config.get("snapshot-every") != 1 or config.get("iterations") != 0:
        raise ValueError("阶段预算要求snapshot-every=1、iterations=0")
    seeds = list(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 100000000 for s in seeds):
        raise ValueError("需要互异、>=100000000的训练seed")
    plan = {"schema": "hsrl.learning_experiment/v1", "config": config, "seeds": seeds,
            "train_library_digest": train_library["digest"], "eval_library_digest": eval_library["digest"],
            "initial_model_sha256": digest(args.checkpoint), "training_source_sha256": training_source_digest(),
            "runner_sha256": digest(__file__), "eval_games_per_seat": args.eval_games,
            "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "os": platform.system()},
            "eval_seed": 8000000, "device": args.device,
            "stop_rule": "rank_delta>=0.15 and CI_low>0 OR top4_delta<=-0.05 and CI_high<0"}
    with writer_lock(out):
        if (out / "plan.json").exists():
            if read(out / "plan.json") != plan:
                raise ValueError("实验配方/源码/模型/库/评测条件已改变，必须新开目录")
        else:
            write(out / "plan.json", plan)
        baseline = out / "baseline.json"
        def evaluate(model, path):
            command("-m", "hsbg.az.train.personal", "eval", "--checkpoint", model,
                    "--library", args.eval_library, "--device", args.device,
                    "--games", args.eval_games, "--out", path)
        if not baseline.exists():
            evaluate(args.checkpoint, baseline)
        if not read(baseline)["valid"] or (read(baseline).get("checkpoint") or {}).get("sha256") != plan["initial_model_sha256"]:
            raise ValueError("固定Best基线无效")
        stage = out / f"target-{args.target_decisions:09d}"
        rows = []
        for seed in seeds:
            arm = out / f"seed-{seed}"
            cfg = arm / "config.json"
            seed_config = {**config, "seed": seed}
            if cfg.exists() and read(cfg) != seed_config:
                raise ValueError("训练seed配置不一致")
            write(cfg, seed_config)
            train = arm / "train"
            milestone = stage / f"seed-{seed}"
            if not (milestone / "checkpoint.json").exists():
                if (arm / "stopped.json").exists():
                    raise RuntimeError("该训练seed已触发退化止损，禁止自动延长")
                def train_to(target, resume):
                    command("-m", "hsbg.az.train.personal", "train", "--config", cfg,
                        "--checkpoint", args.checkpoint, "--library", args.train_library,
                        "--device", args.device, "--out", train, "--stop-at-decisions", target,
                        *(["--resume"] if resume else []))
                if not (train / "latest.json").exists():
                    train_to(1, False)  # 第一个完整batch后保存，随后实际验证恢复。
                current = int(read(train / "latest.json")["cum_decisions"])
                if current > args.target_decisions:
                    # 只允许上次同阶段中断后的一轮超额；不能把更晚权重冒充早期checkpoint。
                    history = [json.loads(x) for x in (train / "history.jsonl").read_text().splitlines() if x.strip()]
                    last_n = int(history[-1]["collect"]["decisions"])
                    if current - last_n >= args.target_decisions:
                        raise ValueError("缺少早期阶段checkpoint，不能用更晚权重补写")
                if current < args.target_decisions:
                    train_to(args.target_decisions, True)
            record = freeze_checkpoint(train, milestone, args.target_decisions)
            training = (read(milestone / "training-report.json") if (milestone / "training-report.json").exists()
                        else training_report(train, milestone, int(record["version"])))
            if training["totals"]["decisions"] != record["actual_decisions"]:
                raise ValueError("阶段训练日志与权威checkpoint决策数不一致，保留现场")
            candidate = milestone / "evaluation.json"
            if not candidate.exists():
                evaluate(milestone / "model.pt", candidate)
            after = read(candidate)
            if (after.get("checkpoint") or {}).get("sha256") != record["sha256"]["model.pt"]:
                raise ValueError("阶段评测与保存权重不匹配")
            pair = compare_personal(read(baseline), after, n_boot=5000, seed=20261102)
            write(milestone / "paired.json", pair)
            stopped = eval_stop(pair)
            if stopped:
                write(arm / "stopped.json", {"target": args.target_decisions, "paired": pair})
            rows.append({"seed": seed, **record, "paired": pair, "training": training, "stopped": stopped})
            complete = len(rows) == len(seeds)
            recommendation = ("stop_recipe" if any(r["stopped"] for r in rows) else
                "incomplete" if not complete else
                "review_then_300k" if args.target_decisions < 300000 else
                "review_hour_stage" if len(rows) >= 2 and all(r["paired"]["delta"] < 0 for r in rows) else
                "hold_and_review_recipe")
            write(stage / "report.json", {"target": args.target_decisions, "arms": rows,
                "complete": complete, "promote_best": False, "recommendation": recommendation,
                "training_seeds": len(seeds), "note": "建议只供人工审阅；不自动晋级，不将评测局数乘以训练seed数冒充独立样本。"})
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/learning.json")
    parser.add_argument("--checkpoint", default="artifacts/models/s14_best_main_model.pt")
    parser.add_argument("--train-library", required=True)
    parser.add_argument("--eval-library", default="artifacts/libraries/bot-v1/dev")
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-decisions", type=int, default=100000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[100000000, 200000000])
    parser.add_argument("--eval-games", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 1 <= args.target_decisions <= 1000000 or not 1 <= args.eval_games <= 256:
        parser.error("阶段预算1..1000000、每座评测1..256局")
    run(args)


if __name__ == "__main__":
    main()
