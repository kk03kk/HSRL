"""冻结 6157 完整状态并验证新版 Bot 库上的 PPO 保存和恢复。"""
import os
import json
from pathlib import Path
import sys
import time

from hsbg.az.train.personal import (checked_library, require_calibrated_library,
                                   training_source_digest, validate_config)
from workflows.build_3702_bot_bank import DRIVER, VERSION
from workflows.migrate_checkpoint_3702 import migrate
from workflows.run_learning_experiment import digest, freeze_checkpoint, read, training_report, write, writer_lock
from workflows.run_overnight_learning import run_process

MODEL_SHA = "088a6f24217b4064b32f9c0d02f5612c2a0fc6bd1db1ed522e78e103512ae1dc"
COMPARISON_SHA = "c8e54393bde8ddd76f2c783523d98f3b6e5e628370bc4cf91e389575141f872c"
ACTION_PROFILE = {"max-actions-per-turn": 80, "move-guard": 1, "move-cap": 5, "move-repeat-limit": 2}


def action_args():
    return [value for key, setting in ACTION_PROFILE.items() for value in (f"--{key}", str(setting))]


def validate_action_protocol(protocol):
    if any(protocol.get(key.replace("-", "_")) != value for key, value in ACTION_PROFILE.items()):
        raise ValueError("37.0.2 评测动作协议不符")


def validate_eval_moves(result):
    validate_action_protocol(result["protocol"])
    for seat in result["per_seat"]:
        for row in seat["end_turn_economy"]:
            if row["action_counts"].get("move", 0) > ACTION_PROFILE["move-cap"]:
                raise ValueError("评测出现超限换位")


def recover(args, out, inherited, manifests, config, evidence, expected):
    """显式故障分叉，核对原实验来源后保留最后已提交的完整 Trainer。"""
    failed = Path(args.recover_from).resolve()
    if failed == out or (out / "train").exists():
        raise ValueError("故障恢复必须使用新的训练目录")
    plan = read(failed / "plan.json")
    old = read(failed / "train/provenance.json")
    latest = read(failed / "train/latest.json")
    if ((failed / "report.json").exists() and read(failed / "report.json").get("complete")):
        raise ValueError("本入口仅恢复未正常收尾的实验")
    if not (failed / "FAILED.json").exists():
        raise ValueError("需要保留的故障记录")
    accepted = dict(evidence["provenance"])
    accepted_source = accepted.pop("training_source_sha256")
    current = dict(expected)
    current.pop("training_source_sha256")
    if (not evidence["passed"] or accepted != current
            or plan["source_sha256"] != accepted_source or old["training_source_sha256"] != accepted_source
            or plan["config"] != config or old["config"] != config
            or old["fingerprint"] != manifests["train"]["fingerprint"]
            or plan["libraries"] != expected["libraries"]
            or old["library_digest"] != manifests["train"]["digest"]
            or plan["inherited"].get("original_sha256") != inherited["original_sha256"]
            or latest["version"] <= 6157 or latest["cum_decisions"] <= inherited["actual_decisions"]):
        raise ValueError("故障来源、工程证据、词表、库或训练配置不匹配")
    source = out / "recovered-state"
    restored = freeze_checkpoint(failed / "train", source, latest["cum_decisions"])
    from hsbg.az.train.personal import inspect_checkpoint
    inspect_checkpoint(source / "model.pt", "cpu")
    record = dict(failed_run=str(failed), failed_plan_sha256=digest(failed / "plan.json"),
                  old_training_source=accepted_source, new_training_source=expected["training_source_sha256"],
                  original_start=inherited, recovered=restored)
    write(out / "recovery.json", record)
    offset = restored["actual_decisions"] - inherited["actual_decisions"]
    args.monitor_first = args.monitor_every - offset % args.monitor_every
    return source, {**restored, "original_sha256": inherited["original_sha256"]}


def inputs(args):
    parent, bank, out = map(Path, (args.continue_from, args.bot_3702, args.out))
    parent, bank, out = parent.resolve(), bank.resolve(), out.resolve()
    previous = read(parent / "plan.json")
    report = read(parent / "report.json")
    latest = read(parent / "train/latest.json")
    if (not report["complete"] or latest["version"] != 6157 or latest["cum_decisions"] != 381659029
            or report["final_evaluations"]["latest"]["hard"]["after_sha256"] != MODEL_SHA
            or digest(args.reference_checkpoint) != MODEL_SHA):
        raise ValueError("续训起点不是已完成的 6157")
    bank_report = read(bank / "report.json")
    if not bank_report["complete"] or bank_report["smoke"]:
        raise ValueError("需要正式新版 Bot 库")
    paths = {"train": bank / "train", "hard": bank / "dev"}
    manifests = {key: checked_library(path) for key, path in paths.items()}
    for key, split in (("train", "train"), ("hard", "dev")):
        manifest = manifests[key]
        require_calibrated_library(manifest)
        if (manifest != bank_report["splits"][split] or manifest["driver"] != DRIVER
                or manifest["split"] != split or manifest["fingerprint"]["engine"]["game_version"] != VERSION
                or manifest["fingerprint"]["vocab_sha256"] !=
                    "5041e83d810e846b3cbc7d7ff056b84e301472d6ff021ffd1971a18a248e05fb"):
            raise ValueError("新版战团库或模型词表不符")
    original = out / "continued-source"
    frozen = freeze_checkpoint(parent / "train", original, latest["cum_decisions"])
    committed = read(Path(args.reference_checkpoint).with_name("checkpoint.json"))
    if frozen["version"] != 6157 or frozen["sha256"] != committed["sha256"] or frozen["sha256"]["model.pt"] != MODEL_SHA:
        raise ValueError("训练器与已冻结 6157 不一致")
    source = out / "initial-state"
    migration = migrate(original / "model.pt", source, original / "trainer.pt")
    frozen = {**frozen, "sha256": migration["outputs"], "original_sha256": migration["inputs"]}
    args.reference_checkpoint = str(source / "model.pt")
    config = validate_config({**previous["config"], **ACTION_PROFILE})
    if config.get("mirror-strict") != 1 or config.get("mirror-sample") != "random":
        raise ValueError("父训练配置不是严格随机镜像")
    if not args.engineering_only:
        evidence = read(Path(args.engineering_report))
        expected = provenance(frozen, manifests, config)
        if getattr(args, "recover_from", None):
            source, frozen = recover(args, out, frozen, manifests, config, evidence, expected)
        elif not evidence["passed"] or evidence["provenance"] != expected:
            raise ValueError("工程验证与当前训练输入不一致")
    run_process([sys.executable, "workflows/check_actor_tensor_path.py", "--checkpoint", source / "model.pt",
                 "--device", "auto", "--require-gpus", "2", "--out",
                 out / "preflight" / f"actor-{os.environ['SLURM_JOB_ID']}.json"],
                time.monotonic() + 300)
    return parent, source, frozen, DRIVER, paths, manifests, config, previous


def provenance(frozen, manifests, config):
    return dict(source=frozen["original_sha256"], libraries={k: v["digest"] for k, v in manifests.items()},
                fingerprint=manifests["train"]["fingerprint"], config=config,
                migration_sha256=digest(Path(__file__).with_name("migrate_checkpoint_3702.py")),
                training_source_sha256=training_source_digest())


def verify(args):
    out = Path(args.out).resolve()
    with writer_lock(out):
        args._writer_acquired = True
        if (out / "train").exists():
            raise FileExistsError("工程验证请使用新的输出目录")
        _, source, frozen, _, paths, manifests, config, _ = inputs(args)
        from hsbg.az.train.personal import inspect_checkpoint
        inspect_checkpoint(source / "model.pt", args.device)
        train = out / "train"
        write(out / "train.config.json", config)
        argv = [sys.executable, "-m", "hsbg.az.train.personal", "train", "--config",
                out / "train.config.json", "--checkpoint", source / "model.pt", "--library",
                paths["train"], "--out", train, "--device", args.device]
        deadline = time.monotonic() + 25 * 60
        run_process(argv + ["--run-iterations", "2", "--initial-trainer-state", source / "trainer.pt"], deadline)
        first = read(train / "latest.json")
        run_process(argv + ["--run-iterations", "1", "--resume"], deadline)
        latest = read(train / "latest.json")
        if (first["version"] != 6159 or latest["version"] != 6160
                or not frozen["actual_decisions"] < first["cum_decisions"] < latest["cum_decisions"]):
            raise RuntimeError("PPO 接续、保存或恢复进度不符")
        report = training_report(train, out, latest["version"])
        totals, strict = report["totals"], report["mirror_strict_counts"]
        with (train / "history.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("iter") is not None:
                    collect = row["collect"]
                    if (not collect.get("move_guard") or collect.get("move_cap") != ACTION_PROFILE["move-cap"]
                            or collect.get("move_repeat_limit") != ACTION_PROFILE["move-repeat-limit"]):
                        raise RuntimeError("采集未启用本轮换位规则")
        if (not totals["optimizer_steps"] or report["unexpected_mirror_rejections"]
                or totals["mirror_fallback"] or totals["mirror_missing"]
                or not totals["mirror_attempt"] == totals["mirror_applied"] == strict["eligible_live_seats"]
                or not strict.get("live_combats") or not strict.get("eligible_live_seats")):
            raise RuntimeError("工程验证未达到严格注入与 PPO 门槛")
        pair = None
        economy = None
        if args.comparison_checkpoint:
            from workflows.paired_panel_delta import compare_personal
            from workflows.report_firestone_learning import by_turn
            if digest(args.comparison_checkpoint) != COMPARISON_SHA:
                raise ValueError("对照模型不是已确认的 4159")
            comparison = out / "comparison-4159"
            migrate(Path(args.comparison_checkpoint), comparison)
            reports = []
            for model, version in ((comparison / "model.pt", 4159), (source / "model.pt", 6157)):
                target = out / f"dev-{version}.json"
                run_process([sys.executable, "-m", "hsbg.az.train.personal", "eval", "--checkpoint", model,
                             "--library", paths["hard"], "--out", target, "--device", args.device,
                             "--games", "8", "--seed", "11000000", "--mirror-strict", "--economy",
                             "--exposure", "1", *action_args()], deadline)
                result = read(target)
                validate_eval_moves(result)
                reports.append(result)
            pair = compare_personal(*reports, n_boot=2000, seed=20261103)
            economy = {version: by_turn(result)
                       for version, result in zip((4159, 6157), reports)}
        from workflows.report_3702_bank import compare
        strength = compare(Path(args.bot_3702))
        if strength["bank_digest"] != manifests["train"]["digest"]:
            raise ValueError("战力对比与训练库摘要不符")
        write(out / "bank-strength.json", strength)
        write(out / "engineering.json", dict(passed=True, provenance=provenance(frozen, manifests, config),
                                               first=first, latest=latest, totals=totals, paired_dev=pair,
                                               dev_economy=economy))
