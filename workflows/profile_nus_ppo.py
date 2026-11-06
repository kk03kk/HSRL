"""从冻结的长训状态跑四轮短探针，只剖析第三轮learner更新。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows.run_learning_experiment import digest, read, training_report, write
from workflows.report_personal_run import summarize
from workflows.run_overnight_learning import run_process


def profile_update(trainer, batch, path, iteration, lag):
    import torch

    activities = [torch.profiler.ProfilerActivity.CPU]
    if trainer.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as profiler:
        with torch.profiler.record_function("hsrl.learner_update"):
            stats = trainer.update(batch, profile=True)

    events = profiler.key_averages()

    def entry(event):
        return dict(
            op=event.key, count=event.count, cpu_total_us=event.cpu_time_total,
            cpu_self_us=event.self_cpu_time_total, device_total_us=event.device_time_total,
            device_self_us=event.self_device_time_total,
        )

    cpu = sorted(events, key=lambda event: event.self_cpu_time_total, reverse=True)
    device = sorted(events, key=lambda event: event.self_device_time_total, reverse=True)
    write(Path(path), dict(
        iteration=iteration, lag=lag, samples=len(batch),
        processed_samples=stats["processed_samples"], optimizer_steps=stats["updates"],
        scopes=[entry(event) for event in events if event.key.startswith("hsrl.")],
        top_cpu=[entry(event) for event in cpu[:25]],
        top_device=[entry(event) for event in device[:25] if event.self_device_time_total > 0],
        copies_and_sync=[entry(event) for event in events
                         if "memcpy" in event.key.lower() or "synchronize" in event.key.lower()],
        note="仅诊断。CPU和GPU事件可能重叠，父子事件不可相加，带profiler的轮次不用于吞吐。",
    ))
    return stats


def run_resume_ab(out, config, model, trainer, library, deadline):
    """两臂用同一完整状态和同一GPU分配，剖析关闭，保留独立输出。"""
    run_process([sys.executable, "-m", "pytest", "-q", "tests/test_ppo_resume_device.py"],
                min(deadline, time.monotonic() + 180))
    config_path = out / "ab.config.json"
    write(config_path, {**read(config), "mirror-strict": 1})
    results = {}
    # 1072511 的旧加载臂在10分钟内只完成9轮，给两臂留出同等余量。
    rounds, warmup = 6, 2
    os.environ.pop("HSRL_PPO_PROFILE_OUT", None)
    for mode, device_type in (("legacy", "cuda"), ("cpu", "cpu")):
        dest = out / mode
        os.environ["HSRL_PPO_BENCHMARK_LOAD"] = mode
        started = time.monotonic()
        run_process([
            sys.executable, "-m", "hsbg.az.train.personal", "train",
            "--config", config_path, "--checkpoint", model, "--initial-trainer-state", trainer,
            "--library", library, "--out", dest, "--device", "cuda:0",
            "--run-iterations", str(rounds), "--run-minutes", "10",
        ], deadline)
        wall = time.monotonic() - started
        devices = read(dest / "adam-step-devices.json")
        if set(devices) != {device_type} or devices[device_type] <= 0:
            raise RuntimeError(f"{mode} Adam计数器放置不符合试验假设：{devices}")
        latest = read(dest / "latest.json")
        quality = training_report(dest, out / "quality" / mode, int(latest["version"]))
        counts = quality.get("mirror_strict_counts", {})
        if (quality["unexpected_mirror_rejections"] or quality["totals"]["mirror_fallback"]
                or not counts.get("live_combats")):
            raise RuntimeError(f"{mode} 的采集未通过严格战团门")
        rows = []
        with (dest / "history.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("iter") is not None:
                    rows.append(row)
        if len(rows) != rounds:
            raise RuntimeError(f"{mode} 完成 {len(rows)} 轮，预期 {rounds}")
        steady = summarize(rows[warmup:])
        totals = steady["totals"]
        if steady["missing_timeline_iterations"] or not totals["processed_samples"]:
            raise RuntimeError(f"{mode} 缺少有效计时或优化样本")
        results[mode] = dict(
            adam_steps_by_device=devices, wall_s=wall,
            wall_effective_samples_per_s=quality["totals"]["processed_samples"] / wall,
            steady_effective_samples_per_s=totals["processed_samples"] / totals["iteration_s"],
            ppo_effective_samples_per_s=totals["processed_samples"] / totals["ppo_s"],
            decisions=totals["decisions"], processed_samples=totals["processed_samples"],
            processed_fraction=totals["processed_samples"] / totals["decisions"],
            audited_combat_count=counts.get("live_combats", 0) + counts.get("ghost_combats", 0),
            optimizer_steps=totals["optimizer_steps"],
            kl_early_stops=totals["kl_early_stop_iterations"],
        )
        write(out / "resume-ab-progress.json", results)
    gains = {key: results["cpu"][key] / results["legacy"][key] - 1
             for key in ("wall_effective_samples_per_s", "steady_effective_samples_per_s",
                         "ppo_effective_samples_per_s")}
    write(out / "resume-ab.json", dict(
        source_model_sha256=digest(model), source_trainer_sha256=digest(trainer),
        config_sha256=digest(config_path), library_digest=read(library / "manifest.json")["digest"],
        rounds_per_arm=rounds, steady_warmup_excluded=warmup, results=results, relative_gain=gains,
        note="单次同分配A/B。整段墙钟含启动和保存；稳态迭代剔除前两轮；两臂均为严格校准对手。",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("必须在Slurm计算节点运行")
    out = args.out.resolve()
    if (out / "train").exists():
        raise FileExistsError("探针训练目录已存在，请使用新作业目录")
    parent = ROOT / "artifacts/runs/nus-human-long-002"
    source = parent / "last-committed/step_004159"
    model, trainer = source / "model.pt", source / "trainer.pt"
    record = read(source / "checkpoint.json")
    config = parent / "train.config.json"
    library = ROOT / "artifacts/runs/nus-curriculum-001/libraries/train"
    if int(record["version"]) != 4159 or any(
        digest(file) != record["sha256"][file.name] for file in (model, trainer)
    ):
        raise ValueError("探针源checkpoint摘要或版本不符")

    deadline = time.monotonic() + 22 * 60
    run_process([
        sys.executable, "workflows/check_actor_tensor_path.py", "--checkpoint", model,
        "--device", "auto", "--require-gpus", "2", "--out", out / "actor-parity.json",
    ], time.monotonic() + 300)
    if os.environ.get("HSRL_PPO_RESUME_AB") == "1":
        run_resume_ab(out, config, model, trainer, library, deadline)
        return
    profile = out / "learner-profile.json"
    os.environ["HSRL_PPO_PROFILE_OUT"] = str(profile)
    run_process([
        sys.executable, "-m", "hsbg.az.train.personal", "train",
        "--config", config, "--checkpoint", model, "--initial-trainer-state", trainer,
        "--library", library, "--out", out / "train", "--device", "cuda:0",
        "--run-iterations", "4", "--run-minutes", "15",
    ], deadline)
    result = read(profile)
    if not result["top_device"]:
        raise RuntimeError("CUDA profiler没有捕获设备事件，请检查作业日志")
    print(f"性能探针报告 {profile}", flush=True)


if __name__ == "__main__":
    main()
