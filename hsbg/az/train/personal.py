"""个人实验入口：复用现有 AZ PPO / 回合末快照，不另建训练引擎。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OPTIONS = {
    "games", "iterations", "minutes", "epochs", "minibatch", "grad-accum",
    "lr", "lr-min-ratio", "max-updates", "clip", "target-kl", "lam", "gamma",
    "entropy-coef", "value-coef", "pi-coef", "seq-coef", "max-grad-norm",
    "seed", "processes", "workers", "batch-size", "max-actions-per-turn",
    "anchor-beta", "anchor-iterations", "anchor-beta-tail", "reset-anchor", "snapshot-every",
    "keep-snapshots", "mirror-sample", "sims", "rc-threads", "local-batch-inference",
    "pipeline", "amp", "anchor-amp", "gpus", "gpu-index-space", "threads",
    "leaf-amp", "leaf-wait-ms", "ipc-mode", "records-dtype", "pipeline-guard",
    "leaf-fused", "mirror-strict", "mirror-seats", "mirror-p",
    # 奖励项（默认全 0 = 纯名次奖励，与原配方逐位一致）。
    # 2026-11-12：诊断显示模型卖随从只有真人的 0.22 倍、搜牌 0.67 倍，
    # 而奖励只认终局名次——"把板面做大"本身没有任何回报。这些键让"奖励盲区"
    # 可以被实验，而不必改代码。
    "terminal-strength-coef", "shaping-coef", "shaping-gamma", "shaping-w-str",
    "shaping-w-tier", "shaping-mode",
    # 专家教学项（默认全关 = 与历史配方逐位一致）。
    # 2026-11-12：data/firestone_bc/bc_samples.jsonl.gz 现在真的有内容了（282,717 条），
    # 所以这些开关第一次是"可用的"而不只是"存在"。注意 --bc-samples 的默认路径
    # 正是 data/firestone_bc/bc_samples.jsonl.gz，无需额外指定。
    "bc-coef", "bc-coef-ramp", "bc-mmr-min", "bc-mmr-boost", "bc-samples",
    "bc-batch", "bc-every", "bc-type-weight", "bc-card-weight", "bc-class-weight",
    "dpo-coef", "dpo-beta", "dpo-batch", "dpo-every",
    "move-guard", "move-cap", "move-repeat-limit", "forbid-cancel", "forbid-unfreeze",
    "pipeline-zero-step-fallback",
    # 周期评测（默认关闭；dev/test 的段后验证由 sbatch 钩子负责）
    "eval-every-decisions", "eval-games", "eval-seats", "eval-seed-base",
    "eval-async", "eval-device", "eval-stop-mean-rank", "eval-panel",
    # 强制升本课程（默认空计划 = 关闭；见 docs/FORCED_TIER.md）。
    # 注意：教师标签只由 ``forced-teacher-coef`` 驱动，不要用 ``teacher-bc-coef``，
    # 否则会同时打开「缩放教师」在线查询，两条标签会互相覆盖。
    "forced-tier-plan", "forced-tier-carry", "forced-tier-schedule",
    "forced-tier-prob", "forced-teacher-until", "forced-teacher-coef",
    # 人类战团对手（默认全关 = 与历史配方逐位一致）：``human-boards`` 给准完整库路径，
    # ``human-board-p``/``human-board-seats`` 给强度，``human-board-fidelity=full`` 才会带上
    # 手牌/金身/关键词位，``human-board-min-turn`` 把真人强度压力限制在后期。
    "human-boards", "human-board-p", "human-board-seats", "human-board-sample",
    "human-board-set-tier", "human-board-fidelity", "human-board-fast-inject",
    "human-board-min-turn",
    # 中后期入场课程（learner 席注入真人完整状态；见 docs/LATE_GAME_ENTRY.md）。
    # 默认 ``entry-p=0`` / ``entry-turn-min=0`` = 完全关闭。
    "entry-boards", "entry-p", "entry-turn-min", "entry-turn-max",
    "entry-sample", "entry-fidelity", "entry-fast-inject", "entry-set-health",
    # 镜像对手的回合上界（0=不限）：中后期课程用它把镜像关在入场回合之前。
    "mirror-max-turn",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha256(path):
    """数据源码按 LF 计算指纹，允许同一 Git 内容跨 Windows/Linux 传输。"""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def library_digest(root):
    files = sorted(Path(root).glob("snap_*.json.gz"))
    if not files:
        raise ValueError(f"没有快照：{root}")
    return hashlib.sha256(json.dumps(
        [(p.name, sha256(p)) for p in files], separators=(",", ":")).encode()).hexdigest()


#: 强制升本课程允许相对**快速基线配方**（``config/p50-7357-train.json``）改变的键。
#: 其它任何差异都算"意外偏离高效长训配方"：必须显式说明后才能提交（见
#: ``docs/nus/FORCED_TIER_DEPLOYMENT.md`` §配置漂移门禁）。``tools/nus_submit.py``
#: 里有一份仅供登录节点（无 numpy）使用的同名清单，``tests/test_config_parity.py``
#: 断言两份完全一致。
FORCED_ALLOWED_CONFIG_DIFF = frozenset({
    "entropy-coef", "anchor-beta", "anchor-beta-tail", "reset-anchor",
    "forced-tier-plan", "forced-tier-carry", "forced-tier-schedule",
    "forced-tier-prob", "forced-teacher-until", "forced-teacher-coef",
})


def config_drift(cfg, baseline, allowed=FORCED_ALLOWED_CONFIG_DIFF):
    """返回 ``cfg`` 相对 ``baseline`` 的**未授权**差异键（排序后）。

    只比较键值集合，不关心顺序；``allowed`` 之外的任何差异都会被返回，
    调用方据此 fail-fast，避免把"跑得慢/学得不对"的配方带进 12 小时长训。
    """
    keys = set(cfg) | set(baseline)
    return sorted(k for k in keys
                  if cfg.get(k) != baseline.get(k) and k not in allowed)


def training_source_digest():
    paths = sorted((ROOT / "hsbg/az/train").glob("*.py")) + [ROOT / "workflows/train_entity_sync.py"]
    h = hashlib.sha256()
    for p in paths:
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def validate_config(config):
    if set(config) - OPTIONS:
        raise ValueError(f"未知配置项：{sorted(set(config) - OPTIONS)}")
    for k in ("games", "minutes", "minibatch", "processes", "workers"):
        if k in config and float(config[k]) <= 0:
            raise ValueError(f"{k} 必须 > 0")
    if int(config.get("iterations", 0)) < 0:
        raise ValueError("iterations必须>=0；0表示按时间/决策预算停止")
    if config.get("sims", 0) != 0:
        raise ValueError("首版个人流程只使用现有 PPO 网络采样，sims 必须为 0")
    processes = int(config.get('processes', 1))
    if not 1 <= processes <= 32:
        raise ValueError('processes必须为1..32')
    if config.get('amp', 'off') not in ('off', 'bf16') or config.get('leaf-amp', 'off') not in ('off', 'bf16'):
        raise ValueError('只允许off/bf16；不启用records半精度存储')
    if config.get('records-dtype', 'fp32') != 'fp32':
        raise ValueError('个人训练records必须保持fp32')
    if config.get('pipeline', 0) not in (0, 1) or (config.get('pipeline', 0) and processes == 1):
        raise ValueError('pipeline=1需要多进程采集')
    if processes > 1:
        if config.get('gpu-index-space') != 'visible' or not str(config.get('gpus', '')).strip():
            raise ValueError('个人多进程必须使用Slurm可见GPU编号')
        if int(config.get('local-batch-inference', 0)):
            raise ValueError('本地batch适配器只用于单进程，pool已有批量推理')
    if config.get("local-batch-inference", 0) not in (0, 1):
        raise ValueError("local-batch-inference只接受0或1")
    if config.get("local-batch-inference", 0) and int(config.get("workers", 1)) != 1:
        raise ValueError("本地批量路径要求单worker")
    if int(config.get("seed", 100000000)) < 100000000:
        raise ValueError("个人训练 seed 从 100000000 起，避免与生成库和评测 seed 重叠")
    mandatory = {
        "max-actions-per-turn": 80,
        "move-guard": 1,
        "move-cap": 5,
        "move-repeat-limit": 2,
        "forbid-cancel": 1,
    }
    for key, expected in mandatory.items():
        actual = int(config.get(key, expected))
        if actual != expected:
            raise ValueError(f"训练必须遵守动作规则 {key}={expected}，当前 {actual}")
    if int(config.get("forbid-unfreeze", 0)) not in (0, 1):
        raise ValueError("forbid-unfreeze 只接受 0/1")
    return config


def fingerprint():
    import hsbg_engine as E
    from .common import engine_info
    if not E.build_is_current(ROOT / "engine"):
        raise ValueError("原生扩展不是从当前 Rust 源码构建，请重新安装 ./engine")
    # 有序词表单独取摘要：同维度并不保证同语义。
    ncols, raw = E.card_static()
    count = len(raw) // (4 * int(ncols))
    ids = [E.card_info(i)[0] for i in range(1, count)]
    return {
        "engine": engine_info(), "cards": count,
        "vocab_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "engine_cards_sha256": text_sha256(ROOT / "engine/crates/hsbg-engine/data/cards.json"),
        "schema_cards_sha256": text_sha256(ROOT / "engine/crates/hsbg-engine/data/cards.json"),
        "obs_floats": 3227, "obs_cards": 124, "actions": 245,
    }


def inspect_checkpoint(path, device):
    import torch
    from .model import load_checkpoint
    from .common import mask_from_bytes, obs_arrays
    from .ppo import PPOEvaluator
    import hsbg_engine as E
    model, meta = load_checkpoint(str(path), device=device)
    if meta.get("arch") in ("entity", "entity18m"):
        if any(k not in meta for k in ("d_model", "n_layers", "n_heads")):
            raise ValueError("entity checkpoint 缺少显式架构元数据")
        if len(model.trunk.layers) != int(meta["n_layers"]) or \
                model.trunk.layers[0].self_attn.num_heads != int(meta["n_heads"]):
            raise ValueError("实际注意力结构与元数据不一致")
        from .entity_model import load_card_tables
        tables = load_card_tables()
        if not np.array_equal(model.card_static.detach().cpu().numpy(), tables.static) or \
                not np.array_equal(model.card_text.detach().cpu().numpy(), tables.text):
            raise ValueError("checkpoint 内的卡牌静态/文本表与当前数据不同")
    # strict load 之外再验证真实观测前向与合法概率。
    g = E.Game(seed=123)
    f, c = obs_arrays(g.observe(0))
    mask = mask_from_bytes(g.legal_mask(0))
    ev = PPOEvaluator(model, device=device, norm_mean=meta.get("norm_mean"),
                      norm_std=meta.get("norm_std"))
    pi, value = ev(f, c, mask, 0)
    if pi.shape != (245,) or not np.isfinite(pi).all() or not np.isfinite(value):
        raise ValueError("真实观测前向输出异常")
    if not np.isclose(pi.sum(), 1) or np.any(pi[~mask] != 0):
        raise ValueError("合法动作概率异常")
    return {"sha256": sha256(path), "parameters": sum(p.numel() for p in model.parameters()),
            "meta": {k: v for k, v in meta.items() if not k.startswith("norm_")},
            "forward_ok": True, "torch": torch.__version__, "device": device}


def checked_library(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["digest"] != library_digest(root):
        raise ValueError("战团库摘要与 manifest 不一致")
    if manifest["fingerprint"] != fingerprint():
        raise ValueError("战团库与当前引擎/词表/数据不一致")
    return manifest


def run_command(argv):
    print("[run]", " ".join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), cwd=ROOT, check=True)


def require_calibrated_library(manifest):
    validation = manifest.get("validation", {})
    driver = manifest.get("driver")
    if driver == "firestone_visible_eot" and (
            manifest.get("build_number") != 255790 or manifest.get("fidelity") != "board_only_eot_approx"):
        raise ValueError("Firestone 严格镜像库版本或字段口径不符")
    if (driver not in {"human_curve_full_snapshots", "firestone_visible_eot", "bot_firestone_mean_3702", "bot_firestone_p50_3702", "model_eot_v1.1", "model_eot_league"}
            or not validation.get("accepted") or validation.get("quarantined") != 0):
        raise ValueError("严格镜像需要逐帧验证且无隔离帧的真人曲线库或 Firestone 回合末库")
    if driver in {"bot_firestone_mean_3702", "bot_firestone_p50_3702"} and (manifest.get("smoke") or manifest.get("last_turn", 0) < 15):
        raise ValueError("新版 Bot 库必须完成正式回合覆盖验收")


def build_library(args):
    from .mirror_snapshots import MirrorSnapshotPool
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("库输出目录必须为空，防止混合不同版本")
    splits = getattr(args, 'splits', 'train,dev,test').split(',')
    if not splits or len(set(splits)) != len(splits) or set(splits)-{'train', 'dev', 'test'}:
        raise ValueError('splits只能是互不重复的train/dev/test')
    seed_base = getattr(args, 'seed_base', 1000000)
    if seed_base < 0:
        raise ValueError('seed-base必须非负')
    for split in splits:
        seed = seed_base + ('train', 'dev', 'test').index(split) * 1000000
        target = out / split
        command = [sys.executable, "workflows/capture_trajectory.py", "--driver", args.driver,
                   "--games", args.games, "--seed", seed, "--from-turn", 1,
                   "--emit-snapshots", 1, "--snap-from-turn", 1,
                   "--out", out / "trajectories" / split, "--snap-dir", target,
                   "--device", args.device]
        if args.driver == "model":
            if not args.checkpoint:
                raise ValueError("model 驱动需要 --checkpoint")
            command += ["--checkpoint", Path(args.checkpoint).resolve()]
        run_command(command)
        pool = MirrorSnapshotPool(str(target), sample="random")
        if pool.frames == 0 or pool.bad_files:
            raise ValueError(f"空库或损坏库：{split}")
        write_json(target / "manifest.json", {
            "schema": "hsrl.library/v1", "split": split, "driver": args.driver,
            "seed_start": seed, "games": args.games, "level": 1,
            "source_checkpoint": sha256(args.checkpoint) if args.checkpoint else None,
            "fingerprint": fingerprint(), "digest": library_digest(target),
            "stats": pool.stats(), "pool_mode": "unknown", "missing": "engine_bot_fallback",
        })


def scale_library(args):
    from .mirror_snapshots import MirrorSnapshotPool
    source = checked_library(args.library)
    if not np.isfinite(args.factor) or args.factor < 1 or args.from_turn < 1:
        raise ValueError("factor >= 1 且 from-turn >= 1")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("输出目录必须为空")
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(args.library).glob("snap_*.json.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            record = json.load(f)
        for fr in record["frames"]:
            if int(fr["turn"]) >= args.from_turn:
                for m in fr["player"]["board"]:
                    m["atk"] = round(m["atk"] * args.factor)
                    m["hp"] = max(1, round(m["hp"] * args.factor))
                    if "max_hp" in m:
                        m["max_hp"] = max(m["hp"], round(m["max_hp"] * args.factor))
                fr["frame_sha256"] = hashlib.sha256(json.dumps(
                    fr["player"], ensure_ascii=False).encode()).hexdigest()
        with gzip.open(out / p.name, "wt", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
    write_json(out / "manifest.json", {**source, "level": args.level,
               "parent_digest": source["digest"], "factor": args.factor,
               "from_turn": args.from_turn, "digest": library_digest(out),
               "stats": MirrorSnapshotPool(str(out)).stats()})


def train(args):
    config = validate_config(json.loads(Path(args.config).read_text(encoding="utf-8")))
    stop_at = int(getattr(args, "stop_at_decisions", 0))
    if stop_at < 0:
        raise ValueError("stop-at-decisions必须>=0")
    run_minutes = float(getattr(args, "run_minutes", 0))
    run_iterations = int(getattr(args, 'run_iterations', 0))
    if run_iterations < 0:
        raise ValueError('run-iterations必须非负')
    if not np.isfinite(run_minutes) or run_minutes < 0:
        raise ValueError("run-minutes必须是有限非负数")
    library = checked_library(args.library)
    if library["split"] != "train":
        raise ValueError("训练只允许 train 库")
    if config.get("mirror-strict", 0):
        require_calibrated_library(library)
    out = Path(args.out).resolve()
    fp = fingerprint()
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    resume = []
    initial_trainer = getattr(args, "initial_trainer_state", "")
    fork = None
    transition = None
    if initial_trainer:
        if args.resume:
            raise ValueError("initial-trainer-state只用于新目录分叉；后续使用resume")
        import torch
        from .model import load_checkpoint
        state = torch.load(initial_trainer, map_location="cpu", weights_only=False)
        if checkpoint is None:
            raise ValueError("完整状态分叉必须同时给checkpoint")
        net, _ = load_checkpoint(str(checkpoint), device="cpu")
        if state["model"].keys() != net.state_dict().keys() or any(
                not torch.equal(v.cpu(), state["model"][k].cpu()) for k, v in net.state_dict().items()):
            raise ValueError("分叉model与trainer模型不一致")
        fork = {"sha256": sha256(initial_trainer), "path": str(Path(initial_trainer).resolve()),
                "cum_decisions": int(state.get("extra", {}).get("cum_decisions", 0)),
                "updates": int(state["updates"]),
                "mode": "explicit_full_state_fork_new_source_or_library; not exact old experiment resume"}
        del state, net
        resume = ["--trainer-state", str(Path(initial_trainer).resolve())]
    if args.resume:
        previous = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
        if previous["fingerprint"] != fp:
            raise ValueError("恢复时引擎/数据/库发生变化；请另开实验")
        old_library = previous['library_digest']
        requested_transition = getattr(args, 'library_transition_from', '')
        if old_library != library['digest']:
            if requested_transition != old_library:
                raise ValueError('课程换库必须显式给出上一库摘要；禁止隐式换库')
            transition = dict(from_digest=old_library, to_digest=library['digest'])
        elif requested_transition:
            raise ValueError('课程换库起止摘要相同；不要伪造换库记录')
        if previous["config"] != config:
            raise ValueError("恢复配置不同；请复制原配置，另开新阶段时只加载权重")
        if previous.get("training_source_sha256") != training_source_digest():
            raise ValueError("训练代码已变更；请另开实验，不能伪装为精确续训")
        latest = json.loads((out / "latest.json").read_text(encoding="utf-8"))
        snap = out / "snapshots" / f"step_{int(latest['version']):06d}"
        from .runlog import recover_committed_record
        recover_committed_record(out / "history.jsonl", snap, int(latest["version"]))
        checkpoint = snap / "model.pt"
        resume = ["--trainer-state", str(snap / "trainer.pt")]
        if transition is not None:
            transition['version'] = latest['version']
            transition['cum_decisions'] = latest['cum_decisions']
    elif out.exists() and any(out.iterdir()):
        raise ValueError("新实验输出目录必须为空；恢复请使用 --resume")
    if checkpoint is None or not checkpoint.is_file():
        raise ValueError("需要有效的 best checkpoint")
    audit = inspect_checkpoint(checkpoint, args.device)
    out.mkdir(parents=True, exist_ok=True)
    revision = "source:" + training_source_digest()
    if not args.resume:
        write_json(out / "provenance.json", {"schema": "hsrl.run/v1", "revision": revision,
                   "fingerprint": fp, "library_digest": library["digest"],
                   "config": config, "initial_model": audit,
                   "initial_trainer": fork,
                   "training_source_sha256": training_source_digest()})
    elif transition is not None:
        previous['library_digest'] = library['digest']
        previous.setdefault('library_transitions', []).append(transition)
        write_json(out/'provenance.json', previous)
    command = [sys.executable, "workflows/train_entity_sync.py", "--checkpoint", str(checkpoint),
               "--device", args.device, "--out", str(out), "--opponent", "bot",
               "--behavior", "net", "--pipeline", "0", "--rust-collector", "0", "--lockstep", "1",
               "--mirror-snaps", str(Path(args.library).resolve()),
               "--mirror-seats", "7", "--mirror-p", "1", "--human-board-p", "0"]
    command += resume
    for k, v in config.items():
        command += ["--" + k, str(v)]
    # 兼容旧配置：未写 eval 键时保持旧的 bot/0 行为。
    if "eval-panel" not in config:
        command += ["--eval-panel", "bot"]
    if "eval-every-decisions" not in config:
        command += ["--eval-every-decisions", "0"]
    # 停止预算是本次调用的运行控制，不改变学习配方、optimizer或恢复契约。
    if stop_at:
        command += ["--stop-at-decisions", str(stop_at)]
    if run_minutes:
        # 运行时上限不改变训练配方/优化器，只能缩短本次子进程。
        command += ["--minutes", str(min(run_minutes, float(config.get("minutes", run_minutes))))]
    if run_iterations:
        configured=int(config.get('iterations',0))
        command += ['--iterations',str(min(run_iterations,configured) if configured else run_iterations)]
    run_command(command)
    for required in ("latest.json", "model.pt", "trainer.pt", "history.jsonl"):
        if not (out / required).is_file():
            raise RuntimeError(f"训练未完整产出：{required}")


def evaluate(args):
    import torch
    from .evaluate import EvalConfig, evaluate_policy
    from .mirror_snapshots import MirrorSnapshotPool
    from .policies import make_policy
    from .model import load_checkpoint
    from .ppo import PPOEvaluator, PPOGreedyPolicy
    library = checked_library(args.library) if args.library else None
    if library and library["split"] not in ("dev", "test"):
        raise ValueError("排行榜只允许 dev/test 库")
    strict = bool(getattr(args, "mirror_strict", False))
    if strict:
        require_calibrated_library(library or {})
    pool = MirrorSnapshotPool(args.library, sample="random") if library else None
    protocol = {"schema": "hsrl.eval/v1", "fingerprint": fingerprint(),
                "training_source_sha256": training_source_digest(),
                "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                            "device_type": torch.device(args.device).type, "os": platform.system()},
                "library_digest": library["digest"] if library else "live_bots",
                "split": library["split"] if library else "bot",
                "seed": args.seed, "games_per_seat": args.games, "seats": list(range(8)),
                  "decode": "greedy", "max_actions_per_turn": int(getattr(args,'max_actions_per_turn',80)), "max_steps": 50000,
                  "move_guard":int(getattr(args,'move_guard',1)),"move_cap":int(getattr(args,'move_cap',5)),
                  "forbid_cancel":1,
                  "move_repeat_limit":int(getattr(args,'move_repeat_limit',2))}
    if strict:
        protocol["mirror_strict"] = True
    if getattr(args, "economy", False):
        protocol["end_turn_economy"] = "before_end_turn/v3_board_gold"
    # 强制升本探针（空 = 关闭）。写进 protocol 使配对门禁把「规则内/规则外」区分开。
    _ft_plan = str(getattr(args, "forced_tier_plan", "") or "").strip()
    if _ft_plan:
        protocol["forced_tier_plan"] = _ft_plan
        protocol["forced_tier_carry"] = int(getattr(args, "forced_tier_carry", 1))
    audit = None
    if args.policy == "net":
        audit = inspect_checkpoint(args.checkpoint, args.device)
        model, meta = load_checkpoint(args.checkpoint, device=args.device)
        policy = PPOGreedyPolicy(PPOEvaluator(model, device=args.device,
                    norm_mean=meta.get("norm_mean"), norm_std=meta.get("norm_std")),
                    max_actions_per_turn=protocol['max_actions_per_turn'],move_guard=bool(protocol['move_guard']),
                    move_cap=protocol['move_cap'],move_repeat_limit=protocol['move_repeat_limit'])
    else:
        policy = make_policy(args.policy, seed=args.seed)
    rows = []
    for seat in range(8):
        rows.append(evaluate_policy(policy, EvalConfig(
            games=args.games, seed=args.seed, policy_seat=seat, mirror_pool=pool,
            collect_exposure=bool(getattr(args, 'exposure', 0)),
            collect_economy=bool(getattr(args, 'economy', False)), mirror_strict=strict,
            forced_tier_plan=_ft_plan,
            forced_tier_carry=int(getattr(args, "forced_tier_carry", 1)))))
    ranks = np.asarray([r for row in rows for r in row["rank_sequence"]])
    valid = bool(np.all((ranks >= 1) & (ranks <= 8)) and all(
        r["illegal_actions"] == r["truncated_combats"] == r["truncated_games"] == 0 for r in rows))
    if pool is not None:
        valid = valid and all(r["mirror"]["mirror_applied"] > 0 and
            r["mirror"]["mirror_errors"] == r["mirror"]["mirror_pool_unbalanced"] ==
            r["mirror"]["mirror_pool_species_unbalanced"] == 0 for r in rows)
    # 种子为重采样单位，八个座位作为同一组，避免把相关样本当独立样本。
    matrix = ranks.reshape(8, args.games)
    rng = np.random.default_rng(714)
    means = np.array([matrix[:, rng.integers(0, args.games, args.games)].mean()
                      for _ in range(2000)])
    result = {"valid": valid, "protocol": protocol, "policy": args.policy,
              "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
              "checkpoint": audit, "mean_rank": float(ranks.mean()) if valid else None,
              "mean_rank_ci95": np.quantile(means, [.025, .975]).tolist() if valid else None,
              "top4_rate": float((ranks <= 4).mean()) if valid else None,
              "win_rate": float((ranks == 1).mean()) if valid else None, "per_seat": rows}
    write_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_seat", "protocol")},
                     ensure_ascii=False, indent=2))
    if not valid:
        raise RuntimeError("评测出现非法动作/截断，禁止入榜")


def leaderboard(args):
    results = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.results]
    if any(not r["valid"] for r in results):
        raise ValueError("含无效评测，禁止入榜")
    if len({r["protocol_sha256"] for r in results}) != 1:
        raise ValueError("不同协议的评测不能放在同一排行榜")
    rows = sorted([{"model": (r["checkpoint"]["sha256"] if r["checkpoint"] else r["policy"]),
                    "mean_rank": r["mean_rank"], "mean_rank_ci95": r["mean_rank_ci95"],
                    "top4_rate": r["top4_rate"], "win_rate": r["win_rate"]} for r in results],
                  key=lambda r: r["mean_rank"])
    write_json(args.out, {"protocol_sha256": results[0]["protocol_sha256"], "ranking": rows})
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def collect_probe(args):
    import torch
    from .collect_diagnostics import probe_config, require_valid_mirror_collect
    from .mirror_snapshots import MirrorSnapshotPool
    from .model import load_checkpoint
    from .parallel_collect import collect_ppo_lockstep
    from .ppo import PPOEvaluator
    config = validate_config(json.loads(Path(args.config).read_text(encoding="utf-8")))
    library = checked_library(args.library)
    if library["split"] != "train":
        raise ValueError("诊断采集只使用 train 库")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("诊断输出目录必须为空，避免覆盖失败证据")
    audit = inspect_checkpoint(args.checkpoint, args.device)
    model, meta = load_checkpoint(args.checkpoint, device=args.device)
    cfg = probe_config(config, iteration=args.iteration)
    evaluator_cls = PPOEvaluator
    if int(config.get("local-batch-inference", 0)):
        from .local_batch_eval import LocalBatchPPOEvaluator
        evaluator_cls = LocalBatchPPOEvaluator
    evaluator = evaluator_cls(model, device=args.device, norm_mean=meta.get("norm_mean"),
                             norm_std=meta.get("norm_std"))
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(args.device))
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    started = time.perf_counter()
    records, stats = collect_ppo_lockstep(model, cfg, eval_fn=evaluator,
        device=args.device, seed=cfg.seed, games=cfg.games, gamemode="bot",
        norm_mean=meta.get("norm_mean"), norm_std=meta.get("norm_std"),
        mirror_pool=MirrorSnapshotPool(args.library, sample=cfg.mirror_sample),
        mirror_seats_count=7, mirror_p=1.0)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(args.device))
    elapsed = time.perf_counter() - started
    hardware = {"machine": platform.machine(), "torch_threads": torch.get_num_threads()}
    if args.device.startswith("cuda"):
        props = torch.cuda.get_device_properties(torch.device(args.device))
        hardware.update({"gpu": props.name, "sm": props.multi_processor_count,
                         "total_bytes": props.total_memory,
                         "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(torch.device(args.device))})
    write_json(out / "probe.json", {"schema": "hsrl.collect_probe/v1",
        "checkpoint": audit, "iteration": args.iteration,
        "config": {**vars(cfg), "local_batch_inference": int(config.get("local-batch-inference", 0))},
        "fingerprint": fingerprint(), "library_digest": library["digest"],
        "collect": stats, "records": len(records), "ppo_updates": 0,
        "inference": evaluator.inference_stats() if hasattr(evaluator, "inference_stats") else {"mode": "serial"},
        "collect_seconds": elapsed, "decisions_per_second": stats["decisions"] / max(elapsed, 1e-9),
        "hardware": hardware})
    print(json.dumps({k: v for k, v in stats.items() if k != "games_meta"},
                     ensure_ascii=False, indent=2), flush=True)
    require_valid_mirror_collect(stats, str(out), iteration=args.iteration, config=vars(cfg))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--checkpoint", required=True)
    pre.add_argument("--device", default="cpu")
    pre.add_argument("--out", required=True)
    build = sub.add_parser("build-library")
    build.add_argument("--games", type=int, default=32)
    build.add_argument("--out", required=True)
    build.add_argument("--driver", choices=["bot", "model"], default="bot")
    build.add_argument("--checkpoint", default="")
    build.add_argument("--device", default="cpu")
    build.add_argument("--seed-base", type=int, default=1000000)
    build.add_argument("--splits", default="train,dev,test", help="新库可只生成train或dev，避免不必要的Test采集")
    scale = sub.add_parser("scale-library")
    scale.add_argument("--library", required=True)
    scale.add_argument("--out", required=True)
    scale.add_argument("--factor", type=float, required=True)
    scale.add_argument("--from-turn", type=int, default=8)
    scale.add_argument("--level", type=int, required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--config", required=True)
    tr.add_argument("--checkpoint", default="")
    tr.add_argument('--run-iterations',type=int,default=0,help='仅限制本次调用轮数，不改变持久配置，用于精确补齐有界验证')
    tr.add_argument("--initial-trainer-state", default="", help="显式新实验完整状态分叉；旧目录不能以此绕过resume检查")
    tr.add_argument("--library", required=True)
    tr.add_argument("--out", required=True)
    tr.add_argument("--device", default="cuda")
    tr.add_argument("--resume", action="store_true")
    tr.add_argument('--library-transition-from', default='', help='固定课程的显式上一库SHA；仍恢复完整Trainer，不能改变源码/配置/引擎')
    tr.add_argument("--run-minutes", type=float, default=0, help="只缩短本次运行时限，不改严格恢复配置")
    tr.add_argument("--stop-at-decisions", type=int, default=0,
                    help="本次调用的累计决策上限；整轮边界停止，可在下一阶段提高预算")
    probe = sub.add_parser("collect-probe", help="只采集一个批次并保存诊断，不进行 PPO 更新")
    probe.add_argument("--config", required=True)
    probe.add_argument("--checkpoint", required=True)
    probe.add_argument("--library", required=True)
    probe.add_argument("--iteration", type=int, required=True)
    probe.add_argument("--device", default="cuda")
    probe.add_argument("--out", required=True)
    ev = sub.add_parser("eval")
    ev.add_argument("--checkpoint", default="")
    ev.add_argument("--policy", choices=["net", "random", "bot"], default="net")
    ev.add_argument("--library", default="")
    ev.add_argument("--games", type=int, default=64)
    ev.add_argument("--seed", type=int, default=8000000)
    ev.add_argument("--device", default="cuda")
    ev.add_argument("--mirror-strict", action="store_true", help="仅使用校准库对手并验证每场交手")
    ev.add_argument("--economy", action="store_true", help="记录学习席每回合EndTurn前的经营状态")
    ev.add_argument("--forced-tier-plan", default="",
                    help="评测期强制升本计划（如 2:2,5:3,6:4,8:5）；空=模型完全自主（正式"
                         "Dev/Test 必须为空，否则棋力不可比）")
    ev.add_argument("--forced-tier-carry", type=int, default=1,
                    help="强制升本的顺延回合数（默认 1）")
    ev.add_argument("--exposure", type=int, choices=[0, 1], default=0)
    ev.add_argument('--max-actions-per-turn',type=int,default=80)
    ev.add_argument('--move-guard',type=int,choices=[0,1],default=1)
    ev.add_argument('--move-cap',type=int,default=5)
    ev.add_argument('--move-repeat-limit',type=int,default=2)
    ev.add_argument("--out", required=True)
    board = sub.add_parser("leaderboard")
    board.add_argument("results", nargs="+")
    board.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if getattr(args, "games", 1) <= 0 or getattr(args, "games", 1) >= 1000000:
        ap.error("games 必须位于 1..999999，保证数据生成 seed 区间不相交")
    import torch
    torch.set_num_threads(int(os.environ.get("HSRL_TORCH_THREADS", "2")))
    if args.command == "preflight":
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("当前计算节点 CUDA 不可用")
        result = {"platform": platform.platform(), "machine": platform.machine(),
                  "python": sys.version, "fingerprint": fingerprint(),
                  "checkpoint": inspect_checkpoint(args.checkpoint, args.device)}
        if args.device.startswith("cuda"):
            prop = torch.cuda.get_device_properties(torch.device(args.device))
            # 实际进行反向和优化器更新，仅更新临时参数。
            x = torch.nn.Parameter(torch.randn(32, 32, device=args.device))
            opt = torch.optim.Adam([x])
            x.square().mean().backward()
            opt.step()
            torch.cuda.synchronize()
            result["gpu"] = {"name": prop.name, "sm": prop.multi_processor_count,
                             "bytes": prop.total_memory, "optimizer_ok": True}
        write_json(args.out, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "build-library":
        build_library(args)
    elif args.command == "scale-library":
        scale_library(args)
    elif args.command == "train":
        train(args)
    elif args.command == "leaderboard":
        leaderboard(args)
    elif args.command == "collect-probe":
        collect_probe(args)
    else:
        evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
