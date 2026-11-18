#!/usr/bin/env python3
"""Entity PPO 训练入口（持久 trainer + 进程级 actor pool + 可选 double-buffer 流水线）。

三条执行模式：

1. **串行**（默认）：collect → update → 快照/日志，语义最简单；
2. **流水线**（`--pipeline 1`，需要 `--processes > 1`）：`collect(B_{k+1})` 与 `update(B_k)`
   并行；actor 用**版本化权重文件**（任务开始时按 `weights_version` 懒加载），因此一批数据
   始终由同一个版本采集 → 滞后恒为 1，可断言；
3. 多线程采集（`--workers > 1 --processes 1`）：老路径，保留兼容。

评测：`--eval-every-decisions N` 到点做评测；`--eval-async 1` 时评测交给独立子进程
（默认 `--eval-device cuda:4`），主循环不等待，结果由轻量线程追加到
`eval/history.json`；`--eval-async 0` 保持旧的同步阻塞行为。`--eval-panel {bot,human,both}`
决定跑哪个面板（默认 both；human=人类战团对手面板，新主指标，见 `docs/EVAL_PANELS.md`），
两个 panel 各起一个子进程、串行排队（并发上限仍为 1）。

日志：`history.jsonl`（逐 iteration 追加，O(1)；不再全量重写 `history.json`），
`summary.json` 只存聚合信息；`--trainer-state` 可精确续训。

超参照 `docs/MULTISTAGE_TRAINING.md` §12（AutoTavern 论文 Table 7 对齐）：
sims=0 / clip 0.2 / target KL 0.05 / λ=0.995 / entropy 0.002 / grad-clip 1.0 /
lr 6e-6→5e-6 / 有效 minibatch = minibatch × grad_accum（8192）/ KL anchor β=0.1 前 100 iter。
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hsbg.az.train.common import engine_available, engine_info, to_jsonable  # noqa: E402
from hsbg.az.train.mirror_snapshots import parse_turn_bands, ramp_p  # noqa: E402
from hsbg.az.train.checkpoint_io import publish_checkpoint  # noqa: E402
from hsbg.az.train.reward_shaping import (  # noqa: E402
    SPELL_BUY_ACTION,
    parse_tier_bonus,
    shaping_gamma_warning,
)
from hsbg.az.train.tier_target import parse_tier_target_table  # noqa: E402
from hsbg.az.train.forced_tier import (  # noqa: E402
    parse_forced_tier_plan,
    parse_prob_schedule,
    plan_spec_of,
    prob_at,
)
from hsbg.az.train.health_gate import (  # noqa: E402
    HealthLimits,
    format_failures,
    hard_collect_failures,
    hard_update_failures,
    soft_warnings,
)
from hsbg.az.train.lobby_filter import parse_race_list  # noqa: E402
from hsbg.az.train.model import (  # noqa: E402
    attach_bc_card_head,
    attach_move_head_split,
    attach_process_head,
    attach_strength_head,
    bc_card_head_dim,
    build_net,
    load_checkpoint,
    save_checkpoint,
)
from hsbg.az.train.curriculum import (  # noqa: E402
    anneal_progress,
    curriculum_schedule,
    gold_mult_at,
)
from hsbg.az.train.ppo import (  # noqa: E402
    BoardRampController,
    CollectConfig,
    PPOConfig,
    PPOEvaluator,
    board_ramp_history,
    collect_ppo_batch,
    forward_model,
    masked_log_softmax_torch,
    parse_decisions,
    ramp_progress_metrics,
    split_model_outputs,
    torch_available,
    value_from_rank_logits_torch,
)
from hsbg.az.train.ppo_trainer import PPOTrainer  # noqa: E402
from hsbg.az.train.batch_eval import BatchLeafServer, BatchPPOEvaluator  # noqa: E402
from hsbg.az.train.parallel_collect import collect_ppo_batch_parallel  # noqa: E402
from hsbg.az.train.process_pool import ProcessActorPool  # noqa: E402
from hsbg.az.train.runlog import (append_record, read_run, slim_collect,  # noqa: E402
                                  write_header, write_summary)
from hsbg.az.train.seq_aux import SeqCoefSchedule  # noqa: E402
from hsbg.az.train.bc_aux import (  # noqa: E402
    bc_pool_summary,
    load_bc_samples,
    parse_bc_class_weights,
)
from hsbg.az.train.dpo_upgrade import (  # noqa: E402
    UpgradeSamples,
    dpo_pair_loss,
    load_upgrade_samples,
)
from workflows.async_eval import (AsyncEvalRunner, build_c0_cmd,  # noqa: E402
                                build_human_cmd, panels_from_spec,
                                parse_panel_result, write_json_atomic)


#: 期望梯度占比的参考量级：PPO total loss（value 项主导）。本 run 无历史时用 s9d 实测均值。
DPO_REF_TOTAL_LOSS = 0.77


def _ppo_total_ref(out_dir: str) -> tuple:
    """本 run 最近一次 PPO ``total`` loss（无历史时退回 s9d 量级常数）；返回 (值, 来源)。"""
    path = os.path.join(str(out_dir), "history.jsonl")
    try:
        last = None
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = line
        if last:
            rec = json.loads(last)
            for e in reversed((rec.get("ppo") or {}).get("epochs") or []):
                if "total" in e:
                    return float(e["total"]), "本 run history"
    except Exception:  # noqa: BLE001 - 只是启动打印的参考值
        pass
    return float(DPO_REF_TOTAL_LOSS), "s9d 量级常数"


def _dpo_startup_probe(model, pool: "UpgradeSamples", *, beta: float,
                       device: str, n: int = 512) -> Dict[str, float]:
    """启动诊断：当前策略在 upgrade 池上的 DPO 初值/接受率（no_grad，不改权重）。"""
    import torch

    sel = np.arange(min(int(n), len(pool)), dtype=np.int64)
    sub = pool.take(sel)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            loss, rate = dpo_pair_loss(model, sub, float(beta), device=str(device))
        return {"n": float(len(sub)), "loss": float(loss),
                "accept_rate": float(rate)}
    finally:
        model.train(was_training)


def _dpo_grad_ref(path: str = "runs/dpo_grad_probe.json"):
    """读离线梯度占比探针结果（``workflows/dpo_grad_probe.py``）；没有则 None。"""
    try:
        with open(str(path), encoding="utf-8") as f:
            data = json.load(f)
        share = float(data["share_per_coef"])       # coef=1 时 ‖g_dpo‖/‖g_ppo‖
        if not np.isfinite(share) or share <= 0.0:
            return None
        return {"path": str(path), "share_per_coef": share,
                "checkpoint": data.get("checkpoint")}
    except Exception:  # noqa: BLE001 - 诊断文件缺失/损坏不影响训练
        return None


def _gpu_used_mib() -> Dict[int, int]:
    """``nvidia-smi`` 查询每张 GPU 已用显存（MiB）；查询失败返回 {}（=不做剔除）。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10, stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001
        return {}
    used: Dict[int, int] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            used[int(parts[0])] = int(float(parts[1]))
        except ValueError:
            continue
    return used


def _read_quality_json(path: str) -> Dict[str, Any]:
    """读离线质量统计 JSON（空路径/读失败 -> ``{}``，绝不打断训练）。

    只用于 ``--ramp-adaptive 1`` 的门控指标：给了 ``--ramp-quality-json`` 就优先用它
    的 ``survive_t12_ratio`` / ``tier_ge4_ratio``（真实 tier 观测），否则退回采集轨迹代理。
    每个 iteration 现读（文件可被外部脚本更新），失败静默退化。
    """
    if not str(path or "").strip():
        return {}
    try:
        with open(str(path), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except BaseException:  # noqa: BLE001 - 门控指标退化为轨迹代理
        return {}


def _atomic_torch_save(obj: Any, path: str) -> None:
    import torch
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _net_size_kwargs(args: Any) -> Dict[str, int]:
    """entity 尺寸参数（值为 0 = 未指定）。只用于**从零构建**；checkpoint 路径以 meta 为准。

    默认全 0 ⇒ 返回空 dict ⇒ ``build_net`` 行为与历史完全相同（逐位等价）。
    """
    if str(getattr(args, "arch", "")) not in ("entity", "entity18m"):
        return {}
    out: Dict[str, int] = {}
    for key in ("d_model", "n_layers", "n_heads"):
        val = int(getattr(args, key, 0) or 0)
        if val > 0:
            out[key] = val
    return out


def _apply_custom_rewards(batch: Any, *, upgrade_w: float = 0.0,
                          win_w: float = 0.0, strength_c: float = 0.0,
                          gamma: float, lam: float) -> Optional[Dict[str, Any]]:
    """三个**默认关闭**的奖励改造，一次改 rewards 后**整批重算 GAE**。

    1. ``upgrade_w``（`--upgrade-reward`）：``upgrade`` 动作（``A_UPGRADE=3``）即时 +w；
       rank-only 下升本收益要 5–8 回合才兑现，GAE λ=0.995 会把它压没。
    2. ``win_w``（`--win-only-reward`）：**终局只有第 1 名 +w**，其余 0；
       用于"减弱名次奖励、只奖励夺冠"（配合 ``--rank-utility-coef 0``）。
    3. ``strength_c``（`--strength-reward-coef`）：**每回合 ΔL(Σ) 密集项**
       （``L = log1p(Σ)/log1p(30000)``，沙盒 telescope 同一曲线）；
       回合末决策上加 ``strength_c × (L(Σ_t) − L(Σ_{t−1}))``，首回合以 0 为基准。
    全部为 0（默认）时**不调用** ⇒ 逐位等价旧路径。
    """
    from hsbg.az.train.ppo import compute_gae, log_strength_from_features  # noqa: PLC0415
    n = len(batch.actions)
    if n <= 0:
        return None
    delta = np.zeros(n, dtype=np.float64)
    n_up = n_win = 0
    if float(upgrade_w) > 0.0:
        a_upgrade = 3  # hsbg/env/action_space_std.py: A_UPGRADE
        hit = (np.asarray(batch.actions) == a_upgrade) & (~np.asarray(batch.forced, dtype=bool))
        delta[hit] += float(upgrade_w)
        n_up = int(hit.sum())
    if float(win_w) > 0.0:
        dones = np.asarray(batch.dones, dtype=np.float32) > 0.5
        ranks = np.asarray(batch.ranks)
        win_hit = dones & (ranks == 1)
        delta[win_hit] += float(win_w)
        n_win = int(win_hit.sum())
    if float(strength_c) > 0.0:
        L = np.asarray(log_strength_from_features(batch.features), dtype=np.float64)
        turns = np.asarray(batch.turns)
        eps = np.asarray(batch.episode_ids)
        # episode 连续（collect 侧保证），逐 episode 找「每回合最后一个决策」
        bounds = np.flatnonzero(np.diff(eps) != 0) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [n]))
        for s, e in zip(starts.tolist(), ends.tolist()):
            t = turns[s:e]
            if t.size == 0:
                continue
            change = np.flatnonzero(np.diff(t) != 0) + s + 1  # 每回合第一个决策
            turn_starts = np.concatenate(([s], change))
            turn_ends = np.concatenate((change - 1, [e - 1]))
            prev_L = 0.0
            for ts, te in zip(turn_starts.tolist(), turn_ends.tolist()):
                cur_L = float(L[te])
                delta[te] += float(strength_c) * (cur_L - prev_L)
                prev_L = cur_L
    if not (n_up or n_win or float(strength_c) > 0.0):
        return None
    rewards = (np.asarray(batch.rewards, dtype=np.float64) + delta)
    adv, ret = compute_gae(rewards, np.asarray(batch.values, dtype=np.float64),
                           np.asarray(batch.dones, dtype=np.float64),
                           gamma=float(gamma), lam=float(lam))
    batch.rewards = rewards.astype(np.float32)
    batch.advantages = adv
    batch.returns = ret
    return {"upgrade_w": float(upgrade_w), "upgrade_hits": n_up,
            "win_w": float(win_w), "win_episodes": n_win,
            "strength_c": float(strength_c),
            "reward_delta_sum": float(delta.sum()),
            "reward_delta_abs": float(np.abs(delta).sum())}


def _critic_param_names(model: Any, *, use_value_head: bool) -> List[str]:
    """默认 critic 路径 = ``rank_head``（8 名次 CE）；``value_head`` 只在 coef>0 时加入。"""
    names = [n for n, _ in model.named_parameters() if n.startswith("rank_head")]
    if use_value_head:
        names += [n for n, _ in model.named_parameters() if n.startswith("value_head")]
    return names


def _value_warmup(trainer: Any, batch: Any, *, iters: int, lr: float,
                  minibatch: int, seed: int) -> Optional[Dict[str, Any]]:
    """critic 预热：只训 critic 头（rank 头[+标量 value 头]）``iters`` 遍。

    为什么必须有：BC 只训策略 ⇒ critic 头仍是**随机初始化**。此时 GAE 的 advantage 与真实
    优势近似无关（沙盒实测符号一致率 52.8% ≈ 抛硬币），PPO 第一个 iteration 就会把策略
    推歪。这里先在**第一个 iteration 的真实轨迹**上把 critic 头校准到「能预测真实名次」。

    为什么只训头：entity 的 critic 与策略**共享 trunk**，整网回传会改变策略 logits
    （沙盒踩过：anchor_kl 首行 0.205、130k 后 Σ 1646→524）。冻结除 critic 头以外的全部
    参数 ⇒ 策略漂移**严格为 0**（由构造保证，不是经验值）。

    默认 ``iters=0`` 时本函数不会被调用 ⇒ 默认路径逐位等价于旧行为。
    """
    import torch  # noqa: PLC0415 - 延迟导入：tiny 路径不保证有 torch
    model = trainer.model
    cfg = trainer.config
    dev = torch.device(trainer.device)
    use_vh = float(getattr(cfg, "value_head_coef", 0.0) or 0.0) > 0.0
    names = _critic_param_names(model, use_value_head=use_vh)
    if not names:
        print("[value-warmup] 模型没有 rank_head/value_head，跳过预热", flush=True)
        return None
    name_set = set(names)
    params = [p for n, p in model.named_parameters() if n in name_set]
    n_params = int(sum(p.numel() for p in params))
    model.eval()

    n = len(batch)
    if n <= 0:
        print("[value-warmup] 批次为空，跳过预热", flush=True)
        return None
    mb = max(1, min(int(minibatch or 1024), int(n)))
    rng = np.random.default_rng([int(seed), 0x5A1])
    ranks_np = np.asarray(batch.ranks, dtype=np.int64)
    rets_np = np.asarray(batch.returns, dtype=np.float32)
    rank_mode = str(getattr(cfg, "value_loss", "rank")).strip().lower() == "rank"
    amp_bf16 = (str(getattr(cfg, "amp", "off")).strip().lower() in ("bf16", "bfloat16")
                and dev.type == "cuda")
    F = torch.nn.functional
    hist: List[float] = []
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=float(lr))
    try:
        for _w in range(max(1, int(iters))):
            order = rng.permutation(int(n))
            tot, cnt = 0.0, 0
            for s0 in range(0, int(n), mb):
                idx = order[s0:s0 + mb]
                x = torch.from_numpy(np.ascontiguousarray(batch.features[idx])).to(dev)
                c = torch.from_numpy(
                    np.ascontiguousarray(batch.cards[idx].astype(np.int64))).to(dev)
                m = torch.from_numpy(
                    np.ascontiguousarray(batch.masks[idx].astype(bool))).to(dev)
                if amp_bf16:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _lg, rank_logits, scalar_value = forward_model(model, x, c, m)
                else:
                    _lg, rank_logits, scalar_value = forward_model(model, x, c, m)
                rl = rank_logits.float()
                if rank_mode:
                    tgt = torch.from_numpy(np.ascontiguousarray(
                        (ranks_np[idx] - 1).clip(0, rl.shape[-1] - 1))).to(dev)
                    loss = F.cross_entropy(rl, tgt)
                else:
                    tgt = torch.from_numpy(
                        np.ascontiguousarray(rets_np[idx])).to(dev)
                    loss = F.mse_loss(value_from_rank_logits_torch(rl), tgt)
                if use_vh and scalar_value is not None:
                    loss = loss + float(cfg.value_head_coef) * F.mse_loss(
                        scalar_value.float().reshape(-1), tgt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * int(len(idx))
                cnt += int(len(idx))
            hist.append(tot / max(1, cnt))
    finally:
        for p in model.parameters():
            p.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        model.eval()
    out = {"iters": int(iters), "critic_params": n_params,
           "critic_param_names": names, "lr": float(lr), "minibatch": mb,
           "loss_mode": "rank_ce" if rank_mode else "mse",
           "loss_first": float(hist[0]) if hist else None,
           "loss_last": float(hist[-1]) if hist else None,
           "n_samples": int(n), "policy_drift_by_construction": 0.0}
    print(f"[value-warmup] critic={names} params={n_params:,} iters={iters} "
          f"lr={lr:g} mb={mb} mode={out['loss_mode']} "
          f"loss {out['loss_first']:.4f} -> {out['loss_last']:.4f} "
          f"(n={n}; 策略/trunk 冻结 ⇒ 策略漂移=0)", flush=True)
    return out


def main() -> int:
    # 2026-11-17：本机控制台是 cp936，help/日志里的非 GBK 字符（如 U+2212 减号、
    # ⇒）会让 argparse 在打印 help 时直接 UnicodeEncodeError。把两条流改成 UTF-8
    # 与 `ops/hsai_ops.py` 同一做法；失败就保持原样（不影响训练）。
    for _stream in (sys.stdout, sys.stderr):
        try:
            if (_stream.encoding or "").lower().replace("-", "") != "utf8":
                _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    from hsbg.az.train.cli import build_parser
    args = build_parser().parse_args()
    if not 0 <= args.lobby_filter_probability <= 1:
        raise ValueError("invalid lobby filter probability")
    if args.lobby_filter_probability < 1 and (args.rust_collector or not args.lobby_exclude):
        raise ValueError("mixed lobby practice requires the Python collector and an exclusion set")
    # 健康门阈值：整轮只构造一次（纯数据，无副作用）。
    health_limits = HealthLimits.from_args(args)
    # 2026-11-16：冠军奖金必须在**任何**策略/训练器构造之前生效——它是原地改写
    # common.RANK_UTILITY，所有 `from .common import RANK_UTILITY` 的模块共享同一数组。
    from hsbg.az.train.common import rank_utility_table, set_rank_utility_table, set_win_bonus
    _wb = set_win_bonus(float(getattr(args, "win_bonus", 0.0) or 0.0))
    # 关键：采集子进程是 spawn 出来的，会把 common 重新 import 一遍。这里把奖金写进
    # 环境变量，common 在 import 时会按它再应用一次，保证**子进程算奖励时也带奖金**。
    os.environ["HSAI_WIN_BONUS"] = str(_wb)
    # 整表优先（2026-11-17）：只奖励前二。同样要传进 spawn 子进程。
    _tbl_raw = str(getattr(args, "rank_utility_table", "") or "").strip()
    if _tbl_raw:
        vals = [float(x) for x in _tbl_raw.replace(" ", "").split(",") if x]
        set_rank_utility_table(vals)
        os.environ["HSAI_RANK_UTILITY"] = ",".join("%g" % v for v in vals)
        if _wb:
            print("[warn] 同时给了 --win-bonus 与 --rank-utility-table：以整表为准", flush=True)
    print(f"[reward] win_bonus={_wb} rank_utility={[round(float(x),3) for x in rank_utility_table()]}",
          flush=True)
    # 2026-11-17（D-070）：校验对着 TRAINING_CONTRACT 做（唯一真值），不再写死字面量。
    from hsbg.az.train.eval_policy import TRAINING_CONTRACT as _CONTRACT
    _bad = []
    if int(args.max_actions_per_turn) != int(_CONTRACT["max_actions_per_turn"]):
        _bad.append(f"max-actions={args.max_actions_per_turn}"
                    f"(契约 {_CONTRACT['max_actions_per_turn']})")
    if int(args.move_guard) != int(bool(_CONTRACT["move_guard"])):
        _bad.append(f"move-guard={args.move_guard}")
    if int(args.move_cap) != int(_CONTRACT["move_cap"]):
        _bad.append(f"move-cap={args.move_cap}(契约 {_CONTRACT['move_cap']})")
    if int(args.move_repeat_limit) != int(_CONTRACT["move_repeat_limit"]):
        _bad.append(f"move-repeat={args.move_repeat_limit}")
    if int(args.forbid_cancel) != int(bool(_CONTRACT["forbid_cancel"])):
        _bad.append(f"forbid-cancel={args.forbid_cancel}")
    if int(args.sims) != 0:
        _bad.append(f"sims={args.sims}")
    if _bad:
        ap.error("强制动作规则与 TRAINING_CONTRACT 不一致：" + ", ".join(_bad)
                 + "（契约见 hsbg/az/train/eval_policy.py）")
    if args.local_batch_inference and (int(args.processes) != 1 or int(args.workers) != 1
                                      or not int(args.lockstep) or int(args.sims) != 0):
        ap.error("local-batch-inference仅支持processes=workers=1、lockstep=1、sims=0")
    args.out = os.path.abspath(args.out)
    from hsbg.az.train.recruit_start import validate_options as validate_recruit_start, load_bank as load_recruit_bank
    validate_recruit_start(args.recruit_start_bank,args.recruit_start_p,
                           lockstep=bool(args.lockstep),rust_collector=bool(args.rust_collector))
    args.recruit_start_manifest_sha256=''
    if args.recruit_start_p:
        if (args.entry_p or args.lobby_exclude or args.mirror_strict
                or str(args.opponent)!='bot' or args.sims):
            raise ValueError('recruit starts require bot-mode snapshot field, sims=0, no legacy entry/filter/strict mirror')
        start_manifest,_start_rows,start_identity=load_recruit_bank(args.recruit_start_bank)
        if str(args.mirror_snaps)!=start_manifest['source_training_field']:
            raise ValueError('recruit start source and training snapshot fields differ')
        args.recruit_start_manifest_sha256=start_identity['sha256']
        print('[recruit-start] '+json.dumps(dict(identity=start_identity,variant=start_manifest['variant'],
                                               states=len(_start_rows),probability=args.recruit_start_p)),flush=True)
    # 经济课程：gold_curriculum=1 且 stage=none 时按 warmup 处理；非法倍数/窗口启动期报错
    if int(args.gold_curriculum) and str(args.curriculum_stage) == "none":
        args.curriculum_stage = "warmup"
    try:
        _curr_cfg = curriculum_schedule(
            stage=str(args.curriculum_stage),
            mult=float(args.curriculum_gold_mult),
            anneal_decisions=float(args.curriculum_anneal_decisions))
    except ValueError as exc:
        print(f"[error] 经济课程参数非法: {exc}", file=sys.stderr)
        return 2
    if (_curr_cfg["curriculum_stage"] != "none"
            and _curr_cfg["curriculum_gold_mult"] == 1.0):
        print("[warn] 经济课程已开启但 --curriculum-gold-mult=1.0：金币不变（等价关闭）",
              flush=True)
    if (float(args.teacher_bc_coef) > 0.0
            or float(args.process_value_coef) > 0.0) and not int(args.lockstep):
        print("[warn] 教师 DAgger / 过程价值标签只在 --lockstep 1 采集路径产出；"
              "本次非 lockstep 将得到空标签（对应损失为 0）", flush=True)
    if float(args.library_p) > 0.0:
        print("[error] --library-p（v1 单座位注入）未通过终审，已禁用；"
              "请用 --trajectory-pool/--trajectory-p", file=sys.stderr)
        return 2
    # 回合末场面快照镜像（默认关闭；--mirror-p 0 且无 ramp 时完全零影响）
    if str(args.mirror_sample).lower().startswith("turnband"):
        try:
            parse_turn_bands(str(args.mirror_sample))
        except ValueError as exc:
            print(f"[error] --mirror-sample turnband 配置错误: {exc}", file=sys.stderr)
            return 2
    _mirror_ramp = str(args.mirror_p_ramp).strip()
    if args.mirror_strict and (not args.lockstep or args.rust_collector or not args.mirror_snaps
            or args.mirror_p != 1 or args.mirror_seats != 7 or _mirror_ramp):
        ap.error("严格镜像仅支持Python lockstep、七个镜像席和固定概率1")

    if _mirror_ramp:
        try:
            ramp_p(0, _mirror_ramp)      # 提前校验格式，避免训练中途才报错
        except ValueError as exc:
            print(f"[error] --mirror-p-ramp 格式错误: {exc}", file=sys.stderr)
            return 2
    if (float(args.mirror_p) > 0.0 or _mirror_ramp) and int(args.mirror_seats) <= 0:
        print("[error] --mirror-p>0/--mirror-p-ramp 需要 --mirror-seats>0"
              "（给随机/固定座位注入）", file=sys.stderr)
        return 2
    if (float(args.mirror_p) > 0.0 or _mirror_ramp) and not int(args.lockstep):
        print("[warn] 回合末快照镜像只在 --lockstep 1 路径生效", flush=True)
    if str(args.mirror_snaps) and not os.path.isdir(str(args.mirror_snaps)):
        print(f"[warn] --mirror-snaps 目录不存在: {args.mirror_snaps}", flush=True)
    # ---- 对手强度课程（human-board 席数/概率/分位按累计 decisions 爬坡）----
    # 默认（spec 全空 + adaptive=0）是恒等映射：下面所有分支都走旧路径。
    try:
        board_ramp = BoardRampController(
            str(args.human_board_ramp), mode=str(args.ramp_mode),
            quantile_spec=str(args.human_board_quantile_ramp),
            adaptive=bool(int(args.ramp_adaptive)),
            base_seats=int(args.human_board_seats),
            base_p=float(args.human_board_p),
            base_sample=str(args.human_board_sample),
            window=int(args.ramp_window),
            survive_threshold=float(args.ramp_survive_threshold),
            tier4_threshold=float(args.ramp_tier4_threshold),
            max_forced_per_game=float(args.ramp_max_forced_per_game),
            delay_decisions=args.ramp_delay_decisions,
            start_decisions=args.ramp_start_decisions)
    except ValueError as exc:
        print(f"[error] 对手强度课程参数非法: {exc}", file=sys.stderr)
        return 2
    if board_ramp.enabled:
        _desc = board_ramp.describe()
        print(f"[ramp] 对手强度课程开启: mode={_desc['mode']} "
              f"bands={_desc['bands']} spec={_desc['spec']!r} "
              f"quantile={_desc['quantile_spec']!r} "
              f"adaptive={int(_desc['adaptive'])} window={_desc['window']} "
              f"start_decisions={_desc['start_decisions']}", flush=True)
        if int(args.ramp_adaptive):
            print(f"[ramp] 门控: survive_t12>={float(args.ramp_survive_threshold):g} 且 "
                  f"tier4>={float(args.ramp_tier4_threshold):g} 且 "
                  f"forced/game<={float(args.ramp_max_forced_per_game):g} 才升档；"
                  f"否则推迟 {int(_desc['delay_decisions'])} decisions", flush=True)
        for _i, (_d, _s, _p) in enumerate(_desc["points"]):
            print(f"[ramp] band {_i}: decisions>={_d} seats={_s} p={_p:g}",
                  flush=True)
        for _i, (_d, _q) in enumerate(_desc["quantile_points"]):
            print(f"[ramp] quantile band {_i}: decisions>={_d} sample={_q}",
                  flush=True)
        if not int(args.lockstep):
            print("[warn] 对手强度课程依赖人类战团注入，只在 --lockstep 1 路径生效",
                  flush=True)
        if int(args.ramp_adaptive) and str(args.ramp_quality_json):
            print(f"[ramp] 自适应门控指标优先读质量统计: {args.ramp_quality_json}",
                  flush=True)
        elif int(args.ramp_adaptive):
            print("[ramp] 自适应门控指标：采集轨迹代理（max_turn>=12 / max_turn>=8）；"
                  "没有真实 tier 观测，建议用 --ramp-quality-json 接 quality_monitor",
                  flush=True)
        if float(_desc["start_decisions"]) > 0.0:
            print(f"[ramp] 起始 decisions 偏移 = {_desc['start_decisions']}"
                  "（续训请传上一次的累计值）", flush=True)
    # ---- 动态难度（闭环）：难度跟着**实测战力**走，不按时间表 ---------------------
    # D-054：先降难度让模型活到中后期，再把对手顶在"略高于自身"的能力边界。
    # 档位只由实测到达率/战斗胜率驱动（hsbg.az.train.opponent_arena），不按 iteration 排期。
    arena = None
    arena_curve = None
    _arena_cfg_path = str(getattr(args, "arena_config", "") or "").strip()
    if _arena_cfg_path:
        from hsbg.az.train.human_curve import HumanCurve, band_center_turn
        from hsbg.az.train.opponent_arena import ArenaController, parse_spec
        if not os.path.isfile(_arena_cfg_path):
            print(f"[error] --arena-config 不存在: {_arena_cfg_path}", file=sys.stderr)
            return 2
        with open(_arena_cfg_path, "r", encoding="utf-8") as _f:
            _acfg = json.load(_f)
        arena_curve = HumanCurve.load(str(_acfg.get("curve", "") or ""))
        if not arena_curve.ok:
            print(f"[error] 难度课程需要真人强度曲线（{_acfg.get('curve')!r} 读不到/为空）；"
                  "先跑 tools/pool_strength_curve.py", file=sys.stderr)
            return 2
        _notes: List[str] = []
        _spec = parse_spec(str(_acfg.get("spec", "") or ""), note=_notes)
        _spec.enabled = bool(int(_acfg.get("enabled", 1)))
        _bounds = {}
        _init = {}
        _init_q = dict(_acfg.get("init_q") or {})
        for _b in ("early", "mid", "late"):
            _bb = arena_curve.band_bounds(_b)
            if _bb is None:
                print(f"[error] 强度曲线缺少 {_b} 段的可用分位", file=sys.stderr)
                return 2
            _bounds[_b] = _bb
            _q0 = float(_init_q.get(_b, 0.30))
            _init[_b] = float(arena_curve.strength(band_center_turn(_b), _q0) or _bb[0])
        _spec.bounds = _bounds
        _state_path = str(getattr(args, "arena_state", "") or "").strip() or \
            os.path.join(str(args.out), "arena_state.json")
        arena = ArenaController(_spec, _init, path=_state_path, note=_notes)
        for _n in getattr(arena, "notes", []) or []:
            print(f"[arena] note: {_n}", flush=True)
        print(f"[arena] 动态难度开启 curve={_acfg.get('curve')} state={_state_path} "
              f"init_q={ {k: round(float(v), 3) for k, v in _init_q.items()} } "
              f"init_strength={ {k: int(v) for k, v in _init.items()} } "
              f"bounds={ {k: (int(v[0]), int(v[1])) for k, v in _bounds.items()} } "
              f"reach_target={float(_spec.reach_target):g}", flush=True)

    def _arena_stats_from_meta(m: Any) -> Dict[str, Any]:
        """采集 meta 的扁平 ``arena_*`` 计数 -> 控制器吃的结构（缺键当 0，绝不抛）。"""
        if not isinstance(m, dict):
            return {"episodes": 0, "reach": {}, "bands": {}}
        out: Dict[str, Any] = {
            "episodes": int(m.get("arena_episodes", 0) or 0),
            "reach": {"9": int(m.get("arena_reach9_n", 0) or 0),
                      "12": int(m.get("arena_reach12_n", 0) or 0)},
            "bands": {},
        }
        for _b in ("early", "mid", "late"):
            _own = m.get(f"arena_own_{_b}")
            _opp = m.get(f"arena_opp_{_b}")
            out["bands"][_b] = {
                "combats": int(m.get(f"arena_combats_{_b}", 0) or 0),
                "losses": int(m.get(f"arena_losses_{_b}", 0) or 0),
                "own_str_sum": int(_own or 0), "own_str_n": 1 if _own else 0,
                "opp_str_sum": int(_opp or 0), "opp_str_n": 1 if _opp else 0,
            }
        return out

    # 人类战团（board-only）注入（默认关闭；p 全 0 时完全零影响）
    if board_ramp.max_p() > 0.0:
        if int(args.human_board_seats) <= 0 and not board_ramp.enabled:
            print("[error] --human-board-p>0 需要 --human-board-seats>0", file=sys.stderr)
            return 2
        if not str(args.human_boards):
            args.human_boards = "data/real_library/library_real_degraded.jsonl"
        if not os.path.isfile(str(args.human_boards)):
            print(f"[error] --human-boards 文件不存在: {args.human_boards}", file=sys.stderr)
            return 2
        _fid = str(getattr(args, "human_board_fidelity", "board") or "board")
        if _fid != "board":
            _need_rows = 0
            try:
                _need_rows = sum(1 for _ in open(args.human_boards, encoding="utf-8")
                                 if '"board_detail"' in _)
            except BaseException:  # noqa: BLE001
                _need_rows = 0
            if _need_rows <= 0:
                print(f"[error] --human-board-fidelity={_fid} 需要准完整库"
                      "（含 board_detail 的行）；当前文件没有："
                      f"{args.human_boards}\n"
                      "  先跑 PYTHONPATH=engine/python:. python "
                      "workflows/build_fidelity_library.py --stats", file=sys.stderr)
                return 2
            print(f"[info] 人类战团保真度 = {_fid}（准完整行 {_need_rows}；"
                  "注入走 apply_turn_snapshots + ended_turn 复位）", flush=True)
        if not int(args.lockstep):
            print("[warn] 人类战团注入只在 --lockstep 1 路径生效", flush=True)
        if int(args.processes) <= 1:
            print("[warn] 人类战团注入只在 --processes>1 的进程池 worker 内生效；"
                  "--processes 1 的逐局采集不会注入（对手强度课程因此无效）", flush=True)
        elif (float(args.mirror_p) > 0.0 or _mirror_ramp):
            print("[info] 人类战团 + 回合末快照镜像同时开启："
                  "同座位同回合镜像快照优先（记 human_board_skipped_by_mirror）", flush=True)
    elif str(args.human_boards):
        print("[warn] --human-boards 已设置但 --human-board-p=0：本次不注入", flush=True)
    # ── 中后期入场课程（learner 席注入真人完整状态；默认关闭）────────────────
    if float(args.entry_p) > 0.0 or int(args.entry_turn_min) > 0:
        if float(args.entry_p) <= 0.0 or int(args.entry_turn_min) <= 0:
            print("[error] 入场课程需要同时给 --entry-p>0 与 --entry-turn-min>0",
                  file=sys.stderr)
            return 2
        if not str(args.entry_boards):
            print("[error] 入场课程需要 --entry-boards（准完整库 JSONL）", file=sys.stderr)
            return 2
        if not os.path.isfile(str(args.entry_boards)):
            print(f"[error] --entry-boards 文件不存在: {args.entry_boards}", file=sys.stderr)
            return 2
        _efid = str(args.entry_fidelity or "full")
        if _efid != "board":
            _erows = 0
            try:
                _erows = sum(1 for _ in open(args.entry_boards, encoding="utf-8")
                             if '"board_detail"' in _)
            except BaseException:  # noqa: BLE001
                _erows = 0
            if _erows <= 0:
                print(f"[error] --entry-fidelity={_efid} 需要准完整库（含 board_detail 的行）；"
                      f"当前文件没有：{args.entry_boards}", file=sys.stderr)
                return 2
        if not int(args.lockstep) or int(args.processes) <= 1:
            print("[error] 入场课程需要 --lockstep 1 且 --processes>1", file=sys.stderr)
            return 2
        if bool(int(args.mirror_strict)):
            print("[error] 入场课程与 --mirror-strict 互斥（严格镜像要求七席全概率且"
                  "禁人类战团）", file=sys.stderr)
            return 2
        _etier = int(getattr(args, "entry_min_tier", 0) or 0)
        _ecards = str(getattr(args, "entry_require_cards", "") or "").strip()
        _ehand = int(getattr(args, "entry_min_hand", 0) or 0)
        if _etier > 0 or _ecards or _ehand:
            # 只保留观测等级 >= _etier 的帧：先把池子建一遍，池子空就直接拒绝启动。
            # （等级未知的 board-only 行会被丢弃，见 human_board_pool.HumanBoardPool.min_tier）
            try:
                from hsbg.az.train.human_board_pool import load_pool as _load_entry_pool
                _epool = _load_entry_pool(str(args.entry_boards),
                                          sample=str(args.entry_sample), min_tier=_etier,
                                          require_any_cards=str(getattr(
                                              args, "entry_require_cards", "") or "") or None,
                                          min_hand=_ehand or None)
            except BaseException as exc:  # noqa: BLE001
                print(f"[error] --entry-min-tier={_etier} 建池失败: {exc!r}",
                      file=sys.stderr)
                return 2
            if _epool is None or not int(getattr(_epool, "kept", 0)):
                print(f"[error] --entry-min-tier={_etier} 过滤后入场池为空"
                      f"（文件 {args.entry_boards}）；放宽等级或换库", file=sys.stderr)
                return 2
            print(f"[info] 入场池按等级过滤：min_tier={_etier} frames={_epool.kept} "
                  f"payload={_epool.payload_kept} filtered={_epool.filtered}",
                  flush=True)
        print(f"[info] 中后期入场课程：p={args.entry_p} 回合 "
              f"{args.entry_turn_min}..{max(int(args.entry_turn_min), int(args.entry_turn_max))}"
              f" fidelity={_efid} sample={args.entry_sample} "
              f"min_tier={_etier} "
              f"health={int(args.entry_set_health)}；T0 之前 learner 由 bot 代打",
              flush=True)
    # ── 搜索作为行为策略 μ=(1−ε)π+ε·π_search（默认 eps=0 = 完全关闭）──────────
    if float(args.behavior_mix_eps) > 0.0:
        if not (0.0 <= float(args.behavior_mix_eps) < 1.0):
            print("[error] --behavior-mix-eps 必须在 [0,1)", file=sys.stderr)
            return 2
        if not int(args.lockstep) or int(args.processes) <= 1:
            print("[error] --behavior-mix-eps>0 需要 --lockstep 1 且 --processes>1"
                  "（混合行为策略只在进程池的 lockstep 采集器里生效）",
                  file=sys.stderr)
            return 2
        if str(args.behavior) != "net":
            print("[error] --behavior-mix-eps>0 需要 --behavior net"
                  "（μ 的网络分量是 π；mcts/mixed 会再叠一层 MCTS 混合）",
                  file=sys.stderr)
            return 2
        print(f"[behavior-mix] 开启：eps={float(args.behavior_mix_eps):g} "
              f"when={args.behavior_mix_when} "
              f"max_per_turn={int(args.behavior_mix_max_per_turn)} "
              f"topk={int(args.behavior_mix_topk)} turns={int(args.behavior_mix_turns)} "
              f"max_ms={float(args.behavior_mix_max_ms):g}；"
              "records 写 logp_mu/logp_pi，PPO ratio 分母 = logp_mu", flush=True)
    if str(args.behavior_mix_report):
        _bm_dir = os.path.dirname(os.path.abspath(str(args.behavior_mix_report)))
        if _bm_dir:
            os.makedirs(_bm_dir, exist_ok=True)
        print(f"[behavior-mix] 逐 iteration 统计 -> {args.behavior_mix_report}",
              flush=True)

    # ---- 势函数奖励塑形（默认关闭；coef=0 时零影响）----
    if float(args.shaping_coef) < 0.0:
        print("[error] --shaping-coef 必须 >= 0", file=sys.stderr)
        return 2
    try:
        parse_tier_bonus(str(args.shaping_tier_bonus))   # 启动期 fail-fast
    except ValueError as exc:
        print(f"[error] --shaping-tier-bonus 非法: {exc}", file=sys.stderr)
        return 2
    # 折扣范围与一致性：γ_s 缺省 = 跟随训练 γ；显式不一致必须警告（不静默）
    if not (0.0 <= float(args.gamma) <= 1.0):
        print("[error] --gamma 必须在 [0, 1] 内", file=sys.stderr)
        return 2
    if args.shaping_gamma is not None and not (0.0 <= float(args.shaping_gamma) <= 1.0):
        print("[error] --shaping-gamma 必须在 [0, 1] 内", file=sys.stderr)
        return 2
    shape_gamma = (float(args.gamma) if args.shaping_gamma is None
                   else float(args.shaping_gamma))
    shape_aligned = (shape_gamma == float(args.gamma))
    if float(args.shaping_coef) > 0.0:
        if str(args.opponent) == "self":
            print("[error] 奖励塑形只支持 --opponent bot|weak（单 learner seat）",
                  file=sys.stderr)
            return 2
        if int(args.rust_collector):
            print("[error] Rust 采集器暂不支持奖励塑形；请设 --shaping-coef 0 "
                  "或 --rust-collector 0", file=sys.stderr)
            return 2
        print(f"[shaping] 势函数奖励塑形开启: coef={float(args.shaping_coef):g} "
              f"γ_s={shape_gamma:g}（训练 γ={float(args.gamma):g}，"
              f"{'对齐' if shape_aligned else '不一致'}） "
              f"Φ={float(args.shaping_w_str):g}*Σ(atk+hp)+"
              f"{float(args.shaping_w_tier):g}*tier_bonus "
              f"tier_bonus={args.shaping_tier_bonus}", flush=True)
        if not shape_aligned:
            print("[shaping][warn] " + shaping_gamma_warning(
                float(args.shaping_coef), shape_gamma, float(args.gamma)),
                file=sys.stderr, flush=True)
        if not int(args.lockstep):
            print("[info] 非 lockstep 路径：塑形在逐局采集（collect_ppo_records_parallel）"
                  "内生效", flush=True)

    # ---- 法术自举补贴（动作条件奖励，默认关闭）----
    if float(getattr(args, "spell_bootstrap_coef", 0.0) or 0.0) != 0.0:
        if str(args.opponent) == "self":
            print("[error] 法术自举补贴只支持 --opponent bot|weak（单 learner seat）",
                  file=sys.stderr)
            return 2
        if float(getattr(args, "spell_bootstrap_coef", 0.0)) < 0.0:
            print("[error] --spell-bootstrap-coef 必须 >= 0", file=sys.stderr)
            return 2
        if int(getattr(args, "spell_bootstrap_per_game", 0) or 0) <= 0:
            print("[error] --spell-bootstrap-per-game 必须 >= 1", file=sys.stderr)
            return 2
        print(f"[spell-bootstrap] 开启：coef={float(args.spell_bootstrap_coef):g} "
              f"每局上限={int(args.spell_bootstrap_per_game)} 动作={SPELL_BUY_ACTION}"
              "（商店法术槽；动作条件奖励，非策略不变）", flush=True)

    # ---- records 数据通路（默认 auto/fp16；`--ipc-mode disk --records-dtype fp32` = 历史路径。
    #      见 docs/TRAIN_THROUGHPUT.md：shm 省掉 335-438 MB/iter 写盘+读盘，fp16 再减半字节）----
    _ipc_requested = str(args.ipc_mode).strip().lower()
    if _ipc_requested == "auto":
        args.ipc_mode = ("shm" if (int(args.processes) >= 2 and os.path.isdir("/dev/shm"))
                         else "disk")
    if str(args.ipc_mode) == "shm":
        if int(args.processes) < 2:
            print("[error] --ipc-mode shm 需要 --processes >= 2（只有进程 pool 才走 IPC）",
                  file=sys.stderr)
            return 2
        if not os.path.isdir("/dev/shm"):
            print("[error] --ipc-mode shm 需要 /dev/shm（本机不存在）", file=sys.stderr)
            return 2
        print(f"[ipc] shm 环形缓冲: slots={int(args.ipc_shm_slots)}（每 slot 按 32 MiB 取整），"
              f"records_dtype={args.records_dtype}", flush=True)
    else:
        print(f"[ipc] disk 通路（ipc_mode={_ipc_requested}->disk），"
              f"records_dtype={args.records_dtype}", flush=True)

    # ---- 换位去抖守卫（默认关闭；--move-guard 0 时零影响）----
    if int(args.move_cap) < 0:
        print("[error] --move-cap 必须 >= 0", file=sys.stderr)
        return 2
    if int(args.move_repeat_limit) < 1:
        print("[error] --move-repeat-limit 必须 >= 1", file=sys.stderr)
        return 2
    if int(args.move_guard):
        if int(args.rust_collector):
            print("[error] Rust 采集器不支持换位守卫（它不经 PPOSelfPlayPolicy 采样）；"
                  "请设 --move-guard 0 或 --rust-collector 0", file=sys.stderr)
            return 2
        print(f"[move-guard] 换位去抖守卫开启: cap={int(args.move_cap)} "
              f"repeat_limit={int(args.move_repeat_limit)} "
              f"（逆换位 + 同动作超限 + 每回合 cap，见 docs/MOVE_GUARD.md）",
              flush=True)
        if not int(args.lockstep):
            print("[info] 非 lockstep 路径：守卫在逐局采集（collect_ppo_records_parallel）"
                  "内同样生效", flush=True)

    # ---- 等级进度奖励（tier-target，默认关闭；coef=0 时零影响）----
    if float(args.tier_target_coef) < 0.0:
        print("[error] --tier-target-coef 必须 >= 0", file=sys.stderr)
        return 2
    if float(args.tier_penalty) < 0.0 or float(args.tier_bonus) < 0.0:
        print("[error] --tier-penalty / --tier-bonus 必须 >= 0", file=sys.stderr)
        return 2
    try:
        parse_tier_target_table(str(args.tier_target_table))
    except ValueError as exc:
        print(f"[error] --tier-target-table 非法: {exc}", file=sys.stderr)
        return 2
    if float(args.tier_target_coef) > 0.0:
        if str(args.opponent) == "self":
            print("[error] 等级进度奖励只支持 --opponent bot|weak（单 learner seat）",
                  file=sys.stderr)
            return 2
        if int(args.rust_collector):
            print("[error] Rust 采集器暂不支持等级进度奖励；请设 --tier-target-coef 0 "
                  "或 --rust-collector 0", file=sys.stderr)
            return 2
        print(f"[tier-target] 等级进度奖励开启: coef={float(args.tier_target_coef):g} "
              f"penalty={float(args.tier_penalty):g} bonus={float(args.tier_bonus):g} "
              f"table={args.tier_target_table}", flush=True)
        if not int(args.lockstep):
            print("[info] 非 lockstep 路径：等级进度奖励在逐局采集"
                  "（collect_ppo_records_parallel / collect_ppo_batch）内生效", flush=True)

    # ---- 阵容覆盖奖励（默认关闭；见 hsbg/az/train/comp_target.py）----
    if float(args.comp_target_coef) > 0.0:
        from hsbg.az.train.comp_target import (  # noqa: E402
            load_comps as _load_comps, parse_tier_weights as _parse_tw)
        if str(args.opponent) == "self":
            print("[error] 阵容覆盖奖励只支持 --opponent bot|weak（单 learner seat）",
                  file=sys.stderr)
            return 2
        if int(args.rust_collector):
            print("[error] Rust 采集器暂不支持阵容覆盖奖励；请设 --comp-target-coef 0 "
                  "或 --rust-collector 0", file=sys.stderr)
            return 2
        try:
            _tw = _parse_tw(str(args.comp_target_tier_w))
            _comps = _load_comps(str(args.comp_target_cards), _tw)
        except (OSError, ValueError) as exc:
            print(f"[error] 阵容库不可用（{args.comp_target_cards}）: {exc}", file=sys.stderr)
            return 2
        _s = sum(1 for c in _comps if c["tier"] == "S")
        _cards = len({cid for c in _comps for cid in c["cards"]})
        print(f"[comp-target] 阵容覆盖奖励开启: coef={float(args.comp_target_coef):g} "
              f"min_turn={int(args.comp_target_min_turn)} comps={len(_comps)}（S={_s}）"
              f" cards={_cards} tier_w={args.comp_target_tier_w} cap="
              f"{float(args.comp_target_cap):g}", flush=True)
    # ---- 留存随从滚雪球奖励（默认关闭）----
    if float(args.retained_growth_coef) != 0.0:
        print(f"[retained-growth] 开启: coef={float(args.retained_growth_coef):g} "
              f"min_turn={int(args.retained_growth_min_turn)} "
              f"normalizer={float(args.retained_growth_normalizer):g}(human=690) "
              f"cap={float(args.retained_growth_cap):g}", flush=True)
        if float(args.retained_growth_normalizer) <= 0.0:
            print("[error] --retained-growth-normalizer 必须 > 0", file=sys.stderr)
            return 2

    # ---- 强制升本课程（默认关闭；见 docs/FORCED_TIER.md）----
    _ft_plan = parse_forced_tier_plan(str(args.forced_tier_plan))
    try:
        _ft_schedule = parse_prob_schedule(
            str(args.forced_tier_schedule), default=float(args.forced_tier_prob))
        _ft_teacher_until = (parse_decisions(str(args.forced_teacher_until))
                             if str(args.forced_teacher_until).strip() else 0.0)
    except ValueError as exc:
        print(f"[error] 强制升本参数非法: {exc}", file=sys.stderr)
        return 2
    if int(args.forced_tier_carry) < 0:
        print("[error] --forced-tier-carry 必须 >= 0", file=sys.stderr)
        return 2
    if float(args.forced_teacher_coef) < 0.0:
        print("[error] --forced-teacher-coef 必须 >= 0", file=sys.stderr)
        return 2
    if _ft_plan:
        if str(args.opponent) == "self":
            print("[error] 强制升本只支持 --opponent bot|weak（单 learner seat），"
                  "self 模式请关闭", file=sys.stderr)
            return 2
        if int(args.rust_collector):
            print("[error] Rust 采集器暂不支持强制升本；请设 --forced-tier-plan '' "
                  "或 --rust-collector 0", file=sys.stderr)
            return 2
        if float(args.teacher_bc_coef) > 0.0 and float(args.forced_teacher_coef) > 0.0:
            # 两条 DAgger 标签源写同一个 record 字段，同时开启会让标签互相覆盖且无法归因。
            print("[error] --teacher-bc-coef 与 --forced-teacher-coef 不能同时 > 0："
                  "两条教师标签共用 record['teacher_action']，会互相覆盖；"
                  "强制升本课程只保留 --forced-teacher-coef", file=sys.stderr)
            return 2
        _ft_teacher_on = bool(_ft_teacher_until > 0 and float(args.forced_teacher_coef) > 0)
        print(f"[forced-tier] 开启: plan={plan_spec_of(_ft_plan)} carry={int(args.forced_tier_carry)} "
              f"schedule={[(int(d), p) for d, p in _ft_schedule]} "
              f"teacher_until={int(_ft_teacher_until)} teacher_coef={float(args.forced_teacher_coef):g}",
              flush=True)
        if _ft_teacher_until > 0 and float(args.forced_teacher_coef) <= 0.0:
            print("[warn] --forced-teacher-until 已设置但 --forced-teacher-coef=0："
                  "标签会写进 record，但不会产生梯度", file=sys.stderr)
        if not _ft_teacher_on:
            pass
        if float(args.forced_tier_prob) <= 0.0 and not _ft_schedule[-1][1] > 0.0:
            print("[warn] 强制概率恒为 0：等价于只观察、不干预", file=sys.stderr)

    # ---- 强制买法术探索课程（默认关闭）----
    if int(args.forced_spell_per_game) < 0:
        print("[error] --forced-spell-per-game 必须 >= 0", file=sys.stderr)
        return 2
    if int(args.forced_spell_turn_min) < 0:
        print("[error] --forced-spell-turn-min 必须 >= 0", file=sys.stderr)
        return 2
    if int(args.forced_spell_per_game) > 0:
        print(f"[forced-spell] 开启: per_game={int(args.forced_spell_per_game)} "
              f"turn_min={int(args.forced_spell_turn_min)} "
              f"prob={float(args.forced_spell_prob):g} "
              f"teacher={int(args.forced_spell_teacher)} action=15", flush=True)

    # ---- 动作序列辅助任务：系数 ramp（默认关闭 = 零影响）----
    try:
        seq_schedule = SeqCoefSchedule.from_spec(
            base_coef=float(args.seq_coef), spec=str(args.seq_coef_ramp),
            total_decisions=(float(args.seq_ramp_total_decisions) if
                             float(args.seq_ramp_total_decisions) > 0 else
                             (float(args.stop_at_decisions) if
                              float(args.stop_at_decisions) > 0 else 3.0e7)))
    except ValueError as exc:
        print(f"[error] --seq-coef-ramp 非法: {exc}", file=sys.stderr)
        return 2
    seq_enabled = bool(seq_schedule.enabled)
    if seq_enabled:
        print(f"[seq] 动作序列辅助任务开启: {json.dumps(seq_schedule.describe(), ensure_ascii=False)}"
              f" coef(dec=0)={seq_schedule.coef(0):.4f}", flush=True)
    elif float(args.seq_coef) < 0.0:
        print("[error] --seq-coef 必须 >= 0", file=sys.stderr)
        return 2

    # ---- 注入（镜像快照 / 人类战团）：先校验引擎构建是否带注入 API ----
    # 典型事故：conda env 里 `hsbg_engine` 指向旧 clone（/tmp/...），训练命令又没设
    # PYTHONPATH → worker 每回合 AttributeError，注入全部静默失败。这里在父进程
    # 启动前 fail-fast，并给出修复命令。
    _inject_wanted = bool(int(args.lockstep)) and (
        (bool(args.mirror_snaps) and int(args.mirror_seats) > 0
         and (float(args.mirror_p) > 0.0 or str(args.mirror_p_ramp).strip()))
        or (bool(args.human_boards) and int(args.human_board_seats) > 0
            and float(args.human_board_p) > 0.0))
    if _inject_wanted:
        try:
            import hsbg_engine as _engine
            _need = []
            if bool(args.mirror_snaps) and int(args.mirror_seats) > 0 and \
                    (float(args.mirror_p) > 0.0 or str(args.mirror_p_ramp).strip()):
                _need.append("apply_turn_snapshots")
            if bool(args.human_boards) and int(args.human_board_seats) > 0 and \
                    float(args.human_board_p) > 0.0:
                _need.append("inject_board")
            _missing = [n for n in _need if not hasattr(_engine.Game, n)]
        except Exception as exc:  # noqa: BLE001
            print(f"[error] 无法 import hsbg_engine：{exc!r}", file=sys.stderr)
            return 2
        if _missing:
            print("[error] 当前 hsbg_engine 缺少注入 API "
                  f"{_missing}（import 自 {getattr(_engine, '__file__', '?')}）。"
                  "这是旧构建，注入会 100% 静默失败。请设置 "
                  "`PYTHONPATH=engine/python:.` 或 `pip install -e engine/python` 后重跑。",
                  file=sys.stderr)
            return 2

    # ---- 真实人类 BC 辅助正则：系数 ramp + 预加载池（默认关闭 = 零影响）----
    if float(args.bc_coef) < 0.0:
        print("[error] --bc-coef 必须 >= 0", file=sys.stderr)
        return 2
    if int(args.bc_batch) <= 0 or int(args.bc_every) <= 0:
        print("[error] --bc-batch 必须 > 0 且 --bc-every 必须 >= 1", file=sys.stderr)
        return 2
    if float(args.bc_type_weight) < 0.0 or float(args.bc_card_weight) < 0.0:
        print("[error] --bc-type-weight/--bc-card-weight 必须 >= 0", file=sys.stderr)
        return 2
    # ---- upgrade 子集 DPO（默认关闭 = 零影响）：系数/β/批量/周期校验 ----
    if not np.isfinite(float(args.dpo_coef)) or float(args.dpo_coef) < 0.0:
        print("[error] --dpo-coef 必须 >= 0", file=sys.stderr)
        return 2
    if not np.isfinite(float(args.dpo_beta)) or float(args.dpo_beta) <= 0.0:
        print("[error] --dpo-beta 必须是有限正数", file=sys.stderr)
        return 2
    if int(args.dpo_batch) <= 0 or int(args.dpo_every) <= 0:
        print("[error] --dpo-batch 必须 > 0 且 --dpo-every 必须 >= 1", file=sys.stderr)
        return 2
    # 场面题 / 强度辅助头（默认关闭）：系数非负、批量/周期合法
    for _nm in ("problem_bc_coef", "strength_aux_coef"):
        _v = float(getattr(args, _nm))
        if not np.isfinite(_v) or _v < 0.0:
            print(f"[error] --{_nm.replace('_', '-')} 必须 >= 0", file=sys.stderr)
            return 2
    for _nm in ("problem_batch", "problem_bc_every", "strength_aux_every"):
        if int(getattr(args, _nm)) <= 0:
            print(f"[error] --{_nm.replace('_', '-')} 必须 >= 1", file=sys.stderr)
            return 2
    # 题池开关（默认关闭）：turn 权重语法 + 强度增量阈值**启动即校验**（非法报错退出）。
    # 只在「启用场面题或显式给了开关」时导入 board_problems（保持默认路径不导入）。
    problem_turn_weights = {}
    _board_flag = (float(args.problem_bc_coef) > 0.0
                   or float(args.strength_aux_coef) > 0.0
                   or bool(str(args.problem_turn_weight or "").strip())
                   or float(args.problem_min_strength_delta) > 0.0)
    if _board_flag:
        try:
            from hsbg.az.train.board_problems import (format_turn_weights,
                                                     parse_turn_weights)
            problem_turn_weights = parse_turn_weights(
                str(args.problem_turn_weight or ""))
        except ValueError as exc:
            print(f"[error] --problem-turn-weight 非法：{exc}", file=sys.stderr)
            return 2
    _pmd = float(args.problem_min_strength_delta)
    if not np.isfinite(_pmd) or _pmd < 0.0:
        print(f"[error] --problem-min-strength-delta 必须有限且 >= 0，"
              f"得到 {args.problem_min_strength_delta!r}", file=sys.stderr)
        return 2
    _strength_enabled = bool(float(args.strength_aux_coef) > 0.0)
    _problem_enabled = bool(float(args.problem_bc_coef) > 0.0)
    _board_enabled = bool(_strength_enabled or _problem_enabled)
    try:
        bc_class_weights = parse_bc_class_weights(str(args.bc_class_weight))
    except ValueError as exc:
        print(f"[error] --bc-class-weight 非法: {exc}", file=sys.stderr)
        return 2
    try:
        bc_schedule = SeqCoefSchedule.from_spec(
            base_coef=float(args.bc_coef), spec=str(args.bc_coef_ramp),
            total_decisions=(float(args.stop_at_decisions) if
                             float(args.stop_at_decisions) > 0 else 3.0e7))
    except ValueError as exc:
        print(f"[error] --bc-coef-ramp 非法: {exc}", file=sys.stderr)
        return 2
    bc_enabled = bool(bc_schedule.enabled)
    if bc_enabled and float(args.bc_type_weight) <= 0.0 \
            and float(args.bc_card_weight) <= 0.0:
        print("[warn] BC 已启用但两层权重都为 0：自动关闭 bc_coef=0", flush=True)
        bc_enabled = False
        args.bc_coef = 0.0
        args.bc_coef_ramp = ""
    #: 是否构建/使用卡牌身份头（只在 BC 启用且 card 权重 > 0 时；bc_coef=0 不建）
    bc_card_head = bool(bc_enabled and float(args.bc_card_weight) > 0.0)
    bc_pool = None
    if bc_enabled:
        try:
            _t_bc = time.time()
            bc_pool = load_bc_samples(
                str(args.bc_samples), mmr_min=int(args.bc_mmr_min),
                mmr_boost=float(args.bc_mmr_boost),
                class_weight_overrides=(bc_class_weights or None))
            if len(bc_pool) == 0:
                raise ValueError("过滤后样本数为 0（动作白名单/MMR 下限）")
            _sum = bc_pool_summary(bc_pool)
            print(f"[bc] 真实人类 BC 池: samples={len(bc_pool)} "
                  f"mmr>={int(args.bc_mmr_min)} load={time.time() - _t_bc:.1f}s "
                  f"path={bc_pool.path}", flush=True)
            print(f"[bc] class_counts={json.dumps(_sum['class_counts'], ensure_ascii=False)}",
                  flush=True)
            print(f"[bc] class_fractions="
                  f"{json.dumps({k: round(v, 4) for k, v in _sum['class_fractions'].items()}, ensure_ascii=False)}",
                  flush=True)
            print(f"[bc] 分层目标: type_weight={float(args.bc_type_weight):g} "
                  f"card_weight={float(args.bc_card_weight):g} "
                  f"card_head={'on' if bc_card_head else 'off'} "
                  f"class_weights={json.dumps(bc_class_weights, ensure_ascii=False)} "
                  f"coef(dec=0)={bc_schedule.coef(0):.4g} "
                  f"ramp={json.dumps(bc_schedule.describe(), ensure_ascii=False)}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001 - BC 可选，失败自动关闭
            print(f"[warn] BC 数据加载失败，自动关闭 bc_coef=0（不中断训练）："
                  f"{type(exc).__name__}: {exc}", flush=True)
            bc_enabled = False
            bc_card_head = False
            args.bc_coef = 0.0
            args.bc_coef_ramp = ""
            bc_pool = None

    if not engine_available():
        print("[error] hsbg_engine 不可用", file=sys.stderr)
        return 2
    if not torch_available():
        print("[error] torch 不可用", file=sys.stderr)
        return 2
    import torch
    if args.minutes <= 0 or args.games <= 0 or args.iterations < 0:
        print("[error] minutes/games 必须 >0，iterations 必须 >=0", file=sys.stderr)
        return 2
    if args.workers <= 0 or args.batch_size <= 0:
        print("[error] workers/batch-size 必须 >0", file=sys.stderr)
        return 2
    # 信用分配（§B）安全边界：默认 gae 完全不进入新路径。
    if int(args.baseline_action_dependent) != 0:
        print("[error] --baseline-action-dependent 只允许 0：动作条件 Q(s,a) 只能进 "
              "critic/TD loss，不能直接当 actor baseline（有偏，§B.3）。"
              "若要做 b(s,a) 必须走双采样/显式加权修正并通过 tests/test_credit.py。",
              file=sys.stderr)
        return 2
    if str(args.adv_estimator) in ("power", "geom"):
        if not (float(args.power_alpha) >= 0.0):
            print("[error] --power-alpha 必须 >= 0", file=sys.stderr)
            return 2
        if not (float(args.power_kappa) > 0.0):
            print("[error] --power-kappa 必须 > 0", file=sys.stderr)
            return 2
        if int(args.power_cutoff) < 0:
            print("[error] --power-cutoff 必须 >= 0", file=sys.stderr)
            return 2
        print(f"[credit] adv_estimator={args.adv_estimator} "
              f"alpha={args.power_alpha:g} kappa={args.power_kappa:g} "
              f"normalize={int(args.power_normalize)} "
              f"cutoff={int(args.power_cutoff)} timebase={args.credit_timebase}",
              flush=True)
    elif str(args.credit_timebase) != "decision":
        print("[error] --credit-timebase turn 需要 --adv-estimator power|geom",
              file=sys.stderr)
        return 2
    if args.trainer_state and not os.path.isfile(args.trainer_state):
        print(f"[error] trainer-state 不存在: {args.trainer_state}", file=sys.stderr)
        return 2
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2 ** 32))
    os.makedirs(args.out, exist_ok=True)
    eng = engine_info()
    print(f"[engine] {json.dumps(eng, ensure_ascii=False)}", flush=True)

    norm_mean = norm_std = None
    if args.checkpoint:
        model, meta = load_checkpoint(args.checkpoint, device=args.device)
        arch = str(meta.get("arch", args.arch))
        norm_mean, norm_std = meta.get("norm_mean"), meta.get("norm_std")
        # --d-model/--n-layers/--n-heads 只对「从零构建」生效；这里只做一致性提示，
        # 避免「以为建了 0.667M，其实加载了 checkpoint 的 384×8」这类静默错配。
        _size_want = _net_size_kwargs(args)
        _size_have = {k: int(meta.get(k, 0) or 0) for k in _size_want}
        _size_mismatch = {k: (v, _size_have[k]) for k, v in _size_want.items()
                          if _size_have[k] != v}
        if _size_mismatch:
            print(f"[warn] --d-model/--n-layers/--n-heads 与 checkpoint meta 不一致"
                  f"（以 meta 为准）：{_size_mismatch}", flush=True)
        print(f"[net] checkpoint meta 尺寸="
              f"{ {k: meta.get(k) for k in ('d_model','n_layers','n_heads')} }",
              flush=True)
        print(f"[net] checkpoint={args.checkpoint} arch={arch}", flush=True)
        # 有向 Move 头：旧 ckpt 没有 move_from/move_to 时，只追加两个投影并继承其余权重
        _move_split_loaded = bool(getattr(model, "move_head_split", False))
        if int(args.move_head_split) and not _move_split_loaded:
            _p0 = sum(p.numel() for p in model.parameters())
            try:
                attach_move_head_split(model)
            except ValueError as exc:
                print(f"[error] --move-head-split 1 不可用: {exc}", file=sys.stderr)
                return 2
            _p1 = sum(p.numel() for p in model.parameters())
            print(f"[net] move_head_split=on（旧 checkpoint 无有向 Move 投影）："
                  f"已追加 move_from/move_to，参数 +{_p1 - _p0:,}"
                  f"（{_p0:,} -> {_p1:,}），其余权重继承", flush=True)
        elif _move_split_loaded:
            print("[net] move_head_split=on（checkpoint meta 已声明）", flush=True)
        if seq_enabled and getattr(model, "seq_head", None) is None:
            print("[warn] checkpoint 没有动作序列辅助头（旧 checkpoint）：本次关闭 seq 辅助任务",
                  flush=True)
            seq_enabled = False
        if bc_card_head and getattr(model, "card_head", None) is None:
            # 旧 checkpoint（如 champion_c0_1975）没有 card_head：加载后追加，
            # 随机初始化，随后由 BC 卡牌身份 CE 训练。
            try:
                attach_bc_card_head(model)
                print(f"[bc] checkpoint 无 card_head：已追加 "
                      f"Linear({bc_card_head_dim(model)}, 3983)", flush=True)
            except Exception as exc:  # noqa: BLE001 - 退化到只做类型级 BC
                print(f"[warn] 追加 card_head 失败（本次只用类型级 BC）：{exc!r}",
                      flush=True)
                bc_card_head = False
                args.bc_card_weight = 0.0
    elif args.trainer_state and os.path.isfile(args.trainer_state):
        # trainer.pt 的 model state 是 strict 加载：老 state 没有 card_head 时不能
        # 在加载前追加（会 unexpected key），因此 trainer-state 路径要求 state 已含该头。
        # 尺寸必须与 state 一致：--d-model/--n-layers/--n-heads 在这里**必须**生效
        # （否则会按默认 384×8 建网，667k 的 state 直接 size mismatch 报错）。
        _size = _net_size_kwargs(args)
        model = build_net(arch=args.arch, seq_head=seq_enabled,
                          bc_card_head=False, **_size)
        if _size:
            print(f"[net] trainer-state 重建 {args.arch} {_size}", flush=True)
        if int(args.move_head_split):
            print("[warn] --trainer-state 路径按 state 的架构加载；--move-head-split 1 "
                  "只对 --checkpoint（model.pt）生效，本次忽略", flush=True)
        meta = {"arch": args.arch, **_size}
        arch = args.arch
        print(f"[net] empty model for trainer-state={args.trainer_state}", flush=True)
    else:
        _size = _net_size_kwargs(args)
        model = build_net(arch=args.arch, seq_head=seq_enabled,
                          bc_card_head=bc_card_head,
                          move_head_split=bool(int(args.move_head_split)),
                          # 2026-11-17（D-072）：三项"借鉴结构"以前**没有 CLI 入口**，
                          # 等于默认关且永远打不开。默认全关 ⇒ 逐位等价旧模型。
                          use_z=bool(int(getattr(args, "use_z", 0) or 0)),
                          n_z=int(getattr(args, "n_z", 46) or 46),
                          use_attr=bool(int(getattr(args, "use_attr", 0) or 0)),
                          attr_f=int(getattr(args, "attr_f", 311) or 311),
                          mag_query=bool(int(getattr(args, "mag_query", 0) or 0)),
                          **_size)
        meta = {"arch": args.arch, **_size}
        arch = args.arch
        if _size:
            print(f"[net] 从零构建 {args.arch} {_size}", flush=True)
    if bc_enabled and bc_card_head and getattr(model, "card_head", None) is None:
        print("[warn] 模型没有 card_head（trainer-state/旧 ckpt）：本次只用类型级 BC "
              "(bc_card_weight=0)", flush=True)
        bc_card_head = False
        args.bc_card_weight = 0.0
    if float(args.process_value_coef) > 0.0:
        if getattr(model, "process_head", None) is None:
            if args.trainer_state:
                print("[warn] --trainer-state 路径的 state 不含 process_head："
                      "本次关闭 process_value_coef=0（请在 --checkpoint 路径追加头）",
                      flush=True)
                args.process_value_coef = 0.0
            else:
                try:
                    attach_process_head(model)
                    print(f"[process] checkpoint 无 process_head：已追加 "
                          f"Linear({bc_card_head_dim(model)}, 3)（log1p 酒馆永久 buff / "
                          f"施法数 / consume 数）", flush=True)
                except Exception as exc:  # noqa: BLE001 - 非 entity/测试替身
                    print(f"[warn] 追加 process_head 失败（本次关闭 "
                          f"process_value_coef=0）：{exc!r}", flush=True)
                    args.process_value_coef = 0.0
        else:
            print("[process] checkpoint 已带 process_head（沿用权重）", flush=True)
    model.to(torch.device(args.device))
    print(f"[net] arch={arch} params={sum(p.numel() for p in model.parameters()):,}",
          flush=True)

    # ---- 场面题池 + 强度辅助头（默认关闭 = 不读文件、不建头、零影响）----------
    # 题目池里同时有「人类参考动作 top-k 软标签」与「t+3 战团强度/tier+1 标签」：
    # 题目 CE 与强度辅助头共用同一批题目 + 同一次 forward_problem（B/C/D 三臂可比）。
    problem_pool = None
    if _board_enabled:
        try:
            from hsbg.az.train.board_problems import load_problem_pool
            _t_prob = time.time()
            problem_pool = load_problem_pool(
                str(args.problem_pool),
                min_strength_delta=float(args.problem_min_strength_delta),
                turn_weights=str(args.problem_turn_weight or ""))
            if len(problem_pool) == 0:
                raise ValueError(f"题目池为空/不存在: {args.problem_pool}")
            _ps = problem_pool.summary()
            print(f"[problems] pool={problem_pool.path} n={_ps['problems']} "
                  f"bc_valid={_ps['bc_valid']}({_ps['bc_valid_rate']:.1%}) "
                  f"aux_valid={_ps['aux_valid']}({_ps['aux_valid_rate']:.1%}) "
                  f"turns={_ps['by_turn']} "
                  f"load={time.time() - _t_prob:.1f}s", flush=True)
            _pfil = dict(_ps.get("filter") or {})
            print(f"[problems] filter: pool_size={_pfil.get('problem_pool_size')} "
                  f"used={_pfil.get('problem_pool_used')} "
                  f"dropped_by_delta={_pfil.get('problem_pool_dropped_by_delta')} "
                  f"dropped_by_turn={_pfil.get('problem_pool_dropped_by_turn')} "
                  f"min_strength_delta={float(_pmd):g} "
                  f"turn_weight={format_turn_weights(problem_turn_weights) or '-'} "
                  f"used_turn_weight="
                  f"{format_turn_weights(problem_pool.turn_weights) or '-'} "
                  f"late_turn(>= {_ps.get('late_turn_min')})="
                  f"{_ps.get('late_turn_problems')} "
                  f"delta_p50={_ps.get('strength_delta_p50', 0.0):.3f}",
                  flush=True)
            print(f"[problems] by_tag={json.dumps(_ps['by_tag'], ensure_ascii=False)} "
                  f"tier_up_rate={_ps['tier_up_rate']:.3f} "
                  f"strength_p50 before={_ps['strength_before_p50']:.0f} "
                  f"after={_ps['strength_after_p50']:.0f} "
                  f"norm={json.dumps(_ps['norm'], ensure_ascii=False)}", flush=True)
            if _problem_enabled and int(_ps["bc_valid"]) == 0:
                print("[warn] 题目池没有任何合法参考解（bc_valid=0）："
                      "自动关闭 problem_bc_coef=0", flush=True)
                args.problem_bc_coef = 0.0
                _problem_enabled = False
            if _strength_enabled and int(_ps["aux_valid"]) == 0:
                print("[warn] 题目池没有任何 t+3 强度标注（aux_valid=0）："
                      "自动关闭 strength_aux_coef=0", flush=True)
                args.strength_aux_coef = 0.0
                _strength_enabled = False
        except Exception as exc:  # noqa: BLE001 - 可选辅助，失败不中断训练
            print(f"[warn] 场面题池加载失败，自动关闭 problem_bc_coef/"
                  f"strength_aux_coef（不中断训练）：{type(exc).__name__}: {exc}",
                  flush=True)
            args.problem_bc_coef = 0.0
            args.strength_aux_coef = 0.0
            _problem_enabled = _strength_enabled = False
            problem_pool = None
    if _strength_enabled and problem_pool is not None:
        _snorm = dict(problem_pool.norm or {"mean": 0.0, "std": 1.0})
        if getattr(model, "strength_head", None) is None:
            try:
                attach_strength_head(model, norm=_snorm)
                print(f"[strength] checkpoint 无 strength_head：已追加 "
                      f"Linear({bc_card_head_dim(model)}, 2)（t+3 log1p 强度 + tier+1），"
                      f"norm={json.dumps(_snorm, ensure_ascii=False)}", flush=True)
            except Exception as exc:  # noqa: BLE001 - 非 entity/测试替身
                print(f"[warn] 追加 strength_head 失败（本次关闭 strength_aux_coef=0）："
                      f"{exc!r}", flush=True)
                args.strength_aux_coef = 0.0
                _strength_enabled = False
        else:
            model.strength_norm = _snorm
            print(f"[strength] checkpoint 已带 strength_head（沿用权重）；"
                  f"norm={json.dumps(_snorm, ensure_ascii=False)}", flush=True)
        if str(args.strength_head_init) == "fit":
            from hsbg.az.train.board_problems import fit_strength_head
            try:
                _rep = fit_strength_head(
                    model, problem_pool, ridge=float(args.strength_head_ridge),
                    device=str(args.device))
                print(f"[strength] 离线拟合线性头（题目池 aux_valid="
                      f"{_rep['n']}）：mse={_rep['mse']:.4f} bce={_rep['bce']:.4f} "
                      f"tier_acc={_rep['tier_acc']:.3f} "
                      f"cv_mse={_rep.get('cv_mse', float('nan')):.4f} "
                      f"pearson(增量)={_rep['pearson_delta']:.3f} "
                      f"ridge={_rep['ridge']:g} frozen={_rep['frozen']}", flush=True)
            except Exception as exc:  # noqa: BLE001 - 拟合失败保留随机头
                print(f"[warn] 强度头离线拟合失败（保留随机初始化）："
                      f"{type(exc).__name__}: {exc}", flush=True)
    if (_problem_enabled or _strength_enabled):
        print(f"[problems] problem_bc_coef={float(args.problem_bc_coef):g} "
              f"every={int(args.problem_bc_every)} "
              f"turn_weight={format_turn_weights(problem_turn_weights) or '-'} "
              f"min_strength_delta={float(args.problem_min_strength_delta):g} "
              f"strength_aux_coef={float(args.strength_aux_coef):g} "
              f"every={int(args.strength_aux_every)} "
              f"batch={int(args.problem_batch)} "
              f"tier_pos_weight={float(args.strength_tier_pos_weight):g}", flush=True)

    # ---- upgrade 子集 DPO：预加载池 + 启动诊断（默认关闭 = 零影响）------------
    # 池 ~3050 条 / ~7s；只在 --dpo-coef>0 时加载。失败自动关闭（不中断训练）。
    dpo_pool: Optional[UpgradeSamples] = None
    if float(args.dpo_coef) > 0.0:
        try:
            _t_dpo = time.time()
            dpo_pool = load_upgrade_samples(str(args.dpo_samples))
            if len(dpo_pool) == 0:
                raise ValueError("过滤后样本数为 0（动作白名单/MMR 下限）")
            print(f"[dpo] upgrade 池: samples={len(dpo_pool)} "
                  f"mmr>={int(dpo_pool.mmr_min)} "
                  f"load={time.time() - _t_dpo:.1f}s path={dpo_pool.path}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001 - DPO 可选，失败自动关闭
            print(f"[warn] upgrade DPO 样本加载失败，自动关闭 dpo_coef=0"
                  f"（不中断训练）：{type(exc).__name__}: {exc}", flush=True)
            args.dpo_coef = 0.0
            dpo_pool = None
    if dpo_pool is not None and len(dpo_pool) > 0:
        # 启动诊断：当前策略在池上的 DPO 初值 / 接受率（no_grad，不改权重）
        _p = _dpo_startup_probe(model, dpo_pool, beta=float(args.dpo_beta),
                                device=str(args.device))
        _term = float(args.dpo_coef) * float(_p["loss"])
        _avg = _term / float(max(1, int(args.dpo_every)))
        _ref, _ref_src = _ppo_total_ref(str(args.out))
        print(f"[dpo] beta={float(args.dpo_beta):g} coef={float(args.dpo_coef):g} "
              f"batch={int(args.dpo_batch)} every={int(args.dpo_every)} "
              f"（每 {int(args.dpo_every)} 个 iteration 加一次 coef*L_dpo）", flush=True)
        print(f"[dpo] 当前策略 on 池: dpo_loss={_p['loss']:.4f} "
              f"accept_rate={_p['accept_rate']:.3f}（n={int(_p['n'])}）"
              f" -> 单步项={_term:.5f}，平均/iteration={_avg:.5f}", flush=True)
        print(f"[dpo] 期望梯度占比（loss 标度参考）: 单步 DPO 项 / |PPO total| ≈ "
              f"{_term / max(abs(_ref), 1e-9) * 100:.2f}%（参考 total={_ref:.4f}，"
              f"来自{_ref_src}）", flush=True)
        _gref = _dpo_grad_ref()
        if _gref is not None:
            _gc = float(args.dpo_coef) * float(_gref["share_per_coef"])
            print(f"[dpo] 期望梯度占比（实测 ‖g_dpo‖/‖g_ppo‖，"
                  f"来自 {_gref['path']}）: 单步≈{_gc * 100:.2f}%，"
                  f"平均/iteration≈{_gc / max(1, int(args.dpo_every)) * 100:.2f}%",
                  flush=True)
        else:
            print("[dpo] 期望梯度占比：暂无实测梯度范数比（可跑 "
                  "workflows/dpo_grad_probe.py 生成 runs/dpo_grad_probe.json）",
                  flush=True)

    p_cfg = PPOConfig(
        epochs=int(args.epochs), minibatch=int(args.minibatch), lr=float(args.lr),
        clip=float(args.clip), gamma=float(args.gamma), lam=float(args.lam),
        target_kl=float(args.target_kl),
        rank_utility_coef=float(args.rank_utility_coef),
        terminal_strength_coef=float(args.terminal_strength_coef),
        adv_estimator=str(args.adv_estimator),
        power_alpha=float(args.power_alpha),
        power_kappa=float(args.power_kappa),
        power_normalize=bool(int(args.power_normalize)),
        power_cutoff=int(args.power_cutoff),
        credit_timebase=str(args.credit_timebase),
        baseline_action_dependent=int(args.baseline_action_dependent),
        entropy_coef=float(args.entropy_coef), pi_coef=float(args.pi_coef),
        max_grad_norm=float(args.max_grad_norm),
        grad_accum=int(args.grad_accum), amp=str(args.amp),
        device=str(args.device), seed=int(args.seed),
        log_every=int(args.log_every),
        compile_train=int(args.compile_train),
        anchor_amp=bool(int(args.anchor_amp)),
        # 动作序列辅助任务：初始系数由 ramp 在 dec=0 处给出（无 ramp 时=--seq-coef）
        seq_coef=(seq_schedule.coef(0) if seq_enabled else 0.0),
        seq_coef_ramp=(str(args.seq_coef_ramp) if seq_enabled else ""),
        # 真实人类 BC 辅助正则（分层目标）：初始系数由 ramp 在 dec=0 处给出；
        # 池在下方注入 trainer。
        bc_coef=(bc_schedule.coef(0) if bc_enabled else 0.0),
        bc_coef_ramp=(str(args.bc_coef_ramp) if bc_enabled else ""),
        bc_mmr_min=int(args.bc_mmr_min),
        bc_mmr_boost=float(args.bc_mmr_boost),
        bc_class_weights=dict(bc_class_weights),
        bc_samples_path=str(args.bc_samples),
        bc_batch=int(args.bc_batch),
        bc_every=int(args.bc_every),
        bc_compile=int(args.bc_compile),
        bc_type_weight=float(args.bc_type_weight),
        bc_card_weight=float(args.bc_card_weight),
        # upgrade 子集 one-sided DPO：系数固定（无 ramp）；池在下方注入 trainer
        dpo_coef=float(args.dpo_coef),
        dpo_beta=float(args.dpo_beta),
        dpo_batch=int(args.dpo_batch),
        dpo_every=int(args.dpo_every),
        dpo_samples=str(args.dpo_samples),
        # 场面题 soft-label CE + 强度辅助头（默认关闭；池在下方注入 trainer）
        problem_bc_coef=float(args.problem_bc_coef),
        problem_bc_every=int(args.problem_bc_every),
        problem_batch=int(args.problem_batch),
        problem_pool_path=str(args.problem_pool),
        problem_turn_weight=str(args.problem_turn_weight or ""),
        problem_min_strength_delta=float(args.problem_min_strength_delta),
        strength_aux_coef=float(args.strength_aux_coef),
        strength_aux_every=int(args.strength_aux_every),
        strength_tier_pos_weight=float(args.strength_tier_pos_weight),
        # 缩放教师 DAgger / 过程价值（默认 0 = 逐位等价）
        # 强制升本的 DAgger 标签**单独记账**：两条标签源共用 record["teacher_action"]，
        # 若把它们合并进 teacher_bc_coef，"在线缩放教师"的开关会被 forced 系数悄悄打开，
        # history 也再分不清标签来源（docs/FORCED_TIER.md §标签源）。
        teacher_bc_coef=float(args.teacher_bc_coef),
        forced_teacher_coef=float(args.forced_teacher_coef),
        process_value_coef=float(args.process_value_coef),
        # 搜索行为混合（默认 eps=0 -> PPO 侧只用于回显/统计，未启用时整组键被移除）
        behavior_mix_eps=float(args.behavior_mix_eps),
        behavior_mix_when=str(args.behavior_mix_when),
        behavior_mix_max_per_turn=int(args.behavior_mix_max_per_turn),
        behavior_mix_topk=int(args.behavior_mix_topk),
        behavior_mix_turns=int(args.behavior_mix_turns),
        behavior_mix_max_ms=float(args.behavior_mix_max_ms),
        behavior_mix_exact=int(args.behavior_mix_exact))
    trainer: Optional[PPOTrainer] = None
    if args.trainer_state and os.path.isfile(args.trainer_state):
        benchmark_load = os.environ.get("HSRL_PPO_BENCHMARK_LOAD")
        trainer = PPOTrainer.load(model, args.trainer_state, config=p_cfg,
                                  device=args.device, bc_pool=bc_pool,
                                  dpo_pool=dpo_pool,
                                  _benchmark_legacy_device_load=benchmark_load == "legacy")
        if benchmark_load:
            from collections import Counter
            devices = Counter(step.device.type for state in trainer.optimizer.state.values()
                              if (step := state.get("step")) is not None)
            from workflows.run_learning_experiment import write
            write(Path(args.out) / "adam-step-devices.json", dict(devices))
        print(f"[trainer] loaded updates={trainer.updates} "
              f"lr={trainer.optimizer.param_groups[0]['lr']:.3g}", flush=True)
        # 续训时 lr 调度参数以 **trainer state** 为准（load_state_dict 会覆盖
        # lr_min_ratio / max_updates，LambdaLR 的闭包读的是实例属性），命令行/配置
        # 里的值不生效。历史复盘里这一点被当成"配置了 2e-6 却跑 1.75e-6"的疑惑，
        # 这里显式打印并对不一致给出警告（不改变行为）。
        _eff_lr = float(trainer.optimizer.param_groups[0]['lr'])
        _cfg_lr = float(getattr(args, "lr", 0.0) or 0.0)
        _cfg_ratio = float(getattr(args, "lr_min_ratio", 0.0) or 0.0)
        _cfg_max = int(getattr(args, "max_updates", 0) or 0)
        print(f"[lr] base={trainer.scheduler.base_lrs} last_epoch={trainer.scheduler.last_epoch} "
              f"lr_min_ratio={trainer.lr_min_ratio:g} max_updates={trainer.max_updates} "
              f"-> effective={_eff_lr:.10g} (config lr={_cfg_lr:g} "
              f"ratio={_cfg_ratio:g} max_updates={_cfg_max})", flush=True)
        if (abs(float(trainer.lr_min_ratio) - _cfg_ratio) > 1e-9
                or int(trainer.max_updates) != _cfg_max):
            print("[lr][warn] 续训沿用 trainer state 的 lr_min_ratio/max_updates，"
                  "命令行值被忽略；如需改退火曲线必须先 fork 新 trainer state", flush=True)
    else:
        trainer = PPOTrainer(model, p_cfg, lr_min_ratio=float(args.lr_min_ratio),
                             max_updates=int(args.max_updates), bc_pool=bc_pool,
                             dpo_pool=dpo_pool)

    # BC 池注入（CLI 已预加载/打印；trainer 若自己加载过则复用）
    if bc_enabled:
        if bc_pool is not None:
            trainer.set_bc_pool(bc_pool)
        elif getattr(trainer, "bc_pool", None) is not None:
            bc_pool = trainer.bc_pool
        if getattr(trainer, "bc_pool", None) is None:
            print("[warn] BC 池不可用，自动关闭 bc_coef=0", flush=True)
            bc_enabled = False
            trainer.config.bc_coef = 0.0
            trainer.config.bc_coef_ramp = ""

    # upgrade DPO 池注入（CLI 已预加载/打印；trainer 若自己加载过则复用）
    dpo_enabled = bool(float(args.dpo_coef) > 0.0)
    if dpo_enabled:
        if dpo_pool is not None:
            trainer.set_dpo_pool(dpo_pool)
        elif getattr(trainer, "dpo_pool", None) is not None:
            dpo_pool = trainer.dpo_pool
            print(f"[dpo] 复用 trainer 已加载的 upgrade 池: samples={len(dpo_pool)}",
                  flush=True)
        if getattr(trainer, "dpo_pool", None) is None:
            print("[warn] upgrade DPO 池不可用，自动关闭 dpo_coef=0", flush=True)
            dpo_enabled = False
            trainer.config.dpo_coef = 0.0

    # 场面题池注入（CLI 已预加载/打印；未启用时为 None，trainer 不会自己加载）
    if _board_enabled and problem_pool is not None and not args.trainer_state:
        trainer.set_problem_pool(problem_pool)

    # ---- 日志：history.jsonl 追加（兼容旧 history.json）----
    history: List[Dict[str, Any]] = []
    history_path = os.path.join(args.out, "history.jsonl")
    legacy_history = os.path.join(args.out, "history.json")
    if args.trainer_state:
        if os.path.isfile(history_path):
            _, history = read_run(history_path)
        elif os.path.isfile(legacy_history):
            _, history = read_run(legacy_history)
    write_header(history_path, {k: v for k, v in vars(args).items()})

    if trainer.extra.get("norm_mean") is not None:
        norm_mean = np.asarray(trainer.extra["norm_mean"], dtype=np.float32)
    if trainer.extra.get("norm_std") is not None:
        norm_std = np.asarray(trainer.extra["norm_std"], dtype=np.float32)
    trainer.extra["norm_mean"] = (None if norm_mean is None else norm_mean.tolist())
    trainer.extra["norm_std"] = (None if norm_std is None else norm_std.tolist())
    start_i = int(trainer.extra.get("next_rollout_index", trainer.updates))
    print(f"[resume] start_iter={start_i} updates={trainer.updates}", flush=True)

    # ---- KL anchor（AutoTavern Table 7：β=0.1，仅前 100 个 iteration）----
    # ``--reset-anchor 1``：在 fork/续训时把 anchor 重建为**当前**策略。
    # 动机（docs/research/LONG1085131_REVIEW.md §4.3）：``load_state_dict`` 优先恢复
    # state 里的 anchor_model，本轮实测它是更早阶段的旧策略（与 7357 L2=9.00），
    # 于是「拉回 stage 起点」实际变成「拉回一个更弱的老策略」。
    if int(getattr(args, "reset_anchor", 0)) and trainer.anchor_model is not None:
        trainer.set_anchor()
        print("[anchor] 已按 --reset-anchor 重建为当前策略（fork 起点）", flush=True)
    if float(args.anchor_beta) > 0.0 and trainer.anchor_model is None:
        trainer.set_anchor()
        print(f"[anchor] beta={args.anchor_beta} for first "
              f"{int(args.anchor_iterations)} iters (stage-start policy frozen)",
              flush=True)
    if int(args.compile_train) > 0:
        tf = getattr(trainer, "train_forward", None)
        if tf is None:
            print("[compile] train compile 请求但未生效（回退 eager）", flush=True)
        else:
            print(f"[compile] train compile={int(args.compile_train)} "
                  f"active={tf.active} main={tf.main_compiled} aux={tf.aux_compiled} "
                  f"warmup={tf.warmup_seconds:.1f}s"
                  + (f" warnings={tf.warnings}" if tf.warnings else ""), flush=True)
        af = getattr(trainer, "anchor_forward", None)
        if af is not None:
            print(f"[compile] anchor compile active={af.active} "
                  f"warmup={af.warmup_seconds:.1f}s", flush=True)

    # ---- 周期 C0 评测（监控曲线；avg rank<=3.0 触发镜像库）----
    # 同步（--eval-async 0，旧行为）在主循环里阻塞；异步（1）把 ckpt 丢给独立子进程，
    # 由轻量线程回收结果。两条路径共用 _post_eval 做 TRIGGER / 早停判定。
    evals: list = []
    eval_dir = os.path.join(args.out, "eval")
    eval_async = bool(int(args.eval_async))
    eval_device = str(args.eval_device or ("cuda:4" if eval_async else args.device))
    # panel 列表：bot=旧 c0_gate 面板；human=人类战团面板；both=两个子进程串行排队。
    eval_panels = panels_from_spec(args.eval_panel)
    eval_human_boards = str(args.eval_human_boards or os.path.join(
        ROOT, "data", "real_library", "library_real_degraded.jsonl"))
    if int(args.eval_every_decisions) > 0 and "human" in eval_panels and not os.path.isfile(eval_human_boards):
        print(f"[error] human 评测面板需要人类战团数据: {eval_human_boards}",
              file=sys.stderr)
        return 2
    eval_stop = threading.Event()
    eval_state: Dict[str, Any] = {"stop_reason": ""}
    eval_history_lock = threading.Lock()
    #: 最近一次各面板结果（写 history.jsonl 的 eval 段；bot -> mean_rank，
    #: human -> human_mean_rank，见 docs/EVAL_PANELS.md）
    latest_eval: Dict[str, Dict[str, Any]] = {}

    def _save_eval_ckpt(decisions_done: int) -> Tuple[str, str]:
        """落一份评测专用 ckpt（与训练权重解耦，子进程只读它）。"""
        os.makedirs(eval_dir, exist_ok=True)
        tag = f"{int(decisions_done):09d}"
        ckpt = os.path.join(eval_dir, f"ckpt_{tag}.pt")
        save_checkpoint(model, ckpt, meta={**meta, "arch": arch},
                        norm_mean=norm_mean, norm_std=norm_std)
        return tag, ckpt

    def _dyn_step(decisions_done: int) -> Optional[Dict[str, Any]]:
        """动态课程一步：跑**真实强度口径**面板（5 席 × p0.9，96 局），按 t15 Σ 加减席。

        规则（Lead 定，可配置）：起点 `--dyn-start-seats` 席 × `--dyn-p`；
        - 若 `S15 >= last_S15 × (1 + +threshold)` 且 `seats < --dyn-max-seats` ⇒ **加一席**；
        - 若 `seats > start` 且 `S15 < last_S15 × (1 - threshold)` ⇒ **退一席**，
          并保持 `--dyn-hold-decisions` 不再加席；
        - 主判据用真实口径，训练剂量只作参考。
        结果写 `out/dynamic_curriculum.json`，并进 `history.jsonl` 的 `ppo.dyn_curriculum`。
        """
        tag, ckpt = _save_eval_ckpt(decisions_done)
        out_json = os.path.join(eval_dir, f"dyn_quality_{tag}.json")
        cmd = [sys.executable, os.path.join(ROOT, "workflows", "quality_monitor.py"),
               "--checkpoint", str(ckpt), "--arch", str(arch),
               "--device", str(args.eval_device or args.device),
               "--games", str(int(args.dyn_quality_games)), "--seed-base", "520000",
               "--human-boards", str(args.human_boards),
               "--human-board-p", "0.9", "--human-board-seats", "5",
               "--human-board-fidelity", "full", "--human-board-sample", "auto",
               "--human-board-set-tier", "1", "--human-board-fast-inject", "1",
               "--move-guard", "1", "--move-cap", "2", "--move-repeat-limit", "2",
               "--out", str(out_json)]
        try:
            _t = time.time()
            subprocess.run(cmd, check=False, timeout=1800)
            with open(out_json, encoding="utf-8") as f:
                q = json.load(f)
            s15 = float((q.get("infer_search") or {}).get("strength_t15_p50") or 0.0)
        except BaseException as exc:  # noqa: BLE001 - 质量面板失败不能中断训练
            print(f"[dyn] 真实强度面板失败（本点跳过）：{exc!r}", flush=True)
            return None
        last = _dyn["last_s15"]
        act = "hold"
        thr = float(args.dyn_threshold)
        if last is not None and s15 >= last * (1.0 + thr) \
                and _dyn["seats"] < int(args.dyn_max_seats) \
                and cum_decisions >= _dyn["hold_until"]:
            _dyn["seats"] += 1
            act = "up"
        elif last is not None and s15 < last * (1.0 - thr) \
                and _dyn["seats"] > int(args.dyn_start_seats):
            _dyn["seats"] -= 1
            _dyn["hold_until"] = cum_decisions + float(args.dyn_hold_decisions)
            act = "down"
        rec = {"decisions": int(decisions_done), "s15_real": s15,
               "last_s15_real": last, "seats": int(_dyn["seats"]),
               "action": act, "seconds": round(time.time() - _t, 1),
               "quality_json": out_json,
               "tier_t15": q.get("tier_p50_t15"),
               "death_turn_p50": q.get("death_turn_p50"),
               "survive_t12": q.get("survive_t12_ratio"),
               "mean_rank": q.get("mean_rank")}
        _dyn["last_s15"] = s15
        _dyn["history"].append(rec)
        try:
            write_json_atomic(os.path.join(args.out, "dynamic_curriculum.json"),
                              {"seats": int(_dyn["seats"]), "p": float(args.dyn_p),
                               "history": _dyn["history"]})
        except BaseException:  # noqa: BLE001
            pass
        print(f"[dyn] dec={decisions_done} S15_real={s15:.0f} (last={last}) "
              f"seats={_dyn['seats']} action={act} tier_t15={rec['tier_t15']} "
              f"death={rec['death_turn_p50']}", flush=True)
        return rec

    def _post_eval(rec: Dict[str, Any]) -> None:
        """评测结果后处理（同步/异步共用，异步时在 monitor 线程执行）。"""
        panel = str(rec.get("panel") or "bot")
        has = (rec.get("human_mean_rank") is not None if panel == "human"
               else rec.get("mean_rank") is not None)
        with eval_history_lock:
            if has:
                latest_eval[panel] = dict(rec)
        # TRIGGER / 早停只按 bot 面板（旧语义），human 面板只作主指标记录。
        mr = rec.get("mean_rank") if panel == "bot" else None
        if mr is None:
            return
        mr = float(mr)
        if mr <= 3.0:
            trig = os.path.join(args.out, "TRIGGER_library.json")
            write_json_atomic(trig, {"decisions": int(rec["decisions"]),
                                     "mean_rank": mr,
                                     "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            print(f"[TRIGGER] avg rank {mr:.3f} <= 3.0 -> "
                  f"构建高质量对局镜像库 ({trig})", flush=True)
        th = float(args.eval_stop_mean_rank)
        if th > 0.0 and mr <= th:
            eval_state["stop_reason"] = (f"mean_rank={mr:.4f} <= {th:.4f} "
                                         f"@dec={int(rec['decisions'])}")
            eval_stop.set()
            print(f"[eval] stop requested: {eval_state['stop_reason']} "
                  f"-> 下一个 iteration 边界停止", flush=True)

    def _eval_snapshot() -> Dict[str, Any]:
        """history.jsonl 的 ``eval`` 段：bot 面板 ``mean_rank`` + human 面板
        ``human_mean_rank``（两个键并存，方便画图脚本直接读）。"""
        with eval_history_lock:
            bot = dict(latest_eval.get("bot") or {})
            human = dict(latest_eval.get("human") or {})
            if not bot and not human:
                return {}
            return {
                "mean_rank": bot.get("mean_rank"),
                "human_mean_rank": human.get("human_mean_rank"),
                "bot_mean_rank": bot.get("bot_mean_rank"),
                "top4": bot.get("top4"),
                "win": bot.get("win"),
                "human_top4": human.get("human_top4"),
                "human_win": human.get("human_win"),
                "human_quantile": human.get("quantile"),
                "bot_decisions": bot.get("decisions"),
                "human_decisions": human.get("decisions"),
            }

    def _append_eval_history(rec: Dict[str, Any]) -> None:
        with eval_history_lock:
            evals.append(rec)
            write_json_atomic(os.path.join(eval_dir, "history.json"), {"evals": evals})

    def _run_eval(decisions_done: int) -> None:
        """同步评测：主循环阻塞到各 panel 的子进程依次结束。"""
        tag, ckpt = _save_eval_ckpt(decisions_done)
        for panel in eval_panels:
            prefix = "c0" if panel == "bot" else "human"
            out_json = os.path.join(eval_dir, f"{prefix}_{tag}.json")
            log_path = os.path.join(eval_dir, f"{prefix}_{tag}.log")
            if panel == "bot":
                cmd = build_c0_cmd(
                    sys.executable, ROOT, ckpt, arch=str(arch), device=eval_device,
                    games=int(args.eval_games), seats=int(args.eval_seats),
                    seed_base=int(args.eval_seed_base), out_json=out_json)
            else:
                cmd = build_human_cmd(
                    sys.executable, ROOT, ckpt, arch=str(arch), device=eval_device,
                    games=int(args.eval_games), seats=int(args.eval_seats),
                    seed_base=int(args.eval_seed_base), out_json=out_json,
                    quantile=str(args.eval_human_quantile),
                    human_boards=eval_human_boards,
                    baseline_bot=bool(int(args.eval_human_baseline_bot)))
            t_eval = time.time()
            print(f"[eval] dec={int(decisions_done)} panel={panel} "
                  f"start games={args.eval_games}", flush=True)
            with open(log_path, "w", encoding="utf-8") as lf:
                rc = subprocess.call(cmd, cwd=ROOT, stdout=lf, stderr=lf)
            rec = parse_panel_result(decisions_done, rc, time.time() - t_eval,
                                     out_json, panel=panel)
            _append_eval_history(rec)
            if panel == "human" and rec.get("human_mean_rank") is not None:
                print(f"[eval] dec={int(decisions_done)} panel=human "
                      f"human_mean_rank={rec['human_mean_rank']:.3f} "
                      f"top4={rec['human_top4']:.3f} win={rec['human_win']:.3f} "
                      f"({rec['seconds']}s)", flush=True)
            elif panel == "bot" and rec.get("mean_rank") is not None:
                print(f"[eval] dec={int(decisions_done)} panel=bot "
                      f"mean_rank={rec['mean_rank']:.3f} "
                      f"bot={rec['bot_mean_rank']:.3f} top4={rec['top4']:.3f} "
                      f"win={rec['win']:.3f} ({rec['seconds']}s)", flush=True)
            else:
                print(f"[eval] dec={int(decisions_done)} panel={panel} FAILED rc={rc} "
                      f"error={rec.get('error')} log={log_path}", flush=True)
            _post_eval(rec)

    # 异步评测：独立子进程 + 一个轻量回收线程（并发上限 1，提交本身只花 ~0.3s 存 ckpt）
    eval_runner: Optional[AsyncEvalRunner] = None
    if eval_async:
        eval_runner = AsyncEvalRunner(
            out_dir=args.out, root=ROOT, arch=arch, device=eval_device,
            games=int(args.eval_games), seats=int(args.eval_seats),
            seed_base=int(args.eval_seed_base),
            panels=eval_panels,
            human_quantile=str(args.eval_human_quantile),
            human_boards=eval_human_boards,
            baseline_bot=bool(int(args.eval_human_baseline_bot)),
            on_result=_post_eval)
        print(f"[eval-async] on：device={eval_device} games={args.eval_games} "
              f"seats={args.eval_seats} panels={eval_panels} "
              f"human_quantile={args.eval_human_quantile}；主循环不等待"
              f"（并发上限 1，panel 串行排队，drain={int(args.eval_drain)}）",
              flush=True)
        if int(args.eval_every_decisions) <= 0:
            print("[warn] --eval-async 1 但 --eval-every-decisions=0：不会触发评测",
                  flush=True)
        if str(eval_device).startswith("cuda") and int(args.gpu_free_check):
            try:
                _eidx = int(str(eval_device).split(":")[1]) if ":" in str(eval_device) else 0
            except ValueError:
                _eidx = None
            _used = _gpu_used_mib()
            if _eidx is not None and _used.get(_eidx, 0) > int(args.gpu_free_mib):
                print(f"[warn] 评测卡 {eval_device} 已被占用 "
                      f"{_used[_eidx]}MiB（>{int(args.gpu_free_mib)}MiB）：评测子进程可能 "
                      f"OOM/变慢，建议换 --eval-device（如 cuda:6/7）", flush=True)

    def _submit_eval(decisions_done: int) -> None:
        """异步提交：只保存 ckpt + spawn，不等待（跳过时 runner 自己记 eval_skipped）。

        已有一个评测在跑时**不落 61MB ckpt**：直接让 runner 记一条 skipped。
        """
        if eval_runner.running():
            eval_runner.submit(decisions_done, "")
            return
        _tag, ckpt = _save_eval_ckpt(decisions_done)
        eval_runner.submit(decisions_done, ckpt)

    search_fn = search_cfg = None
    if int(args.sims) > 0:
        try:
            from hsbg.az import SearchConfig, search as az_search
        except Exception as exc:  # noqa: BLE001
            print(f"[error] search unavailable: {exc!r}", file=sys.stderr)
            return 2
        search_cfg = SearchConfig(simulations=int(args.sims),
                                  determinizations=int(args.determinizations),
                                  seed=int(args.seed))
        search_fn = az_search

    # ---- actor pool / 单进程采集 ----
    pool: Optional[ProcessActorPool] = None
    pool_ckpt = os.path.join(args.out, "_pool_weights.pt")
    if int(args.processes) > 1:
        if not str(args.gpus).strip():
            print("[error] --processes>1 需要 --gpus（按gpu-index-space解释编号）", file=sys.stderr)
            return 2
        from hsbg.az.train.device_assignment import actor_indices
        gpus = actor_indices(args.gpus, args.gpu_index_space,
            visible_count=torch.cuda.device_count(), visibility=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
            cuda=torch.cuda.is_available())
        # 邻居抢占检测：显存已用 > 阈值（默认 1 GiB）的卡从 actor 列表剔除并 warn。
        # 失败（nvidia-smi 不可用）时不做剔除，按用户给的列表跑。
        # nvidia-smi默认列物理编号，不能用来过滤Slurm逻辑卡或MIG实例。
        if int(args.gpu_free_check) and args.gpu_index_space == 'physical':
            used = _gpu_used_mib()
            if used:
                threshold = int(args.gpu_free_mib)
                busy = [g for g in gpus if used.get(g, 0) > threshold]
                if busy:
                    detail = ", ".join(f"cuda:{g}={used.get(g)}MiB" for g in busy)
                    warn = (f"[warn] GPU 已被占用（>{threshold}MiB，疑似邻居进程）：{detail}"
                            f" -> 从 actor 列表剔除")
                    print(warn, flush=True)
                    gpus = [g for g in gpus if g not in busy]
                if not gpus:
                    print("[error] --gpu-free-check 后没有可用 GPU（可用 --gpu-free-check 0 "
                          "跳过检查）", file=sys.stderr)
                    return 2
        per_gpu = {g: sum(1 for k in range(int(args.processes)) if gpus[k % len(gpus)] == g)
                   for g in gpus}
        print(f"[pool] processes={int(args.processes)} gpus={gpus} "
              f"workers/gpu={per_gpu}", flush=True)
        if len(gpus) != len(set(gpus)):
            print(f"[warn] --gpus 有重复项: {gpus}", flush=True)
        save_checkpoint(model, pool_ckpt, meta={**meta, "arch": arch},
                        norm_mean=norm_mean, norm_std=norm_std)
        pool = ProcessActorPool(
            int(args.processes), gpus, pool_ckpt, arch=arch,
            sims=int(args.sims), determinizations=int(args.determinizations),
            seed=int(args.seed), threads=int(args.threads),
            batch_size=int(args.batch_size),
            max_wait_ms=float(args.leaf_wait_ms),
            tmpdir=os.path.join(args.out, "_proc_tmp"),
            ipc_mode=str(args.ipc_mode),
            records_dtype=str(args.records_dtype),
            shm_slots=int(args.ipc_shm_slots),
            collect_sha=bool(int(args.collect_sha)))
        print(f"[pool] {json.dumps(pool.describe(), ensure_ascii=False)}", flush=True)
        if eval_async and str(eval_device).startswith("cuda"):
            try:
                _eidx = int(str(eval_device).split(":")[1]) if ":" in str(eval_device) else 0
            except ValueError:
                _eidx = None
            if _eidx is not None and _eidx in gpus:
                print(f"[warn] 评测卡 {eval_device} 同时是 actor 卡：异步评测不会阻塞主循环，"
                      f"但会与 actor 抢算力（想完全隔离可换 --eval-device cuda:6/7）",
                      flush=True)
    batch_server = None
    evaluator = None
    if pool is None and args.workers > 1:
        batch_server = BatchLeafServer(model, device=args.device,
                                       batch_size=int(args.batch_size),
                                       max_wait_ms=2.0, queue_size=8192,
                                       norm_mean=norm_mean, norm_std=norm_std)
        evaluator = BatchPPOEvaluator(batch_server)
    elif pool is None:
        evaluator_cls = PPOEvaluator
        if args.local_batch_inference:
            if not int(args.lockstep):
                raise ValueError("local-batch-inference需要lockstep")
            from hsbg.az.train.local_batch_eval import LocalBatchPPOEvaluator
            evaluator_cls = LocalBatchPPOEvaluator
        evaluator = evaluator_cls(model, device=args.device,
                                 norm_mean=norm_mean, norm_std=norm_std)

    # ---- 权重版本文件（流水线：worker 在任务开始时按版本懒加载）----
    weights_dir = os.path.join(args.out, "_weights")
    os.makedirs(weights_dir, exist_ok=True)
    published: Dict[str, Any] = {"version": None, "path": None, "history": {}}

    def _publish_weights(version: int) -> str:
        path = os.path.join(weights_dir, f"w_{int(version):06d}.pt")
        tmp = path + ".tmp"
        save_checkpoint(model, tmp, meta={**meta, "arch": arch},
                        norm_mean=norm_mean, norm_std=norm_std)
        os.replace(tmp, path)
        published["version"], published["path"] = int(version), path
        keep = max(2, int(args.keep_versions))
        # 按**写入时间**裁剪，而不是按版本号字符串排序：续训时 trainer.updates 会**回落到
        # checkpoint 里的版本**（如 501），而目录里已有更高版本（536–538）；按名字排序会把
        # 刚写出的 w_000501.pt 当成"最旧"删掉 ⇒ 第一个 iteration 采集直接
        # FileNotFoundError（本 run 踩过）。刚发布的文件永不删。
        olds = sorted(glob.glob(os.path.join(weights_dir, "w_*.pt")),
                      key=lambda p: os.path.getmtime(p))
        for old in olds[:-keep]:
            if os.path.abspath(old) == os.path.abspath(path):
                continue
            try:
                os.remove(old)
            except OSError:
                pass
        return path

    if pool is not None:
        _publish_weights(int(trainer.updates))

    pipeline_on = bool(int(args.pipeline)) and pool is not None
    if int(args.pipeline) and pool is None:
        print("[warn] --pipeline 需要 --processes>1，已退回串行", flush=True)
    if pipeline_on:
        # 实测：learner 与 actor 共用一张卡时两者互相拖慢（collect +42%、PPO +27%），
        # 流水线整体只快 19%；把 learner 放到独立卡可到 +44%。
        learner_idx = None
        if str(args.device).startswith("cuda"):
            learner_idx = int(str(args.device).split(":")[1]) if ":" in str(args.device) else 0
        actor_gpus = [int(x) for x in str(args.gpus).split(",") if x.strip()]
        if learner_idx is not None and learner_idx in actor_gpus:
            suggest = ",".join(str(g) for g in actor_gpus if g != learner_idx) or "（无其它卡）"
            print(f"[warn] 流水线开启但 learner 与 actor 共用 GPU{learner_idx}："
                  f"实测会互抢（2,552 vs 隔离后 3,100 dec/s）。"
                  f"建议 --gpus {suggest} 或 --device cuda:<空闲卡>。", flush=True)

    def _collect_for(idx: int, wv: Optional[int] = None,
                     wp: Optional[str] = None
                     ) -> Tuple[Any, Dict[str, Any], CollectConfig, float, Any]:
        # p ramp：按 iteration 下标求值（空 schedule = 固定 --mirror-p）。
        _mp = float(args.mirror_p)
        if str(args.mirror_p_ramp).strip():
            _mp = float(ramp_p(int(idx), str(args.mirror_p_ramp)))
        # 经济课程：按**已累计 decisions** 求本 iteration 的金币倍数（默认 none -> 1.0）
        _gold_mult = gold_mult_at(
            cum_decisions, stage=_curr_cfg["curriculum_stage"],
            mult=_curr_cfg["curriculum_gold_mult"],
            anneal_decisions=_curr_cfg["curriculum_anneal_decisions"])
        # 对手强度课程：按**已累计 learner decisions** 求本 iteration 生效的
        # seats/p/sample（空 spec -> 原样返回 CLI 值，逐位等价）。
        _hb_state = board_ramp.state(cum_decisions)
        # 动态课程（--dyn-curriculum 1）覆盖阶段式的 seats/p；其余（sample/quantile）不变。
        _hb_seats, _hb_p = int(_hb_state.seats), float(_hb_state.p)
        if int(args.dyn_curriculum) > 0:
            _hb_seats = int(_dyn["seats"])
            _hb_p = float(args.dyn_p)
        # 强制升本：本 iteration 生效的概率与教师开关（按累计 decisions 求值）
        _ft_prob = prob_at(_ft_schedule, cum_decisions) if _ft_plan else 0.0
        _ft_teacher = bool(_ft_plan and _ft_teacher_until > 0
                           and float(cum_decisions) < _ft_teacher_until)
        # 动态难度：控制器当前强度档位 -> "该回合对手应处于真人分布的第几分位"。
        # 采集侧只认分位（战团库在 worker 里），所以这一步换算在训练侧做（曲线是固定文件）。
        _aq: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        if arena is not None and arena_curve is not None:
            _aq = tuple(                      # type: ignore[assignment]
                float(arena_curve.quantile(
                    band_center_turn(_b), float(arena.state.strength.get(_b, 0.0))) or 0.0)
                for _b in ("early", "mid", "late"))
        c_cfg = CollectConfig(
            games=int(args.games), seed=int(args.seed) + idx * 1000,
            opponent=str(args.opponent), behavior=str(args.behavior),
            max_actions_per_turn=int(args.max_actions_per_turn),
            max_steps=int(args.max_steps), fast_act=bool(int(args.fast_act)),
            lockstep=bool(int(args.lockstep)),
            move_guard=bool(int(args.move_guard)),
            forbid_cancel=bool(int(args.forbid_cancel)),
            forbid_unfreeze=bool(int(args.forbid_unfreeze)),
            freeze_cap=int(args.freeze_cap),
        ban_trinkets=bool(int(getattr(args, "ban_trinkets", 1) or 0)),
        # 2026-11-17（D-071）：lobby 种族排除（空 = 不过滤，逐位等价旧行为）。
        lobby_exclude=parse_race_list(getattr(args, "lobby_exclude", "")),
        lobby_filter_probability=args.lobby_filter_probability,
            move_cap=int(args.move_cap),
            archetype_bias_coef=float(args.archetype_bias_coef),
            archetype_min_prob=float(args.archetype_min_prob),
            archetype_cache=bool(int(args.archetype_cache)),
            archetype_table=str(args.archetype_table),
            archetype_model=str(args.archetype_model),
            move_repeat_limit=int(args.move_repeat_limit),
            library_pool=str(args.library_pool), library_p=float(args.library_p),
            trajectory_pool=str(args.trajectory_pool), trajectory_p=float(args.trajectory_p),
            mirror_strict=bool(args.mirror_strict),
            mirror_snaps=str(args.mirror_snaps), mirror_seats=int(args.mirror_seats),
            mirror_p=_mp, mirror_sample=str(args.mirror_sample),
            mirror_p_ramp=str(args.mirror_p_ramp),
            mirror_seat_random=bool(int(args.mirror_seat_random)),
            mirror_max_turn=int(args.mirror_max_turn),
            human_boards=str(args.human_boards),
            human_board_p=_hb_p,
            human_board_seats=_hb_seats,
            human_board_sample=str(_hb_state.sample),
            human_board_set_tier=bool(int(args.human_board_set_tier)),
            human_board_fidelity=str(args.human_board_fidelity),
            human_board_fast_inject=bool(int(args.human_board_fast_inject)),
            human_board_min_turn=int(args.human_board_min_turn),
            # 中后期入场课程（默认 entry_p=0 -> 零影响）
            entry_boards=str(args.entry_boards),
            entry_p=float(args.entry_p),
            entry_turn_min=int(args.entry_turn_min),
            entry_turn_max=int(args.entry_turn_max),
            entry_sample=str(args.entry_sample),
            entry_fidelity=str(args.entry_fidelity),
            entry_fast_inject=bool(int(args.entry_fast_inject)),
            entry_set_health=bool(int(args.entry_set_health)),
            entry_min_tier=int(args.entry_min_tier),
            entry_require_cards=str(args.entry_require_cards or ""),
            entry_min_hand=int(getattr(args, "entry_min_hand", 0) or 0),
            # 专家相对过程奖励（默认 coef=0 -> 零开销）
            board_target_coef=float(args.board_target_coef),
            board_target_w_card=float(args.board_target_w_card),
            board_target_w_str=float(args.board_target_w_str),
            board_target_cap=float(args.board_target_cap),
            frozen_opponent=str(args.frozen_opponent),
            frozen_seats=int(args.frozen_seats),
            # 强制升本课程（默认空计划 -> 逐位等价旧路径）
            forced_tier_plan=str(args.forced_tier_plan),
            forced_tier_carry=int(args.forced_tier_carry),
            forced_tier_prob=float(_ft_prob),
            forced_tier_teacher=bool(_ft_teacher),
            forced_spell_per_game=int(getattr(args, "forced_spell_per_game", 0) or 0),
            forced_spell_turn_min=int(getattr(args, "forced_spell_turn_min", 6) or 0),
            forced_spell_prob=float(getattr(args, "forced_spell_prob", 1.0) or 0.0),
            forced_spell_teacher=bool(int(getattr(args, "forced_spell_teacher", 0) or 0)),
            rust_collector=bool(int(args.rust_collector)),
            rc_threads=int(args.rc_threads), leaf_amp=str(args.leaf_amp),
            leaf_fused=bool(args.leaf_fused),
            leaf_compile=bool(int(args.leaf_compile)),
            leaf_wait_ms=float(args.leaf_wait_ms),
            # 动作序列辅助任务：只在启用时让采集/打包侧产出 seq_target（默认 False）
            seq_aux=bool(seq_enabled),
            # 势函数奖励塑形（默认 coef=0 -> 零影响）
            shaping_coef=float(args.shaping_coef),
            spell_bootstrap_coef=float(getattr(args, "spell_bootstrap_coef", 0.0) or 0.0),
            spell_bootstrap_per_game=int(getattr(args, "spell_bootstrap_per_game", 0) or 0),
            shaping_gamma=args.shaping_gamma,
            shaping_w_str=float(args.shaping_w_str),
            shaping_w_tier=float(args.shaping_w_tier),
            shaping_mode=str(args.shaping_mode),
            shaping_tier_bonus=str(args.shaping_tier_bonus),
            # 等级进度奖励（默认 coef=0 -> 零影响）
            tier_target_coef=float(args.tier_target_coef),            tier_penalty=float(args.tier_penalty),
            tier_bonus=float(args.tier_bonus),
            tier_target_table=str(args.tier_target_table),
            tier_gate_k=float(args.tier_gate_k),
            tier_gate_ref=str(args.tier_gate_ref),
            # 阵容覆盖奖励（默认 coef=0 -> 采集侧不构造对象、不新增 meta 键）
            retained_growth_coef=float(args.retained_growth_coef),
        retained_growth_min_turn=int(args.retained_growth_min_turn),
        retained_growth_normalizer=float(args.retained_growth_normalizer),
        retained_growth_cap=float(args.retained_growth_cap),
        comp_target_coef=float(args.comp_target_coef),
            comp_target_min_turn=int(args.comp_target_min_turn),
            comp_target_cards=str(args.comp_target_cards),
            comp_target_tier_w=str(args.comp_target_tier_w),
            comp_target_cap=float(args.comp_target_cap),
            comp_target_min_hits=int(args.comp_target_min_hits),
            # 经济课程（默认 none+1.0 -> 采集侧不碰金币，逐位等价）
            curriculum_stage=str(_curr_cfg["curriculum_stage"]),
            curriculum_gold_mult=float(_gold_mult),
            recruit_start_bank=str(args.recruit_start_bank),
            recruit_start_p=float(args.recruit_start_p),
            recruit_start_manifest_sha256=str(args.recruit_start_manifest_sha256),
            # 在线缩放教师 / 过程价值标签（默认 False -> 不查教师、不读日志）
            teacher_bc=bool(float(args.teacher_bc_coef) > 0.0),
            process_value=bool(float(args.process_value_coef) > 0.0),
            # 逐回合遥测（默认开：D-052 之后这是中后期能力的唯一直接刻度）
            turn_telemetry=True,
            # 动态难度（默认关：arena 恒 None -> arena_on=0，采集侧走原分位路径）
            arena_on=bool(arena is not None),
            arena_q_early=float(_aq[0]), arena_q_mid=float(_aq[1]),
            arena_q_late=float(_aq[2]),
            # 搜索作为行为策略（默认 eps=0 -> 不构造 mixer、零开销）
            behavior_mix_eps=float(args.behavior_mix_eps),
            behavior_mix_when=str(args.behavior_mix_when),
            behavior_mix_max_per_turn=int(args.behavior_mix_max_per_turn),
            behavior_mix_topk=int(args.behavior_mix_topk),
            behavior_mix_turns=int(args.behavior_mix_turns),
            behavior_mix_max_ms=float(args.behavior_mix_max_ms),
            behavior_mix_exact=int(args.behavior_mix_exact))
        t_c = time.time()
        if pool is not None:
            batch, collect_meta = pool.collect(
                c_cfg, int(args.games), return_records=True, progress=not pipeline_on,
                seed=int(c_cfg.seed), gamma=float(p_cfg.gamma), lam=float(args.lam),
                adv_estimator=str(args.adv_estimator),
                power_alpha=float(args.power_alpha),
                power_kappa=float(args.power_kappa),
                power_normalize=bool(int(args.power_normalize)),
                power_cutoff=int(args.power_cutoff),
                credit_timebase=str(args.credit_timebase),
                rank_utility_coef=float(args.rank_utility_coef),
                terminal_strength_coef=float(args.terminal_strength_coef),
                weights_version=(published["version"] if wv is None else wv),
                weights_path=(published["path"] if wp is None else wp))
            if str(args.mirror_snaps):
                from hsbg.az.train.collect_diagnostics import require_valid_mirror_collect
                require_valid_mirror_collect(collect_meta, args.out, iteration=idx, config=vars(c_cfg))
        elif int(args.lockstep) and str(args.mirror_snaps) and int(args.sims) == 0:
            # 单 GPU / CPU 的个人部署：沿用 lockstep，不修改 Slurm 的可见设备列表。
            from hsbg.az.train.parallel_collect import collect_ppo_lockstep
            from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
            from hsbg.az.train.ppo import build_batch_from_records
            mirror_pool = MirrorSnapshotPool(str(args.mirror_snaps), sample=str(args.mirror_sample))
            if hasattr(evaluator, "reset_inference_stats"):
                evaluator.reset_inference_stats()
            records, collect_meta = collect_ppo_lockstep(
                model, c_cfg, eval_fn=evaluator, device=str(args.device),
                seed=int(c_cfg.seed), games=int(args.games), gamemode=str(args.opponent),
                norm_mean=norm_mean, norm_std=norm_std, mirror_pool=mirror_pool,
                mirror_seats_count=int(args.mirror_seats), mirror_p=float(_mp),
                mirror_seat_random=bool(int(args.mirror_seat_random)))
            collect_meta["inference"] = (evaluator.inference_stats()
                if hasattr(evaluator, "inference_stats") else {"mode": "serial"})
            from hsbg.az.train.collect_diagnostics import require_valid_mirror_collect
            require_valid_mirror_collect(collect_meta, args.out, iteration=idx,
                                         config=vars(c_cfg))
            batch = build_batch_from_records(records, gamma=float(p_cfg.gamma), lam=float(args.lam))
        elif args.workers > 1:
            batch, collect_meta = collect_ppo_batch_parallel(
                model, c_cfg, workers=int(args.workers),
                search_fn=search_fn, search_config=search_cfg,
                eval_fn=evaluator, device=str(args.device),
                seed=int(args.seed) + idx * 1000, gamma=float(p_cfg.gamma),
                lam=float(args.lam),
                norm_mean=norm_mean, norm_std=norm_std, progress=True,
                adv_estimator=str(args.adv_estimator),
                power_alpha=float(args.power_alpha),
                power_kappa=float(args.power_kappa),
                power_normalize=bool(int(args.power_normalize)),
                power_cutoff=int(args.power_cutoff),
                credit_timebase=str(args.credit_timebase),
                rank_utility_coef=float(args.rank_utility_coef),
                terminal_strength_coef=float(args.terminal_strength_coef))
        else:
            batch, collect_meta = collect_ppo_batch(
                model, c_cfg, search_fn=search_fn, search_config=search_cfg,
                eval_fn=evaluator, device=str(args.device),
                seed=int(args.seed) + idx * 1000, gamma=float(p_cfg.gamma),
                lam=float(args.lam),
                norm_mean=norm_mean, norm_std=norm_std, progress=True,
                adv_estimator=str(args.adv_estimator),
                power_alpha=float(args.power_alpha),
                power_kappa=float(args.power_kappa),
                power_normalize=bool(int(args.power_normalize)),
                power_cutoff=int(args.power_cutoff),
                credit_timebase=str(args.credit_timebase),
                rank_utility_coef=float(args.rank_utility_coef),
                terminal_strength_coef=float(args.terminal_strength_coef))
        # 动态难度闭环：用本轮**实测**的到达率 / 战斗胜率更新下一轮的难度档位并落盘。
        # 放在 collect 之后、update 之前：下一轮采集才会用到新档位（与 ramp 同相位）。
        if c_cfg.lobby_filter_probability < 1:
            from hsbg.az.train.lobby_filter import is_excluded
            _, first = np.unique(batch.episode_ids, return_index=True)
            collect_meta["practice_games_observed"] = len(first)
            collect_meta["practice_games_without_races"] = sum(
                not is_excluded(batch.features[j], c_cfg.lobby_exclude) for j in first)
        if arena is not None:
            try:
                collect_meta.update(arena.update(_arena_stats_from_meta(collect_meta)))
                print(f"[arena] it={arena.state.iters} strength="
                      f"{ {k: int(v) for k, v in arena.state.strength.items()} } "
                      f"q={ {k: round(float(v), 3) for k, v in zip(('early','mid','late'), _aq)} } "
                      f"reach_target={float(arena.state.reach_target):g} "
                      f"ratchet={int(arena.state.ratchet_level)}", flush=True)
            except BaseException as _exc:  # noqa: BLE001 - 控制器问题绝不能打断训练
                print(f"[warn] arena update failed: {_exc!r}", flush=True)
        return batch, collect_meta, c_cfg, time.time() - t_c, _hb_state

    def _pre_ratio(batch: Any, n: int = 2048) -> float:
        """用当前参数对 batch 抽样算 mean ratio（流水线的越策略守卫）。"""
        total = len(batch.actions)
        idx = np.arange(total)
        if total > n:
            idx = np.random.default_rng(0).choice(total, size=n, replace=False)
        # forced rows store old_logprob=0 and are excluded from the PPO loss;
        # including them would drag the pipeline-guard mean ratio toward 0.
        valid = ~np.asarray(batch.forced[idx], dtype=bool)
        if not bool(valid.any()):
            return 1.0
        idx = idx[valid]
        dev = torch.device(args.device)
        x = torch.from_numpy(np.ascontiguousarray(batch.features[idx])).to(dev)
        c = torch.from_numpy(np.ascontiguousarray(batch.cards[idx].astype(np.int64))).to(dev)
        m = torch.from_numpy(np.ascontiguousarray(batch.masks[idx].astype(bool))).to(dev)
        a = torch.from_numpy(np.ascontiguousarray(batch.actions[idx].astype(np.int64))).to(dev)
        old = torch.from_numpy(
            np.ascontiguousarray(batch.old_logprobs[idx].astype(np.float32))).to(dev)
        with torch.no_grad():
            logits, _r, _s = split_model_outputs(forward_model(model, x, c, m))
            logp = masked_log_softmax_torch(logits, m).gather(1, a.unsqueeze(1)).squeeze(1)
            ratio = torch.exp(logp - old)
        return float(ratio.mean())

    t0 = time.time()
    deadline = t0 + float(args.minutes) * 60.0
    if os.environ.get("HSAI_ARENA_STOP_EPOCH"):
        deadline = min(deadline, float(os.environ["HSAI_ARENA_STOP_EPOCH"]))
    cum_decisions = int(trainer.extra.get("cum_decisions", 0))
    next_eval = (int(args.eval_every_decisions)
                 if int(args.eval_every_decisions) > 0 else 10 ** 18)
    i = int(start_i)
    end_i = (int(start_i) + int(args.iterations)) if int(args.iterations) > 0 else 0
    last_len = 0.0
    executor = ThreadPoolExecutor(max_workers=1) if pipeline_on else None
    pending = None
    #: critic 预热只在**第一个 iteration** 做一次（--value-warmup-iters>0 时）
    _value_warmed = False
    #: 动态课程状态（--dyn-curriculum 1 时生效；默认 seats=0 表示不覆盖）
    _dyn: Dict[str, Any] = {"seats": int(args.dyn_start_seats) if int(args.dyn_curriculum) > 0 else 0,
                            "last_s15": None, "hold_until": 0.0, "history": []}
    _vwarm: Optional[Dict[str, Any]] = None
    if executor is not None:
        pending = executor.submit(_collect_for, i, published["version"], published["path"])
        print(f"[pipeline] on：collect(k+1) 与 update(k) 并行；"
              f"每 batch 版本钉死（lag<=1）；guard={args.pipeline_guard}", flush=True)

    try:
        while time.time() < deadline:
            stop_file = os.environ.get("HSRL_STOP_FILE", os.path.join(args.out, "STOP_REQUESTED"))
            if os.path.isfile(stop_file) and os.path.isfile(os.path.join(args.out, "latest.json")):
                print("[loop] user stop requested; preserved committed checkpoint", flush=True)
                break
            if end_i > 0 and i >= end_i:
                break
            if float(args.stop_at_decisions) > 0 and cum_decisions >= float(args.stop_at_decisions):
                print(f"[loop] decisions {cum_decisions} >= {int(args.stop_at_decisions)}; stop",
                      flush=True)
                break
            if last_len > 0 and (deadline - time.time()) < max(60.0, last_len):
                print("[loop] remaining time < last iteration; stop", flush=True)
                break
            if eval_stop.is_set():
                # 异步评测结果到达后置 flag：只在下个 iteration 边界停，已完成的 iteration
                # 一定保留（history/snapshot 都已落盘）。
                print(f"[loop] eval stop requested ({eval_state['stop_reason']}); "
                      f"stop at iteration boundary", flush=True)
                if pending is not None:
                    pending.cancel()
                break
            _history_s_prev = locals().get("_history_s", 0.0)
            iter_start = time.time()
            iteration_pipeline_on=pipeline_on
            _tl = {"t_iter_start": time.monotonic()}
            _reload_s = 0.0

            if pending is not None:
                batch, collect_meta, c_cfg, _collect_s, _hb_state = pending.result()
                _reload_s = sum(float(w.get("reload_s", 0.0) or 0.0)
                                for w in (collect_meta.get("workers") or []))
                pending = None
            else:
                batch, collect_meta, c_cfg, _collect_s, _hb_state = _collect_for(i)
            _tl["t_collect_done"] = time.monotonic()

            # 零更新回退只允许影响**一轮**：上一次 after_update 把执行器关掉后（它已经
            # wait 完那一批并留给本轮消费），这里在 pending 已消费完（lag 仍 ≤1）时重新武装。
            # 2026-11-14 实测：不回武装会永久串行，每轮从 19.5s 变 31.9s（吞吐 −39%）。
            if (executor is None and int(args.pipeline) and pool is not None
                    and (end_i <= 0 or i + 1 < end_i)
                    and (deadline - time.time()) > 60.0):
                executor = ThreadPoolExecutor(max_workers=1)
                pipeline_on = True
                iteration_pipeline_on = True
                print("[pipeline] 重新启用重叠采集（上次零更新只影响一轮）", flush=True)

            errs = batch.validate()
            if errs:
                print(f"[error] batch invalid: {errs}", file=sys.stderr)
                return 3

            # ── 健康门（采集侧）：审计 P2-11 —— `TRAINING_RECIPE §5` 的红线以前只在文档里。
            # 硬门 = 工程错误（非法动作/截断/引擎停止/注入率崩塌）⇒ 严格停止并保留现场，
            # 因为继续训练只会把错标签灌进模型（AGENTS §6）。
            if int(args.health_gate):
                _collect_failures = hard_collect_failures(collect_meta, health_limits)
                if _collect_failures:
                    print(format_failures(_collect_failures, where=f"collect iter {i}"),
                          file=sys.stderr, flush=True)
                    raise RuntimeError(
                        "健康门（采集侧）触发，已停止以保留现场："
                        + "; ".join(_collect_failures)
                        + f"（out={args.out}；快照见 snapshots/）")

            # ── 自定义奖励（--upgrade-reward / --win-only-reward / --strength-reward-coef）
            # 必须放在 update 之前：改 rewards 后整批重算 advantages/returns。
            # 三者默认全 0 ⇒ 本块不执行 ⇒ 逐位等价旧路径。
            _rw: Optional[Dict[str, Any]] = None
            if (float(args.upgrade_reward) > 0.0 or float(args.win_only_reward) > 0.0
                    or float(args.strength_reward_coef) > 0.0):
                _rw = _apply_custom_rewards(
                    batch, upgrade_w=float(args.upgrade_reward),
                    win_w=float(args.win_only_reward),
                    strength_c=float(args.strength_reward_coef),
                    gamma=float(args.gamma), lam=float(args.lam))

            batch_version = int(collect_meta.get("policy_version", -1))
            if batch_version < 0:  # 非 pool 路径
                batch_version = int(trainer.updates)
            lag = int(trainer.updates) - batch_version
            if lag < 0 or lag > 1:
                raise RuntimeError(
                    f"流水线滞后异常：batch_version={batch_version} "
                    f"trainer_version={trainer.updates} lag={lag}（上限 1）")

            # 先发起下一轮采集（用当前已发布权重），再跑 update —— 这是流水的关键顺序
            if executor is not None and (end_i <= 0 or i + 1 < end_i) \
                    and (deadline - time.time()) > 60.0:
                pending = executor.submit(_collect_for, i + 1,
                                          published["version"], published["path"])

            pre_ratio = 1.0
            guard_tripped = False
            if pipeline_on and lag >= 1:
                pre_ratio = _pre_ratio(batch)
                if not (0.8 <= pre_ratio <= 1.25):
                    guard_tripped = True
                    print(f"[pipeline] ratio guard tripped: pre_ratio_mean={pre_ratio:.3f} "
                          f"(lag={lag}); guard={args.pipeline_guard}", flush=True)
                    if str(args.pipeline_guard) == 'abort':
                        raise RuntimeError('流水线策略比率越界：在PPO更新前停止，保留最后提交快照')

            trainer.config.anchor_coef = (
                float(args.anchor_beta) if i < int(args.anchor_iterations)
                else float(args.anchor_beta_tail))
            if seq_enabled:
                # 权重 ramp：按**已累计** decisions 的进度插值（本 iteration 生效）
                trainer.config.seq_coef = float(seq_schedule.coef(cum_decisions))
                if getattr(batch, "seq_target", None) is None and i == start_i:
                    print("[warn] seq 辅助任务已开启但批次没有 seq_target"
                          "（--rust-collector 路径暂不产出标签）-> 本批次 seq_loss=0",
                          flush=True)
            if bc_enabled:
                # BC 系数 ramp：与 seq 同一进度口径（累计 decisions）
                trainer.config.bc_coef = float(bc_schedule.coef(cum_decisions))
            # ── critic 预热（--value-warmup-iters>0；默认 0 完全跳过）─────────────
            # 放在**第一个 iteration 的 update 之前**：此时批次来自（预热的）策略、还没被
            # 随机 critic 的噪声梯度动过，用它校准 critic 头最干净。
            # `--value-warmup-every N>0` 时每 N 个 iteration 再重热一次：实测一次预热的效果
            # 会被后续 PPO 的 trunk 漂移抹掉（rank-CE 0.90→2.4），长训练需要周期性重热。
            _vw_due = False
            if int(args.value_warmup_iters) > 0:
                if not _value_warmed:
                    _value_warmed = True
                    _vw_due = True
                elif (int(args.value_warmup_every) > 0 and i > int(start_i)
                      and (i % int(args.value_warmup_every)) == 0):
                    _vw_due = True
            if _vw_due:
                _vwarm = _value_warmup(
                    trainer, batch,
                    iters=int(args.value_warmup_iters),
                    lr=float(args.value_warmup_lr),
                    minibatch=int(args.value_warmup_minibatch),
                    seed=int(args.seed))
            _t_ppo = time.time()
            _tl["t_ppo_start"] = time.monotonic()
            profile_path = os.environ.get("HSRL_PPO_PROFILE_OUT")
            if profile_path and i == start_i + 2:
                from workflows.profile_nus_ppo import profile_update
                stats = profile_update(trainer, batch, profile_path, i, lag)
            else:
                stats = trainer.update(batch)
            gpu_memory = {}
            if str(args.device).startswith('cuda'):
                gpu_memory = dict(allocated_bytes=torch.cuda.memory_allocated(args.device),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                    reserved_bytes=torch.cuda.memory_reserved(args.device))
            # ── 健康门（更新侧）：非有限损失 / 非法动作概率泄漏 ⇒ 严格停止。
            _soft_warn: List[str] = []
            if int(args.health_gate):
                _upd_failures = hard_update_failures(stats, health_limits)
                if _upd_failures:
                    print(format_failures(_upd_failures, where=f"update iter {i}"),
                          file=sys.stderr, flush=True)
                    raise RuntimeError(
                        "健康门（更新侧）触发，已停止以保留现场："
                        + "; ".join(_upd_failures))
                _soft_warn = soft_warnings(stats, collect_meta, health_limits)
                for _w in _soft_warn:
                    print(f"[health-warn] iter {i}: {_w}", flush=True)
            if _rw is not None:
                stats["custom_rewards"] = dict(_rw)
            if _vwarm is not None:
                stats["value_warmup"] = dict(_vwarm)
                _vwarm = None
            # 塑形可观测性：history.jsonl 的 ppo 段带系数与实际 shaping 奖励均值。
            _cmeta = collect_meta.get("collect") if isinstance(collect_meta, dict) else None
            if not isinstance(_cmeta, dict):
                _cmeta = collect_meta if isinstance(collect_meta, dict) else {}
            stats["shaping_coef"] = float(args.shaping_coef)
            stats["shaping_gamma"] = float(shape_gamma)
            stats["shaping_gamma_aligned"] = bool(shape_aligned)
            stats["reward_shaping_mean"] = float(
                _cmeta.get("reward_shaping_mean", 0.0) or 0.0)
            stats["reward_shaping_bias_sum"] = float(
                _cmeta.get("reward_shaping_bias_sum", 0.0) or 0.0)
            # 优势标准差（PPOBatch 已算 advantage_std；history 里此前只有 adv_mean）
            stats["adv_std"] = float(np.std(batch.advantages))
            # 等级进度奖励可观测性（默认关闭时全为 0；history 字段名固定）
            stats["tier_target_coef"] = float(args.tier_target_coef)
            stats["tier_hits"] = int(_cmeta.get("tier_hits", 0) or 0)
            stats["tier_misses"] = int(_cmeta.get("tier_misses", 0) or 0)
            stats["tier_bonus_sum"] = float(_cmeta.get("tier_bonus_sum", 0.0) or 0.0)
            _ppo_s = time.time() - _t_ppo
            _tl["t_ppo_done"] = time.monotonic()
            from hsbg.az.train.pipeline_safety import after_update
            _fallback_start=time.monotonic()
            pipeline_on,executor,zero_fallback=after_update(pipeline_on,int(stats.get('updates',0)),lag,
                executor,bool(int(args.pipeline_zero_step_fallback)))
            _fallback_s=time.monotonic()-_fallback_start
            if zero_fallback:
                print('[pipeline] 旧批次KL零更新：保留已发下一批，后续同步采集；KL阈值保持不变',flush=True)

            # 发布新权重（供下一轮 collect 的 *下一次* 使用；lag 恒为 1）
            if pool is not None:
                _publish_weights(int(trainer.updates))

            slim = slim_collect(collect_meta)
            # ── 对手强度课程：本 iter 实际生效值 + 自适应门控 ──────────────────
            # 注意：``_hb_state`` 由 ``_collect_for`` 在提交采集时求值，pipeline 模式下
            # 最多滞后 1 个 iteration（与 batch 版本滞后同源）；history 记录的就是它。
            _decisions_i = int(slim.get("decisions", 0) or 0)
            _cum_after = int(cum_decisions) + _decisions_i
            _ramp_events: List[Dict[str, Any]] = []
            _ramp_gate: Optional[Dict[str, Any]] = None
            if board_ramp.enabled and int(args.ramp_adaptive):
                _quality = _read_quality_json(str(args.ramp_quality_json))
                _ramp_gate = ramp_progress_metrics(
                    getattr(batch, "turns", None), getattr(batch, "game_ids", None),
                    collect_meta=collect_meta, quality=_quality)
                _ramp_events = board_ramp.observe(i, _cum_after, _ramp_gate)
                for _ev in _ramp_events:
                    print(f"[ramp] iter {i} gate event: {_ev}", flush=True)
            rec = {"iter": i, "collect": slim, "ppo": stats,
                   "learner_gpu_memory": gpu_memory,
                   # 健康门结果（硬门在更早处已经 raise；这里只记录软警告）。
                   "health": {"soft_warnings": list(_soft_warn),
                              "gate_on": bool(int(args.health_gate)),
                              "limits": {"mirror_apply_min": health_limits.mirror_apply_min,
                                         "entropy_warn_below": health_limits.entropy_warn_below,
                                         "kl_warn_above": health_limits.kl_warn_above,
                                         "throughput_warn_below": health_limits.throughput_warn_below}},
                   "pipeline": {"enabled": iteration_pipeline_on, "next_enabled":pipeline_on,
                                "fallback_zero_step":zero_fallback,"lag": int(lag),
                                "batch_version": int(batch_version),
                                "pre_ratio_mean": round(float(pre_ratio), 4),
                                "guard_tripped": bool(guard_tripped)},
                   "profile": {"reload_s": round(_reload_s, 3),
                               "pipeline_fallback_wait_s":round(_fallback_s,3),
                               "collect_s": round(_collect_s, 3),
                               "ppo_s": round(_ppo_s, 3),
                               "build_batch_s": round(float(slim.get("build_batch_s", 0.0)), 3),
                               "parent_read_s": round(float(slim.get("parent_read_s", 0.0)), 3),
                               "records_mb": round(float(slim.get("records_bytes", 0)) / 1e6, 1),
                               "history_prev_s": round(float(_history_s_prev), 3)},
                   "batch_server": (dict(batch_server.stats) if batch_server is not None else None),
                   "seconds": time.time() - t0}
            rec["human_board"] = board_ramp_history(
                _hb_state, spec=str(args.human_board_ramp),
                quantile_spec=str(args.human_board_quantile_ramp),
                mode=str(args.ramp_mode), adaptive=bool(int(args.ramp_adaptive)),
                events=_ramp_events)
            # 平铺三键：history.jsonl 里一眼能查「本 iter 实际生效值」
            rec["human_board_seats"] = int(_hb_state.seats)
            rec["human_board_p"] = round(float(_hb_state.p), 6)
            rec["human_board_sample"] = str(_hb_state.sample)
            if _ramp_gate is not None:
                rec["ramp_gate"] = dict(
                    _ramp_gate, events=list(_ramp_events),
                    window=board_ramp.window_summary())
            if problem_pool is not None:
                # 题池过滤口径（默认 0/空 -> dropped 全 0）：池大小 / 按 delta 丢 /
                # 按 turn 丢，与 ppo 段的 problem_pool_* 同源
                rec["problems"] = dict(problem_pool.filter_report)
            # 经济课程可观测性（默认关闭时也写，字段固定：stage/mult/progress）
            _curr_prog = float(anneal_progress(
                cum_decisions, _curr_cfg["curriculum_anneal_decisions"]))
            rec["curriculum_stage"] = str(_curr_cfg["curriculum_stage"])
            rec["curriculum_gold_mult"] = round(
                float(c_cfg.curriculum_gold_mult), 6)
            rec["curriculum_anneal_progress"] = round(_curr_prog, 6)
            rec["curriculum"] = {
                "stage": str(_curr_cfg["curriculum_stage"]),
                "gold_mult": round(float(c_cfg.curriculum_gold_mult), 6),
                "anneal_progress": round(_curr_prog, 6),
                "decisions": int(cum_decisions),
            }
            # 强制升本课程：本 iter 生效的概率/教师开关 + 实际执行统计
            # 概率/开关取自 collect meta（由采集侧按当时的 CollectConfig 回传，
            # 与 pipeline 滞后无关），不用主循环里的下一轮取值。
            if str(args.forced_tier_plan).strip():
                _ft = _cmeta.get("forced_tier") or {}
                rec["forced_tier"] = {
                    "plan": str(_ft.get("plan") or args.forced_tier_plan),
                    "carry": int(_ft.get("carry", args.forced_tier_carry) or 0),
                    "prob": round(float(_ft.get("prob", 0.0) or 0.0), 6),
                    "teacher": bool(_ft.get("teacher", False)),
                    "forced_actions": int(_ft.get("forced_actions", 0) or 0),
                    "teacher_labels": int(_ft.get("teacher_labels", 0) or 0),
                    "blocked": int(_ft.get("blocked", 0) or 0),
                    "prob_skips": int(_ft.get("prob_skips", 0) or 0),
                    "targets": dict(_ft.get("targets") or {}),
                    "decisions": int(cum_decisions),
                }
            history.append(rec)
            print(f"[iter {i}] ts={time.strftime('%H:%M:%S')} samples={len(batch)} "
                  f"updates={stats['trainer_updates']} "
                  f"lr={stats['lr']:.3g} total={stats['epochs'][-1]['total']:.4f}"
                  + (f" lag={lag} ratio={pre_ratio:.3f}" if pipeline_on else "")
                  + (f" seq={stats.get('seq_loss', 0.0):.4f}"
                     f" seq_coef={stats.get('seq_coef', 0.0):.4g}" if seq_enabled else "")
                  + (f" bc={stats.get('bc_loss', 0.0):.4f}"
                     f" bc_type={stats.get('bc_type_loss', 0.0):.4f}"
                     f" bc_card={stats.get('bc_card_loss', 0.0):.4f}"
                     f" bc_coef={stats.get('bc_coef', 0.0):.4g}"
                     f" bc_samples={stats.get('bc_samples', 0.0):.0f}"
                     if bc_enabled else "")
                  + (f" dpo={stats['dpo_loss']:.4f}"
                     f" dpo_acc={stats['dpo_accept_rate']:.3f}"
                     f" dpo_coef={stats['dpo_coef']:.4g}"
                     f" dpo_n={stats['dpo_samples']:.0f}"
                     if dpo_enabled and "dpo_loss" in stats else "")
                  + (f" pbc={stats['problem_bc_loss']:.4f}"
                     f" pbc_top1={stats.get('problem_bc_top1_acc', 0.0):.3f}"
                     if _problem_enabled and "problem_bc_loss" in stats else "")
                  + (f" saux={stats['strength_aux_loss']:.4f}"
                     f" s_mse={stats.get('strength_mse', 0.0):.4f}"
                     f" s_bce={stats.get('strength_bce', 0.0):.4f}"
                     if _strength_enabled and "strength_aux_loss" in stats else "")
                  + (f" hb={int(_hb_state.seats)}s/p{float(_hb_state.p):.2f}"
                     f"/{_hb_state.sample}/b{int(_hb_state.band)}"
                     if board_ramp.enabled else "")
                  + (f" goldx={c_cfg.curriculum_gold_mult:g}"
                     if str(_curr_cfg["curriculum_stage"]) != "none" else "")
                  + (f" teach={stats['teacher_bc_loss']:.4f}"
                     f" teach_n={stats.get('teacher_samples', 0.0):.0f}"
                     if "teacher_bc_loss" in stats else "")
                  + (f" proc={stats['process_value_loss']:.4f}"
                     f" proc_n={stats.get('process_samples', 0.0):.0f}"
                     if "process_value_loss" in stats else ""),
                  flush=True)

            # 版本化快照：trainer 为权威，model 为导出；manifest 最后原子发布
            trainer.extra["cum_decisions"] = cum_decisions + int(slim.get("decisions", 0))
            trainer.extra["next_rollout_index"] = i + 1
            ver = int(trainer.updates)
            do_snap = (int(args.snapshot_every) <= 1) or ((i % int(args.snapshot_every)) == 0)
            if do_snap:
                _checkpoint_start = time.monotonic()
                snap = os.path.join(args.out, "snapshots", f"step_{ver:06d}")
                os.makedirs(snap, exist_ok=True)
                trainer.save(os.path.join(snap, "trainer.pt"))
                ckpt = os.path.join(snap, "model.pt")
                tmp_ckpt = ckpt + ".tmp"
                save_checkpoint(model, tmp_ckpt,
                                meta={**meta, "arch": arch, "trainer_updates": ver},
                                norm_mean=norm_mean, norm_std=norm_std)
                os.replace(tmp_ckpt, ckpt)
                # 快照文件写完后只原子发布别名，消除重复GPU传输/序列化。
                _publish_start = time.monotonic()
                _publish_modes = [publish_checkpoint(os.path.join(snap, "trainer.pt"),
                                                     os.path.join(args.out, "trainer.pt"))]
                ckpt_flat = os.path.join(args.out, "model.pt")
                _publish_modes.append(publish_checkpoint(ckpt, ckpt_flat))
                rec['profile']['checkpoint_publish_s'] = time.monotonic()-_publish_start
                rec['profile']['checkpoint_publish_modes'] = _publish_modes
                rec['profile']['checkpoint_save_publish_s'] = time.monotonic()-_checkpoint_start
                # 先持久化本轮记录，再发布latest；中断后可补回尚未追加的history。
                write_summary(os.path.join(snap, "record.json"), to_jsonable(rec))
                manifest = {"version": ver, "trainer": os.path.join(snap, "trainer.pt"),
                            "model": os.path.join(snap, "model.pt"),
                            "cum_decisions": int(trainer.extra["cum_decisions"]),
                            "arch": arch, "seed": int(args.seed),
                            "elapsed_s": time.time() - t0,
                            "history_len": len(history)}
                with open(os.path.join(args.out, "latest.json.tmp"), "w",
                          encoding="utf-8") as f:
                    json.dump(manifest, f, ensure_ascii=False, indent=2)
                os.replace(os.path.join(args.out, "latest.json.tmp"),
                           os.path.join(args.out, "latest.json"))
                keep = int(args.keep_snapshots)
                if keep > 0:
                    snap_root = os.path.join(args.out, "snapshots")
                    olds = sorted(d for d in os.listdir(snap_root)
                                  if d.startswith("step_")
                                  and os.path.isdir(os.path.join(snap_root, d)))
                    for d in olds[:-keep]:
                        shutil.rmtree(os.path.join(snap_root, d), ignore_errors=True)

            # history：追加一行（O(1)），不再全量重写
            _th0 = time.time()
            _tl["t_history_done"] = time.monotonic()
            rec["profile"]["timeline"] = dict(_tl)
            rec["profile"]["collect_timeline"] = slim.get("timeline")
            # history.jsonl 的 eval 段：bot 面板 mean_rank + human 面板
            # human_mean_rank（结果到达后写入后续 iteration；绘图见
            # workflows/plot_quality_curves.py）。
            _eval_seg = _eval_snapshot()
            if _eval_seg:
                rec["eval"] = _eval_seg
            append_record(history_path, to_jsonable(rec))
            if str(args.behavior_mix_report):
                # 混合行为策略的逐 iteration 统计（ε 命中率 / 搜索覆盖率 / KL(μ‖π)
                # / 采集侧搜索耗时 / PPO 侧 IS 权重与 ESS）；默认关闭时不落盘。
                _bm_c = (collect_meta.get("behavior_mix")
                         if isinstance(collect_meta, dict) else None) or {}
                _bm_p = (stats.get("behavior_mix") or {}) if isinstance(stats, dict) else {}
                append_record(str(args.behavior_mix_report), to_jsonable({
                    "iter": int(i),
                    "decisions": int(cum_decisions),
                    "updates": int(stats.get("trainer_updates", 0) or 0),
                    "collect_s": float(_collect_s),
                    "collect": _bm_c,
                    "ppo": _bm_p,
                }))
            _history_s = time.time() - _th0

            cum_decisions += int(slim.get("decisions", 0))
            if cum_decisions >= next_eval:
                if eval_runner is not None:
                    # 异步：只花 ~0.3s 存 ckpt + spawn；评测在 cuda:4 独立跑，主循环继续
                    if not eval_stop.is_set():
                        _submit_eval(cum_decisions)
                else:
                    _run_eval(cum_decisions)
                # 动态课程：在主循环里**同步**跑一次真实强度口径面板（5 席 p0.9，~2–3 min），
                # 按 t15 Σ 变化加/减席；结果写 out/dynamic_curriculum.json（默认关闭时不执行）。
                if int(args.dyn_curriculum) > 0:
                    _dyn_step(cum_decisions)
                while next_eval <= cum_decisions:
                    next_eval += int(args.eval_every_decisions)
            last_len = time.time() - iter_start
            i += 1

    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if batch_server is not None:
            batch_server.close()
        if pool is not None:
            pool.close()
        if eval_runner is not None:
            # 默认直接终止在跑的评测（不拖 wall）；--eval-drain 1 才等它落 history
            eval_runner.close(drain=bool(int(args.eval_drain)))

    # 收尾：导出最终模型（评测/续训都要用）
    try:
        from hsbg.az.train.checkpoint_io import publish_final_checkpoint
        publish_final_checkpoint(args.out, trainer, model,
                                 meta={**meta, 'arch': arch, 'trainer_updates': int(trainer.updates)},
                                 norm_mean=norm_mean, norm_std=norm_std,
                                 record=to_jsonable(history[-1]) if history else None,
                                 seed=args.seed, elapsed_s=time.time()-t0, history_len=len(history))
    except Exception as exc:  # noqa: BLE001
        print(f"[error] final save failed: {exc!r}; retained snapshots are authoritative", flush=True)
        raise
    summary = {"iterations": i, "updates": trainer.updates,
               "elapsed_s": time.time() - t0, "config": vars(args),
               "history_len": len(history),
               "evals": (eval_runner.results_snapshot() if eval_runner is not None
                         else evals),
               "pipeline": {"enabled": pipeline_on,
                            "lag_max_allowed": 1},
               "last_records": history[-5:]}
    if eval_runner is not None:
        summary["eval_async"] = eval_runner.stats()
        summary["evals_skipped"] = list(eval_runner.skipped)
    if eval_state["stop_reason"]:
        summary["eval_stop_reason"] = eval_state["stop_reason"]
    if board_ramp.enabled:
        # 对手强度课程：配置 + 收尾实际生效值 + 自适应事件（默认关闭时 summary 不变）
        summary["board_ramp"] = {
            **board_ramp.describe(),
            "final": board_ramp.state(cum_decisions).as_dict(),
            "events": list(board_ramp.events),
            "cum_decisions": int(cum_decisions),
        }
    if seq_enabled:
        # 动作序列辅助任务：ramp 终点与累计 decisions（默认关闭时 summary 不变）
        summary["seq"] = {**seq_schedule.describe(),
                          "coef_final": float(seq_schedule.coef(cum_decisions)),
                          "cum_decisions": int(cum_decisions)}
    if bc_enabled:
        summary["bc"] = {
            **bc_schedule.describe(),
            "coef_final": float(bc_schedule.coef(cum_decisions)),
            "cum_decisions": int(cum_decisions),
            "batch": int(args.bc_batch),
            "every": int(args.bc_every),
            "compile": bool(int(args.bc_compile)),
            "type_weight": float(args.bc_type_weight),
            "card_weight": float(args.bc_card_weight),
            "class_weights": dict(bc_class_weights),
            "card_head": bool(bc_card_head),
            "pool": (bc_pool_summary(bc_pool) if bc_pool is not None else None),
        }
    if dpo_enabled:
        # upgrade 子集 DPO：池规模/系数/周期，以及最终 DPO 接受率（=1-选 upgrade 比例）
        summary["dpo"] = {
            "coef": float(args.dpo_coef), "beta": float(args.dpo_beta),
            "batch": int(args.dpo_batch), "every": int(args.dpo_every),
            "samples_path": str(args.dpo_samples),
            "pool_samples": (int(len(dpo_pool)) if dpo_pool is not None else 0),
        }
    write_summary(os.path.join(args.out, "summary.json"), to_jsonable(summary))
    print(f"[done] iters={i} updates={trainer.updates} "
          f"elapsed={time.time()-t0:.0f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
