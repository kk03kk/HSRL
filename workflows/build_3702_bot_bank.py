#!/usr/bin/env python3
"""采集新版 Bot EOT，并用冻结 Firestone 旧版均值校准。"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean, median
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows.build_human_curve_bank import allocate

CURVE = ROOT / "config/firestone-255790-mean.json"
SPLITS = ("train", "dev", "test")
DRIVER = "bot_firestone_mean_3702"
VERSION = "37.0.2.256589"


def coefficient(turn):
    if turn < 7:
        raise ValueError("第 1–6 回合保留 Bot 原值")
    return .5 if turn <= 12 else {13: .55, 14: .6}.get(turn, .65)


def totals(frame):
    board = frame["player"]["board"]
    return sum(m["atk"] for m in board), sum(m["hp"] for m in board)


def read_frames(paths, engine, dropped, expected_rules=None, expected_source_hash=None):
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            record = json.load(stream)
        if record["engine"] != engine or record["meta"]["driver"] != "bot":
            raise ValueError(f"快照引擎或来源不符 {path}")
        if record['meta'].get('max_actions_per_turn') != 80 or not record['meta'].get('rules_config'):
            raise ValueError(f'原始快照缺少当前操作预算/规则配置证明 {path}')
        if expected_source_hash is not None and record['meta'].get('engine_source_hash', expected_source_hash) != expected_source_hash:
            raise ValueError(f'原始Bot源码指纹不同 {path}')
        if expected_rules is not None and record['meta']['rules_config'] != expected_rules:
            raise ValueError(f'原始Bot规则配置不同 {path}')
        for frame in record["frames"]:
            turn = int(frame["turn"])
            size = len(frame["player"]["board"])
            if size < (3 if turn >= 10 else 1):
                if dropped is not None:
                    dropped[(turn, size)] += 1
                continue
            yield frame


def capture(root, start, count, workers):
    def batch(first):
        games = min(32, start + count - first)
        trajectory = root / "trajectories"
        subprocess.run([sys.executable, "workflows/capture_trajectory.py", "--driver", "bot",
                        "--device", "cpu", "--games", str(games), "--seed", str(first),
                        "--max-actions-per-turn", "80",
                        "--from-turn", "1", "--emit-snapshots", "1", "--out", str(trajectory),
                        "--snap-dir", str(root / "frames")], cwd=ROOT, check=True)
        for seed in range(first, first + games):
            with gzip.open(trajectory / f"traj_{seed:08d}.json.gz", "rt", encoding="utf-8") as stream:
                if json.load(stream)["meta"]["truncated"]:
                    raise RuntimeError(f"Bot 源对局截断 seed={seed}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(batch, range(start, start + count, 32)))


def calibrate(frame, factors):
    board = frame["player"]["board"]
    attack, health = totals(frame)
    fa, fh = factors
    aa = allocate([m["atk"] for m in board], round(attack * fa), 0)
    hh = allocate([m["hp"] for m in board], round(health * fh), 1)
    for minion, atk, hp in zip(board, aa, hh):
        old_hp = minion["hp"]
        minion["max_hp"] = max(hp, round(minion.get("max_hp", old_hp) * hp / old_hp))
        if minion["max_hp"] > 2**31-1:
            raise ValueError("max_hp 超出引擎范围")
        minion["atk"], minion["hp"] = atk, hp


def build(out, workers, smoke=False, profile="legacy-v1", seed_base=3000000,
          games=(2048, 256, 256), rounds=4, consistent_profile=False):
    import hsbg_engine as E
    from hsbg.az.train.mirror_snapshots import engine_stamp, write_snapshot_file
    from hsbg.az.train.personal import fingerprint, library_digest, write_json
    from workflows.mirror_hard_gates import _drive_to_turn

    engine, fp = engine_stamp(), fingerprint()
    if engine["game_version"] != VERSION:
        raise RuntimeError("必须先重建 37.0.2 原生扩展")
    out.mkdir(parents=True)
    refs = json.loads(CURVE.read_text(encoding="utf-8"))["turns"]
    template = _drive_to_turn(E.Game(seed=730000), 1).state_json()

    def inject(frame):
        payload = json.dumps(frame["player"], ensure_ascii=False)
        game = E.Game.from_json(template)
        result = game.apply_turn_snapshots([[0, payload]], False)
        if (not result["pool_balance_ok"] or not result["pool_species_ok"]
                or game.invariant_violations()):
            raise RuntimeError(f"注入破坏引擎不变量 {result}")
        return payload, result

    manifests = {}
    for index, split in enumerate(SPLITS):
        raw, dest = out / "raw" / split, out / split
        seed = seed_base + index * 1000000
        batch = 16 if smoke else games[index]
        threshold, required = (1, 7) if smoke else ((128, 15) if split == "train" else (32, 15))
        for offset in range(0, batch * (1 if smoke else rounds), batch):
            capture(raw, seed + offset, batch, workers)
            paths = sorted((raw / "frames").glob("snap_*.json.gz"))
            candidates, dropped, stale_refs = defaultdict(dict), Counter(), Counter()
            for frame in read_frames(paths, engine, dropped):
                turn = frame["turn"]
                game, seat = frame["source_game_seed"], frame["seat"]
                rank = hashlib.sha256(f"{turn}:{game}:{seat}".encode()).digest()
                previous = candidates[turn].get(game)
                if previous is None or rank < previous[0]:
                    _, result = inject(frame)
                    if result["applied"] == 0 and len(result["rejected"]) == 1 and (
                            str(result["rejected"][0][1]).startswith("unknown_ref:")):
                        stale_refs[turn] += 1
                        raise RuntimeError(f'新引擎仍生成悬空引用，暂停建库 seed={game} turn={turn} {result}')
                    if result["applied"] != 1 or result["rejected"]:
                        raise RuntimeError(f"原始 Bot 帧异常 seed={game} turn={turn} {result}")
                    candidates[turn][game] = rank, seat, totals(frame)
            last = 0
            while len(candidates[last + 1]) >= threshold and last < max(map(int, refs)):
                last += 1
            if last >= required:
                break
        if last < required:
            raise RuntimeError(f"{split} 只连续覆盖到第 {last} 回合，已达建库预算")
        dest.mkdir()
        # 每回合每源局只取一帧，控制 8 个 actor 各自加载完整库的常驻内存。
        limit = 512 if split == "train" else 128
        selected = {turn: dict(sorted(candidates[turn].items(), key=lambda item: item[1][0])[:limit])
                    for turn in range(1, last + 1)}
        targets, factors, observed, shard = {}, {}, defaultdict(list), []
        for turn in range(7, last + 1):
            n, a, h = refs[str(turn)]
            targets[turn] = a/n*coefficient(turn), h/n*coefficient(turn)
        if profile == "economy-window-v2" and split == "train":
            from workflows.derive_3702_economy_bank import targets as economy_targets
            targets.update(economy_targets(refs, last))
        for turn in targets:
            means = tuple(mean(values) for values in zip(*(v[2] for v in selected[turn].values())))
            if min(means) <= 0:
                raise ValueError(f"{split} 第 {turn} 回合 Bot 攻血均值不能缩放")
            factors[turn] = tuple(target / base for target, base in zip(targets[turn], means))

        def flush():
            if shard:
                write_snapshot_file(str(dest / f"snap_{sum(map(len, observed.values())):08d}.json.gz"),
                                    shard, engine=engine, meta={"driver": DRIVER, "split": split})
                shard.clear()

        for frame in read_frames(paths, engine, None):
            turn = frame["turn"]
            choice = selected.get(turn, {}).get(frame["source_game_seed"])
            if choice is None or choice[1] != frame["seat"]:
                continue
            if turn in factors:
                calibrate(frame, factors[turn])
            # 共用单桶使实际抽样和逐帧等权校准口径一致。原玩家对象保持完整。
            frame["source_seat"] = frame["seat"]
            frame["seat"] = 0
            payload, result = inject(frame)
            frame["frame_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
            if result["applied"] != 1 or result["rejected"]:
                raise RuntimeError(f"{split} seed={frame['source_game_seed']} turn={turn} {result}")
            observed[turn].append(totals(frame))
            shard.append(frame)
            if len(shard) == 512:
                flush()
        flush()
        report = {}
        for turn, rows in sorted(observed.items()):
            a, h = map(mean, zip(*rows))
            if turn in targets and any(abs(actual-target) > max(1, target*.02)
                                      for actual, target in zip((a, h), targets[turn])):
                raise ValueError(f"{split} 第 {turn} 回合校准均值超差")
            combined = sorted(a+h for a,h in rows)
            report[turn] = dict(frames=len(rows), games=len(selected[turn]),
                                available_games=len(candidates[turn]), attack=a, health=h,
                                target=targets.get(turn), coefficient=(sum(targets[turn])*refs[str(turn)][0]
                                    /sum(refs[str(turn)][1:])) if turn in targets else None,
                                human_frames=refs[str(turn)][0] if turn in targets else None,
                                median=median(combined),
                                p90=combined[(9*len(combined)-1)//10],
                                dropped_single=dropped[(turn, 1)], dropped_two=dropped[(turn, 2)],
                                raw_unknown_ref_excluded=stale_refs[turn])
        write_json(dest / "calibration.json", report)
        total = sum(map(len, observed.values()))
        manifests[split] = dict(schema="hsrl.library/v1", split=split, driver=DRIVER,
                                digest=library_digest(dest), fingerprint=fp, last_turn=last,
                                seed_start=seed, games=offset+batch, smoke=smoke,
                                curve_sha256=hashlib.sha256(CURVE.read_bytes()).hexdigest(),
                                calibration_profile=profile if (consistent_profile or split == "train") else "legacy-v1",
                                source_max_actions_per_turn=80,
                                rules_config=json.loads(E.Game(seed=0).state_json())["cfg"],
                                validation=dict(checked=total, accepted=total, quarantined=0))
        write_json(dest / "manifest.json", manifests[split])
    audit = strict_audit(out, manifests, smoke)
    write_json(out / "report.json", dict(complete=True, smoke=smoke, splits=manifests,
               calibration_profile=profile, profile_consistent=bool(consistent_profile),
               seed_base=seed_base, rounds=rounds, fresh_generation=True, strict_audit=audit))


def strict_audit(out, manifests, smoke, splits=SPLITS):
    from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
    from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
    from hsbg.az.train.personal import write_json
    from hsbg.az.train.policies import EngineBotPolicy

    audit = {}
    for split in splits:
        pool = MirrorSnapshotPool(str(out / split), sample="random")
        result = evaluate_policy(EngineBotPolicy(), EvalConfig(
            games=4 if smoke else 16, seed=9000000 + SPLITS.index(split)*1000,
            mirror_pool=pool, mirror_strict=True, collect_exposure=True))
        m, s = result["mirror"], result["mirror_strict_counts"]
        audit[split] = dict(mirror=m, strict=s, illegal_actions=result["illegal_actions"],
                            truncated_games=result["truncated_games"],
                            truncated_combats=result["truncated_combats"])
        write_json(out / "strict-audit.json", audit)
        if (pool.bad_files or pool.frames != manifests[split]["validation"]["accepted"]
                or any(audit[split][key] for key in ("illegal_actions", "truncated_games", "truncated_combats"))
                or any(m[key] for key in ("mirror_errors", "mirror_missing", "mirror_rejected",
                                          "mirror_fallback", "mirror_pool_unbalanced",
                                          "mirror_pool_species_unbalanced"))
                or not m["mirror_attempt"] == m["mirror_applied"] == s["eligible_live_seats"]
                or not s["live_combats"]):
            raise RuntimeError(f"{split} 完整对局严格注入验收失败")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--profile", choices=("legacy-v1", "economy-window-v2"), default="legacy-v1")
    parser.add_argument("--seed-base", type=int, default=3000000)
    parser.add_argument("--train-games", type=int, default=2048)
    parser.add_argument("--dev-games", type=int, default=256)
    parser.add_argument("--test-games", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--consistent-profile", action="store_true")
    args = parser.parse_args()
    if args.seed_base < 0 or min(args.train_games, args.dev_games, args.test_games) < 1 or args.rounds < 1:
        parser.error("seed-base/游戏数/轮数必须为正")
    build(args.out, args.workers, args.smoke, args.profile, args.seed_base,
          (args.train_games, args.dev_games, args.test_games), args.rounds, args.consistent_profile)
