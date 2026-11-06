"""把同版本 Firestone 可见战团转换成逐帧验证的严格 EOT 训练库。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsbg.az.train.mirror_snapshots import STAGE_EOT, engine_stamp, write_snapshot_file
from hsbg.az.train.personal import fingerprint, library_digest, write_json
from workflows.compare_board_libraries import install_board
from workflows.mirror_hard_gates import _drive_to_turn
from workflows.run_learning_experiment import digest, read

DRIVER = "firestone_visible_eot"
BUILD = 255790


def convert_split(source, target, split, template, engine, expected_sha):
    import hsbg_engine as E

    if digest(source) != expected_sha:
        raise ValueError(f"{split} 真人源文件摘要不符")
    counts, turns = Counter(), Counter()
    games = {}
    frames = []
    shards = 0
    target.mkdir(parents=True)

    def flush():
        nonlocal shards
        if frames:
            write_snapshot_file(str(target / f"snap_{shards:04d}.json.gz"), frames,
                                engine=engine, meta={"driver": DRIVER, "build": BUILD, "split": split})
            frames.clear()
            shards += 1

    with source.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (row["build_number"] != BUILD or row["split"] != split
                    or row["source"] != "firestone_pms_exact_turn"):
                raise ValueError("仅接受同版本、同分割、精确回合的 Firestone PMS 战团")
            counts["observations"] += 1
            if row["is_ghost"]:
                counts["ghost_excluded"] += 1
                continue
            turn = int(row["turn"])
            if turn <= 0 or not row["board_detail"]:
                raise ValueError("Firestone 源帧回合或阵容为空")
            counts["missing_tier"] += int(row.get("tier") is None)
            sandbox = E.Game.from_json(template)
            install_board(sandbox, 0, row)
            player = json.loads(sandbox.player_json(0))
            for field in ("hand", "shop", "hand_queue", "hand_queue_marks"):
                player[field] = []
            player["gold"] = 0
            player["ended_turn"] = True
            payload = json.dumps(player, ensure_ascii=False)
            receiver = E.Game.from_json(template)
            before = receiver.pool_balance(), receiver.pool_species_violations()
            applied = receiver.apply_turn_snapshots([[0, payload]], False)
            actual = json.loads(receiver.player_json(0))
            visible = ("card", "atk", "hp", "kw", "bonus_kw", "golden", "shield_max")
            if (applied["applied"] != 1 or applied["rejected"] or not actual["ended_turn"]
                    or receiver.pool_balance() != before[0]
                    or receiver.pool_species_violations() != before[1]
                    or receiver.invariant_violations()
                    or len(actual["board"]) != len(row["board_detail"])
                    or any(a[key] != b[key] for a, b in zip(actual["board"], row["board_detail"])
                           for key in visible)):
                raise RuntimeError(f"{split} turn={turn} line={counts['observations']} EOT 校验失败：{applied}")
            game_index = games.setdefault(row["game_id"], len(games))
            # 真人PMS不是模拟器座位分桶，全部写入共享桶；严格镜像按同回合抽取。
            frames.append(dict(turn=turn, seat=0, stage=STAGE_EOT, player=player, held_pool_parts=[],
                               source_game_seed=game_index, pool_mode="unknown", source=DRIVER,
                               frame_sha256=hashlib.sha256(payload.encode()).hexdigest()))
            counts["accepted"] += 1
            turns[turn] += 1
            if len(frames) >= 512:
                flush()
    flush()
    if not counts["accepted"] or set(turns) != set(range(1, max(turns) + 1)):
        raise ValueError(f"{split} 有缺失的完整回合")
    manifest = dict(schema="hsrl.library/v1", split=split, driver=DRIVER, build_number=BUILD,
                    fidelity="board_only_eot_approx", digest=library_digest(target),
                    fingerprint=fingerprint(), source_sha256=expected_sha, games=len(games),
                    validation=dict(accepted=counts["accepted"], quarantined=0, checked=counts["accepted"]))
    write_json(target / "manifest.json", manifest)
    return dict(counts, games=len(games), turns=dict(sorted(turns.items())), digest=manifest["digest"]), set(games)


def build(source, out):
    import hsbg_engine as E

    source_manifest = read(source / "manifest.json")
    engine = engine_stamp()
    if (source_manifest["build_number"] != BUILD
            or engine["game_version"] != f"37.0.0.{BUILD}"):
        raise ValueError("Firestone 与 Rust 引擎版本必须同为 255790")
    if out.exists():
        raise FileExistsError("真人 EOT 输出目录必须是新目录")
    template = _drive_to_turn(E.Game(seed=730000), 1).state_json()
    result, all_games = {}, set()
    for split in ("train", "dev", "test"):
        name = f"{split}.jsonl"
        result[split], games = convert_split(source / name, out / split, split, template, engine,
                                             source_manifest["files"][name]["sha256"])
        if all_games & games:
            raise ValueError("真人源对局跨分割重叠")
        all_games |= games
        print(split, result[split], flush=True)
    write_json(out / "report.json", dict(schema="hsrl.firestone_eot/v1", complete=True,
               build_number=BUILD, source_manifest_sha256=digest(source / "manifest.json"),
               libraries={k: row["digest"] for k, row in result.items()}, splits=result,
               note="原始攻血，可见板EOT近似；幽灵观测保留在原始库，排除出活席抽样。"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    build(args.source, args.out)
