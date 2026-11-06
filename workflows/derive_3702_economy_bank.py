"""从已验收的 37.0.2 Bot 库派生较缓的 Train 曲线。"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from statistics import mean, median
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows.build_3702_bot_bank import CURVE, DRIVER, calibrate, totals, strict_audit

PROFILE = "economy-window-v2"
RATIOS = {2: 1., 9: .40, 10: .35, 11: .30, 12: .22, 13: .17, 14: .17, 15: .16}


def targets(human, last):
    result = {}
    for turn in range(1, last + 1):
        if turn not in RATIOS and turn < 16:
            continue
        n, attack, health = human[str(turn)]
        ratio = RATIOS.get(turn, .15)
        attack, health = attack/n*ratio, health/n*ratio
        if turn >= 16:
            previous = sum(result[turn-1])
            bounded = min(previous*1.25, max(previous, attack+health))
            attack, health = attack*bounded/(attack+health), health*bounded/(attack+health)
        result[turn] = attack, health
    return result


def derive(source, out):
    import hsbg_engine as E
    from hsbg.az.train.personal import checked_library, library_digest, write_json
    from workflows.mirror_hard_gates import _drive_to_turn

    source, out = source.resolve(), out.resolve()
    report = json.loads((source/"report.json").read_text(encoding="utf-8"))
    if not report["complete"] or report.get("raw_source_bank"):
        raise ValueError("需要已发布的原版 Bot 校准库")
    manifests = {split: checked_library(source/split) for split in ("train", "dev", "test")}
    if manifests != report["splits"] or manifests["train"]["driver"] != DRIVER:
        raise ValueError("源库报告或来源不符")
    old = manifests["train"]
    human = json.loads(CURVE.read_text(encoding="utf-8"))["turns"]
    wanted = targets(human, old["last_turn"])
    calibration = json.loads((source/"train/calibration.json").read_text(encoding="utf-8"))
    factors = {turn: tuple(target/calibration[str(turn)][name]
                          for target, name in zip(pair, ("attack", "health")))
               for turn, pair in wanted.items()}

    dest = out/"train"
    dest.mkdir(parents=True)
    template = _drive_to_turn(E.Game(seed=730000), 1).state_json()
    observed = defaultdict(list)
    for path in sorted((source/"train").glob("snap_*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            shard = json.load(stream)
        for frame in shard["frames"]:
            turn = frame["turn"]
            if turn in factors:
                calibrate(frame, factors[turn])
            payload = json.dumps(frame["player"], ensure_ascii=False)
            frame["frame_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
            game = E.Game.from_json(template)
            r = game.apply_turn_snapshots([[0, payload]], False)
            if (r["applied"] != 1 or r["rejected"] or not r["pool_balance_ok"]
                    or not r["pool_species_ok"] or game.invariant_violations()):
                raise RuntimeError(f"派生帧注入失败 {turn} {frame['source_game_seed']} {r}")
            observed[turn].append(totals(frame))
        with gzip.open(dest/path.name, "wt", encoding="utf-8") as stream:
            json.dump(shard, stream, ensure_ascii=False)

    for turn, values in sorted(observed.items()):
        a, h = map(mean, zip(*values))
        row = calibration[str(turn)]
        if len(values) != row["frames"] or any(abs(x-y) > max(1., .02*y)
                    for x, y in zip((a,h), wanted.get(turn, (row["attack"], row["health"])))):
            raise ValueError(f"第 {turn} 回合派生帧计数或目标均值超差")
        combined = sorted(x+y for x,y in values)
        row.update(attack=a, health=h, target=wanted.get(turn),
                   coefficient=sum(wanted[turn])/sum(human[str(turn)][1:])*human[str(turn)][0]
                   if turn in wanted else None, median=median(combined),
                   p90=combined[(9*len(combined)-1)//10])
    if sum(map(len, observed.values())) != old["validation"]["accepted"]:
        raise ValueError("派生库未保持原有 Train 帧数量")
    write_json(dest/"calibration.json", calibration)
    manifests["train"] = {**old, "digest": library_digest(dest), "parent_digest": old["digest"],
                          "calibration_profile": PROFILE,
                          "calibration_sha256": hashlib.sha256((dest/"calibration.json").read_bytes()).hexdigest()}
    write_json(dest/"manifest.json", manifests["train"])
    for split in ("dev", "test"):
        shutil.copytree(source/split, out/split)
    audit = strict_audit(out, manifests, report["smoke"], splits=("train",))
    audit = {**report["strict_audit"], **audit}
    write_json(out/"strict-audit.json", audit)
    write_json(out/"report.json", {**report, "splits": manifests, "raw_source_bank": str(source),
                "strict_audit": audit, "calibration_profile": PROFILE})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    derive(args.source, args.out)
