"""比较已建库的 Bot 原始阵容、Firestone 参考和正式校准模板。"""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

CURVE = Path(__file__).resolve().parents[1] / "config/firestone-255790-mean.json"


def records(directory):
    for path in sorted(directory.glob("snap_*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            yield from json.load(stream)["frames"]


def add(groups, frame):
    board = frame["player"]["board"]
    row = groups[frame["turn"]]
    row[0] += 1
    row[1] += sum(m["atk"] for m in board)
    row[2] += sum(m["hp"] for m in board)


def stats(row):
    if not row or not row[0]:
        return None
    n, a, h = row
    return dict(frames=n, attack=a/n, health=h/n, strength=(a+h)/n)


def compare(bank):
    human = json.loads(CURVE.read_text(encoding="utf-8"))
    manifest = json.loads((bank / "train/manifest.json").read_text(encoding="utf-8"))
    report = json.loads((bank / "report.json").read_text(encoding="utf-8"))
    raw_bank = Path(report.get("raw_source_bank", bank))
    if raw_bank != bank:
        source = json.loads((raw_bank / "train/manifest.json").read_text(encoding="utf-8"))
        if source["digest"] != manifest["parent_digest"]:
            raise ValueError("原始 Bot 库与派生来源不符")
    template, raw, matched = (defaultdict(lambda: [0, 0, 0]) for _ in range(3))
    selected = set()
    for frame in records(bank / "train"):
        key = (frame["source_game_seed"], frame["turn"], frame["source_seat"])
        if key in selected:
            raise ValueError(f"重复校准帧 {key}")
        selected.add(key)
        add(template, frame)
    if len(selected) != manifest["validation"]["accepted"]:
        raise ValueError("校准帧数量与清单不符")
    for frame in records(raw_bank / "raw/train/frames"):
        add(raw, frame)
        key = (frame["source_game_seed"], frame["turn"], frame["seat"])
        if key in selected:
            selected.remove(key)
            add(matched, frame)
    if selected:
        raise ValueError(f"{len(selected)} 张校准帧缺少原始 Bot 帧")

    reference = {int(turn): row for turn, row in human["turns"].items()}
    turns = {turn: dict(bot_raw=stats(raw.get(turn)), bot_matched=stats(matched.get(turn)),
                        firestone=stats(reference.get(turn)), calibrated=stats(template.get(turn)))
             for turn in sorted(set(raw) | set(reference) | set(template))}
    return dict(bank_digest=manifest["digest"], firestone_archive_sha256=human["archive_sha256"],
                calibrated_last_turn=manifest["last_turn"], turns=turns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.bank), ensure_ascii=False, indent=2))
