#!/usr/bin/env python3
"""用Rust实际注入逐帧验证Train库，输出隔离无效帧的新版本；不修改源库。"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare(source, out, *, derived_dev=False):
    import hsbg_engine as E
    from hsbg.az.train.personal import checked_library, library_digest, write_json
    from workflows.mirror_hard_gates import _drive_to_turn
    source, out = Path(source), Path(out)
    manifest = checked_library(source)
    if manifest["split"] != "train" and not (derived_dev and manifest["split"] == "dev"):
        raise ValueError("只派生Train库；Dev/Test必须保持冻结")
    if out.exists():
        raise FileExistsError("派生库目录必须不存在")
    # 标准存活、无pending、未ended的接收座位；每帧从同一状态克隆。
    template = _drive_to_turn(E.Game(seed=730000), 1).state_json()
    records, failures, reasons = [], [], Counter()
    total = accepted = 0
    for path in sorted(source.glob("snap_*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            record = json.load(stream)
        keep = []
        for index, frame in enumerate(record["frames"]):
            total += 1
            game = E.Game.from_json(template)
            payload = json.dumps(frame["player"], ensure_ascii=False)
            before = game.state_json()
            report = game.apply_turn_snapshots([[0, payload]], False)
            if report["applied"] == 1 and not report["rejected"]:
                if game.invariant_violations() or not report["pool_balance_ok"] or not report["pool_species_ok"]:
                    raise RuntimeError("有效帧注入破坏不变量，停止派生")
                keep.append(frame)
                accepted += 1
            else:
                if before != game.state_json():
                    raise RuntimeError("Rust拒绝改变状态，停止派生")
                reason = str(report["rejected"][0][1])
                reasons[reason.split(":")[0]] += 1
                failures.append({"source_file": path.name, "frame_index": index,
                    "turn": frame["turn"], "seat": frame["seat"],
                    "source_game_seed": frame.get("source_game_seed"), "reason": reason})
        record["frames"] = keep
        records.append((path.name, record))
    if accepted == 0:
        raise ValueError("没有可接受帧")
    out.mkdir(parents=True)
    for name, record in records:
        with gzip.open(out / name, "wt", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False)
    validation = {"schema": "hsrl.frame_validation/v1", "source_digest": manifest["digest"],
        "checked": total, "accepted": accepted, "quarantined": len(failures),
        "reasons": dict(reasons), "failures": failures,
        "note": "仅隔离引擎拒绝的完整帧，不修剪引用或修改有效帧；不能替代完整对局硬门。",
        "derived_dev": bool(derived_dev)}
    write_json(out / "validation.json", validation)
    from hsbg.az.train.mirror_snapshots import MirrorSnapshotPool
    write_json(out / "manifest.json", {**manifest, "digest": library_digest(out),
        "stats": MirrorSnapshotPool(str(out), sample="random").stats(),
        "parent_digest": manifest["digest"], "validation": {"checked": total,
            "quarantined": len(failures), "accepted": accepted}})
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--derived-dev", action="store_true", help="只在全新路径建立独立Dev派生版本，绝不修改原Dev或访问Test")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.out, derived_dev=args.derived_dev), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
