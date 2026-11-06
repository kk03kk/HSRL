"""仅迁移 entity checkpoint 内嵌卡表，保留学习参数及完整训练器状态。"""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def migrate(model, out, trainer=None):
    import torch
    from hsbg.az.train.entity_model import load_card_tables
    from hsbg.az.train.personal import fingerprint, sha256, write_json
    from tools.p2_data.check_carddb_3702 import VOCAB

    fp = fingerprint()
    if fp["vocab_sha256"] != VOCAB or fp["engine"]["game_version"] != "37.0.2.256589":
        raise ValueError("迁移只接受已验证的 256589 有序词表")
    inputs = {"model.pt": sha256(model)}
    if trainer:
        inputs["trainer.pt"] = sha256(trainer)
    manifest = out / "migration.json"
    if manifest.exists():
        import json
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if (record["inputs"] != inputs or record["fingerprint"] != fp or
                record["outputs"] != {name: sha256(out / name) for name in inputs}):
            raise ValueError("迁移缓存来源或摘要不同")
        return record
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("保留未完成迁移证据，使用新目录")

    payload = torch.load(model, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    full = torch.load(trainer, map_location="cpu", weights_only=False) if trainer else None
    if full is not None and (state.keys() != full["model"].keys() or
            any(not torch.equal(value, full["model"][key]) for key, value in state.items())):
        raise ValueError("源 model 与完整 Trainer 模型不一致")
    tables = load_card_tables()
    for key, table in (("card_static", tables.static), ("card_text", tables.text)):
        new = torch.from_numpy(table)
        old = state[key]
        if old.shape != new.shape or old.dtype != new.dtype:
            raise ValueError(f"卡表 buffer 布局发生变化 {key}")
        state[key] = new
        if full is not None:
            full["model"][key] = new
            if full.get('anchor_model') is not None:
                anchor = full['anchor_model'][key]
                if anchor.shape != new.shape or anchor.dtype != new.dtype:
                    raise ValueError(f'anchor卡表buffer布局发生变化 {key}')
                full['anchor_model'][key] = new.clone()

    out.mkdir(parents=True, exist_ok=True)
    for name, value in (("model.pt", payload), ("trainer.pt", full)):
        if value is not None:
            tmp = out / (name + ".tmp")
            torch.save(value, tmp)
            os.replace(tmp, out / name)
    record = dict(inputs=inputs, outputs={name: sha256(out / name) for name in inputs},
                  fingerprint=fp, replaced_buffers=["card_static", "card_text"])
    write_json(manifest, record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--trainer", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    migrate(args.model, args.out, args.trainer)
