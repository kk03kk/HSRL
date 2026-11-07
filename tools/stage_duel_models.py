#!/usr/bin/env python3
"""把评测要用的两个权重复制到上传目录，并逐一核对 SHA256/字节数。

为什么需要这一步：``artifacts/models/library-v1/models/<id>/model.pt`` 是指向原始权重的
**硬链接**，整目录接近 2.5 GB；NUS 上只需要这两个文件。本工具把 A/B 两个 ``model.pt``
复制到 ``artifacts/models/duel/<label>/model.pt``（即 ``config/eval-duel-*.json`` 的
``path``），复制后重新计算 SHA256 并与配置里的期望值比对，任何不一致立即失败。

用法::

    python tools/stage_duel_models.py                      # 用配置里的 library_path
    python tools/stage_duel_models.py --source-a /path/model.pt
    python tools/stage_duel_models.py --check-only         # 只核对已存在的目标文件

产物：``artifacts/models/duel/MANIFEST.json``（含 config SHA256、两份权重的 SHA256/字节数），
随权重一起上传；``artifacts/models/duel/`` 在 ``.gitignore`` 的 ``artifacts/`` 下，
**不会**进 Git。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DEFAULT = ROOT / "config/eval-duel-7357-vs-10004.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(CONFIG_DEFAULT))
    parser.add_argument("--source-a", default=None, help="覆盖 A 的源权重路径")
    parser.add_argument("--source-b", default=None, help="覆盖 B 的源权重路径")
    parser.add_argument("--check-only", action="store_true",
                        help="不复制，只核对配置里的目标文件")
    parser.add_argument("--force", action="store_true", help="目标已存在时覆盖")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    entries = []
    for label in ("A", "B"):
        model = cfg["models"][label]
        target = ROOT / model["path"]
        source = Path(args.source_a if label == "A" else args.source_b) \
            if (args.source_a and label == "A") or (args.source_b and label == "B") \
            else ROOT / str(model.get("library_path") or model["path"])
        expected_sha = str(model.get("file_sha256") or "")
        expected_bytes = int(model.get("bytes") or 0)

        if not args.check_only:
            if not source.is_file():
                raise SystemExit(f"缺少源权重 {label}: {source}（先跑 "
                                 "`python tools/export_model_library.py --cohort mainline-v1` "
                                 "或直接改 --source-{a,b}）")
            if target.exists() and not args.force:
                if sha256_file(target) != expected_sha:
                    raise SystemExit(f"目标已存在但 SHA 不符，保留现场：{target}；"
                                     "确认后用 --force 覆盖")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        if not target.is_file():
            raise SystemExit(f"缺少目标权重 {label}: {target}")

        actual_sha = sha256_file(target)
        actual_bytes = target.stat().st_size
        if expected_sha and actual_sha != expected_sha:
            raise SystemExit(f"{label} SHA256 不符：期望 {expected_sha}，实际 {actual_sha}")
        if expected_bytes and actual_bytes != expected_bytes:
            raise SystemExit(f"{label} 字节数不符：期望 {expected_bytes}，实际 {actual_bytes}")
        entries.append({
            "label": label,
            "id": model["id"],
            "path": model["path"],
            "source": str(source.relative_to(ROOT)) if str(source).startswith(str(ROOT))
            else str(source),
            "file_sha256": actual_sha,
            "bytes": int(actual_bytes),
        })
        print(f"[stage] {label} {model['id']}: {actual_bytes} bytes sha256={actual_sha}")

    manifest = {
        "schema": "hsai.eval-duel-models/v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": str(config_path.relative_to(ROOT)) if str(config_path).startswith(str(ROOT))
        else str(config_path),
        "config_sha256": sha256_text(config_path),
        "models": entries,
        "note": "上传这两份 model.pt 后，在 NUS 上跑 `python tools/stage_duel_models.py "
                "--check-only` 复核；SHA 不一致的权重不能进评测。",
    }
    out = ROOT / "artifacts/models/duel/MANIFEST.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[stage] manifest -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
