"""冻结主项目源码、原生扩展及已校验 checkpoint；绝不覆盖既有快照。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from local_offline import prepare


def main():
    handles = prepare()
    import taverngym as tg
    from taverngym.offline.evaluate import checkpoint
    from taverngym.rl.config import identity
    from tools.package_nus import ROOT, payload
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out).resolve()
    if not out.is_relative_to((ROOT / "runs").resolve()):
        p.error("baseline must be inside the main project's runs directory")
    cfg, _ = checkpoint(args.checkpoint)
    out.mkdir(parents=True, exist_ok=False)
    files = payload()
    native = Path(tg.build_info()["file"])
    files[f"python/taverngym/{native.name}"] = native.read_bytes()
    files["baseline-checkpoint.pt"] = Path(args.checkpoint).read_bytes()
    for name, data in files.items():
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    manifest = {"identity": identity(cfg), "native": tg.build_info(),
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    (out / "FROZEN_BASELINE.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": str(out), "files": len(files), "identity": identity(cfg)["digest"]}))


if __name__ == "__main__":
    main()
