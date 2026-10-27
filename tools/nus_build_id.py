"""Native build cache key based on contents, never archive extraction timestamps."""
import argparse
import hashlib
from pathlib import Path


def build_id(root):
    root = Path(root).resolve()
    if not (root / "Cargo.toml").is_file() or not (root / "crates").is_dir():
        raise ValueError("not an AutoTavern source workspace")
    files = [root / name for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "pyproject.toml")
             if (root / name).is_file()]
    files.extend(p for p in (root / "crates").rglob("*")
                 if p.is_file() and p.suffix in {".rs", ".json", ".toml", ".lock"}
                 and not {"target", ".git", "__pycache__"}.intersection(p.relative_to(root).parts))
    h = hashlib.sha256()
    for p in sorted(files):
        h.update(p.relative_to(root).as_posix().encode("utf-8") + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()[:24]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    print(build_id(p.parse_args().root))
