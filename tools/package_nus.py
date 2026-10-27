"""Produce/verify a source-only NUS release; no Git remote or reference repo required."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ("crates", "python", "tests", "configs", "deploy", "requirements", "tavernbench", "tools")
ROOT_FILES = ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "pyproject.toml", "LICENSE",
              "NOTICE", "README.md", "AGENTS.md", "PROJECT_WORKSPACE.md", "project-workspace.json",
              ".gitignore", ".gitattributes", "docs/NUS_PPO_START.md", "tools/package_nus.py",
              "tools/validate_ppo_resume.py")
ROOT_FILES += ("docs/NUS_REPAIR2_START.md", "docs/research/全面修复与复审_2026-10-27.md")
ROOT_FILES += ("docs/research/NUS增量构建缓存修复_2026-10-27.md",)
ROOT_FILES += ("docs/LOCAL_OFFLINE.md", "docs/research/本地离线闭环交付_2026-10-27.md",
               "docs/research/local-offline-2026-10-27/issues.json",
               "docs/research/local-offline-2026-10-27/INPUT_CONTRACT.md")
ROOT_FILES += ("docs/NUS_REPAIR3_START.md", "docs/research/NUS1022804_评测截断修复.md")
ROOT_FILES += ("docs/NUS_PAPER_STAGE_START.md", "docs/research/论文启发实验阶段交付_2026-10-27.md",
               "docs/research/paper-2610.33845/论文分析与训练路线调整.md")
ROOT_FILES += ("docs/NUS_PILOT_RECOVERY.md", "docs/research/首模型实证排查与恢复训练_2026-10-27.md")
ROOT_FILES += ("docs/NUS_ENTITY_LIBRARY_START.md", "assets/entity-v2/rules-lsa128.npz",
               "assets/entity-v2/rules-lsa128.json", "assets/library/manual-game1.json")
ROOT_FILES += ("docs/TRAINING_ACTION_BUDGET.md", "docs/PATCH_37_VERIFICATION.md",
               "docs/TRAINING_PIPELINE_CURRENT.md")
EXCLUDED = {"__pycache__", ".pytest_cache", ".git", "target", ".venv"}
SUFFIXES = {".pyc", ".pyd", ".so", ".dll", ".dylib", ".whl", ".pt", ".npz", ".log"}
SUFFIXES |= {".pem", ".key", ".p12", ".pkl", ".pth", ".ckpt"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def payload(root=ROOT):
    paths = [root / p for p in ROOT_FILES if (root / p).is_file()]
    for folder in PREFIXES:
        paths.extend(p for p in (root / folder).rglob("*") if p.is_file()
                     and not (set(p.relative_to(root).parts) & EXCLUDED) and p.suffix not in SUFFIXES
                     and not p.name.startswith(".env"))
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(set(paths))}


def build(path, base_manifest=None):
    data = payload()
    for name in ("configs/ppo_nus.json", "deploy/nus/train.sbatch", "docs/NUS_PPO_START.md",
                 "tools/overlays/36.6.3.json", "tools/gen_carddb.py", "deploy/nus/rebuild.sbatch",
                 "deploy/nus/qualification.sbatch", "tests/fixtures/cross_seat_choice.json",
                 "docs/NUS_REPAIR2_START.md", "tools/nus_build_id.py",
                 "assets/entity-v2/rules-lsa128.npz", "assets/library/manual-game1.json",
                 "configs/entity_library_nus.json", "deploy/nus/library_pilot.sbatch"):
        if name not in data: raise FileNotFoundError(name)
    try:
        base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError): base = "unavailable"
    manifest = {"format": 1, "upstream_base": base, "ownership": "personal project",
                "files": {name: digest(b) for name, b in data.items()}}
    data["SOURCE_MANIFEST.json"] = json.dumps(manifest, indent=2, ensure_ascii=False).encode()
    if base_manifest is not None:
        baseline_bytes = Path(base_manifest).read_bytes()
        baseline = json.loads(baseline_bytes)["files"]
        removed = sorted(set(baseline) - set(manifest["files"]))
        if removed:
            raise ValueError(f"delta cannot delete source files: {removed}")
        changed = sorted(name for name, sha in manifest["files"].items() if baseline.get(name) != sha)
        data = {name: data[name] for name in changed + ["SOURCE_MANIFEST.json"]}
        data["PATCH_MANIFEST.json"] = json.dumps({"format": 1,
            "base_manifest_sha256": digest(baseline_bytes), "changed_files": changed,
            "instructions": "verify original tree, extract into parent of AutoTavern, verify updated tree"},
            indent=2).encode()
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, b in sorted(data.items()):
            info = tarfile.TarInfo("AutoTavern/" + name)
            info.size, info.mtime, info.mode = len(b), 0, 0o644
            archive.addfile(info, io.BytesIO(b))
    checksum = digest(path.read_bytes())
    path.with_name(path.name + ".sha256").write_text(f"{checksum}  {path.name}\n", encoding="ascii")
    print(json.dumps({"archive": str(path), "sha256": checksum, "files": len(data),
                      "bytes": path.stat().st_size}, ensure_ascii=False))
    return path


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        p = (root / name).resolve()
        if not p.is_relative_to(root.resolve()) or not p.is_file() or digest(p.read_bytes()) != expected:
            raise ValueError(f"release source changed or missing: {name}")
    print(f"Source manifest verified: {len(manifest['files'])} files")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(ROOT / "runs/dist/AutoTavern-nus.tar.gz"))
    p.add_argument("--verify", metavar="ROOT")
    p.add_argument("--delta-from", metavar="SOURCE_MANIFEST", help="source-only repair for a verified deployed release")
    a = p.parse_args()
    verify(a.verify) if a.verify else build(a.out, a.delta_from)
