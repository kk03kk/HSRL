"""Run the whole Python suite from an extracted release, isolated from the development tree.

The current built extension is copied into the extracted source to represent installation.
This checks distribution completeness; Linux/CUDA build compatibility remains a server gate.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive")
    p.add_argument("--out", required=True, help="new isolated validation directory")
    p.add_argument("--native-extension", required=True)
    args = p.parse_args()
    target = Path(args.out).resolve()
    if target.exists(): raise FileExistsError("validation must start with a pristine directory")
    target.mkdir(parents=True)
    with tarfile.open(args.archive) as archive:
        archive.extractall(target, filter="data")
    root = target / "AutoTavern"
    binary = Path(args.native_extension).resolve()
    shutil.copy2(binary, root / "python/taverngym" / binary.name)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "python")
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    code = r'''
import os, sys
from pathlib import Path
handles = []
if os.name == "nt":
    handles = [os.add_dll_directory(p) for p in os.environ.get("HSRL_WINDOWS_DLL_DIRS", "").split(os.pathsep) if p]
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "python"))
import taverngym
assert taverngym.REPO_ROOT == Path.cwd(), (taverngym.REPO_ROOT, Path.cwd())
assert taverngym.build_is_current() is True
from tools.package_nus import verify
verify(Path.cwd())
import pytest
raise SystemExit(pytest.main(["tests", "-q", "-p", "no:cacheprovider"]))
'''
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                            cwd=root, env=environment)
    raise SystemExit(result.returncode)


if __name__ == "__main__": main()
