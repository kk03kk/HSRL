"""跨平台离线工具启动器。Windows DLL 目录仅为本机运行环境，不是参考代码依赖。"""
import os
from pathlib import Path
import runpy
import sys


def prepare():
    root = Path(__file__).resolve().parents[1]
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    handles = []
    if os.name == "nt":
        candidates = [Path(sys.prefix) / "Library/bin", Path(sys.prefix)]
        candidates += list((root.parent / ".audit_tools").glob("llvm-mingw*/bin"))
        candidates += [Path(p) for p in os.environ.get("TAVERNGYM_DLL_DIRS", "").split(os.pathsep) if p]
        for folder in candidates:
            if folder.is_dir():
                handles.append(os.add_dll_directory(str(folder)))
        os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    sys.path.insert(0, str(root / "python"))
    sys.path.insert(0, str(root))
    return handles


def main():
    handles = prepare()
    if len(sys.argv) > 2 and sys.argv[1] == "--baseline":
        baseline = Path(sys.argv[2]).resolve()
        if not (baseline / "python/taverngym/rl/config.py").is_file():
            raise ValueError("baseline must contain frozen source and native extension")
        del sys.argv[1:3]
        sys.path.insert(0, str(baseline / "python"))
        import taverngym
        import taverngym.rl
        # 旧模型仍由冻结的rl/原生扩展解释，审阅工具由当前主项目提供。
        taverngym.__path__.insert(0, str(Path(__file__).resolve().parents[1] / "python/taverngym"))
        # 显式冻结rl路径，避免上面的工具目录优先级改变模型语义。
        taverngym.rl.__path__[:] = [str(baseline / "python/taverngym/rl")]
    sys.argv = ["taverngym.offline", *sys.argv[1:]]
    runpy.run_module("taverngym.offline", run_name="__main__")


if __name__ == "__main__":
    main()
