#!/usr/bin/env python3
"""检查/清除代码目录里的 UTF-8 BOM（U+FEFF）。

为什么要单独一个工具：BOM 本身是合法的 UTF-8 编码（Python 解释器也接受），但它会

1. 让 ``text.read_text(encoding="utf-8") + ast.parse`` 这种**按 utf-8 读再解析**的检查器
   抛 ``SyntaxError: invalid non-printable character U+FEFF``（2026-11-07 NUS CPU 作业
   1092598 就是这样在 15 秒内失败）；
2. 改变源码字节，从而改变 ``training_source_sha256`` / ``runner_sha256`` 这类按字节
   （LF 归一化）计算的摘要，让 GPU 验收报告与当前源码对不上；
3. 在 diff 里制造看不见的改动。

代码目录（``hsbg`` / ``workflows`` / ``tests`` / ``tools`` / ``client`` / ``deploy``）
一律**拒绝** BOM；``records/`` 与 ``artifacts/`` 里的历史证据**不碰**（只读证据）。

用法::

    python tools/strip_source_bom.py            # 只报告（有 BOM 时退出码 1）
    python tools/strip_source_bom.py --fix      # 就地去掉 BOM（保留其余字节）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
BOM = b"\xef\xbb\xbf"
#: 受管代码目录（相对仓库根）。历史证据目录 records/ 与 artifacts/ 故意不在其中。
CODE_DIRS = ("hsbg", "workflows", "tests", "tools", "client", "deploy")
SUFFIXES = (".py", ".sh", ".sbatch")


def iter_sources(root: Path = ROOT) -> List[Path]:
    """代码目录下所有受管源码（排序、去重）。"""
    paths: List[Path] = []
    for name in CODE_DIRS:
        base = Path(root) / name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUFFIXES:
                paths.append(path)
    return sorted(set(paths))


def find_bom(paths: Iterable[Path], root: Path = ROOT) -> List[str]:
    """返回带 BOM 的文件（相对仓库根的 posix 路径）。"""
    out: List[str] = []
    for path in paths:
        try:
            with open(path, "rb") as stream:
                if stream.read(3) == BOM:
                    out.append(Path(path).resolve().relative_to(Path(root).resolve()
                                                              ).as_posix())
        except OSError:  # 读不了就交给别的门禁（不要在这里静默当成"没 BOM"）
            raise
    return out


def strip_bom(path: Path) -> bool:
    """去掉文件开头的 BOM；返回是否真的改过（幂等）。"""
    data = Path(path).read_bytes()
    if not data.startswith(BOM):
        return False
    Path(path).write_bytes(data[len(BOM):])
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fix", action="store_true", help="就地去掉 BOM")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    paths = iter_sources(args.root)
    bad = find_bom(paths, args.root)
    if not bad:
        print(f"[bom] 干净：{len(paths)} 个受管源码没有 BOM")
        return 0
    print(f"[bom] 发现 {len(bad)} 个带 BOM 的文件：")
    for rel in bad:
        print("  ", rel)
    if not args.fix:
        print("[bom] 用 `python tools/strip_source_bom.py --fix` 修复（不要手改编码）")
        return 1
    fixed = sum(1 for rel in bad if strip_bom(Path(args.root) / rel))
    print(f"[bom] 已去掉 {fixed} 个文件的 BOM")
    remaining = find_bom(iter_sources(args.root), args.root)
    if remaining:
        print("[bom] 仍有残留:", remaining)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
