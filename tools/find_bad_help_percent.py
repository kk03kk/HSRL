"""一次性诊断：找出 argparse ``help=`` 字符串里**未转义**的百分号。

argparse 会对 help 文本做 ``%``-格式化（``_expand_help``），所以任何单个 ``%``
后面跟着非格式说明符的字符都会让 ``--help`` / 报错信息直接抛
``ValueError: unsupported format character``。这个脚本用 AST 精确列出所有位置。

用法：``python tools/find_bad_help_percent.py workflows/train_entity_sync.py``
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

#: 合法的 %-格式说明符（argparse 里实际只会用到这些）
OK_SPEC = re.compile(r"%(?:%|\([A-Za-z_]+\)[sdirf]|[-+ #0]*\d*(?:\.\d+)?[sdirfeEgGxXoc])")


class _AnyDict(dict):
    """任何键都返回 1：让 ``%(default)s`` 这类有名说明符能通过格式化。"""

    def __missing__(self, key):        # noqa: D105
        return 1


def bad_percent(text: str) -> list:
    """``text`` 能否被 argparse 的 ``help % params`` 正常格式化。

    返回 ``[]`` 表示 OK；返回 ``[0]`` 表示**整条字符串**会抛 ValueError
    （argparse 里表现为 ``--help`` 与 usage 报错直接崩）。用"真的去格式化一次"
    判断，比正则猜格式说明符可靠。
    """
    if "%" not in text:
        return []
    try:
        text % _AnyDict()
    except (ValueError, TypeError):
        return [0]
    return []


def scan(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    issues = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "help" or not isinstance(kw.value, ast.Constant):
                continue
            text = kw.value.value
            if not isinstance(text, str):
                continue
            for idx in bad_percent(text):
                issues += 1
                snippet = text[max(0, idx - 12):idx + 12].encode(
                    "utf-8", "replace").decode("utf-8", "replace")
                print(f"{path}:{kw.value.lineno}: pos {idx}: ...{snippet}...")
    return issues


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    total = 0
    for name in argv:
        total += scan(Path(name))
    print(f"\n共 {total} 处未转义百分号" + ("（--help 会崩）" if total else "（OK）"))
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
