#!/usr/bin/env python3
"""文档一致性检查：内部链接是否还有效、是否还在引用已退休的路径。

为什么需要它：单 Agent 架构下文档会被反复重组（归档、上移、改名），而 Markdown
的相对链接坏掉是**静默**的——没人点进去就永远发现不了。这个脚本把它变成一次退出码。

用法::

    python ops/check_docs.py            # 检查，有问题退出码 1
    python ops/check_docs.py --verbose  # 列出每个被检查的文件

检查范围：仓库根 `README.md`、`AGENTS.md`、`ops/`、`docs/` 下的所有 `.md`。
`records/` 下的归档文档**不检查**（历史快照按原样保留）。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterator, List, NamedTuple, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

#: 检查入口文件与目录（相对仓库根）
SCAN_FILES = ("README.md", "AGENTS.md")
SCAN_DIRS = ("docs", "ops")

#: 已退休的路径。出现在**现行文档**里就是错误（`records/` 与 DECISIONS 历史除外）。
RETIRED = (
    "docs/sync/",
    "docs/SYNC_MODEL.md",
    "docs/INVENTORY_LOCAL.md",
    "tools/sync_manifest.py",
)

#: 允许提到退休路径的上下文。出现在**现行文档**里就是错误，除非同一行同时说明了
#: 归档/退休——那种提到是必要的（讲清楚东西去哪了），不是坏指针。
RETIREMENT_CUES = ("records/archive/", "退休", "归档", "废止", "取代", "已停用", "上移", "已删除")

LINK_RE = re.compile(r"\[(?P<text>[^\]]*)\]\((?P<target>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE_RE = re.compile(r"^(```|~~~)")


class Problem(NamedTuple):
    path: str
    line: int
    kind: str
    detail: str


def iter_markdown() -> Iterator[Path]:
    seen = set()
    for name in SCAN_FILES:
        p = REPO_ROOT / name
        if p.is_file():
            seen.add(p)
            yield p
    for d in SCAN_DIRS:
        root = REPO_ROOT / d
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.md")):
            if p in seen:
                continue
            seen.add(p)
            yield p


def _strip_code_fences(lines: List[str]) -> List[Tuple[int, str]]:
    """把围栏代码块内的行去掉——里面的路径示例不是链接。"""
    out: List[Tuple[int, str]] = []
    fence = False
    for i, line in enumerate(lines, 1):
        if FENCE_RE.match(line.strip()):
            fence = not fence
            continue
        if not fence:
            out.append((i, line))
    return out


def _rel(path: Path) -> str:
    """仓库内文件给相对路径，仓库外（测试用的临时目录）给绝对路径。"""
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def check_links(path: Path) -> List[Problem]:
    problems: List[Problem] = []
    text = path.read_text(encoding="utf-8")
    for lineno, line in _strip_code_fences(text.splitlines()):
        for m in LINK_RE.finditer(line):
            target = m.group("target")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            rel = target.split("#", 1)[0]
            if not rel:
                continue
            resolved = (path.parent / rel).resolve()
            if not resolved.exists():
                problems.append(
                    Problem(_rel(path), lineno, "broken-link", f"{target}  ->  {rel}")
                )
    return problems


def check_retired(path: Path) -> List[Problem]:
    rel_name = _rel(path)
    if rel_name == "docs/DECISIONS.md":
        return []  # 只追加不改历史：历史 evidence 里的旧路径按规则保留
    # docs/analysis/ 里带日期戳的报告是历史快照，引用的路径在当时是对的
    if re.match(r"docs/analysis/.*\d{8}.*\.md$", rel_name):
        return []
    problems: List[Problem] = []
    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), 1):
        if any(cue in line for cue in RETIREMENT_CUES):
            continue
        for token in RETIRED:
            if token in line:
                problems.append(Problem(rel_name, lineno, "retired-path", line.strip()[:110]))
                break
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="HSAI 文档一致性检查")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    files = list(iter_markdown())
    problems: List[Problem] = []
    for path in files:
        if args.verbose:
            print(f"checked {path.relative_to(REPO_ROOT)}")
        problems += check_links(path)
        problems += check_retired(path)

    print(f"scanned {len(files)} markdown files")
    if not problems:
        print("OK: no broken links, no retired-path references")
        return 0
    print(f"{len(problems)} problem(s):")
    for p in problems:
        print(f"  [{p.kind}] {p.path}:{p.line}: {p.detail}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
