"""Change-hygiene gate for the now single-agent workspace.

历史背景（2026-11-10 → 2026-11-12）：本项目曾经有两个 Agent，各自"拥有"不同的
目录，本脚本的原始用途是**阻止本地改服务器拥有的文件**。那段架构已经退休
（`docs/ARCHITECTURE.md` §5，`docs/DECISIONS.md` D-028）。

现在只有一个 Agent 管两端，"所有权"不再存在。但同一份代码在两台机器上各有一份，
所以仍然需要两个提示：

1. **这个文件平时由哪一端维护**（信息性）。在本地改 `hsbg/` 通常意味着要同步到服务器，
   在服务器改 `client/` 通常意味着要拉回本地。提示一下，避免忘记同步。
2. **这次改动是否会把不该进 Git 的东西带进去**（门禁）。大文件、证据目录、密钥。

`--strict` 只对第 2 类返回非零退出码。第 1 类永远不阻断——单 Agent 有权改任何文件。

用法
----
    python client/tools/check_ownership.py            # 报告
    python client/tools/check_ownership.py --strict   # 有卫生问题时退出码 1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: 单次提交允许的最大单文件体积（与 AGENTS.md §2 一致）
MAX_FILE_BYTES = 5 * 1024 * 1024

#: 平时在本地 Windows 维护
LOCAL_PREFIXES = (
    "client/",
    "config/live/",
    "artifacts/",
    "records/",
    "ops/",
    "docs/live/",
    "docs/LIVE_V1_1_AUDIT.md",
    "docs/E_PHASE_RUNBOOK.md",
    "docs/analysis/",
)

#: 平时在 NUS 维护（在 GPU 上验证，或跑训练才产生）
SERVER_PREFIXES = (
    "engine/",
    "hsbg/",
    "workflows/",
    "deploy/",
    "data/",
    "tools/",
)

#: 两端都改
SHARED_PREFIXES = (
    "tests/",
    "config/",
    "docs/",
    "AGENTS.md",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
)

#: 绝不能出现在提交里的路径特征
FORBIDDEN = (
    ("artifacts/", "运行产物"),
    ("records/", "证据目录"),
    (".ssh/", "私钥目录"),
    (".env", "环境变量（可能含密钥）"),
)
FORBIDDEN_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".onnx",
                      ".tar", ".tar.gz", ".tgz", ".zip", ".7z", ".pdf", ".log")


def git_status() -> list[tuple[str, str]]:
    out = subprocess.run(["git", "status", "--porcelain=v1"], cwd=ROOT,
                         capture_output=True, text=True, timeout=60)
    rows = []
    for line in out.stdout.splitlines():
        if len(line) < 4:
            continue
        code, path = line[:2], line[3:].strip()
        if " -> " in path:                       # rename: judge the destination
            path = path.split(" -> ", 1)[1]
        rows.append((code, path))
    return rows


def classify(path: str) -> str:
    if any(path.startswith(p) for p in LOCAL_PREFIXES):
        return "local"
    if any(path.startswith(p) for p in SERVER_PREFIXES):
        return "server"
    if any(path.startswith(p) for p in SHARED_PREFIXES):
        return "shared"
    return "unknown"


def hygiene(path: str) -> str | None:
    """返回不该提交的理由；None 表示没问题。"""
    for prefix, why in FORBIDDEN:
        if path.startswith(prefix):
            return why
    for suffix in FORBIDDEN_SUFFIXES:
        if path.endswith(suffix):
            return f"大文件/二进制（{suffix}）应走 ssh 传输，不进 Git"
    full = ROOT / path
    if full.is_file() and full.stat().st_size > MAX_FILE_BYTES:
        return f"{full.stat().st_size / 1e6:.1f} MB > 5 MB 阈值"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="HSAI 改动卫生检查")
    ap.add_argument("--strict", action="store_true", help="有卫生问题时退出码 1")
    args = ap.parse_args(argv)

    rows = git_status()
    if not rows:
        print("工作区干净 —— 卫生检查通过")
        return 0

    buckets: dict[str, list[tuple[str, str]]] = {
        "local": [], "server": [], "shared": [], "unknown": []}
    for code, path in rows:
        buckets[classify(path)].append((code, path))

    for key, title in (
        ("local", "平时在本地维护（记得 push 到服务器）"),
        ("server", "平时在服务器维护（记得 pull 回本地）"),
        ("shared", "两端都维护（同步前先 pull）"),
        ("unknown", "未分类（人工看一眼）"),
    ):
        items = buckets[key]
        if not items:
            continue
        print(f"\n{title}: {len(items)}")
        for code, path in items:
            print(f"  {code:>2}  {path}")

    dirty: list[tuple[str, str]] = []
    for _code, path in rows:
        reason = hygiene(path)
        if reason:
            dirty.append((path, reason))

    if dirty:
        print(f"\n卫生问题: {len(dirty)}")
        for path, reason in dirty:
            print(f"  ✗ {path}: {reason}")
    else:
        print("\n卫生检查: 没有大文件 / 证据目录 / 密钥进入改动集")

    if args.strict and dirty:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
