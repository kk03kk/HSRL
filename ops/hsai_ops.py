#!/usr/bin/env python3
"""HSAI 单 Agent 双端运维 CLI。

一个进程管两端：本地 Windows 仓库 + NUS 计算集群。所有远端动作都通过
``ssh hsai-nus``（SSH 配置见 ``~/.ssh/config``，隧道原理见
``docs/architecture/TRANSPORT.md``）以 **stdin 管道** 方式执行，因此不再受
PowerShell/quoted-heredoc 转义问题的折磨。

用法::

    python ops/hsai_ops.py probe            # 隧道是否通
    python ops/hsai_ops.py state            # 双端状态快照（JSON）
    python ops/hsai_ops.py doctor           # 全面自检
    python ops/hsai_ops.py sh "<cmd>"       # 远端执行（带登录节点守卫）
    python ops/hsai_ops.py sh -f script.sh  # 远端执行脚本文件
    python ops/hsai_ops.py git-status       # 双端 git 状态
    python ops/hsai_ops.py push             # 本地 main -> 服务器
    python ops/hsai_ops.py pull             # 服务器 -> 本地
    python ops/hsai_ops.py jobs             # Slurm 作业概览
    python ops/hsai_ops.py models           # 服务器模型清单
    python ops/hsai_ops.py model-pull NAME  # 下载模型到本地（校验 SHA256）

设计约束（不要违反）：

1. **登录节点只有 4 核 / 16 GB，且多人共享。** ``HEAVY_PATTERNS`` 命中的命令默认
   拒绝执行；确实需要时加 ``--allow-heavy``，并在注释里写清理由。真正的重活走
   ``sbatch``。
2. **输出进入 Agent 上下文。** 默认截断到 ``DEFAULT_MAX_BYTES``；要全文用
   ``--full`` 或先重定向到文件再 ``tail``。
3. **所有远端命令留痕。** 写入 ``artifacts/state/ssh-audit.jsonl``。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = REPO_ROOT / "artifacts" / "state"
AUDIT_LOG = STATE_DIR / "ssh-audit.jsonl"

SSH_HOST = os.environ.get("HSAI_SSH_HOST", "hsai-nus")
REMOTE_ROOT = os.environ.get("HSAI_REMOTE_ROOT", "~/projects/HSAI")
DEFAULT_MAX_BYTES = 12000

#: 这些模式只能在计算节点跑。登录节点 4 核共享，跑它们会拖垮自己也会被投诉。
HEAVY_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\bpytest\b", "pytest"),
    (r"\bpython[0-9.]*\s+-m\s+pytest\b", "pytest"),
    (r"\bcargo\b", "cargo"),
    (r"\bpip\s+install\b", "pip install"),
    (r"\bmaturin\b", "maturin"),
    (r"\bimport\s+torch\b", "torch import"),
    (r"\btrain_entity_sync\b", "training"),
    (r"\bworkflows/[a-z_]*train", "training workflow"),
    (r"\bpython[0-9.]*\s+workflows/", "workflow script"),
    (r"\bsrun\b", "srun (use sbatch)"),
    (r"\bray\b", "ray"),
)

ZERO_KEYS = ("illegal", "truncated", "dropped", "errors")


class OpsError(RuntimeError):
    """可预期的运维失败（隧道断、远端命令非零等）。"""


# ────────────────────────── 基础工具 ──────────────────────────


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def audit(entry: Dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    entry = dict(entry)
    entry.setdefault("at", now_iso())
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def truncate(text: str, limit: int = DEFAULT_MAX_BYTES) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    dropped = len(text) - len(head) - len(tail)
    return f"{head}\n... [truncated {dropped} chars] ...\n{tail}"


def ssh_argv() -> List[str]:
    return ["ssh", SSH_HOST]


#: 以这些词开头的行不可能是重活（读文件、查 Slurm、git 操作等）。
#: 有了这个白名单，``git add workflows/train_x.py`` 这类**只是提到**训练脚本的
#: 命令不会误报，而 ``.venv/bin/python workflows/train_x.py`` 仍会被拦下。
SAFE_LEADING_TOKENS = frozenset(
    """
    git echo printf cd pwd ls cat head tail grep egrep fgrep sha256sum md5sum du df
    squeue sacct sinfo stat wc cut tr sort uniq find test set export source .
    realpath date hostname uname mkdir rm mv cp chmod chown readlink basename dirname
    which type true false id whoami env tee sed awk xargs ln touch diff cmp
    scancel sbatch scontrol
    """.split()
)


HEREDOC_RE = re.compile(r"<<-?\s*'?([A-Za-z_][A-Za-z0-9_]*)'?")


def _strip_heredocs(script: str) -> List[str]:
    """丢弃 heredoc 正文。提交信息、内联 python 里的文字不是命令，不该触发守卫。"""
    out: List[str] = []
    tag: Optional[str] = None
    for raw in script.splitlines():
        if tag is not None:
            if raw.strip() == tag:
                tag = None
            continue
        out.append(raw)
        m = HEREDOC_RE.search(raw)
        if m and "<<<" not in raw:
            tag = m.group(1)
    return out


def _logical_lines(script: str) -> List[str]:
    """把反斜杠续行合并成逻辑行；否则 ``git add \\`` 后面的文件名会被当成命令。"""
    out: List[str] = []
    buf = ""
    for raw in _strip_heredocs(script):
        line = raw.rstrip()
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        out.append(buf + line)
        buf = ""
    if buf:
        out.append(buf)
    return out


def _executed_lines(script: str) -> List[str]:
    """粗过滤：去掉注释/空行，跳过明显不可能跑重活的命令行。"""
    kept: List[str] = []
    for raw in _logical_lines(script):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        first = line.split(None, 1)[0]
        # 允许前缀形如 `FOO=bar cmd` 的环境变量赋值
        if "=" in first and not first.startswith("-"):
            parts = line.split(None, 1)
            first = parts[1].split(None, 1)[0] if len(parts) > 1 else ""
        if first in SAFE_LEADING_TOKENS:
            continue
        kept.append(line)
    return kept


def check_heavy(script: str) -> List[str]:
    lines = _executed_lines(script)
    if not lines:
        return []
    hits: List[str] = []
    for pattern, label in HEAVY_PATTERNS:
        if any(re.search(pattern, line) for line in lines):
            hits.append(label)
    return sorted(set(hits))


def remote_run(
    script: str,
    *,
    timeout: float = 120.0,
    allow_heavy: bool = False,
    check: bool = False,
    label: str = "sh",
    cwd: Optional[str] = None,
) -> Tuple[int, str, str]:
    """通过 stdin 把 ``script`` 管道给远端 bash 执行。

    默认先在远端仓库根目录 ``cd``。相对路径（``.venv/bin/python``、
    ``artifacts/...``）因此可以直接写，不必每条命令自己再 cd 一次。
    """
    heavy = check_heavy(script)
    if heavy and not allow_heavy:
        raise OpsError(
            "拒绝在登录节点执行重活："
            + ", ".join(heavy)
            + "。改用 sbatch，或加 --allow-heavy 并写清理由。"
        )
    root = REMOTE_ROOT if cwd is None else cwd
    full = f"cd {root} 2>/dev/null || exit 3\n{script}" if root else script
    cmd = ssh_argv() + ["bash -s"]
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            input=full.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        audit({"label": label, "rc": "timeout", "secs": round(time.time() - started, 1)})
        raise OpsError(f"远端命令超时（{timeout}s）：{script.strip()[:120]}") from exc
    except FileNotFoundError as exc:  # 没有 ssh 客户端
        raise OpsError("找不到 ssh 命令") from exc
    out = proc.stdout.decode("utf-8", "replace")
    err = proc.stderr.decode("utf-8", "replace")
    audit(
        {
            "label": label,
            "rc": proc.returncode,
            "secs": round(time.time() - started, 1),
            "bytes": len(out),
            "heavy": heavy or None,
            "cmd": script.strip().splitlines()[0][:200] if script.strip() else "",
        }
    )
    if check and proc.returncode != 0:
        raise OpsError(f"远端命令失败 rc={proc.returncode}\n{truncate(err, 3000)}")
    return proc.returncode, out, err


def probe_tunnel(timeout: float = 12.0) -> Dict[str, Any]:
    """隧道是否通。断了就抛出带修复命令的异常。"""
    try:
        rc, out, err = remote_run(
            "echo TUNNEL_OK; hostname; date -Is", timeout=timeout, label="probe"
        )
    except OpsError as exc:
        return {"ok": False, "error": str(exc), "fix": TUNNEL_FIX}
    ok = rc == 0 and "TUNNEL_OK" in out
    if not ok:
        return {"ok": False, "error": (err or out).strip()[:400], "fix": TUNNEL_FIX}
    lines = out.strip().splitlines()
    return {
        "ok": True,
        "host": lines[1].strip() if len(lines) > 1 else "",
        "remote_time": lines[2].strip() if len(lines) > 2 else "",
    }


TUNNEL_FIX = (
    "隧道已断。请在 **本地 PowerShell** 里重跑这一条（只需输一次跳板机密码）：\n"
    "  ssh -N -f -L 2222:xlogin2.comp.nus.edu.sg:22 "
    "-o ServerAliveInterval=30 -o ServerAliveCountMax=10 "
    "-o ExitOnForwardFailure=yes hengzhi@sjump.comp.nus.edu.sg\n"
    "或者用本仓库的快捷方式：python ops/hsai_ops.py tunnel"
)


def require_tunnel() -> Dict[str, Any]:
    info = probe_tunnel()
    if not info["ok"]:
        raise OpsError(info.get("error", "tunnel down") + "\n\n" + TUNNEL_FIX)
    return info


# ────────────────────────── 本地 git ──────────────────────────


def _git(args: Sequence[str], *, cwd: Optional[Path] = None) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd or REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise OpsError(proc.stderr.decode("utf-8", "replace").strip() or "git failed")
    return proc.stdout.decode("utf-8", "replace")


def local_git_state() -> Dict[str, Any]:
    dirty = [ln for ln in _git(["status", "--porcelain=v1"]).splitlines() if ln.strip()]
    return {
        "head": _git(["log", "-1", "--format=%H %s"]).strip(),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]).strip(),
        "dirty_count": len(dirty),
        "dirty": dirty[:40],
        "remotes": _git(["remote", "-v"]).strip().splitlines(),
    }


def remote_git_state(script_prefix: str = "") -> Dict[str, Any]:
    script = f"""
set -u
cd {REMOTE_ROOT} || exit 3
echo "@@HEAD"
git log -1 --format='%H %s'
echo "@@BRANCH"
git rev-parse --abbrev-ref HEAD
echo "@@DIRTY"
git status --porcelain=v1
echo "@@END"
"""
    rc, out, err = remote_run(script, timeout=90, label="git-status")
    if rc != 0:
        raise OpsError(f"服务器 git 状态读取失败 rc={rc}: {err.strip()[:300]}")
    sections = _split_sections(out)
    dirty = [ln for ln in sections.get("DIRTY", "").splitlines() if ln.strip()]
    return {
        "head": sections.get("HEAD", "").strip(),
        "branch": sections.get("BRANCH", "").strip(),
        "dirty_count": len(dirty),
        "dirty": dirty[:40],
    }


def _split_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in text.splitlines():
        if line.startswith("@@") and line.strip() not in ("@@END",):
            current = line[2:].strip()
            sections[current] = []
        elif line.strip() == "@@END":
            current = None
        elif current is not None:
            sections[current].append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


# ────────────────────────── 双端状态 ──────────────────────────


SERVER_STATE_SCRIPT = r"""
set -u
cd ~/projects/HSAI || exit 3
echo "@@HEAD"
git log -1 --format='%H %s'
echo "@@BRANCH"
git rev-parse --abbrev-ref HEAD
echo "@@DIRTY"
git status --porcelain=v1
echo "@@JOBS"
squeue -u "$USER" -h -o '%i|%P|%j|%T|%M|%D|%R' 2>/dev/null
echo "@@GPU"
sinfo -h -p gpu,gpu-long -o '%P|%a|%t|%D' 2>/dev/null | sort
echo "@@DISK"
df -h "$HOME" 2>/dev/null | tail -1
echo "@@SIZE"
du -sh . 2>/dev/null | cut -f1
echo "@@MODELS"
ls -1 artifacts/models 2>/dev/null
echo "@@LIBS"
ls -1 artifacts/libraries 2>/dev/null
echo "@@RUNS"
ls -1t artifacts/runs 2>/dev/null | head -6
echo "@@END"
"""


def server_state() -> Dict[str, Any]:
    rc, out, err = remote_run(SERVER_STATE_SCRIPT, timeout=150, label="state")
    if rc != 0:
        raise OpsError(f"服务器状态读取失败 rc={rc}: {err.strip()[:300]}")
    s = _split_sections(out)
    dirty = [ln for ln in s.get("DIRTY", "").splitlines() if ln.strip()]
    jobs = []
    for line in s.get("JOBS", "").splitlines():
        parts = line.split("|")
        if len(parts) >= 7:
            jobs.append(
                {
                    "id": parts[0],
                    "partition": parts[1],
                    "name": parts[2],
                    "state": parts[3],
                    "elapsed": parts[4],
                    "nodes": parts[5],
                    "reason": parts[6].strip(),
                }
            )
    gpu = []
    for line in s.get("GPU", "").splitlines():
        parts = line.split("|")
        if len(parts) >= 4:
            gpu.append(
                {
                    "partition": parts[0].rstrip("*"),
                    "avail": parts[1],
                    "state": parts[2],
                    "nodes": parts[3],
                }
            )
    return {
        "head": s.get("HEAD", "").strip(),
        "branch": s.get("BRANCH", "").strip(),
        "dirty_count": len(dirty),
        "dirty": dirty[:40],
        "jobs": jobs,
        "gpu": gpu,
        "disk": s.get("DISK", "").strip(),
        "repo_size": s.get("SIZE", "").strip(),
        "models": [x for x in s.get("MODELS", "").splitlines() if x.strip()],
        "libraries": [x for x in s.get("LIBS", "").splitlines() if x.strip()],
        "recent_runs": [x for x in s.get("RUNS", "").splitlines() if x.strip()],
    }


def build_state() -> Dict[str, Any]:
    local = local_git_state()
    state: Dict[str, Any] = {"generated_at": now_iso(), "local": local}
    tunnel = probe_tunnel()
    state["tunnel"] = tunnel
    if tunnel["ok"]:
        try:
            state["server"] = server_state()
        except OpsError as exc:
            state["server"] = {"error": str(exc)}
    drift = None
    if tunnel["ok"] and "error" not in state.get("server", {}):
        drift = {
            "head_equal": local["head"].split()[0] == state["server"]["head"].split()[0],
            "local_dirty": local["dirty_count"],
            "server_dirty": state["server"]["dirty_count"],
        }
    state["drift"] = drift
    return state


def print_state(state: Dict[str, Any]) -> None:
    tunnel = state["tunnel"]
    print(f"HSAI dual-end state  {state['generated_at']}")
    print(f"  tunnel : {'OK  ' + tunnel.get('host', '') if tunnel['ok'] else 'DOWN'}")
    loc = state["local"]
    print(f"  local  : {loc['head'][:8]} {loc['branch']} dirty={loc['dirty_count']}")
    srv = state.get("server") or {}
    if "error" in srv:
        print(f"  server : ERROR {srv['error'][:160]}")
    elif srv:
        print(f"  server : {srv['head'][:8]} {srv['branch']} dirty={srv['dirty_count']}"
              f" size={srv.get('repo_size','')} disk={srv.get('disk','')}")
        if srv.get("jobs"):
            for j in srv["jobs"]:
                print(f"           job {j['id']} {j['name']} {j['state']} {j['elapsed']} {j['reason']}")
        else:
            print("           jobs : none")
        print(f"           models={len(srv.get('models', []))} libs={len(srv.get('libraries', []))}")
    if state.get("drift"):
        d = state["drift"]
        flag = "IN SYNC" if d["head_equal"] and not d["local_dirty"] and not d["server_dirty"] else "DRIFT"
        print(f"  drift  : {flag} (heads_equal={d['head_equal']} "
              f"local_dirty={d['local_dirty']} server_dirty={d['server_dirty']})")


# ────────────────────────── 传输 ──────────────────────────


def remote_sha256(paths: Iterable[str]) -> Dict[str, str]:
    quoted = " ".join(shlex.quote(p) for p in paths)
    script = f"cd {REMOTE_ROOT} && sha256sum {quoted} 2>&1"
    rc, out, err = remote_run(script, timeout=120, label="sha256")
    if rc != 0:
        raise OpsError(f"服务器 sha256 失败: {err.strip()[:300]}")
    result: Dict[str, str] = {}
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            result[parts[1].strip()] = parts[0]
    return result


def local_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stream_file_down(remote_rel: str, local_path: Path, *, label: str = "pull") -> int:
    """``ssh cat`` 流式下载单个文件，避免 scp 在 Windows 上的路径怪癖。

    注意：脚本必须走 **stdin**（``bash -s``）。ssh 会把多余参数用空格拼起来，
    ``ssh host bash -c "cd X && cat f"`` 实际只把 ``cd`` 交给 bash，其余变成独立
    词，于是 cat 在 $HOME 里找文件——这个坑已经踩过一次。
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = local_path.with_suffix(local_path.suffix + ".partial")
    script = f"cd {REMOTE_ROOT} && cat -- {shlex.quote(remote_rel)}"
    started = time.time()
    with tmp.open("wb") as fh:
        proc = subprocess.run(
            ssh_argv() + ["bash -s"],
            input=script.encode("utf-8"),
            stdout=fh,
            stderr=subprocess.PIPE,
        )
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise OpsError(
            f"下载 {remote_rel} 失败 rc={proc.returncode}: "
            f"{proc.stderr.decode('utf-8', 'replace')[:300]}"
        )
    tmp.replace(local_path)
    secs = time.time() - started
    size = local_path.stat().st_size
    audit({"label": label, "file": remote_rel, "bytes": size, "secs": round(secs, 2)})
    return size


def cmd_model_pull(args: argparse.Namespace) -> int:
    require_tunnel()
    name = args.name
    rel_dir = f"artifacts/models/{name}"
    rc, out, _ = remote_run(
        f"cd {REMOTE_ROOT} && ls -1 {shlex.quote(rel_dir)} 2>/dev/null", timeout=60, label="model-ls"
    )
    if rc != 0 or not out.strip():
        raise OpsError(f"服务器上没有模型目录 {rel_dir}")
    available = [x.strip() for x in out.splitlines() if x.strip()]
    wanted = [f for f in available if f in args.files] or available
    # 大文件按需拉；trainer.pt 只有显式要求才下载
    if "trainer.pt" in wanted and not args.with_trainer:
        wanted.remove("trainer.pt")
    remote_paths = [f"{rel_dir}/{f}" for f in wanted]
    sums = remote_sha256(remote_paths)
    dest_root = REPO_ROOT / "artifacts" / "models" / name
    total = 0
    started = time.time()
    for f in wanted:
        rel = f"{rel_dir}/{f}"
        local = dest_root / f
        if local.exists() and local_sha256(local) == sums.get(rel):
            print(f"  = {f} (already matches)")
            continue
        size = stream_file_down(rel, local)
        got = local_sha256(local)
        if got != sums.get(rel):
            raise OpsError(f"SHA256 不匹配 {f}: local={got[:16]} remote={str(sums.get(rel))[:16]}")
        total += size
        print(f"  + {f} {size/1e6:.1f} MB  sha256 ok")
    secs = max(time.time() - started, 1e-6)
    print(f"model-pull {name}: {total/1e6:.1f} MB in {secs:.1f}s "
          f"({total/1e6/secs:.1f} MB/s) -> {dest_root}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    require_tunnel()
    script = r"""
cd ~/projects/HSAI/artifacts/models 2>/dev/null || exit 0
for d in */; do
  d=${d%/}
  sz=$(du -sm "$d" 2>/dev/null | cut -f1)
  n=$(ls -1 "$d" 2>/dev/null | wc -l)
  printf '%-42s %5s MB  %s files\n' "$d" "$sz" "$n"
done
"""
    rc, out, err = remote_run(script, timeout=120, label="models")
    print(truncate(out or err, DEFAULT_MAX_BYTES if not args.full else 0))
    return rc


def cmd_jobs(args: argparse.Namespace) -> int:
    require_tunnel()
    script = r"""
echo "=== running/pending ==="
squeue -u "$USER" 2>/dev/null || echo "none"
echo "=== last 8 accounting ==="
sacct -u "$USER" -X -S now-7days -o JobID,JobName%28,State,Elapsed,End 2>/dev/null | tail -9
"""
    rc, out, err = remote_run(script, timeout=120, label="jobs")
    print(truncate(out or err, DEFAULT_MAX_BYTES if not args.full else 0))
    return rc


# ────────────────────────── git 同步 ──────────────────────────


SERVER_REMOTE_URL = f"ssh://{SSH_HOST}/home/h/hengzhi/projects/HSAI"


def ensure_server_remote() -> None:
    remotes = _git(["remote"]).split()
    if "server" in remotes:
        current = _git(["remote", "get-url", "server"]).strip()
        if current != SERVER_REMOTE_URL:
            _git(["remote", "set-url", "server", SERVER_REMOTE_URL])
        return
    _git(["remote", "add", "server", SERVER_REMOTE_URL])


def cmd_push(args: argparse.Namespace) -> int:
    """本地 main -> 服务器 main。

    服务器仓库配置了 ``receive.denyCurrentBranch=updateInstead``，所以推送成功
    会直接更新服务器工作区；服务器有未提交改动时 git 会**拒绝**——这正是我们要的
    漂移保护。
    """
    require_tunnel()
    ensure_server_remote()
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
    if branch != "main":
        raise OpsError(f"当前分支是 {branch}，只允许从 main 推送")
    dirty = [ln for ln in _git(["status", "--porcelain=v1"]).splitlines() if ln.strip()]
    if dirty and not args.allow_dirty:
        raise OpsError(f"本地有 {len(dirty)} 个未提交改动，先提交再推送")
    proc = subprocess.run(
        ["git", "push", "server", f"main:main"],
        cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    print(proc.stdout.decode("utf-8", "replace"))
    audit({"label": "push", "rc": proc.returncode})
    return proc.returncode


def cmd_pull(args: argparse.Namespace) -> int:
    require_tunnel()
    ensure_server_remote()
    fetch = subprocess.run(
        ["git", "fetch", "server", "main"], cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    print(fetch.stdout.decode("utf-8", "replace"))
    if fetch.returncode != 0:
        return fetch.returncode
    ahead = _git(["rev-list", "--count", "HEAD..server/main"]).strip()
    if ahead == "0":
        print("服务器没有新提交")
        return 0
    if args.check_only:
        print(f"服务器领先 {ahead} 个提交（--check 模式，未合并）")
        return 0
    merge = subprocess.run(
        ["git", "merge", "--no-edit", "server/main"], cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    print(merge.stdout.decode("utf-8", "replace"))
    audit({"label": "pull", "rc": merge.returncode})
    return merge.returncode


def cmd_git_status(args: argparse.Namespace) -> int:
    ensure_server_remote()
    print("== local ==")
    print(local_git_state()["head"])
    print("  dirty:", local_git_state()["dirty_count"])
    require_tunnel()
    srv = remote_git_state()
    print("== server ==")
    print(srv["head"])
    print("  dirty:", srv["dirty_count"])
    for ln in srv["dirty"]:
        print("   ", ln)
    return 0


# ────────────────────────── 命令 ──────────────────────────


def cmd_probe(args: argparse.Namespace) -> int:
    info = probe_tunnel()
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0 if info["ok"] else 2


def cmd_tunnel(args: argparse.Namespace) -> int:
    """打印（或执行）~/.ssh/config 里的 hsai-bastion 主机建隧道。"""
    cmd = (
        f"ssh -N -f -L 2222:xlogin2.comp.nus.edu.sg:22 "
        f"-o ServerAliveInterval=30 -o ServerAliveCountMax=10 "
        f"-o ExitOnForwardFailure=yes hengzhi@sjump.comp.nus.edu.sg"
    )
    info = probe_tunnel()
    if info["ok"]:
        print(f"隧道已在运行（{info.get('host')}），无需重建")
        return 0
    print("请在本地 PowerShell 执行（会提示输入跳板机密码）：")
    print("  " + cmd)
    return 2


def cmd_state(args: argparse.Namespace) -> int:
    state = build_state()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (STATE_DIR / f"state-{stamp}.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (STATE_DIR / "latest.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=2))
    else:
        print_state(state)
    return 0 if state["tunnel"]["ok"] else 2


def cmd_sh(args: argparse.Namespace) -> int:
    if args.file:
        script = Path(args.file).read_text(encoding="utf-8")
    else:
        script = " ".join(args.script)
    if not script.strip():
        raise OpsError("没有给出命令")
    rc, out, err = remote_run(
        script, timeout=args.timeout, allow_heavy=args.allow_heavy, label="sh"
    )
    payload = out + (("\n[stderr]\n" + err) if err.strip() else "")
    print(truncate(payload, 0 if args.full else args.max_bytes))
    return rc


def cmd_doctor(args: argparse.Namespace) -> int:
    problems: List[str] = []
    print("HSAI ops doctor")
    print("-" * 60)

    # 1. ssh config
    cfg = Path.home() / ".ssh" / "config"
    print(f"[1/7] ~/.ssh/config           : {'ok' if cfg.exists() else 'MISSING'}")
    if not cfg.exists():
        problems.append("~/.ssh/config 缺失：先建 hsai-nus 别名")

    # 2. 隧道
    info = probe_tunnel()
    print(f"[2/7] tunnel -> {SSH_HOST:<12}: {'ok ' + info.get('host', '') if info['ok'] else 'DOWN'}")
    if not info["ok"]:
        problems.append("SSH 隧道不通")

    # 3. 本地 git
    loc = local_git_state()
    print(f"[3/7] local git               : {loc['head'][:8]} dirty={loc['dirty_count']}")
    if loc["dirty_count"]:
        problems.append(f"本地有 {loc['dirty_count']} 个未提交改动")

    # 4. server git
    if info["ok"]:
        srv = remote_git_state()
        print(f"[4/7] server git              : {srv['head'][:8]} dirty={srv['dirty_count']}")
        if srv["dirty_count"]:
            problems.append(f"服务器有 {srv['dirty_count']} 个未提交改动")
        if srv["head"].split()[0] != loc["head"].split()[0]:
            problems.append("两端 HEAD 不一致")
        # 5. updateInstead
        rc, out, _ = remote_run(
            f"cd {REMOTE_ROOT} && git config --get receive.denyCurrentBranch", timeout=60,
            label="doctor-updateinstead",
        )
        mode = out.strip()
        print(f"[5/7] server updateInstead     : {mode or 'unset'}")
        if mode != "updateInstead":
            problems.append("服务器 receive.denyCurrentBranch 未设为 updateInstead")
        # 6. 作业
        rc, out, _ = remote_run("squeue -u $USER -h -o '%i %j %T' | head -5", timeout=60, label="doctor-jobs")
        jobs = [l for l in out.splitlines() if l.strip()]
        print(f"[6/7] server jobs             : {len(jobs)} active")
        for j in jobs:
            print(f"        {j}")
    else:
        print("[4/7] server git              : skipped (no tunnel)")
        print("[5/7] server updateInstead    : skipped (no tunnel)")
        print("[6/7] server jobs             : skipped (no tunnel)")

    # 7. 远端 remote
    try:
        ensure_server_remote()
        print(f"[7/7] git remote 'server'     : {_git(['remote', 'get-url', 'server']).strip()}")
    except OpsError as exc:
        print(f"[7/7] git remote 'server'     : ERROR {exc}")
        problems.append("git remote 'server' 未配置")

    print("-" * 60)
    if problems:
        print(f"发现 {len(problems)} 个问题：")
        for p in problems:
            print(f"  - {p}")
        return 2
    print("全部通过")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    # 输出是给人和 Agent 看的，不能让 Windows cp936 把中文弄成乱码
    for stream in (sys.stdout, sys.stderr):
        try:
            if (stream.encoding or "").lower().replace("-", "") != "utf8":
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="HSAI 单 Agent 双端运维 CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="隧道连通性")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("tunnel", help="打印建隧道命令")
    p.set_defaults(func=cmd_tunnel)

    p = sub.add_parser("state", help="双端状态快照")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_state)

    p = sub.add_parser("doctor", help="全面自检")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("sh", help="远端执行（登录节点守卫）")
    p.add_argument("script", nargs="*")
    p.add_argument("-f", "--file", help="从文件读取脚本")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    p.add_argument("--full", action="store_true", help="不截断输出")
    p.add_argument("--allow-heavy", action="store_true", help="允许登录节点跑重活")
    p.set_defaults(func=cmd_sh)

    p = sub.add_parser("git-status", help="双端 git 状态")
    p.set_defaults(func=cmd_git_status)

    p = sub.add_parser("push", help="本地 main -> 服务器")
    p.add_argument("--allow-dirty", action="store_true")
    p.set_defaults(func=cmd_push)

    p = sub.add_parser("pull", help="服务器 -> 本地")
    p.add_argument("--check", dest="check_only", action="store_true", help="只看不合并")
    p.set_defaults(func=cmd_pull)

    p = sub.add_parser("jobs", help="Slurm 作业")
    p.add_argument("--full", action="store_true")
    p.set_defaults(func=cmd_jobs)

    p = sub.add_parser("models", help="服务器模型清单")
    p.add_argument("--full", action="store_true")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("model-pull", help="下载模型到本地并校验 SHA256")
    p.add_argument("name")
    p.add_argument("--files", nargs="*", default=[],
                   help="只拉这些文件名（默认拉除 trainer.pt 外全部）")
    p.add_argument("--with-trainer", action="store_true", help="连 trainer.pt 一起拉")
    p.set_defaults(func=cmd_model_pull)

    args = ap.parse_args(argv)
    try:
        return int(args.func(args))
    except OpsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
