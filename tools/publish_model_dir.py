#!/usr/bin/env python
"""把一个 run 目录发布成规范的模型目录（`model.pt + manifest.json + checkpoint.json + MODEL_CARD.md`）。

规范来源：`AGENTS.md` §2 的命名约定。**不复制大文件**：用 ``publish_checkpoint``
（同卷硬链接优先，失败才复制），所以不会占双份磁盘。

用法::

    # 只看会写什么（不落盘）
    python tools/publish_model_dir.py --run artifacts/runs/nus-arena-v3-tier-20261114-1 \
        --name arena-v3-tier-20261114 --dry-run

    # 真的发布
    python tools/publish_model_dir.py --run artifacts/runs/nus-arena-v3-tier-20261114-1 \
        --name arena-v3-tier-20261114 \
        --panel artifacts/runs/gap-v3-vs-champion-20261114/panel.json \
        --panel-model arena_v3

产物（``artifacts/models/<name>/``）：
- ``model.pt`` / ``trainer.pt`` —— 硬链接或复制
- ``checkpoint.json`` —— version / decisions / 两个文件的 sha256
- ``manifest.json`` —— schema ``hsai.model/v1.1``，含引擎锚点、git、作业号、配方摘要
- ``MODEL_CARD.md`` —— 人读卡片；给了 ``--panel`` 就把评测成绩写进去
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.checkpoint_io import publish_checkpoint  # noqa: E402

TZ = timezone(timedelta(hours=8))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def parse_run_env(path: Path) -> Dict[str, str]:
    """run.env 是多行 key=value，但有的行塞了多个键（``job=1140935 node=xgpi13``）。

    返回扁平字典；同时把每一行原文放进 ``__lines__``（列表），供卡片原样展示。
    """
    out: Dict[str, str] = {}
    lines: list = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
        for tok in s.split():
            if "=" in tok:
                k, _, v = tok.partition("=")
                out.setdefault(k.strip(), v.strip())
    out["__lines__"] = "\n".join(lines)  # type: ignore[assignment]
    return out


def parse_engine_from_log(path: Path) -> Optional[Dict[str, Any]]:
    """从 train.log 的 ``[engine] {...}`` 行取引擎锚点（那次运行实际用的）。"""
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[engine] "):
                try:
                    return json.loads(line[len("[engine] "):])
                except ValueError:
                    return None
    return None


def find_parameters(log_path: Path) -> Optional[int]:
    if not log_path.is_file():
        return None
    pat = re.compile(r"\[net\] arch=\S+ params=([\d,]+)")
    with log_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                return int(m.group(1).replace(",", ""))
    return None


def model_card(name: str, man: Dict[str, Any], chk: Dict[str, Any],
               recipe: Dict[str, str], panel: Optional[Dict[str, Any]],
               panel_model: Optional[str]) -> str:
    L = []
    L.append(f"# {name}")
    L.append("")
    L.append(f"- 发布：{man['created_at']}")
    L.append(f"- 来源 run：`{man['source_path']}`（作业 {man.get('training', {}).get('job', '?')}，"
             f"节点 {man.get('training', {}).get('node', '?')}，git `{man.get('training', {}).get('git', '?')}`）")
    L.append(f"- 版本 / 累计决策：{man['version']} / {man['cum_decisions']:,}")
    L.append(f"- 网络：{man['arch']}，{man.get('parameters', '?')} 参数")
    eng = man.get("engine") or {}
    L.append(f"- 引擎锚点：`{eng.get('game_version')}` / 指纹 `{eng.get('source_hash')}` / "
             f"卡 {eng.get('cards')} / 动作 {eng.get('n_actions')} / 观测 {eng.get('obs_floats')}+{eng.get('obs_cards')}")
    L.append("")
    L.append("## 训练配方（该 run 的 run.env 原文）")
    L.append("")
    L.append("```")
    L.append(recipe.get("__lines__", "(没有 run.env)"))
    L.append("```")
    L.append("")
    L.append("## 文件")
    L.append("")
    L.append("| 文件 | 字节 | sha256 |")
    L.append("|---|---:|---|")
    for fn, info in (man.get("files") or {}).items():
        L.append(f"| `{fn}` | {info['bytes']:,} | `{info['sha256'][:16]}…` |")
    L.append("")
    if panel and panel_model:
        m = None
        for cand in panel.get("models") or []:
            if cand.get("name") == panel_model:
                m = cand
                break
        L.append(f"## 评测（`{panel_model}`，场地见 panel.meta）")
        L.append("")
        if m:
            o = m.get("outcome") or {}
            L.append("| mean_rank | 前四率 | 吃鸡率 | 非法动作 | 截断 | 决策/局 |")
            L.append("|---|---|---|---|---|---|")
            mr = o.get("mean_rank")
            t4 = o.get("top4_rate")
            wr = o.get("win_rate")
            L.append("| %s | %s | %s | %s | %s | %s |" % (
                f"{mr:.3f}" if isinstance(mr, (int, float)) else mr,
                f"{t4:.1%}" if isinstance(t4, (int, float)) else t4,
                f"{wr:.1%}" if isinstance(wr, (int, float)) else wr,
                o.get("illegal_actions"), o.get("truncated_games"),
                round(o.get("decisions_per_game") or 0, 1)))
            L.append("")
            L.append("| 回合 | n | M1 中位战力 | M2 平均 tier |")
            L.append("|---:|---:|---:|---:|")
            cv = m.get("curves") or {}
            for k in sorted(cv, key=lambda x: int(x)):
                r = cv[k] or {}
                L.append(f"| {k} | {r.get('n')} | {r.get('strength_p50')} | {r.get('tier_mean')} |")
            L.append("")
        else:
            L.append(f"⚠️ 面板里找不到模型 `{panel_model}`。")
            L.append("")
    L.append("## 判胜结论")
    L.append("")
    L.append("用 `python tools/compare_panels.py <panel.json> --champion <冠军名> "
             f"--candidate {panel_model or '<本模型名>'}` 生成；结论写进 `docs/PROJECT_STATE.md` §3。")
    L.append("")
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="publish a run dir into a model dir")
    ap.add_argument("--run", required=True, help="run 目录（含 train/ 与 run.env）")
    ap.add_argument("--name", required=True, help="模型目录名（artifacts/models/<name>）")
    ap.add_argument("--out-root", default="artifacts/models")
    ap.add_argument("--panel", default="", help="可选：评测 panel.json")
    ap.add_argument("--panel-model", default="", help="可选：面板里本模型的名字")
    ap.add_argument("--status", default="candidate", help="manifest.status（candidate/final/...）")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    run = Path(a.run)
    train = run / "train"
    model_pt, trainer_pt = train / "model.pt", train / "trainer.pt"
    for p in (model_pt, trainer_pt):
        if not p.is_file():
            raise SystemExit(f"缺少 {p}")
    latest = read_json(train / "latest.json") if (train / "latest.json").is_file() else {}
    env = parse_run_env(run / "run.env")

    out = Path(a.out_root) / a.name
    if out.exists() and any(out.iterdir()) and not a.dry_run:
        raise SystemExit(f"输出目录已存在且非空：{out}（拒绝覆盖；换 --name 或先手工处理）")

    created = datetime.now(TZ).isoformat(timespec="seconds")
    shas = {"model.pt": sha256_of(model_pt), "trainer.pt": sha256_of(trainer_pt)}
    files = {fn: {"sha256": shas[fn], "bytes": (train / fn).stat().st_size}
             for fn in shas}
    engine = parse_engine_from_log(run / "train.log") or {}
    man = {
        "schema": "hsai.model/v1.1",
        "id": a.name,
        "status": a.status,
        "created_at": created,
        "source_path": str(train),
        "version": latest.get("version"),
        "cum_decisions": latest.get("cum_decisions"),
        "parameters": find_parameters(run / "train.log"),
        "arch": latest.get("arch"),
        "seed": latest.get("seed"),
        "engine": {k: engine.get(k) for k in
                   ("game_version", "source_hash", "cards", "n_actions",
                    "obs_floats", "obs_cards")},
        "files": files,
        "training": {
            "run": str(run),
            "checkpoint": str(train),
            "job": env.get("job"),
            "node": env.get("node"),
            "git": env.get("git"),
        },
    }
    chk = {
        "target_decisions": latest.get("cum_decisions"),
        "actual_decisions": latest.get("cum_decisions"),
        "version": latest.get("version"),
        "sha256": dict(shas),
    }
    panel = read_json(Path(a.panel)) if a.panel.strip() else None
    card = model_card(a.name, man, chk, env, panel, a.panel_model or None)

    if a.dry_run:
        print(json.dumps({"out": str(out), "manifest": man, "checkpoint": chk},
                         ensure_ascii=False, indent=1))
        print("\n--- MODEL_CARD.md ---\n")
        print(card)
        return 0

    out.mkdir(parents=True, exist_ok=True)
    modes = {}
    for fn, src in (("model.pt", model_pt), ("trainer.pt", trainer_pt)):
        modes[fn] = publish_checkpoint(src, out / fn)
    for fn, obj in (("manifest.json", man), ("checkpoint.json", chk)):
        (out / fn).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n",
                              encoding="utf-8")
    (out / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    print(f"published {out}  modes={modes}")
    print(f"  files: {sorted(p.name for p in out.iterdir())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
