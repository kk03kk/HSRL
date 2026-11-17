#!/usr/bin/env python
"""把分片评测的原始 rows/extra 合并成一份多模型 ``panel.json``。

为什么要重写（2026-11-17 审计 P0-3 / P1-6）
------------------------------------------
* **旧行为**：``eval_human_gap.py`` 把每个模型的 rows 写到**同一个**
  ``rows.jsonl.gz``，一个进程跑多个 ``--checkpoint`` 时后者覆盖前者，于是本工具
  只能重建出**最后一个**模型（``gap-v1.2-plus3-1024`` 的 ``panel.json`` 就只剩
  ``v1.3_r3``，其它三个模型的数据已被覆盖）。现在落盘按模型名分文件
  （``rows-<name>.jsonl.gz`` / ``extra-<name>.json``），本工具按名归并。
  旧命名仍兼容：只有 ``rows.jsonl.gz``/``extra.json`` 的老分片照样能读。
* **有效性计数必须跨分片求和**：``illegal_actions``/``truncated_games`` 是
  ``docs/EVAL_PROTOCOL.md`` §5 的"第 0 步"（不为 0 则整轮作废）。旧的多模型合并
  工具只取第 0 个分片的计数，导致这道门只覆盖 1/24 的局数。
* **配对键要带上 seed**：``rank_sequence`` 只有顺序，没有种子；分片缺失时按下标配对
  会整体错位（审计 P1-6）。合并时同时输出 ``seed_sequence``。
* **协议指纹必须唯一**：所有分片必须同 ``protocol_sha256``，否则拒绝合并
  （``docs/EVAL_PROTOCOL.md`` §4 规则 5）。合并结果把它提到 ``protocol`` 段。

聚合逻辑**不重写**：直接 import ``workflows.eval_human_gap`` 里的同一批函数，
"分片合并 ≡ 串行跑完"由构造保证。
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.az.train.eval_policy import assert_same_protocol
from tools.panel_integrity import validate_identity, validate_rows, validate_samples

from workflows.eval_human_gap import _safe_name  # Compatibility filename convention.
from hsbg.evaluation.metrics import (  # noqa: E402
    _counts_all,
    _counts_by_turn,
    _gold_by_turn,
    _opponents_by_turn,
    _strength_by_seed,
    _trace,
    action_mix_by_turn,
    boards_by_turn,
    combat_by_turn,
    economy_curves,
    final_boards,
    mechanics_by_turn,
    retained_gain_by_turn,
    survival,
)

#: 名次类指标（按逐局名次重算）以外的计数（跨分片求和）。第 3 项 ``decisions`` 是求和。
SUMMED_OUTCOME_KEYS = ("illegal_actions", "truncated_games", "decisions")


# ────────────────────────── 分片读取（多模型 + 旧命名兼容）──────────────────────────
def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise SystemExit(f'{path}: cannot read JSON: {exc}') from exc


def load_shard(shard: Path) -> Dict[str, Dict[str, Any]]:
    """读一个分片目录 -> ``{模型名: {"rows": [...], "extra": {...}}}``。

    兼容三种落盘布局：

    1. 新：``extra-<name>.json`` + ``rows-<name>.jsonl.gz``（每个模型各一份）；
    2. 旧单模型：``extra.json`` + ``rows.jsonl.gz``；
    3. 混合：新文件存在时优先用新的，旧文件只在**同名模型没有新文件**时兜底。
    """
    out: Dict[str, Dict[str, Any]] = {}
    rows_by_name: Dict[str, List[Dict[str, Any]]] = {}

    per_model_rows: Dict[str, Path] = {}
    for p in sorted(shard.glob("rows-*.jsonl.gz")):
        per_model_rows[p.name[len("rows-"):-len(".jsonl.gz")]] = p
    legacy_rows = shard / "rows.jsonl.gz"

    extras: List[Dict[str, Any]] = []
    seen_extra_names: set = set()
    for p in sorted(glob.glob(str(shard / "extra-*.json"))):
        d = _read_json(Path(p))
        if d.get("name"):
            if str(d['name']) in seen_extra_names:
                raise SystemExit(f'{shard}: duplicate model name {d["name"]}')
            extras.append(d)
            seen_extra_names.add(str(d["name"]))
    legacy_extra = shard / "extra.json"
    if legacy_extra.is_file():
        d = _read_json(legacy_extra)
        # 旧名文件与某个新名文件**同名**时视为重复，不重复计数
        if d.get("name") and str(d["name"]) not in seen_extra_names:
            extras.append(d)

    for extra in extras:
        name = str(extra.get("name"))
        rows_path = per_model_rows.get(_safe_name(name))
        if rows_path is None and legacy_rows.is_file() and len(per_model_rows) <= 1:
            rows_path = legacy_rows
        rows: List[Dict[str, Any]] = []
        if rows_path is None or not rows_path.is_file():
            raise SystemExit(f'{shard}: missing rows for {name}')
        if rows_path is not None and rows_path.is_file():
            with gzip.open(rows_path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except ValueError as exc:
                            raise SystemExit(f'{rows_path}: corrupt JSON row') from exc
        rows_by_name[name] = rows
        out[name] = {"rows": rows, "extra": extra}
    return out


def merge_outcome(extras: List[Dict[str, Any]]) -> Dict[str, Any]:
    """按逐局名次重算名次类指标（与 ``evaluate.py`` 同口径）；计数**跨分片求和**。"""
    ranks: List[int] = []
    for e in extras:
        for r in (e.get("rank_sequence") or []):
            try:
                ranks.append(int(r))
            except (TypeError, ValueError):
                pass
    games = sum(int(e.get("games") or 0) for e in extras)
    out: Dict[str, Any] = {
        "games": games,
        "illegal_actions": sum(int((e.get("outcome") or {}).get("illegal_actions") or 0)
                               for e in extras),
        "truncated_games": sum(int((e.get("outcome") or {}).get("truncated_games") or 0)
                               for e in extras),
        "decisions": sum(int((e.get("outcome") or {}).get("decisions") or 0)
                         for e in extras),
        # 截断局（rank=0）**保留**在 rank_sequence 里；名次类指标只用 1..8。
        "truncated_rank_zero": sum(1 for r in ranks if r == 0),
        "rank_known": sum(1 for r in ranks if 1 <= r <= 8),
    }
    out["decisions_per_game"] = (round(out["decisions"] / games, 1) if games else None)
    ok = [r for r in ranks if 1 <= r <= 8]
    if ok:
        out["mean_rank"] = round(sum(ok) / len(ok), 4)
        out["top4_rate"] = round(sum(1 for r in ok if r <= 4) / len(ok), 4)
        out["win_rate"] = round(sum(1 for r in ok if r == 1) / len(ok), 4)
    return out


def _seed_sequence(extras: List[Dict[str, Any]],
                   ranks: List[int]) -> Optional[List[int]]:
    """逐局种子（与 ``rank_sequence`` 一一对应）；旧分片没有该字段时返回 None。"""
    seq: List[int] = []
    for e in extras:
        part = e.get("seed_sequence")
        if not isinstance(part, list):
            return None
        seq.extend(int(x) for x in part)
    if ranks and len(seq) != len(ranks):
        return None
    return seq or None


def _merge_field_stats(extras: List[Dict[str, Any]]) -> Dict[str, Any]:
    """注入统计跨分片求和 + 重算成功率（审计 P1-3：注入率必须能被面板自证）。"""
    acc: Dict[str, Any] = collections.Counter()
    library: Dict[str, Any] = {}
    for e in extras:
        fs = e.get("field_stats") or {}
        for k, v in fs.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                acc[k] += float(v)
            elif k == "library" and isinstance(v, dict) and not library:
                library = dict(v)
    out: Dict[str, Any] = {k: (int(v) if float(v).is_integer() else round(v, 4))
                           for k, v in acc.items()}
    attempt = float(acc.get("mirror_attempt") or 0.0)
    applied = float(acc.get("mirror_applied") or 0.0)
    out["mirror_apply_rate"] = round(applied / attempt, 4) if attempt else None
    if library:
        out["library"] = library
    return out


def _protocol(extras: List[Dict[str, Any]], shard: Path,
              panel_protocol: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """所有分片必须同 ``protocol_sha256``；返回合并后的 ``protocol`` 段。

    2026-11-17：以前这里只带 ``protocol_sha256``，把契约/场地/引擎口径都丢了 ——
    于是一个"能比"的面板没法自证"比的是什么"。现在优先取分片 ``panel.json`` 里
    **完整的 protocol 段**（各分片必然相同）。
    """
    shas = {str((e or {}).get("protocol_sha256") or "") for e in extras}
    if not shas or '' in shas:
        raise SystemExit(
            f"{shard}: 分片没有 protocol_sha256（旧代码产物）。"
            "该分片的可比性无法证明，请重跑评测；不允许把它并进新榜。")
    if len(shas) > 1:
        raise SystemExit(
            f"{shard}: 同一分片内的模型/分片协议不一致 {sorted(s[:12] for s in shas)}"
            "——不同协议不能混榜（docs/EVAL_PROTOCOL.md §4 规则 5）。")
    sha = next(iter(shas))
    if isinstance(panel_protocol, dict) and panel_protocol.get("protocol_sha256"):
        if str(panel_protocol["protocol_sha256"]) == sha:
            return dict(panel_protocol)
        raise SystemExit(f'{shard}: extra 与 panel protocol_sha256 不一致')
    return {"protocol_sha256": sha}


def _model_entry(name: str, shard_map: Dict[str, Dict[str, Any]],
                 ident: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    parts = list(shard_map.values())
    rows = [r for part in parts for r in part["rows"]]
    extras = [part["extra"] for part in parts]
    validate_samples(extras, name=name)
    validate_identity(extras + [ident or {}], name=name)
    all_seeds = [s for e in extras for s in (e.get('seed_sequence') or [])]
    complete = all(isinstance(e.get('seed_sequence'), list) for e in extras)
    truncated = any((e.get('outcome') or {}).get('truncated_games') for e in extras)
    validate_rows(rows, name=name, seeds=all_seeds if complete and not truncated else None)
    hist: collections.Counter = collections.Counter()
    exposure: List[Dict[str, Any]] = []
    ranks: List[int] = []
    for e in extras:
        for k, v in (e.get("action_hist") or {}).items():
            hist[str(k)] += int(v)
        exposure.extend(e.get("combat_exposure") or [])
        for r in (e.get("rank_sequence") or []):
            try:
                ranks.append(int(r))
            except (TypeError, ValueError):
                pass
    entry = {
        "name": name,
        "games": len(_strength_by_seed(rows)) or None,
        "shards": [str(p["extra"].get("shard") or "") for p in parts],
        "outcome": merge_outcome(extras),
        "rank_sequence": ranks or None,
        "seed_sequence": _seed_sequence(extras, ranks),
        "action_hist": dict(hist.most_common()) or None,
        "field_stats": _merge_field_stats(extras),
        "survival": survival(rows),
        "curves": economy_curves(rows),
        "action_mix_by_turn": action_mix_by_turn(rows),
        "combat_by_turn": combat_by_turn(exposure),
        "mechanics_by_turn": mechanics_by_turn(rows),
        "retained_gain_by_turn": retained_gain_by_turn(rows),
        "final_boards": final_boards(rows),
        "boards_by_turn": boards_by_turn(rows),
        "strength_by_seed": _strength_by_seed(rows),
        "trace": _trace(rows),
        "trinkets_by_turn": _counts_by_turn(rows, "trinkets"),
        "opponents_by_turn": _opponents_by_turn(exposure),
        "gold_by_turn": _gold_by_turn(rows),
        "heroes": _counts_all(rows, "hero"),
        "rows": len(rows),
        "decision_traces": [dict(part['extra']['decision_trace'],
                                 directory=str(Path(directory).resolve()))
                            for directory, part in shard_map.items()
                            if part['extra'].get('decision_trace')],
    }
    # 2026-11-17（验证发现）：**模型身份必须跟着走**。以前这里只拼统计量，
    # `checkpoint` / `checkpoint_sha256` / `arch` / `seed_start` / `seed_end` 全丢 ——
    # 合并出来的判胜面板无法自证"评的是哪个权重"。这些字段各分片相同，取第一个即可。
    first = extras[0] if extras else {}
    ident = ident or {}
    for key in ("checkpoint", "checkpoint_sha256", "arch"):
        val = first.get(key, ident.get(key))
        if val is not None:
            entry[key] = val
    # seed 区间必须取**全部**分片的 min/max，不能抄第一个分片的（否则面板会说谎）。
    starts = [int(e["seed_start"]) for e in extras if e.get("seed_start") is not None]
    ends = [int(e["seed_end"]) for e in extras if e.get("seed_end") is not None]
    if not starts and ident.get("seed_start") is not None:
        starts = [int(ident["seed_start"])]
    if not ends and ident.get("seed_end") is not None:
        ends = [int(ident["seed_end"])]
    if starts:
        entry["seed_start"] = min(starts)
    if ends:
        entry["seed_end"] = max(ends)
    entry["shard_count"] = len(extras)
    entry["seconds"] = sum(float(e.get("seconds") or 0.0) for e in extras) or None
    return entry


def _shard_panel_identity(name: str,
                          shard_dirs: List[Path]) -> Dict[str, Any]:
    """从分片 ``panel.json`` 里聚合该模型的**身份字段**（旧分片的 extra 没这些键）。

    * ``checkpoint`` / ``checkpoint_sha256`` / ``arch``：各分片相同，取第一个；
    * ``seed_start`` / ``seed_end``：取**全部**分片的 min / max（合并面板要覆盖整轮）。
    """
    ident: Dict[str, Any] = {}
    starts: List[int] = []
    ends: List[int] = []
    for d in shard_dirs:
        doc = _read_json(Path(d) / "panel.json")
        for mdl in doc.get("models") or []:
            if str(mdl.get("name")) != str(name):
                continue
            for k in ("checkpoint", "checkpoint_sha256", "arch"):
                if k in ('checkpoint_sha256', 'arch') and mdl.get(k) is not None:
                    if k in ident and ident[k] != mdl[k]:
                        raise SystemExit(f'{name}: inconsistent {k} across shard panels')
                if k in mdl and k not in ident:
                    ident[k] = mdl[k]
            if mdl.get("seed_start") is not None:
                starts.append(int(mdl["seed_start"]))
            if mdl.get("seed_end") is not None:
                ends.append(int(mdl["seed_end"]))
    if starts:
        ident["seed_start"] = min(starts)
    if ends:
        ident["seed_end"] = max(ends)
    return ident


def _shard_panel_protocol(shard_dirs: List[Path]) -> Optional[Dict[str, Any]]:
    """从第一个带 ``protocol`` 的分片 ``panel.json`` 里取**完整**协议段。"""
    protocols = []
    for d in shard_dirs:
        doc = _read_json(Path(d) / "panel.json")
        proto = doc.get("protocol")
        if isinstance(proto, dict) and proto.get("protocol_sha256"):
            protocols.append(dict(proto))
    if not protocols:
        return None
    try:
        assert_same_protocol(*[{'protocol': p} for p in protocols])
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return max(protocols, key=len)


def merge(shard_dirs: List[Path], out_path: Path, *,
          expect_games: Optional[int] = None,
          expect_models: Optional[int] = None) -> Dict[str, Any]:
    """合并分片。``expect_*`` 给出时必须**精确相等**，否则拒绝写盘（缺分片不静默）。"""
    if out_path.exists():
        raise SystemExit(f'refusing to overwrite existing panel: {out_path}')
    if len({Path(d).resolve() for d in shard_dirs}) != len(shard_dirs):
        raise SystemExit('duplicate shard directory')
    per_shard = [(d, load_shard(d)) for d in shard_dirs]
    names: List[str] = []
    for _d, m in per_shard:
        for n in m:
            if n not in names:
                names.append(n)
    if not names:
        raise SystemExit("no models found in any shard (rows-*/extra-* 或旧名缺失)")

    merged_models: List[Dict[str, Any]] = []
    protocol = _protocol([v['extra'] for _, models in per_shard for v in models.values()],
                         shard_dirs[0], _shard_panel_protocol(shard_dirs))
    for name in names:
        shard_map = {str(d): m[name] for d, m in per_shard if name in m}
        if len(shard_map) != len(per_shard):
            missing = [str(d) for d, m in per_shard if name not in m]
            raise SystemExit(f"模型 {name!r} 在分片 {missing} 里缺失——拒绝合并（不静默降级）")
        merged_models.append(_model_entry(
            name, shard_map, ident=_shard_panel_identity(name, shard_dirs)))

    total_games = int((merged_models[0].get("outcome") or {}).get("games") or 0)
    doc = {
        "models": merged_models,
        "merged_from": [str(d) for d in shard_dirs],
        "protocol": protocol,
        "merge": {"schema": "hsai.panel.merge/v1", "shards": len(shard_dirs),
                  "models": len(names), "games_per_model": total_games},
    }
    if expect_models is not None and len(names) != int(expect_models):
        raise SystemExit(f"模型数 {len(names)} != 期望 {expect_models}（--expect-models）")
    if expect_games is not None:
        for m in merged_models:
            got = int((m.get("outcome") or {}).get("games") or 0)
            if got != int(expect_games):
                raise SystemExit(
                    f"模型 {m['name']!r} 名次样本 {got} != 期望 {expect_games}："
                    "分片缺失或 seeds 重叠，拒绝写盘")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(doc, ensure_ascii=False))
    for m in merged_models:
        cv = m["curves"]
        print("merged %-16s games=%-5s rows=%-6d T12=%-7s T16=%-7s rank=%-6s "
              "illegal=%-3s trunc=%-3s inject=%s"
              % (m["name"], m["games"], m["rows"],
                 (cv.get("12") or {}).get("strength_p50"),
                 (cv.get("16") or {}).get("strength_p50"),
                 (m["outcome"] or {}).get("mean_rank"),
                 (m["outcome"] or {}).get("illegal_actions"),
                 (m["outcome"] or {}).get("truncated_games"),
                 (m.get("field_stats") or {}).get("mirror_apply_rate")))
    print("protocol_sha256 =", protocol.get("protocol_sha256"))
    print("wrote", out_path)
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shards", nargs="+",
                    help="分片目录（含 rows*.jsonl.gz / extra*.json）或 panel.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-games", type=int, default=None,
                    help="每个模型应有的名次样本数；不符即拒绝写盘（防缺分片）")
    ap.add_argument("--expect-models", type=int, default=None,
                    help="应有的模型数；不符即拒绝写盘")
    a = ap.parse_args(argv)
    dirs = []
    for s in a.shards:
        p = Path(s)
        dirs.append(p.parent if p.is_file() else p)
    merge(dirs, Path(a.out), expect_games=a.expect_games,
          expect_models=a.expect_models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
