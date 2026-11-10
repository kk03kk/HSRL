"""Per-game decision/execution trace with bounded asynchronous writes.

The controller hot path must never wait for JSON formatting or disk fsync.  A
single writer thread owns the four append-only streams:

* ``decisions.jsonl``      - compact decision index (one row per inference)
* ``decisions-full.jsonl.gz`` - exact model/client snapshot for every decision
* ``actions.jsonl``        - physical gesture, timing and causal outcome
* ``trace-errors.jsonl``   - writer/enqueue failures, never silently dropped

At finalization the same module writes a human-readable ``decision-report.md``
and machine-readable ``summary.json`` / ``performance.json``.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
import gzip
import hashlib
import json
from pathlib import Path
import queue
import threading
import time
from typing import Any, Iterable


def decision_id(decision: Any) -> str:
    if is_dataclass(decision):
        payload = asdict(decision)
    else:
        payload = decision
    key = {
        "game_id": payload.get("game_id"),
        "revision": payload.get("revision"),
        "stamp": payload.get("stamp"),
        "action": payload.get("action"),
    }
    return "d-" + hashlib.sha256(
        json.dumps(key, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def client_summary(client: dict | None) -> dict | None:
    """Compact real-observation summary; never a model feature vector."""
    if not client:
        return None
    hero = client.get("hero") or {}
    def card(row: dict) -> dict:
        return {
            "entity_id": row.get("entity_id"),
            "card_id": row.get("card_id"),
            "name": row.get("name"),
            "card_type": row.get("card_type"),
            "position": row.get("position"),
            "attack": row.get("attack"),
            "health": row.get("health"),
            "tier": row.get("tier"),
            "cost": row.get("cost"),
            "tags": {
                key: row["tags"][key]
                for key in (
                    "ATK", "HEALTH", "DAMAGE", "TAUNT", "DIVINE_SHIELD", "REBORN",
                    "WINDFURY", "MEGA_WINDFURY", "VENOMOUS", "POISONOUS", "STEALTH",
                    "MAGNETIC", "FROZEN", "PREMIUM", "CARDRACE", "TECH_LEVEL",
                )
                if key in (row.get("tags") or {})
            },
        }
    return {
        "turn": client.get("turn"),
        "phase": client.get("phase"),
        "gold": client.get("gold"),
        "tavern_tier": client.get("tavern_tier"),
        "upgrade_cost": client.get("upgrade_cost"),
        "health": hero.get("health"),
        "armor": (hero.get("tags") or {}).get("ARMOR"),
        "hero": {
            "entity_id": hero.get("entity_id"),
            "card_id": hero.get("card_id"),
            "name": hero.get("name"),
            "health": hero.get("health"),
            "armor": (hero.get("tags") or {}).get("ARMOR"),
        },
        "board": [card(c) for c in client.get("board") or []],
        "hand": [card(c) for c in client.get("hand") or []],
        "shop": [card(c) for c in client.get("shop") or []],
        "choices": [card(c) for c in client.get("choices") or []],
        "controls": [card(c) for c in client.get("controls") or []],
        "interaction": client.get("interaction"),
        "log_position": client.get("log_position"),
        "resolutions": [
            {
                "kind": r.get("kind"),
                "source": r.get("source"),
                "target": r.get("target"),
                "complete": r.get("complete"),
                "log_position": r.get("log_position"),
            }
            for r in (client.get("resolutions") or [])[-8:]
        ],
    }


class GameTrace:
    SCHEMA = "hsai.live-trace/v1"

    def __init__(self, out: Path, *, queue_size: int = 256):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        # Screenshots are ~10-20 MB decoded. A bounded, separate queue keeps
        # a slow disk from retaining hundreds of frames in memory.
        self._frames: queue.Queue = queue.Queue(maxsize=4)
        self._closed = False
        self._lock = threading.Lock()
        self.enqueued = 0
        self.dropped = 0
        self.frame_dropped = 0
        self.write_errors = 0
        self.sequence = 0
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._writer, name="live-trace-writer", daemon=True)
        self._thread.start()
        self._frame_thread = threading.Thread(target=self._frame_writer, name="live-trace-frames", daemon=True)
        self._frame_thread.start()

    def _enqueue(self, record: dict) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            # A full queue is an engineering error in the trace layer, not a
            # reason to pause a game action. Record the exact loss count.
            self.dropped += 1
        else:
            self.enqueued += 1

    def _write_error(self, exc: BaseException, record: dict) -> None:
        self.write_errors += 1
        try:
            with (self.out / "trace-errors.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(
                    {"at": time.time(), "error": f"{type(exc).__name__}: {exc}",
                     "event": record.get("event"), "decision_id": record.get("decision_id")},
                    ensure_ascii=False,
                ) + "\n")
        except Exception:
            pass

    def _writer(self) -> None:
        paths = {
            "decision": self.out / "decisions.jsonl",
            "full": self.out / "decisions-full.jsonl.gz",
            "action": self.out / "actions.jsonl",
        }
        handles = {
            "decision": paths["decision"].open("a", encoding="utf-8"),
            "full": gzip.open(paths["full"], "at", encoding="utf-8"),
            "action": paths["action"].open("a", encoding="utf-8"),
        }
        try:
            while True:
                record = self._queue.get()
                try:
                    if record is None:
                        break
                    stream = handles.get(record.pop("_stream", "decision"), handles["decision"])
                    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    # Visibility during a live game: these are small, bounded
                    # JSONL rows. Flushing happens on the writer thread, never
                    # on the input hot path; no fsync is needed per decision.
                    stream.flush()
                except BaseException as exc:
                    self._write_error(exc, record or {})
                finally:
                    self._queue.task_done()
        finally:
            for stream in handles.values():
                try:
                    stream.close()
                except Exception:
                    pass

    def _frame_writer(self) -> None:
        from .contracts import replace_file
        while True:
            try:
                record = self._frames.get(timeout=0.2)
            except queue.Empty:
                if self._closed:
                    break
                continue
            try:
                image = record["image"]
                path = Path(record["path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                # Pillow's JPEG writer chooses its encoder from the suffix;
                # a ".jpg.tmp" name would silently become PNG.
                temporary = path.with_name(path.stem + ".tmp.jpg")
                image.convert("RGB").save(temporary, format="JPEG", quality=int(record.get("quality", 88)))
                replace_file(temporary, path)
                metadata = record.get("metadata") or {}
                meta_path = path.with_suffix(".json")
                meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
                meta_tmp.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                replace_file(meta_tmp, meta_path)
            except BaseException as exc:
                self._write_error(exc, record or {})
            finally:
                self._frames.task_done()

    def decision(
        self,
        snapshot: dict,
        decision: Any,
        *,
        policy_ms: float,
        reused: bool,
        bindings: Iterable[Any] = (),
        plan: list | None = None,
        full: bool = True,
        parent: str | None = None,
    ) -> str:
        self.sequence += 1
        did = f"{decision_id(decision)}-{self.sequence:04d}"
        payload = _jsonable(decision)
        bindings = tuple(bindings)
        binding = payload.get("binding") or {}
        client = snapshot.get("client") or {}
        view = snapshot.get("view") or {}
        board = {c.get("entity_id"): c.get("card_id") for c in client.get("board") or []}
        hand = {c.get("entity_id"): c.get("card_id") for c in client.get("hand") or []}
        shop = {c.get("entity_id"): c.get("card_id") for c in client.get("shop") or []}
        choices = {c.get("entity_id"): c.get("card_id") for c in client.get("choices") or []}
        source = binding.get("source")
        target = binding.get("target")
        source_card = (
            hand.get(source) or board.get(source) or shop.get(source)
            or choices.get(source) or f"entity:{source}"
        )
        target_card = (
            board.get(target) or hand.get(target) or shop.get(target)
            or choices.get(target) or f"entity:{target}"
        )
        intent = (
            f"{binding.get('kind')} {source_card}"
            + (f" -> {target_card}" if target is not None else "")
            + (f" @slot {binding.get('position')}" if binding.get("position") is not None else "")
        )
        compact = {
            "_stream": "decision",
            "event": "decision",
            "at": time.time(),
            "sequence": self.sequence,
            "decision_id": did,
            "parent_decision_id": parent,
            "game_id": payload.get("game_id"),
            "revision": payload.get("revision"),
            "stamp": payload.get("stamp"),
            "turn": client.get("turn"),
            "phase": client.get("phase"),
            "action": payload.get("action"),
            "kind": binding.get("kind"),
            "intent": intent,
            "source": binding.get("source"),
            "target": binding.get("target"),
            "position": binding.get("position"),
            "option_id": binding.get("option_id"),
            "choice_id": binding.get("choice_id"),
            "policy_ms": policy_ms,
            "reused": bool(reused),
            "binding_count": len(bindings),
            "legal_actions": [
                {
                    "action": b.action,
                    "kind": b.kind,
                    "source": b.source,
                    "target": b.target,
                    "position": b.position,
                }
                for b in bindings
            ],
            "decision_sha256": payload.get("stamp"),
            "observation_sha256": payload.get("observation_sha256"),
            "mask_sha256": payload.get("mask_sha256"),
            "logits_sha256": payload.get("logits_sha256"),
            "real_observation": client_summary(client),
            "inferred_view": {
                "pending": view.get("pending"),
                "unknown": view.get("unknown"),
                "approximation_fields": sorted((snapshot.get("approximation") or {}).get("fields", {})),
            },
            "plan": plan,
        }
        # Receipt post-state histories are independently archived in the raw
        # Power.log. Keeping the newest identities bounds this per-decision
        # record while preserving the exact current model/client snapshot.
        recorded = dict(snapshot)
        recorded["client"] = dict(client)
        recorded["client"]["resolutions"] = [
            {k: v for k, v in row.items() if k != "post_state"}
            for row in (client.get("resolutions") or [])[-8:]
        ]
        recorded["receipt_history_omitted"] = True
        full_record = {
            "_stream": "full",
            "event": "decision",
            "decision_id": did,
            "parent_decision_id": parent,
            "snapshot": recorded,
            "decision": payload,
            "bindings": [_jsonable(b) for b in bindings],
            "plan": plan,
        }
        self._enqueue(compact)
        if full:
            # The snapshot is treated as immutable once the decision is made.
            # Serialization/compression happens on the writer thread, so the
            # model hot path pays neither a deep copy nor JSON/gzip time.
            self._enqueue(full_record)
        return did

    def execution(
        self,
        did: str,
        *,
        snapshot: dict | None,
        plan: list,
        status: str,
        error: str | None = None,
        timing: dict | None = None,
        native_trace: dict | None = None,
        outcome: str | None = None,
        after: dict | None = None,
        physical: bool = False,
    ) -> None:
        record = {
            "_stream": "action",
            "event": "execution",
            "at": time.time(),
            "decision_id": did,
            "physical": bool(physical),
            "status": status,
            "outcome": outcome,
            "error": error,
            "plan": plan,
            "timing": timing,
            "native_trace": native_trace,
            "before": client_summary((snapshot or {}).get("client")),
            "after": client_summary(after),
        }
        self._enqueue(record)
        if after is not None:
            self._enqueue({
                "_stream": "full",
                "event": "after_state",
                "decision_id": did,
                "status": status,
                "outcome": outcome,
                "after": after,
            })

    def event(self, kind: str, **fields: Any) -> None:
        self._enqueue({"_stream": "action", "event": kind, "at": time.time(), **fields})

    def frame(self, path: Path, image: Any, metadata: dict, *, quality: int = 88) -> bool:
        """Queue a screenshot; JPEG encoding never runs before a mouse press."""
        if self._closed:
            return False
        record = {
            "_stream": "frame",
            "at": time.time(),
            "path": str(path),
            "image": image,
            "metadata": metadata,
            "quality": int(quality),
        }
        try:
            self._frames.put_nowait(record)
        except queue.Full:
            self.frame_dropped += 1
            self.dropped += 1
            return False
        else:
            self.enqueued += 1
            return True

    def close(self) -> dict:
        if not self._closed:
            self._closed = True
            try:
                self._queue.put(None, timeout=5)
            except queue.Full:
                self.dropped += 1
            self._thread.join(timeout=20)
            self._frame_thread.join(timeout=20)
        result = {
            "events_enqueued": self.enqueued,
            "events_dropped": self.dropped,
            "frames_dropped": self.frame_dropped,
            "write_errors": self.write_errors,
            "writer_alive": self._thread.is_alive(),
            "frame_writer_alive": self._frame_thread.is_alive(),
        }
        try:
            summary = summarize(self.out)
            result["summary"] = summary
        except Exception as exc:
            result["summary_error"] = f"{type(exc).__name__}: {exc}"
        return result


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _percentiles(values: list[float]) -> dict:
    import numpy as np

    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not arr.size:
        return {"count": 0, "p50": None, "p90": None, "mean": None, "max": None}
    return {
        "count": int(arr.size),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }


def summarize(out: Path) -> dict:
    out = Path(out)
    decisions = _read_jsonl(out / "decisions.jsonl")
    actions = _read_jsonl(out / "actions.jsonl")
    executions = [a for a in actions if a.get("event") == "execution"]
    physical = [a for a in executions if a.get("physical")]
    applied = [a for a in executions if a.get("outcome") == "applied"]
    failures = [
        a for a in executions
        if a.get("status") in ("cancelled_before_press", "delivery_unknown", "failed")
        or a.get("outcome") in ("rejected", "halted", "blocked")
    ]
    real = [d for d in decisions if not d.get("reused")]
    performance = {
        "schema": "hsai.live-performance/v1",
        "decisions": len(decisions),
        "decisions_with_inference": len(real),
        "decision_reused": len(decisions) - len(real),
        "inference_ms": _percentiles([d.get("policy_ms") for d in real]),
        "executions": len(physical),
        "receipts": len(executions) - len(physical),
        "applied": len(applied),
        "failed_or_rejected": len(failures),
        "action_success_rate": (len(applied) / len(physical)) if physical else None,
        "input_elapsed_ms": _percentiles([
            (o.get("timing") or {}).get("elapsed_ms") for o in physical
        ]),
        "actions_per_decision": (len(physical) / len(decisions)) if decisions else None,
    }
    summary = {
        "schema": GameTrace.SCHEMA,
        "generated_at": time.time(),
        "files": {
            "decisions.jsonl": "one compact row per model decision",
            "decisions-full.jsonl.gz": "exact snapshot + model decision per row (gzip JSONL)",
            "actions.jsonl": "gesture plan, native timing, before/after summary and causal outcome",
            "raw/": "archived Power.log / LoadingScreen.log / GameNetLogger.log byte prefixes",
            "visual-evidence/": "bounded screenshots around visual blocks and dispatches",
        },
        "counts": {
            "decisions": len(decisions),
            "executions": len(physical),
            "applied": len(applied),
            "failed_or_rejected": len(failures),
        },
        "action_kinds": {},
        "decision_rows": [],
    }
    for row in decisions:
        kind = row.get("kind")
        summary["action_kinds"][kind] = summary["action_kinds"].get(kind, 0) + 1
    by_id = {}
    for outcome in executions:
        by_id.setdefault(outcome.get("decision_id"), []).append(outcome)
    for row in decisions:
        executions = by_id.get(row.get("decision_id"), [])
        last = executions[-1] if executions else None
        summary["decision_rows"].append({
            "decision_id": row.get("decision_id"),
            "turn": row.get("turn"),
            "phase": row.get("phase"),
            "action": row.get("action"),
            "kind": row.get("kind"),
            "source": row.get("source"),
            "target": row.get("target"),
            "policy_ms": row.get("policy_ms"),
            "reused": row.get("reused"),
            "outcome": (last or {}).get("outcome"),
            "status": (last or {}).get("status"),
        })
    (out / "performance.json").write_text(
        json.dumps(performance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(out, summary, performance)
    return summary


def _write_report(out: Path, summary: dict, performance: dict) -> None:
    lines = [
        "# 实机决策记录",
        "",
        "本报告由 `hsrl_agent.live.trace` 从每局原始记录生成；不把模型推断值当作真实观测。",
        "",
        "## 性能",
        "",
        f"- 决策数：{performance['decisions']}（真实推理 {performance['decisions_with_inference']}，复用 {performance['decision_reused']}）",
        f"- 推理耗时 p50/p90：{performance['inference_ms']['p50']} / {performance['inference_ms']['p90']} ms",
        f"- 执行阶段数：{performance['executions']}，已确认 {performance['applied']}，失败/拒绝 {performance['failed_or_rejected']}",
        f"- 动作成功率：{performance['action_success_rate']}",
        f"- 输入耗时 p50/p90：{performance['input_elapsed_ms']['p50']} / {performance['input_elapsed_ms']['p90']} ms",
        "",
        "## 逐决策",
        "",
        "| # | decision_id | 回合 | 阶段 | 动作 | 类型 | 源 | 目标 | 推理ms | 结果 |",
        "|---:|---|---:|---|---:|---|---:|---:|---:|---|",
    ]
    for i, row in enumerate(summary.get("decision_rows") or [], 1):
        lines.append(
            f"| {i} | `{row['decision_id']}` | {row['turn']} | {row['phase']} | "
            f"{row['action']} | {row['kind']} | {row['source']} | {row['target']} | "
            f"{row['policy_ms']:.3f} | {row['outcome'] or row['status'] or ''} |"
        )
    lines += [
        "",
        "## 文件说明",
        "",
        "- `decisions-full.jsonl.gz`：每次决策的完整真实客户端快照、模型视图、合法动作与决策。",
        "- `actions.jsonl`：鼠标计划、点击/拖拽坐标、原生输入轨迹、执行前后状态摘要和因果回执。",
        "- `raw/`：本局 Power.log 等原始字节前缀；隐藏信息仍只存在于服务端。",
        "- `visual-evidence/`：两帧稳定性、选择行、派发前截图等有限证据。",
        "",
        "## 数据边界",
        "",
        "真实观测来自客户端日志与截图；模型输入是公开状态投影，`approximation.json` 明确列出不可重建字段。模拟器隐藏状态、随机池差量和未观察到的手牌不能从本记录伪造。",
        "",
    ]
    (out / "decision-report.md").write_text("\n".join(lines), encoding="utf-8")
