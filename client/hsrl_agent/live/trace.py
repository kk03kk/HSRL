"""Async, bounded game trace writer shared by the simulator and the live bridge.

Both producers write the **same** files with the same semantics
(:mod:`hsrl_agent.trace_schema` is the spec):

* ``decisions.jsonl``        one row per *real* model decision
* ``decisions-full.jsonl.gz`` full snapshot + raw model output per decision
* ``actions.jsonl``          one row per execution attempt (before/after/delta/outcome)
* ``waits.jsonl``            wait-loop polls: same state, same action, not a decision
* ``events.jsonl``           bridge/script internals (retry, branch, block, receipt)
* ``observations.npy``       exact model input vectors, in decision order
* ``decisions-resolved.jsonl`` derived join of decision + execution
* ``audit.json`` / ``summary.json`` / ``performance.json`` / ``decision-report.md``

The controller hot path never waits for JSON formatting or disk fsync: a single
writer thread owns the streams and the producer only builds a dict and enqueues
it. A failure to enqueue is counted in ``events_dropped`` and reported at close,
never silently ignored.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import gzip
import json
from pathlib import Path
import queue
import threading
import time
from typing import Any, Iterable

from .. import trace_schema as S


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Bound view objects (live ``Binding``, sim ``_Binding``) and similar
    # records: expose their public fields instead of failing the whole row.
    slots = getattr(type(value), "__slots__", None)
    if slots:
        return {name: _jsonable(getattr(value, name, None)) for name in slots}
    if hasattr(value, "__dict__"):
        return {k: _jsonable(v) for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def client_summary(client: dict | None) -> dict | None:
    """规范化的我方视图；两种记录流共用同一形状。"""
    return S.view(client)


class GameTrace:
    SCHEMA = S.SCHEMA

    def __init__(self, out: Path, *, side: str = "live", queue_size: int = 2048,
                 store_vectors: bool = True, max_vector_rows: int = 12000):
        if side not in S.SIDES:
            raise ValueError(f"unknown trace side: {side}")
        self.side = side
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        # Screenshots are ~10-20 MB decoded. A bounded, separate queue keeps
        # a slow disk from retaining hundreds of frames in memory.
        self._frames: queue.Queue = queue.Queue(maxsize=4)
        self._closed = False
        self.enqueued = 0
        self.dropped = 0
        self.frame_dropped = 0
        self.write_errors = 0
        self.sequence = 0
        self.waits = 0
        self.decisions = 0
        self.executions = 0
        self._last_signature = None
        self._occurrences: dict = {}
        self._turn_actions: dict = {}
        self._last_seen = None
        self._attempts: dict = {}
        self.store_vectors = bool(store_vectors)
        self.max_vector_rows = int(max_vector_rows)
        self._vectors: list = []
        self._cards: list = []
        self.vectors_skipped = 0
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._writer, name="trace-writer", daemon=True)
        self._thread.start()
        self._frame_thread = threading.Thread(target=self._frame_writer, name="trace-frames", daemon=True)
        self._frame_thread.start()

    # ------------------------------------------------------------------ #
    # writer
    # ------------------------------------------------------------------ #
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
            "wait": self.out / "waits.jsonl",
            "event": self.out / "events.jsonl",
        }
        handles = {
            "decision": paths["decision"].open("a", encoding="utf-8"),
            "full": gzip.open(paths["full"], "at", encoding="utf-8"),
            "action": paths["action"].open("a", encoding="utf-8"),
            "wait": paths["wait"].open("a", encoding="utf-8"),
            "event": paths["event"].open("a", encoding="utf-8"),
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

    # ------------------------------------------------------------------ #
    # producers
    # ------------------------------------------------------------------ #
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
        role: str = "model",
        seat: Any = None,
        inputs: dict | None = None,
    ) -> str:
        """Record one model decision.

        ``reused`` means the policy cache avoided a second forward pass. A
        reused decision on the *identical* state and action is the wait loop
        re-reading the same frame; it goes to ``waits.jsonl`` instead of
        ``decisions.jsonl`` so per-turn action statistics stay clean.
        """
        payload = _jsonable(decision)
        bindings = tuple(bindings)
        client = snapshot.get("client") or {}
        stamp = payload.get("stamp") or S.state_key(client)
        signature = (stamp, payload.get("action"))
        turn = client.get("turn")
        if reused and signature == self._last_signature:
            self.waits += 1
            self._enqueue({
                "_stream": "wait",
                "event": "wait",
                "schema": S.SCHEMA,
                "side": self.side,
                "at": time.time(),
                "sequence": self.sequence + 1,
                "turn": turn,
                "phase": client.get("phase"),
                "action": payload.get("action"),
                "kind": S.action_kind(payload.get("action")),
                "stamp": stamp,
            })
            return f"wait-{self.sequence + 1:05d}"
        self._last_signature = signature
        self.sequence += 1
        occurrence = self._occurrences.get((turn, payload.get("action"), stamp), 0)
        self._occurrences[(turn, payload.get("action"), stamp)] = occurrence + 1
        # 同一回合内第几次选择同一个动作：>0 说明这是"同一意图的重读/重试"，
        # 状态变了所以是独立决策，但统计上要能和真正的第二次操作区分开。
        turn_key = (seat, turn)
        prior_same = self._turn_actions.setdefault(turn_key, {})
        repeat_index = prior_same.get(int(payload.get("action") or 0), 0)
        prior_same[int(payload.get("action") or 0)] = repeat_index + 1
        did = S.decision_id(side=self.side, game_id=payload.get("game_id"), seat=seat,
                            turn=turn, state=stamp, action=payload.get("action"),
                            occurrence=occurrence)
        binding = payload.get("binding") or {}
        view = snapshot.get("view") or {}
        zones = {}
        for zone in ("board", "hand", "shop", "choices", "controls"):
            # 没有 entity_id 的行（如模拟器的候选卡）不能写成 None 键，否则
            # 会互相覆盖并伪造出"来源卡牌"。
            zones[zone] = {c.get("entity_id"): c.get("card_id")
                           for c in client.get(zone) or [] if c.get("entity_id") is not None}
        source = binding.get("source")
        target = binding.get("target")
        source_card = next((zones[z].get(source) for z in zones if source in zones[z]), None) \
            or (f"entity:{source}" if source is not None else None)
        target_card = next((zones[z].get(target) for z in zones if target in zones[z]), None) \
            or (f"entity:{target}" if target is not None else None)
        kind = binding.get("kind") or S.action_kind(payload.get("action"))
        intent = (
            f"{kind} {source_card}"
            + (f" -> {target_card}" if target is not None else "")
            + (f" @slot {binding.get('position')}" if binding.get("position") is not None else "")
        )
        approximation = snapshot.get("approximation") or {}
        pre = S.view(client)
        compact = {
            "_stream": "decision",
            "schema": S.SCHEMA,
            "side": self.side,
            "event": "decision",
            "at": time.time(),
            "sequence": self.sequence,
            "decision_index_in_game": self.decisions,
            "decision_id": did,
            "parent_decision_id": parent,
            "game_id": payload.get("game_id"),
            "seat": seat,
            "role": role,
            "visibility": S.VISIBILITY[self.side],
            "revision": payload.get("revision"),
            "turn": turn,
            "phase": client.get("phase"),
            "action": payload.get("action"),
            "kind": kind,
            "action_class": S.action_class(kind),
            "intent": intent,
            "source": source,
            "target": target,
            "source_card_id": source_card if isinstance(source_card, str)
            and not source_card.startswith("entity:") else None,
            "target_card_id": target_card if isinstance(target_card, str)
            and not target_card.startswith("entity:") else None,
            "position": binding.get("position"),
            "option_id": binding.get("option_id"),
            "choice_id": binding.get("choice_id"),
            "policy_ms": policy_ms,
            "reused": bool(reused),
            "repeat_index": repeat_index,
            "binding_count": len(bindings),
            # 完整合法动作集合（245 维掩码的全部为真下标），外加有上限的明细。
            "legal": [int(b.action) for b in bindings],
            "legal_actions": [
                {"action": b.action, "kind": b.kind, "source": b.source,
                 "target": b.target, "position": b.position}
                for b in bindings[:40]
            ],
            "legal_actions_omitted": max(0, len(bindings) - 40),
            # 输入指纹：观察向量 / 掩码 / 原始 logits 各一个 sha256，用于逐位对比。
            "input": {
                "observation_sha256": payload.get("observation_sha256"),
                "mask_sha256": payload.get("mask_sha256"),
                "logits_sha256": payload.get("logits_sha256"),
                "vector": (inputs or {}).get("features_meta"),
                "cards": (inputs or {}).get("cards_meta"),
            },
            "checkpoint_sha256": payload.get("checkpoint_sha256"),
            "state_key": S.state_key(client),
            "board_signature": S.board_signature(client),
            "pre": pre,
            "inferred_view": {
                "pending": view.get("pending"),
                "unknown": view.get("unknown"),
                "approximation_fields": sorted(approximation.get("fields", {})),
                "approximation_mode": approximation.get("mode"),
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
            "schema": S.SCHEMA,
            "side": self.side,
            "event": "decision",
            "decision_id": did,
            "parent_decision_id": parent,
            "sequence": self.sequence,
            "snapshot": recorded,
            "decision": payload,
            "bindings": [_jsonable(b) for b in bindings],
            "plan": plan,
        }
        self.decisions += 1
        self._enqueue(compact)
        self._maybe_combat(turn, pre, seat)
        if full:
            # The snapshot is treated as immutable once the decision is made.
            # Serialization/compression happens on the writer thread, so the
            # model hot path pays neither a deep copy nor JSON/gzip time.
            self._enqueue(full_record)
        if self.store_vectors and inputs:
            features = inputs.get("features")
            cards = inputs.get("cards")
            if features is not None and len(self._vectors) < self.max_vector_rows:
                self._vectors.append(features)
                self._cards.append(cards)
            elif features is not None:
                self.vectors_skipped += 1
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
        channel: str = "mouse",
    ) -> None:
        attempt = self._attempts.get(did, 0)
        self._attempts[did] = attempt + 1
        before = S.view((snapshot or {}).get("client"))
        record = {
            "_stream": "action",
            "schema": S.SCHEMA,
            "side": self.side,
            "event": "execution",
            "at": time.time(),
            "decision_id": did,
            "attempt": attempt,
            "channel": channel,
            "physical": bool(physical),
            "status": status,
            "outcome": outcome,
            "error": error,
            "plan": plan,
            "timing": timing,
            "native_trace": native_trace,
            "before": before,
            "after": S.view(after),
            # 执行前后的状态差值：模拟器由引擎真值算出，实机由回执后的快照算出。
            "delta": S.delta(before, after),
            "received_at_log_position": (after or {}).get("log_position"),
        }
        self.executions += 1
        self._enqueue(record)
        if after is not None:
            self._enqueue({
                "_stream": "full",
                "schema": S.SCHEMA,
                "side": self.side,
                "event": "after_state",
                "decision_id": did,
                "attempt": attempt,
                "status": status,
                "outcome": outcome,
                "after": after,
            })

    def _maybe_combat(self, turn: Any, pre: dict | None, seat: Any) -> None:
        """跨回合的生命差 = 上一场战斗的净伤害；两端同一条规则自动补记。

        实机没有引擎的 combat 事件流，但"上一回合结算后掉了多少血"是可观测的；
        这条记录只陈述观测到的差值，不推断胜负或对手。
        """
        previous = self._last_seen
        health = (pre or {}).get("health")
        self._last_seen = (turn, health)
        if previous is None or not isinstance(turn, int) or not isinstance(previous[0], int):
            return
        if turn <= previous[0]:
            return
        before, after = previous[1], health
        if not isinstance(before, int) or not isinstance(after, int):
            return
        self.note("combat_damage", seat=seat, from_turn=previous[0], to_turn=turn,
                  health=[before, after], damage=max(0, before - after),
                  source="observed between consecutive decisions")

    def wait_poll(self, *, turn: Any, phase: Any, action: Any, stamp: Any, reason: str = "poll") -> None:
        """Explicit wait poll that is *not* a decision (no inference happened)."""
        self.waits += 1
        self._enqueue({
            "_stream": "wait", "schema": S.SCHEMA, "side": self.side, "event": "wait",
            "at": time.time(), "sequence": self.sequence, "turn": turn, "phase": phase,
            "action": action, "kind": S.action_kind(action), "stamp": stamp, "reason": reason,
        })

    def note(self, kind: str, /, **fields: Any) -> None:
        """Bridge/script internal event; never a model decision.

        ``kind`` 是位置参数名，调用方可以自由使用 ``kind=`` 字段（动作类型）。
        """
        self._enqueue({"_stream": "event", "schema": S.SCHEMA, "side": self.side,
                       "event": kind, "at": time.time(), **fields})

    #: 兼容旧调用点（controller.runtime 里的 trace.event(...)）。
    event = note

    def combat(self, **fields: Any) -> None:
        self.note("combat", **fields)

    def turn_change(self, **fields: Any) -> None:
        self.note("turn", **fields)

    def game_end(self, **fields: Any) -> None:
        self.note("game_end", **fields)

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

    # ------------------------------------------------------------------ #
    # finalize
    # ------------------------------------------------------------------ #
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
            "schema": S.SCHEMA,
            "side": self.side,
            "events_enqueued": self.enqueued,
            "events_dropped": self.dropped,
            "frames_dropped": self.frame_dropped,
            "write_errors": self.write_errors,
            "decisions": self.decisions,
            "waits": self.waits,
            "executions": self.executions,
            "writer_alive": self._thread.is_alive(),
            "frame_writer_alive": self._frame_thread.is_alive(),
        }
        try:
            result["vectors"] = self._flush_vectors()
        except Exception as exc:
            result["vectors_error"] = f"{type(exc).__name__}: {exc}"
        try:
            result["summary"] = summarize(self.out, side=self.side)
        except Exception as exc:
            result["summary_error"] = f"{type(exc).__name__}: {exc}"
        return result

    def _flush_vectors(self) -> dict:
        if not self.store_vectors:
            return {"stored": 0, "enabled": False}
        if not self._vectors:
            return {"stored": 0, "skipped": self.vectors_skipped, "enabled": True}
        import numpy as np
        features = np.asarray(self._vectors, dtype=np.float32)
        np.save(self.out / "observations.npy", features)
        if self._cards and all(c is not None for c in self._cards):
            cards = np.asarray(self._cards)
            np.save(self.out / "observations-cards.npy", cards)
        result = {"stored": int(features.shape[0]), "shape": list(features.shape),
                  "skipped": self.vectors_skipped, "enabled": True}
        self._vectors = []
        self._cards = []
        return result


# --------------------------------------------------------------------------- #
# derived artifacts
# --------------------------------------------------------------------------- #
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


def summarize(out: Path, *, side: str | None = None) -> dict:
    """Build ``decisions-resolved.jsonl`` / ``audit.json`` / ``summary.json`` /
    ``performance.json`` / ``decision-report.md`` from the raw streams."""
    out = Path(out)
    decisions = [row for row in _read_jsonl(out / "decisions.jsonl") if row.get("event") == "decision"]
    actions = _read_jsonl(out / "actions.jsonl")
    executions = [a for a in actions if a.get("event") == "execution"]
    events = [a for a in actions if a.get("event") != "execution"]
    events += _read_jsonl(out / "events.jsonl")
    waits = _read_jsonl(out / "waits.jsonl")
    side = side or (decisions[0].get("side") if decisions else (executions[0].get("side") if executions else "live"))
    physical = [a for a in executions if a.get("physical")]
    applied = [a for a in executions if a.get("outcome") == "applied"]
    failures = [
        a for a in executions
        if a.get("status") in ("cancelled_before_press", "delivery_unknown", "failed", "blocked_before_gesture")
        or a.get("outcome") in ("rejected", "halted", "blocked")
    ]
    real = [d for d in decisions if not d.get("reused")]
    rows, problems = S.join_decisions(decisions, executions)
    with (out / "decisions-resolved.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    audit = S.audit(decisions, executions, waits, side=side)
    audit["problems"] = list(dict.fromkeys(audit["problems"] + problems))
    audit["complete"] = not audit["problems"]
    (out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    performance = {
        "schema": "hsai.game-trace-performance/v2",
        "side": side,
        "decisions": len(decisions),
        "decisions_with_inference": len(real),
        "waits": len(waits),
        "inference_ms": _percentiles([d.get("policy_ms") for d in real]),
        "executions": len(physical),
        "receipts": len(executions) - len(physical),
        "applied": len(applied),
        "failed_or_rejected": len(failures),
        "action_success_rate": (len(applied) / len(physical)) if physical else None,
        "input_elapsed_ms": _percentiles([
            (o.get("timing") or {}).get("elapsed_ms") for o in physical
        ]),
        "actions_per_decision": (len(physical) / len(real)) if real else None,
    }
    turns = S.turns_summary(decisions)
    summary = {
        "schema": S.SCHEMA,
        "side": side,
        "visibility": S.VISIBILITY.get(side),
        "generated_at": time.time(),
        "files": S.STREAMS,
        "counts": {
            "decisions": len(decisions),
            "decisions_with_inference": len(real),
            "waits": len(waits),
            "events": len(events),
            "executions": len(physical),
            "applied": len(applied),
            "failed_or_rejected": len(failures),
            "unresolved": audit["unresolved_execution"],
        },
        "action_kinds": {},
        "action_classes": {},
        "turns": turns,
        "complete": audit["complete"],
        "problems": audit["problems"],
        "decision_rows": [],
    }
    for row in decisions:
        kind = row.get("kind")
        summary["action_kinds"][kind] = summary["action_kinds"].get(kind, 0) + 1
        cls = row.get("action_class") or S.action_class(kind)
        summary["action_classes"][cls] = summary["action_classes"].get(cls, 0) + 1
    by_id = {}
    for outcome in executions:
        by_id.setdefault(outcome.get("decision_id"), []).append(outcome)
    for row in decisions:
        rows_for = sorted(by_id.get(row.get("decision_id"), []), key=lambda r: r.get("attempt", 0))
        last = rows_for[-1] if rows_for else None
        after = next((r["after"] for r in reversed(rows_for) if r.get("after") is not None), None)
        summary["decision_rows"].append({
            "decision_id": row.get("decision_id"),
            "turn": row.get("turn"),
            "phase": row.get("phase"),
            "action": row.get("action"),
            "kind": row.get("kind"),
            "action_class": row.get("action_class"),
            "source": row.get("source"),
            "target": row.get("target"),
            "policy_ms": row.get("policy_ms"),
            "reused": row.get("reused"),
            "outcome": (last or {}).get("outcome"),
            "status": (last or {}).get("status"),
            "state_changed": (S.delta(row.get("pre"), after) or {}).get("changed"),
        })
    (out / "performance.json").write_text(
        json.dumps(performance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(out, summary, performance, audit)
    return summary


def _write_report(out: Path, summary: dict, performance: dict, audit: dict) -> None:
    lines = [
        "# 对局记录",
        "",
        f"规范 `{S.SCHEMA}`，来源 `{summary.get('side')}`，可见性 `{summary.get('visibility')}`。",
        "本报告由 `hsrl_agent.trace_schema` 从原始记录生成；不把模型推断值当作真实观测。",
        "",
        "## 性能",
        "",
        f"- 真实决策：{performance['decisions']}（其中重新推理 {performance['decisions_with_inference']}，等待轮询 {performance['waits']}）",
        f"- 推理耗时 p50/p90：{performance['inference_ms']['p50']} / {performance['inference_ms']['p90']} ms",
        f"- 执行阶段：{performance['executions']}，已确认 {performance['applied']}，失败/拒绝 {performance['failed_or_rejected']}",
        f"- 动作成功率：{performance['action_success_rate']}",
        f"- 输入耗时 p50/p90：{performance['input_elapsed_ms']['p50']} / {performance['input_elapsed_ms']['p90']} ms",
        "",
        "## 逐回合动作类型",
        "",
        "| 回合 | 阶段 | 金币 | 等级 | 决策数 | 有效动作 | 动作类型 |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for turn, bucket in (summary.get("turns") or {}).items():
        kinds = ", ".join(f"{k}×{v}" for k, v in sorted(bucket["kinds"].items()))
        lines.append(f"| {turn} | {bucket.get('phase')} | {bucket.get('gold')} | "
                     f"{bucket.get('tier')} | {bucket['decisions']} | "
                     f"{bucket.get('distinct', bucket['decisions'])} | {kinds} |")
    lines += [
        "",
        "## 完整性",
        "",
        f"- 结论：{'完整' if audit.get('complete') else '存在问题'}",
        f"- 缺执行：{audit.get('missing_execution')}，未终态：{audit.get('unresolved_execution')}",
    ]
    for problem in audit.get("problems") or []:
        lines.append(f"- 问题：{problem}")
    lines += [
        "",
        "## 文件说明",
        "",
    ]
    for name, description in S.STREAMS.items():
        lines.append(f"- `{name}`：{description}")
    lines += [
        "",
        "## 数据边界",
        "",
        f"可见性 `{summary.get('visibility')}`："
        + ("模拟器是引擎真值（含隐藏信息），实机是 Power.log 公开投影。"
           if summary.get("side") == "sim"
           else "真实观测来自客户端日志与截图；模拟器隐藏状态、随机池差量和未观察到的手牌不能从本记录伪造。")
        + "`approximation_fields` 列出每局不可重建的字段。",
        "",
    ]
    (out / "decision-report.md").write_text("\n".join(lines), encoding="utf-8")
