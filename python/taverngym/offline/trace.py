"""带严格边界检查的离线轨迹。全局重放状态不属于策略输入。"""
import hashlib
import json
from pathlib import Path

from .actions import decode

FORMAT = "taverngym.trace.v1"


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def public_view(game, seat):
    view = json.loads(game.view(seat))
    # debug.view 的 opponents.board 可含真实隐藏战团，不能直接导出为玩家视图。
    return {"turn": view["turn"], "phase": view["phase"], "seat": seat,
            "me": view["me"], "opponents": [
                {k: o[k] for k in ("id", "health", "armor", "tier", "alive", "placement") if k in o}
                for o in view["opponents"]]}


def write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(dumps(row) + "\n")


def read(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                rows.append(json.loads(line, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s))))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid trace JSON at line {line_no}: {exc}") from exc
    validate(rows)
    return rows


def validate(rows):
    if len(rows) < 2 or rows[0].get("format") != FORMAT or rows[0].get("type") != "header":
        raise ValueError("missing trace v1 header")
    games = len(rows[0].get("initial_debug_states", []))
    if games < 1 or rows[-1].get("type") != "summary":
        raise ValueError("incomplete trace: initial states or summary missing")
    for tick, row in enumerate(rows[1:-1]):
        if row.get("type") != "tick" or row.get("tick") != tick:
            raise ValueError("trace tick sequence is invalid")
        for field in ("actions", "ready", "applied", "rejected", "done", "placement"):
            if len(row.get(field, [])) != games * 8:
                raise ValueError(f"invalid trace width: {field}")
        if len(row.get("post_hashes", [])) != games:
            raise ValueError("missing post-state hashes")
        for i, a in enumerate(row["actions"]):
            if a == -1 and not row["ready"][i] and not row["applied"][i]:
                continue  # 原生 bot 的非活动席位哨兵，不是一个模型动作。
            decode(a)
        for d in row.get("decisions", []):
            i = d["slot"]
            if not 0 <= i < games * 8 or d["action"] != row["actions"][i]:
                raise ValueError("decision/slot action mismatch")
            if not row["ready"][i] or d["applied"] != row["applied"][i]:
                raise ValueError("decision/applied mismatch")
            obs = d["observation"]
            import math
            if len(obs.get("features", [])) != 3227 or len(obs.get("cards", [])) != 124:
                raise ValueError("invalid observation dimensions")
            if not all(isinstance(f, (int, float)) and math.isfinite(f) for f in obs["features"]):
                raise ValueError("invalid observation numeric values")
            if len(obs["mask"]) != 245 or not obs["mask"][d["action"]]:
                raise ValueError("decision is outside legal mask")
    return rows


def replay(rows):
    import taverngym as tg
    validate(rows)
    header = rows[0]
    if header["native"]["source_hash"] != tg.build_info()["source_hash"]:
        raise ValueError("replay requires matching native source identity")
    n = len(header["initial_debug_states"])
    env = tg.BatchEnv(n, controlled=range(8))
    env.auto_reset = False
    for g, state in enumerate(header["initial_debug_states"]):
        env.load(g, dumps(state))
    placements = [0] * (n * 8)
    for row in rows[1:-1]:
        actual_obs = env.observe()
        actual_extras = env.training_extras()
        if actual_obs["ready"].tolist() != row["ready"]:
            raise ValueError("replay ready-mask mismatch")
        for decision in row["decisions"]:
            i, obs = decision["slot"], decision["observation"]
            for source, target in (("features", "features"), ("cards", "cards"), ("mask", "action_mask")):
                if obs[source] != actual_obs[target][i].tolist():
                    raise ValueError(f"replay observation mismatch: {source}")
            if dumps(obs["extras"]) != dumps(actual_extras[i]):
                raise ValueError("replay effect tokens mismatch")
        # bot_actions 的内部 RNG 消耗也属于采集协议，不能省略。
        env.bot_actions()
        _, done, info = env.step(row["actions"])
        for name, actual in (("done", done), ("applied", info["applied"]),
                             ("rejected", info["rejected"]), ("placement", info["placement"])):
            if actual.tolist() != row[name]:
                raise ValueError(f"replay mismatch tick {row['tick']} {name}")
        hashes = [digest(json.loads(env.save(g))) for g in range(n)]
        if hashes != row["post_hashes"]:
            raise ValueError(f"replay state mismatch tick {row['tick']}")
        for i, finished in enumerate(done):
            if finished:
                placements[i] = int(info["placement"][i])
    expected = [r for lobby in rows[-1].get("placements", []) for r in lobby]
    if expected and expected != placements:
        raise ValueError("replay summary placement mismatch")
    return {"ticks": len(rows) - 2, "state_hashes_verified": True}


def statistics(rows):
    from collections import Counter
    validate(rows)
    groups, by_turn = Counter(), {}
    entropies, repeats = [], 0
    last = {}
    for row in rows[1:-1]:
        for d in row["decisions"]:
            if not d["applied"]:
                continue
            group = decode(d["action"])["kind"]
            groups[group] += 1
            turn = str(d["turn"])
            by_turn.setdefault(turn, Counter())[group] += 1
            if d.get("entropy") is not None:
                entropies.append(d["entropy"])
            key = (d["slot"], d["turn"])
            stamp = (d["action"], digest(d["before"]["me"]))
            repeats += int(last.get(key) == stamp)
            last[key] = stamp
    return {"groups": dict(groups), "by_turn": {k: dict(v) for k, v in by_turn.items()},
            "effective_decisions": sum(groups.values()), "repeated_same_action_and_state": repeats,
            "mean_entropy": sum(entropies) / len(entropies) if entropies else None,
            "deferred": sum(sum(r and not a for r, a in zip(x["ready"], x["applied"])) for x in rows[1:-1]),
            "rejected": sum(sum(x["rejected"]) for x in rows[1:-1]), "outcome": rows[-1]}


def import_failure(path, out):
    """原 format-2 快照转玩家回放；概率未知，禁止事后补推理冒充历史。"""
    import taverngym as tg
    source = json.loads(Path(path).read_text(encoding="utf-8"))
    if source.get("format") != 2 or "actions" not in source:
        raise ValueError("legacy snapshot has no action trace; exact replay unavailable")
    if source.get("native", {}).get("source_hash") != tg.build_info()["source_hash"]:
        raise ValueError("failure import requires the original native identity")
    env = tg.BatchEnv(1, controlled=range(8))
    env.auto_reset = False
    env.load(0, dumps(source["initial"]))
    rows = [{"type": "header", "format": FORMAT, "identity": {"schema": "legacy-native-only", "digest": None},
             "native": source["native"], "seed": source.get("seed"), "policy": "not_recorded",
             "initial_debug_states": [source["initial"]], "source": str(path)}]
    ranks, counts = [0] * 8, [0] * 8
    for tick, actions in enumerate(source["actions"]):
        obs, extras = env.observe(), env.training_extras()
        before = {i: public_view(env.game(0), i) for i in range(8) if obs["ready"][i]}
        env.bot_actions()
        _, done, info = env.step(actions)
        decisions = []
        for i, view in before.items():
            if info["rejected"][i]:
                continue  # 非法动作在 tick 原始事件中可查，不伪装为有效策略样本。
            counts[i] += int(info["applied"][i])
            decisions.append({"slot": i, "lobby": 0, "seat": i, "turn": view["turn"], "decision": counts[i],
                              "action": int(actions[i]), "semantic": decode(int(actions[i])),
                              "applied": bool(info["applied"][i]), "memory_reset": None,
                              "observation": {"features": obs["features"][i].tolist(), "cards": obs["cards"][i].tolist(),
                                              "mask": obs["action_mask"][i].tolist(), "extras": extras[i]},
                              "before": view, "after": public_view(env.game(0), i),
                              "value": None, "logp": None, "entropy": None, "topk": None})
        for i in range(8):
            if done[i]:
                ranks[i] = int(info["placement"][i])
        rows.append({"type": "tick", "tick": tick, "actions": actions, "ready": obs["ready"].tolist(),
                     "applied": info["applied"].tolist(), "rejected": info["rejected"].tolist(),
                     "done": done.tolist(), "placement": info["placement"].tolist(), "errors": info["errors"],
                     "decisions": decisions, "post_hashes": [digest(json.loads(env.save(0)))]})
    if source.get("post") and digest(source["post"]) != digest(json.loads(env.save(0))):
        raise ValueError("failure snapshot final state does not reproduce")
    rows.append({"type": "summary", "status": "legacy_snapshot", "placements": [ranks],
                 "ranks": [rank if rank else None for rank in ranks], "probabilities": "not_recorded"})
    validate(rows)
    write(out, rows)
    return {"ticks": len(rows) - 2, "probabilities": "not_recorded", "out": str(out)}
