"""从合法自然轨迹导出回合末战团库；没有把合成战团注入训练环境。"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import taverngym as tg

from taverngym.rl.config import Config
from taverngym.rl.collect import observation
from taverngym.rl.opponents import profile_action


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def export_natural(path, *, seed, games=2, split="train", profile="native"):
    if split not in ("train", "validation", "test") or games < 1:
        raise ValueError("invalid bank split or game count")
    cfg = Config(device="cpu", n_games=games, threads=1, bot_profile=profile)
    cfg.validate()
    env = tg.BatchEnv(games, seed=seed, controlled=range(8), config=cfg.rules)
    env.auto_reset = False
    records, roots = [], {}
    done = np.zeros(games * 8, dtype=bool)
    ranks = np.zeros(games * 8, dtype=int)
    for tick in range(cfg.max_ticks):
        obs = env.observe()
        actions = env.bot_actions()
        extras = env.training_extras()
        for i in np.flatnonzero(obs["ready"]):
            actions[i] = profile_action(observation(obs, extras, i), int(actions[i]), profile, int(i))
        candidates = {}
        for i in np.flatnonzero(obs["ready"] & (actions == 0)):
            g, seat = int(i) // 8, int(i) % 8
            state = json.loads(env.save(g))["game"]
            root = fingerprint({"seed": state["seed"], "native": tg.build_info()["source_hash"]})
            roots[g] = root
            view = json.loads(env.game(g).view(seat))["me"]
            candidates[int(i)] = {"root": root, "split": split, "seat": seat, "turn": state["turn"],
                                  "capture_moment": "before_applied_end_turn_not_after_end_of_turn_effects",
                                  "source": "synthetic_legal_bot_trajectory", "profile": profile,
                                  "human_calibrated": False, "native_player": state["players"][seat],
                                  "visible_warband": view["board"], "provenance": {"base_seed": seed, "game": g, "tick": tick}}
        _, dn, info = env.step(actions)
        if info["rejected"].any() or any(a or b or c for a, b, c in env.diagnostics()):
            raise RuntimeError("bank generation guard: incomplete lobbies are not silently accepted")
        for i, row in candidates.items():
            if info["applied"][i]:
                records.append(row)
        done |= dn
        ranks[dn] = info["placement"][dn]
        if done.all():
            break
    else:
        raise RuntimeError("bank generation tick limit")
    for row in records:
        row["final_placement_metadata_only"] = int(ranks[row["provenance"]["game"] * 8 + row["seat"]])
    result = {"format": "taverngym.warband-bank.v1", "native": tg.build_info()["source_hash"],
              "engine": tg.game_version(), "rules": cfg.rules, "split": split, "roots": sorted(set(roots.values())),
              "source": "synthetic_legal_bot_trajectory", "human_calibrated": False,
              "policy_input": False, "training_import_allowed": False, "records": records}
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
    return result


def summarize(bank):
    turns = {}
    for row in bank["records"]:
        board = row["visible_warband"]
        values = turns.setdefault(str(row["turn"]), {"attack": [], "health": [], "size": [], "keywords": Counter()})
        values["attack"].append(sum(c["atk"] for c in board))
        values["health"].append(sum(c["hp"] for c in board))
        values["size"].append(len(board))
        for c in board:
            values["keywords"].update(c["keywords"])
    return {"source": bank["source"], "human_calibrated": False,
            "split": bank["split"], "roots": len(bank["roots"]), "records": len(bank["records"]),
            "turns": {turn: {**{k: {"n": len(v[k]), "p10_p50_p90": np.quantile(v[k], [.1, .5, .9]).tolist()}
                                     for k in ("attack", "health", "size")}, "keywords": dict(v["keywords"])}
                      for turn, v in turns.items()}}


def check_splits(banks):
    seen = {}
    for bank in banks:
        if bank["native"] != banks[0]["native"] or bank["rules"] != banks[0]["rules"]:
            raise ValueError("bank engine/rules mismatch")
        for root in bank["roots"]:
            if root in seen and seen[root] != bank["split"]:
                raise ValueError("same source lobby leaked across splits")
            seen[root] = bank["split"]
    return {"root_disjoint": True, "roots": len(seen),
            "notice": "split by source lobby; generic common boards may recur; not a human-strength exam"}


def import_client(path, out, *, split="test", max_lines=20000):
    """只读取旧系统JSONL公共己方战团，明确不是人类高手/完整原生状态。"""
    if split not in ("train", "validation", "test") or max_lines < 1:
        raise ValueError("invalid client import limits")
    path = Path(path)
    records, seen, roots, builds = [], set(), set(), set()
    unknown_fields = malformed = lines = 0
    source_hash = hashlib.sha256()
    complete = True
    with path.open("rb") as f:
        for index, raw in enumerate(f):
            if index >= max_lines:
                complete = False
                break
            lines += 1
            source_hash.update(raw)
            try:
                event = json.loads(raw)
            except ValueError:
                malformed += 1
                continue
            if not isinstance(event, dict):
                malformed += 1
                continue
            obj = event.get("sample", event)
            if not isinstance(obj, dict):
                malformed += 1
                continue
            obj = obj.get("observation", obj)
            if not isinstance(obj, dict):
                malformed += 1
                continue
            if obj.get("phase") != "recruit" or not obj.get("board") or not obj.get("game_id") or not obj.get("build"):
                continue
            def known_number(x):
                return isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)
            if (not known_number(obj.get("turn")) or obj["turn"] < 1
                    or not isinstance(obj["board"], list)
                    or any(not isinstance(c, dict) or not known_number(c.get("attack"))
                           or not known_number(c.get("health")) or not c.get("card_id")
                           or not isinstance(c.get("tags", {}), dict) for c in obj["board"])):
                unknown_fields += 1
                continue
            root = fingerprint({"capture": path.parent.parent.name, "game_id": obj["game_id"], "build": obj["build"]})
            board = [{"id": c["card_id"], "atk": c["attack"], "hp": c["health"], "tier": c.get("tier"),
                      "keywords": [k for k in ("TAUNT", "DIVINE_SHIELD", "VENOMOUS", "REBORN", "DEATHRATTLE")
                                   if str(c.get("tags", {}).get(k, "0")) == "1"]} for c in obj["board"]]
            key = fingerprint({"root": root, "turn": obj.get("turn"), "board": board})
            if key in seen:
                continue
            seen.add(key)
            roots.add(root)
            builds.add(str(obj["build"]))
            records.append({"root": root, "split": split, "turn": obj.get("turn"), "visible_warband": board,
                            "build": str(obj["build"]), "source": "historical_automated_agent_client",
                            "capture_moment": "recruit_snapshot_not_combat_entry", "human_calibrated": False,
                            "missing_fields": ["complete_native_counters", "effect_provenance", "opponent_hidden_hand"],
                            "provenance": {"line": lines}})
    result = {"format": "taverngym.warband-bank.v1", "native": None, "engine": "unmapped_client_builds",
              "rules": None, "split": split, "roots": sorted(roots), "builds": sorted(builds),
              "source": "historical_automated_agent_client", "human_calibrated": False,
              "training_import_allowed": False, "policy_input": False, "records": records,
              "source_prefix_sha256": source_hash.hexdigest(), "source_complete": complete,
              "lines_read": lines, "malformed_lines": malformed, "unknown_board_fields_skipped": unknown_fields}
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
    return result
