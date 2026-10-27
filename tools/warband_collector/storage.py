"""Versioned raw observations, atomic persistence and portable evidence backups."""
import copy
from datetime import datetime, timezone
import gzip
import io
import json
from pathlib import Path
import re
import threading
import uuid
import zipfile

FORMAT = "taverngym.manual-warband.v1"
PHASES = {"unknown", "recruit_end", "pre_start", "starting", "after_start", "mid_combat"}
MAX_BYTES = 40 * 1024 * 1024
TRIBES = {"BEAST", "DEMON", "DRAGON", "ELEMENTAL", "MECHANICAL", "MURLOC", "PIRATE", "QUILBOAR", "UNDEAD", "ABERRATION"}


def now():
    return datetime.now(timezone.utc).isoformat()


def uid():
    return uuid.uuid4().hex


def fresh():
    return {"format": FORMAT, "revision": 0, "created_at": now(), "updated_at": now(), "games": []}


def number(value, label, minimum=None, maximum=None):
    if value is None:
        return
    if type(value) is not int or abs(value) > 9007199254740991 or (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f"{label} 必须为整数或留空（未知）")


def quality(game, shot):
    blockers = []
    quick = game.get("capture_mode") == "quick_combat" or shot.get("capture_mode") == "quick_combat"
    if quick and shot.get("entry_touched") is False and not shot.get("submitted"):
        blockers.append("尚未填写的自动草稿")
    if not quick and not game.get("source", "").strip(): blockers.append("缺录像来源")
    if not game.get("patch", "").strip(): blockers.append("版本未知")
    if not quick and not shot.get("timestamp", "").strip(): blockers.append("缺视频时间戳")
    if not quick and not shot.get("reviewed"): blockers.append("战场尚未核对")
    board = shot.get("board", [])
    if not board and not shot.get("empty_board_confirmed"): blockers.append("空战场未确认")
    for i, card in enumerate(board, 1):
        if not card.get("name", "").strip() or any(card.get(k) is None for k in ("attack", "health", "golden")):
            blockers.append(f"第{i}张牌缺卡名/形态/攻血")
        if not quick and card.get("review") != "observed": blockers.append(f"第{i}张牌待确认或含推断")
    combat = list(blockers)
    if shot.get("phase") != "pre_start": combat.append("不是已确认开战效果前状态")
    if not quick and shot.get("hand_status") == "unknown": combat.append("手牌状态未知")
    if any(not c.get("name", "").strip() or (not quick and c.get("review") != "observed") for c in shot.get("hand", [])):
        combat.append("手牌存在空项/待核对/推断内容")
    if not quick and shot.get("effects_status") == "unknown": combat.append("全局/技能效果未知")
    if not quick and not shot.get("combat_details_checked"): combat.append("附件/额外亡语/战斗计数尚未审阅")
    return {"statistics_ready": not blockers, "statistics_blockers": blockers,
            "capture_mode": "quick_combat" if quick else "legacy_evidence",
            "hand_scope": shot.get("hand_scope", "full_hand"),
            "external_audit_required": not quick,
            "combat_review_candidate": not combat, "combat_blockers": combat,
            "native_import_ready": False, "training_import_allowed": False,
            "notice": "原始录像观察；精确编译、版本映射与引擎支持仍须另验收"}


def validate(data):
    if not isinstance(data, dict) or data.get("format") != FORMAT or not isinstance(data.get("games"), list):
        raise ValueError("不支持的采集格式；请导入采集器导出的备份 JSON/ZIP")
    if len(data["games"]) > 5000: raise ValueError("单库最多5000局，请分库")
    identifiers, source_splits = set(), {}
    def identifier(obj):
        value = obj.get("id")
        if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", value) or value in identifiers:
            raise ValueError("记录 ID 缺失、无效或重复")
        identifiers.add(value)
    def text(obj, keys):
        for k in keys:
            if not isinstance(obj.get(k, ""), str) or len(obj.get(k, "")) > 50000:
                raise ValueError(f"{k} 必须为文本，长度不超过50000")
            # Missing optional text remains explicitly unknown/empty, never a numeric zero.
            obj.setdefault(k, "")
    for game in data["games"]:
        if not isinstance(game, dict): raise ValueError("局记录必须为对象")
        identifier(game)
        text(game, ("source", "patch", "hero", "streamer", "notes", "start_timestamp", "source_id"))
        if not game.get("source_id"): raise ValueError("缺少原始局来源标识")
        tribes=game.get("available_tribes", [])
        if not isinstance(tribes,list) or any(not isinstance(t,str) or t not in TRIBES for t in tribes) or len(set(tribes))!=len(tribes):
            raise ValueError("无效或重复的本局可用种族")
        split = game.get("split", "unassigned")
        if split not in {"unassigned", "train", "validation", "test"}: raise ValueError("无效的数据划分")
        source = game["source_id"]
        if source in source_splits and source_splits[source] != split:
            raise ValueError("同一来源局不能跨训练/验证/考试划分")
        source_splits[source] = split
        number(game.get("placement"), "最终名次", 1, 8)
        if not isinstance(game.get("snapshots"), list) or len(game["snapshots"]) > 10000: raise ValueError("回合记录过多或无效")
        for shot in game["snapshots"]:
            if not isinstance(shot, dict): raise ValueError("回合必须为对象")
            identifier(shot)
            number(shot.get("turn"), "回合", 1, 200)
            if shot.get("turn") is None: raise ValueError("回合不能留空")
            for k in ("tier", "armor", "alive", "damage_cap"):
                number(shot.get(k), k, 0)
            number(shot.get("hand_count"), "总手牌数", 0, 10)
            if shot.get("hand_count") is not None and shot["hand_count"] < len(shot.get("hand", [])):
                raise ValueError("总手牌数不能少于已录战斗手牌数")
            number(shot.get("health"), "英雄生命")
            if shot.get("phase") not in PHASES or shot.get("role") not in {"player", "opponent", "ghost"}: raise ValueError("无效的状态时点/席位类型")
            for flag in ("reviewed", "empty_board_confirmed", "combat_details_checked"):
                if type(shot.get(flag)) is not bool: raise ValueError(f"{flag} 必须为布尔值")
            for flag in ("entry_touched", "submitted"):
                if flag in shot and type(shot[flag]) is not bool: raise ValueError(f"{flag} 必须为布尔值")
            text(shot, ("timestamp", "hero", "notes", "skill_notes", "global_notes", "seat", "battle_id", "result"))
            for k in ("hand_status", "effects_status"):
                if shot.get(k) not in {"unknown", "none", "recorded"}: raise ValueError(f"无效的{k}")
            if shot["hand_status"] != "recorded" and shot.get("hand"): raise ValueError("已有手牌时不能标记为空/未知；请删除牌或改为已记录")
            for zone, limit in (("board", 7), ("hand", 10)):
                cards = shot.get(zone)
                if not isinstance(cards, list) or len(cards) > limit: raise ValueError(f"{zone} 卡牌数量不合法")
                for card in cards:
                    if not isinstance(card, dict): raise ValueError("卡牌必须为对象")
                    identifier(card)
                    text(card, ("name", "card_id", "catalog_version", "details", "text"))
                    for k in ("attack", "health", "max_health"): number(card.get(k), k)
                    if card.get("golden") is not None and type(card["golden"]) is not bool: raise ValueError("金色状态必须为是/否/未知")
                    if card.get("review") not in {"pending", "observed", "inferred"}: raise ValueError("无效卡牌证据状态")
                    if not isinstance(card.get("keywords"), list) or not all(isinstance(x, str) for x in card["keywords"]): raise ValueError("无效关键字")
            evidence = shot.get("evidence", [])
            if not isinstance(evidence, list) or any(not isinstance(e, str) or not re.fullmatch(r"[a-f0-9]{32}\.(png|jpg|jpeg|webp|mp4|webm)", e) for e in evidence):
                raise ValueError("无效证据附件标识")
    return data


def dumps(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def atomic(path, value):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        f.write(value)
        f.flush()
        import os
        os.fsync(f.fileno())
    temp.replace(path)


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.path = self.directory / "collection.json"
        self.history = self.directory / "history"
        self.attachments = self.directory / "attachments"
        self.history.mkdir(exist_ok=True)
        self.attachments.mkdir(exist_ok=True)
        if self.path.exists():
            self.data = validate(json.loads(self.path.read_text(encoding="utf-8")))
        else:
            self.data = fresh()
            atomic(self.path, dumps(self.data))

    def load(self):
        with self.lock: return copy.deepcopy(self.data)

    def save(self, data, revision):
        with self.lock:
            if type(revision) is not int or revision != self.data["revision"]:
                raise Conflict("另一页面已保存新版本；本页草稿已保留，请先导出草稿再重新加载，勿覆盖")
            data = copy.deepcopy(validate(data))
            data["revision"] = revision + 1
            data["created_at"] = self.data["created_at"]
            data["updated_at"] = now()
            # Snapshot before replacement; all revisions remain inspectable.
            atomic(self.history / f"{revision:08d}.json.gz", gzip.compress(dumps(self.data)))
            atomic(self.path, dumps(data))
            self.data = data
            return self.load()

    def attach(self, name, content):
        ext = Path(name).suffix.lower()
        if ext not in {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm"} or len(content) > MAX_BYTES:
            raise ValueError("附件仅支持图片/MP4/WebM，单个最多40MB")
        filename = uid() + ext
        atomic(self.attachments / filename, content)
        return filename

    def backup(self):
        with self.lock:
            data = self.load()
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("collection.json", dumps(data))
                for name in sorted({e for g in data["games"] for s in g["snapshots"] for e in s.get("evidence", [])}):
                    p = self.attachments / name
                    if not p.is_file(): raise ValueError(f"缺少证据附件 {name}，不能生成完整备份")
                    z.write(p, "attachments/" + name)
            return buffer.getvalue()

    def merge(self, content, revision, zipped=False):
        with self.lock:
            attachments = {}
            if zipped:
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    infos = z.infolist()
                    if len(infos) > 20000 or sum(i.file_size for i in infos) > 200 * 1024 * 1024:
                        raise ValueError("备份解压总量超过200MB或文件过多")
                    if len({i.filename for i in infos}) != len(infos): raise ValueError("备份包含重名文件")
                    incoming = json.loads(z.read("collection.json"))
                    for i in infos:
                        if i.filename == "collection.json": continue
                        if not re.fullmatch(r"attachments/[a-f0-9]{32}\.(png|jpg|jpeg|webp|mp4|webm)", i.filename):
                            raise ValueError("备份包含非采集器附件路径")
                        if i.file_size > MAX_BYTES: raise ValueError("附件超过40MB")
                        attachments[i.filename.split("/")[1]] = z.read(i)
            else:
                incoming = json.loads(content)
            validate(incoming)
            data = self.load()
            old_ids = {g["id"] for g in data["games"]}
            if old_ids & {g["id"] for g in incoming["games"]}:
                raise ValueError("存在相同局ID，拒绝重复导入。恢复完整备份请用新的空数据目录，原库不覆盖")
            required = {e for g in incoming["games"] for s in g["snapshots"] for e in s.get("evidence", [])}
            if any(e not in attachments and not (self.attachments / e).is_file() for e in required):
                raise ValueError("JSON所引用附件不齐，请导入完整ZIP备份")
            data["games"].extend(incoming["games"])
            validate(data)
            if revision != self.data["revision"]: raise Conflict("导入前数据已变化，请重新加载")
            for name, payload in attachments.items():
                dest = self.attachments / name
                if dest.exists() and dest.read_bytes() != payload: raise ValueError("证据附件ID冲突")
            for name, payload in attachments.items(): atomic(self.attachments / name, payload)
            return self.save(data, revision)

    def jsonl(self):
        data = self.load()
        def untouched(s):
            return s.get("capture_mode") == "quick_combat" and s.get("entry_touched") is False and not s.get("submitted")
        rows = [{"format": FORMAT, "type": "header", "revision": data["revision"],
                 "training_import_allowed": False, "human_calibrated": False,
                 "omitted_auto_drafts": sum(untouched(s) for g in data["games"] for s in g["snapshots"])}]
        for game in data["games"]:
            meta = {k: v for k, v in game.items() if k != "snapshots"}
            rows.append({"type": "game", "game": meta})
            for shot in game["snapshots"]:
                if untouched(shot): continue
                rows.append({"type": "snapshot", "game_id": game["id"], "source_id": game["source_id"],
                             "split": game.get("split", "unassigned"), "snapshot": shot, "quality": quality(game, shot)})
        return ("\n".join(json.dumps(r, ensure_ascii=False, allow_nan=False) for r in rows) + "\n").encode("utf-8")
