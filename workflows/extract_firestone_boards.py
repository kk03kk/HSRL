#!/usr/bin/env python3
"""只提取真人战团：复用卡表/库行构造；精确读取PMS gameState.currentTurn。"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import re
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows.parse_firestone_replay_xml import CardDb, KW_BITS, XmlReplay, _row_from_board

FLAGS = {"taunt": "TAUNT", "divineShield": "DIVINE_SHIELD", "windfury": "WINDFURY",
         "reborn": "REBORN", "venomous": "VENOMOUS", "poisonous": "POISONOUS", "stealth": "STEALTH"}


def split_game(game):
    bucket = int(hashlib.sha256(game.encode()).hexdigest()[:8], 16) % 10
    return "test" if bucket == 0 else "dev" if bucket == 1 else "train"


def convert_board(raw, cdb):
    if not isinstance(raw, list) or not 1 <= len(raw) <= 7:
        raise ValueError("empty_or_bad_board_size")
    result = []
    for pos, m in enumerate(raw, 1):
        cid = m.get("cardId")
        info = cdb.info(cid)
        if info is None or cdb.cid(cid) is None or info["kind"] != "minion":
            raise ValueError("unmapped_or_non_minion:" + str(cid))
        atk, hp = m.get("attack"), m.get("health")
        if isinstance(atk, bool) or isinstance(hp, bool) or not isinstance(atk, int) or not isinstance(hp, int) or atk < 0 or hp <= 0:
            raise ValueError("invalid_stats")
        if atk + hp > 100000:
            raise ValueError("current_pool_stat_cap")
        printed = cdb.printed_kw(cid)
        kw = printed
        for field, name in FLAGS.items():
            if field in m:
                kw &= ~KW_BITS[name]
                if m[field]:
                    kw |= KW_BITS[name]
        result.append(dict(card=cdb.cid(cid), card_id=cid, atk=atk, hp=hp,
            golden=cid.endswith("_G"), kw=kw, bonus_kw=kw & ~printed,
            shield_max=1, position=pos))
    if sum(m["atk"] + m["hp"] for m in result) > 250000:
        raise ValueError("current_pool_strength_cap")
    return result


def make_row(feed, board, turn, perspective, player, source, mmr=None):
    game = feed.get("originalReviewId") or feed["reviewId"]
    row = _row_from_board(game_id=game, turn=turn, board=board, hand=[], hand_available=False,
        tier=player.get("tavernTier"), gold=None, hero_cid=None, hero_card_id=player.get("cardId"),
        hp=player.get("hpLeft"), armor=None, placement=None, main_placement=None,
        source=source, perspective=perspective, opponent_hero=None, build=feed.get("buildNumber"),
        mmr=mmr if perspective == "self" else None,
        archetype=feed.get("bgsCompArchetype") if perspective == "self" else None,
        tribes=feed.get("bgsAvailableTribes"), banned=feed.get("bgsBannedTribes"),
        real_pms=source == "firestone_pms_exact_turn", action_counts={}, split=split_game(game))
    if row:
        row.update(schema="hsrl.human_board_observation/v1", fidelity="board_only_approx",
                   review_id=feed["reviewId"], main_player_mmr=mmr,
                   mmr_scope="uploader_only", is_ghost=bool(player.get("hpLeft") is not None and player["hpLeft"] <= 0))
        row["availability"].update(keywords="observed_flags_with_printed_fallback",
                                    shield_max="unavailable", hidden_counters="unavailable")
    return row


def pms_rows(feed, data, cdb):
    rows, dropped = [], Counter()
    for idx, battle in enumerate(data.get("battleResultHistory") or []):
        info = battle.get("battleInfo")
        if not info or battle.get("actualResult") is None:
            continue
        if info.get("playerTeammateBoard") is not None or info.get("opponentTeammateBoard") is not None:
            dropped["duos_battle"] += 1
            continue
        turn = (info.get("gameState") or {}).get("currentTurn")
        if not isinstance(turn, int) or isinstance(turn, bool) or not 1 <= turn <= 100:
            dropped["missing_exact_turn"] += 1
            continue
        for side, perspective in (("playerBoard", "self"), ("opponentBoard", "opponent")):
            raw = info.get(side) or {}
            try:
                board = convert_board(raw.get("board"), cdb)
                row = make_row(feed, board, turn, perspective, raw.get("player") or {},
                               "firestone_pms_exact_turn", data.get("oldMmr"))
                if row:
                    row.update(battle_index=idx, pms_outer_turn=battle.get("turn"),
                               round_alignment="battleInfo.gameState.currentTurn")
                    rows.append(row)
            except ValueError as exc:
                dropped[str(exc)] += 1
    return rows, dropped


def remove_player_display_names(data):
    """只移除固定Player头的显示名称，不处理Tag/Block或任何游戏数值。"""
    pattern = rb'(<Player\b[^>]*?)\sname=.*?(\sisMainPlayer="(?:true|false)"\s*>)'
    return re.sub(pattern, rb'\1\2', data, flags=re.S)


def parse_board_xml(path, cdb):
    try:
        return XmlReplay(cdb).parse(path), False
    except ET.ParseError:
        with zipfile.ZipFile(path) as archive:
            name = next(n for n in archive.namelist() if n.lower().endswith('.xml'))
            raw = archive.read(name)
        clean = remove_player_display_names(raw)
        if clean == raw:
            raise
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            archive.writestr('replay.xml', clean)
        buffer.seek(0)
        return XmlReplay(cdb).parse(buffer), True


def xml_worker(task):
    feed, path, cards, wait = task
    cdb = CardDb(cards)
    try:
        deadline = time.monotonic() + 900
        while wait and not Path(path).exists() and not (Path(path).parent.parent / "download-summary.json").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("XML下载等待超过15分钟")
            time.sleep(.5)
        xml, sanitized = parse_board_xml(path, cdb)
        rows = []
        for snap in xml["snapshots"]:
            board = snap["self_board"]
            if not board or any(m.get("card") is None for m in board):
                continue
            row = make_row(feed, board, snap["turn"], "self", {}, "firestone_xml_observed")
            if row:
                row.update(stage="xml_" + snap["reason"], round_alignment="XML_main_player_TURN",
                           xml_display_names_removed=sanitized,
                           completeness="observed_not_guaranteed_precombat_complete")
                rows.append(row)
        return feed["reviewId"], rows, None
    except Exception as exc:
        return feed["reviewId"], [], str(exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--xml-supplement", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wait-for-xml", action="store_true", help="并行下载时等待XML落盘，最多15分钟")
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    if out.exists():
        raise FileExistsError("提取输出必须为新目录")
    out.mkdir(parents=True)
    cards = str(ROOT / "engine/crates/hsbg-engine/data/cards.json")
    cdb = CardDb(cards)
    feed = json.loads((root / "index.json").read_text(encoding="utf-8"))
    stats = Counter()
    dropped = Counter()
    coverage, tasks, xml_errors = {}, [], []
    # 原始CardId观察另存，保留旧版本、超出现有池阈值等数据，不伪装成可注入帧。
    with gzip.open(out / "raw_observed_boards.jsonl.gz", "wt", encoding="utf-8") as raw_out:
        for game in feed:
            path = root / "pms" / (game["reviewId"] + ".json")
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            observed = False
            for idx, battle in enumerate(data.get("battleResultHistory") or []):
                info = battle.get("battleInfo")
                if not info or battle.get("actualResult") is None:
                    continue
                for side, perspective in (("playerBoard", "self"), ("opponentBoard", "opponent")):
                    raw = info.get(side) or {}
                    if not raw.get("board"):
                        continue
                    row = dict(schema="hsrl.raw_human_board/v1", review_id=game["reviewId"],
                        game_id=game.get("originalReviewId") or game["reviewId"],
                        build_number=game.get("buildNumber"), perspective=perspective, battle_index=idx,
                        turn=(info.get("gameState") or {}).get("currentTurn"),
                        raw_board=raw["board"], duos=info.get("playerTeammateBoard") is not None)
                    raw_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stats["raw_observed_boards"] += 1
                    observed = True
            stats["games_with_raw_boards"] += int(observed)
    files = {s: (out / f"{s}.jsonl").open("w", encoding="utf-8") for s in ("train", "dev", "test")}
    seen = set()
    with (out / "library_primary.jsonl").open("w", encoding="utf-8") as library:
        for game in feed:
            if game.get("buildNumber") != 255790:
                stats["historical_build_games_excluded"] += 1
                continue
            stats["compatible_catalog_games"] += 1
            path = root / "pms" / (game["reviewId"] + ".json")
            if path.exists():
                stats["pms_files"] += 1
                rows, failures = pms_rows(game, json.loads(path.read_text(encoding="utf-8")), cdb)
                dropped.update(failures)
                for row in rows:
                    signature = (row["game_id"], row["turn"], row["perspective"], json.dumps(row["board_detail"], sort_keys=True))
                    if signature in seen:
                        stats["duplicates"] += 1
                        continue
                    seen.add(signature)
                    line = json.dumps(row, ensure_ascii=False) + "\n"
                    library.write(line)
                    files[row["split"]].write(line)
                    stats["rows"] += 1
                    stats[row["perspective"] + "_rows"] += 1
                    coverage[game["reviewId"]] = coverage.get(game["reviewId"], 0) + 1
            xml = root / "xmlz" / (game["reviewId"] + ".xml.zip")
            if args.xml_supplement and (xml.exists() or args.wait_for_xml):
                tasks.append((game, str(xml), cards, args.wait_for_xml))
    for f in files.values():
        f.close()
    if tasks:
        with (out / "library_xml_supplement.jsonl").open("w", encoding="utf-8") as f:
            with ProcessPoolExecutor(max_workers=max(1, min(4, args.workers))) as executor:
                for n, (game, rows, error) in enumerate(executor.map(xml_worker, tasks, chunksize=2), 1):
                    stats["xml_games_ok" if error is None else "xml_errors"] += 1
                    if error:
                        xml_errors.append({"review_id": game, "error": error})
                    if any(row.get("xml_display_names_removed") for row in rows):
                        stats["xml_display_name_sanitized_games"] += 1
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stats["xml_supplement_rows"] += 1
                    if n % 50 == 0:
                        print("XML", n, "/", len(tasks), flush=True)
    report = dict(schema="hsrl.firestone_boards/v1", stats=dict(stats), games_with_primary_boards=len(coverage),
                  xml_errors=xml_errors,
                  dropped=dict(dropped), game_board_counts=coverage,
                  note="主库是PMS真实战斗双方，精确回合；XML补充单独保存；按原始对局分割，无完整隐状态。")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest(root, out, cards)
    print(json.dumps({k: v for k, v in report.items() if k != "game_board_counts"}, ensure_ascii=False), flush=True)


def write_manifest(root, out, cards):
    root, out = Path(root), Path(out)
    checksums = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            with path.open("rb") as stream:
                checksums[path.name] = {"bytes": path.stat().st_size,
                    "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    manifest = dict(schema="hsrl.firestone_board_manifest/v1", build_number=255790,
        fidelity="board_only_approx", files=checksums,
        card_database_sha256=hashlib.sha256(Path(cards).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        source_catalog=json.loads((root / "catalog-summary.json").read_text(encoding="utf-8")),
        split_unit="originalReviewId_or_reviewId", primary_round_field="battleInfo.gameState.currentTurn")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
