import copy
import json
from pathlib import Path

import pytest

from tools.opponent_capture.parsers import Catalog, PowerParser, HDTParser
from tools.opponent_capture import store as storage


@pytest.fixture
def catalog():
    return Catalog({"meta": {"client_version": "reference"}, "cards": [
        {"id": "BUSKER", "name": "Southsea Busker", "kind": "minion", "golden": "BUSKER_G"},
        {"id": "BUSKER_G", "name": "Southsea Busker", "kind": "minion", "normal": "BUSKER"}]})


def power(message, channel="GameState.DebugPrintPower"):
    return f"D 12:00:00 {channel}() - {message}\n"


def entity(eid, card, controller, kind, **tags):
    rows = [power(f"FULL_ENTITY - Creating ID={eid} CardID={card}")]
    for k, v in dict(CONTROLLER=controller, CARDTYPE=kind, ZONE="PLAY", **tags).items():
        rows.append(power(f"tag={k} value={v}"))
    return "".join(rows)


def fight(mode="GT_BATTLEGROUNDS", end=True):
    rows = power("CREATE_GAME") + power("GameEntity EntityID=1")
    rows += power("tag=TURN value=4")
    rows += f"D 0 GameState.DebugPrintGame() - GameType={mode}\n"
    rows += "D 0 GameState.DebugPrintGame() - BuildNumber=255130\n"
    rows += "D 0 GameState.DebugPrintGame() - PlayerID=1\n"
    rows += entity(10, "HERO_A", 1, "HERO", HEALTH=30, ARMOR=8)
    rows += entity(11, "BUSKER", 1, "MINION", ZONE_POSITION=1, ATK=3, HEALTH=1)
    rows += power("TAG_CHANGE Entity=GameEntity tag=BACON_IN_COMBAT_PHASE value=1")
    rows += entity(20, "HERO_B", 9, "HERO", HEALTH=25, ARMOR=3)
    rows += entity(21, "BUSKER", 9, "MINION", ZONE_POSITION=1, ATK=5, HEALTH=6)
    rows += power("BLOCK_START BlockType=ATTACK Entity=11 Target=21")
    rows += power("TAG_CHANGE Entity=21 tag=HEALTH value=1")
    if end:
        rows += power("TAG_CHANGE Entity=GameEntity tag=BACON_IN_COMBAT_PHASE value=0")
    return rows


def parse(parser, text):
    return [r for line in text.splitlines() for r in parser.feed(line)]


def test_power_strict_bg_and_single_channel_no_constructed_contamination():
    assert not parse(PowerParser("source"), fight("GT_RANKED"))
    parser = PowerParser("source")
    text = fight()
    # A second channel logging the exact same messages must not create extra combats.
    out = parse(parser, text + text.replace("GameState.DebugPrintPower", "PowerTaskList.DebugPrintPower")
                .replace("GameState.DebugPrintGame", "Other.DebugPrintGame"))
    assert len({r["encounter_id"] for r in out}) == 1
    row = out[-1]
    assert row["capture_complete"] and row["turn"] == 2 and row["turn_raw"] == 4
    assert row["opponent"]["board"][0]["health"] == 6  # pre-attack, not the later damaged state
    assert row["opponent"]["hero_id"] == "HERO_B"
    assert row["boundary"] == "post_start_before_first_attack"
    assert row["actual_result"] is None and not row["training_import_allowed"]


def test_power_resume_preserves_boundary_and_does_not_invent_hidden_hand():
    parser = PowerParser("source")
    out = parse(parser, fight(end=False))
    resumed = PowerParser("source", json.loads(json.dumps(parser.s)))
    row = parse(resumed, power("TAG_CHANGE Entity=GameEntity tag=BACON_IN_COMBAT_PHASE value=0"))[-1]
    assert row["encounter_id"] == out[0]["encounter_id"]
    assert row["opponent"]["hand"] == []
    assert row["opponent"]["hand_scope"] != 'complete'
    assert "opponent.unrevealed_hand_or_counters" in row["missing_fields"]
    assert len(row["events"]) >= 1


def test_unknown_side_not_inferred_from_unrelated_lobby_boards():
    text = fight(end=False).replace(power("BLOCK_START BlockType=ATTACK Entity=11 Target=21"),
        entity(30, "THIRD_HERO", 3, "HERO", HEALTH=30) + power("BLOCK_START BlockType=ATTACK Entity=11 Target=21"))
    row = parse(PowerParser("s"), text)[0]
    assert row["opponent_controller"] is None
    assert not row["opponent_board_recorded"]


def test_hidden_shop_cards_not_mistaken_for_opponent_and_show_entity_numeric_reference():
    text = fight(end=False)
    attack = power("BLOCK_START BlockType=ATTACK Entity=11 Target=21")
    before = entity(25, "STALE_SHOP_OFFER", 9, "MINION", ZONE_POSITION=2, ATK=99, HEALTH=99)
    before += power("HIDE_ENTITY - Entity=25 tag=ZONE value=SETASIDE")
    before += power("SHOW_ENTITY - Updating Entity=21 CardID=REVEALED_OPPONENT")
    row = parse(PowerParser("s"), text.replace(attack, before + attack))[0]
    assert [c['card_id'] for c in row['opponent']['board']] == ['REVEALED_OPPONENT']


def test_debug_game_lists_both_players_local_identity_from_shop_controls():
    text = fight().replace("D 0 GameState.DebugPrintGame() - PlayerID=1\n", "")
    marker = power("BLOCK_START BlockType=ATTACK Entity=11 Target=21")
    meta = "D 0 GameState.DebugPrintGame() - PlayerID=1, PlayerName=PrivateA#1234\n"
    meta += "D 0 GameState.DebugPrintGame() - PlayerID=9, PlayerName=PrivateB#5678\n"
    meta += entity(99, "TB_BaconShopLockAll_Button", 1, "SPELL")
    row = parse(PowerParser("s"), text.replace(marker, meta + marker))[0]
    assert row['local_controller'] == 1 and row['opponent_controller'] == 9
    assert row['local_controller_source'] == 'unique_shop_control_owner'


def test_historical_reference_cannot_resurrect_removed_shop_offer():
    attack = power("BLOCK_START BlockType=ATTACK Entity=11 Target=21")
    prior = entity(25, "STALE_SHOP_OFFER", 9, "MINION", ZONE_POSITION=2, ATK=99, HEALTH=99)
    prior += power("TAG_CHANGE Entity=25 tag=ZONE value=REMOVEDFROMGAME")
    prior += power("TAG_CHANGE Entity=[entityName=Private id=25 zone=PLAY zonePos=2 cardId=STALE_SHOP_OFFER player=9] tag=ATK value=99")
    row = parse(PowerParser('s'), fight().replace(attack, prior + attack))[0]
    assert [c['entity_id'] for c in row['opponent']['board']] == [21]


def test_observed_attachment_and_revealed_minion_hand_preserved_only_as_context():
    attack = power("BLOCK_START BlockType=ATTACK Entity=11 Target=21")
    prior = entity(40, "ENCHANTMENT", 9, "ENCHANTMENT", ATTACHED=21, TAG_SCRIPT_DATA_NUM_1=3)
    prior += power('FULL_ENTITY - Creating ID=41 CardID=HAND_MINION')
    prior += power('tag=CONTROLLER value=1') + power('tag=CARDTYPE value=MINION')
    prior += power('tag=ZONE value=HAND') + power('tag=ATK value=15') + power('tag=HEALTH value=17')
    row = parse(PowerParser('s'), fight().replace(attack, prior + attack))[0]
    assert row['opponent']['board'][0]['enchantments'][0]['observed_tags']['TAG_SCRIPT_DATA_NUM_1'] == '3'
    assert row['player']['hand'][0]['attack'] == 15
    assert not row['policy_input']


def hdt_input(team="null", name="Southsea Busker", close=True):
    rows = ["--- Player turn 3 ---", "----- Simulation Input -----",
            "Player: heroPower=HERO_POWER_A, used=False, data=0", "Hand:", "[Southsea Busker 3/1]", "---",
            "Opponent: heroPower=HERO_POWER_B, used=True, data=2", "Hand: SpellCardEntity",
            f"[{name} 12/8, G, Div]", "PlayerTeammate: " + team, "OpponentTeammate: null"]
    if close:
        rows.append("----- End of Input -----")
    return "\n".join("00:00|Info|BobsBuddyInvoker.RunSimulation >> " + x for x in rows) + "\n"


def test_hdt_gold_mapping_outcomes_bound_to_correct_input_and_team_not_solo(catalog):
    parser = HDTParser("s", catalog)
    text = hdt_input(team="heroPower=TEAM")
    text += "X >> WinRate=20% (Lethal=0%), TieRate=30%, LossRate=50%\n"
    text += "ValidateSimulationResultAsync >> result=Win\n"
    text += hdt_input()
    out = parse(parser, text)
    first = next(r for r in reversed(out) if r["combat_index"] == 1)
    second = out[-1]
    assert first["opponent"]["board"][0]["card_id"] == "BUSKER_G"
    assert first["actual_result"]["outcome"] == "win"
    assert 'actual_result' not in first['missing_fields']
    assert first["second_implementation"]["official"] is False
    assert "unsupported_team_context" in first["missing_fields"]
    assert second["actual_result"] is None  # never reuse a previous combat's result
    assert first["root"] == second["root"]  # conservative source-session split grouping
    assert "opponent.unparsed_hand" in second["missing_fields"]
    assert not first["training_import_allowed"]


def test_hdt_unknown_card_and_missing_input_end_are_not_silently_accepted(catalog):
    parser = HDTParser("s", catalog)
    out = parse(parser, hdt_input(name="Unknown Future Card", close=False) + hdt_input())
    first = out[0]
    assert first["opponent"]["board"][0]["card_id"] is None
    assert "incomplete_simulation_input" in first["missing_fields"]
    assert first["capture_complete"] is False


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    result = storage.Store(tmp_path / "runs/capture")
    yield result
    result.close()


def test_incremental_utf8_resume_idempotence_and_transactional_offset(store, tmp_path, catalog):
    path = tmp_path / "Power.log"
    text = fight(end=False).encode()
    partial = power("TAG_CHANGE Entity=GameEntity tag=BACON_IN_COMBAT_PHASE value=0").encode()
    path.write_bytes(text + partial[:-3])
    store.ingest(path, "power", catalog)
    assert store.summary()["encounters"] == 1
    assert store.summary()["completed_boundaries"] == 0
    assert store.ingest(path, "power", catalog)["bytes"] == 0
    with path.open("ab") as f:
        f.write(partial[-3:])
    store.ingest(path, "power", catalog)
    assert store.summary()["completed_boundaries"] == 1
    assert store.ingest(path, "power", catalog)["changed"] == 0
    store.export(); first = (store.directory / "opponents.jsonl").read_bytes()
    store.export(); assert first == (store.directory / "opponents.jsonl").read_bytes()
    assert "player" not in json.loads(first)


def test_replaced_log_preserves_old_encounters_and_new_generation(store, tmp_path, catalog):
    path = tmp_path / "Power.log"; path.write_text(fight())
    store.ingest(path, "power", catalog)
    path.write_text(fight().replace("ATK value=5", "ATK value=7"))
    store.ingest(path, "power", catalog)
    assert store.summary()["encounters"] == 2
    assert store.summary()["diagnostics"][0]["reason"] == "log_replaced_or_truncated"


def test_no_account_or_raw_entity_names_in_persisted_state(store, tmp_path, catalog):
    path = tmp_path / "Power.log"
    private = "PrivateBattleTag#1234"
    text = fight().replace("Entity=21 tag=HEALTH", f"Entity=[entityName={private} id=21 zone=PLAY zonePos=1 cardId=BUSKER player=9] tag=HEALTH")
    text += power("Player EntityID=5 PlayerID=3 GameAccountId=[hi=123 lo=999]")
    text += power("tag=ACCOUNT_ID value=9999888777")
    text += power(f"TAG_CHANGE Entity={private} tag=ATK value=999")
    path.write_text(text)
    store.ingest(path, "power", catalog); store.export()
    assert private not in (store.directory / "encounters.jsonl").read_text()
    states = store.db.execute("SELECT state FROM sources").fetchall()
    assert private not in json.dumps(states) and "GameAccountId" not in json.dumps(states)
    assert '9999888777' not in json.dumps(states)


def test_output_cannot_write_into_old_project(tmp_path):
    with pytest.raises(ValueError, match="main project"):
        storage.output_directory(tmp_path / "not-project-runs")


def test_single_writer_lock_prevents_duplicate_collectors(store):
    with storage.single_writer(store.directory):
        with pytest.raises(RuntimeError, match='already running'):
            with storage.single_writer(store.directory):
                pass


def test_failed_chunk_rolls_back_emitted_rows_and_source_cursor(store, tmp_path, catalog, monkeypatch):
    path = tmp_path / 'Power.log'; path.write_text(fight())
    original = store.put
    def broken(row, catalog=None):
        original(row, catalog)
        raise RuntimeError('simulated interruption before source checkpoint')
    with monkeypatch.context() as patch:
        patch.setattr(store, 'put', broken)
        with pytest.raises(RuntimeError, match='simulated'):
            store.ingest(path, 'power', catalog)
    assert store.summary()['encounters'] == 0
    assert store.db.execute('SELECT count(*) FROM sources').fetchone()[0] == 0
    store.ingest(path, 'power', catalog)
    assert store.summary()['encounters'] == 1


def test_training_export_never_auto_splits_cross_source_duplicates(store, tmp_path, catalog):
    path = tmp_path / 'Power.log'; path.write_text(fight())
    store.ingest(path, 'power', catalog); store.export()
    row=json.loads((store.directory/'opponents.jsonl').read_text())
    assert row['split']=='unassigned' and not row['training_import_allowed']
    assert 'cross_source' in row['split_blocker']


def test_team_context_stays_in_quarantine_not_solo_opponent_export(store, tmp_path, catalog):
    path = tmp_path / 'hdt_log.txt'; path.write_text(hdt_input(team='heroPower=TEAM'))
    store.ingest(path, 'hdt', catalog); store.export()
    assert (store.directory/'opponents.jsonl').read_text() == ''
    row=json.loads((store.directory/'quarantine.jsonl').read_text())
    assert row['mode']=='team_context_not_solo' and not row['solo_training_candidate']


def test_golden_card_id_relationship_survives_missing_premium_tag(store, tmp_path, catalog):
    path=tmp_path/'Power.log'; path.write_text(fight().replace('ID=21 CardID=BUSKER','ID=21 CardID=BUSKER_G'))
    store.ingest(path,'power',catalog);store.export()
    row=json.loads((store.directory/'opponents.jsonl').read_text())
    assert row['opponent']['board'][0]['golden'] is True
    assert row['opponent']['board'][0]['attack']==5
