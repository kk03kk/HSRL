"""Regression for crowded-hand dispatch and local Choose One continuations."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageEnhance
import pytest

from hsrl_agent.live import branch_ui
from hsrl_agent.live.actions import bind_actions
from hsrl_agent.live.contracts import Binding, Blocked, Decision
from hsrl_agent.live.controller import Controller
from hsrl_agent.live.interactions import branch_snapshot
from hsrl_agent.live.logs import LogReader
from hsrl_agent.live.practical import PracticalGeometry
from hsrl_agent.live.verification import verify


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts/runs"
RUN = RUNS / "live-control-20261121-134752-94400/match-0001"
PREVIOUS = RUNS / "live-control-20261121-122256-31d1d/match-0001"
PROFILE = json.loads((ROOT / "config/live/autoplay-255790.json").read_text(encoding="utf8"))


def decision(binding):
    return Decision("game", 1, "stamp", binding.action, binding, "", "")


def read_artifact(path):
    if not path.is_file():
        pytest.skip("local authentic evidence not present")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def restore(ledger):
    value = ledger["decision"]
    return Decision(**{**value, "binding": Binding(**value["binding"])})


@pytest.mark.parametrize("count", [7, 8, 9, 10])
def test_all_crowded_slots_keep_own_entity_and_insertion(count):
    hand = [dict(entity_id=100 + i, position=i+1, card_type="MINION") for i in range(count)]
    snapshot = dict(view=dict(pending=0), client=dict(hand=hand, board=[], shop=[], choices=[]))
    geometry = PracticalGeometry(PROFILE)
    positions = []
    for i, card in enumerate(hand):
        binding = Binding(46 + i*8, "play", source=card["entity_id"], position=0)
        drag = geometry.plan(decision(binding), snapshot)[0]
        assert drag[0] == "drag"
        assert drag[1] == list(geometry.locations(snapshot)[0]["entity:"+str(card["entity_id"])])
        assert all(0 < v < 1 for v in drag[2])
        positions.append(drag[1])
    assert all(right[0] - left[0] >= .025 for left, right in zip(positions, positions[1:]))
    if count in (9, 10):
        wrong = deepcopy(snapshot)
        wrong["client"]["hand"][0]["position"] = 2
        with pytest.raises(Blocked, match="calibrated slot order"):
            geometry.locations(wrong)


@pytest.mark.parametrize(("run", "slot", "card", "box"), [
    (RUN, 7, "BG36_369", (1362, 1438, 1408, 1495)),
    (PREVIOUS, 9, None, (1470, 1457, 1511, 1510)),
])
def test_grip_sits_within_annotated_exposed_card_art(run, slot, card, box):
    ledger = read_artifact(run / "unresolved-input.json")
    shot = run / "visual-evidence/dispatch/0.jpg"
    if not shot.is_file():
        pytest.skip("dispatch screenshot not present")
    with Image.open(shot) as image:
        assert image.size == (2560, 1600)
        size = image.size
    action = restore(ledger)
    row = ledger["before"]["client"]["hand"][slot-1]
    assert row["entity_id"] == action.binding.source
    if card:
        assert row["card_id"] == card
    plan = PracticalGeometry(PROFILE).plan(action, ledger["before"])
    x, y = [round(v*s) for v, s in zip(plan[0][1], size)]
    left, top, right, bottom = box
    assert left < x < right and top < y < bottom


@pytest.fixture(scope="module")
def previous_match():
    folder = RUN / "session-archives"
    if not folder.is_dir():
        pytest.skip("local raw replay not present")
    raw, = folder.rglob("Power.log")
    branch = None
    with gzip.open(RUN / "decisions-full.jsonl.gz", "rt", encoding="utf8") as stream:
        for row in map(json.loads, stream):
            if row.get("decision_id") == "d-958e04ab399b14d8":
                branch = row["snapshot"]
                break
    assert branch is not None
    held = read_artifact(RUN / "unresolved-input.json")
    terminal_cursor = read_artifact(RUN / "status.json")["client"]["log_position"]
    reader = LogReader(RUN, held["before"]["client"]["local_player_id"])
    reader.state.balanced_async_blocks = True
    markers = {}
    with raw.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not reader.history_complete and "GameState.DebugPrintPower() - CREATE_GAME" not in line:
                continue
            reader.feed(line, "power", 1795240935)
            cursor = reader.state.power_lines
            if cursor == branch["client"]["log_position"]:
                markers["branch_options"] = deepcopy(reader.state.options)
            if cursor == terminal_cursor:
                after = reader.state.observe().to_dict()
                after["game_id"] = held["before"]["game_id"]
                markers["wrong_receipt"] = after
                break
    return branch, held, markers


def test_real_log_rejects_the_wrong_hand_entity(previous_match):
    _, held, markers = previous_match
    original = restore(held)
    after = markers["wrong_receipt"]
    response = next(r for r in after["resolutions"] if r["log_position"] == 126185)
    assert response["source"] == response["submission"]["source"] == 6687
    assert response["submission"]["option_id"] == 7
    assert original.binding.source == 6637 and original.binding.option_id == 6
    assert verify(original, held["before"], {**held["before"], "client":after, "revision":after["revision"]},
                  require_playback=False, allow_options_boundary=True) == "rejected"


def test_real_branch_persists_and_uses_raw_suboptions(previous_match, monkeypatch, tmp_path):
    original, _, markers = previous_match
    data = original["client"]
    request = data["interaction"]
    root = next(row for row in markers["branch_options"] if row.get("entity") == 5179)
    branches = tuple((r["entity"], r["id"], r["error"], tuple(r["targets"]))
                     for r in root["sub_options"])
    assert branches[0][:3] == (5180, 0, "NONE") and len(branches[0][3]) == 12
    binding = Binding(62, "play", source=5179, option_id=root["id"], branches=branches)
    parent = Decision(original["game_id"], original["revision"], request["request_id"].split(":",1)[1],
                      binding.action, binding, "", "")
    snapshot = dict(original, options=markers["branch_options"],
                    client=dict(data, choices=[], interaction=None))
    before = deepcopy(snapshot)
    pending = (parent, before, 0.)
    # The trace preserved the actual measured frame boxes, but did not retain
    # the original RGB frame of this T8 branch. Only raster detection is mocked.
    prior_visual = original["visual"]
    assert prior_visual["branch_ui"]["selectable"] == [False, True]
    monkeypatch.setattr(branch_ui, "regions", lambda image: prior_visual["branch_ui"]["boxes"])
    proof = branch_ui.inspect(Image.new("RGB", (2560, 1600)), PROFILE, snapshot["client"], pending)
    assert proof["selectable"] == [True, True]
    stage = branch_snapshot(snapshot, pending, {
        "branch_open":True, "branch_source_card_id":proof["source_card_id"],
        "branch_entities":proof["entities"],
    })
    selected = bind_actions(stage)[0]
    assert selected.source == 5180
    stage["visual"] = {"branch_ui": proof}
    x, y, width, height = proof["boxes"][0]
    plan = PracticalGeometry(PROFILE).plan(decision(selected), stage)
    assert plan[0] == ("click", [x+width/2, y+height*.56])

    changed = deepcopy(snapshot)
    changed_root = next(row for row in changed["options"] if row.get("entity") == 5179)
    changed_root["sub_options"][0]["error"] = "REQ_TARGET_TO_PLAY"
    with pytest.raises(Blocked, match="current server sub-options"):
        branch_snapshot(changed, pending, {
            "branch_open":True, "branch_source_card_id":proof["source_card_id"],
            "branch_entities":proof["entities"],
        })
    controller = Controller(SimpleNamespace(peer_round=True), None, None, tmp_path,
                            practical=True, clock=lambda: 3.)
    controller.pending = pending
    assert controller.unsent_reconciliation(snapshot) is None
    assert controller.pending is pending
    later = {**snapshot, "client":{**snapshot["client"], "turn":9}}
    assert controller.unsent_reconciliation(later)
    assert controller.pending is None


def test_branch_hover_continuity_on_real_miner_panel():
    shot = RUNS / "live-player150-input-fixes-20261121-02/visual-evidence/dispatch/branch.jpg"
    if not shot.is_file():
        pytest.skip("local Miner branch panel not present")
    with Image.open(shot) as source:
        before = source.convert("RGB")
    boxes = branch_ui.regions(before)
    assert len(boxes)==2
    x, y, w, h = boxes[1]
    hit = (x+w/2, y+h/2)
    dimmed = ImageEnhance.Brightness(before).enhance(.76)
    assert branch_ui.matches_admitted_row(before, dimmed, boxes, hit)
    assert not branch_ui.matches_admitted_row(before, dimmed, boxes, (.50, .16))
    replaced = before.copy()
    left, top, right, bottom = (round(a*b) for a,b in zip((x,y,x+w,y+h),before.size*2))
    replaced.paste((26, 27, 28),(left,top,right,bottom))
    assert not branch_ui.matches_admitted_row(before, replaced, boxes, hit)


def test_real_hover_preview_obscures_only_the_other_branch():
    folder = RUNS / "local-live-fixes-20261121-01"
    evidence = read_artifact(folder / "miner-hover.json")
    boxes = evidence["admitted"]["boxes"]
    x, y, width, height = boxes[1]
    hit = (x+width/2, y+height*.56)
    assert evidence["matched_layout"] and not evidence["hover"]["visible"]
    with Image.open(folder / "miner-admitted.jpg") as before, Image.open(folder / "miner-hover.jpg") as hover:
        assert branch_ui.matches_admitted_row(before, hover, boxes, hit)
        # When the overlaid branch becomes the requested one, it no longer
        # has a complete center proof and must not receive a press.
        x, y, width, height = boxes[0]
        assert not branch_ui.matches_admitted_row(before, hover, boxes, (x+width/2, y+height*.56))
