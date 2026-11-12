"""Offline-dialog recovery: the failure that cost the 2026-11-11 overnight run.

The OCR rows below are the real capture from
``artifacts/runs/live-e-20261111-108/match-0007/startup/blocked-last.json``
(06:14, client build 255790). Nine batches x seven attempts all died on this one
dialog because ``重新连接`` was classified as "needs a human", so nothing was ever
clicked (``queue-report.json``: elapsed 8.8 s, actions 0, inputs_sent 0).
"""
import pytest

from hsrl_agent.live.contracts import Blocked
from hsrl_agent.live.lobby import LobbyFlow, reconnect_dialog

#: Verbatim rows from the archived failure (noise rows included on purpose).
OFFLINE_ROWS = [
    dict(text='重新连接', confidence=0.99876, point=[0.5, 0.230078125]),
    dict(text='当前处于离线状态', confidence=0.99908, point=[0.5009765625, 0.53984375]),
    dict(text='距离你的上一次操作已经过了很', confidence=0.9996, point=[0.49951171875, 0.6078125]),
    dict(text='长时间，游戏连接中断。', confidence=0.99976, point=[0.494140625, 0.65390625]),
    dict(text='重新连接', confidence=0.9964, point=[0.4072265625, 0.75625]),
    dict(text='取消', confidence=0.99986, point=[0.58251953125, 0.75625]),
    dict(text='06:14', confidence=0.99985, point=[0.07275390625, 0.9765625]),
    dict(text='8', confidence=0.99995, point=[0.7880859375, 0.9765625]),
    dict(text='340', confidence=0.99996, point=[0.87744140625, 0.97578125]),
    dict(text='x', confidence=0.73424, point=[0.94482421875, 0.978125]),
]


def rows_with(**override):
    """Offline dialog with one row replaced by ``{'text':..,'x':..,'y':..}``."""
    out = []
    for row in OFFLINE_ROWS:
        out.append(dict(row))
    for text, x, y, conf in override.get('replace', ()):
        out = [r for r in out if not (r['text'] == text)]
        out.append(dict(text=text, confidence=conf, point=[x, y]))
    return out


def test_offline_dialog_anchors_the_button_not_the_title():
    """Title and button share the text 重新连接; only the lower button may be clicked."""
    button = reconnect_dialog(OFFLINE_ROWS)
    assert button is not None
    assert tuple(button['point']) == (0.4072265625, 0.75625)
    # The title must never be returned: clicking it would send input to the board.
    assert button['point'][1] > 0.6


def test_offline_dialog_requires_body_text_not_just_the_word():
    """A stray 重新连接 hit (queue UI, tooltip) must not authorise a modal click."""
    rows = [r for r in OFFLINE_ROWS if '离线' not in r['text'] and '中断' not in r['text']]
    assert reconnect_dialog(rows) is None


def test_offline_dialog_refuses_when_cancel_is_missing():
    """Without a positive 取消 witness the layout is unverified; never click blind."""
    rows = [r for r in OFFLINE_ROWS if r['text'] != '取消']
    assert reconnect_dialog(rows) is None


def test_offline_dialog_refuses_when_button_is_right_of_cancel():
    """The left-of-取消 rule is what makes pressing 取消 (which closes the client) impossible."""
    rows = rows_with(replace=(('重新连接', 0.70, 0.75625, 0.996),))
    # Now two 重新连接 rows exist at y>0.6? No: the title stays at y=0.23.
    assert reconnect_dialog(rows) is None


def test_offline_dialog_refuses_when_buttons_are_not_on_one_row():
    rows = rows_with(replace=(('取消', 0.58251953125, 0.60, 0.999),))
    assert reconnect_dialog(rows) is None


def test_offline_dialog_refuses_low_confidence_button():
    rows = rows_with(replace=(('重新连接', 0.4072265625, 0.75625, 0.55),))
    assert reconnect_dialog(rows) is None


def test_flow_emits_reconnect_after_two_frames():
    flow = LobbyFlow(max_restarts=3)
    assert flow.inspect(OFFLINE_ROWS, 'token-1', 100.0) is None      # first frame: proof only
    action = flow.inspect(OFFLINE_ROWS, 'token-1', 100.4)
    assert action is not None and action.kind == 'reconnect'
    assert action.point == (0.4072265625, 0.75625)


def test_commit_reconnect_counts_and_enters_pending():
    flow = LobbyFlow(max_restarts=3)
    flow.inspect(OFFLINE_ROWS, 't', 100.0)
    action = flow.inspect(OFFLINE_ROWS, 't', 100.4)
    flow.commit(action, 101.0)
    assert flow.state == 'reconnect_pending'
    assert flow.reconnects == 1
    assert flow.needs_restart is True


def test_reconnect_is_spaced_by_backoff():
    """A failing reconnect must not be re-clicked every frame."""
    flow = LobbyFlow(max_restarts=3)
    flow.inspect(OFFLINE_ROWS, 't', 100.0)
    flow.commit(flow.inspect(OFFLINE_ROWS, 't', 100.4), 101.0)
    # Still offline 5 s later: inside the 10 s backoff, so no new click.
    assert flow.inspect(OFFLINE_ROWS, 't', 106.0) is None
    assert flow.inspect(OFFLINE_ROWS, 't', 106.4) is None
    # Past the backoff the dialog may be clicked again (needs a fresh two-frame proof).
    assert flow.inspect(OFFLINE_ROWS, 't', 112.0) is None
    assert flow.inspect(OFFLINE_ROWS, 't', 112.4) is not None


def test_reconnect_attempts_are_bounded():
    flow = LobbyFlow(max_restarts=3)
    flow.reconnects = 3
    with pytest.raises(Blocked, match='reconnect recovery limit'):
        flow.inspect(OFFLINE_ROWS, 't', 100.0)


def test_dialog_clearing_resets_to_lobby_and_requests_restart():
    flow = LobbyFlow(max_restarts=3)
    flow.inspect(OFFLINE_ROWS, 't', 100.0)
    flow.commit(flow.inspect(OFFLINE_ROWS, 't', 100.4), 101.0)
    lobby_rows = [dict(text='酒馆战棋', confidence=0.99, point=[0.37, 0.02])]
    assert flow.inspect(lobby_rows, 't2', 130.0) is None
    assert flow.state == 'idle' and flow.needs_restart is True


@pytest.mark.parametrize('text', ['验证码', '登录'])
def test_credential_dialogs_still_require_a_human(text):
    """A real credential prompt must never be auto-dismissed."""
    rows = [
        dict(text='重新连接', confidence=0.998, point=[0.5, 0.23]),
        dict(text=text, confidence=0.998, point=[0.5, 0.54]),
        dict(text='重新连接', confidence=0.996, point=[0.407, 0.756]),
        dict(text='取消', confidence=0.999, point=[0.582, 0.756]),
    ]
    flow = LobbyFlow(max_restarts=3)
    with pytest.raises(Blocked, match='login/connection intervention'):
        flow.inspect(rows, 't', 100.0)


# ── quest overlay (observed 2026-11-12 00:23 and 00:25) ────────────────────────

QUEST_ROWS = [
    dict(text='你的每周任务', confidence=0.999, point=[0.5, 0.2]),
    dict(text='使用30张畸变怪或野兽牌。', confidence=0.99, point=[0.29, 0.53]),
    dict(text='15 / 30', confidence=0.99, point=[0.29, 0.58]),
    dict(text='等级', confidence=0.99, point=[0.77, 0.16]),
    dict(text='3152', confidence=0.99, point=[0.77, 0.2]),
]

LOBBY_READY = [
    dict(text='酒馆战棋', confidence=0.99, point=[0.5, 0.02]),
    dict(text='开始', confidence=0.99, point=[0.78, 0.79]),
]


def test_quest_overlay_is_dismissed_at_a_fixed_empty_point():
    """The overlay has no dismiss control, so the popup itself is the authorisation."""
    flow = LobbyFlow(max_restarts=3)
    assert flow.inspect(QUEST_ROWS, 't', 100.0) is None
    action = flow.inspect(QUEST_ROWS, 't', 100.4)
    assert action is not None and action.kind == 'dismiss_quests'
    assert action.point == (0.12, 0.88)


def test_quest_overlay_dismissal_is_bounded():
    flow = LobbyFlow(max_restarts=3)
    flow.quest_dismissals = 3
    with pytest.raises(Blocked, match='quest overlay dismissal limit'):
        flow.inspect(QUEST_ROWS, 't', 100.0)


def test_quest_overlay_does_not_steal_a_ready_lobby():
    """A normal lobby with 开始 must start a match, never click the empty corner."""
    flow = LobbyFlow(max_restarts=3)
    assert flow.inspect(LOBBY_READY, 't', 100.0) is None
    action = flow.inspect(LOBBY_READY, 't', 100.4)
    assert action is not None and action.kind == 'start'


def test_quest_overlay_is_not_dismissed_with_unresolved_gameplay():
    flow = LobbyFlow(max_restarts=3)
    with pytest.raises(Blocked, match='unresolved gameplay input'):
        flow.inspect(QUEST_ROWS, 't', 100.0, pending_gameplay=True)
