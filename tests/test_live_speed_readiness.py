"""Fast polling must never grant stale input authority."""
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'client'))

from hsrl_agent.live.logs import LogReader
from hsrl_agent.live.actions import bind_actions
from hsrl_agent.live.timing import RAPID
from hsrl_agent.live.windows import NativeBackend


def test_stationary_log_reuses_observation_but_partial_read_revokes_authority(tmp_path):
    session = tmp_path / 'Logs' / 'Hearthstone_test'
    session.mkdir(parents=True)
    power = session / 'Power.log'
    power.write_text('D 15:00:00.000000 GameState.DebugPrintPower() - CREATE_GAME\n', encoding='utf8')
    network = session / 'GameNetLogger.log'
    network.write_text('GameMgr.OnGameSetup()\n', encoding='utf8')
    reader = LogReader(tmp_path)
    reader.start_latest = True
    first = reader.poll()
    assert reader.live_tail_current()
    assert reader.poll() is first

    with network.open('a', encoding='utf8') as stream:
        stream.write('Network.OnGameServerDiscon')
    assert reader.poll() is first
    assert not reader.caught_up and not reader.live_tail_current()

    with network.open('a', encoding='utf8') as stream:
        stream.write('nectEvent()\n')
    assert reader.poll() is not first
    assert reader.caught_up and reader.connection == 'disconnected'
    assert not reader.live_tail_current()


def test_pointer_parks_during_receipt_wait_only_after_safe_dwell():
    now = [0.]
    movements = []
    window = SimpleNamespace(hwnd=1, rect=(0, 0, 2560, 1600), token=lambda: 'old')
    backend = NativeBackend.__new__(NativeBackend)
    backend.u = SimpleNamespace(GetAsyncKeyState=lambda _: 0, WindowFromPoint=lambda _: 1,
                                GetAncestor=lambda hwnd, _: hwnd)
    backend.timing, backend.held = RAPID, False
    backend.pointer_ready_at, backend.deferred_park = 0., None
    backend.clock = lambda: now[0]
    backend.sleep = lambda delay: now.__setitem__(0, now[0] + delay)
    backend.window = lambda: window
    backend.stopped = lambda: False
    backend._event = lambda flags, point=None, token=None, **kw: movements.append((flags, point, now[0]))
    backend.execute([('click', [.5, .5]), ('move_pointer', [.96, .91])],
                    {'visual': {'window_token': 'old'}})
    released_at = now[0]
    assert movements[-1][0] == 4
    backend.service_pointer()
    assert movements[-1][0] == 4
    now[0] += RAPID.release
    backend.service_pointer()
    assert movements[-1] == (0, (2457, 1455), released_at + RAPID.release)
    last = list(movements)
    backend.service_pointer()
    assert movements == last


def test_untargeted_secondary_power_does_not_replace_witnessed_choose_one_play():
    card = dict(entity_id=2419, card_id='BG30_123', card_type='MINION', tags={'CHOOSE_ONE': '1'})
    branch = dict(id=1, entity=2419, type='POWER', error='NONE', targets=[], has_targets=False,
                  sub_options=[dict(id=0, entity=2420, error='NONE', targets=[]),
                               dict(id=1, entity=2421, error='NONE', targets=[])])
    plain = dict(id=2, entity=2419, type='POWER', error='NONE', targets=[], has_targets=False)
    board = [dict(entity_id=11, card_id='TEST', card_type='MINION')]
    snapshot = dict(client=dict(hand=[card], board=board), view={}, action_profile='player150')
    for options in ([branch, plain], [plain, branch]):
        bindings = bind_actions(dict(snapshot, options=options))
        assert [b.action for b in bindings] == [0, 46, 47]
        assert [(b.option_id, b.branches) for b in bindings[1:]] == [
            (1, ((2420, 0, 'NONE', ()), (2421, 1, 'NONE', ())))] * 2

    # Without the witnessed card tag, only the ambiguous slots are masked.
    card['tags'].clear()
    assert [b.action for b in bind_actions(dict(snapshot, options=[branch, plain]))] == [0]


def test_only_a_certified_unsent_matching_gesture_can_degrade_rapid_timing():
    backend = NativeBackend.__new__(NativeBackend)
    backend.timing = RAPID
    backend.degraded_gestures = set()
    backend.last_gesture = ('drag', ('session', 6, 42), RAPID.name)
    snapshot = {'game_id': 'session', 'client': {'turn': 6, 'log_position': 43}}
    assert backend.note_unsent(snapshot) is None
    assert backend.degraded_gestures == set()
    snapshot['client']['log_position'] = 42
    assert backend.note_unsent(snapshot) == 'drag'
    assert backend.degraded_gestures == {'drag'}
