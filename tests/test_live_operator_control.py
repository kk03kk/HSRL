from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from hsrl_agent.live.contracts import Blocked, InputNotSent
from hsrl_agent.live.operator_control import OperatorControl, reset_admission
from hsrl_agent.live.runtime import load, save


class Clock:
    now = 100.

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def write_command(folder, clock, command, sequence):
    save(folder / 'command.json', dict(command=command, sequence=sequence, at=clock()))


def test_pause_ack_requires_an_explicit_fresh_resume(tmp_path):
    clock = Clock()
    control = OperatorControl(tmp_path, clock=clock, wall_clock=clock)
    assert control.poll() == 'stop'
    write_command(tmp_path, clock, 'run', 1)
    control.require_new_input()
    write_command(tmp_path, clock, 'pause', 2)
    with pytest.raises(Blocked, match='no input sent'):
        control.require_new_input()
    control.publish('pausing', pending=True, turn=6)
    assert load(tmp_path / 'control-status.json')['pending']
    control.publish('paused', pending=False, turn=6)
    assert load(tmp_path / 'control-status.json')['sequence'] == 2
    write_command(tmp_path, clock, 'run', 1)
    assert control.poll(force=True) == 'pause'
    write_command(tmp_path, clock, 'run', 2)
    assert control.poll(force=True) == 'pause'
    write_command(tmp_path, clock, 'run', 3)
    control.require_new_input()


def test_missing_gui_heartbeat_fails_closed_and_cannot_restart(tmp_path):
    clock = Clock()
    control = OperatorControl(tmp_path, clock=clock, wall_clock=clock)
    write_command(tmp_path, clock, 'run', 1)
    control.require_new_input()
    clock.sleep(11)
    assert control.poll() == 'run'
    clock.sleep(1.1)
    assert control.poll() == 'stop'
    write_command(tmp_path, clock, 'run', 2)
    assert control.poll(force=True) == 'stop'


def test_resume_discards_proof_without_mutating_pending_transaction():
    ledger = object()
    controller = SimpleNamespace(candidate=object(), pending=ledger, ended=('game', 6))
    pipeline = SimpleNamespace(admission=object(),
        gate=SimpleNamespace(previous=object(), admitted=object(), admitted_binding=object()))
    reset_admission(controller, pipeline)
    assert controller.candidate is None
    assert controller.pending is ledger
    assert controller.ended == ('game', 6)
    assert pipeline.admission is None
    assert all(value is None for value in vars(pipeline.gate).values())


def fake_backend():
    from hsrl_agent.live import windows
    backend = object.__new__(windows.NativeBackend)
    clock = Clock()
    backend.clock, backend.sleep = clock, clock.sleep
    backend.window = lambda: SimpleNamespace(hwnd=42, rect=(0, 0, 200, 200))
    backend._check = lambda token: None
    backend.stopped = lambda: False
    backend._neutral = lambda *args: None
    backend.deferred_park, backend.held = None, False
    backend.u = SimpleNamespace(GetAsyncKeyState=lambda key: 0,
                                GetAncestor=lambda handle, flags: handle, WindowFromPoint=lambda point: 42)
    backend.timing = SimpleNamespace(name='test', hover=.04, hold=.04, release=.1,
                                     double_gap=.05, overlap_release=False)
    calls = []
    backend._event = lambda flag, *args, **kwargs: calls.append(flag)
    observation = {'visual': {'window_token': object()}}
    return backend, calls, observation


def test_native_pause_checks_only_the_first_press_of_a_transaction():
    backend, calls, observation = fake_backend()

    def check():
        if 2 in calls:
            raise AssertionError('operator control checked again after mouse down')
        calls.append('boundary')

    backend.before_first_down = check
    backend.execute([('double_click', [.4, .4])], observation)
    assert calls.count('boundary') == 1
    assert [flag for flag in calls if flag in (2, 4)] == [2, 4, 2, 4]
    calls.clear()

    def paused():
        raise Blocked('pause')

    backend.before_first_down = paused
    with pytest.raises(InputNotSent, match='pause'):
        backend.execute([('click', [.4, .4])], observation)
    assert 2 not in calls and 4 not in calls


def test_hdt_occlusion_skips_only_pointer_parking():
    backend, calls, observation = fake_backend()
    backend.timing.overlap_release = True
    backend.pointer_ready_at = 0.
    backend.optional_pointer_park = True
    backend.u.WindowFromPoint = lambda point: 7 if point.x > 175 else 42
    backend.execute([('click', [.4, .4]), ('move_pointer', [.96, .91])], observation)
    assert backend.last_input_trace['pointer_park_skipped']
    assert backend.deferred_park is None
    assert [flag for flag in calls if flag in (2, 4)] == [2, 4]
    calls.clear()
    with pytest.raises(InputNotSent, match='action point occluded'):
        backend.execute([('click', [.96, .91])], observation)
    assert calls == []


@pytest.mark.parametrize(('games', 'stop_after', 'expected'), [(1, None, 1), (2, None, 2), (0, 3, 3)])
def test_desktop_batch_counts_terminal_results_and_stops_at_target(tmp_path, monkeypatch, games, stop_after, expected):
    from hsrl_agent.live import practical, startup
    output = tmp_path / 'artifacts/runs/live-control-test'
    output.mkdir(parents=True)
    write_command(output, time.time, 'run', 1)
    config = tmp_path / 'profile.json'
    save(config, {})
    opts = SimpleNamespace(games=games, desktop_control=True, out=output, shadow=False,
                           no_queue=False, profile=config)
    attempts = []
    returned = []

    def fake_match(root, child):
        attempts.append(child.out)
        child.preloaded_backend = object()
        save(Path(child.out) / 'result.json', dict(terminal=True))
        if len(attempts) == stop_after:
            write_command(output, time.time, 'stop', 2)
            child.operator_control.poll(force=True)
        return 0

    monkeypatch.setattr(practical, '_one_game', fake_match)
    monkeypatch.setattr(startup, 'wait_for_match', lambda *args, **kwargs: returned.append(kwargs['return_lobby']))
    assert practical._autoplay_locked(tmp_path, opts) == 0
    assert len(attempts) == expected
    assert len(load(output / 'matches.json')) == expected
    assert load(output / 'control-status.json')['completed'] == expected
    assert returned == ([] if stop_after else [True])


def test_desktop_batch_stops_before_queueing_after_a_match_error(tmp_path, monkeypatch):
    from hsrl_agent.live import practical
    output = tmp_path / 'artifacts/runs/live-control-test'
    output.mkdir(parents=True)
    write_command(output, time.time, 'run', 1)
    opts = SimpleNamespace(games=0, desktop_control=True, out=output, shadow=False,
                           no_queue=False)
    attempts = []

    def fake_match(root, child):
        attempts.append(child.out)
        raise Blocked('input interrupted; delivery unknown')

    monkeypatch.setattr(practical, '_one_game', fake_match)
    assert practical._autoplay_locked(tmp_path, opts) == 2
    assert len(attempts) == 1
    assert load(output / 'control-status.json')['state'] == 'error'
    assert load(output / 'control-status.json')['completed'] == 0


def test_final_lobby_proof_never_clicks_start(tmp_path):
    from hsrl_agent.live.startup import wait_for_match
    from PIL import Image
    image = Image.new('RGB', (200, 200))
    window = SimpleNamespace(dpi=144, token=lambda: ('stable',))
    backend = SimpleNamespace(stopped=lambda: False, capture=lambda: (image, window))
    options = SimpleNamespace(game_dir=tmp_path, player_id=None, queue_timeout=10, profile=tmp_path / 'profile.json')
    context = lambda: dict(active=False, rebuilding=False, completed=True)
    clock = Clock()
    from unittest.mock import patch
    rows = [dict(text='酒馆战棋', point=(.5, .04), confidence=.99),
            dict(text='开始', point=(.78, .79), confidence=.99)]
    with patch('hsrl_agent.live.startup.lobby_ocr_rows', return_value=rows):
        wait_for_match(tmp_path, options, backend, dict(client_size=[200, 200], dpi=144),
                       tmp_path / 'run', clock=clock, sleep=clock.sleep, context=context,
                       ocr=object(), return_lobby=True)
    assert load(tmp_path / 'run/startup/queue-report.json') == dict(status='lobby_ready', inputs_sent=0)
