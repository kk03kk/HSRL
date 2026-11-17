import gzip
import json

import pytest

from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
from hsbg.az.train.policies import Policy, Decision
from hsbg.evaluation.trace import DecisionRecorder, validate_trace


class EndTurnPolicy(Policy):
    def decide(self, game, seat):
        mask = bytes(game.legal_mask(seat))
        action = 0 if mask[0] else next(i for i, valid in enumerate(mask) if valid)
        return Decision(action)


def test_recording_is_observational_and_every_step_replays(tmp_path):
    pytest.importorskip('hsbg_engine')
    config = EvalConfig(games=2, seed=637201, collect_economy=True, collect_action_hist=True)
    plain = evaluate_policy(EndTurnPolicy(), config)
    path = tmp_path/'decisions.jsonl.gz'
    recorder = DecisionRecorder(path, identity={'test': True})
    config.decision_recorder = recorder
    recorded = evaluate_policy(EndTurnPolicy(), config)
    manifest = recorder.finish(recorded)
    for key in ('rank_sequence', 'seed_sequence', 'decisions', 'action_hist', 'end_turn_economy'):
        assert recorded[key] == plain[key], key
    assert manifest['actions'] == plain['decisions']
    assert validate_trace(path, replay=10000)['replayed'] == plain['decisions']
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    assert [row['actions'] for row in rows if row['event'] == 'game_end']


def test_partial_trace_is_never_published_as_complete(tmp_path):
    path = tmp_path/'decisions.jsonl.gz'
    recorder = DecisionRecorder(path, identity={})
    recorder.abort()
    assert not path.exists()
    assert recorder.partial.exists()
    assert not recorder.manifest_path.exists()
