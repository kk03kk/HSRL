import gzip
import json

import pytest

from hsbg.az.train.evaluate import EvalConfig, evaluate_policy
from hsbg.evaluation.trace import DecisionRecorder, validate_trace
from hsbg.runtime.artifacts import file_identity
from tests.test_decision_recording import EndTurnPolicy


@pytest.mark.parametrize('mutation', ['footer_count', 'rank', 'trailing_event', 'game_index', 'manifest_incomplete'])
def test_trace_rejects_semantically_corrupt_evidence_even_with_updated_hash(tmp_path, mutation):
    pytest.importorskip('hsbg_engine')
    path = tmp_path/'decisions.jsonl.gz'
    recorder = DecisionRecorder(path, identity={})
    result = evaluate_policy(EndTurnPolicy(), EvalConfig(games=1, seed=637201, decision_recorder=recorder))
    manifest = recorder.finish(result)
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    if mutation == 'footer_count': rows[-1]['actions'] += 1
    if mutation == 'rank': next(r for r in rows if r['event']=='game_end')['rank'] = 0
    if mutation == 'trailing_event': rows.append({'event':'unknown'})
    if mutation == 'game_index': next(r for r in rows if r['event']=='action')['game'] = 99
    if mutation == 'manifest_incomplete': manifest['complete'] = False
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        for row in rows: stream.write(json.dumps(row)+'\n')
    manifest.update(file_identity(path))
    recorder.manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError): validate_trace(path)
