import copy
import gzip
import json

import pytest

from hsbg.az.train.eval_policy import EvalProtocol, assert_same_protocol, protocol_from_args
from tools import merge_panels as mp, merge_panels_multi as mm
from tests.test_merge_panels_validity import _write_shard, _sub_panel
from tests.test_merge_panel_identity import _shard
from types import SimpleNamespace


def mutate(path, fn):
    data = json.loads(path.read_text(encoding='utf-8'))
    fn(data)
    path.write_text(json.dumps(data), encoding='utf-8')


@pytest.mark.parametrize('fault', ['different_models_protocol', 'missing_protocol', 'duplicate_seeds',
                                  'short_ranks', 'short_seeds', 'bad_rank', 'missing_counter', 'missing_rows', 'bad_rows'])
def test_merge_refuses_corrupt_evidence(tmp_path, fault):
    a = _write_shard(tmp_path, 's0', {'vA': 10, 'vB': 100})
    b = _write_shard(tmp_path, 's1', {'vA': 30, 'vB': 130})
    if fault == 'different_models_protocol':
        for s in (a,b): mutate(s/'extra-vB.json', lambda d: d.update(protocol_sha256='b'*64))
    elif fault == 'missing_protocol':
        mutate(b/'extra-vA.json', lambda d: d.pop('protocol_sha256'))
    elif fault == 'duplicate_seeds':
        mutate(b/'extra-vA.json', lambda d: d.update(seed_sequence=list(range(10,22))))
    elif fault == 'short_ranks':
        mutate(b/'extra-vA.json', lambda d: d['rank_sequence'].pop())
    elif fault == 'short_seeds':
        mutate(b/'extra-vA.json', lambda d: d['seed_sequence'].pop())
    elif fault == 'bad_rank':
        mutate(b/'extra-vA.json', lambda d: d['rank_sequence'].__setitem__(0, 9))
    elif fault == 'missing_counter':
        mutate(b/'extra-vA.json', lambda d: d['outcome'].pop('illegal_actions'))
    elif fault == 'missing_rows':
        (b/'rows-vA.jsonl.gz').unlink()
    else:
        with gzip.open(b/'rows-vA.jsonl.gz', 'at') as fh: fh.write('BROKEN\n')
    with pytest.raises(SystemExit):
        mp.merge([a,b], tmp_path/'out.json', expect_games=24)
    assert not (tmp_path/'out.json').exists()


def test_merge_refuses_mixed_model_weights(tmp_path):
    a = _shard(tmp_path/'a', 10, with_ident_in_extra=True)
    b = _shard(tmp_path/'b', 30, with_ident_in_extra=True)
    mutate(b/'extra-va.json', lambda d: d.update(checkpoint_sha256='b'*64))
    with pytest.raises(SystemExit, match='checkpoint_sha256'):
        mp.merge([a,b], tmp_path/'out.json')


def test_full_protocol_is_verified_not_just_its_label():
    proto = EvalProtocol().as_dict()
    proto['protocol_sha256'] = EvalProtocol().sha256()
    changed = copy.deepcopy(proto)
    changed['contract']['move_cap'] = 999
    with pytest.raises(ValueError, match='正文'):
        assert_same_protocol({'protocol': proto}, {'protocol': changed})


def test_zero_injection_probability_is_not_replaced_by_one():
    assert protocol_from_args(SimpleNamespace(mirror_p=0)).mirror_p == 0


def test_legacy_merge_cannot_launder_missing_protocol(tmp_path):
    files = []
    for i, proto in enumerate(('a'*64, None)):
        path = tmp_path/f'{i}.json'
        path.write_text(json.dumps(_sub_panel('vA', i*100, proto=proto)), encoding='utf-8')
        files.append(path)
    with pytest.raises(SystemExit, match='legacy'):
        mm.merge(files, tmp_path/'out.json')


@pytest.mark.parametrize('merger', [mp.merge, mm.merge])
def test_merging_preserves_an_existing_report(tmp_path, merger):
    out = tmp_path/'out.json'
    out.write_bytes(b'existing report')
    with pytest.raises(SystemExit, match='overwrite'):
        merger([], out)
    assert out.read_bytes() == b'existing report'
