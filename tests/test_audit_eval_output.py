import json

import pytest

from workflows.eval_human_gap import _preflight_outputs
from tools.compare_panels import _pair_keys, load_models, main


@pytest.mark.parametrize('names', [['x', 'x'], ['a/b', 'a_b'], ['A', 'a'], ['---']])
def test_output_names_cannot_overwrite_each_other(tmp_path, names):
    with pytest.raises(SystemExit, match='unique'):
        _preflight_outputs(names, tmp_path/'panel.json')


@pytest.mark.parametrize('name', ['panel.json', 'rows-x.jsonl.gz', 'extra.json'])
def test_existing_evidence_survives_preflight(tmp_path, name):
    path = tmp_path/name
    path.write_bytes(b'old evidence')
    with pytest.raises(SystemExit, match='overwrite'):
        _preflight_outputs(['x'], tmp_path/'panel.json')
    assert path.read_bytes() == b'old evidence'


def test_preflight_allows_run_metadata(tmp_path):
    (tmp_path/'run.env').write_text('metadata')
    _preflight_outputs(['x', 'y'], tmp_path/'panel.json')


@pytest.mark.parametrize('seeds', [[1], [1, 1], ['1', '2']])
def test_bad_seed_evidence_never_falls_back_to_index(seeds):
    with pytest.raises(ValueError):
        _pair_keys([1, 2], [2, 3], seeds, [1, 2])


def test_duplicate_model_names_require_explicit_merge(tmp_path):
    p = tmp_path/'p.json'
    p.write_text(json.dumps({'models': [{'name': 'x'}, {'name': 'x'}]}))
    with pytest.raises(SystemExit, match='duplicate'):
        load_models([str(p)])


def test_legacy_flag_cannot_bypass_known_protocol_mismatch(tmp_path):
    paths = []
    for name in ['a', 'b']:
        p = tmp_path/f'{name}.json'
        p.write_text(json.dumps({'protocol': {'protocol_sha256': name*64},
                                 'models': [{'name': name}]}))
        paths.append(str(p))
    with pytest.raises(SystemExit, match='不一致'):
        main([*paths, '--champion', 'a', '--candidate', 'b', '--legacy-no-protocol'])
