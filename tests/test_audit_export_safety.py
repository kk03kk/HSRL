import json
import sys

import pytest

from tools import export_model_library as export


def setup_export(monkeypatch, tmp_path, model='m'):
    lib = tmp_path/'library'
    (lib/'cohorts').mkdir(parents=True)
    (lib/'manifest.json').write_text(json.dumps({'models': [{'id': model}], 'counts': {}}))
    (lib/'cohorts/cohorts.json').write_text('{}')
    monkeypatch.setattr(export, 'LIB', lib)
    monkeypatch.setattr(sys, 'argv', ['export', '--models', model, '--out', str(tmp_path/'out.tar.gz')])
    return lib


@pytest.mark.parametrize('name', ['out.tar.gz', 'out.tar.gz.sha256', 'out.staging'])
def test_export_preserves_existing_evidence(monkeypatch, tmp_path, name):
    setup_export(monkeypatch, tmp_path)
    path = tmp_path/name
    path.write_bytes(b'previous evidence')
    with pytest.raises(SystemExit, match='overwrite'):
        export.main()
    assert path.read_bytes() == b'previous evidence'


def test_export_refuses_missing_weights(monkeypatch, tmp_path):
    setup_export(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match='incomplete'):
        export.main()
    assert not (tmp_path/'out.tar.gz').exists()


def test_export_refuses_unsafe_manifest_ids(monkeypatch, tmp_path):
    setup_export(monkeypatch, tmp_path, '../outside')
    with pytest.raises(SystemExit, match='safe'):
        export.main()


def test_export_success_is_self_contained(monkeypatch, tmp_path):
    import tarfile
    lib = setup_export(monkeypatch, tmp_path)
    model = lib/'models/m'
    model.mkdir(parents=True)
    (model/'model.pt').write_bytes(b'test-only-opaque-weight-file')
    (model/'model.json').write_text('{}')
    assert export.main() == 0
    with tarfile.open(tmp_path/'out.tar.gz') as tar:
        assert tar.extractfile('library-v1/models/m/model.pt').read() == (model/'model.pt').read_bytes()
    assert (tmp_path/'out.tar.gz.sha256').read_text().startswith(export.sha256_file(tmp_path/'out.tar.gz'))
