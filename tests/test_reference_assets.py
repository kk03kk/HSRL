import hashlib
import json
import pytest
from hsbg.runtime.references import reference_directory


def test_optional_reference_missing_modified_and_unexpected_files_fail_closed(tmp_path):
    (tmp_path/'config').mkdir()
    directory = tmp_path/'artifacts/references/test'
    manifest = {'schema':'hsai.reference-assets/v1','test':{'directory':'artifacts/references/test',
                'files':{'code.py':hashlib.sha256(b'pass\n').hexdigest()}}}
    (tmp_path/'config/reference-assets.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(FileNotFoundError): reference_directory('test', root=tmp_path)
    directory.mkdir(parents=True)
    (directory/'code.py').write_bytes(b'pass\n')
    assert reference_directory('test', root=tmp_path) == directory
    (directory/'code.py').write_bytes(b'changed\n')
    with pytest.raises(ValueError, match='content changed'): reference_directory('test', root=tmp_path)
    (directory/'code.py').write_bytes(b'pass\n')
    (directory/'extra.py').write_bytes(b'pass\n')
    with pytest.raises(ValueError, match='inventory'): reference_directory('test', root=tmp_path)
