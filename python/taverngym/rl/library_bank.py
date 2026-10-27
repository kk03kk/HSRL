"""Fingerprint and split checks for native exogenous opponent libraries."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import taverngym as tg


@lru_cache(maxsize=4)
def bank_text(path, expected_sha, usage=None):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('library bank checksum mismatch')
    bank = json.loads(raw)
    if bank['native_source'] != tg.build_info()['source_hash']:
        raise ValueError('library native source mismatch; regenerate or explicitly migrate')
    if bank['split'] not in ('train', 'validation', 'test'):
        raise ValueError('library requires assigned source split')
    if usage == 'train' and bank['split'] != 'train':
        raise ValueError('cannot train on a held-out library')
    if usage == 'evaluation' and bank['split'] not in ('validation','test'):
        raise ValueError('independent library evaluation cannot use training sources')
    return raw.decode('utf-8')


def assert_disjoint(paths):
    groups = set()
    for path in paths:
        b = json.loads(Path(path).read_text(encoding='utf-8'))
        here = {t['source_group'] for t in b['traces']}
        if here & groups:
            raise ValueError('library source groups overlap across splits')
        groups |= here
    return True
