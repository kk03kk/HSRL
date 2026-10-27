"""Read installed Unity DBF tables; write evidence only beneath an explicit output directory.

Requires UnityPy (may be installed in a project-local --deps directory).
Does not launch the client, read credentials, or replace simulator data.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import re
import sys


def inspect(install, out, deps=None):
    if deps:
        sys.path.insert(0, str(Path(deps).resolve()))
    import UnityPy
    install, out = Path(install).resolve(), Path(out).resolve()
    if out == install or install in out.parents:
        raise ValueError('Output must be outside the game installation')
    out.mkdir(parents=True, exist_ok=True)
    # Installed UnityPlayer file version is 6000.3.11.3145967; stripped bundle
    # headers require a compatible explicit Unity serialization fallback.
    UnityPy.config.FALLBACK_UNITY_VERSION = '6000.3.11f1'
    src = install/'Data/Win/dbf.unity3d'
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    tables, index = {}, []
    for obj in UnityPy.load(str(src)).objects:
        if obj.type.name != 'MonoBehaviour':
            continue
        data = obj.read_typetree()
        name, rows = data.get('m_Name', ''), data.get('Records', [])
        index.append({'table': name, 'rows': len(rows)})
        if name in ('CARD', 'CARD_TAG', 'CARD_RACE', 'POWER_DEFINITION', 'CARD_CHANGE'):
            tables[name] = rows
            (out/(name+'.json')).write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    tags = defaultdict(dict)
    for row in tables['CARD_TAG']:
        tags[row['m_cardId']][str(row['m_tagId'])] = row['m_tagValue']
    cards = []
    for row in tables['CARD']:
        ts = tags[row['m_ID']]
        # Positive pool flags are client evidence, NOT a live server whitelist.
        if not (row['m_noteMiniGuid'].startswith(('BG', 'Bacon', 'TB_Bacon'))
                or ts.get('1456') or ts.get('3081') or ts.get('1440')):
            continue
        def localized(field, index):
            values = row.get(field, {}).get('m_locValues', [])
            return values[index] if len(values) > index else None
        cards.append({'id': row['m_noteMiniGuid'], 'dbf_id': row['m_ID'],
                      'name_zh': localized('m_name', 12), 'name_en': localized('m_name', 0),
                      'text_zh': localized('m_textInHand', 12), 'text_en': localized('m_textInHand', 0),
                      'tags': ts, 'active_event': row.get('m_battlegroundsActiveEvent'),
                      'every_game_event': row.get('m_battlegroundsEveryGameEvent')})
    product = install/'.product.db'
    builds = sorted(set(re.findall(rb'\d{2}\.\d+\.\d+\.\d{6}', product.read_bytes()))) if product.exists() else []
    report = {'installed_versions': [x.decode() for x in builds], 'source': str(src),
              'source_sha256': before, 'source_unchanged': before == hashlib.sha256(src.read_bytes()).hexdigest(),
              'tables': index, 'card_records': len(tables['CARD']), 'tag_records': len(tables['CARD_TAG']),
              'bg_candidate_records': len(cards),
              'pool_minion_flag_records': sum(c['tags'].get('1456') == 1 for c in cards),
              'tag_3081_flag_records': sum(c['tags'].get('3081') == 1 for c in cards),
              'bg_types': dict(Counter(str(c['tags'].get('202')) for c in cards)),
              'qualification': 'Includes historical/golden/token entries; client flags are not verified live pool membership. Server hotfixes and full resolution order not established.'}
    (out/'battlegrounds-candidates.json').write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return {k: v for k, v in report.items() if k != 'tables'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--install', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--deps')
    a = p.parse_args()
    print(json.dumps(inspect(a.install, a.out, a.deps), ensure_ascii=False, indent=2))
