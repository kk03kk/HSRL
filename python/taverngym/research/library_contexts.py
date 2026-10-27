"""Cover the native random lobby distribution without changing its rules or seeds."""
import itertools
import json
from pathlib import Path
import re
import taverngym as tg


def context(state):
    return (state['lobby_races'], state['deity'], state['paradox'])


def expected_contexts(rules):
    # Derive the card lists from the pinned native source; keep the eligibility
    # conditions in lockstep with flow.rs and verify every planned seed natively.
    root = Path(tg.__file__).resolve().parents[2]
    flow = (root/'crates/taverngym/src/flow.rs').read_text(encoding='utf-8')
    def ids(name):
        section = re.search(r'pub const '+name+r':.*?=\s*\[(.*?)\];', flow, re.S)
        if not section:
            raise ValueError('native lobby constants changed; review context enumeration')
        return re.findall(r'"([^"]+)"', section[1])
    catalog = {c['id']:c for c in json.loads(tg.catalog_json())}
    # Python observations reserve embedding zero for padding; saved native CardId
    # is zero-based. card_info accepts the observation/embedding index.
    index = {tg.card_info(i)[0]:i-1 for i in range(1,tg.N_CARDS)}
    names = ['BEAST','DEMON','DRAGON','ELEMENTAL','MECHANICAL','MURLOC',
             'PIRATE','QUILBOAR','UNDEAD','ABERRATION']
    forced = set(rules.get('lobby_types', []))
    if rules.get('aberration_forced', False):
        forced.add('ABERRATION')
    count = rules.get('lobby_races', 5)
    result = set()
    for combo in itertools.combinations(names, count):
        if not forced.issubset(combo):
            continue
        mask = sum(1<<names.index(n) for n in combo)
        for deity in ids('DEITIES'):
            for paradox in ids('PARADOX_VARIANTS'):
                tier = catalog[paradox]['tier']
                if 'UNDEAD' in combo and tier == 5:
                    continue
                if deity == 'BGFYM_011' and 'ABERRATION' in combo and tier in (5,6):
                    continue
                result.add((mask,index[deity],index[paradox]))
    if not result:
        raise ValueError('empty native context space')
    return result


def seed_plan(rules, required, *, start, copies=1, max_scan=250000):
    """Independent seed ranges per split; no mutation of generated game state."""
    found = {k:[] for k in required}
    config = json.dumps({**rules,'capture_library':True})
    for offset in range(max_scan):
        seed = start+offset
        key = context(json.loads(tg.Game(seed, config).state_json()))
        if key not in found:
            # A deliberately restricted test panel may ignore other real contexts.
            continue
        if len(found[key]) < copies:
            found[key].append(seed)
            if all(len(v) == copies for v in found.values()):
                return [seed for key in sorted(found) for seed in found[key]]
    missing = [k for k,v in found.items() if len(v)<copies]
    raise RuntimeError(f'context seed scan exhausted; {len(missing)} contexts missing')
