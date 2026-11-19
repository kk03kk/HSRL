"""Read archived recruit boards, combat outcomes and lobby survival for one panel."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json, file_identity
from tools.compare_research_panels import load_sealed_panel


def lobby_openings(ref):
    path = Path(ref['directory']) / ref['file']
    if file_identity(path)['sha256'] != ref['sha256']:
        raise ValueError('trace fingerprint differs')
    rows = []
    last = None
    prefix = re.compile(rb'^\{"event":"action","game":\d+,"seed":(\d+),"action_index":\d+,"turn":(\d+),')
    with gzip.open(path, 'rb') as stream:
        for line in stream:
            hit = prefix.match(line)
            if hit is None:
                continue
            pair = tuple(map(int, hit.groups()))
            if pair == last or pair[1] == 0:
                continue
            last = pair
            action = json.loads(line)
            players = action['state_before']['players']
            seat = action['seat']
            alive = [i for i,p in enumerate(players) if p['alive'] and p['placement'] == 0]
            if seat not in alive:
                raise ValueError('eliminated learner in action trace')
            effective = [p['health'] + p['armor'] for p in players]
            rows.append([*pair,len(alive),effective[seat],
                         [effective[i] for i in alive if i != seat]])
    return rows


def run(panel_path, out):
    panel = load_sealed_panel(panel_path)
    model, = panel['models']
    assert model['checkpoint_sha256'] == 'e1117934c76c37090212ec02e75d8ef52b21280c8d47f7dd5eb6089a228f74ad'
    library = Path(model['field_stats']['library']['root'])
    manifest = json.loads((library/'manifest.json').read_text())
    frozen = json.loads((panel_path.parent/'run.json').read_text())['inputs']
    frame_values = defaultdict(list)
    for file in sorted(library.iterdir()):
        if file_identity(file) != frozen[str(file)]:
            raise ValueError('source library changed')
        if file.name.startswith('snap_'):
            with gzip.open(file, 'rt') as stream:
                for f in json.load(stream)['frames']:
                    assert f['source_turn'] == f['turn']
                    value = sum(u['atk']+u['hp'] for u in f['player']['board'])
                    frame_values[f['turn']].append([value,f['source_side'],f['source_game_id']])
    assert Counter({str(t):len(v) for t,v in frame_values.items()}) == manifest['turns']
    exposures = {}
    for ref in model['decision_traces']:
        extra_path = Path(ref['directory']) / ('extra-'+model['name']+'.json')
        shard = json.loads(extra_path.read_text())
        assert shard['checkpoint_sha256'] == model['checkpoint_sha256']
        assert shard['protocol_sha256'] == panel['protocol']['protocol_sha256']
        assert shard['decision_trace']['sha256'] == ref['sha256']
        for row in shard['combat_exposure']:
            k = (row['seed'],row['turn'])
            assert k not in exposures
            opp = row['source_strength']
            if not row['ghost']:
                assert row['mirror_applied'] and row['turn'] == row['source_turn']
                assert opp == sum(u['atk']+u['hp'] for u in row['opponent_board'])
            else:
                assert opp is None
            exposures[k] = [opp,int(row['win']),int(row['tie']),row['damage_taken']]
    ranks = dict(zip(model['seed_sequence'],model['rank_sequence']))
    own = {(r['seed'],r['turn']):sum(c['atk']+c['hp'] for c in r['roster']) for r in model['trace']}
    assert all(own[r['seed'],r['turn']] == r['strength'] for r in model['trace'])
    assert own.keys() == exposures.keys()
    with ProcessPoolExecutor(max_workers=8) as pool:
        lobby = {(s,t):[n,hp,others] for part in pool.map(lobby_openings,model['decision_traces'])
                 for s,t,n,hp,others in part}
    assert own.keys() == lobby.keys()
    atomic_json(out,dict(panel=file_identity(panel_path),library=manifest['digest'],
                         source_games=len(manifest['games']),protocol=panel['protocol']['protocol_sha256'],
                         field={t:[[v,side] for v,side,_ in vals] for t,vals in frame_values.items()},
                         origins={t:len({gid for _,_,gid in vals}) for t,vals in frame_values.items()},
                         columns=['seed','turn','rank','strength','opponent','win','tie','damage_taken',
                                  'alive_at_open','own_hp_armor','other_alive_hp_armor'],
                         encounters=[[s,t,ranks[s],own[s,t],*exposures[s,t],*lobby[s,t]] for s,t in sorted(own)]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel',type=Path)
    parser.add_argument('out',type=Path)
    args = parser.parse_args()
    run(args.panel,args.out)
