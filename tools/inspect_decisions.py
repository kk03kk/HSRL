"""Extract exact decision evidence for a model, seed and optional turn/action.

Selects the relevant shard using its manifest; does not load a model or run a game.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json, file_identity


def extract(panel_path, name, seed, *, turn=None, action_index=None):
    panel_path = Path(panel_path).resolve()
    panel = json.loads(panel_path.read_text(encoding='utf-8'))
    models = [m for m in panel['models'] if m['name']==name]
    if len(models) != 1: raise ValueError('model must match exactly one panel entry')
    model = models[0]
    references = model.get('decision_traces') or [dict(model['decision_trace'], directory=str(panel_path.parent))]
    matches = [r for r in references if seed in r['seeds']]
    if len(matches) != 1: raise ValueError('seed must match exactly one trace')
    reference = matches[0]
    path = Path(reference['directory'])/reference['file']
    if file_identity(path) != {k:reference[k] for k in ('bytes','sha256')}:
        raise ValueError('trace file differs from panel identity')
    rows = []
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('seed') != seed: continue
            if row['event']=='action' and (turn is None or row['turn']==turn) and (action_index is None or row['action_index']==action_index):
                rows.append(row)
            if row['event']=='game_end': break
    if not rows: raise ValueError('no decision matches the selection')
    return {'schema':'hsai.decision-extract/v1','identity':reference['identity'],
            'trace':{k:reference[k] for k in ('file','sha256')}, 'seed':seed, 'decisions':rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--turn', type=int)
    parser.add_argument('--action-index', type=int)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.panel,args.model,args.seed,turn=args.turn,action_index=args.action_index)
    atomic_json(args.out,result)
    print(json.dumps({'decisions':len(result['decisions']),'out':str(args.out)}))
