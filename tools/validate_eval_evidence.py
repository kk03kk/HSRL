"""Verify complete per-action evidence against a single or merged evaluation panel."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.az.train.eval_policy import assert_same_protocol
from hsbg.evaluation.trace import validate_trace
from hsbg.runtime.artifacts import atomic_json, file_identity


def validate_panel(path, *, replay=16):
    path = Path(path).resolve()
    panel = json.loads(path.read_text(encoding='utf-8'))
    protocol = assert_same_protocol(panel)
    results = []
    for model in panel['models']:
        traces = model.get('decision_traces') or ([dict(model['decision_trace'], directory=str(path.parent))]
                                                 if model.get('decision_trace') else [])
        if not traces:
            raise ValueError(f'{model["name"]}: missing full decision evidence')
        games = actions = 0
        seeds = []
        ranks = []
        for reference in traces:
            trace = Path(reference['directory'])/reference['file']
            identity = reference['identity']
            if identity['name'] != model['name'] or identity['checkpoint_sha256'] != model['checkpoint_sha256']:
                raise ValueError('trace belongs to another model')
            assert_same_protocol(panel, {'protocol':identity['protocol']})
            if file_identity(trace) != {k:reference[k] for k in ('bytes','sha256')}:
                raise ValueError('panel trace identity mismatch')
            result = validate_trace(trace, replay=replay)
            if result['identity'] != identity or result['seeds'] != reference['seeds']:
                raise ValueError('panel trace reference differs from manifest/records')
            games += result['games']
            actions += result['actions']
            seeds.extend(result['seeds'])
            ranks.extend(result['ranks'])
        if seeds != model['seed_sequence'] or len(set(seeds)) != games:
            raise ValueError('panel/trace seed coverage differs')
        if games != len(model['rank_sequence']) or actions != model['outcome']['decisions']:
            raise ValueError('panel/trace coverage differs')
        if ranks != model['rank_sequence']:
            raise ValueError('panel ranks differ from terminal game evidence')
        if any(model['outcome'].get(k, 0) for k in ('illegal_actions', 'truncated_games')):
            raise ValueError('invalid evaluation outcome')
        results.append({'name':model['name'],'games':games,'actions':actions,'trace_files':len(traces)})
    return {'schema':'hsai.eval-evidence-validation/v1', 'valid':True,
            'protocol_sha256':protocol, 'panel':file_identity(path), 'models':results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--replay', type=int, default=16)
    args = parser.parse_args()
    result = validate_panel(args.panel, replay=args.replay)
    atomic_json(args.out, result)
    print(json.dumps(result))
