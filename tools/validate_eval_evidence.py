"""Verify complete per-action evidence against a single or merged evaluation panel."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hsbg.az.train.eval_policy import assert_same_protocol
from hsbg.evaluation.trace import validate_trace
from hsbg.runtime.artifacts import atomic_json, file_identity


def _validate_reference(item):
    reference, replay = item
    trace = Path(reference['directory'])/reference['file']
    if file_identity(trace) != {k:reference[k] for k in ('bytes','sha256')}:
        raise ValueError('panel trace identity mismatch')
    result = validate_trace(trace, replay=replay)
    if result['identity'] != reference['identity'] or result['seeds'] != reference['seeds']:
        raise ValueError('panel trace reference differs from manifest/records')
    return result


def validate_panel(path, *, replay=16, workers=1):
    path = Path(path).resolve()
    panel = json.loads(path.read_text(encoding='utf-8'))
    protocol = assert_same_protocol(panel)
    models = panel.get('models') or []
    if not models or len({m['name'] for m in models}) != len(models):
        raise ValueError('panel models must be nonempty and uniquely named')
    references = []
    groups = []
    for model in panel['models']:
        traces = model.get('decision_traces') or ([dict(model['decision_trace'], directory=str(path.parent))]
                                                 if model.get('decision_trace') else [])
        if not traces: raise ValueError(f'{model["name"]}: missing full decision evidence')
        groups.append(traces)
        references.extend(traces)
    if type(workers) is not int or workers < 1: raise ValueError('workers must be a positive integer')
    tasks = [(reference,replay) for reference in references]
    if workers == 1:
        validated = list(map(_validate_reference,tasks))
    else:
        # Spawn avoids inheriting Torch/native thread pools or GPU contexts.
        with ProcessPoolExecutor(max_workers=min(workers,len(tasks)),
                                 mp_context=multiprocessing.get_context('spawn')) as pool:
            validated = list(pool.map(_validate_reference,tasks))
    checked = iter(validated)
    results = []
    for model,traces in zip(panel['models'],groups):
        games = actions = replayed = 0
        seeds = []
        ranks = []
        for reference in traces:
            identity = reference['identity']
            if identity['name'] != model['name'] or identity['checkpoint_sha256'] != model['checkpoint_sha256']:
                raise ValueError('trace belongs to another model')
            assert_same_protocol(panel, {'protocol':identity['protocol']})
            result = next(checked)
            games += result['games']
            actions += result['actions']
            replayed += result['replayed']
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
        results.append({'name':model['name'],'games':games,'actions':actions,
                        'trace_files':len(traces),'replayed_actions':replayed})
    return {'schema':'hsai.eval-evidence-validation/v1', 'valid':True,
            'protocol_sha256':protocol, 'panel':file_identity(path), 'models':results,
            'replay_steps_per_trace':replay,
            'replayed_actions':sum(r['replayed_actions'] for r in results)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--replay', type=int, default=16)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    result = validate_panel(args.panel, replay=args.replay, workers=args.workers)
    atomic_json(args.out, result)
    print(json.dumps(result))
