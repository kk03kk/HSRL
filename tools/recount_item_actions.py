"""Recover item subtypes directly from sealed pre-action engine records."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsbg.az.train.evaluate import item_action_label
from hsbg.runtime.artifacts import atomic_json, file_identity
from tools.compare_research_panels import load_sealed_panel


def count(ref):
    path = Path(ref['directory']) / ref['file']
    if file_identity(path) != {k: ref[k] for k in ('sha256', 'bytes')}:
        raise ValueError('source trace identity differs')
    counts = Counter()
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        if json.loads(next(stream))['identity'] != ref['identity']:
            raise ValueError('source header differs')
        for line in stream:
            row = json.loads(line)
            if row['event'] != 'action':
                continue
            counts['decisions'] += 1
            label = item_action_label(row['state_before']['players'][row['seat']], row['action'])
            if label:
                counts[label] += 1
                if row['turn'] >= 9:
                    counts['T9plus_' + label] += 1
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', type=Path)
    parser.add_argument('name')
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    panel = load_sealed_panel(args.panel)
    model, = [m for m in panel['models'] if m['name'] == args.name]
    with ProcessPoolExecutor(max_workers=4) as pool:
        counts = sum(pool.map(count, model['decision_traces']), Counter())
    if counts['decisions'] != model['outcome']['decisions']:
        raise ValueError('incomplete action coverage')
    atomic_json(args.out, dict(panel=file_identity(args.panel), name=args.name,
                model_sha256=model['checkpoint_sha256'], games=model['games'], counts=counts))
