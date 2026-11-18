"""Construct a late-turn opponent curriculum using intact training frames."""
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hsbg.runtime.artifacts import atomic_json, file_identity


def build(source, destination):
    manifest = json.loads((source / 'manifest.json').read_text())
    path = source / manifest['file']
    source_id = file_identity(path)
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        data = json.load(stream)
    by_turn = defaultdict(list)
    for frame in data['frames']:
        player = json.dumps(frame['player'], sort_keys=True, ensure_ascii=False).encode()
        if hashlib.sha256(player).hexdigest() != frame['frame_sha256']:
            raise ValueError('source player hash mismatch')
        if frame['stage'] != 'recruitment_end_before_combat' or frame['seat'] != 0:
            raise ValueError('unexpected frame semantics')
        by_turn[frame['turn']].append(frame)
    counts = {str(t): len(frames) for t, frames in by_turn.items()}
    if (manifest['bad_rows'] or counts != manifest['per_turn']
            or set(by_turn) != set(range(1, 21)) or len(data['frames']) != manifest['frames']):
        raise ValueError('incomplete training source')
    mapping = {t: t if t < 12 else t - 1 for t in range(1, 21)}
    data['frames'] = [dict(frame, turn=t) for t, origin in mapping.items()
                      for frame in by_turn[origin]]
    data['meta'] = dict(data['meta'], frames_total=len(data['frames']),
                        parent_library=str(source), parent_file=source_id, turn_mapping=mapping)
    destination.mkdir(parents=True, exist_ok=False)
    output = destination / manifest['file']
    with gzip.open(output, 'xt', encoding='utf-8', compresslevel=2) as stream:
        json.dump(data, stream, ensure_ascii=False, separators=(',', ':'))
    with gzip.open(output, 'rt', encoding='utf-8') as stream:
        on_disk = json.load(stream)
    if on_disk['frames'] != data['frames'] or source_id != file_identity(path):
        raise ValueError('sealed frame content differs')
    per_turn = Counter(frame['turn'] for frame in data['frames'])
    atomic_json(destination / 'manifest.json', dict(manifest, frames=len(data['frames']),
                per_turn=per_turn, turn_mapping=mapping, source_sha256=source_id['sha256'],
                content_sha256=file_identity(output)['sha256']))


if __name__ == '__main__':
    build(Path('artifacts/libraries/arena-cal-v14-snap'),
          Path('artifacts/libraries/arena-cal-v14-delay12-20261118-snap'))
