#!/usr/bin/env python3
"""离线物化固定来源配比；每(turn,seat)明确配额，按源游戏等权抽取。"""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import random


def mix(sources, quotas, out, split, seed=20261103):
    from hsbg.az.train.personal import checked_library, fingerprint, library_digest, write_json
    from hsbg.az.train.mirror_snapshots import write_snapshot_file, engine_stamp, MirrorSnapshotPool
    if set(sources) != set(quotas) or any(not isinstance(n, int) or n <= 0 for n in quotas.values()):
        raise ValueError('每个来源需要正整数配额')
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    buckets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    parents = {}
    for name, path in sources.items():
        manifest = checked_library(path)
        if manifest['split'] != split or not manifest.get('validation', {}).get('accepted'):
            raise ValueError('混合来源须已验证且分割一致')
        parents[name] = manifest['digest']
        for file in sorted(Path(path).glob('snap_*.json.gz')):
            with gzip.open(file, 'rt', encoding='utf-8') as stream:
                for row in json.load(stream)['frames']:
                    buckets[(row['turn'], row['seat'])][name][row['source_game_seed']].append(row)
    rng = random.Random(seed)
    frames, missing, allocation = [], {}, {}
    for key, sources_at_key in sorted(buckets.items()):
        missing[str(key)] = sorted(set(sources)-set(sources_at_key))
        allocation[str(key)] = {}
        for source, games in sorted(sources_at_key.items()):
            allocation[str(key)][source] = quotas[source]
            for _ in range(quotas[source]):
                game = rng.choice(sorted(games))
                row = rng.choice(games[game])
                frames.append(dict(row, mixture_source=source))
    out.mkdir(parents=True)
    write_snapshot_file(str(out/'snap_mixture.json.gz'), frames, seed=seed, engine=engine_stamp(),
                        meta=dict(quotas=quotas, split=split, materialized_sampling=True))
    write_json(out/'manifest.json', dict(schema='hsrl.library/v1', split=split, driver='fixed_mixture',
        fingerprint=fingerprint(), digest=library_digest(out), parents=parents, quotas=quotas,
        allocation=allocation, missing_sources={k:v for k,v in missing.items() if v},
        stats=MirrorSnapshotPool(str(out), sample='random').stats(), missing='engine_bot_fallback',
        note='配额精确到现有(turn,seat)；缺来源不跨回合补帧；物化抽样可能重复完整帧。'))
