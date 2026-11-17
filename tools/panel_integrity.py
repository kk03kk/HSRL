"""Validate evidence identity and sample keys before aggregating statistics."""
from typing import Any


def validate_samples(parts: list[dict[str, Any]], *, name: str) -> None:
    seen = set()
    for part in parts:
        ranks = part.get('rank_sequence')
        games = part.get('games', (part.get('outcome') or {}).get('games'))
        if not isinstance(games, int) or isinstance(games, bool) or games < 1:
            raise SystemExit(f'{name}: invalid games count {games!r}')
        if not isinstance(ranks, list) or len(ranks) != games:
            raise SystemExit(f'{name}: rank_sequence length != games')
        if any(type(r) is not int or not 0 <= r <= 8 for r in ranks):
            raise SystemExit(f'{name}: invalid rank (expected integer 0..8)')
        outcome = part.get('outcome') or {}
        for key in ('illegal_actions', 'truncated_games'):
            value = outcome.get(key)
            if type(value) is not int or value < 0:
                raise SystemExit(f'{name}: missing or invalid {key}')
        if ranks.count(0) > outcome['truncated_games']:
            raise SystemExit(f'{name}: unknown ranks not covered by truncated_games')
        seq = part.get('seed_sequence')
        if seq is not None:
            if not isinstance(seq, list) or len(seq) != games or any(type(s) is not int for s in seq):
                raise SystemExit(f'{name}: invalid seed_sequence length or type')
            if len(set(seq)) != len(seq) or seen.intersection(seq):
                raise SystemExit(f'{name}: duplicate seed across samples/shards')
            seen.update(seq)


def validate_identity(parts: list[dict[str, Any]], *, name: str) -> None:
    for key in ('checkpoint_sha256', 'arch'):
        values = {p[key] for p in parts if p.get(key) is not None}
        if len(values) > 1:
            raise SystemExit(f'{name}: inconsistent {key} across shards')


def validate_rows(rows: list[dict[str, Any]], *, name: str, seeds=None) -> None:
    keys = set()
    for row in rows:
        if not isinstance(row, dict) or type(row.get('seed')) is not int or type(row.get('turn')) is not int:
            raise SystemExit(f'{name}: row missing integer seed/turn')
        key = (row['seed'], row['turn'])
        if key in keys:
            raise SystemExit(f'{name}: duplicate seed/turn row {key}')
        keys.add(key)
    if seeds is not None and {s for s, _ in keys} != set(seeds):
        raise SystemExit(f'{name}: rows and seed_sequence cover different games')
