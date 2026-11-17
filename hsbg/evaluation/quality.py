"""Fail-closed quality checks for research suffixes with snapshot opponents."""

MIRROR_ERRORS=('mirror_rejected','mirror_missing','mirror_fallback','mirror_errors',
    'mirror_pool_unbalanced','mirror_pool_species_unbalanced',
    'mirror_applied_with_hand_queue','mirror_pool_preexisting','mirror_pool_species_preexisting')

def suffix_quality(result):
    if result['games_finished']!=result['games']:
        raise ValueError('suffix game did not finish')
    for key in ('illegal_actions','truncated_games','truncated_combats'):
        if result[key]:raise ValueError('suffix quality failure: '+key)
    mirror=result['mirror']
    for key in MIRROR_ERRORS:
        if mirror[key]:raise ValueError('suffix injection quality failure: '+key)
    if mirror['mirror_attempt']!=mirror['mirror_applied']:
        raise ValueError('suffix injection attempts were not all applied')
    return {'passed':True,'games':result['games'],'actions':result['decisions'],
        'mirror_attempt':mirror['mirror_attempt'],'mirror_applied':mirror['mirror_applied'],
        'mirror_skipped_ended':result['mirror_skipped_ended'],
        'scope':'Already-ended seats may be skipped; no missing/fallback/rejected injection or truncated combat accepted'}
