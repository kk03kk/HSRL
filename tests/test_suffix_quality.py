import copy
import pytest
from hsbg.evaluation.quality import suffix_quality,MIRROR_ERRORS

def result():
    return {'games':1,'games_finished':1,'decisions':300,'illegal_actions':0,'truncated_games':0,
        'truncated_combats':0,'mirror_skipped_ended':7,
        'mirror':dict.fromkeys(MIRROR_ERRORS,0)|{'mirror_attempt':35,'mirror_applied':35}}

def test_suffix_quality_accepts_ended_seat_skips():
    assert suffix_quality(result())['passed']

@pytest.mark.parametrize('field',MIRROR_ERRORS+('truncated_combats','truncated_games','illegal_actions'))
def test_suffix_quality_rejects_each_failure_class(field):
    r=result();target=r['mirror'] if field in MIRROR_ERRORS else r;target[field]=1
    with pytest.raises(ValueError,match='quality failure'):suffix_quality(r)

def test_suffix_quality_rejects_unfinished_or_incomplete_injection():
    r=result();r['games_finished']=0
    with pytest.raises(ValueError,match='finish'):suffix_quality(r)
    r=result();r['mirror']['mirror_applied']-=1
    with pytest.raises(ValueError,match='not all applied'):suffix_quality(r)
