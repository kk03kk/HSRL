import json
import numpy as np
import pytest
from hsbg.az.train.quilboar_option import basis,roles,TERMS
from hsbg.env import obs_std as O


def test_option_reads_own_public_cards_and_preserves_other_lobbies():
    f=np.zeros(3227,np.float32)
    c=np.zeros(124,np.int64)
    rider=int(np.flatnonzero(roles()[0][:,0])[0])
    rat=int(np.flatnonzero(roles()[0][:,1])[0])
    f[0]=.50
    f[5]=1
    c[O.SLICE_CARD_OPP_BOARDS.start]=rider
    f[O.GLOBAL_LOBBY_RACES_START+7]=1
    assert basis(f,c) is None
    c[O.SLICE_CARD_BOARD.start]=rat
    c[O.SLICE_CARD_SHOP.start]=rider
    x=basis(f,c)
    assert x[8,TERMS.index('complete_rider')]>0
    assert x[16,TERMS.index('retain_supply')]<0
    f[O.GLOBAL_LOBBY_RACES_START+7]=0
    assert basis(f,c) is None


@pytest.mark.parametrize('card,expected',[(None,0),('BG31_323',2),('BG31_323_G',4)])
def test_native_choose_one_counter(card,expected):
    import hsbg_engine as E
    from hsbg.az.train.economic_tasks import debug,own
    from workflows.mirror_hard_gates import _drive_to_turn
    from workflows.quilboar_study import Recorder
    game=next(g for g in (E.Game(seed=i) for i in range(40))
              if json.loads(g.state_json())['lobby_races'] & 128)
    _drive_to_turn(game,3)
    debug(game,'clear_zone',zone='board')
    debug(game,'clear_zone',zone='hand')
    if card:
        debug(game,'add_minion',zone='board',card=card)
    debug(game,'add_minion',zone='board',card='BG31_326')
    debug(game,'add_spell',zone='hand',card='BG31_893')
    recorder=Recorder()
    recorder.begin_game(game,0,0,0)
    for a in (next(i for i in range(46,54) if game.legal_mask(0)[i]),30):
        before=recorder.before(game,0)
        game.step(0,a)
        recorder.after(game,before,None,a,False)
    assert all(m['gems']==expected for m in own(game)['board'])
    assert [e[2] for e in recorder.events]==([2*expected] if card else [])
