import copy
import json

import pytest
import taverngym as tg
from taverngym.research.manual_bank import resolve_card,make_fixture,command


def catalog():
    return {c['id']:c for c in json.loads(tg.catalog_json())}


def test_explicit_shape_normalizes_reference_ids_without_touching_stats():
    c=catalog()
    ordinary={'card_id':'BG34_142_G','name':'狂热变装鱼人','golden':False,'attack':16,'health':17}
    before=copy.deepcopy(ordinary)
    assert resolve_card(ordinary,c)['id']=='BG34_142'
    assert ordinary==before
    golden={'card_id':'BG20_104','name':'狂棍野猪人','golden':True,'attack':4,'health':14}
    assert resolve_card(golden,c)['id']=='BG20_104_G'


def test_post_start_hand_scaling_proxy_does_not_reapply_entered_attack():
    c=catalog()
    meta={'hero':'钩牙船长','available_tribes':['PIRATE','ABERRATION','MURLOC','DRAGON','QUILBOAR']}
    shot={'turn':8,'tier':5,'health':30,'armor':14,'hand':[],
          'board':[{'card_id':'BG34_142_G','name':'狂热变装鱼人','golden':False,
                    'attack':16,'health':17,'keywords':['DIVINE_SHIELD','TAUNT'],'details':''}]}
    with pytest.raises(ValueError,match='最高攻击'):make_fixture(meta,shot,c,after_start=False)
    saved,board,changes,limits=make_fixture(meta,shot,c,after_start=True)
    assert board[0]['id']=='BG34_142' and board[0]['attack']==16
    assert changes and limits
    g=tg.Game.from_json(saved);before=g.state_json()
    assert json.loads(g.view(0))['me']['hand']==[]
    command(g,'add_minion',p=1,zone='board',card='BG20_100',atk=1,hp=1)
    preview=command(g,'combat_preview',a=0,b=1,n=4,details=True)
    assert preview['win_a']==1 and not any(t['truncated'] for t in preview['trials'])
    # Native opening buff=0 with the empty proxy hand; it must not become 32 Attack.
    trace=command(g,'combat_preview',a=0,b=1,n=1,details=False)['sample']
    attacks=[m['atk'] for frame in trace['frames'] for m in frame['boards'][0] if m['id']=='BG34_142']
    assert attacks and set(attacks)=={16}
