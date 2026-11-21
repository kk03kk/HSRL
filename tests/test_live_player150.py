from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from hsbg.az.train import player_observation
from hsbg.env import obs_std as O
from hsrl_agent.live.action_profiles import PLAYER150, PROFILES
from hsrl_agent.live.contracts import Binding, Decision
from hsrl_agent.live.controller import Controller
from hsrl_agent.live.player150 import encode_observation, observed_context, public_facts
from tests.live_fixtures import snapshot


def controller(tmp_path, **options):
    return Controller(SimpleNamespace(peer_round=True,action_profile=PLAYER150),
                      None, None, tmp_path, practical=True, **options)


def test_new_limits_keep_same_code_and_prohibit_immediate_reverse(tmp_path):
    c=controller(tmp_path)
    s=snapshot()
    turn=(s['game_id'],s['client']['turn'])
    end, freeze, move, inverse=(Binding(0,'end'),Binding(2,'freeze'),Binding(197,'move'),Binding(203,'move'))
    choices=(end,freeze,move,inverse)
    c._note_action(turn,197)
    assert c._guarded_bindings(s,choices)==(end,freeze,move)
    c._note_action(turn,197)
    assert c._guarded_bindings(s,choices)==(end,freeze,move)
    c._note_action(turn,197)
    assert c._guarded_bindings(s,choices)==(end,freeze)
    c._note_action(turn,2)
    assert c._guarded_bindings(s,choices)==(end,freeze)
    c._note_action(turn,2)
    assert c._guarded_bindings(s,choices)==(end,)
    s['client']['turn']+=1
    assert c._guarded_bindings(s,choices)==choices


def test_live_move_mask_matches_frozen_move_guard(tmp_path):
    from hsbg.az.train.move_guard import MoveGuard

    c=controller(tmp_path)
    s=snapshot()
    turn=s['client']['turn']
    allowed=np.zeros(245,dtype=bool)
    allowed[0]=True
    allowed[[197,198,203,205,210,211]]=True
    kinds=[Binding(0,'end'), *(Binding(int(a),'move') for a in np.flatnonzero(allowed[196:])+196)]
    rules=PROFILES[PLAYER150]
    training=MoveGuard(cap=rules.moves,repeat_limit=0,hard_cap=True)
    for taken in [197,197,205,0]:
        expected,_=training.guarded_mask(allowed,game_id=1,seat=0,turn=turn)
        assert [b.action for b in c._guarded_bindings(s,kinds)]==np.flatnonzero(expected).tolist()
        if taken:
            assert expected[taken]
            training.note_action(taken,game_id=1,seat=0,turn=turn)
            c._note_action((s['game_id'],turn),taken)


def test_budget_reserves_and_charges_model_steps_once(tmp_path):
    c=controller(tmp_path,enable_input=True)
    s=snapshot()
    s['client']['game_id']=s['game_id']
    s['client']['log_position']=40
    turn=(s['game_id'],s['client']['turn'])
    d=Decision(s['game_id'],s['revision'],'stamp',46,Binding(46,'play',source=1,targets=(2,)), 'sha','logits')
    c.turn_counts[turn]=dict(actions=148,moves=0)
    c.pending=(d,s,c.clock())
    assert c.remaining_actions(s['client'])==1
    d=replace(d,binding=replace(d.binding,target=2))
    c.pending=(d,s,c.clock());c.target_sent=True
    assert c.budget_expired(s['client'])
    c._charge_receipt(turn,d,s);c._save_state();c.persist()
    c=controller(tmp_path,enable_input=True)
    c._charge_receipt(turn,d,s)
    assert c.turn_counts[turn]['actions']==150


def test_exhausted_budget_stops_before_inference_even_with_owned_choice(tmp_path):
    c=controller(tmp_path)
    s=snapshot()
    s['client']['game_id']=s['game_id']
    s['client']['interaction']=dict(owner=s['client']['local_player_id'])
    s['client']['choices']=[dict(entity_id=88,card_type='MINION')]
    c.turn_counts[(s['game_id'],s['client']['turn'])]=dict(actions=150,moves=0)
    assert c.tick(s)=='waiting_budget_native_resolution'
    assert c.pending is None


def test_public_encoder_uses_exact_slot_schema_and_omits_unannounced_opponents():
    s=snapshot()
    s['view']['global']['hero_select']=0
    s['view']['power_entities']=[5]
    s['view']['pending']=0
    rogers=dict(entity_id=7,card_id='BG33_823',tags={'TAG_SCRIPT_DATA_NUM_1':'4','TAG_SCRIPT_DATA_NUM_3':'9'})
    s['client']['board']=[rogers];s['client']['hand']=[];s['client']['shop']=[]
    s['client']['controls']=[dict(entity_id=5,card_id='P',tags={})]
    info={'P':{'text':'Use this.'},'BG33_823':{'text':'After you spend 9 Gold, get a random Bounty. (9 Gold left!)'}}
    s['remaining_actions']=148
    s['player150_public']=dict(growth=[None]*8+[2,4],health_prices=[],last_seen_turn=6,
                              encounter=dict(slot=1,hero='H'))
    facts=public_facts(s,info)
    assert facts['faces']==[[1,'P','Use this.'],[19,'BG33_823','After you spend 9 Gold, get a random Bounty. (4 Gold left!)']]
    f,c=encode_observation(np.ones(3227,dtype=np.float32),np.ones(124,dtype=np.int32),s,{'H':7},info)
    assert f.shape==(4967,) and c.shape==(124,)
    np.testing.assert_array_equal(f[3227:],player_observation.encode(facts))
    assert f[O.SLICE_OPPONENTS].sum()==2 and not f[O.SLICE_OPP_BOARDS].any()
    assert c[O.SLICE_CARD_OPP_HEROES].tolist()==[0,7,0,0,0,0,0]
    assert not c[O.SLICE_CARD_OPP_BOARDS].any()


def test_public_context_needs_announced_identity_and_tracks_known_values_only():
    selected=SimpleNamespace(card_id='H',tags=dict(PLAYER_ID='1',CARDTYPE='HERO',ZONE='PLAY',
                                                  PLAYER_LEADERBOARD_PLACE='5'))
    own=SimpleNamespace(tags=dict(NEXT_OPPONENT_PLAYER_ID='1',BACON_BLOODGEMBUFFATKVALUE='3'))
    reader=SimpleNamespace(state=SimpleNamespace(entities={4:own,5:selected},players={8:4}),
                           encounter_turns={1:4},history_complete=True,combat_seen=True)
    got=observed_context(reader,dict(local_player_id=8,phase='recruit',shop=[{}]))
    assert got==dict(growth=[None]*8+[3,None],health_prices=[None],
                    encounter=dict(slot=0,hero='H'),last_seen_turn=4)
    own.tags.pop('NEXT_OPPONENT_PLAYER_ID')
    assert observed_context(reader,dict(local_player_id=8,phase='recruit'))['encounter'] is None
