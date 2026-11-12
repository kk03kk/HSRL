"""离线原生观测对照；不证明真实游戏日志完整，也不调用Windows输入。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'client'))
import numpy as np
import pytest
from tests.live_fixtures import native_view,game_to_turn,vocabulary,snapshot
from hsrl_agent.live.contracts import Blocked
from hsrl_agent.live.encoding import encode

@pytest.mark.parametrize('turn,seed',[(0,91),(1,91),(2,91),(3,91),(2,17),(2,71)])
def test_encoder_remains_equal_to_current_native_observation(turn,seed):
 import hsbg_engine as e
 game=e.Game(seed=seed) if not turn else game_to_turn(turn,seed)
 f,c=encode(native_view(game),vocabulary());raw=game.observe(0)
 np.testing.assert_array_equal(f,np.frombuffer(raw[0],np.float32))
 np.testing.assert_array_equal(c,np.frombuffer(raw[1],np.int32))

def test_model_name_never_bypasses_pinned_identity(tmp_path):
 from hsrl_agent.live.policy import ModelPolicy
 p=tmp_path/'4159.pt';p.write_bytes(b'wrong weights')
 with pytest.raises(Blocked,match='approved pinned'):ModelPolicy(p)

def test_missing_model_does_not_fall_back_to_old_client():
 """CLI 默认 checkpoint 必须来自 config/live/model.json，不能回落到旧客户端的权重。"""
 import json
 from hsrl_agent.live.cli import parser
 root=Path(__file__).resolve().parents[1]
 config=json.loads((root/'config/live/model.json').read_text(encoding='utf8'))
 a=parser(root).parse_args(['doctor'])
 assert Path(a.checkpoint)==root/config['checkpoint']

def test_old_profile_is_rejected_before_backend_creation():
 from hsrl_agent.live.selection import validate_live_profile
 root=Path(__file__).resolve().parents[1]
 model=root/'artifacts/models/nus-1080772/step-006039-210m/model.pt'
 if not model.exists():pytest.skip('本地归档权重不入Git')
 with pytest.raises(Blocked,match='256589'):
  validate_live_profile(model,dict(review_state='reviewed',allowed_builds=[255790]))


def test_build_number_type_never_silently_mismatches(tmp_path):
 """profile JSON stores builds as strings, POLICY_SPECS as ints (same build must match)."""
 import hashlib
 from hsrl_agent.live.selection import validate_live_profile
 from hsrl_agent.live.policy import POLICY_SPECS
 probe=tmp_path/'probe.pt';probe.write_bytes(b'probe weights')
 sha=hashlib.sha256(probe.read_bytes()).hexdigest()
 POLICY_SPECS[sha]=dict(id='probe',auxiliary=False,required_client_build=255790)
 try:
  assert validate_live_profile(probe,dict(review_state='reviewed',allowed_builds=['255790']))['id']=='probe'
  with pytest.raises(Blocked,match='255790'):
   validate_live_profile(probe,dict(review_state='reviewed',allowed_builds=['256589']))
  with pytest.raises(Blocked,match='reviewed'):
   validate_live_profile(probe,dict(review_state='reference_only',allowed_builds=['255790']))
 finally:
  POLICY_SPECS.pop(sha,None)


def test_v1_1_is_the_selected_pinned_model():
 """当前选中的 pin 必须与 config/live/model.json 一致，且必须是已登记的策略。"""
 import hashlib,json
 from hsrl_agent.live.policy import POLICY_SPECS
 from hsrl_agent.live.selection import selected_checkpoint,validate_live_profile
 root=Path(__file__).resolve().parents[1]
 config=json.loads((root/'config/live/model.json').read_text(encoding='utf8'))
 checkpoint=selected_checkpoint(root)
 if not checkpoint.exists():pytest.skip('选中权重为本地产物，不入Git')
 sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
 assert config['sha256']==sha
 assert config['checkpoint']==str(checkpoint.relative_to(root)).replace('\\','/')
 spec=POLICY_SPECS[sha]
 assert spec['id']==config['model_id']
 assert spec['parameters']==config.get('parameters',spec['parameters'])
 assert spec['auxiliary'] is False
 assert spec.get('trained_engine_version',config.get('trained_game_version'))==config['trained_game_version']
 profile=json.loads((root/'config/live/autoplay-255790.json').read_text(encoding='utf8'))
 assert validate_live_profile(checkpoint,profile)['id']==config['model_id']


def test_live_cancel_matches_v1_1_forbid_cancel_rule():
 """V1.1 训练时 forbid_cancel=1，实机不得把 Cancel(7) 放进正常掩码。"""
 from hsrl_agent.live.actions import bind_actions
 s=snapshot()
 s['client'].update(
  interaction=dict(min_select=1,max_select=1,mode=None,cancel_authorized=True,request_id=1),
  choices=[dict(entity_id=41,card_id='BG20_100',card_type='MINION',tags={})])
 bindings=bind_actions(s)
 assert 7 not in {b.action for b in bindings}
 assert [b.kind for b in bindings]==['choose'] and bindings[0].action==30
 # 四个英雄位全部锁定、只剩取消可授权时，必须保留 7，否则空掩码会死锁。
 locked=dict(BACON_LOCKED_MULLIGAN_HERO='1',BACON_HERO_CAN_BE_DRAFTED='1')
 s['client'].update(interaction=dict(min_select=1,max_select=1,mode=None,cancel_authorized=True,request_id=2),
  choices=[dict(entity_id=100+i,card_id='HERO_01',card_type='HERO',tags=locked) for i in range(4)])
 bindings=bind_actions(s)
 assert [b.action for b in bindings]==[7]


def test_lobby_accepts_start_and_start_match_labels():
 """255790 实测按钮 OCR 为「开始」；「开始匹配」是同一控件的另一种文案。"""
 from hsrl_agent.live.lobby import LobbyFlow
 rows=[dict(text='酒馆战棋',confidence=.99,point=[.37,.02]),
       dict(text='开始匹配',confidence=.99,point=[.785,.79])]
 flow=LobbyFlow(max_restarts=1)
 assert flow.inspect(rows,'token-1',100.0) is None          # 双帧证明的第一帧
 action=flow.inspect(rows,'token-1',100.4)
 assert action is not None and action.kind=='start' and action.point==(.785,.79)


def test_click_control_is_never_authorised_by_a_stale_frame():
 """点击前必须在新鲜帧上重新识别同一控件（authorising pass 已过期 0.3-2.8 s）。"""
 from hsrl_agent.live.startup import control_texts
 from hsrl_agent.live.lobby import STARTS
 assert control_texts('start') is STARTS and '开始' in STARTS and '开始匹配' in STARTS
 for kind in ('enter_lobby','dismiss_error','dismiss_result'):
  assert control_texts(kind)


def test_v1_1_decides_against_pinned_weights_not_a_fallback():
 """Real forward: the decision must come from V1.1's own tensors."""
 from hsrl_agent.live.actions import bind_actions
 from hsrl_agent.live.policy import ModelPolicy
 root=Path(__file__).resolve().parents[1]
 checkpoint=root/'artifacts/models/v1.1/model.pt'
 if not checkpoint.exists():pytest.skip('V1.1 权重为本地产物，不入Git')
 policy=ModelPolicy(checkpoint)
 assert policy.policy_id=='hsai-v1.1'
 assert sum(p.numel() for p in policy.model.parameters())==16436034
 assert policy.model.training is False
 s=snapshot();bindings=bind_actions(s)
 d=policy.decide(s,bindings)
 assert d.binding in bindings and 0<=d.action<245
 assert d.checkpoint_sha256==policy.sha
 # Only EndTurn is authorized on an empty recruiting snapshot.
 assert d.action==0 and d.binding.kind=='end'
