"""离线原生观测对照；不证明真实游戏日志完整，也不调用Windows输入。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'client'))
import numpy as np
import pytest
from tests.live_fixtures import native_view,game_to_turn,vocabulary
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
 from hsrl_agent.live.cli import parser
 root=Path(__file__).resolve().parents[1]
 a=parser(root).parse_args(['doctor'])
 assert Path(a.checkpoint)==root/'artifacts/models/nus-1080772/step-006039-210m/model.pt'

def test_old_profile_is_rejected_before_backend_creation():
 from hsrl_agent.live.selection import validate_live_profile
 root=Path(__file__).resolve().parents[1]
 model=root/'artifacts/models/nus-1080772/step-006039-210m/model.pt'
 if not model.exists():pytest.skip('本地归档权重不入Git')
 with pytest.raises(Blocked,match='256589'):
  validate_live_profile(model,dict(review_state='reviewed',allowed_builds=[255790]))
