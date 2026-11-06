"""实机入口：默认仅检查接口，必须显式--enable-input才操作游戏。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--game-dir',type=Path,default=Path(os.environ.get('HEARTHSTONE_DIR','Hearthstone')))
 p.add_argument('--checkpoint',type=Path,help='仅接受旧候选动作模型；不是245动作HSRL checkpoint')
 p.add_argument('--enable-input',action='store_true')
 p.add_argument('--out',type=Path)
 p.add_argument('--seconds',type=int,default=2400)
 a=p.parse_args()
 if not a.enable_input:
  from bg_agent.model import Action,Observation
  from legacy_policy.actor import LiveActorPolicy
  print(json.dumps(dict(ready=True,input_enabled=False,client_interface=Action.__module__,
    legacy_actor_interface=LiveActorPolicy.__module__,current_245_action_model_bridge=False),ensure_ascii=False))
  return 0
 from bg_agent.live_lease import LiveLease
 from bg_agent.runtime import AuditLog
 from bg_agent.live_session import run_session
 from legacy_policy.actor import LiveActorPolicy
 a.policy='heuristic';a.dark_gifts_per_turn=1;a.resume=None
 a.ui_profile=None;a.hand_profile=None;a.geometry_mode='geometry';a.layout_catalog=None;a.local_layout=None
 a.decision_ahead=False;a.interaction_speed='safe';a.gesture_clock='precise'
 a.out=a.out or ROOT.parent/'artifacts/runs'/('client-'+time.strftime('%Y%m%d-%H%M%S'))
 a.out.mkdir(parents=True,exist_ok=False)
 factory={}
 if a.checkpoint:
  policy=LiveActorPolicy(a.checkpoint)
  factory['policy']=lambda *args,**kwargs:policy
 with LiveLease(a.game_dir):
  return run_session(a,a.out,AuditLog(a.out/'agent.jsonl'),factories=factory)

if __name__=='__main__':raise SystemExit(main())
