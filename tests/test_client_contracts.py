"""离线检查实机组件边界，不连接游戏、不启用输入。"""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_client_dry_run_never_enables_input(tmp_path):
 r=subprocess.run([sys.executable,str(ROOT/'client/run.py'),'--out',str(tmp_path/'not-created')],capture_output=True,text=True,encoding='utf-8')
 assert r.returncode==0,r.stderr
 assert json.loads(r.stdout)['input_enabled'] is False
 assert not (tmp_path/'not-created').exists()

def test_client_resources_and_legacy_contract_are_separate():
 from bg_agent.resources import calibration_root
 from legacy_policy.spec import NUM_ACTION_KINDS
 from hsbg.az.train.common import N_ACTIONS
 assert calibration_root().is_relative_to(ROOT/'client')
 assert list(calibration_root().rglob('*.json'))
 assert NUM_ACTION_KINDS==13 and N_ACTIONS==245
