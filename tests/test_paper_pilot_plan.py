import json
from pathlib import Path
import subprocess
import sys


def test_actual_pilot_generator_equal_budget_and_distinct_seeds(tmp_path):
    root=Path(__file__).resolve().parents[1]
    script=(root/'deploy/nus/paper_pilot.sh').read_text(encoding='utf-8')
    generator=script.split("<<'PILOT_CONFIG'\n",1)[1].split('\nPILOT_CONFIG',1)[0]
    target=tmp_path/'pilot'
    subprocess.run([sys.executable,'-c',generator,str(target)],cwd=root,check=True,capture_output=True)
    plan=json.loads((target/'plan.json').read_text())
    assert len(plan['runs'])==4
    configs=[json.loads(Path(r['config']).read_text()) for r in plan['runs']]
    assert {c['seed'] for c in configs}=={701,1701}
    assert {c['policy_kind'] for c in configs}=={'compact_gru','compact_ff'}
    assert all(c['decision_budget']==200000 and c['bc_waves']==16 for c in configs)
    assert len({r['run'] for r in plan['runs']})==4
    assert all(c['opponent_mode']=='bots' for c in configs)
    wrapper=(target/'checked-train.sbatch').read_text()
    assert "['ppo_decisions']>=200000" in wrapper
    assert 'independent-bc.json' in wrapper and 'independent-final.json' in wrapper
