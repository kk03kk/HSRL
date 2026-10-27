import importlib.util
import json
from pathlib import Path

import numpy as np
import taverngym as tg


def diagnostic():
    p=Path(__file__).resolve().parents[1]/'tools/diagnose_pilot.py'
    spec=importlib.util.spec_from_file_location('pilot_diagnosis_tool',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_replays_real_native_guard_without_disabling_it(tmp_path):
    tool=diagnostic()
    e=tg.BatchEnv(1,seed=91,controlled=range(8),config={'max_combat_attacks':1})
    e.auto_reset=False
    initial=json.loads(e.save(0));actions=[]
    for _ in range(3000):
        a=e.bot_actions();actions.append(a.tolist());e.step(a)
        if any(e.diagnostics()[0]):break
    else:raise AssertionError('test fixture never triggered the deliberately tiny combat guard')
    assert e.diagnostics()[0][1]>0
    f=tmp_path/'failure-test.json'
    f.write_text(json.dumps({'format':2,'native':tg.build_info(),'initial':initial,
        'post':json.loads(e.save(0)),'actions':actions,'diagnostic':e.diagnostics()[0]}))
    before=f.read_bytes();out=tmp_path/'audit';out.mkdir()
    r=tool.replay_failure(f,out)
    assert r['guard_reproduced'] and r['post_matches_saved'] and r['stopped_at_expected_row']
    assert f.read_bytes()==before


def test_behavior_measures_effective_actions_and_does_not_label_ends_as_failures():
    tool=diagnostic();l=dict(tg.layout())
    f=np.zeros(tg.OBS_FLOATS,dtype=np.float32);f[0]=.1;f[7]=.6;f[10]=1/3
    mask=np.ones(tg.N_ACTIONS,dtype=bool)
    result=tool.behavior([{'rank':8,'samples':[{'action':0,'obs':{'features':f,'mask':mask}}]}])
    assert result['actions']=={'EndTurn':1}
    assert result['end_turn_states'][0]['gold']==6
    assert result['end_turn_states'][0]['buy_available']


def test_diagnostic_submission_has_no_build_or_training_and_archives_evidence():
    s=(Path(__file__).resolve().parents[1]/'deploy/nus/diagnose_first_model.sh').read_text()
    assert '--partition=normal' in s and '--gres=' not in s
    assert 'python -m taverngym.rl.train' not in s and 'cargo build' not in s
    assert 'first-model-evidence.tar.gz' in s
    assert 'afterany' not in s


def test_actual_bundle_code_keeps_models_failure_and_checksums_without_modifying_source(tmp_path):
    import subprocess,sys,hashlib,tarfile,os
    root=Path(__file__).resolve().parents[1]
    text=(root/'deploy/nus/diagnose_first_model.sh').read_text()
    code=text.split("<<'BUNDLE'\n",1)[1].split('\nBUNDLE',1)[0]
    run=tmp_path/'training';audit=tmp_path/'audit';run.mkdir();audit.mkdir()
    (run/'latest.pt').write_bytes(b'checkpoint-fixture-not-a-real-model')
    (run/'failure-example.json').write_text('{}')
    (audit/'report.json').write_text('{}')
    before=(run/'latest.pt').read_bytes()
    subprocess.run([sys.executable,'-c',code,str(run),str(audit)],check=True,capture_output=True,
                   env={**os.environ,'USERPROFILE':str(tmp_path),'HOME':str(tmp_path)})
    archive=audit/'first-model-evidence.tar.gz'
    assert archive.with_name(archive.name+'.sha256').read_text().split()[0]==hashlib.sha256(archive.read_bytes()).hexdigest()
    with tarfile.open(archive) as t:
        assert {'training/latest.pt','training/failure-example.json','audit/report.json'} <= set(t.getnames())
    assert before==(run/'latest.pt').read_bytes()
    assert (tmp_path/'AutoTavern-first-model-evidence-1023992.tar.gz').read_bytes()==archive.read_bytes()
