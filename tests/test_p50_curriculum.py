"""P50校准不能用均值、不能修改原帧；课程必须可审计并保留源回合。"""
import copy,json,random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import pytest
from workflows.build_p50_bank import calibrated,scale_frame,select_rows
from workflows.report_firestone_learning import strength_quantiles


def frame(a,h,seed=1,turn=10,n=5):
    return dict(turn=turn,seat=2,source_game_seed=seed,player={'board':[dict(atk=a,hp=h,max_hp=h,peak_atk=a,peak_hp=h,card=i+1) for i in range(n)],'hand':[]})

def test_p50_is_not_mean_and_tail_is_bounded():
    rows=[frame(a,10,i) for i,a in enumerate([10,20,30,40,100000])];before=copy.deepcopy(rows)
    result=calibrated(rows,700)
    values=[sum(m['atk']+m['hp'] for m in f['player']['board']) for f in result]
    assert strength_quantiles(values)['p50_strength']==700
    assert max(values)<=2100 and min(values)>=175
    assert rows==before
    assert all(m['hp']>=1 and m['max_hp']>=m['hp'] for f in result for m in f['player']['board'])
    assert all(len(f['player']['board'])==5 for f in result)

def test_p50_of_sums_is_not_sum_of_marginal_medians():
    assert strength_quantiles([101,101,2])['p50_strength']==101

def test_zero_and_overflow_rejected():
    with pytest.raises(ValueError):calibrated([frame(1,1)],float('nan'))
    with pytest.raises(ValueError):scale_frame(frame(1,1),1e20,1e20)

def test_tail_source_is_counted_without_bot_fallback():
    from hsbg.az.train.strict_mirror import StrictLibrary
    from hsbg.az.train.mirror_snapshots import SnapshotFrame
    f=SnapshotFrame(turn=19,seat=0,player={'board':[]},source_turn=17)
    pool=SimpleNamespace(turns=lambda:list(range(1,20)),has=lambda t,s:s==0,get=lambda t,s,rng:f)
    library=StrictLibrary(pool);counts=Counter()
    assert library.sample(19,3,random.Random(1),counts) is f
    assert counts['tail_reserve']==1 and counts['cross_seat']==1
    assert library.sample(20,3,random.Random(1),counts) is f
    assert counts['tail_reserve']==2

def test_strict_calibrated_driver_gate():
    from hsbg.az.train.personal import require_calibrated_library
    good=dict(driver='bot_firestone_p50_3702',last_turn=19,validation={'accepted':20,'quarantined':0},smoke=False)
    require_calibrated_library(good)
    with pytest.raises(ValueError):require_calibrated_library(dict(good,smoke=True))

def test_schedule_and_config_remain_frozen():
    from workflows.run_7357_long import SCHEDULE
    from hsbg.az.train.personal import validate_config
    root=Path(__file__).resolve().parents[1]
    cfg=validate_config(json.loads((root/'config/p50-7357-train.json').read_text()))
    assert cfg['move-cap']==5 and cfg['max-actions-per-turn']==80
    assert SCHEDULE==[[0,'warm0'],[10000000,'warm1'],[20000000,'warm2'],[30000000,'train']]
    assert cfg['lr']==2e-6 and cfg['entropy-coef']==.002


def test_curriculum_boundaries_and_resume_do_not_restart_schedule():
    from workflows.run_sustained_learning import curriculum_segment
    from workflows.run_7357_long import SCHEDULE
    paths={k:k for k in ('train','warm0','warm1','warm2')};base=558873732
    assert curriculum_segment(base,base,base+30000000,paths,SCHEDULE)==('warm0',base+10000000)
    assert curriculum_segment(base+10001000,base,base+30000000,paths,SCHEDULE)==('warm1',base+20000000)
    assert curriculum_segment(base+30001000,base,base+60000000,paths,SCHEDULE)==('train',base+60000000)
    with pytest.raises(ValueError):curriculum_segment(base,base,base+1,paths,[[1,'warm0']])


def test_explicit_library_transition_preserves_checkpoint(tmp_path,monkeypatch):
    from hsbg.az.train import personal,runlog
    out=tmp_path/'train';snap=out/'snapshots/step_007358';snap.mkdir(parents=True)
    for name in ('model.pt','trainer.pt'):(snap/name).write_bytes(name.encode())
    cfg={'games':1,'minutes':1,'processes':1};config=tmp_path/'config.json';config.write_text(json.dumps(cfg))
    previous=dict(fingerprint={'engine':'same'},library_digest='old',config=cfg,training_source_sha256='code')
    (out/'provenance.json').write_text(json.dumps(previous));(out/'latest.json').write_text(json.dumps({'version':7358,'cum_decisions':568873732}))
    monkeypatch.setattr(personal,'checked_library',lambda _:dict(split='train',digest='new'))
    monkeypatch.setattr(personal,'fingerprint',lambda:{'engine':'same'})
    monkeypatch.setattr(personal,'training_source_digest',lambda:'code')
    monkeypatch.setattr(personal,'inspect_checkpoint',lambda *a:{'passed':True})
    monkeypatch.setattr(runlog,'recover_committed_record',lambda *a:None)
    commands=[]
    def execute(command):
        commands.append(command)
        for n in ('model.pt','trainer.pt','history.jsonl'):(out/n).write_bytes(b'committed')
    monkeypatch.setattr(personal,'run_command',execute)
    args=SimpleNamespace(config=config,library='new',out=out,checkpoint=snap/'model.pt',device='cpu',resume=True,library_transition_from='')
    with pytest.raises(ValueError,match='显式'):personal.train(args)
    args.library_transition_from='old';personal.train(args)
    changed=json.loads((out/'provenance.json').read_text())
    assert changed['library_transitions']==[dict(from_digest='old',to_digest='new',version=7358,cum_decisions=568873732)]
    assert '--trainer-state' in commands[0] and str(snap/'trainer.pt') in commands[0]
    assert (snap/'trainer.pt').read_bytes()==b'trainer.pt'
    assert changed['config']==cfg
