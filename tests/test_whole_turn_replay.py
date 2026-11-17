import json
import numpy as np
import pytest
from hsbg.runtime.artifacts import atomic_json,file_identity
from tools.export_whole_turn_replay import export


def test_only_train_sources_enter_replay_and_no_fake_outcomes(tmp_path):
    source=tmp_path/'source';source.mkdir()
    mask=np.zeros((3,245),bool);mask[:,[0,1,8]]=True
    targets=np.zeros_like(mask);targets[np.arange(3),[0,1,8]]=True
    np.savez_compressed(source/'labels.npz',features=np.zeros((3,3282),np.float32),
        cards=np.zeros((3,124),np.int16),masks=mask,targets=targets,
        seeds=[10,11,12],dev=[True,False,False],turns=[8,8,9],kinds=['normal']*3)
    atomic_json(source/'manifest.json',{'schema':'hsai.whole-turn-process-dataset/v1',
        'files':{'labels.npz':file_identity(source/'labels.npz')}})
    result=export(source,tmp_path/'out')
    assert result['rows']==2 and result['source_seeds']==[11,12]
    assert not result['outcome_auxiliary_labels']
    rows=[json.loads(line) for line in (tmp_path/'out/process.jsonl').read_text().splitlines()]
    assert all(not row['label']['available'] for row in rows)
    assert [r['ref']['actions'] for r in rows]==[[1],[8]]
    from hsbg.az.train.board_problems import load_problem_pool
    pool=load_problem_pool(str(tmp_path/'out/process.jsonl'))
    assert np.isfinite(pool.norm['std']) and not pool.aux_valid.any()
    with (source/'labels.npz').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='identity'):export(source,tmp_path/'bad')
