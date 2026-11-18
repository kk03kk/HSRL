import gzip,json
from pathlib import Path
import numpy as np
import hsbg_engine as E
from hsbg.az.train import mechanism_obs as M
from hsbg.az.train.conditional_economy import current_gate_spec
from hsbg.az.train.public_turn_history import PublicTurnHistory
from hsbg.az.train.research_economy_policy import EconomicOptionPolicy
from hsbg.runtime.artifacts import file_identity
from workflows.build_payoff_dataset import label
from workflows.annotate_payoff_history import annotate
from workflows.probe_whole_turn_economy import fixture
from tests.test_economic_tasks import natural_source
from tools.prove_play_history_alias import public


def test_history_sidecar_matches_online_public_views_without_changing_labels(tmp_path):
    game=E.Game.from_json(json.dumps(fixture(natural_source(),'complete_gold')[0]))
    def fallback(features,cards,mask,seat):
        p=np.asarray(mask,dtype=np.float32);return p/p.sum(),0.
    policy=EconomicOptionPolicy(fallback,operator_mode='growth');history=PublicTurnHistory()
    seed=11802001;root=tmp_path/'original';dest=root/'sources'/str(seed);dest.mkdir(parents=True)
    path=dest/'decisions.jsonl.gz';labels=[];expected=[];counts=[]
    with gzip.open(path,'wt',encoding='utf-8') as stream:
        for index in range(101):
            before_state=json.loads(game.state_json());before=public(game);marks,count=history.before(before)
            decision=policy.decide(game,0);f,c=M.observe(game,0,M.SCHEMA)
            row=label(np.frombuffer(f,dtype='<f4'),np.frombuffer(c,dtype='<i4'),policy.evaluator.last_mask,decision,game.turn)
            if row is not None:labels.append(row);expected.append(marks);counts.append(count)
            game.step(0,decision.action);history.after(before,public(game),decision.action)
            stream.write(json.dumps({'event':'action','seat':0,'turn':before['turn'],'action_index':index,
                'action':decision.action,'state_before':before_state,'state_after':json.loads(game.state_json()),
                'policy_info':decision.info})+'\n')
            if decision.action==0:break
    n=len(labels)
    np.savez_compressed(dest/'labels.npz',targets=np.stack([r[3] for r in labels]),turns=[r[5] for r in labels],
        reasons=[r[6] for r in labels],seeds=np.full(n,seed),dev=np.zeros(n,dtype=bool))
    original=file_identity(dest/'labels.npz')
    task={'seed':seed,'split':'train','labels':n,'labels_file':original,
        'evidence':{'trace':{**file_identity(path),'actions':index+1}}}
    out=tmp_path/'annotations';result=annotate((root,task,out,current_gate_spec('gold_engine')['embedding_ids']))
    assert result['rows']==n and result['actions']==index+1
    with np.load(out/'sources'/str(seed)/'history.npz') as data:
        assert np.array_equal(data['played_by_slot'],np.stack(expected))
        assert data['actions_so_far'].tolist()==counts
    assert file_identity(dest/'labels.npz')==original and not any(game.diagnostic_counts())
