import gzip,json
from types import SimpleNamespace
import pytest
from hsbg.az.train.policies import Decision,Policy
from hsbg.az.train.evaluate import EvalConfig,evaluate_policy
from hsbg.evaluation.trace import DecisionRecorder
from workflows.replay_snapshot_tails import Replay

class EndTurn(Policy):
    def decide(self,game,seat):
        mask=bytes(game.legal_mask(seat))
        return Decision(0 if mask[0] else next(i for i,x in enumerate(mask) if x))

def test_complete_native_sealed_action_replay(tmp_path):
    pytest.importorskip('hsbg_engine')
    path=tmp_path/'trace.gz';recorder=DecisionRecorder(path,identity={'diagnostic_only':True})
    original=evaluate_policy(EndTurn(),EvalConfig(games=1,seed=76123,decision_recorder=recorder))
    recorder.finish(original)
    with gzip.open(path,'rt',encoding='utf-8') as stream:rows=[json.loads(x) for x in stream]
    start=next(x for x in rows if x['event']=='game_start')
    final=next(x for x in rows if x['event']=='game_end')
    actions=[x for x in rows if x['event']=='action']
    replay=Replay(start,actions,final,SimpleNamespace(game_ref=None))
    actual=evaluate_policy(replay,EvalConfig(games=1,seed=76123,decision_recorder=replay))
    assert actual['rank_sequence']==original['rank_sequence'] and replay.i==len(actions)
    actions[0]['state_before']['players'][0]['gold']+=1
    bad=Replay(start,actions,final,SimpleNamespace(game_ref=None))
    with pytest.raises(ValueError,match='input differs'):
        evaluate_policy(bad,EvalConfig(games=1,seed=76123,decision_recorder=bad))
