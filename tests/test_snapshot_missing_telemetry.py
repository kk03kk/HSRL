import copy,json,random
from hsbg.az.train.mirror_snapshots import MirrorStats,inject_turn
from hsbg.az.train.runlog import slim_collect

class MissingPool:
    def get(self,turn,seat,rng=None):return None

class Game:
    def __init__(self,alive):
        self.state={'seed':123,'players':[{'alive':alive,'health':12 if alive else -3}]}
    def state_json(self):return json.dumps(self.state)

def test_missing_telemetry_classifies_and_does_not_change_rng_or_game():
    combined=MirrorStats()
    for alive in (True,False):
        game=Game(alive);before=copy.deepcopy(game.state)
        rng=random.Random(871);prior=rng.getstate()
        stats=inject_turn(game,MissingPool(),21,seats=[1,2],rng=rng,skip_inactive=False)
        assert stats.missing==2 and stats.fallback==2
        assert stats.missing_learner_alive==2*alive
        assert stats.missing_learner_dead==2*(not alive)
        assert stats.missing_learner_unknown==0
        assert game.state==before and rng.getstate()==prior
        combined.add(stats)
    meta=combined.to_meta()
    assert meta['mirror_missing']==4
    assert len(meta['mirror_missing_observations'])==4
    assert slim_collect(meta)['mirror_missing_learner_alive']==2
    merged=slim_collect({'worker_meta':[{'collect':meta},{'collect':meta}]})
    assert merged['mirror_missing_learner_alive']==4
    assert merged['mirror_missing_learner_dead']==4
    assert len(merged['mirror_missing_observations'])==8

def test_missing_telemetry_keeps_unknown_explicit():
    stats=inject_turn(object(),MissingPool(),21,seats=[1],skip_inactive=False)
    assert stats.missing==stats.missing_learner_unknown==1
    assert stats.missing_learner_alive==stats.missing_learner_dead==0
    malformed=Game(True);del malformed.state['seed']
    stats=inject_turn(malformed,MissingPool(),21,seats=[1],skip_inactive=False)
    assert stats.missing==stats.missing_learner_unknown==1
    assert stats.missing_learner_alive==0
