import copy,gzip,json
from types import SimpleNamespace
from hsbg.runtime.artifacts import atomic_json,file_identity
from tests.test_economic_tasks import natural_source
from tools.probe_economic_starts import run_turn,canonical_items,select_sources
from workflows.build_recruit_curriculum import endow
from tools.probe_economic_starts import ProtectCoresView
import hsbg_engine as engine


class Script:
    def __init__(self,actions):self.actions=actions
    def reset(self):self.index=0
    def decide(self,game,seat):
        a=self.actions[self.index];self.index+=1
        return SimpleNamespace(action=a)


def test_end_turn_intent_never_resets_gold_or_runs_combat():
    state,_=endow(natural_source(11800001));original=copy.deepcopy(state)
    metrics,trace=run_turn(Script([0]),state,canonical_items())
    assert metrics['actions']==0 and metrics['gross_gold_gains']==0
    assert metrics['gold_start']==metrics['gold_end'] and metrics['all_cores_retained']
    assert trace['final']==original and state==original
    assert trace['end_intent']['executed'] is False


def test_sales_are_logged_as_income_not_strength_progress():
    state,_=endow(natural_source(11800001))
    # Endowed cores are the final three slots; sell Brann using its actual slot.
    slot=len(state['players'][0]['board'])-3
    metrics,trace=run_turn(Script([16+slot,0]),state,canonical_items())
    assert metrics['core_sells']==1 and metrics['cores_retained']==2
    assert metrics['gross_gold_gains']==1 and metrics['gold_spent']==0
    assert metrics['strength_delta']<0 and len(trace['steps'])==1


def test_diagnostic_only_uses_source_game_holdout(tmp_path):
    rows=[]
    for seed in (11800000,11800001):
        rows.append({'seed':seed,'seat':0,'turn':8,'state':natural_source(seed)})
    with gzip.open(tmp_path/'states.jsonl.gz','wt') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    atomic_json(tmp_path/'manifest.json',{'data':file_identity(tmp_path/'states.jsonl.gz')})
    assert [r['seed'] for r in select_sources(tmp_path)]==[11800000]
    import pytest
    with pytest.raises(ValueError,match='assessment-only'):select_sources(tmp_path,source_split='all_assessment')
    atomic_json(tmp_path/'manifest.json',{'data':file_identity(tmp_path/'states.jsonl.gz'),'assessment_only':True},replace=True)
    assert [r['seed'] for r in select_sources(tmp_path,source_split='all_assessment')]==[11800000,11800001]
    from workflows.build_recruit_curriculum import build
    with pytest.raises(ValueError,match='cannot produce training'):build(tmp_path,tmp_path/'forbidden-training')


def test_core_constraint_changes_only_policy_sale_mask():
    state,_=endow(natural_source(11800001))
    game=engine.Game.from_json(json.dumps(state));before=game.state_json()
    native=bytes(game.legal_mask(0));filtered=ProtectCoresView(game,canonical_items()).legal_mask(0)
    changed=[i for i,(a,b) in enumerate(zip(native,filtered)) if a!=b]
    end=len(state['players'][0]['board'])
    assert changed==list(range(16+end-3,16+end))
    assert game.state_json()==before and bytes(game.legal_mask(0))==native


def test_collection_namespaces_are_explicit_disjoint_and_bounded():
    import pytest
    from workflows.collect_recruit_states import validate_source_namespace
    validate_source_namespace(11800000,10000)
    validate_source_namespace(12602000,128,True)
    validate_source_namespace(12602112,16,True)
    for seed,games,assessment in ((12602000,128,False),(11800000,16,True),
                                  (12602000,129,True),(12602000,0,True)):
        with pytest.raises(ValueError,match='namespace'):
            validate_source_namespace(seed,games,assessment)
