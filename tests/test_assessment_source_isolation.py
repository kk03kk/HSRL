import importlib,json,sys
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('module',[
    'build_recruit_curriculum','build_economic_dataset','build_cycle_dataset',
    'build_whole_turn_dataset','build_payoff_dataset','build_payoff_queries'])
def test_all_training_builders_reject_assessment_before_reading_states(tmp_path,monkeypatch,module):
    bank=tmp_path/'bank';bank.mkdir()
    (bank/'manifest.json').write_text(json.dumps({'assessment_only':True}),encoding='utf-8')
    out=tmp_path/'out';builder=importlib.import_module('workflows.'+module)
    args=SimpleNamespace(bank=str(bank),out=str(out))
    if module=='build_payoff_queries':
        original=tmp_path/'original';original.mkdir()
        (original/'manifest.json').write_text('{}',encoding='utf-8')
        plan=tmp_path/'plan.json'
        plan.write_text(json.dumps({'bank':str(bank),'original_dataset':str(original)}),encoding='utf-8')
        args.plan=str(plan)
    with pytest.raises(ValueError,match='assessment-only bank cannot produce training data'):
        if module=='build_recruit_curriculum':builder.build(bank,out)
        elif module in ('build_economic_dataset','build_cycle_dataset'):
            argv=['builder','--bank',str(bank),'--out',str(out)]
            if module=='build_cycle_dataset':argv+=['--checkpoint',str(tmp_path/'never-read.pt')]
            monkeypatch.setattr(sys,'argv',argv)
            builder.main()
        else:builder.main(args)
    assert not list(out.rglob('*.npz'))
