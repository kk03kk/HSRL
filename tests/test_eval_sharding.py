import pytest
from workflows.evaluate_suite import shard_ranges


@pytest.mark.parametrize('games,shards', [(512,8),(1031,8),(2,2),(1,1)])
def test_sharding_covers_each_planned_seed_once(games, shards):
    result = shard_ranges(games, shards, 930000)
    seeds = [seed+i for seed,count in result for i in range(count)]
    assert seeds == list(range(930000,930000+games))
    assert max(n for _,n in result)-min(n for _,n in result) <= 1


@pytest.mark.parametrize('games,shards', [(0,8),(8,0),(2,3),(-1,1)])
def test_invalid_sharding_is_rejected(games, shards):
    with pytest.raises(ValueError): shard_ranges(games,shards,0)


def test_legacy_entry_preserves_explicit_scientific_options():
    from workflows.evaluate_legacy import plan_from_environment
    env = {'HSAI_GAP_MODELS':'old=a.pt;new=b.pt', 'HSAI_GAP_MIRROR_SNAPS':'library',
           'HSAI_GAP_GAMES':'1031', 'HSAI_GAP_FREEZE_CAP':'0',
           'HSAI_GAP_MIRROR_STRICT':'1', 'HSAI_GAP_LOBBY_EXCLUDE':'beast',
           'HSAI_GAP_FORCED_TIER':'2:2'}
    plan = plan_from_environment(env)
    assert plan['games'] == 1031
    assert plan['models'][1] == {'name':'new','checkpoint':'b.pt'}
    assert plan['evaluation']['freeze-cap'] == 0
    assert plan['evaluation']['mirror-strict'] == 1
    assert plan['evaluation']['lobby-exclude'] == 'beast'
    assert plan['evaluation']['forced-tier-plan'] == '2:2'
