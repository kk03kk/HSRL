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
