import pytest
from hsbg.az.train.runlog import aggregate_injection


def test_global_median_is_computed_from_games_not_shard_medians():
    parts = [dict(learner_final_str_p50=1, learner_final_str_samples=[0,1,2]),
             dict(learner_final_str_p50=100, learner_final_str_samples=[100])]
    result = aggregate_injection(parts)
    assert result['learner_final_str_p50'] == 1.5  # not (1+100)/2
    assert result['learner_final_str_mean'] == 25.75


def test_legacy_shard_medians_are_not_fabricated_into_global_p50():
    result = aggregate_injection([{'learner_final_str_p50': 1}, {'learner_final_str_p50': 100}])
    assert result['learner_final_str_p50'] is None


def test_survivor_means_use_survivor_counts():
    result = aggregate_injection([{'tel_n_12': 1, 'tel_tier_12': 2},
                                  {'tel_n_12': 9, 'tel_tier_12': 6}])
    assert result['tel_tier_12'] == pytest.approx(5.6)


def test_nested_copy_of_counter_is_not_counted_twice():
    assert aggregate_injection([{'mirror_attempt': 3, 'collect': {'mirror_attempt': 3}}])['mirror_attempt'] == 3
