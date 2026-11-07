"""预检的强制升本门禁：样本比例/到达率/未达比例必须真的生效（不能因缺字段静默跳过）。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from workflows.run_7357_preflight import check_forced_tier


def _train(tmp_path, *, five=None, forced_actions=4, plan='2:2,5:3,6:4,8:5'):
    train = tmp_path / 'train'
    train.mkdir(parents=True, exist_ok=True)
    row = dict(iter=7357, forced_tier=dict(
        plan=plan, prob=1.0, teacher=False, forced_actions=forced_actions,
        teacher_labels=0, blocked=0,
        targets={'5': five or {'on_time': 0, 'deferred': 0, 'missed': 0, 'self': 0}}))
    header = dict(type='header', args={})
    (train / 'history.jsonl').write_text(
        json.dumps(header) + '\n' + json.dumps(row) + '\n', encoding='utf-8')
    return train


def test_passes_when_samples_and_reach_are_fine(tmp_path):
    train = _train(tmp_path, five={'on_time': 360, 'deferred': 40, 'missed': 0, 'self': 0})
    c = check_forced_tier(train, 7360, games=512)
    assert c['ok'] is True
    assert c['tier5_on_time'] == pytest.approx(0.9)


def test_fails_when_no_forced_action(tmp_path):
    train = _train(tmp_path, forced_actions=0,
                   five={'on_time': 400, 'deferred': 0, 'missed': 0, 'self': 0})
    c = check_forced_tier(train, 7360, games=512)
    assert c['ok'] is False and '一次强制都没执行' in c['reason']


def test_fails_when_sample_ratio_too_small(tmp_path):
    """多数局在 T8 前结束 -> 5 本样本不足 -> 必须失败（这正是训练场过难的信号）。"""
    train = _train(tmp_path, five={'on_time': 4, 'deferred': 0, 'missed': 0, 'self': 0})
    c = check_forced_tier(train, 7360, games=512)
    assert c['ok'] is False and '样本过少' in c['reason']


def test_fails_when_games_unknown(tmp_path):
    """局数拿不到时不能静默跳过比例检查。"""
    train = _train(tmp_path, five={'on_time': 400, 'deferred': 0, 'missed': 0, 'self': 0})
    c = check_forced_tier(train, 7360, games=0)
    assert c['ok'] is False and '无法确定本轮局数' in c['reason']


def test_fails_when_reach_too_low(tmp_path):
    train = _train(tmp_path, five={'on_time': 100, 'deferred': 20, 'missed': 380, 'self': 0})
    c = check_forced_tier(train, 7360, games=512)
    assert c['ok'] is False and '到达率过低' in c['reason']


def test_fails_when_missed_too_high(tmp_path):
    train = _train(tmp_path, five={'on_time': 300, 'deferred': 60, 'missed': 152, 'self': 0})
    c = check_forced_tier(train, 7360, games=512)
    assert c['ok'] is False and ('未达比例过高' in c['reason'] or '到达率过低' in c['reason'])


def test_fails_without_forced_tier_section(tmp_path):
    train = tmp_path / 'train'
    train.mkdir()
    (train / 'history.jsonl').write_text(
        json.dumps(dict(type='header', args={})) + '\n' + json.dumps(dict(iter=1)) + '\n',
        encoding='utf-8')
    c = check_forced_tier(train, 1, games=512)
    assert c['ok'] is False and '没有 forced_tier' in c['reason']
