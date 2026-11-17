"""Translate legacy shard environment variables to the single evaluation runner.

Compatibility only: new assessments should commit an explicit JSON plan.
"""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflows.evaluate_suite import run


def plan_from_environment(env):
    models = []
    for value in env['HSAI_GAP_MODELS'].split(';'):
        name, checkpoint = value.split('=', 1)
        if not name or not checkpoint: raise ValueError('expected name=checkpoint')
        models.append({'name':name, 'checkpoint':checkpoint})
    options = {
        'mirror-snaps':env['HSAI_GAP_MIRROR_SNAPS'],
        'human-baseline':env.get('HSAI_GAP_BASELINE', 'artifacts/runs/human-baseline-20261112/baseline.json'),
        'mirror-sample':env.get('HSAI_GAP_MIRROR_SAMPLE', 'p50band'),
        'mirror-strict':int(env.get('HSAI_GAP_MIRROR_STRICT', '0')),
        'lobby-exclude':env.get('HSAI_GAP_LOBBY_EXCLUDE', ''),
        'freeze-cap':int(env.get('HSAI_GAP_FREEZE_CAP', '1')),
        'max-actions-per-turn':100, 'move-cap':2, 'move-repeat-limit':2, 'forbid-unfreeze':1,
    }
    if env.get('HSAI_GAP_FORCED_TIER'):
        options['forced-tier-plan'] = env['HSAI_GAP_FORCED_TIER']
    return {'schema':'hsai.eval-plan/v1', 'models':models, 'evaluation':options,
            'games':int(env.get('HSAI_GAP_GAMES', '1024')),
            'shards':int(env.get('HSAI_SHARDS', '8')), 'seed':int(env.get('HSAI_GAP_SEED', '930000'))}


if __name__ == '__main__':
    run(plan_from_environment(os.environ), os.environ['HSAI_GAP_OUT'])
