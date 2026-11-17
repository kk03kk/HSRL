import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from hsbg.az.train.health_gate import hard_collect_failures, hard_update_failures


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, 0.5, 'bad', None, True])
def test_malformed_health_counts_cannot_pass(value):
    assert hard_collect_failures({'illegal_actions': value})


@pytest.mark.parametrize('key', ['truncated_recruit_contexts', 'mirror_errors', 'dropped_decisions'])
def test_engineering_errors_are_hard_failures(key):
    assert hard_collect_failures({key: 1})


def test_invalid_mirror_counts_and_probabilities_fail_closed():
    assert hard_collect_failures({'mirror_attempt': 10, 'mirror_applied': 11})
    assert hard_update_failures({'illegal_prob_max': float('nan')})


def test_checkpoint_preserves_actual_attention_head_count(tmp_path):
    torch = pytest.importorskip('torch')
    from hsbg.az.train.model import build_net, load_checkpoint, save_checkpoint
    from hsbg.az.train.ppo import forward_model, split_model_outputs
    net = build_net(arch='entity', d_model=32, n_layers=1, n_heads=4)
    net.eval()
    path = tmp_path/'model.pt'
    save_checkpoint(net, str(path))
    restored, meta = load_checkpoint(str(path), 'cpu')
    restored.eval()
    assert meta['n_heads'] == 4
    assert restored.trunk.layers[0].self_attn.num_heads == 4
    f = torch.zeros(2, 3227)
    c = torch.zeros(2, 124, dtype=torch.long)
    mask = torch.ones(2, 245, dtype=torch.bool)
    with torch.no_grad():
        a = split_model_outputs(forward_model(net, f, c, mask))[0]
        b = split_model_outputs(forward_model(restored, f, c, mask))[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_final_export_io_error_is_not_reported_as_success():
    # Exercise the production exception handler without starting training.
    source = Path('workflows/train_entity_sync.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)
                and 'final save failed' in (ast.get_source_segment(source, n) or '')]
    assert len(handlers) == 1
    failure = ast.Raise(exc=ast.Call(func=ast.Name(id='OSError', ctx=ast.Load()),
                                    args=[ast.Constant('disk full')], keywords=[]), cause=None)
    module = ast.Module(body=[ast.Try(body=[failure], handlers=[copy.deepcopy(handlers[0])],
                                     orelse=[], finalbody=[])], type_ignores=[])
    with pytest.raises(OSError, match='disk full'):
        exec(compile(ast.fix_missing_locations(module), '<final-export-test>', 'exec'), {})
