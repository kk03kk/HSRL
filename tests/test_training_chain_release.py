"""Bounded release audit: context coverage, source splits, resources and library PPO."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import taverngym as tg
from taverngym.rl.config import Config, identity
from taverngym.rl.model import Policy
from taverngym.rl.collect import collect
from taverngym.rl.train import update, atomic_save, load_policy
from taverngym.research.library_contexts import context, expected_contexts, seed_plan
from taverngym.research import resources


ROOT = Path(__file__).resolve().parents[1]


def test_random_native_context_enumerator_matches_actual_native_ids():
    rules = Config().rules
    expected = expected_contexts(rules)
    assert len(expected) == 2072
    for seed in range(100):
        assert context(json.loads(tg.Game(seed,json.dumps(rules)).state_json())) in expected


def test_gpu_oom_scaling_preserves_virtual_batch_and_never_changes_architecture(monkeypatch):
    cfg=Config(chunks_per_batch=8,gradient_accumulation=8)
    def probe(c):
        if c.chunks_per_batch>2:
            raise torch.cuda.OutOfMemoryError('synthetic memory limit')
        return {'parameters':123,'weights_discarded':True}
    monkeypatch.setattr(resources,'probe',probe)
    effective,result=resources.qualify_resources(cfg)
    assert effective.chunks_per_batch==2 and effective.gradient_accumulation==32
    assert effective.dim==cfg.dim and result['effective_max_states']==2048
    assert len(result['attempts'])==2


def test_source_disjoint_complete_context_libraries_reach_ppo_and_strict_load(tmp_path):
    # A small explicit fixture exercises all 8 contexts for this lobby. Production
    # configs remain random and require all 2072; no global lobby restriction.
    from tools.prepare_library_stage import prepare_stage
    cfg=Config.load(ROOT/'configs/entity_library_cpu.json')
    cfg.rules={**cfg.rules,'lobby_types':['PIRATE','ABERRATION','MURLOC','DRAGON','QUILBOAR']}
    config=tmp_path/'fixture.json';config.write_text(json.dumps(cfg.to_dict()),encoding='utf-8')
    out=tmp_path/'banks'
    report=prepare_stage(config,out,(8,8,8),ROOT/'assets/library/manual-game1.json')
    assert report['required_contexts']==8 and report['source_splits_disjoint']
    groups=[]
    for split in ('train','validation','test'):
        bank=json.loads((out/f'{split}.json').read_text(encoding='utf-8'))
        assert {(t['lobby'],t['deity'],t['paradox']) for t in bank['traces']}==expected_contexts(cfg.rules)
        groups.append({t['source_group'] for t in bank['traces']})
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    c=replace(Config.load(out/'train-config.json'),n_games=1,epochs=1)
    torch.set_num_threads(2);torch.manual_seed(17);np.random.seed(17)
    model=Policy(c)
    tr,stats=collect(c,model,17)
    assert stats['opponent_counts']=={'library':7} and stats['truncated_games']==0
    result=update(model,torch.optim.Adam(model.parameters(),lr=1e-4),tr,c)
    assert result['updates']>0
    path=tmp_path/'checkpoint.pt'
    atomic_save({'model':model.state_dict(),'identity':identity(c),'config':c.to_dict()},path)
    loaded=load_policy(path,c,identity(c))
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())


def test_resource_probe_discards_updates_and_restores_sampling_rng():
    c=Config(device='cpu',policy_kind='compact_ff',threads=1,torch_threads=1,
             dim=16,hidden=16,layers=1,heads=2,chunks_per_batch=1,seq_len=8)
    torch.manual_seed(99);before=torch.get_rng_state().clone()
    result=resources.probe(c)
    assert result['optimizer_updates']>0 and result['weights_discarded']
    assert torch.equal(before,torch.get_rng_state())
