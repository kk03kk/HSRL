import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import taverngym as tg
from taverngym.rl.config import Config, identity
from taverngym.rl.model import Policy, pack
from taverngym.rl.collect import collect, observation
from taverngym.rl.train import update
from taverngym.rl.library_bank import assert_disjoint, bank_text


ROOT = Path(__file__).resolve().parents[1]


def small():
    return Config.load(ROOT/'configs/entity_library_cpu.json')


def fixture_bank(tmp_path, split='train'):
    c = small()
    env = tg.BatchEnv(1,seed=29,controlled=range(8),config={**c.rules,'capture_library':True})
    env.auto_reset = False
    for _ in range(1000):
        env.step(env.bot_actions())
        capture = env.captured_library(0)
        if capture:
            break
    else:
        raise AssertionError('source did not reach first combat')
    rules = json.loads(env.save(0))['game']['cfg']; rules['capture_library'] = False
    p = copy.deepcopy(next(p for p in capture['players'] if p['board']))
    p['pending'] = []
    p['board'] = p['board'][:1]
    p['board'][0].update(atk=17,hp=23,max_hp=23)
    # Every opponent intentionally shares one short trajectory: parking after learner death
    # must prevent endless mirror-library battles from invalidating its natural placement.
    bank = {'format':'taverngym.library.v1','split':split,'rules':rules,
            'card_table':capture['card_table'],'native_source':tg.build_info()['source_hash'],
            'traces':[{'id':'opening','source_group':'source-game-'+split,'lobby':capture['lobby'],
                       'deity':capture['deity'],'paradox':capture['paradox'],'frames':{'1':p}}]}
    path = tmp_path/(split+'.json');path.write_text(json.dumps(bank))
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), bank


def test_entity_assets_and_large_model_size_are_real():
    c = Config.load(ROOT/'configs/entity_library_nus.json')
    m = Policy(c)
    count = sum(p.numel() for p in m.parameters())
    assert 17_000_000 < count < 19_000_000
    assert not m.is_recurrent
    assert len(m.entities.layers) == 8
    assert m.entities.layers[0].linear1.out_features == 1536
    bad = replace(c,rule_asset_sha256='0'*64)
    with pytest.raises(ValueError,match='checksum'):
        Policy(bad)


def test_native_library_binding_preserves_rng_and_hides_unseen_boards(tmp_path):
    path, sha, bank = fixture_bank(tmp_path)
    c = small()
    # Source metadata is seed-specific: locate exactly the matching fresh native lobby.
    env = tg.BatchEnv(1,seed=29,controlled=range(8),config=c.rules)
    env.auto_reset = False
    before = json.loads(env.save(0))['game']
    env.bind_library(bank_text(str(path),sha),[0])
    after = json.loads(env.save(0))['game']
    assert before['rng'] == after['rng']
    assert after['library']['learner'] == 0
    assert len(after['library']['seats']) == 8
    assert after['library']['seats'][0] is None
    assert before['players'][0] == after['players'][0]
    visible = env.observe(); own_extras = env.training_extras()[0]
    modified = json.loads(env.save(0))
    frame = modified['game']['library']['seats'][1]['frames']['1']
    frame['gold'] = 999
    if frame['board']:
        frame['board'][0]['atk'] = 999
    env.load(0,json.dumps(modified))
    assert np.array_equal(visible['features'][0],env.observe()['features'][0])
    assert np.array_equal(visible['cards'][0],env.observe()['cards'][0])
    assert own_extras == env.training_extras()[0]

    assert np.array_equal(env.observe()['action_mask'][0],tg.BatchEnv(1,seed=29,controlled=range(8),config=c.rules).observe()['action_mask'][0])


def test_bank_checksums_and_source_split_overlap_fail_closed(tmp_path):
    path, sha, _ = fixture_bank(tmp_path)
    with pytest.raises(ValueError,match='checksum'):
        bank_text(str(path),'0'*64)
    with pytest.raises(ValueError,match='training sources'):
        bank_text(str(path),sha,'evaluation')
    other = tmp_path/'duplicate.json';other.write_bytes(path.read_bytes())
    with pytest.raises(ValueError,match='overlap'):
        assert_disjoint([path,other])


def test_library_rollout_and_update_reach_natural_learner_terminal(tmp_path):
    path,sha,bank = fixture_bank(tmp_path)
    # collect's per-lobby seed transform is identical to fixture_bank's native BatchEnv.
    c = replace(small(),n_games=1,opponent_mode='library',library_bank_path=str(path),library_bank_sha256=sha)
    torch.set_num_threads(2);torch.manual_seed(77)
    m = Policy(c)
    trajectories,stats = collect(c,m,29)
    assert stats['opponent_counts'] == {'library':7}
    assert stats['truncated_games'] == 0
    assert len(trajectories)==1 and 1<=trajectories[0]['rank']<=8
    assert stats['samples'] > 0
    before = m.action.weight.detach().clone()
    result = update(m,torch.optim.Adam(m.parameters(),lr=1e-4),trajectories,c)
    assert result['updates']>0 and not torch.equal(before,m.action.weight)


def test_gradient_accumulation_matches_one_large_batch_on_same_data():
    c = replace(small(),n_games=1,seq_len=16,chunks_per_batch=2,gradient_accumulation=2,epochs=1,target_kl=100.)
    torch.set_num_threads(2);torch.manual_seed(30)
    m = Policy(c)
    tr,_ = collect(c,None,120,teacher=True)
    # One deterministic four-chunk update, with no stale policy ratio.
    tr[0]['samples'] = tr[0]['samples'][:64]
    assert len(tr[0]['samples'])>=48
    for sample in tr[0]['samples']:
        with torch.no_grad():
            logits,value,_,_=m(pack([sample['obs']],'cpu'),torch.zeros(1,c.hidden))
        sample['logp']=float(logits.log_softmax(-1)[0,sample['action']]);sample['value']=float(value[0])
    twin = copy.deepcopy(m)
    # Compare gradients with SGD: Adam can amplify ~1e-9 roundoff in the policy
    # head's shared additive bias (whose true softmax gradient is zero). This
    # test checks accumulation equivalence; production training still uses Adam.
    np.random.seed(31);a=update(m,torch.optim.SGD(m.parameters(),lr=.1),copy.deepcopy(tr),c)
    np.random.seed(31);b=update(twin,torch.optim.SGD(twin.parameters(),lr=.1),copy.deepcopy(tr),replace(c,chunks_per_batch=4,gradient_accumulation=1))
    assert a['updates']==b['updates']==1
    assert all(torch.allclose(x.grad,y.grad,atol=2e-7,rtol=2e-5)
               for x,y in zip(m.parameters(),twin.parameters()) if x.grad is not None)
    assert all(torch.allclose(x,y,atol=2e-6,rtol=2e-5) for x,y in zip(m.parameters(),twin.parameters()))


def test_private_opponent_hand_does_not_change_player_policy_input():
    c=small();env=tg.BatchEnv(1,seed=44,controlled=range(8),config=c.rules)
    before=env.observe();extras=env.training_extras()
    state=json.loads(env.save(0));state['game']['players'][1]['gold']=99
    # A private nonpublic counter must not enter the learner's observed extras.
    state['game']['players'][1]['stats']['gold_spent']=999
    env.load(0,json.dumps(state));after=env.observe()
    assert np.array_equal(before['features'][0],after['features'][0])
    assert np.array_equal(before['cards'][0],after['cards'][0])
    assert extras[0]==env.training_extras()[0]


def test_library_trace_replays_exactly_without_hidden_policy_state(tmp_path):
    from taverngym.offline.capture import capture
    from taverngym.offline.trace import read,replay
    path,sha,_=fixture_bank(tmp_path)
    c=replace(small(),n_games=1,opponent_mode="library",library_bank_path=str(path),library_bank_sha256=sha)
    torch.manual_seed(51);m=Policy(c)
    target=tmp_path/"library.jsonl"
    result=capture(c,seed=29,actor=m,path=target)
    assert result['status']=="completed"
    assert replay(read(target))['state_hashes_verified']
