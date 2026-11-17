import numpy as np
import torch


def test_native_credit_and_gradients_leave_weights_unchanged():
    from hsbg.az.train.model import build_net
    from hsbg.az.train.ppo import CollectConfig,PPOEvaluator,build_batch_from_records
    from hsbg.az.train.common import set_rank_utility_table,rank_utility_table
    from hsbg.az.train.parallel_collect import collect_ppo_lockstep
    from workflows.probe_training_credit import UTILITY,credit_summary,gradient_summary
    torch.set_num_threads(1);torch.manual_seed(27)
    model=build_net(arch='entity',d_model=32,n_layers=1,n_heads=4)
    initial={k:v.clone() for k,v in model.state_dict().items()}
    saved=rank_utility_table().copy()
    try:
        set_rank_utility_table(UTILITY)
        cfg=CollectConfig(games=2,seed=11919000,opponent='bot',behavior='net',lockstep=True,
            forced_tier_plan='2:2,5:3,7:4,8:5',forced_tier_prob=0,forced_tier_teacher=True)
        records,meta=collect_ppo_lockstep(model,cfg,eval_fn=PPOEvaluator(model),seed=11919000,games=2)
        batch=build_batch_from_records(records,gamma=1,lam=.995,meta=meta)
        assert not batch.validate()
        rows,mc=credit_summary(batch)
        assert all(r['games']<=2 for r in rows)
        assert np.allclose(mc,np.asarray(UTILITY)[batch.ranks-1]-batch.values,atol=2e-6)
        gradients=gradient_summary(model,batch,mc,'cpu',38,1)
        assert gradients and all(np.isfinite(list(r['norms'].values())).all() for r in gradients)
        assert all(torch.equal(initial[k],v) for k,v in model.state_dict().items())
    finally:
        set_rank_utility_table(saved)
