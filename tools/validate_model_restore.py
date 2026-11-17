"""Fixed-batch update/save/restore equivalence using a real model and Adam state.

Probe output is engineering evidence, never a training starting point.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from hsbg.az.train.model import load_checkpoint
from hsbg.az.train.ppo import PPOBatch, PPOConfig, forward_model, split_model_outputs, masked_log_softmax_torch
from hsbg.az.train.ppo_trainer import PPOTrainer
from hsbg.runtime.artifacts import atomic_json, file_identity


def validate(source, out, device):
    import hsbg_engine as engine
    torch.set_num_threads(2)
    source, out = Path(source), Path(out)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    before = {n:file_identity(source/n) for n in ('model.pt','trainer.pt')}
    state = torch.load(source/'trainer.pt', map_location='cpu', weights_only=False)
    cfg = PPOConfig(**state['config'])
    cfg.device = device
    if device == 'cpu': cfg.amp = 'off'
    model, _ = load_checkpoint(str(source/'model.pt'), device=device)
    trainer = PPOTrainer(model, cfg, state=state)
    version = trainer.updates
    fs, cs, ms = [], [], []
    for seed in (619301,619302):
        game = engine.Game(seed=seed)
        f,c = game.observe(0)
        fs.append(np.frombuffer(f,dtype='<f4').copy())
        cs.append(np.frombuffer(c,dtype='<i4').copy())
        ms.append(np.frombuffer(game.legal_mask(0),dtype=np.uint8).astype(bool))
    fs,cs,ms = np.stack(fs),np.stack(cs),np.stack(ms)
    with torch.inference_mode():
        logits,_,_ = split_model_outputs(forward_model(model,torch.from_numpy(fs).to(device),
                                                      torch.from_numpy(cs).long().to(device),torch.from_numpy(ms).to(device)))
        lp = masked_log_softmax_torch(logits,torch.from_numpy(ms).to(device))
        actions = lp.argmax(dim=-1)
        old = lp.gather(1,actions[:,None]).squeeze(1).cpu().numpy().copy()
    batch = PPOBatch(features=fs,cards=cs,masks=ms,actions=actions.cpu().numpy(),old_logprobs=old,
                     values=np.zeros(2,dtype=np.float32),rewards=np.array([1,-1],dtype=np.float32),
                     dones=np.ones(2,dtype=np.float32),advantages=np.array([1,-1],dtype=np.float32),
                     returns=np.array([1,-1],dtype=np.float32),seats=np.zeros(2,dtype=np.int32),
                     game_ids=np.array([0,1]),ranks=np.array([1,8]),turns=np.ones(2,dtype=np.int32),
                     target_pi=np.zeros((2,245),dtype=np.float32),has_pi=np.zeros(2,dtype=bool),
                     episode_ids=np.arange(2,dtype=np.int32))
    first = trainer.update(batch)
    assert first['updates'] > 0 and trainer.updates == version+1
    trainer.save(str(out/'probe-trainer.pt'))
    second = trainer.update(batch)
    expected = {k:v.detach().cpu().clone() for k,v in trainer.model.state_dict().items()}
    expected_scheduler = trainer.scheduler.state_dict()
    expected_optimizer = {key:{name:value.detach().cpu().clone() if torch.is_tensor(value) else value
                              for name,value in fields.items()}
                          for key,fields in trainer.optimizer.state_dict()['state'].items()}
    del trainer, model
    gc.collect()
    if device.startswith('cuda'): torch.cuda.empty_cache()
    restored_model,_ = load_checkpoint(str(source/'model.pt'),device=device)
    restored = PPOTrainer.load(restored_model,str(out/'probe-trainer.pt'),config=cfg,device=device)
    resumed = restored.update(batch)
    assert restored.updates == version+2
    for key,value in restored_model.state_dict().items():
        assert torch.equal(value.detach().cpu(), expected[key]), key
    assert restored.scheduler.state_dict() == expected_scheduler
    for key,fields in restored.optimizer.state_dict()['state'].items():
        for name,value in fields.items():
            if torch.is_tensor(value): assert torch.equal(value.cpu(),expected_optimizer[key][name]), (key,name)
            else: assert value == expected_optimizer[key][name]
    assert before == {n:file_identity(source/n) for n in before}
    report = {'valid':True,'source_generation':version,'probe_generation':version+2,
              'device':device,'model_bitwise_equal':True,'adam_bitwise_equal':True,
              'scheduler_equal':True,'source_unchanged':True,
              'optimizer_steps':[first['updates'],second['updates'],resumed['updates']],
              'scope':'two synthetic samples; engineering probe; never publish as trained model'}
    atomic_json(out/'validation.json',report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    print(json.dumps(validate(args.source,args.out,args.device)))
