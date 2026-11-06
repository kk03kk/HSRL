"""本地真实网络的合成小批次更新/保存/恢复验证，不作为棋力实验。"""
import gc,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from hsbg.az.train.model import load_checkpoint
from hsbg.az.train.ppo import PPOConfig,PPOBatch,forward_model,split_model_outputs,masked_log_softmax_torch
from hsbg.az.train.ppo_trainer import PPOTrainer
from hsbg.az.train.personal import sha256,write_json
import hsbg_engine as E

def main():
    torch.set_num_threads(2)
    source=ROOT/'artifacts/models/nus-1080772/step-007357-360m'
    before={n:sha256(source/n) for n in ('model.pt','trainer.pt')}
    state=torch.load(source/'trainer.pt',map_location='cpu',weights_only=False)
    model,_=load_checkpoint(str(source/'model.pt'),device='cpu')
    cfg=PPOConfig(**state['config']);cfg.device='cpu';cfg.amp='off';cfg.epochs=1;cfg.minibatch=2;cfg.grad_accum=1;cfg.compile_train=0
    trainer=PPOTrainer(model,cfg,state=state);steps0=min(float(v['step']) for v in trainer.optimizer.state.values())
    games=[E.Game(seed=s) for s in (7357,7358)]
    fs=[];cs=[];ms=[]
    for g in games:
        obs=g.observe(0);fs.append(np.frombuffer(obs[0],dtype=np.float32).copy());cs.append(np.frombuffer(obs[1],dtype=np.int32).copy().astype(np.int64));ms.append(np.frombuffer(g.legal_mask(0),dtype=np.uint8).astype(bool))
    fs=np.stack(fs);cs=np.stack(cs);ms=np.asarray(ms,dtype=bool)
    def batch():
        with torch.inference_mode():logits,_,_=split_model_outputs(forward_model(model,torch.from_numpy(fs),torch.from_numpy(cs),torch.from_numpy(ms)));lp=masked_log_softmax_torch(logits,torch.from_numpy(ms));actions=torch.argmax(lp,dim=-1);old=lp.gather(1,actions[:,None]).squeeze(1).numpy().copy()
        return PPOBatch(features=fs,cards=cs,masks=ms,actions=actions.numpy(),old_logprobs=old,values=np.zeros(2,dtype=np.float32),rewards=np.asarray([1,-1],dtype=np.float32),dones=np.ones(2,dtype=np.float32),advantages=np.asarray([1,-1],dtype=np.float32),returns=np.asarray([1,-1],dtype=np.float32),seats=np.zeros(2,dtype=np.int32),game_ids=np.asarray([1,2]),ranks=np.asarray([1,8]),turns=np.ones(2,dtype=np.int32),target_pi=np.zeros((2,245),dtype=np.float32),has_pi=np.zeros(2,dtype=bool),episode_ids=np.arange(2,dtype=np.int32))
    q1=trainer.update(batch());out=ROOT/'artifacts/local/7357-update-restore';out.mkdir(parents=True,exist_ok=True);trainer.save(str(out/'trainer.pt'))
    saved=torch.load(out/'trainer.pt',map_location='cpu',weights_only=False)
    del trainer;gc.collect()
    trainer=PPOTrainer(model,cfg,state=saved)
    assert trainer.updates==7358 and trainer.scheduler.state_dict()==saved['scheduler']
    assert torch.equal(torch.get_rng_state(),saved['torch_rng'])
    q2=trainer.update(batch());steps1=min(float(v['step']) for v in trainer.optimizer.state.values())
    assert trainer.updates==7359 and q1['updates']>0 and q2['updates']>0 and steps1>steps0
    assert before=={n:sha256(source/n) for n in before}
    write_json(ROOT/'records/research/long1080772/local-update-restore.json',dict(passed=True,source_version=7357,final_probe_version=7359,adam_step_before=steps0,adam_step_after=steps1,optimizer_steps=[q1['updates'],q2['updates']],source_unchanged=True,scope='real16M_model_synthetic_two_sample_CPU_batch_not_gameplay_or_GPU_performance',probe_output_not_training_start=True))
    print('Real7357 update/save/restore/update passed; source unchanged.')
if __name__=='__main__':main()
