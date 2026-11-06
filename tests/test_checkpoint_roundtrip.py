"""加载后再次保存不能把numpy规范化统计泄漏进weights-only权重文件。"""
import numpy as np
import torch
from hsbg.az.train.model import build_net,save_checkpoint,load_checkpoint

def test_loaded_metadata_can_be_saved_and_resumed(tmp_path):
 model=build_net(16)
 mean=np.zeros(3227,dtype=np.float32);std=np.ones(3227,dtype=np.float32)
 first=tmp_path/'first.pt';second=tmp_path/'second.pt'
 save_checkpoint(model,str(first),meta={'hidden':16},norm_mean=mean,norm_std=std)
 loaded,meta=load_checkpoint(str(first))
 save_checkpoint(loaded,str(second),meta=meta,norm_mean=meta['norm_mean'],norm_std=meta['norm_std'])
 payload=torch.load(second,weights_only=True)
 assert isinstance(payload['meta']['norm_mean'],list)
 restored,restored_meta=load_checkpoint(str(second))
 assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
 assert np.array_equal(restored_meta['norm_mean'],mean)
