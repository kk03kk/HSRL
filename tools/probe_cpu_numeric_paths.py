"""Diagnose exact-output test failures without changing model or kernel defaults."""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tests.test_conditional_economy import small_parent,inputs
from tests.test_mechanism_observation import native_target_source_pair
from hsbg.az.train.conditional_economy import fork_conditional
from hsbg.az.train.model import build_net,save_checkpoint,load_checkpoint
from workflows.learn_cycle_process import configure_parameters
from hsbg.runtime.artifacts import atomic_json


def differences(a,b):
    return [{'exact':torch.equal(x,y),'max_abs':float((x-y).abs().max()),
             'argmax_equal':bool(torch.equal(x.argmax(-1),y.argmax(-1))) if x.ndim>1 else None}
            for x,y in zip(a,b)]


def probe(out):
    torch.set_num_threads(1);result={'torch':torch.__version__,'platform':sys.platform}
    for fast in (True,False):
        torch.backends.mha.set_fastpath_enabled(fast)
        base,meta=small_parent();reference=copy.deepcopy(base);model=fork_conditional(base,meta).eval()
        f,c,mask=inputs()
        with torch.no_grad():before=model(f,c,mask);reference_before=reference(f,c,mask)
        configure_parameters(model,'all')
        with torch.no_grad():frozen=model(f,c,mask)
        opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
        action=int(mask[1].nonzero()[0]);loss=-model(f,c,mask)[0][1,action]
        loss.backward();opt.step()
        with torch.no_grad():after=model(f,c,mask);direct=model.base(f,c,mask)
        path=out/f'conditional-{fast}.pt';save_checkpoint(model,str(path),meta=meta)
        restored,_=load_checkpoint(str(path))
        with torch.no_grad():roundtrip=restored(f,c,mask)
        comparison={'initial':differences(before,reference_before),'freeze':differences(frozen,before),
            'fallback_after_learning':differences(tuple(x[:1] for x in after),tuple(x[:1] for x in frozen)),
            'critic_after_learning':differences(after[1:],frozen[1:]),
            'fallback_vs_direct':differences(tuple(x[:1] for x in after),tuple(x[:1] for x in direct)),
            'roundtrip':differences(after,roundtrip),
            'base_state_exact':all(torch.equal(v,reference.state_dict()[k]) for k,v in model.base.state_dict().items())}
        a,b=native_target_source_pair();fa,ca=a.observe(0);fb,cb=b.observe(0)
        torch.manual_seed(513);net=build_net(arch='entity',d_model=16,n_layers=1,n_heads=2).eval()
        f=torch.tensor(np.stack([np.frombuffer(fa,dtype='<f4'),np.frombuffer(fb,dtype='<f4')]))
        c=torch.tensor(np.stack([np.frombuffer(ca,dtype='<i4'),np.frombuffer(cb,dtype='<i4')]),dtype=torch.long)
        identical=c.clone();identical[1]=identical[0]
        with torch.no_grad():actual=net(f,c);same=net(f,identical);again=net(f,c)
        comparison.update(alias_same_batch_position=differences(actual,same),repeat=differences(actual,again),
            duplicate_rows=differences(tuple(x[:1] for x in same),tuple(x[1:] for x in same)))
        result[str(fast)]=comparison
    atomic_json(out/'complete.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);probe(out)
