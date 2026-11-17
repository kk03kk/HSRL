import pytest
import torch

from hsbg.az.train.process_learning import process_loss,log_policy,forward_kl


def test_set_loss_is_probability_mass_and_has_finite_gradient():
    logits=torch.zeros((2,4),requires_grad=True)
    masks=torch.tensor([[1,1,1,0],[1,1,0,0]],dtype=torch.bool)
    targets=torch.tensor([[1,1,0,0],[0,1,0,0]],dtype=torch.bool)
    loss=process_loss(logits,masks,targets)
    expected=-(torch.tensor(2/3).log()+torch.tensor(.5).log())/2
    assert torch.allclose(loss,expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0,3]==0 and logits.grad[0,2]>0 and logits.grad[0,0]<0
    lp=log_policy(logits,masks)
    assert forward_kl(lp,lp).item()==0


def test_invalid_or_illegal_label_fails_closed():
    x=torch.zeros((1,3));mask=torch.tensor([[1,0,0]],dtype=torch.bool)
    with pytest.raises(ValueError):process_loss(x,mask,~mask)
    with pytest.raises(ValueError):process_loss(x,mask,torch.zeros_like(mask))
