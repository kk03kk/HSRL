"""Set-valued process supervision with explicit legal-action semantics."""
import torch


def log_policy(logits, masks):
    if not bool(masks.any(-1).all()):raise ValueError('empty legal action set')
    if not bool(torch.isfinite(logits).all()):raise ValueError('non-finite logits')
    return logits.float().masked_fill(~masks, -1e9).log_softmax(-1)


def process_loss(logits,masks,targets):
    if not bool(targets.any(-1).all()) or bool((targets & ~masks).any()):
        raise ValueError('invalid optimal action set')
    lp=log_policy(logits,masks)
    return -torch.logsumexp(lp.masked_fill(~targets,-1e9),dim=-1).mean()


def forward_kl(reference_logp,student_logp):
    return (reference_logp.exp()*(reference_logp-student_logp)).sum(-1).mean()
