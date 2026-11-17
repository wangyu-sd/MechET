"""Whole-route token-sum surrogate plus separately normalized TRAIN CE.

The KL estimate is importance-corrected at behavior-sampled observations and
tokens, not an assertion of exact current-policy trajectory-distribution KL.
"""


def route_advantages(rewards):
    if len(rewards)<2:
        raise ValueError('RLOO_REQUIRES_TWO_INDEPENDENT_EPISODES')
    if any(r is None for r in rewards):
        return None
    total=sum(rewards)
    return [float(r)-(total-r)/(len(rewards)-1) for r in rewards]


def action_loss(new, behavior, reference, advantage, *, clip, beta, normalizer):
    import torch
    if new.shape!=behavior.shape or new.shape!=reference.shape or not new.numel():
        raise ValueError('POLICY_TOKEN_RECEIPT_SHAPE_MISMATCH')
    if normalizer<=0 or not 0<clip<1 or beta<0:
        raise ValueError('INVALID_ROUTE_OBJECTIVE')
    ratio=(new-behavior.detach()).exp()
    pg=-torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage).sum()/normalizer
    delta=reference.detach()-new
    kl=(ratio*(delta.exp()-delta-1)).sum()/normalizer
    return dict(total=pg+beta*kl,pg=pg,kl=kl,tokens=new.numel())


def policy_sequences(report):
    result=[]
    for record in report['records']:
        p,r=record.get('prompt_token_ids'),record.get('generated_token_ids')
        b=record.get('behavior_logprobs')
        if r:
            if not p or b is None or len(b)!=len(r):
                raise ValueError('INCOMPLETE_BEHAVIOR_TOKEN_RECEIPT')
            result.append((p,r,b))
    return result


def supervised_sequence(tokenizer,row,max_context):
    from .qwen35_protocol import encode_tool_decision
    encoded,info=encode_tool_decision(tokenizer,row,max_length=max_context)
    if info['exceeds_max_length']:
        raise ValueError('GT_CONTEXT_OVERFLOW_NO_TRUNCATION')
    start,end=info['assistant_spans'][0]
    return encoded['input_ids'][:start],encoded['input_ids'][start:end]
