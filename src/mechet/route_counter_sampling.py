"""Opt-in categorical draws with request/decode-counter RNG and one GPU search.

This changes the RNG realization, not temperature/top-p/tool distribution.
The HF outer loop selects the pinned token with argmax; it must not sample a
second categorical distribution over the pin. Original logprobs are captured
before pinning. Golden PrivateSampling remains the default.
"""
import hashlib


class CounterCategorical:
    def __init__(self,uids,seed,device,capture,*,on_failure=None,prompt_width=0):
        self.uids=list(uids);self.seed=seed;self.device=device;self.capture=capture;self.step=0
        self.on_failure,self.prompt_width=on_failure,prompt_width

    def __call__(self,input_ids,scores):
        import torch
        logps=self.capture.previous
        if logps is None:logps=scores.float().log_softmax(-1)
        probabilities=logps.exp()
        valid=torch.isfinite(probabilities).all(-1)&(probabilities.sum(-1)>0)
        if not bool(valid.all()):
            from .route_sampling_diagnostics import describe_distribution_failure
            report=describe_distribution_failure(scores,input_ids,bad_rows=(~valid).nonzero().flatten().tolist(),
                request_ids=self.uids,finished=self.capture.done,decode_step=self.step,prompt_width=self.prompt_width)
            if self.on_failure:self.on_failure(dict(event='policy-distribution-failure',**report))
            raise ValueError('INVALID_COUNTER_POLICY_DISTRIBUTION')
        # 52 bits give strictly interior exactly representable FP64 uniforms.
        uniforms=[((int.from_bytes(hashlib.sha256(
            f'{self.seed}:{uid}:{self.step}'.encode()).digest()[:8],'big')>>12)+.5)/2**52
            for uid in self.uids]
        u=torch.tensor(uniforms,device=scores.device,dtype=torch.float64)[:,None]
        cdf=probabilities.cumsum(-1,dtype=torch.float64)
        cdf=cdf/cdf[:,-1:]
        chosen=torch.searchsorted(cdf.contiguous(),u,right=True).clamp_max(scores.shape[-1]-1)
        self.step+=1
        return torch.full_like(scores,-float('inf')).scatter(1,chosen,0.)
