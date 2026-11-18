"""Bounded immutable-reference scores; actor outputs and gradients never enter.

Entries live as detached CPU tensors. Every consumer receives its own tensor;
only exact prompt/response IDs under the same frozen numerical contract reuse a
reference forward. A restart simply recomputes these immutable reference scores.
"""
from collections import OrderedDict,Counter

from .external_route_data import signature


def reference_identity(cfg):
    if not cfg.get('parent_sha256') or not cfg.get('base_manifest'):
        raise ValueError('REFERENCE_CACHE_IMMUTABLE_IDENTITY_REQUIRED')
    return dict(reference_sha256=cfg['parent_sha256'],tokenizer_manifest=cfg['base_manifest'],
        precision=cfg.get('policy_precision','legacy_fp32_lora'),tf32=cfg.get('policy_allow_tf32'),
        pad_token_id=cfg.get('pad_token_id'),mask='left_prompt_right_response_nonpadding_v1')


class ReferenceLogprobCache:
    def __init__(self,capacity):
        if type(capacity) is not int or capacity<1:raise ValueError('REFERENCE_CACHE_CAPACITY')
        self.capacity=capacity;self.entries=OrderedDict()
        self.counts=Counter(dict.fromkeys(('cache_hits','cache_misses','coalesced_pairs',
            'evictions','physical_sequences','physical_batches'),0))

    def snapshot(self):
        return dict(self.counts,entries=len(self.entries),capacity=self.capacity,
            bytes=sum(v.numel()*v.element_size() for v in self.entries.values()))

    def lookup(self,pairs,*,identity,compute,device):
        import torch
        if identity is None:raise ValueError('REFERENCE_CACHE_IMMUTABLE_IDENTITY_REQUIRED')
        keys=[];values={};missing=OrderedDict()
        for p,r in pairs:
            if not p or not r:raise ValueError('REFERENCE_CACHE_EMPTY_SEQUENCE')
            key=signature([identity,p,r]);keys.append(key)
            if key in values or key in missing:
                self.counts['coalesced_pairs']+=1;continue
            if key in self.entries:
                self.counts['cache_hits']+=1;self.entries.move_to_end(key);values[key]=self.entries[key]
            else:
                self.counts['cache_misses']+=1;missing[key]=(p,r)
        if missing:
            computed=compute(list(missing.values()))
            if len(computed)!=len(missing):raise ValueError('REFERENCE_CACHE_FORWARD_LENGTH')
            material=[]
            for value,(_,response) in zip(computed,missing.values()):
                detached=value.detach().to('cpu').clone()
                if (detached.ndim!=1 or len(detached)!=len(response)
                        or not bool(torch.isfinite(detached).all()) or bool((detached>0).any())):
                    raise ValueError('REFERENCE_CACHE_INVALID_PROBABILITIES')
                material.append(detached)
            self.counts['physical_sequences']+=len(material);self.counts['physical_batches']+=1
            for key,value in zip(missing,material):
                values[key]=value;self.entries[key]=value
                while len(self.entries)>self.capacity:
                    self.entries.popitem(last=False);self.counts['evictions']+=1
        return [values[k].to(device=device,copy=True) for k in keys]
