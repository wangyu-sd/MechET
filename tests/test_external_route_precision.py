"""End-to-end tensor regression with tiny random Qwen3.5+LoRA; no corpus/model experiment."""
from copy import deepcopy
import math

import pytest


@pytest.mark.parametrize('counter',[False,True])
@pytest.mark.parametrize('dtype_name,scale',[('float32',1.),('bfloat16',1.),('bfloat16',32.),('bfloat16',128.)])
def test_native_cached_generation_then_dr_grpo_backward_is_supported(monkeypatch,dtype_name,scale,counter):
    import torch
    # This is a CPU tensor regression. Select upstream's CPU fallback before
    # importing the model, rather than importing unused GPU-only FLA kernels.
    from transformers.utils import import_utils
    monkeypatch.setattr(import_utils,'is_flash_linear_attention_available',lambda:False)
    monkeypatch.setattr(import_utils,'is_causal_conv1d_available',lambda:False)
    from transformers import Qwen3_5TextConfig,Qwen3_5ForCausalLM,GenerationConfig,LogitsProcessorList
    from transformers.models.qwen3_5 import modeling_qwen3_5 as upstream
    from peft import LoraConfig,get_peft_model
    from mechet.route_policy_sampling import TokenLogprobs,PrivateSampling
    from mechet.external_route_rl import optimize_routes,DR_GRPO_ALGORITHM
    for name in ('FusedRMSNormGated','chunk_gated_delta_rule','fused_recurrent_gated_delta_rule',
                 'causal_conv1d_fn','causal_conv1d_update'):
        monkeypatch.setattr(upstream,name,None)
    torch.set_num_threads(1);torch.manual_seed(19)
    config=Qwen3_5TextConfig(vocab_size=19,hidden_size=16,intermediate_size=24,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,head_dim=8,
        linear_num_key_heads=2,linear_num_value_heads=2,linear_key_head_dim=8,linear_value_head_dim=8,
        layer_types=['linear_attention','full_attention'],max_position_embeddings=128,
        rope_parameters=dict(rope_type='default',rope_theta=10000.,partial_rotary_factor=1.,mrope_section=[1,1,2]))
    base=Qwen3_5ForCausalLM(config).to(getattr(torch,dtype_name))
    with torch.no_grad():base.lm_head.weight.mul_(scale)
    actor=get_peft_model(base,LoraConfig(task_type='CAUSAL_LM',r=2,lora_alpha=4,
        target_modules=['q_proj','v_proj','in_proj_qkv'],lora_dropout=0.)).eval()
    reference=deepcopy(actor).eval()
    for p in reference.parameters():p.requires_grad_(False)
    prompts=[list(range(1,2+i%6)) for i in range(64)]
    ids=torch.zeros((64,6),dtype=torch.long);mask=torch.zeros_like(ids)
    for i,p in enumerate(prompts):ids[i,-len(p):]=torch.tensor(p);mask[i,-len(p):]=1
    capture=TokenLogprobs(18)
    processor=PrivateSampling([str(i) for i in range(64)],17,'cpu')
    if counter:
        from mechet.route_counter_sampling import CounterCategorical
        processor=CounterCategorical([str(i) for i in range(64)],17,'cpu',capture)
    generation=GenerationConfig(do_sample=not counter,top_k=0,top_p=1.,temperature=1.,
        pad_token_id=0,eos_token_id=18,use_cache=True)
    with torch.inference_mode():
        output=actor.generate(input_ids=ids,attention_mask=mask,max_new_tokens=4,generation_config=generation,
            logits_processor=LogitsProcessorList([capture,processor]))
    capture.finish(output[:,-1]);logps=capture.values();reports=[]
    for i,row in enumerate(output):
        response=row[6:].tolist()
        if 18 in response:response=response[:response.index(18)+1]
        reports.append(dict(reward=[0.,1.,0.,.2][i%4],records=[dict(prompt_token_ids=prompts[i],
            generated_token_ids=response,behavior_logprobs=logps[i])]))
    groups=[reports[i:i+4] for i in range(0,64,4)]
    optimizer=torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad],lr=2e-6)
    cfg=dict(algorithm=DR_GRPO_ALGORITHM,candidates=4,loss_batch_size=2,pad_token_id=0,
        clip_epsilon=.2,kl_beta=.02,loss_token_normalizer=1024.,max_grad_norm=.5,
        behavior_logprob_tolerance=.25)
    result=optimize_routes(actor,reference,optimizer,groups,cfg)
    assert result['optimizer_updated'] and math.isfinite(result['grad_norm'])
    assert result['sampled_action_records']==64
