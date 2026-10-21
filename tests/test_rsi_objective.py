import torch


def test_dpo_update_improves_preferred_logprob_without_reference_gradient():
    from mechet.rsi_objective import dpo_loss
    theta=torch.nn.Parameter(torch.tensor(0.));ref=torch.tensor(0.,requires_grad=True)
    optimizer=torch.optim.SGD([theta],lr=1.)
    loss=dpo_loss(theta,-theta,ref,beta=.1)
    loss.backward();assert ref.grad is None
    optimizer.step();assert theta.item()>0
    assert torch.allclose(loss.detach(),torch.tensor(.6931472))


def test_action_logp_mask_includes_eos_but_not_prompt():
    from mechet.rsi_objective import action_inputs
    ids,mask=action_inputs([4,5],[6,7,151645],device='cpu')
    assert ids.tolist()==[[4,5,6,7,151645]]
    assert mask.tolist()==[[0,0,1,1,1]]


def tiny_qwen():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=64,
        num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=64)).eval()


def test_selected_projection_matches_full_teacher_forced_logps_and_gradients():
    from mechet.selected_policy_logps import install_selected_logps_forward
    from mechet.rsi_objective import action_inputs
    torch.manual_seed(17);model=tiny_qwen();ids,mask=action_inputs([1,2,3],[4,5,6],device='cpu')
    logits=model(input_ids=ids).logits[:,:-1].float()
    expected=(logits.log_softmax(-1).gather(-1,ids[:,1:,None]).squeeze(-1)*mask[:,1:]).sum()
    before=torch.autograd.grad(expected,model.model.layers[0].self_attn.q_proj.weight,retain_graph=True)[0]
    install_selected_logps_forward(model,chunk_size=2)
    got=model(input_ids=ids,attention_mask=torch.ones_like(ids),loss_token_mask=mask)['selected_logps'].sum()
    after=torch.autograd.grad(got,model.model.layers[0].self_attn.q_proj.weight)[0]
    assert torch.allclose(got,expected,atol=1e-5)
    assert torch.allclose(before,after,atol=1e-5)


def test_real_lora_dpo_saves_new_weights_and_keeps_reference_unchanged(tmp_path):
    from peft import get_peft_model,LoraConfig
    from mechet.selected_policy_logps import install_selected_logps_forward
    from mechet.rsi_objective import PolicyScorer
    torch.manual_seed(17);base=tiny_qwen();install_selected_logps_forward(base)
    config=LoraConfig(r=4,lora_alpha=8,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')
    model=get_peft_model(base,config,adapter_name='reference');model.add_adapter('forward_judge',config)
    model.save_pretrained(tmp_path/'initial',selected_adapters=['reference'])
    scorer=PolicyScorer.__new__(PolicyScorer);scorer.model=model;scorer.device='cpu';scorer.reference_cache={}
    class Tokenizer:
        def save_pretrained(self,path):return None
    scorer.tokenizer=Tokenizer()
    reference={n:p.detach().clone() for n,p in model.named_parameters() if '.reference.' in n}
    result=scorer.train_dpo([{'id':'pair','prompt_ids':[1,2,3],'chosen_ids':[4,6],
        'rejected_ids':[5,6],'tool':'apply_electron_flow'}],tmp_path/'initial/reference','r0',tmp_path/'trained',steps=2,accum=1,lr=.001)
    assert result['status']=='updated' and result['optimizer_steps']==2 and result['weight_delta_l2']>0
    assert (tmp_path/'trained/policy/adapter_model.safetensors').exists()
    assert all(torch.equal(reference[n],p) for n,p in model.named_parameters() if n in reference)
    again=scorer.train_dpo([{'id':'pair','prompt_ids':[1,2,3],'chosen_ids':[4,6],
        'rejected_ids':[5,6],'tool':'apply_electron_flow'}],tmp_path/'initial/reference','r0',tmp_path/'trained',steps=2,accum=1,lr=.001)
    assert again['weight_sha256']==result['weight_sha256']


def test_optimizer_recovers_after_interruption_without_repeating_committed_step(tmp_path,monkeypatch):
    import pytest
    from peft import get_peft_model,LoraConfig
    from mechet.selected_policy_logps import install_selected_logps_forward
    from mechet.rsi_objective import PolicyScorer
    from mechet.rsi_contract import JsonJournal
    torch.manual_seed(17);base=tiny_qwen();install_selected_logps_forward(base)
    cfg=LoraConfig(r=4,lora_alpha=8,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')
    model=get_peft_model(base,cfg,adapter_name='reference');model.add_adapter('forward_judge',cfg)
    model.save_pretrained(tmp_path/'initial',selected_adapters=['reference'])
    s=PolicyScorer.__new__(PolicyScorer);s.model=model;s.device='cpu';s.reference_cache={}
    class Tokenizer:
        def save_pretrained(self,path):return None
    s.tokenizer=Tokenizer();pairs=[{'id':'p','prompt_ids':[1,2],'chosen_ids':[3,5],'rejected_ids':[4,5],'tool':'apply_electron_flow'}]
    original=JsonJournal.append
    with monkeypatch.context() as m:
        def interrupt(journal,row):
            if journal.path.name=='training_steps.jsonl':raise RuntimeError('interruption after optimizer commit')
            return original(journal,row)
        m.setattr(JsonJournal,'append',interrupt)
        with pytest.raises(RuntimeError,match='interruption'):
            s.train_dpo(pairs,tmp_path/'initial/reference','r0',tmp_path/'trained',steps=2,accum=1,lr=.001)
    saved=torch.load(tmp_path/'trained/optimizer_recovery.pt',weights_only=False)
    assert saved['step']==1
    result=s.train_dpo(pairs,tmp_path/'initial/reference','r0',tmp_path/'trained',steps=2,accum=1,lr=.001)
    assert result['optimizer_steps']==2 and result['recovered_optimizer_steps']==1
