from collections import Counter
import torch
import pytest


def test_small_pool_cannot_be_repeated_for_200_exposures():
    from mechet.rsi_objective import anchored_plan
    pairs=[{'id':str(i),'source_id':str(i//2)} for i in range(11)]
    replay=[{'id':'real','source_id':'independent','tool':'apply_electron_flow','reference_class':'bond_change'}]
    plan=anchored_plan(pairs,replay)
    counts=Counter(i for step in plan for i in step['pairs'])
    assert len(plan)<=20 and sum(counts.values())<=44 and max(counts.values())<=4
    assert all(len(s['pairs'])==3 for s in plan)


@pytest.mark.parametrize('mode',['replay_only','dpo_only','anchored'])
def test_actual_qwen_lora_anchored_branch_gradients_frozen_reference_and_saved_weight(tmp_path,mode):
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from peft import get_peft_model,LoraConfig
    from mechet.selected_policy_logps import install_selected_logps_forward
    from mechet.rsi_objective import PolicyScorer
    torch.manual_seed(17)
    base=Qwen3ForCausalLM(Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=64,
        num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=64)).eval()
    install_selected_logps_forward(base)
    cfg=LoraConfig(r=4,lora_alpha=8,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')
    model=get_peft_model(base,cfg,adapter_name='reference');model.add_adapter('forward_judge',cfg)
    model.save_pretrained(tmp_path/'initial',selected_adapters=['reference'])
    scorer=PolicyScorer.__new__(PolicyScorer);scorer.model=model;scorer.device='cpu';scorer.reference_cache={}
    class Tokenizer:
        def save_pretrained(self,path):pass
    scorer.tokenizer=Tokenizer()
    refs={n:p.detach().clone() for n,p in model.named_parameters() if '.policy.' not in n}
    pairs=[{'id':'p'+str(i),'source_id':'s'+str(i),'prompt_ids':[1,2,3],
        'chosen_ids':[4,6],'rejected_ids':[5,6],'tool':'apply_electron_flow'} for i in range(3)]
    replay=[{'id':'real','source_id':'other','prompt_ids':[1,2,3],'completion_ids':[7,6],
        'tool':'finish_trace','reference_class':'bond_change'}]
    result=scorer.train_anchored(pairs,replay,tmp_path/'initial/reference','r0',tmp_path/mode,
        mode=mode,max_steps=1,lr=.001)
    assert result['status']=='updated' and result['weight_delta_l2']>0
    assert result['optimizer_steps']==1 and result['scored_assistant_tokens']==14
    assert result['active_loss_tokens']==({'replay_only':2,'dpo_only':12,'anchored':14}[mode])
    assert result['dpo_gradient_norm_max']>0 if mode!='replay_only' else result['dpo_gradient_norm_max']==0
    assert result['replay_gradient_norm_max']>0 if mode!='dpo_only' else result['replay_gradient_norm_max']==0
    assert all(torch.equal(refs[n],p) for n,p in model.named_parameters() if n in refs)
    assert (tmp_path/mode/'policy/adapter_model.safetensors').exists()
    model.delete_adapter('policy')  # fresh process: no previously active control
    again=scorer.train_anchored(pairs,replay,tmp_path/'initial/reference','r0',tmp_path/mode,mode=mode,max_steps=1,lr=.001)
    assert result['weight_sha256']==again['weight_sha256']
    assert 'policy' in model.peft_config


def test_no_preferences_do_not_fabricate_dpo_update(tmp_path):
    from mechet.rsi_objective import PolicyScorer
    scorer=PolicyScorer.__new__(PolicyScorer)
    result=scorer.train_anchored([],[],tmp_path,'r0',tmp_path/'out',mode='anchored')
    assert result['status']=='no_update' and result['optimizer_steps']==0


def test_recursive_controls_share_token_cap_and_update_count_without_cropping_actions():
    from mechet.rsi_objective import matched_round_plans
    def pairs(size):
        return [{'id':str(i),'source_id':str(i),'chosen_ids':[1]*size,'rejected_ids':[2]*size} for i in range(3)]
    replay=[{'id':'real','source_id':'r','tool':'finish_trace','completion_ids':[3,4]}]
    plans,report=matched_round_plans({'dynamic':pairs(10),'static':pairs(6)},replay)
    assert len(plans['dynamic'])==len(plans['static'])==report['optimizer_steps']
    assert all(n<=report['common_scored_assistant_token_cap'] for n in report['spent_assistant_tokens'].values())
    assert report['unused_token_cap']['static']>=0
