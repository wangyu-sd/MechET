"""Full-action DPO, exact selected-token scoring, immutable reference margins."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def dpo_loss(chosen,rejected,reference_margin,*,beta=.1):
    return -F.logsigmoid(beta*(chosen-rejected-reference_margin.detach()))


def action_inputs(prompt_ids,completion_ids,*,device):
    if not prompt_ids or not completion_ids:raise ValueError('empty prompt or action')
    ids=torch.tensor([list(prompt_ids)+list(completion_ids)],device=device)
    mask=torch.tensor([[0]*len(prompt_ids)+[1]*len(completion_ids)],device=device)
    return ids,mask


class PolicyScorer:
    def train_anchored(self,pairs,replay,initial_adapter,r0_sha,output,**options):
        return train_anchored(self,pairs,replay,initial_adapter,r0_sha,output,**options)

    def __init__(self,base_path,r0,f0,device='cuda:1'):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from .selected_policy_logps import install_selected_logps_forward
        from .rsi_contract import BASE_REVISION
        self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(base_path,revision=BASE_REVISION,local_files_only=True)
        base=AutoModelForCausalLM.from_pretrained(base_path,revision=BASE_REVISION,local_files_only=True,
             torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device)
        install_selected_logps_forward(base)
        self.model=PeftModel.from_pretrained(base,str(r0),adapter_name='reference',is_trainable=False)
        self.model.load_adapter(str(f0),adapter_name='forward_judge',is_trainable=False)
        self.model.eval();self.reference_cache={};self.forward_cache={}
        for p in self.model.parameters():p.requires_grad_(False)

    def logp(self,prompt_ids,completion_ids,adapter):
        if len(prompt_ids)+len(completion_ids)>16384:raise ValueError('scoring context exceeded; no cropping allowed')
        self.model.set_adapter(adapter)
        # PEFT set_adapter changes requires_grad; never let reference/forward
        # parameters become trainable as a side effect of selecting them.
        for name,p in self.model.named_parameters():p.requires_grad_('.policy.' in name and adapter=='policy')
        ids,mask=action_inputs(prompt_ids,completion_ids,device=self.device)
        return self.model(input_ids=ids,attention_mask=torch.ones_like(ids),loss_token_mask=mask)['selected_logps'].sum()

    def witness_score(self,witness,f0_sha):
        from .assistant_masking import render_chat,render_qwen_sft_tool_prefix
        from .forward_qwen_events import forward_tools
        from .rsi_contract import fingerprint
        total=0.;tokens=0
        for step in witness:
            messages=step['messages'];tools=forward_tools()
            prompt=render_qwen_sft_tool_prefix(self.tokenizer,messages,tools=tools)
            full=render_chat(self.tokenizer,messages+[{'role':'assistant','content':'','tool_calls':[{
                'id':'rsi-witness','type':'function','function':{'name':step['name'],'arguments':step['arguments']}}]}],tools=tools)
            if not full.startswith(prompt):raise ValueError('witness SFT prefix mismatch')
            p=self.tokenizer.encode(prompt,add_special_tokens=False)
            a=self.tokenizer.encode(full[len(prompt):],add_special_tokens=False)
            # Match sampled complete calls: include im_end, not the trailing
            # template newline after EOS.
            eos=self.tokenizer.convert_tokens_to_ids('<|im_end|>')
            if eos not in a:raise ValueError('witness missing EOS')
            a=a[:a.index(eos)+1];key=fingerprint([f0_sha,p,a,'full_inverse_terminal_mixture_v1'])
            if key not in self.forward_cache:
                with torch.no_grad():self.forward_cache[key]=float(self.logp(p,a,'forward_judge'))
            total+=self.forward_cache[key];tokens+=len(a)
        return {'sum_logp':total,'tokens':tokens,'mean_logp':total/max(tokens,1),
                'metric':'teacher_forced_path_support_NOT_autonomous_success'}

    def train_dpo(self,pairs,initial_adapter,r0_sha,output,*,steps=50,lr=2e-6,beta=.1,accum=4,reference_adapter='reference'):
        import time
        from pathlib import Path
        from .rsi_contract import fingerprint,sha256,write_json,JsonJournal
        if not pairs:return {'status':'no_update','reason':'no_authentic_same_state_pairs','optimizer_steps':0}
        output=Path(output)
        contract=fingerprint({'pairs':pairs,'parent_sha':sha256(Path(initial_adapter)/'adapter_model.safetensors'),
            'reference_sha':r0_sha,'steps':steps,'lr':lr,'beta':beta,'accum':accum,'reference_adapter':reference_adapter})
        if (output/'contract.json').exists():
            import json
            if json.loads((output/'contract.json').read_text())['sha']!=contract:raise ValueError('DPO resume contract changed')
            if (output/'manifest.json').exists():
                previous=json.loads((output/'manifest.json').read_text())
                if sha256(Path(previous['path'])/'adapter_model.safetensors')!=previous['weight_sha256']:raise ValueError('completed adapter corrupt')
                return previous
        # Load a fresh trainable adapter for this round, not the current F0 or
        # frozen R0. Both rounds keep the SAME R0 probability reference.
        if 'policy' in self.model.peft_config:self.model.delete_adapter('policy')
        self.model.load_adapter(str(initial_adapter),adapter_name='policy',is_trainable=True)
        before={n:p.detach().clone() for n,p in self.model.named_parameters() if '.policy.' in n}
        references={n:p.detach().clone() for n,p in self.model.named_parameters() if '.reference.' in n or '.forward_judge.' in n}
        margins=[]
        for pair in pairs:
            key=fingerprint([r0_sha,pair['prompt_ids'],pair['chosen_ids'],pair['rejected_ids']])
            if key not in self.reference_cache:
                with torch.no_grad():self.reference_cache[key]=float(self.logp(pair['prompt_ids'],pair['chosen_ids'],reference_adapter)-self.logp(pair['prompt_ids'],pair['rejected_ids'],reference_adapter))
            margins.append(self.reference_cache[key])
        self.model.eval() # gradients enabled, dropout disabled for DPO likelihoods
        self.model.set_adapter('policy')
        parameters=[p for n,p in self.model.named_parameters() if '.policy.' in n]
        optimizer=torch.optim.AdamW(parameters,lr=lr);start=time.monotonic()
        output.mkdir(parents=True,exist_ok=True);write_json(output/'contract.json',{'sha':contract})
        write_json(output/'reference_margins.json',{'reference_sha':r0_sha,'pair_margins':dict(zip([p['id'] for p in pairs],margins))})
        journal=JsonJournal(output/'training_steps.jsonl');tokens=0;tools={};start_step=0;previous_elapsed=0.
        recovery=output/'optimizer_recovery.pt'
        if recovery.exists():
            from peft import set_peft_model_state_dict
            saved=torch.load(recovery,map_location=self.device,weights_only=False)
            if saved['contract']!=contract:raise ValueError('optimizer recovery contract mismatch')
            set_peft_model_state_dict(self.model,saved['adapter'],adapter_name='policy')
            optimizer.load_state_dict(saved['optimizer']);start_step=saved['step'];tokens=saved['tokens'];tools=saved['tools']
            previous_elapsed=saved.get('elapsed_seconds',0.)
        for step in range(start_step,steps):
            optimizer.zero_grad();loss_sum=0.;margin_sum=0.
            for micro in range(accum):
                index=(step*accum+micro)%len(pairs);pair=pairs[index]
                cp=self.logp(pair['prompt_ids'],pair['chosen_ids'],'policy')
                rp=self.logp(pair['prompt_ids'],pair['rejected_ids'],'policy')
                loss=dpo_loss(cp,rp,cp.new_tensor(margins[index]),beta=beta)
                (loss/accum).backward();loss_sum+=float(loss.detach())/accum
                margin_sum+=float((cp-rp).detach())/accum
                tokens+=len(pair['chosen_ids'])+len(pair['rejected_ids']);tools[pair['tool']]=tools.get(pair['tool'],0)+1
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(norm):raise ValueError('nonfinite DPO gradients')
            optimizer.step()
            import os
            from peft import get_peft_model_state_dict
            temporary=recovery.with_suffix('.tmp')
            with temporary.open('wb') as handle:
                torch.save({'contract':contract,'step':step+1,'tokens':tokens,'tools':tools,'elapsed_seconds':previous_elapsed+time.monotonic()-start,
                    'adapter':get_peft_model_state_dict(self.model,adapter_name='policy'),
                    'optimizer':optimizer.state_dict()},handle);handle.flush();os.fsync(handle.fileno())
            temporary.replace(recovery)
            record={'optimizer_step':step+1,'loss':loss_sum,'policy_margin':margin_sum,
                    'grad_norm':float(norm),'elapsed_seconds':previous_elapsed+time.monotonic()-start,'tool_examples':dict(tools)}
            journal.append(record);print({'phase':'RSI-DPO',**record},flush=True)
        journal.close()
        delta=sum(float((p.detach()-before[n]).float().square().sum()) for n,p in self.model.named_parameters() if n in before)**.5
        if not delta:raise ValueError('DPO had no actual weight change')
        for n,p in self.model.named_parameters():
            if n in references and not torch.equal(p.detach(),references[n]):raise ValueError('frozen judge/reference mutated')
        self.model.save_pretrained(output,selected_adapters=['policy'],safe_serialization=True)
        adapter=output/'policy';self.tokenizer.save_pretrained(adapter)
        result={'status':'updated','path':str(adapter),'weight_sha256':sha256(adapter/'adapter_model.safetensors'),
                'parent_weight_sha256':sha256(Path(initial_adapter)/'adapter_model.safetensors'),'reference_sha':r0_sha,
                'unique_pairs':len(pairs),'optimizer_steps':steps,'recovered_optimizer_steps':start_step,'reused_examples':steps*accum,
                'weight_delta_l2':delta,'elapsed_seconds':previous_elapsed+time.monotonic()-start,'assistant_tokens':tokens,
                'per_tool_examples':tools,'peak_vram_bytes':torch.cuda.max_memory_allocated(self.device) if str(self.device).startswith('cuda') else 0}
        write_json(output/'manifest.json',result);write_json(output/'reference_margins.json',{'reference_sha':r0_sha,'pair_margins':dict(zip([p['id'] for p in pairs],margins))})
        return result


def anchored_plan(pairs,replay,*,max_steps=20,max_exposures=4):
    """One source-balanced, common scored-work plan for all three controls.

    Ablations disable a loss, not its scored examples. Active-gradient token
    budgets necessarily differ; these are explicitly reported, not conflated.
    """
    from collections import defaultdict,deque
    from .rsi_contract import stable_seed
    if not replay:return []
    groups=defaultdict(list)
    for i,pair in enumerate(pairs):groups[pair['source_id']].append(i)
    order=[]
    queues=[deque(indices) for _,indices in sorted(groups.items())]
    while any(queues):
        for q in queues:
            if q:order.append(q.popleft())
    # Replay stratification alternates tool/class bins and independent sources.
    bins=defaultdict(list)
    for i,row in enumerate(replay):bins[(row['tool'],row.get('reference_class','unknown'))].append(i)
    rq=[deque(sorted(indices,key=lambda i:stable_seed(17,replay[i]['source_id'],replay[i]['id'])))
        for _,indices in sorted(bins.items())]
    real=[]
    while any(rq):
        for q in rq:
            if q:real.append(q.popleft())
    uses=order*max_exposures
    count=min(max_steps,len(uses)//3) if pairs else min(max_steps,len(real))
    return [{'pairs':uses[step*3:step*3+3] if pairs else [],'replay':real[step%len(real)]}
            for step in range(count)]


def matched_round_plans(pools,replay):
    """Same recursive update count and common action-token *cap* for both arms.

    Complete tool calls are indivisible: actual spent/unused tokens differ and
    are disclosed. No partial-action cropping or fake padding work is used.
    """
    plans={name:anchored_plan(pairs,replay) for name,pairs in pools.items()}
    def cost(name,step):
        return sum(len(pools[name][i]['chosen_ids'])+len(pools[name][i]['rejected_ids']) for i in step['pairs'])+len(replay[step['replay']]['completion_ids'])
    cap=min(sum(cost(name,s) for s in plan) for name,plan in plans.items())
    fit={}
    for name,plan in plans.items():
        used=0;count=0
        for step in plan:
            n=cost(name,step)
            if used+n>cap:break
            used+=n;count+=1
        fit[name]=count
    steps=min(fit.values())
    plans={name:plan[:steps] for name,plan in plans.items()}
    spent={name:sum(cost(name,s) for s in plan) for name,plan in plans.items()}
    return plans,{'optimizer_steps':steps,'common_scored_assistant_token_cap':cap,
        'spent_assistant_tokens':spent,'unused_token_cap':{name:cap-n for name,n in spent.items()},
        'boundary':'equal_token_cap_and_updates_not_exact_identical_spent_tokens'}


def train_anchored(scorer,pairs,replay,initial_adapter,r0_sha,output,*,mode='anchored',
                   max_steps=20,max_exposures=4,lr=2e-6,beta=.1,replay_weight=.25,plan_override=None):
    import json,os,time
    from collections import Counter
    from pathlib import Path
    from .rsi_contract import fingerprint,sha256,write_json,JsonJournal
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    if mode not in {'anchored','dpo_only','replay_only'}:raise ValueError('unknown anchored control')
    if (not pairs and mode!='replay_only') or not replay:
        return {'status':'no_update','reason':'no_strong_preferences_or_authentic_replay','optimizer_steps':0}
    if len({p['id'] for p in pairs})!=len(pairs):raise ValueError('duplicate unique preferences')
    plan=plan_override if plan_override is not None else anchored_plan(pairs,replay,max_steps=max_steps,max_exposures=max_exposures)
    exposures=Counter(i for step in plan for i in step['pairs'])
    if len(plan)>max_steps or any(v>max_exposures for v in exposures.values()):
        raise ValueError('unique preference exposure / update cap exceeded')
    if not plan:return {'status':'no_update','reason':'insufficient_independent_pairs_for_three_slots','optimizer_steps':0}
    output=Path(output);parent_sha=sha256(Path(initial_adapter)/'adapter_model.safetensors')
    identity={'pairs':pairs,'replay':replay,'parent_sha':parent_sha,'reference_sha':r0_sha,
        'mode':mode,'plan':plan,'lr':lr,'beta':beta,'replay_weight':replay_weight,
        'objective':'full_action_DPO_plus_token_mean_real_CE_v1'}
    contract=fingerprint(identity)
    if (output/'contract.json').exists():
        if json.loads((output/'contract.json').read_text())['sha']!=contract:raise ValueError('anchored resume contract changed')
        if (output/'manifest.json').exists():
            previous=json.loads((output/'manifest.json').read_text())
            if sha256(Path(previous['path'])/'adapter_model.safetensors')!=previous['weight_sha256']:raise ValueError('saved anchored adapter corrupt')
            # A resumed process may have no policy adapter, or the previous
            # control's policy. Restore the exact saved one before retention.
            if 'policy' in scorer.model.peft_config:scorer.model.delete_adapter('policy')
            scorer.model.load_adapter(previous['path'],adapter_name='policy',is_trainable=True)
            return previous
    if 'policy' in scorer.model.peft_config:scorer.model.delete_adapter('policy')
    scorer.model.load_adapter(str(initial_adapter),adapter_name='policy',is_trainable=True)
    before={n:p.detach().clone() for n,p in scorer.model.named_parameters() if '.policy.' in n}
    frozen={n:p.detach().clone() for n,p in scorer.model.named_parameters() if '.reference.' in n or '.forward_judge.' in n}
    margins=[]
    for pair in pairs:
        key=fingerprint([r0_sha,pair['prompt_ids'],pair['chosen_ids'],pair['rejected_ids']])
        if key not in scorer.reference_cache:
            with torch.no_grad():scorer.reference_cache[key]=float(scorer.logp(pair['prompt_ids'],pair['chosen_ids'],'reference')-
                scorer.logp(pair['prompt_ids'],pair['rejected_ids'],'reference'))
        margins.append(scorer.reference_cache[key])
    scorer.model.eval();scorer.model.set_adapter('policy')
    parameters=[p for n,p in scorer.model.named_parameters() if '.policy.' in n]
    optimizer=torch.optim.AdamW(parameters,lr=lr)
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'contract.json',{'sha':contract,'mode':mode,'plan':plan,'budget_boundary':
        'equal_scored_examples_updates; active_loss_tokens_differ_by_ablation'})
    write_json(output/'reference_margins.json',{'reference_sha':r0_sha,'margins':dict(zip([p['id'] for p in pairs],margins))})
    journal=JsonJournal(output/'training_steps.jsonl');start=time.monotonic();first=0
    totals={'scored_assistant_tokens':0,'active_loss_tokens':0,'dpo_gradient_norm_max':0.,'replay_gradient_norm_max':0.,
        'elapsed_seconds':0.,'replay_examples':0,'pair_uses':0,'tool_examples':{}}
    recovery=output/'optimizer_recovery.pt'
    if recovery.exists():
        saved=torch.load(recovery,map_location=scorer.device,weights_only=False)
        if saved['contract']!=contract:raise ValueError('anchored optimizer recovery mismatch')
        set_peft_model_state_dict(scorer.model,saved['adapter'],adapter_name='policy')
        optimizer.load_state_dict(saved['optimizer']);first=saved['step'];totals=saved['totals']
    elapsed_before=totals['elapsed_seconds']
    def grad_norm():
        return sum(float(p.grad.detach().float().square().sum()) for p in parameters if p.grad is not None)**.5
    for step in range(first,len(plan)):
        optimizer.zero_grad(set_to_none=True);branch=plan[step];dpo_value=0.;margin_value=0.;initial_margin=0.
        for index in branch['pairs']:
            pair=pairs[index]
            context=torch.no_grad() if mode=='replay_only' else torch.enable_grad()
            with context:
                cp=scorer.logp(pair['prompt_ids'],pair['chosen_ids'],'policy')
                rp=scorer.logp(pair['prompt_ids'],pair['rejected_ids'],'policy')
                loss=dpo_loss(cp,rp,cp.new_tensor(margins[index]),beta=beta)
                if mode!='replay_only':(loss/len(branch['pairs'])).backward()
            dpo_value+=float(loss.detach())/len(branch['pairs'])
            margin_value+=float((cp-rp).detach())/len(branch['pairs']);initial_margin+=margins[index]/len(branch['pairs'])
            tokens=len(pair['chosen_ids'])+len(pair['rejected_ids'])
            totals['scored_assistant_tokens']+=tokens;totals['pair_uses']+=1
            if mode!='replay_only':totals['active_loss_tokens']+=tokens
            totals['tool_examples'][pair['tool']]=totals['tool_examples'].get(pair['tool'],0)+1
        dpo_norm=grad_norm()
        dpo_grads=[p.grad.detach().clone() if p.grad is not None else None for p in parameters]
        real=replay[branch['replay']]
        context=torch.no_grad() if mode=='dpo_only' else torch.enable_grad()
        with context:
            ce=-scorer.logp(real['prompt_ids'],real['completion_ids'],'policy')/len(real['completion_ids'])
            if mode!='dpo_only':(replay_weight*ce).backward()
        replay_norm=sum(float((p.grad.detach()-(g if g is not None else 0)).float().square().sum())
            for p,g in zip(parameters,dpo_grads) if p.grad is not None)**.5
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(norm):raise ValueError('nonfinite anchored gradient')
        optimizer.step()
        tokens=len(real['completion_ids']);totals['scored_assistant_tokens']+=tokens
        if mode!='dpo_only':totals['active_loss_tokens']+=tokens
        totals['replay_examples']+=1;totals['elapsed_seconds']=elapsed_before+time.monotonic()-start
        totals['dpo_gradient_norm_max']=max(totals['dpo_gradient_norm_max'],dpo_norm)
        totals['replay_gradient_norm_max']=max(totals['replay_gradient_norm_max'],replay_norm)
        record={'phase':'RSI-anchored','mode':mode,'optimizer_step':step+1,'dpo_loss':dpo_value,
            'token_mean_replay_ce':float(ce.detach()),'replay_weight':replay_weight,
            'dpo_gradient_norm':dpo_norm,'weighted_replay_gradient_norm':replay_norm,'combined_gradient_norm':float(norm),
            'behavior_chosen_rejected_logratio_drift':margin_value-initial_margin,
            'drift_is_pair_margin_not_full_behavior_KL':True,**totals}
        temporary=recovery.with_suffix('.tmp')
        with temporary.open('wb') as f:
            torch.save({'contract':contract,'step':step+1,'totals':totals,'optimizer':optimizer.state_dict(),
                'adapter':get_peft_model_state_dict(scorer.model,adapter_name='policy')},f);f.flush();os.fsync(f.fileno())
        temporary.replace(recovery);journal.append(record);print(record,flush=True)
    journal.close()
    delta=sum(float((p.detach()-before[n]).float().square().sum()) for n,p in scorer.model.named_parameters() if n in before)**.5
    if not delta:raise ValueError('anchored update changed no parameters')
    if any(not torch.equal(p.detach(),frozen[n]) for n,p in scorer.model.named_parameters() if n in frozen):
        raise ValueError('frozen original R0/F0 mutated')
    scorer.model.save_pretrained(output,selected_adapters=['policy'],safe_serialization=True)
    adapter=output/'policy';scorer.tokenizer.save_pretrained(adapter)
    result={'status':'updated','path':str(adapter),'weight_sha256':sha256(adapter/'adapter_model.safetensors'),
        'mode':mode,'parent_weight_sha256':parent_sha,'reference_sha':r0_sha,'unique_pairs':len(pairs),
        'independent_pair_sources':len({p['source_id'] for p in pairs}), 'optimizer_steps':len(plan),
        'pair_exposures':{pairs[i]['id']:v for i,v in exposures.items()},'weight_delta_l2':delta,
        'scored_plan_sha':fingerprint(plan),'budget_boundary':'equal_scored_work_not_equal_active_gradient_tokens',
        'recovered_optimizer_steps':first,**totals,
        'peak_vram_bytes':torch.cuda.max_memory_allocated(scorer.device) if str(scorer.device).startswith('cuda') else 0}
    write_json(output/'manifest.json',result)
    return result
