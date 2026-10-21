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
        self.model.load_adapter(str(f0),adapter_name='forward',is_trainable=False)
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
                with torch.no_grad():self.forward_cache[key]=float(self.logp(p,a,'forward'))
            total+=self.forward_cache[key];tokens+=len(a)
        return {'sum_logp':total,'tokens':tokens,'mean_logp':total/max(tokens,1),
                'metric':'teacher_forced_path_support_NOT_autonomous_success'}

    def train_dpo(self,pairs,initial_adapter,r0_sha,output,*,steps=50,lr=2e-6,beta=.1,accum=4,reference_adapter='reference'):
        import time
        from pathlib import Path
        from .rsi_contract import fingerprint,sha256,write_json,JsonJournal
        if not pairs:return {'status':'no_update','reason':'no_authentic_same_state_pairs','optimizer_steps':0}
        output=Path(output)
        if output.exists():raise ValueError('immutable output exists')
        # Load a fresh trainable adapter for this round, not the current F0 or
        # frozen R0. Both rounds keep the SAME R0 probability reference.
        if 'policy' in self.model.peft_config:self.model.delete_adapter('policy')
        self.model.load_adapter(str(initial_adapter),adapter_name='policy',is_trainable=True)
        before={n:p.detach().clone() for n,p in self.model.named_parameters() if '.policy.' in n}
        references={n:p.detach().clone() for n,p in self.model.named_parameters() if '.reference.' in n or '.forward.' in n}
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
        output.mkdir(parents=True);journal=JsonJournal(output/'training_steps.jsonl');tokens=0;tools={}
        for step in range(steps):
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
            record={'optimizer_step':step+1,'loss':loss_sum,'policy_margin':margin_sum,
                    'grad_norm':float(norm),'elapsed_seconds':time.monotonic()-start,'tool_examples':dict(tools)}
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
                'unique_pairs':len(pairs),'optimizer_steps':steps,'reused_examples':steps*accum,
                'weight_delta_l2':delta,'elapsed_seconds':time.monotonic()-start,'assistant_tokens':tokens,
                'per_tool_examples':tools,'peak_vram_bytes':torch.cuda.max_memory_allocated(self.device)}
        write_json(output/'manifest.json',result);write_json(output/'reference_margins.json',{'reference_sha':r0_sha,'pair_margins':dict(zip([p['id'] for p in pairs],margins))})
        return result
