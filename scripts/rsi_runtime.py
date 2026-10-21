"""Bounded independent Qwen episodes; GPU owner overlaps independent CPU tools."""
from __future__ import annotations
import copy
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from mechet.rsi_contract import stable_seed,fingerprint,JsonJournal


def rolling_agent_batch(controllers,make_request,consume,owner,*,cpu_workers=16,max_inflight=64,on_done=None):
    ready=deque(controllers);gpu={};cpu={};done=[]
    with ThreadPoolExecutor(max_workers=cpu_workers) as pool:
        while ready or gpu or cpu:
            while ready and len(gpu)+len(cpu)<max_inflight:
                c=ready.popleft();request=make_request(c)
                if request is None:c['done']=True;done.append(c);on_done and on_done(c);continue
                gpu[owner.submit(request)]=c
            output=owner.poll(timeout=.005)
            if output:
                identifier,result=output
                if identifier not in gpu:raise ValueError('unknown or duplicate GPU callback')
                c=gpu.pop(identifier);cpu[pool.submit(consume,c,result)]=c
            for future in list(cpu):
                if not future.done():continue
                c=future.result();del cpu[future]
                if c.get('done'):done.append(c);on_done and on_done(c)
                else:ready.append(c)
    return done


class GPUOwner:
    """Exactly one caller owns synchronous vLLM, including add/step/abort."""
    def __init__(self,engine,params_factory,*,max_inflight=64):
        self.engine=engine;self.params_factory=params_factory;self.capacity=max_inflight
        self.inputs=queue.Queue(maxsize=max_inflight);self.outputs=queue.Queue();self.stop=threading.Event()
        self.thread=threading.Thread(target=self._run,daemon=True,name='RSI-vllm-owner');self.thread.start()
    def submit(self,request):
        if self.stop.is_set():raise RuntimeError('GPU owner stopped')
        identifier=uuid.uuid4().hex;self.inputs.put((identifier,request));return identifier
    def poll(self,timeout=0.):
        try:identifier,result,error=self.outputs.get(timeout=timeout)
        except queue.Empty:return None
        if error:raise RuntimeError(str(error)) from error
        return identifier,result
    def _run(self):
        active=set()
        try:
            while not self.stop.is_set():
                while len(active)<self.capacity:
                    try:identifier,request=self.inputs.get(timeout=.01 if not active else 0.)
                    except queue.Empty:break
                    self.engine.add_request(identifier,{'prompt_token_ids':request['prompt_ids']},
                        self.params_factory(request),lora_request=request['lora'])
                    active.add(identifier)
                if active:
                    for output in self.engine.step():
                        if output.finished:
                            active.remove(str(output.request_id));self.outputs.put((str(output.request_id),output,None))
        except BaseException as exc:self.outputs.put((None,None,exc))
        finally:
            if active:self.engine.abort_request(list(active))
    def close(self):
        self.stop.set();self.thread.join(30)
        if self.thread.is_alive():raise RuntimeError('GPU owner did not stop safely')


def reverse_prompt(node,tokenizer):
    from mechet.assistant_masking import render_qwen_sft_tool_prefix
    from scripts.run_natural_language_value_search import policy_prompt,SYSTEM,TOOLS
    messages=[{'role':'system','content':SYSTEM},{'role':'user','content':policy_prompt(node.target,node.state,
        include_inventory=True,actions=node.actions,compact_history=True)}]
    return render_qwen_sft_tool_prefix(tokenizer,messages,tools=TOOLS)


def forward_prompt_text(state,history,tokenizer):
    from mechet.forward_qwen_events import SYSTEM,forward_prompt,forward_tools
    from mechet.assistant_masking import render_qwen_sft_tool_prefix
    messages=[{'role':'system','content':SYSTEM},{'role':'user','content':forward_prompt(state,history)}]
    return render_qwen_sft_tool_prefix(tokenizer,messages,tools=forward_tools())


def collect_episodes(rows,*,owner,tokenizer,lora,actor_sha,round_id,sample_indices,
                     direction='reverse',journal_path,max_attempts=24,max_tokens=384,cpu_workers=16):
    from scripts.run_natural_language_value_search import Node,private_product_state,visible
    from mechet.in_place_grounded_flow import mapped_atom_numbers
    controllers=[];raw=JsonJournal(str(journal_path)+'.actions.jsonl');episodes=JsonJournal(journal_path)
    for row in rows:
        for candidate in sample_indices:
            state=private_product_state(row['target']) if direction=='reverse' else row['initial_state']
            node=Node(target=row.get('target',''),state=state,next_map=max(mapped_atom_numbers(state))+1,visited={visible(state)})
            controllers.append({'id':row['id'],'episode_id':fingerprint([round_id,actor_sha,row['id'],candidate,direction]),
              'candidate':candidate,'round':round_id,'actor_sha':actor_sha,'node':node,'initial_state':state,
              'state':state,'target':row.get('target',''),'steps':[],'history':[],'flow_count':0,'terminal':False})
    start=time.monotonic()
    def request(c):
        if len(c['steps'])>=max_attempts:c['stop_reason']='ATTEMPT_BUDGET';return None
        prompt=reverse_prompt(c['node'],tokenizer) if direction=='reverse' else forward_prompt_text(c['state'],c['history'],tokenizer)
        ids=tokenizer.encode(prompt,add_special_tokens=False)
        if len(ids)+max_tokens>16384:c['stop_reason']='CONTEXT_BUDGET';return None
        c['request']={'prompt_text':prompt,'prompt_ids':ids,'seed':stable_seed(17,round_id if direction=='reverse' else 'frozen_forward',c['id'],c['candidate'],len(c['steps']),actor_sha),
                      'max_tokens':max_tokens,'lora':lora};return c['request']
    def consume(c,result):
        cpu_start=time.monotonic()
        from scripts.eval_natural_language_event_local import prediction_call
        from scripts.run_natural_language_value_search import Action,execute
        from mechet.forward_qwen_events import execute_forward_call
        out=result.outputs[0];ids=list(out.token_ids);raw_text=tokenizer.decode(ids,skip_special_tokens=False)
        eos=tokenizer.convert_tokens_to_ids('<|im_end|>');complete=bool(ids and ids[-1]==eos)
        name,arguments,error=prediction_call(raw_text,tokenizer)
        if not complete:error='TRUNCATED_ACTION_WITHOUT_EOS'
        before=c['state'];after=before;accepted=False
        if not error:
            if direction=='reverse':
                new,error=execute(c['node'],Action(name,arguments,raw_text,0.,len(ids)),max_imports=32,reject_target_retained_finish=True)
                if new is not None:c['node']=new;after=new.state;accepted=True;c['terminal']=new.terminal
            else:
                try:
                    replay=execute_forward_call(before,name,arguments)
                    if not replay.get('ok'):error=str(replay.get('code','FORWARD_EXECUTION_FAILED'))
                    else:
                        after=replay['state_smiles'];accepted=True;c['terminal']=bool(replay.get('terminal'))
                        c['history'].append({'name':name,'arguments':arguments})
                except Exception as exc:error=f'{type(exc).__name__}:{exc}'
        step={'attempt':len(c['steps']),'prompt_text':c['request']['prompt_text'],'prompt_ids':c['request']['prompt_ids'],
              'completion_ids':ids,'raw':raw_text,'complete':complete,'name':name,'arguments':arguments,'accepted':accepted,'error':error,
              'state_before':before,'state_after':after,'successor':visible(after) if accepted else None,
              'seed':c['request']['seed'],'finish_reason':str(out.finish_reason),
              'cpu_executor_wall_seconds':time.monotonic()-cpu_start,
              'generated_sum_logp':sum(float(scores[token].logprob) for scores,token in zip(out.logprobs,ids)) if out.logprobs is not None else None}
        raw.append({'episode_id':c['episode_id'],'actor_sha':actor_sha,**step});c['steps'].append(step);c['state']=after
        if accepted and name in {'apply_electron_flow','apply_forward_electron_flow'}:c['flow_count']+=1
        c['done']=c['terminal'] or len(c['steps'])>=max_attempts
        return c
    def finish(c):
        c['final_state']=c['state'];c['elapsed_seconds']=time.monotonic()-start
        saved={k:v for k,v in c.items() if k not in {'node','request'}};episodes.append(saved)
        print({'phase':'RSI-episode','direction':direction,'round':round_id,'id':c['id'],'candidate':c['candidate'],
               'terminal':c['terminal'],'attempts':len(c['steps']),'elapsed_seconds':c['elapsed_seconds']},flush=True)
    try:result=rolling_agent_batch(controllers,request,consume,owner,cpu_workers=cpu_workers,on_done=finish)
    finally:raw.close();episodes.close()
    return [{k:v for k,v in c.items() if k not in {'node','request'}} for c in result]
