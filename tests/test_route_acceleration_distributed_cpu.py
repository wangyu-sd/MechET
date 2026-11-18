"""Two-process transport and real chemistry unit integration, no model experiment."""
from pathlib import Path


def _cpu_worker(rank,rendezvous,root,fault=None):
    import json
    import sqlite3
    from types import SimpleNamespace
    import torch
    import torch.distributed as dist
    import mechet.external_route_io as io
    from mechet.route_policy_sampling import append_receipt
    from mechet.route_policy_protocol import route_tool_prefix
    from mechet.route_process_reward import ProcessRouteReward,ALGORITHM
    from mechet.route_failure_policy import configure_first_rejection
    from mechet.route_refit_runtime import DEFAULTS
    from mechet.route_acceleration import DEFAULT_ACCELERATION
    from mechet.endpoints import ROUTE_ENDPOINT_CONTRACT
    from test_route_native_observation import episode,flow
    root=Path(root);torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    args=flow((lambda ep:(ep.next_decision(),ep)[1])(episode()))
    class Tokenizer:
        pad_token_id=0;pad_token='pad';padding_side='left'
        def apply_chat_template(self,*a,**kw):return 'prompt'
        def __call__(self,*a,**kw):return {'input_ids':[1]}
    class StubBackend:
        def __init__(self,actor,tokenizer,cfg,raw_path,execution_writer):
            if fault=='startup' and rank==1:raise OSError('injected startup failure')
            self.cfg,self.raw_path=cfg,Path(raw_path)
            self.writer=io.ReceiptBatch(raw_path,cfg=cfg)
            self.execution_writer=execution_writer
        def generate(self,requests):
            values=[]
            append_receipt(self.raw_path,dict(event='gpu-requested',requests=[vars(r) for r in requests],prompt_token_ids=[[1] for r in requests]))
            for r in requests:
                name='apply_electron_flow' if r.request_id.endswith('/a0') else 'finish_trace'
                values.append(dict(request_id=r.request_id,behavior_version=r.behavior_version,prompt_token_ids=[1],
                    generated_token_ids=[2],behavior_logprobs=[-.2],raw='<tool_call>'+json.dumps(
                    dict(name=name,arguments=args if name=='apply_electron_flow' else {}))+'</tool_call>',truncated=False))
            for value in values:append_receipt(self.raw_path,dict(event='behavior-receipt',**value))
            return values
    io.BatchedNativeRouteSampler=StubBackend
    cfg=dict(_sha='contract',world_size=2,candidates=4,seed=17,max_action_tokens=2048,
        max_context=32768,generation_batch_size=8,executor_workers_per_rank=2,
        algorithm=ALGORITHM,dataset=str(root),partial_reward_weight=.2,
        route_endpoint_contract=ROUTE_ENDPOINT_CONTRACT,stock_material_contract='exact_material_stock_v1',
        public_observation_contract='route_sft_aligned_feedback_v9',inference_backend='hf',
        refit=DEFAULTS,acceleration=DEFAULT_ACCELERATION,collective_timeout_seconds=30,
        pad_token_id=0,policy_precision='fp32')
    configure_first_rejection(cfg)
    rows=[{'task_id':str(i),'target':'CO'} for i in range(6)]
    from mechet.route_distributed_collection import collect_distributed_round
    if fault=='summary' and rank==0:
        import mechet.route_distributed_collection as module
        original=module.atomic_json
        def publish(path,value):
            if Path(path).name=='scheduler_summary.json':raise OSError('injected summary failure')
            return original(path,value)
        module.atomic_json=publish
    stock={'[CH3+]','[OH-]'}
    with ProcessRouteReward(root/'references.sqlite',stock,failure_policy=cfg['route_failure_policy']) as oracle:
        try:groups=collect_distributed_round(rows,stock,set(),torch.nn.Linear(1,1),Tokenizer(),cfg,root/'round','v1',oracle)
        except RuntimeError as exc:
            if not fault:raise
            (root/f'failure{rank}.txt').write_text(str(exc));dist.destroy_process_group();return
        if fault:raise AssertionError('injected failure was lost')
    torch.save([g[0]['task_id'] for g in groups],root/f'owned{rank}.pt')
    # A rank with zero eligible groups must still join the same gradient sum.
    from copy import deepcopy
    from mechet.external_route_rl import optimize_routes,DR_GRPO_ALGORITHM
    from scripts.pr96_earho_contract import token_logps_batch
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.emb=torch.nn.Embedding(5,3);self.head=torch.nn.Linear(3,5)
        def forward(self,input_ids,attention_mask,logits_to_keep,use_cache,position_ids=None):
            return SimpleNamespace(logits=self.head((self.emb(input_ids)*attention_mask[...,None]).cumsum(1)[:,logits_to_keep]))
    torch.manual_seed(73);tiny=Tiny();ref=deepcopy(tiny)
    old=token_logps_batch(tiny,[([1],[2,3])])[0].detach().tolist()
    record=dict(prompt_token_ids=[1],generated_token_ids=[2,3],behavior_logprobs=old)
    training=[[dict(reward=r,records=[record]) for r in [0.,1.,0.,0.]]] if rank==0 else []
    stats=optimize_routes(tiny,ref,torch.optim.SGD(tiny.parameters(),lr=.1),training,
        dict(algorithm=DR_GRPO_ALGORITHM,candidates=4,loss_batch_size=4,pad_token_id=0,
             clip_epsilon=.2,kl_beta=.02,loss_token_normalizer=1024.,max_grad_norm=1.,behavior_logprob_tolerance=.25))
    assert stats['global_policy_groups']==1 and stats['optimizer_updated']
    torch.save(tiny.state_dict(),root/f'updated{rank}.pt')
    dist.destroy_process_group()


def _database(tmp_path):
    import sqlite3
    from mechet.external_route_data import edge_signature
    with sqlite3.connect(tmp_path/'references.sqlite') as db:
        db.execute('CREATE TABLE tasks (id TEXT,target TEXT,split TEXT,reference_steps INTEGER)')
        db.execute('CREATE TABLE edges (task_id TEXT,signature TEXT)')
        for i in range(6):
            db.execute('INSERT INTO tasks VALUES (?,?,?,?)',(str(i),'CO','training',1))
            db.execute('INSERT INTO edges VALUES (?,?)',(str(i),edge_signature('CO',['[CH3+]','[OH-]'])))


def test_two_process_global_collection_routes_real_chemistry_and_preserves_denominator(tmp_path,monkeypatch):
    monkeypatch.setenv('MKL_THREADING_LAYER','GNU')
    import json
    import torch
    import torch.multiprocessing as mp
    _database(tmp_path)
    mp.spawn(_cpu_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path)),nprocs=2,join=True)
    owned=[x for r in range(2) for x in torch.load(tmp_path/f'owned{r}.pt',weights_only=True)]
    assert sorted(owned)==list(map(str,range(6)))
    summary=json.loads((tmp_path/'round/scheduler_summary.json').read_text())
    assert summary['complete']==summary['expected']==6 and not summary['error']
    assert summary['counts']['completed_requests']>0
    a=torch.load(tmp_path/'updated0.pt',weights_only=True);b=torch.load(tmp_path/'updated1.pt',weights_only=True)
    for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)


def test_single_rank_startup_error_reaches_both_ranks(tmp_path,monkeypatch):
    _check_fault(tmp_path,monkeypatch,'startup')


def test_summary_publication_error_reaches_both_ranks(tmp_path,monkeypatch):
    _check_fault(tmp_path,monkeypatch,'summary')


def _check_fault(tmp_path,monkeypatch,kind):
    monkeypatch.setenv('MKL_THREADING_LAYER','GNU')
    import time
    import torch.multiprocessing as mp
    _database(tmp_path)
    ctx=mp.spawn(_cpu_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path),kind),nprocs=2,join=False)
    deadline=time.monotonic()+40
    while time.monotonic()<deadline:
        if ctx.join(timeout=1):break
    else:
        for p in ctx.processes:p.terminate();p.join(5)
        raise AssertionError('collective failure caused a hang')
    for rank in range(2):assert 'injected '+kind+' failure' in (tmp_path/f'failure{rank}.txt').read_text()
