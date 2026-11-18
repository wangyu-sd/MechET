"""Dynamic whole-target leases plus same-version GPU inference for all owners."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
from threading import Thread

from .external_route_data import file_sha,signature
from .route_training_contract import atomic_json


def owner_collection_config(cfg,broker,cache):
    return dict(cfg,_policy_broker=broker,_execution_cache=cache,executor_workers_per_rank=1,
                physical_executor_concurrency=cfg['world_size']*cfg['executor_workers_per_rank'])


def collect_owned_targets(broker,rank,directory,collect,*,workers=16):
    directory=Path(directory)
    def work():
        while True:
            job=broker.claim(rank)
            if job is None:return
            row=job['task'];target_dir=directory/'groups'/signature(row['task_id'])
            try:
                groups=collect(row,target_dir)
                if len(groups)!=1 or len(groups[0])!=4:raise ValueError('DYNAMIC_GROUP_NOT_FULL_G4')
                path=target_dir/'scored_group.json';atomic_json(path,groups[0])
                broker.finish(job['lease_id'],rank,str(path),file_sha(path))
            except BaseException as exc:
                broker.fail(type(exc).__name__+':'+str(exc));raise
    with ThreadPoolExecutor(workers) as pool:
        futures=[pool.submit(work) for _ in range(workers)]
        for future in futures:future.result()
    return [json.loads(Path(r['result_path']).read_text()) for r in broker.results() if r['rank']==rank]


def collect_distributed_round(rows,stock,protected,actor,tokenizer,cfg,directory,version,oracle):
    import torch
    import torch.distributed as dist
    from scripts.train_external_route_rl import collect_round,score_groups,backend_collection_metrics
    from .route_work_broker import WorkBroker
    from .route_global_sampler import broker_server,connect_broker,serve_gpu,LockedTokenizer
    from .route_gpu_backend import make_gpu_sampler,close_gpu_sampler,recover_gpu_journal
    from scripts.pr96_earho_contract import token_logps_batch
    rank,world=dist.get_rank(),dist.get_world_size()
    if int(os.environ.get('LOCAL_WORLD_SIZE',world))!=world:
        raise ValueError('PR111_FIRST_VERSION_REQUIRES_SINGLE_FOUR_GPU_HOST')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    server=None;control=[None];error=''
    if rank==0:
        try:
            from .route_acceleration import dispatch_settings
            state=WorkBroker(rows,actor_version=version,contract_sha=cfg['_sha'],journal_path=directory/'claims.jsonl',
                             dispatch=dispatch_settings(cfg))
            # Restore physical GPU work once, regardless of which CPU owner or
            # GPU receives the request after restart. No fresh sample is forged.
            def recompute(p,r):
                from .route_precision import precision_mode
                with torch.no_grad():
                    return token_logps_batch(actor,[(p,r)],cfg['pad_token_id'],precision=precision_mode(cfg))[0].tolist()
            for gpu in range(world):
                path=directory/f'gpu-rank{gpu:02d}.jsonl'
                for request,value in recover_gpu_journal(path,version,cfg,recompute):
                    state.seed_result(request,value)
            if cfg.get('acceleration',{}).get('version')=='pr111_full_acceleration_v2':
                from .route_request_accounting import restore_finished_request_labels
                restore_finished_request_labels(state,version)
            server,address,key=broker_server(state,host='127.0.0.1')
            control[0]=dict(address=address,key=key,error='')
        except BaseException as exc:control[0]=dict(error=type(exc).__name__+':'+str(exc))
    dist.broadcast_object_list(control,src=0)
    if control[0]['error']:raise RuntimeError('DYNAMIC_BROKER_START_FAILED:'+control[0]['error'])
    broker=worker=sampler=None
    try:
        broker=connect_broker(control[0]['address'],control[0]['key'])
        tokenizer=LockedTokenizer(tokenizer)
        sampler=make_gpu_sampler(actor,tokenizer,cfg,directory,version,rank)
        worker=Thread(target=serve_gpu,args=(broker,sampler,rank),daemon=True);worker.start()
    except BaseException as exc:error=type(exc).__name__+':'+str(exc)
    startup_errors=[None]*world;dist.all_gather_object(startup_errors,error)
    if any(startup_errors):
        if broker:broker.fail(json.dumps(startup_errors))
        if worker and worker.ident is not None:worker.join(timeout=5)
        if sampler is not None and (worker is None or not worker.is_alive()):close_gpu_sampler(sampler,cfg)
        if server:server.stop_event.set()
        raise RuntimeError('DYNAMIC_WORKER_START_FAILED:'+json.dumps(startup_errors))
    groups=[]
    try:
        from .route_execution_cache import ExecutionCache
        cache=ExecutionCache(cfg['acceleration']['execution_cache_capacity'])
        def collect(row,target_dir):
            local=owner_collection_config(cfg,broker,cache)
            reports=collect_round([row],stock,protected,actor,tokenizer,local,target_dir,version)
            # SQLite connections are thread-owned; never use the main learner's
            # hidden-reference connection from a target worker.
            from .route_failure_policy import reward_kwargs
            from .route_process_reward import ALGORITHM
            with type(oracle)(Path(cfg['dataset'])/'references.sqlite',stock,
                    partial_weight=cfg['partial_reward_weight'],
                    **reward_kwargs(cfg,process=cfg['algorithm']==ALGORITHM)) as own_oracle:
                return score_groups([row],reports,cfg['candidates'],own_oracle)
        groups=collect_owned_targets(broker,rank,directory,collect,workers=cfg['executor_workers_per_rank'])
        # Waiting is CPU RPC, not an NCCL polling kernel competing with a GPU
        # that is still serving other owners' requests.
        while not broker.wait_collection(.25):pass
        if rank==0:broker.stop()
    except BaseException as exc:
        error=type(exc).__name__+':'+str(exc);broker.fail(error)
    worker.join(timeout=cfg.get('collective_timeout_seconds',604800))
    if worker.is_alive():
        error='GPU_OWNER_DID_NOT_STOP_BEFORE_LEARNER';broker.fail(error)
    else:
        try:close_gpu_sampler(sampler,cfg)
        except BaseException as exc:
            error=type(exc).__name__+':'+str(exc);broker.fail(error)
    # Another worker may have failed after its local CPU work finished.
    observed=broker.snapshot()
    errors=[None]*world;dist.all_gather_object(errors,error)
    publication_error=''
    try:
        if rank==0:atomic_json(directory/'scheduler_summary.json',observed)
    except BaseException as exc:publication_error=type(exc).__name__+':'+str(exc)
    try:
        publication_errors=[None]*world;dist.all_gather_object(publication_errors,publication_error)
    finally:
        if server:server.stop_event.set()
    if any(publication_errors):raise RuntimeError('DYNAMIC_SUMMARY_PUBLICATION_FAILED:'+json.dumps(publication_errors))
    if any(errors) or observed['error']:
        raise RuntimeError('DYNAMIC_COLLECTION_FAILED:'+json.dumps([*errors,observed['error']]))
    if observed['complete']!=len(rows):raise ValueError('DYNAMIC_TARGET_DENOMINATOR_MISMATCH')
    metrics=backend_collection_metrics(directory/f'gpu-rank{rank:02d}.jsonl')
    if '_last_native_engine_metrics' in cfg:
        metrics['native_engine']=cfg['_last_native_engine_metrics']
        atomic_json(directory/f'native-engine-rank{rank:02d}.json',cfg['_last_native_engine_metrics'])
    if '_last_native_journal_metrics' in cfg:
        metrics['native_journal']=cfg['_last_native_journal_metrics']
    if 'request_accounting' in observed:metrics['physical_request_accounting']=observed['request_accounting']
    if cfg.get('acceleration',{}).get('version')=='pr111_full_acceleration_v2':
        metrics['cache_witness_provenance']=cache.persist_provenance(directory/f'cache-witnesses-rank{rank:02d}.json')
    metrics.update(dynamic_owned_target_groups=len(groups),deterministic_execution_cache=cache.metrics(),
                   global_scheduler=observed['counts'],physical_generation_owner_rank=rank)
    cfg['_last_collection_metrics']=metrics
    return groups
