"""Independent full-route Dr.GRPO with hidden reference rewards and full VALID selection."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import time

from mechet.external_route_data import file_sha, iter_records, signature
from mechet.route_training_contract import atomic_json
from mechet.route_phase_profile import phase
from scripts.train_pr98_route_policy import (worker_lease,publish_checkpoint,
    discover_checkpoint,verify_checkpoint,emit)


def validate_public_tasks(rows):
    seen = set()
    for row in rows:
        if set(row) != {'task_id','target'}:
            raise ValueError('PRIVATE_FIELDS_IN_PUBLIC_TASK')
        if row['task_id'] in seen or not isinstance(row['target'],str) or not row['target']:
            raise ValueError('INVALID_OR_DUPLICATE_PUBLIC_TASK')
        seen.add(row['task_id'])


def training_order(rows,seed):
    validate_public_tasks(rows)
    return [r['task_id'] for r in sorted(rows,key=lambda r:signature([seed,r['task_id']]))]


def make_episodes(rows,stock,protected,cfg,version):
    from mechet.external_route_rl import TargetOnlyRouteEpisode
    validate_public_tasks(rows)
    episode_cfg = dict(behavior_version=version,max_action_tokens=cfg['max_action_tokens'],
        protected_identities=protected,concurrent_executors=cfg['world_size']*cfg['executor_workers_per_rank'])
    for key in ('public_observation_contract','max_consecutive_recoveries'):
        if key in cfg:
            episode_cfg[key]=cfg[key]
    if 'route_memory' in cfg:
        episode_cfg['route_memory'] = cfg['route_memory']
    episode_type = TargetOnlyRouteEpisode
    namespace = version
    if '_validation_rng_namespace' in cfg:
        from mechet.external_route_validation import ValidationRouteEpisode
        episode_type = ValidationRouteEpisode
        namespace = cfg['_validation_rng_namespace']
    if 'route_memory' in cfg:
        from mechet.route_memory_integration import MemoryTargetOnlyRouteEpisode, MemoryValidationRouteEpisode
        episode_type = (MemoryValidationRouteEpisode if '_validation_rng_namespace' in cfg
                        else MemoryTargetOnlyRouteEpisode)
    return [episode_type(episode_cfg,row['target'],stock,
                namespace+'/'+row['task_id']+'/'+str(k),cfg['seed'])
            for row in rows for k in range(cfg['candidates'])]


def score_groups(rows,reports,candidates,oracle):
    if len(reports) != len(rows)*candidates:
        raise ValueError('INCOMPLETE_K_EPISODE_COLLECTION')
    groups=[]
    for i,row in enumerate(rows):
        group=[]
        for report in reports[i*candidates:(i+1)*candidates]:
            result=oracle.score(row['task_id'],report)
            group.append(dict(report,task_id=row['task_id'],reward=result['reward'],reward_evidence=result))
        groups.append(group)
    return groups


def sampled_behavior_entropy(groups):
    """Monte Carlo action-token surprisal, including censored sampled tokens.

    This is a sampled-action token entropy estimator, not an exact entropy
    calculation over each full vocabulary distribution.
    """
    count=0;surprisal=0.
    for group in groups:
        for report in group:
            for record in report['records']:
                values=record.get('behavior_logprobs',[])
                count+=len(values);surprisal-=sum(map(float,values))
    return dict(behavior_token_surprisal_sum=surprisal,behavior_token_surprisal_count=count,
                policy_entropy_estimate=surprisal/count if count else None,
                policy_entropy_scope='sampled_action_token_entropy_estimator_all_rollouts_including_censored')


def backend_collection_metrics(raw_path):
    """Count durable backend request attempts and actual returned generations.

    A requested receipt precedes generate(), so an interrupted attempt does not
    assert that generation completed. Context guards produce neither event.
    Existing raw journal bytes are counted on replay; reused results are not new
    model calls and no number is inferred from output-token volume.
    """
    metrics=dict(backend_generation_batches=0,backend_request_attempts=0,
                 backend_completed_batches=0,backend_completed_requests=0)
    path=Path(raw_path)
    if path.exists():
        with path.open() as reader:
            for line in reader:
                event=json.loads(line)
                if event['event']=='gpu-requested':
                    metrics['backend_generation_batches']+=1
                    metrics['backend_request_attempts']+=len(event['requests'])
                elif event['event']=='gpu-returned':
                    metrics['backend_completed_batches']+=1
                    metrics['backend_completed_requests']+=len(event['requests'])
                elif event['event']=='gpu-batch-timing':
                    for source,target in (('generation_seconds','measured_generation_seconds'),
                                          ('request_journal_seconds','request_journal_seconds'),
                                          ('return_journal_seconds','return_journal_seconds')):
                        metrics[target]=metrics.get(target,0.)+event[source]
                elif event['event']=='vllm-submitted':
                    metrics.pop('backend_generation_batches',None)
                    metrics.pop('backend_completed_batches',None)
                    metrics['backend_physical_batch_count_scope']='native_engine_internal_not_inferred_from_requests'
                    metrics['backend_request_attempts']+=1
                elif event['event']=='vllm-completed':
                    metrics['backend_completed_requests']+=1
    return metrics


def collect_round(rows,stock,protected,actor,tokenizer,cfg,directory,version):
    from mechet.route_policy_sampling import (collect_route_groups,
        read_owned_journal,recover_behavior_receipts)
    from mechet.external_route_io import ReceiptBatch,BatchedNativeRouteSampler
    from scripts.pr96_earho_contract import token_logps_batch
    import torch
    directory.mkdir(parents=True,exist_ok=True)
    binding=dict(behavior_version=version,contract_sha256=cfg['_sha'],task_ids=[r['task_id'] for r in rows])
    if '_validation_rng_namespace' in cfg:
        binding['validation_rng_namespace'] = cfg['_validation_rng_namespace']
    path=directory/'binding.json'
    if path.exists():
        if json.loads(path.read_text()) != binding:
            raise ValueError('PENDING_COLLECTION_BINDING_CHANGED')
    else:
        atomic_json(path,binding)
    episodes=make_episodes(rows,stock,protected,cfg,version)
    indexed={ep.uid:ep for ep in episodes};seen=set();journal=directory/'executed_calls.jsonl'
    for receipt in read_owned_journal(journal):
        uid=receipt['request_id'].rsplit('/r',1)[0]
        if receipt['request_id'] in seen:
            raise ValueError('DUPLICATE_EXECUTED_BEHAVIOR_RECEIPT')
        indexed[uid].accept(receipt);seen.add(receipt['request_id'])
    execution_writer=ReceiptBatch(journal,cfg=cfg)
    native=cfg.get('inference_backend','hf')=='vllm'
    if native:
        from mechet.route_vllm_backend import (VllmRouteSampler,recover_native_receipts,
                                               bind_rollout_adapter)
        from mechet.route_async_collection import collect_async_route_groups
        bind_rollout_adapter(cfg,version)
        sampler=VllmRouteSampler(cfg['_vllm_process'],tokenizer,cfg,directory/'raw_generation.jsonl')
    else:
        sampler=BatchedNativeRouteSampler(actor,tokenizer,cfg,directory/'raw_generation.jsonl',execution_writer)
    def recompute(prompt,response):
        actor.eval()
        with torch.no_grad():
            from mechet.route_precision import precision_mode
            precision_kwargs=({'precision':precision_mode(cfg)} if 'policy_precision' in cfg else {})
            return token_logps_batch(actor,[(prompt,response)],tokenizer.pad_token_id,
                                    **precision_kwargs)[0].tolist()
    recovered=(recover_native_receipts(sampler.raw_path,version) if native
               else recover_behavior_receipts(sampler.raw_path,version,recompute))
    def record(receipt):
        if receipt['request_id'] in seen:
            raise ValueError('DUPLICATE_NEW_BEHAVIOR_RECEIPT')
        execution_writer.append(receipt);seen.add(receipt['request_id'])
    next_snapshot=0.
    def snapshot(_episode=None,*,force=False):
        nonlocal next_snapshot
        now=time.monotonic()
        if force or now>=next_snapshot:
            next_snapshot=now+30.
            atomic_json(directory/'collection_progress.json',dict(
                behavior_version=version,contract_sha256=cfg['_sha'],updated_unix=time.time(),
                targets=len(rows),sampled_route_episodes=len(episodes),
                completed_route_episodes=sum(ep.status is not None for ep in episodes),
                active_route_episodes=sum(ep.status is None for ep in episodes),
                model_decision_requests=sum(len(ep.records) for ep in episodes),
                completed_reaction_programs=sum(len(ep.reactions) for ep in episodes),
                native_stock_solved=sum(ep.status=='stock_solved' for ep in episodes),
                scheduler_metrics=dict(getattr(sampler,'scheduler_metrics',{})),
                censored_route_episodes=sum(ep.status in ('resource_censored','protected_censored') for ep in episodes),
                max_decisions_inflight=max((len(ep.records) for ep in episodes if ep.status is None),default=0)))
    snapshot(force=True)
    try:
        collector=collect_async_route_groups if native else collect_route_groups
        return collector(episodes,sampler,workers=cfg['executor_workers_per_rank'],
                                    journal=record,recovered=recovered,snapshot=snapshot)
    finally:
        if native:
            sampler.close()
            cfg['_vllm_process'].rpc('drain',{}).result(timeout=cfg['_vllm_process'].timeout)
        execution_writer.flush()
        snapshot(force=True)
        metrics=backend_collection_metrics(sampler.raw_path)
        metrics.update(native_decision_request_attempts=sum(len(ep.records) for ep in episodes),
                       scheduler_metrics=dict(getattr(sampler,'scheduler_metrics',{})),
                       decision_request_metric_scope='native_policy_requests_including_physical_censor_attempts',
                       backend_metric_scope='cumulative_original_durable_gpu_receipts_including_recovered_work')
        atomic_json(directory/'collection_metrics.json',metrics)
        cfg['_last_collection_metrics']=metrics


def full_validation(rows,stock,protected,actor,tokenizer,cfg,root,checkpoint_round,actor_sha,fractions):
    """Complete fixed VALID, then select by the declared root-level criterion.

    Each rank collects independently. The only collectives are at checkpoint
    entry and after local collection, so unequal shard/batch counts are safe.
    """
    import numpy as np
    import torch
    import torch.distributed as dist
    from mechet.external_route_rl import HiddenRouteReward
    from mechet.external_route_validation import (validation_outcomes,
        summarize_validation,selection_key)
    rank,world=dist.get_rank(),dist.get_world_size()
    protocol=cfg['validation_protocol']
    if len(rows)!=cfg['validation_targets']:
        raise ValueError('FULL_VALIDATION_DENOMINATOR_CHANGED')
    validate_public_tasks(rows)
    directory=root/'validation'/f'checkpoint-{checkpoint_round:06d}'
    binding=dict(contract_sha256=cfg['_sha'],actor_sha256=actor_sha,
        checkpoint_round=checkpoint_round,protocol_sha256=signature(protocol),
        task_ids_sha256=signature([r['task_id'] for r in rows]),
        denominator=len(rows),fractions=fractions)
    summary_path=directory/'summary.json'
    cached=[None,None]
    if rank==0:
        try:
            directory.mkdir(parents=True,exist_ok=True)
            if summary_path.exists():
                saved=json.loads(summary_path.read_text())
                if saved['binding']!=binding or not saved['complete']:
                    raise ValueError('VALIDATION_COMPLETION_BINDING_CHANGED')
                best_path=root/'validation'/'best_checkpoint.json'
                best=json.loads(best_path.read_text()) if best_path.exists() else None
                if best is not None and best['binding']['contract_sha256']!=cfg['_sha']:
                    raise ValueError('BEST_CHECKPOINT_CONTRACT_MISMATCH')
                if best is None or selection_key(saved)<selection_key(best):
                    atomic_json(best_path,saved)
                cached[0]=saved
        except Exception as exc:
            cached[1]=type(exc).__name__+':'+str(exc)
    dist.broadcast_object_list(cached,src=0)
    if cached[1]:
        raise RuntimeError('COLLECTIVE_VALIDATION_CACHE_FAILED:'+cached[1])
    if cached[0] is not None:
        return cached[0]
    # Evaluation never advances the learner's RNG streams, including on errors.
    rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),
             python=random.getstate(),numpy=np.random.get_state())
    was_training=actor.training
    local_outcomes=[];error='';observed_seconds=0.;local_backend={}
    emit(root,rank,'external-route-full-validation-start',checkpoint_round=checkpoint_round,
         actor_sha256=actor_sha,validation_targets=len(rows),candidates=protocol['candidates'],
         fractions=fractions,optimizer_updates_during_validation=0)
    try:
        actor.eval()
        eval_cfg=dict(cfg,seed=protocol['seed'],candidates=protocol['candidates'],
            _validation_rng_namespace=protocol['id']+':'+signature(protocol))
        own=rows[rank::world]
        chunk_size=max(1,cfg['products_per_round']//world)
        version=f'validation{checkpoint_round:06d}:{actor_sha}'
        rank_dir=directory/f'rank{rank:02d}'
        with HiddenRouteReward(Path(cfg['dataset'])/'references.sqlite',stock,
                               split='validation',partial_weight=cfg['partial_reward_weight'],
                               **({'material_contract':cfg['stock_material_contract']}
                                  if 'stock_material_contract' in cfg else {})) as oracle:
            for batch_index,start in enumerate(range(0,len(own),chunk_size)):
                batch_rows=own[start:start+chunk_size]
                batch_dir=rank_dir/f'batch-{batch_index:05d}'
                completion=batch_dir/'outcomes.json'
                batch_binding=dict(binding,task_ids=[r['task_id'] for r in batch_rows],rank=rank)
                if completion.exists():
                    saved=json.loads(completion.read_text())
                    if saved['binding']!=batch_binding:
                        raise ValueError('VALIDATION_BATCH_BINDING_CHANGED')
                    outcomes=saved['outcomes'];seconds=saved['seconds']
                    backend=saved['backend_metrics']
                    if [v['task_id'] for v in outcomes]!=batch_binding['task_ids']:
                        raise ValueError('VALIDATION_BATCH_ROOT_ALIGNMENT')
                else:
                    began=time.monotonic()
                    eval_cfg.pop('_last_collection_metrics',None)
                    with torch.no_grad():
                        # Hidden VALID labels are absent from every collection argument.
                        reports=collect_round(batch_rows,stock,protected,actor,tokenizer,
                                              eval_cfg,batch_dir,version)
                    backend=eval_cfg.get('_last_collection_metrics')
                    if backend is None:
                        backend=backend_collection_metrics(batch_dir/'raw_generation.jsonl')
                    groups=score_groups(batch_rows,reports,protocol['candidates'],oracle)
                    atomic_json(batch_dir/'rewarded_episodes.json',groups)
                    outcomes=validation_outcomes(batch_rows,groups,protocol['candidates'])
                    seconds=time.monotonic()-began
                    atomic_json(completion,dict(binding=batch_binding,outcomes=outcomes,seconds=seconds,backend_metrics=backend))
                local_outcomes.extend(outcomes);observed_seconds+=seconds
                for key,value in backend.items():
                    if key.startswith('backend_') and isinstance(value,(int,float)):
                        local_backend[key]=local_backend.get(key,0)+value
                atomic_json(rank_dir/'progress.json',dict(binding=binding,
                    expected_local_roots=len(own),completed_local_roots=len(local_outcomes),
                    sampled_route_episodes=sum(v['sampled_route_episodes'] for v in local_outcomes),
                    model_decision_requests=sum(v['model_decision_requests'] for v in local_outcomes),
                    completed_reaction_programs=sum(v['completed_reaction_programs'] for v in local_outcomes),
                    verified_closed_roots=sum(v['verified_closed'] for v in local_outcomes),
                    censored_roots=sum(v['censored'] for v in local_outcomes)))
                emit(root,rank,'external-route-validation-batch-complete',checkpoint_round=checkpoint_round,
                     completed_local_roots=len(local_outcomes),expected_local_roots=len(own),
                     model_decision_requests=sum(v['model_decision_requests'] for v in local_outcomes))
    except Exception as exc:
        from mechet.external_route_runtime import record_phase_failure
        record_phase_failure(cfg,root,rank,exc,'full_validation')
        error=type(exc).__name__+':'+str(exc)
    finally:
        torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'])
        random.setstate(rng['python']);np.random.set_state(rng['numpy'])
        actor.train(was_training)
    parts=[None]*world
    dist.all_gather_object(parts,dict(error=error,outcomes=local_outcomes,seconds=observed_seconds,backend_metrics=local_backend))
    errors=[v['error'] for v in parts]
    if any(errors):
        raise RuntimeError('COLLECTIVE_FULL_VALIDATION_FAILED:'+json.dumps(errors))
    result=[None,None]
    if rank==0:
        try:
            outcomes=[v for part in parts for v in part['outcomes']]
            summary=summarize_validation([r['task_id'] for r in rows],outcomes,protocol['candidates'])
            if not summary['complete']:
                raise ValueError('FULL_VALIDATION_COVERAGE_INCOMPLETE')
            for part in parts:
                for key,value in part['backend_metrics'].items():
                    summary[key]=summary.get(key,0)+value
            summary.update(binding=binding,checkpoint_round=checkpoint_round,actor_sha256=actor_sha,
                decision_request_metric_scope='native_policy_requests_including_physical_censor_attempts',
                backend_metric_scope='cumulative_original_durable_gpu_receipts_including_recovered_work',
                checkpoint_path=cfg['parent'] if checkpoint_round==0 else str(root/'checkpoints'/f'round-{checkpoint_round:06d}'),
                protocol=protocol,validation_reference_split='validation',
                optimizer_updates_during_validation=0,
                worker_gpu_hours_observed=sum(v['seconds'] for v in parts)/3600.,
                wall_seconds_observed=max(v['seconds'] for v in parts),
                timing_scope='completed_batch_worker_wall; interrupted_unpublished_batch_time_not_included')
            atomic_json(directory/'root_outcomes.json',sorted(outcomes,key=lambda v:v['task_id']))
            atomic_json(summary_path,summary)
            best_path=root/'validation'/'best_checkpoint.json'
            best=json.loads(best_path.read_text()) if best_path.exists() else None
            if best is not None and best['binding']['contract_sha256']!=cfg['_sha']:
                raise ValueError('BEST_CHECKPOINT_CONTRACT_MISMATCH')
            if best is None or selection_key(summary)<selection_key(best):
                atomic_json(best_path,summary)
            result[0]=summary
        except Exception as exc:
            result[1]=type(exc).__name__+':'+str(exc)
    dist.broadcast_object_list(result,src=0)
    if result[1]:
        raise RuntimeError('COLLECTIVE_VALIDATION_PUBLICATION_FAILED:'+result[1])
    emit(root,rank,'external-route-full-validation-complete',checkpoint_round=checkpoint_round,
         actor_sha256=actor_sha,verified_closed_roots=result[0]['verified_closed_roots'],
         validation_denominator=len(rows),censored_roots=result[0]['censored_roots'],
         model_decision_requests=result[0]['model_decision_requests'])
    flush_phase_profile(cfg,root,rank,'validation',checkpoint_round)
    return result[0]


def collective_checkpoint_publish(root,version,state,writer):
    import torch.distributed as dist
    result=[None,None]
    if dist.get_rank()==0:
        try:
            saved=publish_checkpoint(root,version,state,writer)
            result[0]=file_sha(saved/'adapter_model.safetensors')
        except Exception as exc:
            result[1]=type(exc).__name__+':'+str(exc)
    dist.broadcast_object_list(result,src=0)
    if result[1]:
        raise RuntimeError('COLLECTIVE_CHECKPOINT_PUBLICATION_FAILED:'+result[1])
    return result[0]


def complete_deferred_baseline(rows,stock,protected,reference,tokenizer,cfg,root):
    """Resume original observations, using only the frozen starting reference."""
    from contextlib import ExitStack
    import torch.distributed as dist
    from mechet.external_route_schedule import (validate_deferred_baseline,
        evaluate_frozen_baseline,baseline_for_selection)
    from mechet.external_route_validation import selection_key
    code_root=Path(__file__).resolve().parents[1]
    rank,world=dist.get_rank(),dist.get_world_size()
    with ExitStack() as leases:
        error=''
        try:
            parent=validate_deferred_baseline(cfg,code_root)
            if Path(parent['output_dir']).resolve()!=Path(root).resolve():
                leases.enter_context(worker_lease(Path(parent['output_dir']),rank))
            atomic_json(root/f'baseline_continuation.rank{rank:02d}.json',dict(
                controller_contract_sha256=cfg['_sha'],parent_contract_sha256=parent['_sha'],
                parent_output=parent['output_dir'],reference_actor_sha256=cfg['parent_sha256'],
                controller_source=str(code_root),sampling_compatibility=cfg['initial_baseline']))
        except Exception as exc:error=type(exc).__name__+':'+str(exc)
        errors=[None]*world;dist.all_gather_object(errors,error)
        if any(errors):raise RuntimeError('COLLECTIVE_BASELINE_LEASE_FAILED:'+json.dumps(errors))
        summary=evaluate_frozen_baseline(cfg,code_root,rows,stock,protected,reference,tokenizer,full_validation)
    result=[None]
    if rank==0:
        try:
            selected=baseline_for_selection(cfg,summary)
            directory=root/'validation'/'checkpoint-000000';directory.mkdir(parents=True,exist_ok=True)
            atomic_json(directory/'summary.json',selected)
            path=root/'validation'/'best_checkpoint.json'
            best=json.loads(path.read_text()) if path.exists() else None
            if best is not None and best['binding']['contract_sha256']!=cfg['_sha']:
                raise ValueError('BEST_CHECKPOINT_CONTRACT_MISMATCH')
            if best is None or selection_key(selected)<selection_key(best):atomic_json(path,selected)
        except Exception as exc:result[0]=type(exc).__name__+':'+str(exc)
    dist.broadcast_object_list(result,src=0)
    if result[0]:raise RuntimeError('COLLECTIVE_BASELINE_SELECTION_FAILED:'+result[0])
    emit(root,rank,'external-route-deferred-baseline-complete',actor_sha256=cfg['parent_sha256'])


def create_phase_profile(cfg,rank,contract_sha256):
    enabled=cfg.get('phase_profiling',False)
    if type(enabled) is not bool:raise ValueError('PHASE_PROFILING_BOOL_REQUIRED')
    if not enabled:return None
    from mechet.route_phase_profile import RoutePhaseProfile
    return RoutePhaseProfile(rank,contract_sha256)


def flush_phase_profile(cfg,root,rank,kind,round_index):
    profiler=cfg.get('_phase_profile')
    if profiler is None:return None
    path=Path(root)/'runtime_profiles'/f'rank{rank:02d}'/f'{kind}-{round_index:06d}-{time.time_ns()}.json'
    try:
        return profiler.flush(path,window={'kind':kind,'round':round_index})
    except Exception as exc:
        # Optional telemetry must not replace an actor/executor/checkpoint error.
        print(json.dumps(dict(event='external-route-profile-flush-failed',rank=rank,
                              exception_type=type(exc).__name__)),flush=True)
        return None


def round_profile_kind(schedule_index,start_index):
    return 'cold' if schedule_index==start_index else 'round'


def train(contract_path):
    # Select NumPy's compatible threading layer before importing MKL/Torch.
    os.environ['MKL_THREADING_LAYER']='GNU'
    # Activate the profile before Transformers imports FLA's decorators, and
    # require the isolated startup check before loading four model replicas.
    from mechet.route_runtime_efficiency import configure_profile,require_kernel_unit
    raw_cfg=json.loads(Path(contract_path).read_text())
    profiler=create_phase_profile(raw_cfg,int(os.environ.get('RANK','0')),file_sha(contract_path))
    if profiler is not None:raw_cfg['_phase_profile']=profiler
    configure_profile(raw_cfg,Path(__file__).resolve().parents[1])
    require_kernel_unit(raw_cfg,file_sha(contract_path))
    with phase(raw_cfg,'startup.imports'):
        import numpy as np
        import torch
        import torch.distributed as dist
        from mechet.route_transformers_import import import_transformers_indexed
        transformers=import_transformers_indexed(raw_cfg)
        import peft
        import triton
        import rdkit
        from peft import PeftModel
        from transformers import AutoTokenizer
        from mechet.qwen35_protocol import load_language_model,OFFICIAL_REVISION,validate_runtime_versions
        from mechet.external_route_rl import HiddenRouteReward,optimize_routes
        from scripts.prepare_external_route_rl import verify_contract
    cfg=verify_contract(contract_path);cfg['_sha']=file_sha(contract_path)
    if profiler is not None:cfg['_phase_profile']=profiler
    from mechet.route_precision import configure_precision
    cfg['_effective_policy_precision']=configure_precision(cfg)
    transformers.utils.logging.disable_progress_bar()
    validate_runtime_versions(torch_version=torch.__version__,cuda_version=torch.version.cuda,triton_version=triton.__version__)
    if transformers.__version__!='5.5.0' or peft.__version__!='0.18.1' or rdkit.__version__!='2026.03.4':
        raise ValueError('PINNED_EXTERNAL_ROUTE_RUNTIME_REQUIRED')
    rank,world,local=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']),int(os.environ['LOCAL_RANK'])
    if world != cfg['world_size'] or local >= torch.cuda.device_count():
        raise ValueError('DISTRIBUTED_GPU_WORLD_MISMATCH')
    if cfg['gpu_name'] not in torch.cuda.get_device_name(local):
        raise ValueError('FROZEN_GPU_ALLOCATION_CHANGED')
    torch.cuda.set_device(local);torch.set_num_threads(cfg['cpu_threads_per_rank'])
    dist.init_process_group('nccl',timeout=timedelta(seconds=cfg['collective_timeout_seconds']))
    root=Path(cfg['output_dir']);root.mkdir(parents=True,exist_ok=True)
    dataset=Path(cfg['dataset'])
    rows=list(iter_records(dataset/'training.jsonl'));validate_public_tasks(rows)
    if len(rows) != cfg['train_targets']:
        raise ValueError('FULL_TRAINING_DENOMINATOR_CHANGED')
    stock=set((dataset/'stock.txt').read_text().splitlines())
    protected=json.loads((dataset/'training_protected_identities.json').read_text())
    validation_protected=json.loads((dataset/'validation_protected_identities.json').read_text())
    validation_rows=list(iter_records(dataset/'validation.jsonl'));validate_public_tasks(validation_rows)
    if len(validation_rows)!=cfg['validation_targets']:
        raise ValueError('FULL_VALIDATION_DENOMINATOR_CHANGED')
    ids=training_order(rows,cfg['seed']);byid={r['task_id']:r for r in rows}
    schedule=[(epoch,ids[i:i+cfg['products_per_round']]) for epoch in range(cfg['epochs'])
              for i in range(0,len(ids),cfg['products_per_round'])]
    order_sha=signature(schedule)
    from mechet.external_route_validation import validation_boundaries
    from mechet.external_route_schedule import deferred,validation_due,required_prior_validations
    boundaries=validation_boundaries(schedule,cfg['validation_protocol']['fractions'])
    reward_type=HiddenRouteReward
    from mechet.route_process_reward import ALGORITHM as PROCESS_ALGORITHM
    if cfg['algorithm']==PROCESS_ALGORITHM:
        from mechet.route_process_reward import ProcessRouteReward
        reward_type=ProcessRouteReward
    with worker_lease(root,rank), reward_type(dataset/'references.sqlite',stock,
                                partial_weight=cfg['partial_reward_weight'],
                                **({'material_contract':cfg['stock_material_contract']}
                                   if 'stock_material_contract' in cfg else {})) as oracle:
        checkpoint=discover_checkpoint(root,cfg['_sha'])
        state=verify_checkpoint(checkpoint,cfg['_sha']) if checkpoint else None
        if state and state['training_order_sha256'] != order_sha:
            raise ValueError('RESUME_FULL_TRAINING_ORDER_CHANGED')
        with phase(cfg,'startup.model_load'):
            tokenizer=AutoTokenizer.from_pretrained(cfg['base_path'],revision=OFFICIAL_REVISION,local_files_only=True)
            tokenizer.padding_side='left'
            if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
            cfg['pad_token_id']=tokenizer.pad_token_id
            actor,_=load_language_model(cfg['base_path'],dtype=torch.bfloat16)
            actor=PeftModel.from_pretrained(actor,checkpoint or cfg['parent'],is_trainable=True).to(local)
            reference,_=load_language_model(cfg['base_path'],dtype=torch.bfloat16)
            reference=PeftModel.from_pretrained(reference,cfg['parent'],is_trainable=False).to(local).eval()
            for p in reference.parameters():p.requires_grad_(False)
            for m in actor.modules():
                if isinstance(m,torch.nn.Dropout):m.p=0.
            actor.eval()
            optimizer=torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad],
                lr=cfg['learning_rate'],betas=tuple(cfg['adam_betas']),eps=cfg['adam_eps'],
                weight_decay=cfg['weight_decay'])
        start,step=(state['next_round'],state['optimizer_step']) if state else (0,0)
        actor_sha=file_sha(checkpoint/'adapter_model.safetensors') if state else cfg['parent_sha256']
        cfg['_rollout_adapter_path']=str(checkpoint or cfg['parent'])
        if cfg.get('inference_backend','hf')=='vllm':
            from mechet.route_vllm_backend import start_vllm_process
            start_vllm_process(cfg)
        if state:
            optimizer.load_state_dict(torch.load(checkpoint/'optimizer.pt',map_location=f'cuda:{local}',weights_only=False))
            rng=torch.load(checkpoint/'rng.pt',map_location='cpu',weights_only=False)[rank]
            torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'])
            random.setstate(rng['python']);np.random.set_state(rng['numpy'])
        else:
            torch.manual_seed(cfg['seed']+rank);random.seed(cfg['seed']+rank);np.random.seed(cfg['seed']+rank)
        for p in actor.parameters():
            if p.requires_grad:dist.broadcast(p.data,src=0)
        emit(root,rank,'external-route-actor-ready',parent_sha256=cfg['parent_sha256'],
             actor_sha256=actor_sha,actor_directory=str(checkpoint or cfg['parent']),
             algorithm=cfg['algorithm'],contract_sha256=cfg['_sha'],
             gpu_name=torch.cuda.get_device_name(local),visible_gpu_count=torch.cuda.device_count(),
             torch_version=torch.__version__,transformers_version=transformers.__version__,
             peft_version=peft.__version__,rdkit_version=rdkit.__version__,
             train_targets=len(ids),total_rounds=len(schedule),next_round=start,teacher_action_tokens=0)
        flush_phase_profile(cfg,root,rank,'startup',start)
        # A checkpoint can have been published immediately before an interrupted
        # scheduled validation. Complete that validation before the next update.
        prior_error=''
        try:
            for boundary in required_prior_validations(cfg,boundaries,start):
                if boundary<start:
                    prior=root/'validation'/f'checkpoint-{boundary:06d}'/'summary.json'
                    if not prior.exists():
                        raise ValueError('REQUIRED_PRIOR_FULL_VALIDATION_MISSING:'+str(boundary))
                    previous=json.loads(prior.read_text())
                    if (not previous['complete'] or previous['binding']['contract_sha256']!=cfg['_sha']
                            or previous['denominator']!=len(validation_rows)):
                        raise ValueError('REQUIRED_PRIOR_FULL_VALIDATION_INVALID:'+str(boundary))
        except Exception as exc:
            prior_error=type(exc).__name__+':'+str(exc)
        prior_errors=[None]*world;dist.all_gather_object(prior_errors,prior_error)
        if any(prior_errors):
            raise RuntimeError('COLLECTIVE_REQUIRED_PRIOR_VALIDATION_FAILED:'+json.dumps(prior_errors))
        if deferred(cfg):
            emit(root,rank,'external-route-train-first',initial_baseline='deferred_after_training',
                 prior_validation_output=cfg['initial_baseline']['parent_output'],optimizer_step=step)
        if validation_due(cfg,boundaries,start):
            full_validation(validation_rows,stock,validation_protected,actor,tokenizer,cfg,root,
                            start,actor_sha,boundaries[start])
        for index in range(start,len(schedule)):
            epoch,chunk=schedule[index];own=[byid[k] for k in chunk[rank::world]]
            version=f'round{index:06d}:{actor_sha}';directory=root/f'round-{index:06d}'/f'rank{rank:02d}'
            began=time.monotonic();error='';groups=[]
            try:
                # The oracle is deliberately absent from the collection API.
                from mechet.external_route_runtime import run_recorded
                with phase(cfg,'collection.total',targets=len(own)):
                    reports=run_recorded(contract_path,
                        lambda _:collect_round(own,stock,protected,actor,tokenizer,cfg,directory,version),rank=rank)
                with phase(cfg,'reward.score',groups=len(own)):
                    groups=score_groups(own,reports,cfg['candidates'],oracle)
                with phase(cfg,'reward.persist'):
                    atomic_json(directory/'rewarded_episodes.json',groups)
            except Exception as exc:
                error=type(exc).__name__+':'+str(exc)
            errors=[None]*world
            with phase(cfg,'sync.collective'):
                dist.all_gather_object(errors,error)
            if any(errors):raise RuntimeError('COLLECTIVE_ROUTE_COLLECTION_FAILED:'+json.dumps(errors))
            sampling_seconds=time.monotonic()-began
            with phase(cfg,'learner.total',groups=len(groups)):
                stats=optimize_routes(actor,reference,optimizer,groups,cfg)
            from mechet.external_route_runtime import require_route_update
            require_route_update(stats)
            step+=int(stats['optimizer_updated'])
            stats.update(sampled_behavior_entropy(groups))
            stats.update(cfg.get('_last_collection_metrics',{}))
            stats.update(sampling_seconds=sampling_seconds,targets=len(own),
                sampled_route_episodes=sum(len(g) for g in groups),
                model_decision_requests=sum(len(r['records']) for g in groups for r in g),
                completed_reaction_programs=sum(len(r['reactions']) for g in groups for r in g),
                native_stock_solved=sum(r['status']=='stock_solved' for g in groups for r in g),
                verified_stock_solved=sum(r['reward_evidence']['verified_stock_closed'] for g in groups for r in g),
                teacher_action_tokens=0)
            parts=[None]*world
            with phase(cfg,'sync.collective'):
                dist.all_gather_object(parts,stats)
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),python=random.getstate(),numpy=np.random.get_state())
            rngs=[None]*world
            with phase(cfg,'sync.collective'):
                dist.all_gather_object(rngs,rng)
            def writer(path):
                actor.save_pretrained(path,safe_serialization=True)
                torch.save(optimizer.state_dict(),path/'optimizer.pt');torch.save(rngs,path/'rng.pt')
            with phase(cfg,'checkpoint.publish'):
                actor_sha=collective_checkpoint_publish(root,index+1,dict(contract_sha256=cfg['_sha'],next_round=index+1,
                        optimizer_step=step,training_order_sha256=order_sha,epoch=epoch,
                        visited_target_episodes=sum(len(v) for _,v in schedule[:index+1]),round_metrics=parts),writer)
            cfg['_rollout_adapter_path']=str(root/'checkpoints'/f'round-{index+1:06d}')
            emit(root,rank,'external-route-round-complete',round=index+1,total_rounds=len(schedule),
                 train_denominator=len(ids),epoch=epoch,optimizer_step=step,actor_sha256=actor_sha,metrics=parts)
            flush_phase_profile(cfg,root,rank,round_profile_kind(index,start),index+1)
            if validation_due(cfg,boundaries,index+1):
                full_validation(validation_rows,stock,validation_protected,actor,tokenizer,cfg,root,
                                index+1,actor_sha,boundaries[index+1])
        if deferred(cfg):
            complete_deferred_baseline(validation_rows,stock,validation_protected,reference,tokenizer,cfg,root)
        completion_error=[None]
        if rank==0:
            try:
                atomic_json(root/'completion.json',dict(contract_sha256=cfg['_sha'],train_targets=len(ids),
                    epochs=cfg['epochs'],visited_target_episodes=len(ids)*cfg['epochs'],actor_sha256=actor_sha,
                    validation_targets=len(validation_rows),
                    selected_checkpoint=json.loads((root/'validation'/'best_checkpoint.json').read_text()),
                    chemical_benchmark_improvement='NOT_EVALUATED'))
            except Exception as exc:
                completion_error[0]=type(exc).__name__+':'+str(exc)
        dist.broadcast_object_list(completion_error,src=0)
        if completion_error[0]:
            raise RuntimeError('COLLECTIVE_TRAINING_COMPLETION_PUBLICATION_FAILED:'+completion_error[0])
    if '_vllm_process' in cfg:cfg['_vllm_process'].close()
    dist.barrier();dist.destroy_process_group()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',required=True,type=Path)
    from mechet.external_route_runtime import worker_main
    contract=parser.parse_args().contract
    worker_main(contract,train)
