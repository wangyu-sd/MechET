"""Synchronize complete on-policy collection, score distinct committed edges, then train."""
from collections import Counter
import time
from pathlib import Path
from .route_joint_cache import content_key,get_scores,score_key
from .route_joint_devices import release_for_teacher,run_teacher_device,sample_on_devices
from .route_joint_forward import apply_scores
from .route_joint_inventory import extract_forward_input

def attach_joint_evidence(groups,cfg):
    import torch.distributed as dist
    opts=cfg['joint_forward']
    if not opts['lambda_f'] and not opts['lambda_d']:
        return {'query_count':0,'teacher_seconds':0.}
    protocol=opts['teacher_protocol']
    observations={};work=[];mask=Counter();local_error=None
    try:
        for group in groups:
            for root in group:
                for report in [root,*root['refit']['suffix_reports']]:
                    if report['reward'] is None:continue
                    edges=[]
                    for i,rxn in enumerate(report['reactions']):
                        q=extract_forward_input(report,i)
                        if q['status']!='eligible':
                            mask[q['reason']]+=1;continue
                        inputs={k:q[k] for k in ('teacher_reactants_mapped','target_product','target_heavy_teacher_maps')}
                        key=content_key(inputs);observations[key]=inputs
                        edges.append((key,dict(index=q['finish_index'],occurrence=rxn['occurrence'],
                            reaction_key=content_key([rxn['product'],sorted(rxn['precursors']),q['target_product']]),
                            protocol=content_key(protocol),finish_receipt=q['finish_request_id'])))
                    work.append((report,edges))
    except Exception as exc:local_error=repr(exc)
    multi=dist.is_available() and dist.is_initialized()
    parts=[None]*(dist.get_world_size() if multi else 1)
    if multi:dist.all_gather_object(parts,(observations,local_error))
    else:parts[0]=(observations,local_error)
    if any(e for _,e in parts):raise ValueError('joint inventory collection failed: '+str([e for _,e in parts if e]))
    pool={k:v for part,_ in parts for k,v in part.items()};ids=sorted(pool)
    profile=opts.get('profile','colocated');began=time.monotonic()
    # Release only on a cache miss. Rank zero decides once so every rank closes
    # and restores its own engine in the same phase, even with an empty shard.
    release=[False,None]
    if profile=='time_slice' and (not multi or dist.get_rank()==0):
        try:
            storage=Path(opts['cache_dir'])
            release[0]=any(not (storage/(score_key(protocol,pool[k])+'.json')).is_file() for k in ids)
        except OSError as exc:release[1]=repr(exc)
    if multi:dist.broadcast_object_list(release,src=0)
    if release[1]:raise RuntimeError(release[1])
    local_error=None;released=False
    try:released=release_for_teacher(cfg,profile,release[0])
    except Exception as exc:local_error=repr(exc)
    errors=[None]*len(parts)
    if multi:dist.all_gather_object(errors,local_error)
    else:errors[0]=local_error
    if any(errors):raise RuntimeError('student engine release failed: '+str(errors))
    result=[None]
    if not multi or dist.get_rank()==0:
        try:
            metrics=dict(executed_queries=0,model_seconds=0.,device_seconds={})
            def batch(rows):
                started=time.monotonic();metrics['executed_queries']+=len(rows)
                devices=list(range(cfg['world_size'])) if profile=='time_slice' else [opts['teacher_local_rank']]
                values,seconds=sample_on_devices(rows,devices,lambda d,items:run_teacher_device(opts,d,items))
                metrics['device_seconds']=seconds
                metrics['model_seconds']+=time.monotonic()-started
                return values
            cached=get_scores([pool[k] for k in ids],protocol,opts['cache_dir'],batch)
            result[0]=(dict(zip(ids,cached)),metrics,None)
        except Exception as exc:
            # An engine-wide failure is operational; the round can retry its authentic
            # saved collection. Never silently train a 64-update J arm without a teacher.
            result[0]=(None,None,repr(exc))
    if multi:dist.broadcast_object_list(result,src=0)
    scores,metrics,err=result[0]
    if err:raise RuntimeError('FlowER scoring failed, receipts retained: '+err)
    if released:
        from .route_vllm_backend import start_vllm_process
        try:start_vllm_process(cfg)
        except Exception as exc:local_error=repr(exc)
    if multi:dist.all_gather_object(errors,local_error)
    else:errors[0]=local_error
    if any(errors):raise RuntimeError('student engine restart failed: '+str(errors))
    route_rewards=[];teacher_rewards=[];scored_finishes=0;eligible_finishes=0
    report_error=None
    try:
        for report,edges in work:
            decisions=[dict(fields,score=scores[key]['score'],teacher_cache_key=scores[key]['key']) for key,fields in edges]
            eligible_finishes+=len(decisions);scored_finishes+=sum(d['score'] is not None for d in decisions)
            route_rewards.append(report['reward'])
            apply_scores(report,decisions,opts['lambda_f'])
            teacher_rewards.append(report['reward']-route_rewards[-1])
    except Exception as exc:report_error=repr(exc)
    report_errors=[None]*len(parts)
    if multi:dist.all_gather_object(report_errors,report_error)
    else:report_errors[0]=report_error
    if any(report_errors):raise ValueError('cannot bind forward scores to route credit: '+str(report_errors))
    return dict(query_count=len(observations),global_distinct=len(scores),available=sum(v['score'] is not None for v in scores.values()),
                global_input_sha256=content_key([pool[key] for key in ids]),
                mask_reasons=dict(mask),eligible_finish_appearances=eligible_finishes,scored_finish_appearances=scored_finishes,
                observed_route_reward_sum=sum(route_rewards),observed_teacher_reward_sum=sum(teacher_rewards),
                scored_reports=len(route_rewards),global_executed_queries=metrics['executed_queries'],
                teacher_profile=profile,student_engine_restarted=released,teacher_device_seconds=metrics['device_seconds'],
                global_cache_hits=len(scores)-metrics['executed_queries'],global_model_seconds=metrics['model_seconds'],
                teacher_seconds=time.monotonic()-began)
