"""Bridge executed TRAIN ReFIT journals to offline native-action distillation.

Trust scope: already executed journals plus a frozen TRAIN reward oracle, NOT
independent replay of native chemistry and NOT authentication of loaded weights.
The caller authenticates actual_actor_sha against its collector/model snapshot.
"""
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
import math

from .route_self_distill import OBJECTIVE, HEADER_FIELDS, content_sha256, _sha, _tokens

SCOPE='executed_journal_and_frozen_training_reward_oracle_not_independent_replay'
CENSORED={'resource_censored','protected_censored'}
COMPLETE={'stock_solved','rejected_action','recovery_exhausted','cycle','nonproductive_finish','already_stock'}


def export_search_states(groups, *, actual_actor_sha, training_task_ids, behavior_version,
                         tokenizer_sha, tool_schema_sha, executor_sha, stock_contract_sha,
                         budget_sha, reward_contract_sha, reward_oracle):
    """Export receipt-unique matched states from score_groups/ReFIT.pack output.

    Recompute process reward with the supplied frozen TRAIN oracle and validate
    its digest/returns against the recorded credit. Reused prefix requests keep
    their original continuation outcome, never a descendant's improved suffix.
    Any resource-censored actual draw masks all candidates at its matched node.
    """
    from .route_process_reward import process_sequences
    from .route_refit import decision_identity, grounded_action_key
    from .route_refit_runtime import validate_refit
    from .route_training_contract import chemical_identity
    for name,value in [('actor',actual_actor_sha),('tokenizer',tokenizer_sha),('tools',tool_schema_sha),
                       ('executor',executor_sha),('stock',stock_contract_sha),('budget',budget_sha),
                       ('reward',reward_contract_sha)]:
        _sha(value,name)
    if reward_oracle is None or getattr(reward_oracle,'split',None)!='training':
        raise ValueError('FROZEN_TRAIN_REWARD_ORACLE_REQUIRED')
    if not isinstance(behavior_version,str) or not behavior_version:
        raise ValueError('BEHAVIOR_VERSION_REQUIRED')
    allowed=set(training_task_ids)
    if not allowed or any(not isinstance(t,str) or not t for t in allowed):
        raise ValueError('TRAIN_TASK_ID_ALLOWLIST_REQUIRED')
    states=OrderedDict();receipt_payloads={};unique_records={};masked=set();node_tasks={}
    checked_reports=0
    for group in groups:
        if not group:raise ValueError('EMPTY_SCORED_GROUP')
        for root in group:
            task_id=root.get('task_id')
            if task_id not in allowed:raise ValueError('TRAIN_TASK_ID_NOT_ALLOWLISTED')
            if isinstance(training_task_ids,Mapping) and chemical_identity(root['root'])!=chemical_identity(training_task_ids[task_id]):
                raise ValueError('TRAIN_TASK_ROOT_MISMATCH')
            refit=root.get('refit')
            if refit is None:
                reports=[root];conditions={};refit_config=None
            else:
                refit_config=dict(validate_refit(refit['config']))
                conditions=refit['conditioning'];reports=[root,*refit['suffix_reports']]
                masked.update(k for k,v in refit.get('node_censored_receipts',{}).items() if v)
            for report in reports:
                if report.get('task_id')!=task_id:raise ValueError('TRAIN_SUFFIX_TASK_ID_MISMATCH')
                if report.get('behavior_version')!=behavior_version:
                    raise ValueError('JOURNAL_BEHAVIOR_ACTOR_BINDING_MISMATCH')
                if report.get('gt_used_for_generation_or_stopping') is not False:
                    raise ValueError('HIDDEN_REFERENCE_GENERATION_FORBIDDEN')
                status=report.get('status')
                if status not in COMPLETE|CENSORED:raise ValueError('INCOMPLETE_REAL_STATUS_REWARD')
                records=report['records'];credit=report.get('reward_evidence',{}).get('process_credit')
                if report['reward'] is not None:
                    try:list(process_sequences(report,[0.]*len(records)))
                    except Exception as exc:raise ValueError('PROCESS_CREDIT_INVALID') from exc
                elif credit is not None:raise ValueError('CENSORED_PROCESS_CREDIT_MUST_BE_NONE')
                try:rescored=reward_oracle.score(task_id,report)
                except Exception as exc:raise ValueError('FROZEN_REWARD_PROOF_INVALID') from exc
                if (content_sha256(rescored)!=content_sha256(report['reward_evidence'])
                        or rescored['reward']!=report['reward']):
                    raise ValueError('RECORDED_REWARD_PROOF_MISMATCH')
                censored=rescored.get('censored') is True
                if censored!=(status in CENSORED):raise ValueError('REAL_STATUS_REWARD_CENSOR_MISMATCH')
                if any(r.get('resource_error') or r.get('truncated') for r in records) and not censored:
                    raise ValueError('REAL_RESOURCE_REWARD_MISMATCH')
                report_sha=content_sha256(report);checked_reports+=1
                for record in records:
                    uid=record['request_id'];signature=content_sha256(record)
                    if uid in unique_records and unique_records[uid]!=(task_id,signature):
                        raise ValueError('SHARED_PREFIX_RECORD_CONFLICT')
                    unique_records[uid]=(task_id,signature)
                for event in report.get('refit_path',[]):
                    node=event['node_id']
                    if node not in conditions:continue
                    context=conditions[node];index=event['index']
                    if type(index) is not int or not 0<=index<len(records):
                        raise ValueError('REFIT_CONDITIONING_RECORD_INDEX')
                    record=records[index];uid=record['request_id']
                    history=[{k:r.get(k) for k in ('raw','state_before','state_after')}
                             for r in records[:index] if r.get('accepted') is True]
                    if (decision_identity(**context)!=node or context['tree_id']!=report['episode_id']
                            or chemical_identity(context['target_id'])!=chemical_identity(report['root'])
                            or context['prompt_token_ids']!=record.get('prompt_token_ids')
                            or context['mapped_state']!=record['state_before']
                            or context['occurrence']!=record['occurrence']
                            or context['behavior_version']!=record['behavior_version']
                            or context['behavior_version']!=behavior_version
                            or context['accepted_history']!=history
                            or event['request_id']!=uid
                            or event['action']!=grounded_action_key(record['state_before'],record['raw'])):
                        raise ValueError('EXACT_REFIT_CONDITIONING_MISMATCH')
                    if event.get('proposal')!='original_policy':raise ValueError('UNKNOWN_REFIT_PROPOSAL')
                    if node in node_tasks and node_tasks[node]!=task_id:raise ValueError('NODE_TRAIN_TASK_ID_CONFLICT')
                    node_tasks[node]=task_id
                    if node not in states:
                        states[node]=dict(split='training',node_id=node,source_actor_sha=actual_actor_sha,
                            behavior_policy_version=behavior_version,refit_config_sha=content_sha256(refit_config),
                            budget_sha=budget_sha,tokenizer_sha=tokenizer_sha,tool_schema_sha=tool_schema_sha,
                            executor_sha=executor_sha,stock_contract_sha=stock_contract_sha,
                            external_teacher_calls=0,prompt_token_ids=list(context['prompt_token_ids']),
                            conditioning=dict(target_id=task_id,reaction_occurrence_id=context['occurrence'],
                                observation_sha=content_sha256([context['mapped_state'],context['tools'],context['prompt_token_ids']]),
                                history_sha=content_sha256(history),route_context_sha=content_sha256(context['route_context']),
                                inventory_sha=content_sha256(context['mapped_state']),refit_preimage=deepcopy(context)),candidates=[])
                    state=states[node]
                    # Validate duplicate paths above; attach only the first actual
                    # continuation for this shared sampled request below.
                    if uid in receipt_payloads:
                        if receipt_payloads[uid]['source_node_id']!=node:
                            raise ValueError('RECEIPT_MULTIPLE_DECISION_STATES')
                        continue
                    if censored:masked.add(node)
                    c=dict(semantic_key=event['action'],generated_role='assistant',
                        generated_token_ids=list(record.get('generated_token_ids',[])),
                        Q_search=None if credit is None else credit['returns'][index],receipt_id=uid,
                        conditioning_sha=content_sha256({k:state[k] for k in HEADER_FIELDS}),
                        source_actor_sha=actual_actor_sha,behavior_policy_version=behavior_version,
                        refit_config_sha=state['refit_config_sha'],budget_sha=budget_sha,
                        proposal_kind='independent_policy_sample')
                    tier=('reference_supported_closed' if rescored.get('verified_stock_closed') is True
                          else 'executor_rejected' if record.get('accepted') is False
                          else 'stock_closed_unverified' if rescored.get('structural_stock_closed') is True
                          else 'observed_partial')
                    receipt_payloads[uid]=dict(c,split='training',verified=True,completed=not censored,
                        censored=censored,executor_accepted=record.get('accepted') is True,trust_tier=tier,
                        external_teacher_calls=0,source_node_id=node,source_record_sha256=content_sha256(record),
                        source_report_sha256=report_sha,process_credit_sha256=None if credit is None else credit['sha256'],
                        reward_contract_sha=reward_contract_sha,verification_scope=SCOPE,
                        independent_executor_replay_verified=False)
                    state['candidates'].append(c)
    for node,state in states.items():
        state['matched_group_censored']=node in masked
        for c in state['candidates']:
            receipt=receipt_payloads[c['receipt_id']]
            if node in masked:
                receipt['censored']=True;receipt['matched_group_censored']=True
            c['receipt_sha256']=content_sha256(receipt)
    return dict(states=list(states.values()),receipt_manifest=dict(verifier_sha=reward_contract_sha,
        receipts=receipt_payloads,scope=SCOPE,actual_actor_sha=actual_actor_sha,
        independent_executor_replay_verified=False),statistics=dict(checked_reports=checked_reports,
        unique_journal_requests=len(unique_records),exported_decision_states=len(states),
        exported_receipts=len(receipt_payloads),masked_matched_nodes=len(set(states)&masked)),
        status='prepared_not_trained')


def distillation_update(actor, optimizer, prepared_examples, cfg):
    """One reusable weighted native sequence-CE step using existing forwards.

    No rollout, critic, GRPO ratio, reference KL or search selection occurs here.
    All state/source masses and q_T probabilities must already be prepared.
    """
    import torch
    import torch.distributed as dist
    from scripts.pr96_earho_contract import token_logps_batch,synchronize_gradients
    from .route_training_batches import training_batches
    from .route_training_memory import checkpointed_policy_forward
    from .route_precision import precision_mode
    distributed=dist.is_available() and dist.is_initialized()
    def checked(error):
        if distributed:
            errors=[None]*dist.get_world_size()
            dist.all_gather_object(errors,None if error is None else repr(error))
            if any(errors):raise ValueError('COLLECTIVE_SELF_DISTILL_FAILURE:'+repr(errors)) from error
        elif error is not None:raise error
    compact=OrderedDict();seen=set();mass=0.;tokens=0;error=None
    try:
        _sha(cfg.get('source_actor_sha'),'source_actor')
        if cfg.get('actual_actor_sha')!=cfg['source_actor_sha']:raise ValueError('DISTILL_ACTOR_INIT_BINDING')
        precision=precision_mode(cfg)
        for row in prepared_examples:
            if row.get('split')!='training':raise ValueError('TRAINING_ONLY')
            if row.get('status')!='prepared' or row.get('objective')!=OBJECTIVE or row.get('external_teacher_calls')!=0:
                raise ValueError('PREPARED_STANDARD_DISTILLATION_REQUIRED')
            if row.get('kind') not in {'self_search','original_sft','actor_state'}:
                raise ValueError('UNKNOWN_SOURCE_KIND')
            if row['kind']!='original_sft' and row.get('source_actor_sha')!=cfg['source_actor_sha']:
                raise ValueError('DISTILL_ACTOR_SOURCE_MISMATCH')
            if row['state_key'] in seen:raise ValueError('DUPLICATE_DISTILL_STATE')
            seen.add(row['state_key']);weight=row.get('mixture_weight')
            if type(weight) not in (int,float) or not math.isfinite(weight) or weight<0:
                raise ValueError('INVALID_STATE_SOURCE_MASS')
            p=_tokens(row.get('prompt_token_ids'),'prompt');targets=row['targets']
            probabilities=[t.get('probability') for t in targets]
            if (not probabilities or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in probabilities)
                    or not math.isclose(math.fsum(probabilities),1.,rel_tol=0.,abs_tol=1e-12)):
                raise ValueError('INVALID_DISTILL_TARGET_PROBABILITY')
            mass+=weight
            for target,q in zip(targets,probabilities):
                r=_tokens(target.get('generated_token_ids'),'assistant',cfg['eos_token_id'])
                if weight*q:
                    key=(tuple(p),tuple(r));compact[key]=compact.get(key,0.)+weight*q
                    tokens+=len(r)
        # Validate every physical batch before touching gradients/parameters.
        entries=list(compact.items())
        batches=list(training_batches(entries,cfg['loss_batch_size'],cfg['training_max_padded_tokens']))
        if not math.isfinite(cfg['max_grad_norm']) or cfg['max_grad_norm']<=0:
            raise ValueError('INVALID_GRADIENT_NORM_CAP')
    except Exception as exc:error=exc
    checked(error)
    device=next(actor.parameters()).device
    denominator=torch.tensor(mass,dtype=torch.float64,device=device)
    if distributed:dist.all_reduce(denominator)
    global_mass=denominator.item()
    result=dict(objective=OBJECTIVE,states=len(seen),policy_tokens=tokens,unique_sequences=len(compact),
                batches=len(batches),global_state_source_mass=global_mass,loss=0.,optimizer_updated=False)
    if not global_mass:return result
    optimizer.zero_grad(set_to_none=True);numerator=0.;previous=actor.training;error=None
    try:
        actor.eval()
        with checkpointed_policy_forward(actor,cfg):
            for batch in batches:
                pairs=[(list(key[0]),list(key[1])) for key,_ in batch]
                logps=token_logps_batch(actor,pairs,cfg['pad_token_id'],precision=precision)
                loss=None
                for values,(_,weight),(_,response) in zip(logps,batch,pairs):
                    if values.numel()!=len(response) or not torch.isfinite(values).all():
                        raise ValueError('INVALID_NATIVE_RESPONSE_LOGPS')
                    part=-values.sum()*weight
                    loss=part if loss is None else loss+part
                if not torch.isfinite(loss):raise ValueError('NONFINITE_DISTILLATION_LOSS')
                numerator+=loss.detach().item();loss.backward()
    except Exception as exc:error=exc
    finally:actor.train(previous)
    checked(error)
    synchronize_gradients(actor,global_mass)
    norm=torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad],cfg['max_grad_norm'])
    if not torch.isfinite(norm):raise ValueError('NONFINITE_DISTILLATION_GRADIENT')
    optimizer.step()
    total=torch.tensor(numerator,dtype=torch.float64,device=device)
    if distributed:dist.all_reduce(total)
    result.update(loss=total.item()/global_mass,grad_norm=norm.item(),optimizer_updated=True)
    return result
