"""Receipt-unique branch credit replaces (never adds to) independent G4 credit."""
from collections import defaultdict

from .route_process_reward import process_group_advantages,process_sequences
from .route_refit import decision_identity,grounded_action_key,branch_advantages,unique_credit


def training_records(group):
    fallback_adv=process_group_advantages([
        None if r['reward'] is None else r['reward_evidence']['process_credit']['event_rewards'] for r in group])
    fallback={}
    if fallback_adv is not None:
        for report,advantages in zip(group,fallback_adv):
            list(process_sequences(report,advantages))
            fallback.update((r['request_id'],a) for r,a in zip(report['records'],advantages))
    candidates=[];branch={};conditional=0;masked=0
    for root in group:
        refit=root.get('refit',{})
        reports=[root,*refit.get('suffix_reports',[])]
        conditions=refit.get('conditioning',{})
        rows=defaultdict(list);seen=set()
        for report in reports:
            credit=report.get('reward_evidence',{}).get('process_credit')
            if report['reward'] is not None:
                list(process_sequences(report,[0.]*len(report['records'])))
            candidates.extend((record,fallback.get(record['request_id'])) for record in report['records'])
            for event in report.get('refit_path',[]):
                if event['name']!='apply_electron_flow' or event['node_id'] not in conditions:continue
                index=event['index']
                if type(index) is not int or not 0<=index<len(report['records']):
                    raise ValueError('REFIT_CONDITIONING_RECORD_INDEX')
                record=report['records'][index];uid=record['request_id']
                context=conditions[event['node_id']]
                history=[dict(raw=r['raw'],state_before=r['state_before'],state_after=r.get('state_after'))
                         for r in report['records'][:index] if r.get('accepted') is True]
                if (decision_identity(**context)!=event['node_id'] or
                        context['prompt_token_ids']!=record.get('prompt_token_ids') or
                        context['mapped_state']!=record['state_before'] or
                        context['occurrence']!=record['occurrence'] or
                        context['behavior_version']!=record['behavior_version'] or
                        context['accepted_history']!=history or
                        event['request_id']!=uid or
                        event['action']!=grounded_action_key(record['state_before'],record['raw'])):
                    raise ValueError('REFIT_CONDITIONING_MISMATCH')
                # Shared prefix has only its original sampled continuation;
                # descendant leaves cannot turn it into extra independent draws.
                if uid in seen:continue
                seen.add(uid)
                rows[event['node_id']].append(dict(action=event['action'],receipt_id=uid,
                    value=None if credit is None else credit['returns'][index],
                    complete=report.get('status') not in (None,'resource_censored','protected_censored'),
                    proposal=event['proposal']))
        for node,values in rows.items():
            advantages=(None if refit.get('node_censored_receipts',{}).get(node) else
                branch_advantages(values,matched_continuations=refit['config']['matched_continuations']))
            if advantages is None:masked+=1
            else:
                if set(branch)&set(advantages):raise ValueError('REFIT_RECEIPT_MULTIPLE_BRANCH_CREDITS')
                branch.update(advantages);conditional+=1
    records=unique_credit(candidates,branch)
    return records,dict(conditional_groups=conditional,masked_conditional_groups=masked,
        original_g4_eligible=fallback_adv is not None,unique_receipts=len(records),
        effective=any(a!=0 for _,a in records))
