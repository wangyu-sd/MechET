"""Offline scoring detail for already audited saved Route Dr.GRPO groups.

No actor, electron executor, optimizer, or new reward is run. Original frozen
reward replay is a required input; this report adds transparent diagnostic axes.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


def category(row):
    evidence=row['reward_evidence']
    if evidence.get('censored'):return 'resource_or_protection_censored'
    if evidence.get('graph_error'):return 'invalid_route_graph'
    if evidence.get('verified_stock_closed'):return 'reference_supported_full_closure'
    if evidence.get('structural_stock_closed'):return 'structural_closure_without_full_reference_support'
    if evidence.get('supported_reactions',0)>0:return 'trusted_partial_progress_unclosed'
    if not row['reactions']:return 'no_completed_reaction'
    return 'completed_reactions_without_trusted_reference_progress'


def group_scorecards(group):
    if len(group)!=4:raise ValueError('ORIGINAL_INDEPENDENT_G4_REQUIRED')
    if (len({r['task_id'] for r in group})!=1 or len({r['behavior_version'] for r in group})!=1
            or len({r['episode_id'] for r in group})!=4):
        raise ValueError('ORIGINAL_ROOT_OR_BEHAVIOR_IDENTITY_MISMATCH')
    rewards=[r['reward'] for r in group]
    if any(v is not None and (type(v) not in (int,float) or not math.isfinite(v)) for v in rewards):
        raise ValueError('NONFINITE_REWARD')
    usable=all(v is not None for v in rewards)
    mean=math.fsum(rewards)/4 if usable else None
    advantages=[v-mean for v in rewards] if usable else None
    cards=[];generated=policy=nonzero=0
    for i,row in enumerate(group):
        tokens=0;execution=Counter()
        for record in row['records']:
            p,r,b=record.get('prompt_token_ids'),record.get('generated_token_ids'),record.get('behavior_logprobs')
            if r and (not p or not isinstance(b,list) or len(b)!=len(r)
                      or any(type(v) not in (int,float) or not math.isfinite(v) or v>0 for v in b)):
                raise ValueError('INCOMPLETE_OR_INVALID_TOKEN_RECEIPT')
            tokens+=len(r or [])
            if record.get('resource_error') or record.get('truncated'):event='resource_or_truncation'
            elif record.get('accepted') is True:event='accepted_recorded'
            elif record.get('accepted') is False:event='rejected_recorded'
            elif record.get('error'):event='rejected_execution_receipt'
            elif 'state_after' in record:event='accepted_execution_receipt'
            else:event='unknown_missing_execution_evidence'
            execution[event]+=1
            if 'state_after' in record and 'state_before' in record and record['state_after']!=record['state_before']:
                execution['recorded_state_changes']+=1
        if tokens!=row['generated_tokens']:raise ValueError('GENERATED_TOKEN_COUNT_MISMATCH')
        generated+=tokens
        if usable:
            policy+=tokens
            if advantages[i]!=0:nonzero+=tokens
        ev=row['reward_evidence']
        cards.append(dict(episode_id=row['episode_id'],category=category(row),
            training_reward_unchanged=row['reward'],group_centered_advantage=advantages[i] if usable else None,
            generated_assistant_tokens=tokens,action_count=len(row['records']),execution_receipt_counts=dict(execution),
            completed_reaction_occurrences=len(row['reactions']),
            supported_unique_edge_types=ev.get('supported_reactions'),
            unsupported_reaction_occurrences=ev.get('unsupported_reactions'),
            untrusted_reference_matches=ev.get('untrusted_reference_matches'),
            structural_stock_closed=ev.get('structural_stock_closed'),
            reference_supported_closed=ev.get('verified_stock_closed'),
            pending_occurrences=len(row.get('pending',[])),stock_leaf_occurrences=len(row.get('stock_leaves',[])),
            terminal_status=row['status'],executor_replay_verified=False,auxiliary_role_audit_verified=False))
    return dict(task_id=group[0]['task_id'],behavior_version=group[0]['behavior_version'],G=4,
                rewards=rewards,advantages=advantages,usable=usable,
                informative=usable and any(v!=0 for v in advantages),generated_tokens=generated,
                learner_policy_tokens=policy,kl_tokens=policy,nonzero_advantage_policy_tokens=nonzero,
                effective_clipped_policy_gradient='NOT_MEASURED',scorecards=cards)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as reader:
        for block in iter(lambda:reader.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def atomic(path,data):
    descriptor,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.writing',dir=path.parent)
    temporary=Path(name)
    try:
        with os.fdopen(descriptor,'w') as writer:
            json.dump(data,writer,indent=2);writer.write('\n');writer.flush();os.fsync(writer.fileno())
        os.link(temporary,path)
    finally:temporary.unlink()


def replay(audit_path,output):
    audit_path,output=Path(audit_path).resolve(),Path(output).resolve()
    shared=Path('/aaa/fionafyang/buddy1')
    if not audit_path.is_relative_to(shared) or not output.is_relative_to(shared):
        raise ValueError('SHARED_STORAGE_REQUIRED')
    if output.exists():raise FileExistsError(output)
    audit=json.loads(audit_path.read_text())
    if (audit.get('scope')!='original_first_round_saved256_reward_replay_and_signal_ledger'
            or audit.get('status')!='PASS' or audit.get('mismatches')
            or not audit.get('witness_agreement') or not all(audit['witness_agreement'].values())):
        raise ValueError('COMPLETE_ORIGINAL_FROZEN_REWARD_REPLAY_REQUIRED')
    if audit.get('counts',{}).get('groups')!=64 or audit.get('counts',{}).get('episodes')!=256:
        raise ValueError('FULL64_GROUPS_256_EPISODES_REQUIRED')
    bindings=dict(audit['source_files_sha256']);bindings.update(audit['raw_artifacts_sha256'])
    bindings[audit['source_contract']]=audit['contract_sha256']
    def verify(item):
        path,wanted=item
        if sha(path)!=wanted:raise ValueError('AUDITED_INPUT_CHANGED:'+path)
    with ThreadPoolExecutor(max_workers=16) as pool:list(pool.map(verify,bindings.items()))
    ledger=[];categories=Counter();zero=Counter();totals=Counter()
    for path in sorted(audit['raw_artifacts_sha256']):
        if not path.endswith('/rewarded_episodes.json'):continue
        for group in json.loads(Path(path).read_text()):
            value=group_scorecards(group);value['source_path']=path;ledger.append(value)
            for key in ('generated_tokens','learner_policy_tokens','kl_tokens','nonzero_advantage_policy_tokens'):
                totals[key]+=value[key]
            totals['informative_groups']+=value['informative'];totals['groups']+=1
            for card in value['scorecards']:
                categories[card['category']]+=1;totals['episodes']+=1
                if card['training_reward_unchanged']==0:zero[card['category']]+=1
    expected=audit['counts']
    checks={'groups':'groups','episodes':'episodes','informative_groups':'informative_groups',
            'generated_tokens':'generated_tokens','learner_policy_tokens':'policy_tokens','kl_tokens':'policy_tokens',
            'nonzero_advantage_policy_tokens':'policy_tokens_with_nonzero_advantage'}
    if any(totals[k]!=expected[v] for k,v in checks.items()):raise ValueError('FULL_FIRST_ROUND_LEDGER_MISMATCH')
    report=dict(schema='route_reward_scorecards_v1',status='SCALAR_PRESERVING_AUDIT_PASS',
        source_audit=str(audit_path),source_audit_sha256=sha(audit_path),source_files_sha256=bindings,
        source_script_sha256=sha(__file__),counts=dict(totals),categories=dict(categories),zero_reward_categories=dict(zero),
        training_reward_changed=False,new_training_reward_enabled=False,chemistry_executions=0,optimizer_updates=0,
        limitation='Stored reports only. Independent executor/auxiliary-role audit and effective clipped gradients remain unverified.',
        group_scorecards=ledger)
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic(output,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('group_scorecards','source_files_sha256')},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();replay(args.audit,args.output)
