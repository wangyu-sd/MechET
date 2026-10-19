"""Authenticate actual generation-wave costs at an immutable saved boundary.

Live ledgers may append. Only a uniquely matched complete prefix is charged;
tokens returned by a shared cache are never relabelled as new GPU generation.
"""
import hashlib
import math
from pathlib import Path


def snapshot_wave_prefix(path,*,output_tokens,solved_ids):
    import json
    if type(output_tokens) is not int or output_tokens<0:
        raise ValueError('invalid frozen generation-token boundary')
    solved_ids=set(solved_ids);seen=set();digest=hashlib.sha256();size=0;matches=[]
    integer_fields=('output_tokens','model_forward_calls','prompt_tokens','policy_request_batches',
                    'shared_policy_cache_hits','reused_output_tokens')
    float_fields=('inference_elapsed_seconds','cpu_executor_thread_seconds')
    previous={k:0 for k in (*integer_fields,*float_fields)}
    with Path(path).open('rb') as handle:
        for i,line in enumerate(handle):
            if not line.endswith(b'\n'):break
            row=json.loads(line)
            if type(row.get('output_tokens')) is not int or row['output_tokens']<0:
                raise ValueError('invalid generation counter')
            if row['output_tokens']>output_tokens:break
            if row.get('boundary')!='post_executor_whole_generation_wave_v1' or row.get('wave')!=i:
                raise ValueError('unsupported/reordered generation-wave boundary')
            for key in integer_fields:
                if type(row.get(key)) is not int or row[key]<previous[key]:
                    raise ValueError('invalid/regressing generation-wave counter: '+key)
            for key in float_fields:
                value=row.get(key)
                if (type(value) not in (int,float) or not math.isfinite(value) or value<previous[key]):
                    raise ValueError('invalid/regressing elapsed counter: '+key)
            added=row['new_stock_solved_ids']
            if len(added)!=len(set(added)) or set(added)&seen:
                raise ValueError('phantom or duplicate stock-route credit')
            seen.update(added)
            if row['credited_stock_solved_targets']!=len(seen):
                raise ValueError('stock credit count differs from actual IDs')
            if i==0 and any(row[k]!=0 for k in (*integer_fields,'cpu_executor_thread_seconds')):
                raise ValueError('unaccounted initial generation-wave cost')
            previous={k:row[k] for k in previous}
            digest.update(line);size+=len(line)
            # A cache-only wave can solve another target without generating
            # tokens. Authenticate the historical boundary with BOTH counters
            # and its solved-ID census, not every later token-plateau row.
            if row['output_tokens']==output_tokens and seen==solved_ids:
                matches.append((row,sorted(seen),digest.hexdigest(),size))
    if len(matches)!=1:
        raise ValueError('frozen token boundary missing or ambiguous in actual work ledger')
    row,credited,sha,size=matches[0]
    if set(credited)!=solved_ids:
        raise ValueError('frozen stock-route witnesses differ from work-ledger prefix')
    return {'source_path':str(Path(path).resolve()),'source_prefix_bytes':size,'source_prefix_sha256':sha,
        'last_whole_wave':row['wave'],'actual_generated_output_tokens':row['output_tokens'],
        'unique_scheduled_model_forward_calls':row['model_forward_calls'],
        'logical_prompt_tokens_not_uncached_prefill':row['prompt_tokens'],
        'policy_request_batches':row['policy_request_batches'],
        'reused_output_tokens_not_charged':row['reused_output_tokens'],
        'shared_policy_cache_hits':row['shared_policy_cache_hits'],
        'cpu_executor_thread_seconds':row['cpu_executor_thread_seconds'],
        'worker_inference_elapsed_seconds':row['inference_elapsed_seconds'],
        'frozen_stock_solved_ids':credited,'actual_gpu_active_seconds':None,
        'scope':'all work in original shard, including unfinished targets; not solved-target-only cost',
        'matched_compute_claim':False}


def audit_r1_snapshot(snapshot_path,freeze_root):
    """Align each actual shard's whole-wave prefix to the PR88 frozen census."""
    import gzip
    import json
    from .literature_sources import file_sha256
    snapshot_path=Path(snapshot_path);freeze_root=Path(freeze_root)
    freeze_path=freeze_root/'prediction_freeze_manifest.json'
    freeze=json.loads(freeze_path.read_bytes())
    if freeze.get('complete') is not True or file_sha256(snapshot_path)!=freeze['progress_snapshot_sha256']:
        raise ValueError('work audit requires authenticated complete prediction snapshot')
    for name,description in freeze['output_files'].items():
        if file_sha256(freeze_root/name)!=description['sha256']:
            raise ValueError('frozen prediction table checksum differs: '+name)
    def rows(name):
        with gzip.open(freeze_root/(name+'.jsonl.gz'),'rt') as f:
            return [json.loads(l) for l in f]
    snapshot=json.loads(snapshot_path.read_bytes());run='r1_mcts_pr88'
    population=[r for r in rows('target_population') if r['run_id']==run]
    if len({r['target_id'] for r in population})!=len(population):
        raise ValueError('duplicated work population')
    observed={r['target_id'] for r in population if r['report_status']=='observed'}
    sources=[r for r in freeze['source_reports'] if r['run_id']==run]
    if len(sources)!=len(observed) or {r['target_id'] for r in sources}!=observed:
        raise ValueError('observed work population differs from frozen report inventory')
    solved={r['target_id'] for r in rows('frozen_routes') if r['run_id']==run and
            r.get('stock_verified') is True and r.get('reaction_keys') and not r.get('already_in_stock')}
    summary=snapshot['r1']
    if (summary['full_denominator']!=len(population) or summary['saved_reports']!=len(observed) or
        summary['nonzero_stock_connected']!=len(solved) or
        summary['unfinished']!=len(population)-len(observed)):
        raise ValueError('snapshot and frozen stock-route populations differ')
    shards=[];seen_ranks=set();seen_targets=set()
    for entry in snapshot['sources']['r1_progress']:
        rank=entry['rank'];root=Path(entry['path']).parent.resolve()
        if rank in seen_ranks:raise ValueError('duplicate work rank')
        seen_ranks.add(rank)
        current=[s for s in sources if Path(s['report_path']).resolve().is_relative_to(root)]
        if not current:raise ValueError('work shard has no frozen source-report binding')
        manifest_path=root/'manifest.json';contract=json.loads(manifest_path.read_bytes())
        sha=file_sha256(manifest_path)
        if (contract.get('rank')!=rank or contract.get('adapter_sha')!=freeze['actor_sha256'] or
            any(Path(s['manifest_path']).resolve()!=manifest_path or s['manifest_sha256']!=sha or
                s['actor_sha256']!=freeze['actor_sha256'] for s in current)):
            raise ValueError('work shard actor/manifest differs from frozen report contract')
        ids={s['target_id'] for s in current}
        if ids&seen_targets:raise ValueError('work target assigned to multiple ranks')
        seen_targets.update(ids)
        if len(ids&solved)!=entry['stock_solved_targets']:
            raise ValueError('work snapshot solved-count differs from frozen route witnesses')
        result=snapshot_wave_prefix(root/'work_waves.jsonl',output_tokens=entry['output_tokens'],solved_ids=ids&solved)
        shards.append({**result,'rank':rank,'shard_target_count':contract['shard_rows'],
            'source_manifest_path':str(manifest_path),'source_manifest_sha256':sha,
            'declared_world':contract['world']})
    if (seen_targets!=observed or any(s['declared_world']!=len(shards) for s in shards) or
        sum(s['shard_target_count'] for s in shards)!=len(population) or
        seen_ranks!=set(range(len(shards)))):
        raise ValueError('work-shard census does not cover complete original population')
    additive=('actual_generated_output_tokens','unique_scheduled_model_forward_calls',
        'logical_prompt_tokens_not_uncached_prefill','policy_request_batches',
        'reused_output_tokens_not_charged','shared_policy_cache_hits','cpu_executor_thread_seconds')
    return {'schema':'lgrv_frozen_r1_actual_work_v1','run_id':run,
        'snapshot_path':str(snapshot_path.resolve()),'snapshot_sha256':file_sha256(snapshot_path),
        'prediction_freeze_manifest_sha256':file_sha256(freeze_path),
        'actor_sha256':freeze['actor_sha256'],'original_target_denominator':len(population),
        'observed_reports':len(observed),'pending_reports':len(population)-len(observed),
        'nonzero_stock_connected_targets':len(solved),'shards':shards,
        'totals':{k:sum(s[k] for s in shards) for k in additive},
        'summed_worker_inference_elapsed_seconds':sum(s['worker_inference_elapsed_seconds'] for s in shards),
        'worker_elapsed_is_global_wall_time':False,'actual_gpu_active_seconds':None,
        'fixed_observed_reports_only_cost_claim':False,'matched_compute_claim':False,'new_model_inference':False}


if __name__=='__main__':
    import argparse
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--progress-snapshot',type=Path,required=True)
    parser.add_argument('--prediction-freeze-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit_r1_snapshot(args.progress_snapshot,args.prediction_freeze_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'phase':'frozen_actual_work_audited','targets':result['original_target_denominator'],
        'actual_generated_output_tokens':result['totals']['actual_generated_output_tokens'],'output':str(args.output)}),flush=True)
