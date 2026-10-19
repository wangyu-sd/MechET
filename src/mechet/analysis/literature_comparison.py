"""Compare chemical discoveries at identical target IDs and explicit work units."""
import math

from .literature_route_metrics import source_supported


def reaction_target_outcomes(occurrences, evidence, *, universe):
    universe=list(universe)
    if len(universe)!=len(set(universe)):
        raise ValueError('duplicate target universe')
    by_target={target:set() for target in universe}
    for row in occurrences:
        if row['target_id'] not in by_target:
            raise ValueError('reaction outside target universe')
        by_target[row['target_id']].add(row['reaction_key'])
    targets={}
    for target,keys in by_target.items():
        supported={key for key in keys if source_supported(evidence.get(key,{}))}
        exact={key for key in supported if evidence[key]['tier']=='A'}
        checked={key for key in keys if evidence.get(key,{}).get('verification_status') in
                 {'source_passage_checked','source_human_reviewed'}}
        targets[target]={'reaction_keys':sorted(keys),'supported_reaction_keys':sorted(supported),
                         'exact_reaction_keys':sorted(exact),
                         'distinct_generated_reactions':len(keys),
                         'distinct_supported_disconnections':len(supported),
                         'source_checked_fraction':len(checked)/len(keys) if keys else None}
    unique=set().union(*by_target.values())
    return {'denominator':len(universe),'targets':targets,
            'any_supported_targets':[t for t,r in targets.items() if r['supported_reaction_keys']],
            'exact_supported_targets':[t for t,r in targets.items() if r['exact_reaction_keys']],
            'unique_generated_reactions':len(unique),
            'source_checked_unique_reactions':sum(evidence.get(key,{}).get('verification_status') in
                {'source_passage_checked','source_human_reviewed'} for key in unique)}


def compare_disconnections(method, baseline, evidence, *, universe):
    left=reaction_target_outcomes(method,evidence,universe=universe)
    right=reaction_target_outcomes(baseline,evidence,universe=universe)
    targets={};pairs=[0,0,0];all_left=set();all_right=set()
    for target in left['targets']:
        a=set(left['targets'][target]['supported_reaction_keys'])
        b=set(right['targets'][target]['supported_reaction_keys'])
        targets[target]={'both':sorted(a&b),'method_only':sorted(a-b),'baseline_only':sorted(b-a)}
        for index,keys in enumerate((a&b,a-b,b-a)):
            pairs[index]+=len(keys)
        all_left|=a;all_right|=b
    return {'denominator':left['denominator'],'targets':targets,
            'unique_supported_reactions':{'both':len(all_left&all_right),
                'method_only':len(all_left-all_right),'baseline_only':len(all_right-all_left)},
            'target_reaction_pairs':dict(zip(('both','method_only','baseline_only'),pairs)),
            'interpretation':'recorded-run descriptive; unequal work is not a causal search gain'}


def discovery_efficiency(work, *, supported_disconnections, covered_targets):
    values=[row['values'] for row in work.get('per_target',[])]
    def valid(value):
        return (isinstance(value,(int,float)) and not isinstance(value,bool)
                and math.isfinite(value) and value>=0)
    names=('sampled_tokens','reused_output_tokens','executor_calls','wall_seconds')
    missing={name:sum(not valid(row.get(name)) for row in values) for name in names}
    sums={name:sum(row[name] for row in values if valid(row.get(name))) for name in names}
    new_tokens=None
    if values and not missing['sampled_tokens'] and not missing['reused_output_tokens']:
        # Native sampled_tokens includes cache returns. Resume inheritance can
        # differ from current-runtime cost counters: expose any inconsistency,
        # and do not relabel this difference as measured whole-cluster compute.
        differences=[row['sampled_tokens']-row['reused_output_tokens'] for row in values]
        if all(n>=0 for n in differences):
            new_tokens=sum(differences)
    calls=sums['executor_calls'] if values and not missing['executor_calls'] else None
    def ratio(numerator,denominator):
        return numerator/denominator if denominator is not None and denominator>0 else None
    return {'observed_work_targets':len(values),'reported_work_sums':sums,
            'missing_cost_target_counts':missing,
            'report_accounted_nonreused_output_tokens':new_tokens,
            'supported_disconnections_per_million_report_accounted_nonreused_tokens':ratio(1000000*supported_disconnections,new_tokens),
            'precedent_covered_targets_per_million_report_accounted_nonreused_tokens':ratio(1000000*covered_targets,new_tokens) if covered_targets is not None else None,
            'supported_disconnections_per_executor_call':ratio(supported_disconnections,calls),
            'summed_target_elapsed_seconds':sums['wall_seconds'] if values and not missing['wall_seconds'] else None,
            'elapsed_is_global_wall_time':False,'actual_cluster_gpu_seconds':None,
            'work_basis':'source-report accounting; shared-cache and resumed-history costs are not inferred GPU FLOPs/time',
            'actual_generation_token_total_requires_run_ledger':True,
            'matched_compute_gain_claim':False}
