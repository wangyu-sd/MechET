"""All-edge literature coverage; target pairing never pools program contexts."""
from collections import defaultdict
import random


def source_supported(evidence):
    return (evidence.get('supported') is True and evidence.get('tier') in {'A','B'}
            and evidence.get('verification_status') in {'source_passage_checked','source_human_reviewed'})


def route_evidence(route, evidence):
    keys = sorted(set(route['reaction_keys']))
    checked = [key for key in keys if source_supported(evidence.get(key,{}))]
    exact = [key for key in checked if evidence[key]['tier']=='A']
    generated = bool(keys) and route.get('already_in_stock') is not True
    stock = route.get('stock_verified') is True and generated
    return {**route,'distinct_essential_reaction_count':len(keys),'supported_reaction_keys':checked,
            'unresolved_reaction_keys':[key for key in keys if key not in checked],
            'coverage':len(checked)/len(keys) if keys else None,
            'any_exact':stock and bool(exact),'all_exact':stock and len(exact)==len(keys),
            'precedent_covered':stock and len(checked)==len(keys),'nonzero_stock_solved':stock,
            'whole_route_laboratory_validated':False}


def paired_targets(universe, method_positive, baseline_positive, *, bootstrap_draws=2000, seed=17):
    universe = list(universe)
    if len(universe) != len(set(universe)):
        raise ValueError('target universe contains duplicate IDs')
    method, baseline = set(method_positive), set(baseline_positive)
    if not (method | baseline) <= set(universe):
        raise ValueError('paired success outside target universe')
    values = [int(key in method)-int(key in baseline) for key in universe]
    result = {'denominator':len(universe),'both':len(method&baseline),'method_only':len(method-baseline),
              'baseline_only':len(baseline-method),'neither':len(universe)-len(method|baseline),
              'difference':sum(values)/len(values) if values else None,
              'interpretation':'recorded-run descriptive; unequal work is not a causal search gain',
              'bootstrap_draws':bootstrap_draws,'seed':seed}
    if not values:
        result['difference_95ci'] = None
        return result
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(bootstrap_draws))
    result['difference_95ci'] = [draws[int(.025*(len(draws)-1))], draws[int(.975*(len(draws)-1))]] if draws else None
    return result


def paired_observed_targets(universe, method_positive, baseline_positive, *,
                            method_observed, baseline_observed, bootstrap_draws=2000, seed=17):
    universe=list(universe);allowed=set(universe)
    method_observed,baseline_observed=map(set,(method_observed,baseline_observed))
    method_positive,baseline_positive=map(set,(method_positive,baseline_positive))
    if ((method_observed|baseline_observed)-allowed or
        method_positive-method_observed or baseline_positive-baseline_observed):
        raise ValueError('observed/positive targets disagree with paired universe')
    result=paired_targets(universe,method_positive,baseline_positive,
                          bootstrap_draws=bootstrap_draws,seed=seed)
    common=method_observed&baseline_observed
    result.update(method_pending_reports=len(allowed-method_observed),
                  baseline_pending_reports=len(allowed-baseline_observed),pending_is_failure=False,
                  full_population_outcome='recorded positive discovery lower bound; neither includes unobserved targets',
                  observed_both=paired_targets([t for t in universe if t in common],
                    method_positive&common,baseline_positive&common,
                    bootstrap_draws=bootstrap_draws,seed=seed))
    return result


def disconnection_counts(occurrences, evidence, *, universe):
    by_target = {key:set() for key in universe}
    for row in occurrences:
        key = row.get('reaction_key')
        if row['target_id'] not in by_target:
            raise ValueError('reaction target outside declared universe')
        if key and source_supported(evidence.get(key,{})):
            by_target[row['target_id']].add(key)
    return {'unique_supported_reactions':len(set().union(*by_target.values())),
            'supported_per_target':{key:len(values) for key,values in by_target.items()},
            'any_supported_targets':sum(bool(values) for values in by_target.values()),
            'denominator':len(by_target)}


def aggregate_routes(routes, evidence, *, universe, observed_ids):
    by_target = defaultdict(list)
    for route in routes:
        by_target[route['target_id']].append(route_evidence(route,evidence))
    if set(by_target)-set(universe):
        raise ValueError('route outside original target universe')
    outcomes = []
    for key in universe:
        rows = by_target[key]
        outcomes.append({'target_id':key,'report_status':'observed' if key in observed_ids else 'pending',
                         'StockSolved@B':any(r['nonzero_stock_solved'] for r in rows),
                         'AnyExactStepPrecedent@B':any(r['any_exact'] for r in rows),
                         'AllStepExactPrecedentSolved@B':any(r['all_exact'] for r in rows),
                         'PrecedentCoveredSolved@B':any(r['precedent_covered'] for r in rows)})
    names=['StockSolved@B','AnyExactStepPrecedent@B','AllStepExactPrecedentSolved@B','PrecedentCoveredSolved@B']
    return {'denominator':len(universe),'observed_reports':len(observed_ids),
            'pending_reports':len(set(universe)-set(observed_ids)),
            'counts':{name:sum(row[name] for row in outcomes) for name in names},'targets':outcomes,
            'route_evidence':[row for rows in by_target.values() for row in rows]}
