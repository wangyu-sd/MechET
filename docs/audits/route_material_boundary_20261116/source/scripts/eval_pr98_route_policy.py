"""Complete-population same-budget 2x2 route-result ingestion; no Top-K ranking.

Missing conditions/targets remain errors, not dropped denominators or invented
predictions. This does not launch a benchmark implicitly during training.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

from mechet.route_training_contract import atomic_json as durable_json
from mechet.route_training_contract import chemical_identity,public_molecule
from scripts.audit_pr98_route_train import sha
from mechet.endpoints import ROUTE_ENDPOINT_CONTRACT


CONDITIONS=('stage2_independent','stage2_tree','route_rl_independent','route_rl_tree')


def validate_factorial(conditions):
    if set(conditions)!=set(CONDITIONS):
        raise ValueError('INCOMPLETE_FACTORIAL_CONDITIONS')
    if any(c.get('route_endpoint_contract')!=ROUTE_ENDPOINT_CONTRACT for c in conditions.values()):
        raise ValueError('UNRECOGNIZED_ROUTE_ENDPOINT_CONTRACT')
    from mechet.route_materials import material_contract
    if len({material_contract(c) for c in conditions.values()})!=1:
        raise ValueError('UNMATCHED_FACTORIAL:stock_material_contract')
    for key in ('population_sha','stock_sha','token_budget','executor_sha','prompt_sha'):
        if len({conditions[name][key] for name in CONDITIONS})!=1:
            raise ValueError('UNMATCHED_FACTORIAL:'+key)
    for stage in ('stage2','route_rl'):
        if conditions[stage+'_independent']['actor_sha256']!=conditions[stage+'_tree']['actor_sha256']:
            raise ValueError('UNMATCHED_FACTORIAL_ACTOR:'+stage)


def validate_route_certificate(report,stock):
    """Reconstruct all occurrences; never trust a self-reported solved flag."""
    if report.get('pending'):
        raise ValueError('INCOMPLETE_ROUTE_OBLIGATIONS')
    from mechet.route_materials import material_contract, validate_material_edge, EXACT_MATERIAL
    material=material_contract(report)
    edges={r['occurrence']:r for r in report['reactions']}
    if len(edges)!=len(report['reactions']):
        raise ValueError('DUPLICATE_ROUTE_OCCURRENCE')
    pending=[(chemical_identity(report['root']),'0',frozenset(),0)]
    used=set(); leaves=[]; depth=0
    while pending:
        product,occurrence,ancestors,level=pending.pop()
        if product in ancestors:
            raise ValueError('ROUTE_OBLIGATION_CYCLE')
        if product in stock:
            leaves.append(product); depth=max(depth,level)
            continue
        if occurrence not in edges:
            raise ValueError('UNRESOLVED_ROUTE_OBLIGATION:'+occurrence)
        edge=edges[occurrence]; used.add(occurrence)
        if chemical_identity(edge['product'])!=product or not edge['precursors']:
            raise ValueError('WRONG_ROUTE_OBLIGATION_EDGE')
        children=[chemical_identity(c) for c in edge['precursors']]
        validate_material_edge(edge,stock,material)
        mixture=chemical_identity('.'.join(children))
        if mixture==product:
            raise ValueError('NONPRODUCTIVE_ROUTE_EDGE')
        # Material aliases are per original contributor occurrence. Joining
        # their results must not merge distinct occurrences into a stock salt.
        expected=children if material!=EXACT_MATERIAL else ([mixture] if mixture in stock else mixture.split('.'))
        if children!=expected:
            raise ValueError('ROUTE_SALT_OR_MULTIPLICITY_CHANGED')
        pending.extend((c,occurrence+'.'+str(i),ancestors|{product},level+1) for i,c in enumerate(children))
    if used!=set(edges) or not used:
        raise ValueError('UNUSED_OR_ZERO_STEP_ROUTE_EDGES')
    if Counter(leaves)!=Counter(report['stock_leaves']) or depth!=report['depth']:
        raise ValueError('ROUTE_LEAF_OR_DEPTH_EVIDENCE_MISMATCH')
    return dict(depth=depth,reaction_count=len(used),stock_leaves=leaves)


def replay_complete_route(report):
    """Verify every reaction action is both raw-predicted and executable."""
    from mechet.endpoints import ROUTE_ENDPOINT_CONTRACT
    if report.get('route_endpoint_contract') != ROUTE_ENDPOINT_CONTRACT:
        raise ValueError('UNRECOGNIZED_ROUTE_ENDPOINT_CONTRACT')
    from mechet.route_materials import material_contract
    material_contract(report)
    from mechet.route_policy_episode import AGENTIC_ROUTE_OBSERVATIONS,RouteEpisode
    if report.get('public_observation_contract') in AGENTIC_ROUTE_OBSERVATIONS:
        # Failed attempts are part of the actual policy trajectory. Replaying
        # only the accepted subsequence would not verify its feedback/history.
        stock={chemical_identity(v) for v in report['stock_leaves']}
        ep=RouteEpisode(dict(behavior_version=report['behavior_version'],
            route_endpoint_contract=report['route_endpoint_contract'],
            episode_token_budget=float('inf'),public_observation_contract=report['public_observation_contract'],
            max_consecutive_recoveries=report['max_consecutive_recoveries'],
            **({k:report[k] for k in ('stock_material_contract',) if k in report})),
            report['public_root'],stock,report['episode_id'],report['seed'])
        for output in report['records']:
            ep.accept(output)
        if ep.status!='stock_solved':
            raise ValueError('ROUTE_REPLAY_NOT_STOCK_CLOSED')
        actual=ep.finish_report()
        for field in ('reactions','stock_leaves','depth','pending'):
            if actual[field]!=report[field]:
                raise ValueError('ROUTE_REPLAY_EVIDENCE_MISMATCH:'+field)
        return
    from mechet.route_policy_episode import NativeChemistry
    from mechet.pr96_evaluation import parsed
    from scripts.build_mech31k_main_product_only_sft import same_chemical_state
    chemistry=NativeChemistry()
    records=report['records']
    for edge in report['reactions']:
        actions=[r for r in records if r['occurrence']==edge['occurrence']]
        if len(actions)!=len(edge['actions']) or not actions:
            raise ValueError('ROUTE_RAW_ACTION_EVIDENCE_MISSING')
        if chemical_identity(edge['public_product'])!=edge['product']:
            raise ValueError('PUBLIC_REACTION_OBSERVATION_IDENTITY_MISMATCH')
        if [chemical_identity(v) for v in edge['public_precursors']]!=edge['precursors']:
            raise ValueError('PUBLIC_PRECURSOR_OBSERVATION_IDENTITY_MISMATCH')
        node=chemistry.root(edge['public_product']); initial=node
        if not same_chemical_state(node.state,edge['initial_state']):
            raise ValueError('ROUTE_INITIAL_STATE_MISMATCH')
        flow=False
        for output,claimed in zip(actions,edge['actions']):
            name,args,error=parsed(output)
            if error or name!=claimed['name'] or args!=claimed['arguments'] or node.terminal:
                raise ValueError('ROUTE_PREDICTED_PROGRAM_MISMATCH')
            child,error=chemistry.step(node,dict(output,behavior_logprobs=output.get('behavior_logprobs',[])))
            if child is None:
                raise ValueError('ROUTE_ELECTRON_REPLAY:'+error)
            flow=flow or name=='apply_electron_flow'; node=child
        from mechet.route_materials import material_contract,project_material_endpoints
        actual=project_material_endpoints(chemistry.precursors(node,initial),
            set(report['stock_leaves']),material_contract(report))['precursors']
        if not node.terminal or not flow or actual!=edge['precursors']:
            raise ValueError('ROUTE_ELECTRON_ENDPOINT_MISMATCH')
        if not same_chemical_state(node.state,edge['final_state']):
            raise ValueError('ROUTE_FINAL_STATE_MISMATCH')


def verify_result_bundle(name,condition):
    if condition.get('route_endpoint_contract')!=ROUTE_ENDPOINT_CONTRACT:
        raise ValueError('UNRECOGNIZED_ROUTE_ENDPOINT_CONTRACT')
    manifest_path=Path(condition['result_manifest'])
    if sha(manifest_path)!=condition['result_manifest_sha256']:
        raise ValueError('RESULT_MANIFEST_ARTIFACT_CHANGED')
    manifest=json.loads(manifest_path.read_text())
    from mechet.route_materials import material_contract
    if material_contract(manifest)!=material_contract(condition):
        raise ValueError('RESULT_PROVENANCE_MISMATCH:stock_material_contract')
    for key in ('population_sha','stock_sha','token_budget','executor_sha','prompt_sha','actor_sha256','route_endpoint_contract'):
        if manifest[key]!=condition[key]:
            raise ValueError('RESULT_PROVENANCE_MISMATCH:'+key)
    if manifest['condition']!=name or manifest['actor_file']!=condition['actor_file']:
        raise ValueError('RESULT_ACTOR_CONDITION_MISMATCH')
    if sha(manifest['actor_file'])!=manifest['actor_sha256']:
        raise ValueError('RESULT_ACTOR_ARTIFACT_CHANGED')
    files=manifest['files_sha256']
    if manifest['results_file'] not in files or not manifest['raw_files'] or any(p not in files for p in manifest['raw_files']):
        raise ValueError('RESULT_RAW_ARTIFACTS_MISSING')
    for path,digest in files.items():
        if sha(path)!=digest:
            raise ValueError('RESULT_ARTIFACT_CHANGED:'+path)
    raw={}
    for path in manifest['raw_files']:
        with Path(path).open() as reader:
            for line in reader:
                event=json.loads(line)
                if event.get('event') in ('behavior-receipt','recovered-behavior-receipt'):
                    raw[event['request_id']]=event
                elif event.get('event')=='gpu-generation-returned':
                    for uid,tokens,text in zip(event['request_ids'],event['generated_token_ids'],event['raw_outputs']):
                        raw[uid]=dict(generated_token_ids=tokens,raw=text)
    rows=json.loads(Path(manifest['results_file']).read_text())
    for row in rows:
        if material_contract(row)!=material_contract(condition):
            raise ValueError('ROW_MATERIAL_PROVENANCE_MISMATCH')
        if row.get('route_endpoint_contract')!=ROUTE_ENDPOINT_CONTRACT:
            raise ValueError('UNRECOGNIZED_ROUTE_ENDPOINT_CONTRACT')
        if row['actor_sha256']!=condition['actor_sha256']:
            raise ValueError('ROW_ACTOR_PROVENANCE_MISMATCH')
        if row['generated_tokens']>condition['token_budget']:
            raise ValueError('ROW_EXCEEDED_FROZEN_BUDGET')
        for record in row['records']:
            saved=raw.get(record['request_id'])
            if saved is None or any(saved[k]!=record[k] for k in ('generated_token_ids','raw')):
                raise ValueError('ROW_RAW_PREDICTION_EVIDENCE_MISMATCH')
    return rows


def summarize_routes(population,rows,stocks=None,roots=None):
    ids=[r['target_id'] for r in rows]
    if len(set(ids))!=len(ids) or set(ids)!=set(population) or len(ids)!=len(population):
        raise ValueError('INCOMPLETE_OR_DUPLICATE_EVAL_POPULATION')
    if stocks is None or roots is None or set(stocks)!=set(population) or set(roots)!=set(population):
        raise ValueError('FROZEN_STOCK_AND_PUBLIC_ROOTS_REQUIRED')
    if any(r.get('route_endpoint_contract')!=ROUTE_ENDPOINT_CONTRACT for r in rows):
        raise ValueError('UNRECOGNIZED_ROUTE_ENDPOINT_CONTRACT')
    counts=Counter(r['status'] for r in rows)
    solved=[r for r in rows if r['status']=='stock_solved']
    for r in solved:
        if r['reward']!=1. or not r['reactions'] or r['depth']<1:
            raise ValueError('ZERO_STEP_OR_INCOMPLETE_ROUTE_REWARDED')
        validate_route_certificate(r,stocks[r['target_id']])
    for r in rows:
        if chemical_identity(r['root'])!=chemical_identity(roots[r['target_id']]):
            raise ValueError('RESULT_PUBLIC_ROOT_CHANGED')
        if r['status']=='already_stock' and (r['root'] not in stocks[r['target_id']] or r['reactions'] or r['reward']!=0.):
            raise ValueError('FALSE_ZERO_STEP_STOCK_RESULT')
    return dict(denominator=len(population),solved=len(solved),
        solved_rate_lower_bound=len(solved)/len(population),censored=counts['resource_censored'],
        already_stock=counts['already_stock'],statuses=dict(counts),
        solved_depth_histogram=dict(Counter(r['depth'] for r in solved)),
        generated_tokens=sum(r['generated_tokens'] for r in rows))


def paired_bootstrap(population,left,right,seed=17,reps=2000):
    a={r['target_id']:r['status']=='stock_solved' for r in left}
    b={r['target_id']:r['status']=='stock_solved' for r in right}
    delta=[int(b[x])-int(a[x]) for x in population]
    rng=random.Random(seed)
    values=sorted(sum(rng.choice(delta) for _ in delta)/len(delta) for _ in range(reps))
    return dict(delta=sum(delta)/len(delta),ci95=[values[int(.025*reps)],values[int(.975*reps)]],
                censored_counted_as_unsolved_lower_bound=True)


def evaluate_routes(eval_contract,actors=None):
    validate_factorial(eval_contract['conditions'])
    # Public root/stock assignment frozen independently of result labels.
    if sha(eval_contract['population_file'])!=eval_contract['conditions']['stage2_independent']['population_sha']:
        raise ValueError('EVAL_POPULATION_ARTIFACT_CHANGED')
    targets=[json.loads(line) for line in Path(eval_contract['population_file']).open() if line.strip()]
    population=[r['id'] for r in targets]
    if population!=eval_contract['target_ids'] or len(set(population))!=len(population):
        raise ValueError('EVAL_FROZEN_POPULATION_MISMATCH')
    stock_spec=eval_contract['stock_assignments']
    from mechet.shared_electron_search import fingerprint
    if fingerprint(stock_spec)!=eval_contract['conditions']['stage2_independent']['stock_sha']:
        raise ValueError('EVAL_STOCK_ASSIGNMENT_CHANGED')
    stocks={}
    for name,record in stock_spec.items():
        if sha(record['path'])!=record['sha256']:
            raise ValueError('EVAL_STOCK_ARTIFACT_CHANGED')
        stocks[name]=set(Path(record['path']).read_text().splitlines())
    by_target={r['id']:stocks[r['stock']] for r in targets}
    roots={r['id']:r['product_unmapped'] for r in targets}
    # Callbacks may schedule inference but their unverified return values are
    # NEVER used as scores. Only authenticated, published bundles are consumed.
    if actors:
        for name in CONDITIONS:
            actors[name](population,eval_contract['conditions'][name])
    outputs={name:verify_result_bundle(name,eval_contract['conditions'][name]) for name in CONDITIONS}
    reports={name:summarize_routes(population,rows,by_target,roots) for name,rows in outputs.items()}
    for rows in outputs.values():
        for row in rows:
            if row['public_root']!=public_molecule(roots[row['target_id']]):
                raise ValueError('ACTOR_PUBLIC_INPUT_OBSERVATION_CHANGED')
            if row['status']=='stock_solved':
                replay_complete_route(row)
    reports['paired_training_effect']={mode:paired_bootstrap(population,outputs['stage2_'+mode],outputs['route_rl_'+mode])
                                       for mode in ('independent','tree')}
    return reports


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args(); cfg=json.loads(args.contract.read_text())
    durable_json(args.output,evaluate_routes(cfg))
