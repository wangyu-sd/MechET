"""Actual saved electronic programs -> frozen complete chemical reactions.

Program novelty is relative to the specified actor's training exposure and the
historical v1 signature abstraction. It is neither new chemistry nor a C1
benchmark label. Experimental evidence is attached separately by reaction key.
"""
from collections import Counter, defaultdict

from .literature_reaction_keys import canonical_components, make_reaction, stable_hash
from .literature_route_metrics import route_evidence, source_supported


def authenticate_exposure(exposure, actor_sha):
    if exposure.get('model_sha256') != actor_sha:
        raise ValueError('training exposure does not belong to frozen actor')
    if exposure.get('signature_version') != 'source_to_sink_execution_moves_v1':
        raise ValueError('unsupported training signature version')
    actual = exposure.get('actual_train_exposure', {})
    if not isinstance(actual.get('program_counts'), dict) or not isinstance(actual.get('primitive_counts'), dict):
        raise ValueError('complete actual actor training counts required')
    return actual


def bind_program_states(actions, root_state, terminal_state):
    """Bind saved path endpoints and available public result-state continuity."""
    from rdkit import Chem
    def private(smiles):
        parser=Chem.SmilesParserParams();parser.removeHs=False
        molecule=Chem.MolFromSmiles(smiles,parser)
        if molecule is None:
            raise ValueError('unparseable saved private state')
        return Chem.MolToSmiles(molecule,canonical=True,isomericSmiles=True)
    if not actions:
        return {'valid':False,'reason':'missing action sequence','continuity_checks':0}
    checked=0
    try:
        if private(actions[0]['state_before'])!=private(root_state):
            raise ValueError('first action state does not match native reaction root')
        if private(actions[-1]['state_before'])!=private(terminal_state):
            raise ValueError('last action state does not match terminal context')
        for left,right in zip(actions,actions[1:]):
            observed=left.get('result',{}).get('current_state')
            if observed:
                checked+=1
                if canonical_components(observed)!=canonical_components(right['state_before']):
                    raise ValueError('recorded public result differs from next private state')
        final=actions[-1].get('result',{}).get('derived_precursor')
        if final and canonical_components(final)!=canonical_components(terminal_state):
            raise ValueError('finish result differs from terminal state')
    except (ValueError,KeyError,RuntimeError) as error:
        return {'valid':False,'reason':str(error),'continuity_checks':checked}
    return {'valid':True,'continuity_checks':checked,
            'basis':'map-preserving endpoints plus available map-free public result continuity; not independent executor replay'}


def _complexity(actions, root_state, structural_state):
    from .tree_chemistry import molecular_delta
    flow = [a for a in actions if a.get('name') == 'apply_electron_flow']
    counts = [len(a.get('arguments', {}).get('electron_flow', [])) for a in flow]
    result = {'flow_events':len(flow), 'source_sink_move_count':sum(counts),
              'coupled_flow_events':sum(n > 1 for n in counts),
              'action_count':len(actions),
              'basis':'native product-to-structural-precursor graph difference; imports can change atom inventory',
              'physical_elementary_mechanism_established':False}
    try:
        delta = molecular_delta(root_state, structural_state)
        result.update(changed_bond_count=len(delta['changed_bonds']),
                      changed_atom_count=len(delta['changed_atom_maps']),
                      cycle_rank_delta=delta['cycle_rank_delta'],
                      net_charge_delta=delta['net_charge_delta'],
                      atom_inventory_preserved=delta['atom_inventory_preserved'],
                      delta_status='available')
    except (ValueError, RuntimeError) as error:
        result.update(delta_status='unavailable', delta_error=str(error),
                      changed_bond_count=None, changed_atom_count=None)
    return result


def join_report_programs(report, tree, occurrences, exposure, *, actor_sha):
    """Keep every saved terminal program for an eligible reaction in THIS report.

    Structural ownership uses the executor's existing product-atom projection.
    New retrieval identities are not invented. All program paths and failed
    annotations survive; an absent/incomplete signature is never called seen.
    """
    actual = authenticate_exposure(exposure, actor_sha)
    if tree.get('case_id') != report['id'] or canonical_components(tree['target']) != canonical_components(report['target']):
        raise ValueError('tree/report target identity differs')
    for key in ('sampled_tokens', 'raw_candidates', 'node_expansions', 'executor_calls'):
        left, right = report.get('work', {}).get(key), tree.get('work', {}).get(key)
        if left is not None and right is not None and left != right:
            raise ValueError('tree snapshot work differs from frozen report: ' + key)
    eligible = defaultdict(list)
    ownership={(r['run_id'],r['source']['report_sha256']) for r in occurrences}
    if len(ownership)>1:
        raise ValueError('occurrences mix different report or run identities')
    for row in occurrences:
        if row['target_id'] != report['id'] or row['source']['actor_sha256'] != actor_sha:
            raise ValueError('occurrence belongs to another target or actor')
        eligible[row['reaction_key']].append(row)
    from mechet.endpoints import split_precursor_endpoints
    from .tree_chemistry import program_support
    records = []
    if 'reaction_trees' in tree:
        inners=[(product,inner,'/reaction_trees/'+product.replace('~','~0').replace('/','~1'))
                for product,inner in tree['reaction_trees'].items()]
    elif 'contexts' in tree and 'root_context' in tree:
        inners=[(tree['target'],tree,'')]
    else:
        raise ValueError('unsupported saved native tree layout')
    for product, inner, inner_pointer in inners:
        contexts = {c['id']:c for c in inner.get('contexts', [])}
        if len(contexts) != len(inner.get('contexts', [])):
            raise ValueError('duplicate native context in reaction tree')
        root_state = contexts[inner['root_context']]['state']
        if canonical_components(root_state) != canonical_components(product):
            raise ValueError('native reaction root differs from recorded product')
        for index, context in enumerate(inner.get('contexts', [])):
            if context.get('terminal') is not True:
                continue
            parts = split_precursor_endpoints(context['state'], root_state)
            if not parts.structural:
                continue
            reaction = make_reaction(product, parts.structural,
                mapped_product=root_state, mapped_precursors=parts.structural,
                auxiliary=parts.auxiliary)
            matching = eligible.get(reaction['reaction_key'], [])
            if not matching:
                continue
            actions = context.get('actions', [])
            binding=bind_program_states(actions,root_state,context['state'])
            try:
                if binding['valid'] is not True:
                    raise ValueError(binding['reason'])
                support = program_support(actions, actual)
                signature = support.get('complete_signature')
                classification = ('seen_ordinary' if support['complete_signature_training_count'] > 0 else 'unseen_ordinary') if signature else (
                    'nonordinary' if support.get('ordinary_source_sink_program') is False else 'unavailable')
            except (ValueError, KeyError, TypeError, RuntimeError) as error:
                support = {'complete_signature':None, 'complete_signature_training_count':None,
                           'program_complete':False, 'support_error':str(error)}
                classification = 'unavailable'
            pointer = inner_pointer + f'/contexts/{index}/actions'
            base = matching[0]
            records.append({**support,
                'program_record_id':stable_hash([base['run_id'], report['id'], product, context['id']]),
                'run_id':base['run_id'], 'target_id':report['id'], 'cohort':report.get('cohort'),
                'reaction_key':reaction['reaction_key'], 'reaction_product':product,
                'occurrence_ids':sorted(r['occurrence_id'] for r in matching),
                'native_context_id':context['id'], 'program_source_pointer':pointer,
                'program_payload_sha256':stable_hash(actions),
                'exposure_class':classification,
                'program_state_binding':binding,
                'whole_program_unseen':classification == 'unseen_ordinary' if support.get('complete_signature') else None,
                'actor_exposure_authenticated':True, 'actor_sha256':actor_sha,
                'signature_scope':'actual actor exposure; historical v1 ignores import identities',
                'route_C1_claim':False, 'independent_chemical_precedent':False,
                'mapped_product':root_state, 'mapped_structural_precursors':parts.structural,
                'mapping_status':reaction['mapping_status'],
                'reaction_family_key':reaction['reaction_family_key'],
                'complexity':_complexity(actions, root_state, parts.structural)})
    return records


def composition_metrics(programs, evidence, routes, *, universe):
    universe = list(universe)
    if len(set(universe)) != len(universe):
        raise ValueError('duplicate target in composition denominator')
    if {r['target_id'] for r in programs} - set(universe):
        raise ValueError('program outside target universe')
    if {r['target_id'] for r in routes} - set(universe):
        raise ValueError('route outside target universe')
    if len({r['run_id'] for r in [*programs,*routes] if r.get('run_id')})>1:
        raise ValueError('composition aggregation mixes different runs')
    def group(rows):
        supported = [r for r in rows if source_supported(evidence.get(r['reaction_key'], {}))]
        checked = [r for r in rows if evidence.get(r['reaction_key'],{}).get('verification_status') in
                   {'source_passage_checked','source_human_reviewed'}]
        complexity = Counter()
        for row in rows:
            complexity[str(row.get('complexity', {}).get('changed_bond_count'))] += 1
        return {'program_occurrences':len(rows),
                'unique_program_signatures':len({r['complete_signature'] for r in rows if r.get('complete_signature')}),
                'unique_complete_reactions':len({r['reaction_key'] for r in rows}),
                'targets':len({r['target_id'] for r in rows}),
                'source_supported_unique_reactions':len({r['reaction_key'] for r in supported}),
                'source_supported_targets':len({r['target_id'] for r in supported}),
                'source_checked_coverage_fraction':len(checked)/len(rows) if rows else None,
                'source_supported_program_fraction':len(supported)/len(rows) if rows else None,
                'graph_changed_bond_count_by_program_occurrence':dict(complexity)}
    result = {'denominator':len(universe), 'targets_with_programs':len({r['target_id'] for r in programs}),
              'reference_composition_claim':False, 'chemical_novelty_claim':False,
              'reference_C0_C1_C2':'Unknown; generated programs do not label benchmark references'}
    for label in ('seen_ordinary', 'unseen_ordinary', 'nonordinary', 'unavailable'):
        result[label] = group([r for r in programs if r['exposure_class']==label])
    unseen = [r for r in programs if r['exposure_class']=='unseen_ordinary']
    result['unseen_combination_supported_primitives'] = group([
        r for r in unseen if r.get('all_primitives_seen_at_least_five') is True])
    supported_unseen = [r for r in unseen if source_supported(evidence.get(r['reaction_key'], {}))]
    result['UnseenProgramToSupportedReaction'] = {
        'unique_reactions':len({r['reaction_key'] for r in supported_unseen}),
        'targets':len({r['target_id'] for r in supported_unseen})}
    target_unseen = defaultdict(set)
    for row in unseen:
        target_unseen[row['target_id']].add(row['reaction_key'])
    covered = [r for r in routes if route_evidence(r, evidence)['precedent_covered']
               and set(r['reaction_keys']) & target_unseen[r['target_id']]]
    result['NovelProgramPrecedentCoveredRoute'] = {'targets':len({r['target_id'] for r in covered}),
                                                 'recorded_route_witnesses':len(covered)}
    result['NovelProgramSupportedDisconnection'] = len({r['reaction_key'] for r in supported_unseen})
    result['source_support_is_complete_reaction_not_internal_arrow'] = True
    return result
