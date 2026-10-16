"""Reference matching tests: no inferred arrow ground truth or model evaluation."""
from copy import deepcopy


def edge():
    return {'source_product_smiles':'CO','source_precursor_smiles':['[CH3+]','[OH-]'],
            'mapping_valid':True,'annotation_status':'mapping_only',
            'program_signature':None,'primitive_signatures':None}


def record():
    return {'source_id':'recorded-rxn','source_split':'train',
            'target_smiles':'[CH3:1][OH:2]',
            'structural_precursor':'[CH3+:1].[OH-:2]',
            'executor_replayed':True,'program_signature':'recorded-program',
            'primitive_signatures':['recorded-primitive'],
            'signature_version':'source_to_sink_execution_moves_v1',
            'ordinary_source_sink_program':True,'electronic_program':{'steps':[{
                'state_before':'[CH3:1][OH:2]','state_after':'[CH3+:1].[OH-:2]',
                'moves':[{'source':{'kind':'BOND','atoms':[1,2]},
                          'sink':{'kind':'ATOM','atoms':[2]},'electrons':2}]}]}}


def test_match_requires_whole_precursor_multiset_not_only_same_product():
    from mechet.frozen_route_mechanisms import resolve_reference_mechanism
    wrong=record();wrong['structural_precursor']='C.O'
    result=resolve_reference_mechanism(edge(),[wrong])
    assert result['annotation_status']=='no_verified_match'
    assert result['program_signature'] is None


def test_map_renumbering_and_component_order_preserve_exact_chemistry_match():
    from mechet.frozen_route_mechanisms import resolve_reference_mechanism
    source=record();source['structural_precursor']='[OH-:99].[CH3+:42]'
    result=resolve_reference_mechanism(edge(),[source])
    assert result['annotation_status']=='verified_mechanism'
    assert result['program_signature']=='recorded-program'
    assert result['matched_source_ids']==['recorded-rxn']
    assert result['electronic_programs'][0]['steps'][0]['moves'][0]['source']['kind']=='BOND'


def test_unverified_record_does_not_turn_mapping_into_verified_arrows():
    from mechet.frozen_route_mechanisms import resolve_reference_mechanism
    source=record();source['executor_replayed']=False
    result=resolve_reference_mechanism(edge(),[source])
    assert result['annotation_status']=='unsupported_mechanism'
    assert result['program_signature'] is None


def test_multiple_recorded_programs_are_not_selected_using_training_seen_status():
    from mechet.frozen_route_mechanisms import resolve_reference_mechanism
    other=deepcopy(record());other.update(source_id='other',program_signature='other-program')
    result=resolve_reference_mechanism(edge(),[record(),other])
    assert result['annotation_status']=='unsupported_mechanism'
    assert result['mechanism_match_status']=='AMBIGUOUS_RECORDED_PROGRAMS'
    assert result['matched_source_ids']==['other','recorded-rxn']


def test_be_delta_is_not_mislabeled_as_an_ordinary_source_sink_program():
    from mechet.frozen_route_mechanisms import resolve_reference_mechanism
    source=record();source['ordinary_source_sink_program']=False
    result=resolve_reference_mechanism(edge(),[source])
    assert result['annotation_status']=='unsupported_mechanism'
    assert result['primitive_signatures'] is None


def test_missing_source_mapping_cannot_support_composition_classification():
    from mechet.frozen_route_mechanisms import resolve_reference_mechanism
    reference=edge();reference['mapping_valid']=False
    result=resolve_reference_mechanism(reference,[record()])
    assert result['annotation_status']=='no_verified_match'
    assert result['mechanism_match_status']=='REFERENCE_MAPPING_UNAVAILABLE'


def test_exact_match_does_not_neutralize_distinct_charge_or_stereo():
    from mechet.frozen_route_mechanisms import reaction_chemistry_key
    assert reaction_chemistry_key('CO',['[CH3+]','[OH-]']) != reaction_chemistry_key('CO',['C','O'])
    assert reaction_chemistry_key('F[C@H](Cl)Br',['C']) != reaction_chemistry_key('F[C@@H](Cl)Br',['C'])
    assert reaction_chemistry_key('CO',['O','O']) != reaction_chemistry_key('CO',['O'])


def decision(index, name, args, annotated='<A01>C<A02>O'):
    return {'id':f'r::{index}','source_id':'r','target_smiles':'CO',
        'expected_precursor':'[CH3+].[OH-]',
        'metadata':{'executor_replayed':True,'decision_index':index},
        'messages':[{'role':'user','content':f'ANNOTATED CURRENT STATE: {annotated}\n'},
                    {'role':'assistant','tool_calls':[{'function':{'name':name,'arguments':args}}]},
                    {'role':'tool','name':name,'content':'{"ok":true,"code":"PASS","current_state":"[CH3+].[OH-]"}'}]}


def flow_args(source='the bond between atoms A01 and A02',destination='atom A02'):
    return {'direction':'retrosynthetic','electron_flow':[{'source':source,'destination':destination,
        'instruction':'Return the bonding pair to oxygen.'}],'bond_order_changes':[],'charge_changes':[]}


def test_sft_event_signature_uses_actual_grounded_labels_not_old_compiler_metadata():
    from mechet.frozen_route_mechanisms import sft_event_features
    row=decision(0,'apply_electron_flow',flow_args())
    row['metadata']['execution_primitive_signatures']=['obsolete']
    result=sft_event_features(row)
    assert result['primitive_sequence']==[
        '{"electrons":2,"sink_atoms":[[8,0,0]],"sink_bond_order":0,"sink_kind":"ATOM",'
        '"source_atoms":[[6,0,0],[8,0,0]],"source_bond_order":1,"source_kind":"BOND"}']
    assert result['ordinary_source_sink_event'] is True


def test_sft_alias_numbers_are_local_and_signature_invariant():
    from mechet.frozen_route_mechanisms import sft_event_features
    original=decision(0,'apply_electron_flow',flow_args())
    shifted=decision(0,'apply_electron_flow',flow_args('the bond between atoms A08 and A99','atom A99'),
                     annotated='<A08>C<A99>O')
    assert sft_event_features(original)['primitive_sequence']==sft_event_features(shifted)['primitive_sequence']


def test_sft_be_delta_never_receives_the_empty_source_sink_pseudo_primitive():
    from mechet.frozen_route_mechanisms import sft_event_features
    row=decision(0,'apply_electron_flow',{'direction':'retrosynthetic','electron_flow':[],
        'bond_order_changes':[{'atoms':['A01','A02'],'delta':-1}],'charge_changes':[]})
    result=sft_event_features(row)
    assert result['ordinary_source_sink_event'] is False
    assert result['primitive_sequence']==[]


def test_reaction_summary_requires_contiguous_complete_actual_sft_decisions():
    import pytest
    from mechet.frozen_route_mechanisms import summarize_sft_reaction
    rows=[decision(0,'apply_electron_flow',flow_args()),decision(1,'finish_trace',{})]
    result=summarize_sft_reaction(rows,'train',structural_precursor='[CH3+].[OH-]')
    assert result['source_id']=='r'
    assert result['reference_event_length']==1
    assert result['coupled_electronic_moves']==1
    assert result['program_signature'] is not None
    with pytest.raises(ValueError,match='finish'):
        summarize_sft_reaction(rows[:1],'train',structural_precursor='[CH3+].[OH-]')
    with pytest.raises(ValueError,match='index'):
        summarize_sft_reaction([rows[1],rows[0]],'train',structural_precursor='[CH3+].[OH-]')


def test_missing_or_duplicate_annotated_atoms_cannot_invent_grounding():
    import pytest
    from mechet.frozen_route_mechanisms import sft_event_features
    for annotated in ['<A01>CO','<A01>C<A01>O']:
        with pytest.raises(ValueError,match='annotat'):
            sft_event_features(decision(0,'apply_electron_flow',flow_args(),annotated))


def test_annotation_merges_only_frozen_reference_edges_and_preserves_target_population():
    from mechet.frozen_route_mechanisms import annotate_reference_targets
    ref=edge();ref.update(reference_route_id='ref',reference_edge_id='ref:0')
    target={'id':'t','already_in_stock':False,'reference_reactions':[ref],
            'reference_routes':[{'reference_route_id':'ref'}]}
    source=record();source.update(reference_event_length=1,coupled_electronic_moves=1,imported_participants=0)
    result=annotate_reference_targets([target],{('CO',('[CH3+]','[OH-]')):[source]},
                                      {'recorded-program'},{'recorded-primitive':10})
    assert len(result)==1
    assert result[0]['id']=='t'
    assert result[0]['composition_stratum']=='C0'
    assert result[0]['reference_routes'][0]['reference_event_length']==1
    assert target['reference_reactions'][0]['annotation_status']=='mapping_only'


def test_actual_sft_import_participants_exclude_context_but_count_multiplicity():
    from mechet.frozen_route_mechanisms import summarize_sft_reaction
    imported=decision(0,'import_fragments',{'fragments':[
        {'smiles':'O','count':2,'purpose':'electron_participant'},
        {'smiles':'C','count':3,'purpose':'endpoint_context'}]})
    rows=[imported,decision(1,'apply_electron_flow',flow_args()),decision(2,'finish_trace',{})]
    assert summarize_sft_reaction(rows,'train',structural_precursor='[CH3+].[OH-]')['imported_participants']==2


def test_post_search_annotation_merge_preserves_verified_stock_flags_and_source_order():
    from mechet.frozen_route_mechanisms import merge_reference_annotations
    reference=edge();reference.update(reference_route_id='ref',reference_edge_id='ref:0')
    original={'id':'t','already_in_stock':False,'product_unmapped':'CO',
        'reference_reactions':[reference],
        'reference_routes':[{'reference_route_id':'ref','all_reference_leaves_in_stock':True,'depth':1}]}
    annotated=deepcopy(original);annotated['already_in_stock']=None
    annotated['reference_routes'][0].update(all_reference_leaves_in_stock=None,composition_stratum='C0')
    annotated['composition_stratum']='C0'
    result=merge_reference_annotations([original],[annotated])
    assert result[0]['already_in_stock'] is False
    assert result[0]['reference_routes'][0]['all_reference_leaves_in_stock'] is True
    assert result[0]['composition_stratum']=='C0'
    assert 'composition_stratum' not in original


def test_post_search_annotation_cannot_change_a_reference_reaction_or_denominator():
    import pytest
    from mechet.frozen_route_mechanisms import merge_reference_annotations
    reference=edge();reference.update(reference_route_id='ref',reference_edge_id='ref:0')
    original={'id':'t','product_unmapped':'CO','reference_reactions':[reference],
              'reference_routes':[{'reference_route_id':'ref','depth':1}]}
    changed=deepcopy(original);changed['reference_reactions'][0]['source_precursor_smiles']=['C','O']
    with pytest.raises(ValueError,match='reference'):
        merge_reference_annotations([original],[changed])
    with pytest.raises(ValueError,match='denominator'):
        merge_reference_annotations([original],[])
