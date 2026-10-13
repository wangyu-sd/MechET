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
