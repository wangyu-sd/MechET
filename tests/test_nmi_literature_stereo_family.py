"""Legacy-index compatibility must repair traversal parity, not erase stereo."""
import importlib
import json
from pathlib import Path

import pytest

from mechet.analysis.literature_index import ReactionIndex
from mechet.analysis.literature_policy_controls import variant_smiles
from mechet.analysis.literature_reaction_keys import make_reaction


def implementation():
    try:return importlib.import_module('mechet.analysis.literature_stereo_family')
    except ModuleNotFoundError:pytest.fail('full-stereo preserving local-key compatibility missing')


def source():
    path=Path(__file__).parents[1]/'configs/analysis/lgrv_independent_retrieval_controls_20261018.json'
    return json.loads(path.read_bytes())['sources'][2]


def remote_query(s):
    p=variant_smiles(s['mapped_product'],'remote_benzyl_fluoro')
    r=variant_smiles(s['mapped_precursors'],'remote_benzyl_fluoro')
    return make_reaction(p,[r])


def add(index,name,s):
    index.add({'source_id':name,'product':s['mapped_product'],
               'precursors':[s['mapped_precursors']],'license':'CC-BY-SA-4.0'})


def test_actual_source_c_remote_substituent_keeps_local_stereo_and_finds_legacy_index(tmp_path):
    m=implementation();s=source();q=remote_query(s)
    original=m.family_context(s['mapped_precursors'],s['mapped_product'])
    context=m.family_context(q['precursors_raw'][0],q['product_raw'])
    assert context['family_key_v2']==original['family_key_v2']
    assert '0e84eb6addbe6a4ea8b137c1226e4fdc623c2318f191678b4f2355236e56a1ba' in context['legacy_family_keys']
    index=ReactionIndex(tmp_path/'legacy.sqlite');add(index,'actual-C',s);index.commit()
    hits,audit=m.query_sources(index,q,context,cap=5)
    assert [h['source_id'] for h in hits]==['actual-C']
    assert hits[0]['match_type']=='local_edit_analogue_candidate'
    assert audit['context_status']=='available' and audit['legacy_alias_space_complete'] is True
    index.close()


def test_invalid_local_parity_aliases_never_make_opposite_full_enantiomer_compatible(tmp_path):
    m=implementation();s=source();q=remote_query(s)
    wrong={**s,'mapped_product':s['mapped_product'].replace('[C@@H:25]','[C@H:25]')}
    context=m.family_context(q['precursors_raw'][0],q['product_raw'])
    opposite=m.family_context(wrong['mapped_precursors'],wrong['mapped_product'])
    assert context['family_key_v2']!=opposite['family_key_v2']
    index=ReactionIndex(tmp_path/'legacy.sqlite');add(index,'opposite-enantiomer',wrong);index.commit()
    hits,audit=m.query_sources(index,q,context,cap=5)
    assert hits==[]
    assert audit['incompatible_family_candidates']>=1
    index.close()


def test_wrong_enantiomer_does_not_consume_checked_distinct_source_cap(tmp_path):
    m=implementation();s=source();q=remote_query(s)
    wrong={**s,'mapped_product':s['mapped_product'].replace('[C@@H:25]','[C@H:25]')}
    index=ReactionIndex(tmp_path/'legacy.sqlite');add(index,'0-wrong',wrong);add(index,'1-right',s);index.commit()
    hits,audit=m.query_sources(index,q,m.family_context(q['precursors_raw'][0],q['product_raw']),cap=1)
    assert [h['source_id'] for h in hits]==['1-right']
    assert audit['incompatible_family_candidates']==1
    index.close()


def test_full_stereo_local_family_is_invariant_to_consistent_map_renumbering():
    m=implementation();s=source()
    a=m.family_context(s['mapped_precursors'],s['mapped_product'])
    b=m.family_context(variant_smiles(s['mapped_precursors'],'consistent_map_permutation'),
                       variant_smiles(s['mapped_product'],'consistent_map_permutation'))
    assert a['family_key_v2']==b['family_key_v2']
    assert a['legacy_family_keys']==b['legacy_family_keys']


def test_bounded_legacy_alias_space_is_explicitly_unknown_not_silently_complete(tmp_path):
    m=implementation();s=source();q=remote_query(s)
    context=m.family_context(q['precursors_raw'][0],q['product_raw'],alias_limit=1)
    assert context['status']=='alias_space_unresolved'
    assert context['legacy_alias_space_complete'] is False
    index=ReactionIndex(tmp_path/'legacy.sqlite');add(index,'actual-C',s);index.commit()
    hits,audit=m.query_sources(index,q,context,cap=5)
    assert hits==[] and audit['context_status']=='alias_space_unresolved'
    index.close()


def test_unmapped_context_keeps_exact_candidates_and_never_invents_analogue(tmp_path):
    m=implementation();q=make_reaction('CO',['CBr','[OH-]'])
    context=m.family_context('CBr.[OH-]','CO')
    assert context['status']=='mapping_unknown'
    index=ReactionIndex(tmp_path/'index.sqlite')
    index.add({'source_id':'exact','product':'CO','precursors':['CBr','[OH-]'],'license':'CC-BY-SA-4.0'})
    index.commit();hits,_=m.query_sources(index,q,context,cap=5)
    assert [h['source_id'] for h in hits]==['exact'] and hits[0]['match_type']=='exact'
    wrong=m.family_context('[CH3:1][NH2:2]','[CH3:1][NH2:2]')
    with pytest.raises(ValueError):m.query_sources(index,q,wrong,cap=5)
    index.close()


def test_absent_center_is_not_a_universal_identity_reaction_family():
    m=implementation();s=source()
    context=m.family_context(s['mapped_product'],s['mapped_product'])
    assert context['status']=='no_reaction_center'
    assert context['legacy_family_keys']==()


def test_complete_double_bond_geometry_is_not_erased_by_local_compatibility():
    m=implementation()
    precursor='[CH3:1][CH2:2][CH2:3][CH3:4]'
    e=m.family_context(precursor,'[CH3:1]/[CH:2]=[CH:3]/[CH3:4]')
    z=m.family_context(precursor,'[CH3:1]/[CH:2]=[CH:3]\\[CH3:4]')
    assert e['family_key_v2']!=z['family_key_v2']


def test_aromatic_boundary_is_a_local_graph_not_a_complete_aromatic_molecule(tmp_path):
    path=Path(__file__).parents[1]/'configs/analysis/lgrv_independent_retrieval_controls_20261018.json'
    s=json.loads(path.read_bytes())['sources'][1];m=implementation();q=remote_query(s)
    context=m.family_context(q['precursors_raw'][0],q['product_raw'])
    assert context['status']=='available'
    index=ReactionIndex(tmp_path/'legacy.sqlite');add(index,'actual-B',s);index.commit()
    hits,_=m.query_sources(index,q,context,cap=5)
    assert [h['source_id'] for h in hits]==['actual-B']
    index.close()


@pytest.mark.parametrize('case',['same_stereo','opposite_stereo','no_manual_analogue_review'])
def test_checked_analogue_grading_uses_full_stereo_context_not_unstable_legacy_hash(tmp_path,case):
    from test_nmi_literature_protocol import primary
    from mechet.analysis.literature_evidence import assess
    m=implementation();s=source();q=remote_query(s)
    q['stereo_family_context']=m.family_context(q['precursors_raw'][0],q['product_raw'])
    product=s['mapped_product'] if case!='opposite_stereo' else s['mapped_product'].replace('[C@@H:25]','[C@H:25]')
    checked=primary(tmp_path,source_product=product,source_precursors=[s['mapped_precursors']],
        analogue_query_reaction_key=q['reaction_key'],
        analogue_compatibility_checked=case!='no_manual_analogue_review',
        substrate_differences='hypothetical remote aryl F in this SOFTWARE fixture',
        functional_group_compatibility='unchanged N-O/alkene reacting environment',stereo_compatibility_checked=True)
    result=assess(q,[],[checked])
    assert result['tier']==('B' if case=='same_stereo' else 'C')
    assert result['physical_mechanism_verified'] is False


@pytest.mark.parametrize('binding',[None,'another-query'])
def test_analogue_notes_without_this_query_identity_cannot_support_a_family(tmp_path,binding):
    from test_nmi_literature_protocol import primary
    from mechet.analysis.literature_evidence import assess
    m=implementation();s=source();q=remote_query(s)
    q['stereo_family_context']=m.family_context(q['precursors_raw'][0],q['product_raw'])
    checked=primary(tmp_path,source_product=s['mapped_product'],source_precursors=[s['mapped_precursors']],
        analogue_query_reaction_key=binding,analogue_compatibility_checked=True,
        substrate_differences='hypothetical remote aryl F; SOFTWARE fixture only',
        functional_group_compatibility='unchanged reacting environment',stereo_compatibility_checked=True)
    result=assess(q,[],[checked])
    assert result['tier']=='C'
    assert result['supported'] is False


def test_review_of_one_remote_substituent_does_not_promote_another_query(tmp_path):
    from test_nmi_literature_protocol import primary
    from mechet.analysis.literature_evidence import assess
    m=implementation();s=source();reviewed=remote_query(s)
    unreviewed=make_reaction(reviewed['product_raw'].replace('[F:10000]','[Cl:10000]'),
        [reviewed['precursors_raw'][0].replace('[F:10000]','[Cl:10000]')])
    for q in (reviewed,unreviewed):
        q['stereo_family_context']=m.family_context(q['precursors_raw'][0],q['product_raw'])
    assert reviewed['reaction_key']!=unreviewed['reaction_key']
    assert reviewed['stereo_family_context']['family_key_v2']==unreviewed['stereo_family_context']['family_key_v2']
    checked=primary(tmp_path,source_product=s['mapped_product'],source_precursors=[s['mapped_precursors']],
        analogue_query_reaction_key=reviewed['reaction_key'],analogue_compatibility_checked=True,
        substrate_differences='hypothetical remote aryl F; SOFTWARE fixture only',
        functional_group_compatibility='unchanged reacting environment',stereo_compatibility_checked=True)
    assert assess(reviewed,[],[checked])['tier']=='B'
    assert assess(unreviewed,[],[checked])['tier']=='C'
