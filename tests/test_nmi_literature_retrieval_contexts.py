"""Corrected retrieval contexts preserve every frozen query and native witness."""
import importlib

import pytest

from mechet.analysis.literature_reaction_keys import make_reaction


def implementation():
    try:return importlib.import_module('mechet.analysis.literature_retrieval_contexts')
    except ModuleNotFoundError:pytest.fail('authenticated stereo retrieval contexts missing')


def fixture():
    q=make_reaction('CCCO',['CCC','O'])
    p={'program_record_id':'p','reaction_key':q['reaction_key'],'actor_sha256':'a'*64,
        'program_state_binding':{'valid':True},
        'mapped_product':'[CH3:1][CH2:2][CH2:3][OH:4]',
        'mapped_structural_precursors':'[CH3:1][CH2:2][CH3:3].[OH2:4]'}
    return q,p


@pytest.mark.parametrize('workers',[1,2])
def test_contexts_cover_the_entire_query_pool_and_bind_all_programs(tmp_path,workers):
    q,p=fixture();missing=make_reaction('CO',['CBr','[OH-]'])
    rows=implementation().prepare_contexts([q,missing],[p,{**p,'program_record_id':'second'}],
        actor_sha256='a'*64,workers=workers)
    assert set(rows)=={q['reaction_key'],missing['reaction_key']}
    context=rows[q['reaction_key']]
    assert context['status']=='available' and context['family_key_v2']
    assert context['program_record_ids']==['p','second']
    assert context['representative_program_record_id']=='p'
    assert rows[missing['reaction_key']]['status']=='native_mapping_unavailable'


def test_non_equivalent_native_correspondences_remain_unknown():
    q,p=fixture()
    other={**p,'program_record_id':'second',
        'mapped_structural_precursors':'[CH2:1]([CH3:2])[CH3:3].[OH2:4]'}
    rows=implementation().prepare_contexts([q],[p,other],actor_sha256='a'*64)
    assert rows[q['reaction_key']]['status']=='conflicting_native_families'
    assert rows[q['reaction_key']]['family_key_v2'] is None


@pytest.mark.parametrize('mutation',['actor','chemical_identity','binding','duplicate_program'])
def test_contexts_refuse_unbound_program_data(mutation):
    q,p=fixture()
    if mutation=='actor':p['actor_sha256']='f'*64
    if mutation=='chemical_identity':p['mapped_product']='[CH3:1][CH2:2][CH2:3][NH2:4]'
    if mutation=='binding':p['program_state_binding']['valid']=False
    programs=[p,p] if mutation=='duplicate_program' else [p]
    with pytest.raises(ValueError):implementation().prepare_contexts([q],programs,actor_sha256='a'*64)
