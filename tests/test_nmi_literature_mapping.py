"""Native mappings enrich retrieval, never change the frozen chemistry pool."""
import copy
import gzip
import importlib
import json

import pytest

from mechet.analysis.literature_reaction_keys import make_reaction
from mechet.analysis.literature_sources import file_sha256


def fixture(tmp_path):
    freeze=tmp_path/'freeze';freeze.mkdir()
    programs=tmp_path/'programs';programs.mkdir()
    q=make_reaction('CCCO',['CCC','O'])
    mapped_product='[CH3:1][CH2:2][CH2:3][OH:4]'
    mapped_precursors='[CH3:1][CH2:2][CH3:3].[OH2:4]'
    native=make_reaction(q['product_raw'],q['precursors_raw'],
        mapped_product=mapped_product,mapped_precursors=mapped_precursors)
    p={'program_record_id':'p1','actor_sha256':'a'*64,
       'actor_exposure_manifest_sha256':'e'*64,'reaction_key':q['reaction_key'],
       'mapped_product':mapped_product,'mapped_structural_precursors':mapped_precursors,
       'program_state_binding':{'valid':True},
       'mapping_status':native['mapping_status'],'reaction_family_key':native['reaction_family_key']}
    def table(path,rows):
        with gzip.open(path,'wt') as f:
            for r in rows:f.write(json.dumps(r)+'\n')
    table(freeze/'unique_reaction_queries.jsonl.gz',[q])
    fm={'complete':True,'actor_sha256':'a'*64,'output_files':{
        'unique_reaction_queries.jsonl.gz':{'sha256':file_sha256(freeze/'unique_reaction_queries.jsonl.gz')}}}
    (freeze/'prediction_freeze_manifest.json').write_text(json.dumps(fm))
    def save_programs(rows):
        table(programs/'program_reaction_links.jsonl.gz',rows)
        pm={'complete':True,'actor_sha256':'a'*64,'actor_exposure_manifest_sha256':'e'*64,
            'prediction_freeze_manifest_sha256':file_sha256(freeze/'prediction_freeze_manifest.json'),
            'unmatched_occurrence_count':0,'output_files':{'program_reaction_links.jsonl.gz':{
                'sha256':file_sha256(programs/'program_reaction_links.jsonl.gz')}}}
        (programs/'program_join_manifest.json').write_text(json.dumps(pm))
    save_programs([p])
    return freeze,programs,q,p,save_programs


def implementation():
    try:return importlib.import_module('mechet.analysis.literature_mapping')
    except ModuleNotFoundError:pytest.fail('authenticated native mapping enrichment missing')


def read_table(root):
    with gzip.open(root/'unique_reaction_queries_enriched.jsonl.gz','rt') as f:
        return [json.loads(l) for l in f]


def test_mapping_enrichment_keeps_every_frozen_identity_and_raw_smiles(tmp_path):
    freeze,programs,q,p,_=fixture(tmp_path)
    before=(freeze/'unique_reaction_queries.jsonl.gz').read_bytes()
    out=tmp_path/'derived'
    manifest=implementation().enrich_queries(freeze,programs,out)
    rows=read_table(out)
    assert len(rows)==1 and rows[0]['reaction_key']==q['reaction_key']
    assert rows[0]['product_raw']=='CCCO' and rows[0]['precursors_raw']==['CCC','O']
    assert rows[0]['mapping_status']=='mapped_center_available'
    assert rows[0]['changed_bond_count']==1
    assert rows[0]['native_mapping']['status']=='unique_native_family'
    assert rows[0]['native_mapping']['program_record_ids']==['p1']
    assert manifest['counts']['newly_resolved']==1
    assert manifest['complete'] is True and manifest['new_model_inference'] is False
    assert (freeze/'unique_reaction_queries.jsonl.gz').read_bytes()==before
    assert manifest['output_files']['unique_reaction_queries_enriched.jsonl.gz']['sha256']==file_sha256(out/'unique_reaction_queries_enriched.jsonl.gz')
    with pytest.raises(FileExistsError):implementation().enrich_queries(freeze,programs,out)


def test_conflicting_native_families_remain_unknown_without_selecting_a_favourite(tmp_path):
    freeze,programs,q,p,save=fixture(tmp_path)
    other=copy.deepcopy(p);other['program_record_id']='p2'
    other['mapped_structural_precursors']='[CH2:1]([CH3:2])[CH3:3].[OH2:4]'
    from mechet.analysis.literature_reaction_keys import mapped_center
    other['reaction_family_key']=mapped_center(other['mapped_structural_precursors'],other['mapped_product'])['reaction_family_key']
    assert other['reaction_family_key']!=p['reaction_family_key']
    save([p,other])
    out=tmp_path/'derived';m=implementation().enrich_queries(freeze,programs,out)
    row=read_table(out)[0]
    assert row['reaction_key']==q['reaction_key']
    assert row['mapping_status']=='mapping_unknown' and row['reaction_family_key'] is None
    assert row['native_mapping']['status']=='conflicting_native_families'
    assert len(row['native_mapping']['family_keys'])==2
    assert m['counts']['conflicting']==1


@pytest.mark.parametrize('mutation',['actor','exposure','binding','chemical_identity','family_payload','duplicate_program','unfrozen_bytes','unbound_program_table','unbound_query_table'])
def test_mapping_enrichment_rejects_unbound_native_metadata_before_creating_output(tmp_path,mutation):
    freeze,programs,_,p,save=fixture(tmp_path)
    if mutation=='actor':p['actor_sha256']='b'*64
    if mutation=='exposure':p['actor_exposure_manifest_sha256']='f'*64
    if mutation=='binding':p['program_state_binding']['valid']=False
    if mutation=='chemical_identity':p['mapped_product']='[CH3:1][CH2:2][CH2:3][NH2:4]'
    if mutation=='family_payload':p['reaction_family_key']='f'*64
    save([p,p] if mutation=='duplicate_program' else [p])
    if mutation=='unfrozen_bytes':
        with (programs/'program_reaction_links.jsonl.gz').open('ab') as f:f.write(b'edited')
    if mutation in ('unbound_program_table','unbound_query_table'):
        path=(programs/'program_join_manifest.json' if mutation=='unbound_program_table'
              else freeze/'prediction_freeze_manifest.json')
        data=json.loads(path.read_bytes());data['output_files']={}
        path.write_text(json.dumps(data))
        if mutation=='unbound_query_table':
            # Rebind the parent manifest so the only invalid boundary is the
            # query table's missing original checksum, not the parent hash.
            pm=programs/'program_join_manifest.json';data=json.loads(pm.read_bytes())
            data['prediction_freeze_manifest_sha256']=file_sha256(path)
            pm.write_text(json.dumps(data))
    out=tmp_path/'derived'
    with pytest.raises(ValueError):implementation().enrich_queries(freeze,programs,out)
    assert not out.exists()


def test_missing_native_program_preserves_unresolved_query_and_denominator(tmp_path):
    freeze,programs,q,_,save=fixture(tmp_path);save([])
    out=tmp_path/'derived';m=implementation().enrich_queries(freeze,programs,out)
    rows=read_table(out)
    assert len(rows)==1 and rows[0]['reaction_key']==q['reaction_key']
    assert rows[0]['mapping_status']=='mapping_unknown'
    assert rows[0]['native_mapping']['status']=='native_mapping_unavailable'
    assert m['counts']['query_count']==1 and m['counts']['unavailable']==1


def test_archived_full_mixture_auxiliary_projection_is_preserved_without_inventing_raw_imports(tmp_path):
    freeze,programs,q,p,save=fixture(tmp_path)
    # Actual single-reaction serializer stores a canonical full-mixture
    # remainder, not a raw import log; auxiliary_raw can legitimately be empty.
    q['auxiliary_components']=['[Cl-]']
    q['auxiliary_role_basis']='recorded_structural_projection_full_mixture_difference'
    table=freeze/'unique_reaction_queries.jsonl.gz'
    with gzip.open(table,'wt') as f:f.write(json.dumps(q)+'\n')
    fm=freeze/'prediction_freeze_manifest.json';m=json.loads(fm.read_bytes())
    m['output_files'][table.name]['sha256']=file_sha256(table);fm.write_text(json.dumps(m))
    save([p])
    out=tmp_path/'derived';implementation().enrich_queries(freeze,programs,out)
    row=read_table(out)[0]
    assert row['auxiliary_components']==['[Cl-]']
    assert row['auxiliary_raw']==[]
    assert row['auxiliary_role_basis']==q['auxiliary_role_basis']
    assert row['mapping_status']=='mapped_center_available'


def test_query_view_consumer_revalidates_saved_programs_not_just_declared_counts(tmp_path):
    freeze,programs,q,_,_=fixture(tmp_path)
    out=tmp_path/'derived';module=implementation();module.enrich_queries(freeze,programs,out)
    queries,manifest=module.load_query_view(freeze,programs,out)
    assert len(queries)==1 and queries[0]['reaction_key']==q['reaction_key']
    assert queries[0]['native_mapping']['status']=='unique_native_family'
    assert manifest['counts']['newly_resolved']==1


@pytest.mark.parametrize('mutation',['family','raw_precursors','auxiliary_projection','duplicate_query','claimed_count','parent_manifest'])
def test_query_view_consumer_refuses_self_consistent_but_invented_mapping(tmp_path,mutation):
    freeze,programs,_,_,_=fixture(tmp_path)
    out=tmp_path/'derived';module=implementation();module.enrich_queries(freeze,programs,out)
    table=out/'unique_reaction_queries_enriched.jsonl.gz';rows=read_table(out)
    if mutation=='family':rows[0]['reaction_family_key']='f'*64
    if mutation=='raw_precursors':rows[0]['precursors_raw']=['CCC','N']
    if mutation=='auxiliary_projection':rows[0]['auxiliary_components']=['[Na+]']
    if mutation=='duplicate_query':rows.append(copy.deepcopy(rows[0]))
    with gzip.open(table,'wt') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    path=out/'mapping_enrichment_manifest.json';m=json.loads(path.read_bytes())
    m['output_files'][table.name]['sha256']=file_sha256(table)
    if mutation=='claimed_count':m['counts']['newly_resolved']=99
    if mutation=='parent_manifest':m['program_join_manifest_sha256']='f'*64
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError):module.load_query_view(freeze,programs,out)
