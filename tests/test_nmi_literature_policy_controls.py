"""Independent source-anchored policy controls do not become model evidence."""
import importlib

import pytest


def fixture():
    variants=[{'id':v,'expected_retrieval':e} for v,e in [
        ('recorded_structures','exact'),('consistent_map_permutation','exact'),
        ('remote_benzyl_fluoro','local_edit_analogue_candidate'),
        ('reactive_environment_element_change','unresolved')]]
    source={'id':'source',
        'mapped_precursors':'[CH3:5][C:6](=[O:8])[Cl:9].[OH:1][N:2]([CH3:3])[c:4]1[cH:7][cH:10][cH:11][cH:12][cH:13]1',
        'mapped_product':'[CH3:5][C:6](=[O:8])[O:1][N:2]([CH3:3])[c:4]1[cH:7][cH:10][cH:11][cH:12][cH:13]1'}
    return {'radius':1,'primary_search_cap':5,'variants':variants,'sources':[source]}


def implementation():
    try:return importlib.import_module('mechet.analysis.literature_policy_controls')
    except ModuleNotFoundError:pytest.fail('source-anchored structural policy validator missing')


def test_structure_policy_retains_remote_analogue_but_rejects_changed_reactive_environment(tmp_path):
    result=implementation().evaluate_controls(fixture(),tmp_path/'index.sqlite')
    assert result['total']==4 and result['passed']==4 and result['failed']==0
    assert result['all_passed'] is True
    by_id={r['variant']:r for r in result['controls']}
    assert by_id['recorded_structures']['observed_retrieval']=='exact'
    assert by_id['consistent_map_permutation']['observed_retrieval']=='exact'
    assert by_id['remote_benzyl_fluoro']['observed_retrieval']=='local_edit_analogue_candidate'
    assert by_id['reactive_environment_element_change']['observed_retrieval']=='unresolved'
    assert by_id['remote_benzyl_fluoro']['query_reaction_key']!=by_id['recorded_structures']['query_reaction_key']
    assert all(r['machine_only_tier']=='C' for r in result['controls'])
    assert result['chemical_success_rate_established'] is False


def test_unmet_predeclared_control_is_reported_not_hidden_or_relabelled(tmp_path):
    spec=fixture();spec['variants'][2]['expected_retrieval']='exact'
    result=implementation().evaluate_controls(spec,tmp_path/'index.sqlite')
    assert result['all_passed'] is False and result['failed']==1
    assert result['total']==4
    wrong=[r for r in result['controls'] if not r['passed']]
    assert len(wrong)==1 and wrong[0]['variant']=='remote_benzyl_fluoro'


def test_policy_control_runner_refuses_to_overwrite_existing_index(tmp_path):
    path=tmp_path/'existing.sqlite';path.write_bytes(b'original')
    with pytest.raises(FileExistsError):implementation().evaluate_controls(fixture(),path)
    assert path.read_bytes()==b'original'


def test_versioned_full_stereo_reader_repairs_the_frozen_failed_control_without_changing_labels(tmp_path):
    import json
    from pathlib import Path
    p=Path(__file__).parents[1]/'configs/analysis/lgrv_independent_retrieval_controls_20261018.json'
    spec=json.loads(p.read_bytes())
    result=implementation().evaluate_controls(spec,tmp_path/'legacy.sqlite',family_policy='full_stereo_local_v2')
    assert result['total']==12 and result['passed']==12 and result['failed']==0
    assert result['all_passed'] is True
    assert result['family_policy']=='full_stereo_local_v2'
    assert all(r['machine_only_tier']=='C' for r in result['controls'])


@pytest.mark.parametrize('mutation',['none','source_structure','source_checksum','source_overlap'])
def test_policy_orchestration_binds_actual_source_structures_bytes_and_independence(tmp_path,mutation):
    import gzip,json
    from google.protobuf.struct_pb2 import Struct
    from mechet.analysis.literature_reaction_keys import make_reaction
    from mechet.analysis.literature_sources import file_sha256
    pa=pytest.importorskip('pyarrow');pq=pytest.importorskip('pyarrow.parquet')
    spec=fixture();spec.update(ord_relative_path='data/source.parquet',ord_dataset_id='software-fixture',
        primary_url='https://example.invalid/software-fixture')
    spec['sources'][0].update(ord_row=0,ord_reaction_id='fixture',primary_locator='software fixture only')
    sp=tmp_path/'spec.json';sp.write_text(json.dumps(spec))
    raw={'reaction_id':'fixture','inputs':{'a':{'components':[
        {'reaction_role':'REACTANT','identifiers':[{'type':'SMILES','value':'CC(=O)Cl'}]},
        {'reaction_role':'REACTANT','identifiers':[{'type':'SMILES','value':'ON(C)c1ccccc1'}]}]}},
        'outcomes':[{'products':[{'is_desired_product':True,'identifiers':[{'type':'SMILES',
            'value':'CC(=O)ON(C)c1ccccc1' if mutation!='source_structure' else 'CC(=O)N(C)c1ccccc1'}]}]}]}
    message=Struct();message.update(raw)
    root=tmp_path/'corpus';(root/'data').mkdir(parents=True)
    source=root/'data/source.parquet'
    schema=pa.schema([('reaction_id',pa.string()),('reaction',pa.binary())],metadata={b'ord.dataset_id':b'software-fixture'})
    pq.write_table(pa.Table.from_pylist([{'reaction_id':'fixture','reaction':message.SerializeToString()}],schema=schema),source)
    catalog=root/'source_corpus_manifest.json';catalog.write_text(json.dumps({'download_complete':True,
        'files':[{'path':'data/source.parquet','sha256':file_sha256(source)}]}))
    # A real protobuf message codec for the software fixture, not an ORD model
    # output or a bypass in production code.
    codec=tmp_path/'codec.py';codec.write_text('from google.protobuf.struct_pb2 import Struct as Reaction\n')
    freeze=tmp_path/'freeze';freeze.mkdir();table=freeze/'unique_reaction_queries.jsonl.gz'
    query=make_reaction('CC(=O)ON(C)c1ccccc1',['CC(=O)Cl','ON(C)c1ccccc1']) if mutation=='source_overlap' else make_reaction('CO',['CBr','[OH-]'])
    with gzip.open(table,'wt') as f:f.write(json.dumps(query)+'\n')
    (freeze/'prediction_freeze_manifest.json').write_text(json.dumps({'complete':True,
        'output_files':{table.name:{'sha256':file_sha256(table)}}}))
    if mutation=='source_checksum':
        with source.open('ab') as f:f.write(b'changed after catalogue freeze')
    out=tmp_path/'output'
    if mutation in ('source_structure','source_checksum'):
        with pytest.raises(ValueError):implementation().run_controls(sp,catalog,codec,file_sha256(codec),freeze,out)
        assert not out.exists()
    else:
        result,manifest=implementation().run_controls(sp,catalog,codec,file_sha256(codec),freeze,out)
        assert result['total']==4 and result['passed']==4
        assert result['all_passed'] is (mutation!='source_overlap')
        assert result['source_independence_passed'] is (mutation!='source_overlap')
        assert len(result['source_rows_reconciled'])==1
        assert manifest['all_controls_passed']==result['all_passed']
        assert manifest['output_files']['policy_control_results.json']['sha256']==file_sha256(out/'policy_control_results.json')
