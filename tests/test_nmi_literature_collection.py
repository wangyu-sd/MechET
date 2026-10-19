"""A repaired subset must never masquerade as the original complete corpus."""
import importlib
import json
from pathlib import Path

import pytest


def implementation():
    try:
        return importlib.import_module('scripts.assemble_nmi_literature_index')
    except ModuleNotFoundError:
        pytest.fail('full original-catalogue shard assembly is not implemented')


def collection(tmp_path):
    import gzip
    import pyarrow as pa
    import pyarrow.parquet as pq
    from mechet.analysis.literature_index import ReactionIndex
    from mechet.analysis.literature_sources import file_sha256
    from scripts.build_nmi_literature_index import implementation_identity
    source=tmp_path/'source';source.mkdir();donors=[];entries=[]
    for i in range(2):
        name=f'ord_dataset-{i}'
        raw=source/(name+'.parquet');pq.write_table(pa.table({'reaction':[b'x']}),raw)
        entries.append({'path':raw.name,'sha256':file_sha256(raw)})
        donor=tmp_path/f'donor{i}';donor.mkdir();donors.append(donor)
        index_path=donor/(name+'.sqlite');index=ReactionIndex(index_path)
        index.add({'source_id':str(i),'license':'fixture-only','product':'CO',
                   'precursors':['CBr','[OH-]']});index.close()
        failure=donor/(name+'.failures.jsonl.gz')
        with gzip.open(failure,'wt'):pass
        receipt={'complete':True,'source_path':str(raw),'source_sha256':file_sha256(raw),
            'codec_sha256':'a'*64,'source_rows':1,'indexed_structure_records':1,
            'index_path':str(index_path),'index_sha256':file_sha256(index_path),
            'failure_ledger':str(failure),'failure_sha256':file_sha256(failure),'failures':{},
            'implementation':implementation_identity()}
        (donor/(name+'.json')).write_text(json.dumps(receipt))
    manifest=source/'source_corpus_manifest.json';manifest.write_text(json.dumps({
        'download_complete':True,'declared_file_count':2,'files':entries,
        'license':'fixture-only','github_revision':'frozen-fixture'}))
    return manifest,donors


def test_assembly_joins_all_original_files_across_repair_roots_and_keeps_sources(tmp_path):
    from mechet.analysis.literature_index import ReactionIndex
    manifest,donors=collection(tmp_path)
    result=implementation().assemble(manifest,donors,tmp_path/'all',codec_sha='a'*64)
    assert result['complete'] is True
    assert result['source_reaction_rows']==2
    assert len(result['files'])==2
    index=ReactionIndex(tmp_path/'all/reactions.sqlite',readonly=True)
    assert index.count()==2
    index.close()
    assert all((root/f'ord_dataset-{i}.sqlite').is_file() for i,root in enumerate(donors))
    assert result['source_corpus']['declared_file_count']==2


@pytest.mark.parametrize('failure', ['missing','source_hash','row_count','ledger_hash','producer'])
def test_assembly_rejects_partial_or_corrupt_collection_before_creating_output(tmp_path,failure):
    manifest,donors=collection(tmp_path)
    receipt_path=donors[1]/'ord_dataset-1.json'
    receipt=json.loads(receipt_path.read_bytes())
    if failure=='missing':receipt['complete']=False
    if failure=='source_hash':receipt['source_sha256']='0'*64
    if failure=='row_count':receipt['source_rows']=99
    if failure=='ledger_hash':receipt['failure_sha256']='0'*64
    if failure=='producer':receipt['implementation']['producer_sha256']='0'*64
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        implementation().assemble(manifest,donors,tmp_path/'must-not-exist',codec_sha='a'*64)
    assert not (tmp_path/'must-not-exist').exists()


def test_legacy_adoption_requires_explicit_opt_in_and_preserves_receipt(tmp_path):
    manifest,donors=collection(tmp_path);p=donors[0]/'ord_dataset-0.json'
    receipt=json.loads(p.read_bytes());receipt.pop('implementation');p.write_text(json.dumps(receipt))
    before=p.read_bytes()
    with pytest.raises(ValueError):
        implementation().assemble(manifest,donors,tmp_path/'rejected',codec_sha='a'*64)
    result=implementation().assemble(manifest,donors,tmp_path/'all',codec_sha='a'*64,allow_legacy_v1=True)
    assert result['files'][0]['compatible_legacy_v1_reuse'] is True
    assert p.read_bytes()==before


def test_completed_assembly_rejects_changed_codec_instead_of_reusing_success(tmp_path):
    manifest,donors=collection(tmp_path);run=implementation().assemble
    run(manifest,donors,tmp_path/'all',codec_sha='a'*64)
    with pytest.raises(ValueError,match='codec'):
        run(manifest,donors,tmp_path/'all',codec_sha='b'*64)


def test_final_sqlite_rename_before_manifest_complete_can_resume(tmp_path):
    manifest,donors=collection(tmp_path);run=implementation().assemble
    old=run(manifest,donors,tmp_path/'all',codec_sha='a'*64)
    path=tmp_path/'all/source_corpus_manifest.json'
    # Precisely the interrupted finalization state: all shard transactions and
    # atomic final rename committed, but complete manifest save not committed.
    pending={k:v for k,v in old.items() if k not in {'index_path','record_count','index_sha256'}}
    pending['complete']=False;path.write_text(json.dumps(pending))
    result=run(manifest,donors,tmp_path/'all',codec_sha='a'*64)
    assert result['complete'] is True
    assert result['index_sha256']==old['index_sha256']
    assert result['record_count']==2


@pytest.mark.parametrize('alias',['direct','symlink','source_manifest'])
def test_output_cannot_overwrite_donor_or_original_source_catalogue(tmp_path,alias):
    from mechet.analysis.literature_sources import file_sha256
    manifest,donors=collection(tmp_path)
    if alias=='direct':output=donors[0]
    elif alias=='symlink':
        output=tmp_path/'alias';output.symlink_to(donors[0],target_is_directory=True)
    else:output=manifest.parent
    if alias!='source_manifest':
        protected=donors[0]/'source_corpus_manifest.json'
        protected.write_text(json.dumps({'source_manifest_sha256':file_sha256(manifest),
            'codec_sha256':'a'*64,'complete':False,'owner':'old live donor'}))
    else:protected=manifest
    before=protected.read_bytes()
    with pytest.raises(ValueError):
        implementation().assemble(manifest,donors,output,codec_sha='a'*64)
    assert protected.read_bytes()==before
    assert not (output/'reactions.sqlite').exists()


def test_hardlinked_building_database_cannot_modify_donor_before_failure(tmp_path):
    import os
    manifest,donors=collection(tmp_path)
    protected=donors[0]/'ord_dataset-0.sqlite';before=protected.read_bytes()
    out=tmp_path/'all';out.mkdir()
    os.link(protected,out/'reactions.building.sqlite')
    with pytest.raises(ValueError):
        implementation().assemble(manifest,donors,out,codec_sha='a'*64)
    assert protected.read_bytes()==before
