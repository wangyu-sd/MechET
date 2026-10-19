#!/usr/bin/env python3
"""Index every downloaded official ORD/Lowe reaction without model inference.

Independent source-file workers stream protobuf rows into recoverable SQLite
shards. No full-corpus Python object, GPU, model or test-label lookup is needed.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mechet.analysis.literature_sources import file_sha256,iter_ord_messages,ord_records
from mechet.analysis.literature_index import ReactionIndex

IMPLEMENTATION_ID='lgrv_index_v1_1'


def implementation_identity():
    from rdkit import rdBase
    from mechet.analysis.literature_reaction_keys import stable_hash
    paths=[Path(__file__).resolve(),ROOT/'src/mechet/analysis/literature_reaction_keys.py',
           ROOT/'src/mechet/analysis/literature_sources.py',ROOT/'src/mechet/analysis/literature_index.py']
    return {'id':IMPLEMENTATION_ID,'producer_sha256':stable_hash({str(p.relative_to(ROOT)):file_sha256(p) for p in paths}),
            'rdkit_version':rdBase.rdkitVersion,
            'normalization':'rdkit_map_free_isomeric_component_multiset_no_standardizer_v1'}


def save(path,value):
    temp = Path(str(path)+'.tmp'); temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def merge_shards(receipts, output_root):
    output_root=Path(output_root)
    building=output_root/'reactions.building.sqlite'
    final=output_root/'reactions.sqlite'
    if final.exists():
        raise FileExistsError('final index already exists; validate its completed manifest')
    combined=ReactionIndex(building)
    combined.connection.execute('CREATE TABLE IF NOT EXISTS merged_shards(path TEXT PRIMARY KEY,sha256 TEXT NOT NULL)')
    combined.commit()
    for receipt in receipts:
        if file_sha256(receipt['index_path'])!=receipt['index_sha256']:
            raise ValueError('source shard changed before merge')
        merged=combined.connection.execute('SELECT sha256 FROM merged_shards WHERE path=?',(receipt['index_path'],)).fetchone()
        if merged:
            if merged[0]!=receipt['index_sha256']:
                raise ValueError('resumed merge shard identity differs')
            continue
        combined.connection.execute('ATTACH DATABASE ? AS shard',(receipt['index_path'],))
        with combined.connection:
            combined.connection.execute('INSERT INTO reactions SELECT * FROM shard.reactions')
            combined.connection.execute('INSERT INTO merged_shards VALUES (?,?)',(receipt['index_path'],receipt['index_sha256']))
        combined.connection.execute('DETACH DATABASE shard')
    count=combined.count();combined.close()
    os.replace(building,final)
    return {'index_path':str(final.resolve()),'record_count':count,'index_sha256':file_sha256(final)}


def index_file(task):
    source_root,entry,codec_path,codec_sha,output,implementation,allow_v1 = task
    from google.protobuf.json_format import MessageToDict
    from rdkit import rdBase
    if file_sha256(codec_path) != codec_sha:
        raise ValueError('official protobuf codec checksum mismatch')
    spec = importlib.util.spec_from_file_location('lgrv_ord_reaction_codec',codec_path)
    codec = importlib.util.module_from_spec(spec); spec.loader.exec_module(codec)
    source = Path(source_root)/entry['path']
    if file_sha256(source) != entry['sha256']:
        raise ValueError('official source data checksum mismatch')
    name = source.stem
    output = Path(output)
    receipt_path = output/(name+'.json')
    database = output/(name+'.sqlite')
    failure_path = output/(name+'.failures.jsonl.gz')
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_bytes())
        if receipt['source_sha256']!=entry['sha256'] or receipt['codec_sha256']!=codec_sha or file_sha256(database)!=receipt['index_sha256']:
            raise ValueError('resumed source index identity differs')
        if file_sha256(receipt['failure_ledger'])!=receipt['failure_sha256']:
            raise ValueError('resumed conversion failure ledger checksum differs')
        if receipt.get('implementation')!=implementation:
            if not (allow_v1 and 'implementation' not in receipt):
                raise ValueError('source index implementation identity differs; use a new output')
            receipt={**receipt,'compatible_legacy_v1_reuse':True,
                     'legacy_reuse_basis':'same exact normalization/source/codec; reader dedup corrected; failed family serializations were not completed shards'}
        return receipt
    index = ReactionIndex(database)
    failures = Counter(); source_rows = 0
    def decode(payload):
        reaction = codec.Reaction(); reaction.ParseFromString(payload)
        return MessageToDict(reaction,preserving_proto_field_name=True)
    # Failed source rows remain in a typed failure ledger; they are not chemical
    # negatives and never change the prediction target denominator.
    with gzip.open(failure_path,'wt') as failure_handle, rdBase.BlockLogs():
        for row_number,dataset,raw in iter_ord_messages(source,decode):
            source_rows += 1
            try:
                records = ord_records(raw,dataset=dataset,row_index=row_number)
            except (ValueError,KeyError) as error:
                failures[type(error).__name__+':'+str(error)[:200]] += 1
                failure_handle.write(json.dumps({'parquet_row':row_number,'reaction_id':raw.get('reaction_id'),
                                                'error':str(error)[:500]})+'\n')
                continue
            for record in records:
                try:
                    index.add(record)
                except (ValueError,KeyError) as error:
                    failures[type(error).__name__+':'+str(error)[:200]] += 1
                    failure_handle.write(json.dumps({'parquet_row':row_number,'source_id':record['source_id'],
                                                    'error':str(error)[:500]})+'\n')
            if source_rows%1000==0:
                index.commit()
                save(output/(name+'.progress.json'),{'source_rows':source_rows,'indexed_records':index.count()})
    count = index.count(); index.close()
    receipt = {'source_path':str(source),'source_sha256':entry['sha256'],'codec_sha256':codec_sha,
               'source_rows':source_rows,'indexed_structure_records':count,'failures':dict(failures),
               'index_path':str(database),'index_sha256':file_sha256(database),
               'failure_ledger':str(failure_path),'failure_sha256':file_sha256(failure_path),
               'implementation':implementation,
               'complete':True,'performed_source_verification':False}
    save(receipt_path,receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest',type=Path,required=True)
    parser.add_argument('--codec-file',type=Path,required=True)
    parser.add_argument('--codec-sha256',required=True)
    parser.add_argument('--codec-source-revision',required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--allow-compatible-v1-shards',action='store_true',
                        help='Explicitly adopt checked pre-v1.1 successful shards; exact keys unchanged, provenance retained')
    args = parser.parse_args()
    source = json.loads(args.source_manifest.read_bytes())
    if source.get('download_complete') is not True:
        raise ValueError('official source download is incomplete')
    args.output_root.mkdir(parents=True,exist_ok=True)
    shards = args.output_root/'shards';shards.mkdir(exist_ok=True)
    manifest = {'schema':'lgrv_structured_source_index_v1','complete':False,'source_manifest':str(args.source_manifest.resolve()),
                'source_manifest_sha256':file_sha256(args.source_manifest),'source_corpus':source,
                'codec_sha256':args.codec_sha256,'codec_revision':args.codec_source_revision,
                'codec_source':'https://github.com/open-reaction-database/ord-schema',
                'codec_license':'Apache-2.0','data_license':'CC-BY-SA-4.0',
                'verification_scope':'structured retrieval candidates, not checked original performed patent/paper examples',
                'implementation_id':IMPLEMENTATION_ID,'implementation':implementation_identity(),
                'compatible_v1_adoption_requested':args.allow_compatible_v1_shards,'files':[],'failed_sources':[]}
    path = args.output_root/'source_corpus_manifest.json'
    if path.exists():
        old = json.loads(path.read_bytes())
        if old['source_manifest_sha256']!=manifest['source_manifest_sha256'] or old['codec_sha256']!=args.codec_sha256:
            raise ValueError('output already belongs to a different corpus or codec')
        if old.get('complete') is True:
            if old.get('implementation_id')!=IMPLEMENTATION_ID or old.get('implementation')!=manifest['implementation']:
                raise ValueError('completed index belongs to a different implementation; preserve and migrate explicitly')
            if file_sha256(args.output_root/'reactions.sqlite')!=old['index_sha256']:
                raise ValueError('completed index checksum differs')
            print(json.dumps({'phase':'index_already_complete'}),flush=True)
            return
    combined_path=args.output_root/'reactions.sqlite'
    if combined_path.exists():
        if not args.allow_compatible_v1_shards:
            raise ValueError('legacy partial combined index requires explicit adoption; no files changed')
        preserved=args.output_root/('reactions.legacy-partial-'+file_sha256(combined_path)+'.sqlite')
        if preserved.exists():
            raise FileExistsError('preserved partial index already exists')
        os.replace(combined_path,preserved)
        manifest['preserved_legacy_partial_index']=str(preserved)
    save(path,manifest)
    tasks = [(str(args.source_manifest.parent.resolve()),entry,str(args.codec_file.resolve()),args.codec_sha256,str(shards.resolve()),
              manifest['implementation'],args.allow_compatible_v1_shards)
             for entry in source['files']]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        submitted={pool.submit(index_file,task):task[1] for task in tasks}
        for future in as_completed(submitted):
            try:
                receipt=future.result()
            except Exception as error:
                failure={'source':submitted[future]['path'],'error':type(error).__name__+':'+str(error)[:2000]}
                manifest['failed_sources'].append(failure)
                save(path,manifest)
                print(json.dumps({'phase':'source_index_failed',**failure}),flush=True)
                continue
            manifest['files'].append(receipt);manifest['files'].sort(key=lambda item:item['source_path'])
            save(path,manifest)
            print(json.dumps({'phase':'index_ord','files_done':len(manifest['files']),'files_total':len(tasks),
                              'source_rows':sum(x['source_rows'] for x in manifest['files']),
                              'structure_records':sum(x['indexed_structure_records'] for x in manifest['files'])}),flush=True)
    if manifest['failed_sources']:
        raise RuntimeError('source files failed; successful shards retained, see failed_sources in manifest')
    merged=merge_shards(manifest['files'],args.output_root)
    manifest.update({'complete':True,**merged,'source_reaction_rows':sum(x['source_rows'] for x in manifest['files'])})
    save(path,manifest)
    print(json.dumps({'phase':'index_complete','record_count':manifest['record_count']}),flush=True)


if __name__ == '__main__':
    main()
