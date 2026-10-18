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


def save(path,value):
    temp = Path(str(path)+'.tmp'); temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def index_file(task):
    source_root,entry,codec_path,codec_sha,output = task
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
                'files':[]}
    path = args.output_root/'source_corpus_manifest.json'
    if path.exists():
        old = json.loads(path.read_bytes())
        if old['source_manifest_sha256']!=manifest['source_manifest_sha256'] or old['codec_sha256']!=args.codec_sha256:
            raise ValueError('output already belongs to a different corpus or codec')
    save(path,manifest)
    tasks = [(str(args.source_manifest.parent.resolve()),entry,str(args.codec_file.resolve()),args.codec_sha256,str(shards.resolve()))
             for entry in source['files']]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for receipt in as_completed([pool.submit(index_file,task) for task in tasks]):
            manifest['files'].append(receipt.result());manifest['files'].sort(key=lambda item:item['source_path'])
            save(path,manifest)
            print(json.dumps({'phase':'index_ord','files_done':len(manifest['files']),'files_total':len(tasks),
                              'source_rows':sum(x['source_rows'] for x in manifest['files']),
                              'structure_records':sum(x['indexed_structure_records'] for x in manifest['files'])}),flush=True)
    combined_path = args.output_root/'reactions.sqlite'
    if combined_path.exists():
        raise FileExistsError('combined index exists; preserve it and use its completed manifest')
    combined = ReactionIndex(combined_path)
    for receipt in manifest['files']:
        combined.connection.execute('ATTACH DATABASE ? AS shard',(receipt['index_path'],))
        combined.connection.execute('INSERT INTO reactions SELECT * FROM shard.reactions')
        combined.commit();combined.connection.execute('DETACH DATABASE shard')
    manifest.update({'complete':True,'index_path':str(combined_path.resolve()),'record_count':combined.count(),
                     'source_reaction_rows':sum(x['source_rows'] for x in manifest['files']),
                     'index_sha256':file_sha256(combined_path)})
    combined.close();save(path,manifest)
    print(json.dumps({'phase':'index_complete','record_count':manifest['record_count']}),flush=True)


if __name__ == '__main__':
    main()
