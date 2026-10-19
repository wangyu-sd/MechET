#!/usr/bin/env python3
"""Assemble the ORIGINAL complete ORD catalogue from immutable finished shards.

Completed repairs are read in place, never copied over live partial databases.
A 23-file repair manifest is not eligible to replace the original 93-file
catalogue. The explicit original catalogue is retained verbatim in the output.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from mechet.analysis.literature_sources import file_sha256
from scripts.build_nmi_literature_index import implementation_identity,merge_shards,save


def assemble(source_manifest,donor_roots,output_root,*,codec_sha,allow_legacy_v1=False,workers=8):
    import pyarrow.parquet as pq
    source_manifest,output_root=(Path(p).resolve() for p in (source_manifest,output_root))
    donor_roots=[Path(p).resolve() for p in donor_roots]
    protected_roots=[*donor_roots,source_manifest.parent]
    if any(output_root.is_relative_to(root) or root.is_relative_to(output_root) for root in protected_roots):
        raise ValueError('assembly output overlaps protected donor or original source directory')
    for name in ['source_corpus_manifest.json','source_corpus_manifest.json.tmp',
                 'reactions.sqlite','reactions.building.sqlite']:
        if any((output_root/name).resolve().is_relative_to(root) for root in protected_roots):
            raise ValueError('assembly output artifact aliases protected donor/source data')
    source=json.loads(source_manifest.read_bytes())
    if source.get('download_complete') is not True:
        raise ValueError('original source download is incomplete')
    if source.get('original_source_manifest') or source.get('full_collection_file_count'):
        raise ValueError('repair input is not the original complete source catalogue')
    entries=source['files'];names=[Path(e['path']).stem for e in entries]
    if len(set(names))!=len(names) or len(entries)!=source['declared_file_count']:
        raise ValueError('duplicate or missing original catalogue source files')
    if workers<1:
        raise ValueError('assembly workers must be positive')
    old=None
    if output_root.exists() and (output_root/'source_corpus_manifest.json').is_file():
        old=json.loads((output_root/'source_corpus_manifest.json').read_bytes())
        if old.get('source_manifest_sha256')!=file_sha256(source_manifest):
            raise ValueError('existing assembly belongs to another original catalogue')
        if old.get('codec_sha256')!=codec_sha:
            raise ValueError('existing assembly codec checksum differs')
        if old.get('complete') is True:
            if (old.get('implementation')!=implementation_identity() or
                file_sha256(output_root/'reactions.sqlite')!=old['index_sha256']):
                raise ValueError('completed assembly implementation or checksum differs')
            return old
    # Find every completed source first; incomplete donors must not create an
    # apparent full-corpus output, and no active partial SQLite is touched.
    selected=[];missing=[]
    for entry,name in zip(entries,names):
        candidates=[]
        for root in map(Path,donor_roots):
            for path in (root/(name+'.json'),root/'shards'/(name+'.json')):
                if path.is_file():
                    receipt=json.loads(path.read_bytes())
                    if receipt.get('complete') is True:
                        candidates.append((path,receipt))
        if not candidates:
            missing.append(name);continue
        if len({r['index_sha256'] for _,r in candidates})>1:
            raise ValueError('ambiguous completed donor indices for '+name)
        # Identical index bytes can have multiple provenance receipts; caller's
        # ordered donor preference is explicit and preserved in the manifest.
        selected.append((entry,*candidates[0]))
    if missing:
        raise ValueError('original catalogue still missing completed sources: '+','.join(missing))
    protected_files=[source_manifest]
    for _,receipt_path,receipt in selected:
        protected_files.extend([receipt_path,Path(receipt['source_path']),
                                Path(receipt['index_path']),Path(receipt['failure_ledger'])])
    for name in ['source_corpus_manifest.json','source_corpus_manifest.json.tmp',
                 'reactions.sqlite','reactions.building.sqlite']:
        planned=output_root/name
        if planned.exists() and any(planned.samefile(p) for p in protected_files):
            raise ValueError('assembly output aliases protected donor/source inode')
    implementation=implementation_identity()
    def validate(item):
        entry,receipt_path,receipt=item
        raw=source_manifest.parent/entry['path']
        if (receipt['source_sha256']!=entry['sha256'] or
            receipt['codec_sha256']!=codec_sha or
            Path(receipt['source_path']).resolve()!=raw.resolve() or
            file_sha256(raw)!=entry['sha256']):
            raise ValueError('source or codec checksum/path differs: '+str(receipt_path))
        if receipt['source_rows']!=pq.ParquetFile(raw).metadata.num_rows:
            raise ValueError('completed receipt did not scan every original source row')
        if (file_sha256(receipt['index_path'])!=receipt['index_sha256'] or
            file_sha256(receipt['failure_ledger'])!=receipt['failure_sha256']):
            raise ValueError('completed source index or failure ledger checksum differs')
        if receipt.get('implementation')!=implementation:
            if not (allow_legacy_v1 and 'implementation' not in receipt):
                raise ValueError('completed source producer differs')
            receipt={**receipt,'compatible_legacy_v1_reuse':True,
                'legacy_reuse_basis':'same original source/codec/exact normalization; failed partial sources are not adopted'}
        return {**receipt,'source_receipt_path':str(receipt_path.resolve()),
                'source_receipt_sha256':file_sha256(receipt_path)}
    receipts=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for receipt in pool.map(validate,selected):
            receipts.append(receipt)
            print(json.dumps({'phase':'assembly_source_verified','files_done':len(receipts),
                              'files_total':len(entries)}),flush=True)
    output_root.mkdir(parents=True,exist_ok=True)
    manifest={'schema':'lgrv_structured_source_index_v1','complete':False,
        'source_manifest':str(source_manifest.resolve()),'source_manifest_sha256':file_sha256(source_manifest),
        'source_corpus':source,'codec_sha256':codec_sha,'implementation':implementation,
        'implementation_id':implementation['id'],'codec_license':'Apache-2.0',
        'data_license':source['license'],'files':receipts,'failed_sources':[],
        'assembly_donor_roots':[str(Path(p).resolve()) for p in donor_roots],
        'compatible_v1_adoption_requested':allow_legacy_v1,
        'verification_scope':'structured retrieval candidates, not checked original performed patent/paper examples',
        'source_reaction_rows':sum(r['source_rows'] for r in receipts)}
    path=output_root/'source_corpus_manifest.json'
    final=output_root/'reactions.sqlite'
    if final.exists():
        expected={r['index_path']:r['index_sha256'] for r in receipts}
        if old is None or {r['index_path']:r['index_sha256'] for r in old.get('files',[])}!=expected:
            raise ValueError('unclaimed final database or changed source shards; preserve for inspection')
        connection=sqlite3.connect(final.as_uri()+'?mode=ro',uri=True)
        try:
            healthy=connection.execute('PRAGMA integrity_check').fetchall()==[('ok',)]
            ledger=dict(connection.execute('SELECT path,sha256 FROM merged_shards'))
            count=connection.execute('SELECT count(*) FROM reactions').fetchone()[0]
        finally:
            connection.close()
        if (not healthy or ledger!=expected or
            count!=sum(r['indexed_structure_records'] for r in receipts)):
            raise ValueError('interrupted final database inventory differs from completed original shards')
        merged={'index_path':str(final),'index_sha256':file_sha256(final),'record_count':count}
        manifest['finalization_recovered_from_validated_merge']=True
    else:
        save(path,manifest)
        merged=merge_shards(receipts,output_root)
    manifest.update(merged,complete=True);save(path,manifest)
    print(json.dumps({'phase':'full_original_catalogue_assembled','files':len(receipts),
                      'source_rows':manifest['source_reaction_rows'],'structure_records':merged['record_count']}),flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest',type=Path,required=True)
    parser.add_argument('--donor-root',type=Path,action='append',required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--codec-sha256',required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--allow-compatible-v1-shards',action='store_true')
    args=parser.parse_args()
    assemble(args.source_manifest,args.donor_root,args.output_root,codec_sha=args.codec_sha256,
             workers=args.workers,allow_legacy_v1=args.allow_compatible_v1_shards)


if __name__=='__main__':
    main()
