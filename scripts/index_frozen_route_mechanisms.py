#!/usr/bin/env python3
"""Full actual-SFT exposure census and exact reference correspondence, CPU only.

No actor generation, new arrow inference, reference-conditioned policy input,
subset experiment or mutation of the already frozen inference assets.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from mechet.frozen_route_assets import normalize_stock_smiles,file_sha256
from mechet.frozen_route_mechanisms import reaction_chemistry_key,summarize_sft_reaction,annotate_reference_targets
from scripts.prepare_frozen_route_benchmarks import write_json,write_jsonl


def _sft_init(lookup,split):
    global _LOOKUP,_SPLIT
    _LOOKUP,_SPLIT=lookup,split


def _sft_worker(rows):
    source=rows[0]['source_id'];parent=_LOOKUP.get(source)
    value=summarize_sft_reaction(rows,_SPLIT,structural_precursor=parent['structural_precursor'] if parent else None)
    if parent and (normalize_stock_smiles(value['target_smiles'])!=parent['product_unmapped']
                   or normalize_stock_smiles(value['full_precursor_state'])!=parent['full_precursor_state']):
        raise ValueError(f'parent versus actual SFT endpoint drift: {source}')
    if parent:value['parent_record_sha256']=parent.get('record_sha256')
    return {'source_id':source,'decisions':len(rows),'program_signature':value['program_signature'],
            'ordinary':value['ordinary_source_sink_program'],
            'primitives':value['observed_ordinary_primitive_signatures'],'matched_record':value if parent else None}


def _checked_lines(path,stats):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for raw in handle:
            digest.update(raw)
            if raw.strip():yield raw
    stats['sha256']=digest.hexdigest()


def _reaction_groups(path,stats):
    group=[];active=None
    for raw in _checked_lines(path,stats):
        row=json.loads(raw)
        if row.get('source_id')!=active:
            if group:yield group
            group=[];active=row.get('source_id')
            if not active:raise ValueError('missing actual SFT source ID')
        group.append(row)
    if group:yield group


def scan_sft_split(path,sha,*,expected_decisions,expected_source_ids,lookup,split,workers):
    stats={};seen=set();decisions=0;unsupported=0;programs=Counter();primitives=Counter();records=[]
    groups=_reaction_groups(path,stats)
    pool=mp.Pool(workers,initializer=_sft_init,initargs=(lookup,split)) if workers>1 else None
    if pool is None:_sft_init(lookup,split)
    try:
        results=pool.imap(_sft_worker,groups,chunksize=16) if pool else map(_sft_worker,groups)
        for result in results:
            source=result['source_id']
            if source in seen:raise ValueError(f'duplicate/noncontiguous actual SFT reaction: {source}')
            seen.add(source);decisions+=result['decisions'];unsupported+=not result['ordinary']
            if result['program_signature']:programs.update([result['program_signature']])
            primitives.update(set(result['primitives']))
            if result['matched_record']:records.append(result['matched_record'])
            if len(seen)%4096==0:
                print({'phase':'actual-SFT-exposure','split':split,'reactions':len(seen),
                       'decisions':decisions,'expected_reactions':len(expected_source_ids)},flush=True)
        if stats.get('sha256')!=sha:raise ValueError('actual SFT SHA mismatch')
        if seen!=expected_source_ids:raise ValueError('actual SFT reaction census mismatch')
        if decisions!=expected_decisions:raise ValueError('actual SFT decision census mismatch')
    finally:
        if pool:pool.terminate();pool.join()
    return {'source_path':str(path),'source_sha256':sha,'reaction_rows':len(seen),'decision_rows':decisions,
            'program_counts':dict(programs),'primitive_counts':dict(primitives),
            'non_ordinary_programs':unsupported,'matched_chemistry_records':len(records)},records


def _parent_init(wanted,roots):
    global _WANTED,_ROOTS
    _WANTED,_ROOTS=wanted,roots


def _parent_worker(raw):
    row=json.loads(raw);source=row.get('source_id') or row['id']
    product=normalize_stock_smiles(row['target_smiles'])
    record=None
    if product in _WANTED:
        structural=normalize_stock_smiles(row['structural_precursor'])
        if reaction_chemistry_key(product,[structural]) in _WANTED[product]:
            record={'structural_precursor':structural,'product_unmapped':product,
                    'full_precursor_state':normalize_stock_smiles(row['full_precursor_state']),
                    'record_sha256':hashlib.sha256(raw).hexdigest()}
    return source,record,product if product in _ROOTS else None


def scan_parent(path,sha,*,expected_rows,wanted,roots,workers):
    stats={};ids=set();lookup={};overlap=Counter()
    pool=mp.Pool(workers,initializer=_parent_init,initargs=(wanted,roots)) if workers>1 else None
    if pool is None:_parent_init(wanted,roots)
    try:
        lines=_checked_lines(path,stats)
        results=pool.imap(_parent_worker,lines,chunksize=32) if pool else map(_parent_worker,lines)
        for source,record,root in results:
            if source in ids:raise ValueError('duplicate parent reaction ID')
            ids.add(source)
            if record:lookup[source]=record
            if root:overlap.update([root])
            if len(ids)%16384==0:print({'phase':'frozen-parent-census','rows':len(ids),'total':expected_rows},flush=True)
        if stats.get('sha256')!=sha:raise ValueError('parent corpus SHA mismatch')
        if len(ids)!=expected_rows:raise ValueError('parent full reaction denominator mismatch')
    finally:
        if pool:pool.terminate();pool.join()
    return ids,lookup,dict(overlap)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--adapter',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args()
    from scripts.infer_dynamic_electron_trees import ADAPTER_SHA
    if file_sha256(args.adapter/'adapter_model.safetensors')!=ADAPTER_SHA:raise ValueError('wrong frozen Stage-II model')
    contract=json.loads((args.adapter/'data_contract.json').read_text())
    shared=Path('/aaa/fionafyang/buddy1/whaleywang/MechET')
    sft=shared/Path(contract['train_file']).parent
    history=json.loads((sft/'manifest.json').read_text())
    if history['splits']['train']['output_sha256']!=contract['train_file_sha256']:
        raise ValueError('actual Stage-II exposure manifest mismatch')
    state=Path(history['source_artifact']);state_manifest=json.loads((state/'manifest.json').read_text())
    if history['splits']['train']['source_sha256']!=state_manifest['splits']['train']['output_sha256']:
        raise ValueError('history/state source chain mismatch')
    initial_path=shared/contract['initial_adapter_path']
    initial_contract=json.loads((initial_path/'data_contract.json').read_text())
    if initial_contract['train_file_sha256']!=history['splits']['train']['source_sha256']:
        raise ValueError('Stage-I actual exposure does not match lossless history source')
    if file_sha256(initial_path/'adapter_model.safetensors')!=contract['initial_adapter_model_sha256']:
        raise ValueError('wrong Stage-I parent weights')
    parent=Path(state_manifest['source_artifact']);parent_manifest=parent/'training_manifest.json'
    if file_sha256(parent_manifest)!=state_manifest['source_manifest_sha256']:
        raise ValueError('frozen parent manifest mismatch')
    parent_info=json.loads(parent_manifest.read_text())
    assets=json.loads(args.assets.read_text());targets={};wanted=defaultdict(set);roots=set()
    for name,cohort in assets['cohorts'].items():
        path=Path(cohort['targets_path'])
        if file_sha256(path)!=cohort['targets_sha256']:raise ValueError('frozen route source changed')
        rows=[json.loads(line) for line in path.open()];targets[name]=rows
        roots.update(row['product_unmapped'] for row in rows)
        for row in rows:
            for edge in row.get('reference_reactions',[]):
                key=reaction_chemistry_key(edge['source_product_smiles'],edge['source_precursor_smiles'])
                wanted[key[0]].add(key)
    workers=min(args.workers,len(os.sched_getaffinity(0)))
    if workers<1:raise ValueError('positive CPU worker count required')
    scans={};records=[];overlap={}
    for split in ['train','valid','test']:
        info=parent_info['splits'][split]
        ids,lookup,overlap[split]=scan_parent(parent/f'{split}.jsonl',info['sha256'],
            expected_rows=info['rows'],wanted=dict(wanted),roots=roots,workers=workers)
        actual=history['splits'][split]
        scans[split],matched=scan_sft_split(sft/f'{split}.jsonl',actual['output_sha256'],
            expected_decisions=actual['decision_rows'],expected_source_ids=ids,lookup=lookup,split=split,workers=workers)
        records.extend(matched)
    by_key=defaultdict(list)
    for row in records:by_key[reaction_chemistry_key(row['target_smiles'],[row['structural_precursor']])].append(row)
    manifest={'schema_version':'actual_sft_frozen_route_annotation_v1','model_sha256':ADAPTER_SHA,
        'source_config_sha256':assets['config_sha256'],'source_parent_manifest_sha256':file_sha256(parent_manifest),
        'source_history_manifest_sha256':file_sha256(sft/'manifest.json'),
        'code_sha':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        'signature_version':'source_to_sink_execution_moves_v1',
        'signature_limitations':['omits_fragment_import_identities','coupled_move_order_sensitive',
                                'BE_DELTA_not_ordinary_source_sink_primitives'],
        'primitive_frequency_unit':'distinct_eligible_train_reactions','actor_training_exposure':'Stage-I + lossless-history Stage-II same labels',
        'actual_train_exposure':scans['train'],'corpus_census':scans,'target_product_overlap':overlap,
        'cohorts':{},'new_model_inference':False}
    for name,rows in targets.items():
        annotated=annotate_reference_targets(rows,by_key,set(scans['train']['program_counts']),scans['train']['primitive_counts'])
        destination=args.output/name/'targets.annotated.jsonl';write_jsonl(destination,annotated)
        manifest['cohorts'][name]={'rows':len(rows),'target_ids':[row['id'] for row in rows],
            'annotations_path':str(destination.resolve()),'annotations_sha256':file_sha256(destination),
            'reference_reactions_sha256':assets['cohorts'][name]['reference_reactions_sha256'],
            'strata':dict(Counter(row['composition_stratum'] for row in annotated)),
            'annotation_status_counts':dict(Counter(edge['annotation_status'] for row in annotated for edge in row['reference_reactions']))}
    write_json(args.output/'manifest.json',manifest)
    print({'phase':'actual-SFT-full-census-sealed','output':str(args.output),
           'strata':{key:value['strata'] for key,value in manifest['cohorts'].items()}},flush=True)


if __name__=='__main__':main()
