#!/usr/bin/env python3
"""Freeze the full source-defined R1 cohorts/stocks before model inference.

Source originals and generated structures stay outside Git. Stock preparation
uses the remote job's CPU pool; --references-only performs no stock chemistry
processing and leaves stock-closure labels explicitly pending.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import itertools
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from mechet.frozen_route_assets import file_sha256, verify_source, normalize_stock_smiles, extract_reference_route


def open_text(path):
    return gzip.open(path,'rt') if str(path).endswith('.gz') else Path(path).open()


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    with temporary.open('w') as handle:
        for chunk in json.JSONEncoder(ensure_ascii=False,indent=2).iterencode(value):handle.write(chunk)
        handle.write('\n')
    os.replace(temporary,path)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    with temporary.open('w') as handle:
        for row in rows:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    os.replace(temporary,path)


def normalize_stock_row(item):
    index,smiles = item
    try:return index,normalize_stock_smiles(smiles),None
    except Exception as exc:return index,None,{'row':index,'smiles':smiles,'error':str(exc)}


def freeze_assets(config_path, output, *, workers, normalize_stocks):
    import rdkit
    config_path,output = Path(config_path),Path(output)
    config = json.loads(config_path.read_text())
    raw_root = Path(config['raw_root'])
    config_sha = file_sha256(config_path)
    existing=output/'manifest.json'
    if existing.exists() and json.loads(existing.read_text())['config_sha256'] != config_sha:
        raise ValueError('existing cohort frozen with different source config; use new output')
    # Verify EVERY original before writing any derived target or stock.
    for item in itertools.chain(config['stocks'].values(),config['cohorts'].values()):
        verify_source(raw_root/item['path'],item['sha256'])
    manifest={'schema_version':'mechet_frozen_route_assets_v1','config_sha256':config_sha,
        'rdkit_version':rdkit.__version__,'normalization':'mapless_canonical_stereo_isotope_charge_preserved_whole_salts_v1',
        'stocks':{},'cohorts':{},'stock_preparation_complete':normalize_stocks,
        'composition_annotation_complete':False,'raw_redistribution':False}
    normalized={}
    for name,item in config['stocks'].items():
        with open_text(raw_root/item['path']) as handle:
            raw=[row[item['smiles_column']] for row in csv.DictReader(handle)]
        if len(raw)!=item['rows']:raise ValueError(f'full stock denominator mismatch: {name}')
        information={**item,'source_rows':len(raw),'normalized_rows':None,'canonical_sha256':None}
        if normalize_stocks:
            known=set();errors=[]
            def consume(results):
                for processed,(_,canonical,error) in enumerate(results,1):
                    if error:errors.append(error)
                    else:known.add(canonical)
                    if processed % 4096==0:print({'phase':'stock-normalization','stock':name,'done':processed,'total':len(raw)},flush=True)
            if workers==1:
                consume(map(normalize_stock_row,enumerate(raw)))
            else:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    consume(pool.map(normalize_stock_row,enumerate(raw),chunksize=256))
            destination=output/'stocks'/f'{name}.txt'
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_text(''.join(x+'\n' for x in sorted(known)))
            write_jsonl(output/'stocks'/f'{name}.invalid.jsonl',errors)
            information.update(normalized_rows=len(known),invalid_rows=len(errors),canonical_path=str(destination.resolve()),
                               canonical_sha256=file_sha256(destination))
            normalized[name]=known
        manifest['stocks'][name]=information
        print({'phase':'stock-source-frozen','stock':name,'rows':len(raw),'normalized':normalize_stocks},flush=True)
    structures={}; source_ids={}
    for name,item in config['cohorts'].items():
        original = raw_root/item['path']
        with open_text(original) as handle:
            if item['format']=='csv':
                source=[{'id':row[item['id_column']],'smiles':row[item['smiles_column']],'acceptable_routes':[]}
                        for row in csv.DictReader(handle)]
            else:source=list(json.load(handle)['targets'].values())
        if len(source)!=item['rows'] or len({row['id'] for row in source})!=len(source):
            raise ValueError(f'full target denominator/ID mismatch: {name}')
        rows=[]; reference_routes=[];reference_reactions=[]
        for raw in source:
            identifier=f"{name}:{raw['id']}"
            canonical=normalize_stock_smiles(raw['smiles'])
            route_info=[];edges=[]
            for index,route in enumerate(raw['acceptable_routes']):
                info,items=extract_reference_route(route,identifier,index)
                info['all_reference_leaves_in_stock']=(all(x in normalized[item['stock']] for x in info['leaf_smiles'])
                                                      if normalize_stocks else None)
                route_info.append(info);edges.extend(items)
            rows.append({'id':identifier,'source_id':raw['id'],'cohort':name,'product_unmapped':canonical,
                'source_product_smiles':raw['smiles'],'stock':item['stock'],
                'already_in_stock':canonical in normalized[item['stock']] if normalize_stocks else None,
                'reference_routes':route_info,'reference_reactions':edges,
                'reference_depth':route_info[0]['depth'] if route_info else None,
                'composition_stratum':'Unknown'})
            reference_routes.extend(route_info);reference_reactions.extend(edges)
        cohort_dir=output/name
        write_jsonl(cohort_dir/'targets.jsonl',rows)
        write_jsonl(cohort_dir/'r2_reference_routes.jsonl',reference_routes)
        write_jsonl(cohort_dir/'r2_reference_reactions.jsonl',reference_reactions)
        depths=Counter(row['reference_depth'] for row in rows if row['reference_depth'] is not None)
        if name=='retrocast_mkt_cnv160' and depths!=Counter({2:40,3:40,4:40,5:40}):
            raise ValueError('source mkt-cnv160 depth strata mismatch')
        manifest['cohorts'][name]={**item,'rows':len(rows),'target_ids':[row['id'] for row in rows],
            'targets_path':str((cohort_dir/'targets.jsonl').resolve()),'targets_sha256':file_sha256(cohort_dir/'targets.jsonl'),
            'reference_routes_sha256':file_sha256(cohort_dir/'r2_reference_routes.jsonl'),
            'reference_reactions_sha256':file_sha256(cohort_dir/'r2_reference_reactions.jsonl'),
            'reference_depth_counts':dict(depths),'reference_routes':len(reference_routes),
            'reference_reactions':len(reference_reactions),'zero_step_targets':sum(row['already_in_stock'] is True for row in rows) if normalize_stocks else None,
            'reference_stock_closed_routes':sum(r['all_reference_leaves_in_stock'] is True for r in reference_routes) if normalize_stocks else None}
        structures[name]={row['product_unmapped'] for row in rows}
        source_ids[name]={row['source_id'] for row in rows}
        print({'phase':'cohort-frozen','cohort':name,'rows':len(rows),'reference_depths':dict(depths)},flush=True)
    manifest['overlap']=[{'cohorts':[a,b],'same_structure_count':len(structures[a]&structures[b]),
                         'shared_source_ids':sorted(source_ids[a]&source_ids[b])}
                         for a,b in itertools.combinations(config['cohorts'],2)]
    manifest['full_target_rows']=sum(item['rows'] for item in manifest['cohorts'].values())
    write_json(output/'manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=64)
    parser.add_argument('--references-only',action='store_true')
    args=parser.parse_args()
    workers=min(args.workers,len(os.sched_getaffinity(0)))
    if workers<1:raise ValueError('positive CPU worker count required')
    manifest=freeze_assets(args.config,args.output,workers=workers,normalize_stocks=not args.references_only)
    print({'phase':'route-assets-frozen','rows':manifest['full_target_rows'],'stocks_ready':manifest['stock_preparation_complete']},flush=True)


if __name__=='__main__':main()
