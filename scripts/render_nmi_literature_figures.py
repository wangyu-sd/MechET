#!/usr/bin/env python3
"""Render frozen source-checked chemical cases without new model inference.

This case renderer does not substitute for population Figures 2/3. Primary
document checks are repeated at drawing time; an unchecked source stays C.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from mechet.analysis.figure_qa import inspect_figure
from mechet.analysis.literature_evidence import load_primary_registries
from mechet.analysis.literature_figures import prepare_case,render_case
from mechet.analysis.literature_predictions import resolve_tree_artifact
from mechet.analysis.literature_programs import join_report_programs
from mechet.analysis.literature_sources import file_sha256


def _rows(path):
    with gzip.open(path,'rt') as handle:
        yield from (json.loads(line) for line in handle)


def render_selected_case(spec_path,freeze_root,exposure_path,primary_registry_paths,style_path,output_root):
    import yaml
    from rdkit import rdBase
    from scripts.render_nmi_tree_figures import export
    spec_path,freeze_root,exposure_path,style_path,output_root=map(Path,
        (spec_path,freeze_root,exposure_path,style_path,output_root))
    spec=json.loads(spec_path.read_bytes());freeze_path=freeze_root/'prediction_freeze_manifest.json'
    for path,expected in [(freeze_path,spec['prediction_freeze_manifest_sha256']),
                          (exposure_path,spec['actor_exposure_manifest_sha256'])]:
        if file_sha256(path)!=expected:
            raise ValueError('case input checksum differs: '+str(path))
    freeze=json.loads(freeze_path.read_bytes())
    if freeze.get('complete') is not True:
        raise ValueError('case requires complete frozen prediction pool')
    for name,record in freeze['output_files'].items():
        if file_sha256(freeze_root/name)!=record['sha256']:
            raise ValueError('frozen table checksum differs: '+name)
    routes=[r for r in _rows(freeze_root/'frozen_routes.jsonl.gz') if
            (r['run_id'],r['target_id'],r['route_key'])==(spec['run_id'],spec['target_id'],spec['route_key'])]
    if len(routes)!=1:
        raise ValueError('selected route not unique in frozen prediction pool')
    route=routes[0];route_source=route['source']
    matching=[s for s in freeze['source_reports'] if
        (s['run_id'],s['target_id'])==(route['run_id'],route['target_id']) and
        all(s.get(k)==v for k,v in route_source.items())]
    if len(matching)!=1:
        raise ValueError('route source absent from frozen report inventory')
    source=matching[0]
    if file_sha256(source['report_path'])!=source['report_sha256']:
        raise ValueError('case report checksum differs')
    data=json.loads(Path(source['report_path']).read_bytes())
    report=data['rows'][source['report_row_index']] if 'report_row_index' in source else data
    if source.get('manifest_path'):
        if file_sha256(source['manifest_path'])!=source['manifest_sha256']:
            raise ValueError('actor contract checksum differs')
        contract=json.loads(Path(source['manifest_path']).read_bytes())
        if contract.get('adapter_sha')!=freeze['actor_sha256']:
            raise ValueError('case actor differs from frozen checkpoint')
    native_path=resolve_tree_artifact(report,source)
    if file_sha256(native_path)!=spec['native_tree_sha256']:
        raise ValueError('case native tree checksum differs')
    tree=json.loads(native_path.read_bytes())
    occurrences=[r for r in _rows(freeze_root/'generated_reaction_steps.jsonl.gz') if
                 (r['run_id'],r['target_id'])==(spec['run_id'],spec['target_id'])]
    exposure=json.loads(exposure_path.read_bytes())
    programs=join_report_programs(report,tree,occurrences,exposure,actor_sha=freeze['actor_sha256'])
    wanted=set(spec['program_record_ids'])
    selected=[r for r in programs if r['program_record_id'] in wanted]
    if len(selected)!=len(wanted) or len(wanted)!=len(spec['program_record_ids']):
        raise ValueError('missing or duplicate selected native programs')
    registry=load_primary_registries(primary_registry_paths)
    prepared=prepare_case(route,occurrences,selected,tree,registry['records'])
    style=yaml.safe_load(style_path.read_text())
    main,details,metadata=render_case(prepared,style,title=spec['title'])
    # No output directory before all file, chemical and action bindings pass.
    output_root.mkdir(parents=True,exist_ok=False)
    figures=[]
    for name,svg in [('case_net_reactions',main),*[(f'case_inverse_program_{i+1:02d}',svg) for i,svg in enumerate(details)]]:
        files=export(svg,output_root/'figures'/name,style)
        qa=inspect_figure(files,style)
        figures.append({'id':name,'files':files,'qa':qa})
        if not qa['machine_checks_passed']:
            raise ValueError('publication export QA failed: '+str(qa['problems']))
    metadata_path=output_root/'case_drawing_sources.json'
    metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')
    captions=[f'# {spec["title"]}',
        'Actual frozen model-generated route; net reaction schemes are shown in the forward synthetic direction. '
        'Separate action plates show every recorded inverse IMPORT/FLOW/FINISH, in order, '
        'with source–sink arrows compiled from the saved pre-action state. '
        'FINISH panels show structural precursors; auxiliary endpoint species remain in the source metadata. '
        'Executor states are symbolic constructions, not experimentally observed intermediates. '
        'Stock membership refers to the frozen benchmark catalogue, not present-day vendor availability. '
        + ('This is a source-checked case, not population validation or a matched-compute comparison.'
         if all(s['evidence']['supported'] for s in prepared['steps']) else
         'This case includes unresolved or conflicting reaction evidence; it is not a source-supported route or population validation.')]
    for i,step in enumerate(prepared['steps'],1):
        citation=step['citation'];p=step['program']
        captions.append(f'## Reaction {i}')
        captions.append(f'Canonical reaction `{step["reaction_key"]}`. Program exposure: '
            f'{p["exposure_class"]}; ordered signature `{p.get("complete_signature")}`. '
            'Signature novelty is actor-specific electronic composition, not new chemistry or strict H2.')
        if citation:
            kind=('Documented contradiction' if step['evidence']['tier']=='D' else 'Exact/analogue net-transformation evidence')
            captions.append(f'{kind}: Tier {step["evidence"]["tier"]}; '
                f'[{citation["source_id"]}]({citation["source_url"]}). '+citation['locator'])
            captions.append(citation.get('source_conditions','Conditions unavailable.'))
            captions.append(citation.get('scope_limit','Net-reaction precedent does not establish microscopic arrows.'))
        else:
            captions.append('Tier C: no checked primary evidence. Unresolved is not chemically impossible.')
    (output_root/'captions.md').write_text('\n\n'.join(captions)+'\n')
    with (output_root/'case_registry.csv').open('w') as handle:
        writer=csv.DictWriter(handle,lineterminator='\n',fieldnames=['run_id','target_id','route_key','reaction_key',
            'program_record_id','program_source_pointer','tier','source_id','source_url','native_tree_sha256'])
        writer.writeheader()
        for step in prepared['steps']:
            writer.writerow({**{k:spec[k] for k in ['run_id','target_id','route_key']},
                'reaction_key':step['reaction_key'],**{k:step['program'][k] for k in ['program_record_id','program_source_pointer']},
                'tier':step['evidence']['tier'],'source_id':step['evidence']['selected_source'],
                'source_url':step['evidence'].get('source_url'),'native_tree_sha256':spec['native_tree_sha256']})
    paths=[spec_path,freeze_path,exposure_path,style_path,Path(source['report_path']),native_path]
    producer=[Path(__file__),ROOT/'src/mechet/analysis/literature_figures.py',ROOT/'scripts/render_nmi_tree_figures.py',
              ROOT/'src/mechet/analysis/literature_evidence.py',ROOT/'src/mechet/analysis/literature_programs.py',
              ROOT/'src/mechet/natural_language_electron_flow.py',ROOT/'src/mechet/analysis/figure_qa.py']
    manifest={'schema':'lgrv_source_bound_case_figures_v1','case_id':spec['target_id'],
        'route_key':route['route_key'],'actor_sha256':freeze['actor_sha256'],
        'rdkit_version':rdBase.rdkitVersion,'figures':figures,
        'source_files':[{'path':str(p.resolve()),'sha256':file_sha256(p)} for p in paths],
        'producer_files':{str(p.relative_to(ROOT)):file_sha256(p) for p in producer},
        'primary_registries':registry['source_manifests'],
        'case_drawing_sources_sha256':file_sha256(metadata_path),
        'precedent_covered':prepared['precedent_covered'],
        'physical_mechanism_verified':False,'population_figure_completion':False,
        'visual_inspection_status':'pending','new_model_inference':False}
    (output_root/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['case-spec','prediction-freeze-root','actor-exposure-manifest','style','output-root']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--primary-source-registry',action='append',type=Path,default=[])
    args=parser.parse_args()
    result=render_selected_case(args.case_spec,args.prediction_freeze_root,args.actor_exposure_manifest,
        args.primary_source_registry,args.style,args.output_root)
    print(json.dumps({'phase':'case_figures_exported','figures':len(result['figures']),
                      'precedent_covered':result['precedent_covered'],'output_root':str(args.output_root)}),flush=True)


if __name__=='__main__':
    main()
