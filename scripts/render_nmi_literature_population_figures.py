#!/usr/bin/env python3
"""Export source-bound R1/R2 population figures, never reference Top-K plots."""
import argparse
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from mechet.analysis.figure_qa import inspect_figure
from mechet.analysis.literature_population_figures import prepare_panels,render_population,RUNS
from mechet.analysis.literature_programs import composition_metrics
from mechet.analysis.literature_mapping import load_query_view
from mechet.analysis.literature_evidence import assess
from mechet.analysis.literature_reaction_keys import make_reaction
from mechet.analysis.literature_route_metrics import aggregate_routes,source_supported
from mechet.analysis.literature_sources import file_sha256


def rows(path):
    with gzip.open(path,'rt') as handle:
        yield from (json.loads(line) for line in handle)


def authenticate_original_corpus(corpus):
    """An original catalogue is a set of pinned source identities, not a count."""
    source_path=Path(corpus.get('source_manifest',''))
    if not source_path.is_file() or file_sha256(source_path)!=corpus.get('source_manifest_sha256'):
        raise ValueError('original corpus catalogue checksum differs')
    catalogue=json.loads(source_path.read_bytes())
    if catalogue!=corpus['source_corpus'] or catalogue.get('download_complete') is not True:
        raise ValueError('original corpus catalogue identity differs')
    originals={str((source_path.parent/r['path']).resolve()):r['sha256'] for r in catalogue['files']}
    receipts=corpus['files']
    converted={str(Path(r['source_path']).resolve()):r['source_sha256'] for r in receipts}
    if (len(originals)!=93 or len(catalogue['files'])!=93 or len(converted)!=93 or
        converted!=originals or any(r.get('complete') is not True for r in receipts)):
        raise ValueError('completed corpus receipts differ from original93 distinct source identities')
    index_path=Path(corpus.get('index_path',''))
    if not index_path.is_file() or file_sha256(index_path)!=corpus.get('index_sha256'):
        raise ValueError('completed corpus index checksum differs')


def authenticate_analysis(root,freeze_path,corpus_path,config_path):
    manifest_path=root/'analysis_manifest.json'
    if not manifest_path.is_file():
        raise ValueError('completed analysis output manifest required')
    manifest=json.loads(manifest_path.read_bytes())
    if (manifest.get('complete') is not True or
        manifest.get('prediction_freeze_manifest_sha256')!=file_sha256(freeze_path) or
        manifest.get('source_corpus_manifest_sha256')!=file_sha256(corpus_path) or
        manifest.get('retrieval_config_sha256')!=file_sha256(config_path)):
        raise ValueError('analysis manifest input identity differs')
    required={'metrics.json','evidence_coverage.json','reaction_evidence.jsonl.gz',
              'literature_hits.jsonl.gz','program_reaction_links.jsonl.gz',
              'prediction_freeze_manifest.json','source_corpus_manifest.json'}
    outputs=manifest.get('output_files',{})
    if not required<=outputs.keys():
        raise ValueError('analysis manifest does not bind required publication tables')
    for name,record in outputs.items():
        if Path(name).name!=name or file_sha256(root/name)!=record['sha256']:
            raise ValueError('analysis output table checksum differs: '+name)
    return manifest


def render_population_figures(analysis_root,freeze_root,assets_path,case_spec,exposure_path,
                              primary_registries,style_path,config_path,output_root,*,program_join_root=None):
    import yaml
    from scripts.render_nmi_literature_figures import load_selected_case
    from scripts.render_nmi_tree_figures import export
    analysis_root,freeze_root,assets_path,style_path,config_path,output_root=map(Path,
        (analysis_root,freeze_root,assets_path,style_path,config_path,output_root))
    corpus_path=analysis_root/'source_corpus_manifest.json'
    corpus=json.loads(corpus_path.read_bytes())
    if (corpus.get('complete') is not True or len(corpus.get('files',[]))!=93 or
        corpus.get('source_corpus',{}).get('declared_file_count')!=93 or
        corpus.get('source_corpus',{}).get('full_collection_file_count')):
        raise ValueError('population figures require completed original93-file corpus, not a repair subset')
    authenticate_original_corpus(corpus)
    context=load_selected_case(case_spec,freeze_root,exposure_path,primary_registries)
    freeze=context['freeze'];fpath=freeze_root/'prediction_freeze_manifest.json'
    delivery=authenticate_analysis(analysis_root,fpath,corpus_path,config_path)
    if json.loads((analysis_root/fpath.name).read_bytes())!=freeze:
        raise ValueError('population analysis belongs to another frozen prediction pool')
    if file_sha256(assets_path)!=freeze['route_assets_manifest_sha256']:
        raise ValueError('population route asset checksum differs')
    assets=json.loads(assets_path.read_bytes());target_path=Path(assets['full_targets_path'])
    if file_sha256(target_path)!=assets['full_targets_sha256']:
        raise ValueError('population original target-file checksum differs')
    targets=[json.loads(l) for l in target_path.read_text().splitlines() if l]
    if len({t['id'] for t in targets})!=len(targets):
        raise ValueError('duplicate target in original figure population')
    cohorts={t['id']:t['cohort'] for t in targets}
    config=yaml.safe_load(config_path.read_text())
    if (len(targets)!=460 or config['r1']['full_declared_target_denominator']!=460 or
        Counter(cohorts.values())!=Counter(config['r1']['datasets']) or
        config['actor']['weight_sha256']!=freeze['actor_sha256']):
        raise ValueError('population figures require all original460 targets and frozen actor')
    coverage_path=analysis_root/'evidence_coverage.json';metrics_path=analysis_root/'metrics.json'
    coverage=json.loads(coverage_path.read_bytes());metrics=json.loads(metrics_path.read_bytes())
    evidence_rows=list(rows(analysis_root/'reaction_evidence.jsonl.gz'))
    evidence={r['reaction_key']:r for r in evidence_rows}
    queries=list(rows(freeze_root/'unique_reaction_queries.jsonl.gz'))
    mapping_sha=delivery.get('native_mapping_enrichment_manifest_sha256')
    mapping_inputs=[]
    if mapping_sha:
        if program_join_root is None:
            raise ValueError('native query view requires authenticated complete program join')
        names=('mapping_enrichment_manifest.json','unique_reaction_queries_enriched.jsonl.gz')
        if not set(names)<=delivery['output_files'].keys():
            raise ValueError('analysis does not bind both native query-view artifacts')
        if file_sha256(analysis_root/names[0])!=mapping_sha:
            raise ValueError('native query-view manifest checksum differs')
        queries,_=load_query_view(freeze_root,program_join_root,analysis_root)
        mapping_inputs=[analysis_root/name for name in names]
    elif any((analysis_root/name).exists() for name in
             ('mapping_enrichment_manifest.json','unique_reaction_queries_enriched.jsonl.gz')):
        raise ValueError('unbound native query-view artifacts may not be silently ignored')
    if (len(evidence_rows)!=len(evidence) or set(evidence)!={r['reaction_key'] for r in queries} or
        coverage['unique_reaction_queries']!=len(queries) or
        Counter(r['tier'] for r in evidence_rows)!=Counter(metrics['literature_tiers']) or
        Counter(coverage['tier_counts'])!=Counter(metrics['literature_tiers'])):
        raise ValueError('population reaction evidence coverage or counts differ from frozen full query pool')
    if {r['sha256'] for r in coverage['primary_registry_manifests']}!={r['sha256'] for r in context['registry']['source_manifests']}:
        raise ValueError('case and population primary evidence registries differ')
    if (delivery.get('actor_sha256')!=freeze['actor_sha256'] or
        {r['sha256'] for r in delivery['primary_registry_manifests']}!=
        {r['sha256'] for r in context['registry']['source_manifests']}):
        raise ValueError('analysis actor or primary registry identity differs')
    candidates=defaultdict(list);primary=defaultdict(list)
    for hit in rows(analysis_root/'literature_hits.jsonl.gz'):
        key=hit['query_reaction_key']
        if key not in evidence:raise ValueError('source hit outside frozen reaction query pool')
        candidates[key].append({k:v for k,v in hit.items() if k!='query_reaction_key'})
    for record in context['registry']['records']:
        keys={make_reaction(record['source_product'],record['source_precursors'])['reaction_key']}
        keys.update(record[k] for k in ['analogue_query_reaction_key','contradicted_reaction_key'] if record.get(k))
        for key in keys:primary[key].append(record)
    for reaction in queries:
        key=reaction['reaction_key']
        if evidence[key]!=assess(reaction,candidates[key],primary[key]):
            raise ValueError('population source assignment differs from pinned primary evidence: '+key)
    checked=sum(r.get('verification_status') in {'source_passage_checked','source_human_reviewed'} for r in evidence_rows)
    supported=sum(source_supported(r) for r in evidence_rows)
    if coverage['source_checked_reactions']!=checked or coverage['source_supported_reactions']!=supported:
        raise ValueError('population checked/source-supported coverage differs from evidence rows')
    for step in context['prepared']['steps']:
        if evidence[step['reaction_key']]!=step['evidence']:
            # Indexed candidate IDs can be present in population records but
            # not the direct primary-source case. Check substantive support.
            if any(evidence[step['reaction_key']].get(k)!=step['evidence'].get(k) for k in
                   ['tier','supported','selected_source','verification_status','document_sha256']):
                raise ValueError('case evidence differs from population source assignment')
    if program_join_root is None:
        raise ValueError('authenticated full program join required for population composition figure')
    program_join_root=Path(program_join_root);program_path=program_join_root/'program_join_manifest.json'
    program=json.loads(program_path.read_bytes())
    if (program.get('complete') is not True or program['actor_sha256']!=freeze['actor_sha256'] or
        program['prediction_freeze_manifest_sha256']!=file_sha256(fpath) or
        program.get('actor_exposure_manifest_sha256')!=file_sha256(exposure_path) or
        delivery.get('program_join_manifest_sha256')!=file_sha256(program_path) or
        file_sha256(program_path)!=metrics['r2_program_join_manifest_sha256']):
        raise ValueError('population composition does not belong to authenticated completed program join')
    link_path=analysis_root/'program_reaction_links.jsonl.gz'
    if file_sha256(link_path)!=program['output_files'][link_path.name]['sha256']:
        raise ValueError('population program-link table checksum differs')
    programs=defaultdict(list);routes_by_run=defaultdict(list);populations=defaultdict(list)
    for r in rows(link_path):programs[r['run_id']].append(r)
    for r in rows(freeze_root/'frozen_routes.jsonl.gz'):routes_by_run[r['run_id']].append(r)
    for r in rows(freeze_root/'target_population.jsonl.gz'):populations[r['run_id']].append(r)
    for run in RUNS:
        population=populations[run]
        if (len(population)!=len(cohorts) or {r['target_id'] for r in population}!=set(cohorts) or
            any(r['report_status'] not in {'observed','pending'} for r in population)):
            raise ValueError('frozen report population differs from original route target universe')
        observed={r['target_id'] for r in population if r['report_status']=='observed'}
        actual_routes=aggregate_routes(routes_by_run[run],evidence,universe=cohorts,observed_ids=observed)
        for field in ['denominator','observed_reports','pending_reports','counts','targets']:
            expected=actual_routes[field];reported=metrics['runs'][run][field]
            if field=='targets':
                expected=sorted(expected,key=lambda r:r['target_id']);reported=sorted(reported,key=lambda r:r['target_id'])
            if expected!=reported:
                raise ValueError('population route outcomes differ from frozen routes/evidence/report status: '+field)
        actual=composition_metrics(programs[run],evidence,routes_by_run[run],universe=cohorts)
        if actual!=metrics['composition'][run]:
            raise ValueError('population composition summary differs from authenticated source rows')
    data=prepare_panels(metrics,cohorts)
    style=yaml.safe_load(style_path.read_text())
    drawings=render_population(data,context['prepared'],style)
    output_root.mkdir(parents=True,exist_ok=False)
    figures={}
    for name,svg in drawings.items():
        files=export(svg,output_root/'figures'/name,style)
        qa=inspect_figure(files,style)
        figures[name]={'files':files,'qa':qa}
        if not qa['machine_checks_passed']:
            raise ValueError('population vector-export QA failed: '+str(qa['problems']))
    panel_path=output_root/'panel_source_data.json';panel_path.write_text(json.dumps(data,indent=2)+'\n')
    inputs=[*mapping_inputs,analysis_root/'analysis_manifest.json',corpus_path,Path(corpus['source_manifest']),fpath,
            assets_path,target_path,coverage_path,metrics_path,link_path,program_path,
            analysis_root/'reaction_evidence.jsonl.gz',Path(case_spec),Path(exposure_path),style_path,config_path]
    producer=[Path(__file__),ROOT/'src/mechet/analysis/literature_population_figures.py',
              ROOT/'scripts/render_nmi_literature_figures.py',ROOT/'scripts/render_nmi_tree_figures.py',
              ROOT/'src/mechet/analysis/literature_figures.py',ROOT/'src/mechet/analysis/figure_qa.py',
              ROOT/'src/mechet/analysis/literature_mapping.py']
    manifest={'schema':'lgrv_population_figure_manifest_v1','figures':figures,
        'actor_sha256':freeze['actor_sha256'],'original_route_target_denominator':460,
        'unique_reaction_queries':len(queries),'matched_compute_claim':False,'ranking_primary':False,
        'reference_C0_C1_C2':'Unknown','new_model_inference':False,
        'native_mapping_enrichment_manifest_sha256':mapping_sha,
        'source_files':[{'path':str(p.resolve()),'sha256':file_sha256(p)} for p in inputs],
        'producer_files':{str(p.relative_to(ROOT)):file_sha256(p) for p in producer},
        'panel_source_data_sha256':file_sha256(panel_path),'visual_inspection_status':'pending',
        'remaining_primary_review':coverage['remaining_work']}
    (output_root/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    n=len(queries)
    case_text=f'Selected generated case: {context["spec"]["title"]}. Net transformations are in forward orientation. '
    for i,step in enumerate(context['prepared']['steps'],1):
        citation=step['citation'];tier=step['evidence']['tier']
        if citation:
            claim='net-reaction precedent' if tier in {'A','B'} else 'documented conflict, not supporting precedent'
            case_text+=(f'Reaction {i}: Tier {tier} {claim}; '
                f'[{citation["source_id"]}]({citation["source_url"]}), {citation["locator"]}. '
                +citation.get('source_conditions','')+' '+citation.get('scope_limit','')+' ')
        else:case_text+=f'Reaction {i}: Tier C, unresolved; no checked primary support. '
    exposure=json.loads(Path(exposure_path).read_bytes())['actual_train_exposure']
    primitive_count=len(exposure['primitive_counts']);signature_count=len(exposure['program_counts'])
    captions=[
        '# Figure 2 | Recorded stock-connected routes and complete-reaction experimental precedent',
        '(a) All original460 targets and their frozen cohort denominators. MCTS is an incomplete saved-report lower bound; unfinished targets remain pending. Greedy I and II are complete archived single-path conditions. These are recorded-run descriptive comparisons with unequal work, not matched-budget causal effects. '
        '(b) Target counts for nonzero frozen-stock AND closure, any exact precedent on a complete route, every essential reaction exact, and every essential reaction source-checked A/B. A route is covered only when every essential reaction is supported. '
        f'Across all six frozen conditions, {checked:,}/{n:,} distinct complete reactions have checked primary-source evidence (including any checked contradictions); this is review coverage, not a chemical success rate. '
        '(c) '+case_text+'A net precedent does not validate every symbolic arrow or establish a laboratory-tested whole route.',
        '# Figure 3 | Electronic program composition and supported chemical transformations',
        '(a) Seen/training-unseen ordinary electronic program occurrences, distinct ordered signatures and distinct whole reactions under the authenticated actor exposure. These have different units; they are not a common-denominator funnel. '
        '(b) Source-supported distinct reactions and targets for seen versus unseen MCTS programs; the same reaction may be reachable through both groups, so reaction counts are not additive. This is not chemical novelty, strict H2 or a reference-C1 label. All160 reference C0/C1/C2 labels remain Unknown. '
        '(c) Selected actual saved inverse FLOW actions from the fixed case program, with source–sink arrows compiled from their pre-action states. The full program and every import/finish remain in the linked case plates. These states are symbolic executor constructions, not experimentally observed intermediates. '
        f'The authenticated current actor exposure has {primitive_count:,} local primitive and {signature_count:,} ordinary whole signatures. Other historical censuses are not substituted here.',
        '# Source versions and unresolved coverage',
        'All93 official ORD production files, including public Lowe/USPTO extracts, are indexed. An indexed structural candidate is not checked experimental precedent. Tier C remains unresolved, never automatically invalid. Source/training overlap remains unknown unless separately documented. Broader primary-source checking, analogue calibration and work-ledger attribution remain unfinished. These figures are an actual current-evidence draft, not a completed population chemical validation.']
    (output_root/'captions.md').write_text('\n\n'.join(captions)+'\n')
    print(json.dumps({'phase':'population_figures_exported','figures':len(figures),'targets':460,'output_root':str(output_root)}),flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['analysis-root','prediction-freeze-root','route-assets-manifest','case-spec',
                 'actor-exposure-manifest','style','retrieval-config','output-root','program-join-root']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--primary-source-registry',type=Path,action='append',default=[])
    a=parser.parse_args()
    render_population_figures(a.analysis_root,a.prediction_freeze_root,a.route_assets_manifest,a.case_spec,
        a.actor_exposure_manifest,a.primary_source_registry,a.style,a.retrieval_config,a.output_root,
        program_join_root=a.program_join_root)


if __name__=='__main__':main()
