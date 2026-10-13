#!/usr/bin/env python3
"""Analyze immutable full-cohort R1 outputs; reuse mkt-cnv160 for R2.

Only observed route outcomes enter metrics/figures. Unknown electronic
annotation is not C2. Greedy/tree actual compute is reported, not claimed
matched merely because nominal task budget ceilings agree.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from mechet.frozen_route_assets import normalize_stock_smiles, file_sha256
from mechet.frozen_route_mechanisms import merge_reference_annotations


def load_mechanism_index(path,assets,*,model_sha,train_sha,decisions,reactions):
    value=json.loads(Path(path).read_text())
    if (value.get('schema_version')!='actual_sft_frozen_route_annotation_v1'
            or value.get('model_sha256')!=model_sha
            or value.get('source_config_sha256')!=assets['config_sha256']):
        raise ValueError('mechanism annotation source/model mismatch')
    exposure=value.get('actual_train_exposure',{})
    if (exposure.get('source_sha256')!=train_sha or exposure.get('decision_rows')!=decisions
            or exposure.get('reaction_rows')!=reactions):
        raise ValueError('mechanism actual full training exposure mismatch')
    for name,cohort in assets['cohorts'].items():
        if value.get('cohorts',{}).get(name,{}).get('reference_reactions_sha256')!=cohort['reference_reactions_sha256']:
            raise ValueError('mechanism annotation frozen references mismatch')
    return value


def paired_bootstrap(before,after,*,resamples=2000,seed=17):
    if len(before)!=len(after) or not before:raise ValueError('paired target lengths must agree and be nonempty')
    differences=np.asarray(after,dtype=float)-np.asarray(before,dtype=float)
    rng=np.random.default_rng(seed)
    samples=differences[rng.integers(0,len(before),size=(resamples,len(before)))].mean(axis=1)
    return {'n':len(before),'resamples':resamples,'delta':float(differences.mean()),
            'ci95':[float(x) for x in np.quantile(samples,[.025,.975])]}


def edge_key(product,precursors):
    return (normalize_stock_smiles(product),tuple(sorted(normalize_stock_smiles(x) for x in precursors)))


def published_route_recovered(generated,reference_reactions):
    if not generated or not reference_reactions:return False
    actual={edge_key(edge['product'],edge['precursors']) for edge in generated}
    alternatives=defaultdict(set)
    for edge in reference_reactions:
        alternatives[edge['reference_route_id']].add(edge_key(edge['source_product_smiles'],edge['source_precursor_smiles']))
    return any(actual==reference for reference in alternatives.values())


def graph_contains_published_route(generated,reference_reactions):
    if not generated or not reference_reactions:return False
    actual={edge_key(edge['product'],edge['precursors']) for edge in generated}
    alternatives=defaultdict(set)
    for edge in reference_reactions:
        alternatives[edge['reference_route_id']].add(edge_key(edge['source_product_smiles'],edge['source_precursor_smiles']))
    return any(reference<=actual for reference in alternatives.values())


def validate_shard_contract(manifest,expected,*,rank,world,mode):
    required={**expected,'rank':rank,'world':world,'mode':mode,
              'evaluation_unit':'stock_connected_AND_route','route_cohort':'all'}
    for key,value in required.items():
        if manifest.get(key)!=value:raise ValueError(f'route shard contract mismatch: {key}')


def rate_interval(values,*,seed=17):
    if not values:return {'solved':None,'ci95':None,'n':0}
    array=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    rates=array[rng.integers(0,len(array),size=(2000,len(array)))].mean(axis=1)
    return {'solved':float(array.mean()),'ci95':[float(x) for x in np.quantile(rates,[.025,.975])],'n':len(values)}


def aggregate_cohort(targets,conditions):
    identifiers=[row['id'] for row in targets]
    if len(set(identifiers))!=len(identifiers):raise ValueError('source denominator contains duplicate IDs')
    indexed={}
    for mode,rows in conditions.items():
        if len(rows)!=len(identifiers) or {row['id'] for row in rows}!=set(identifiers):
            raise ValueError(f'full prediction denominator mismatch: {mode}')
        indexed[mode]={row['id']:row for row in rows}
        for target in targets:
            row=indexed[mode][target['id']]
            if row.get('already_in_stock')!=target['already_in_stock']:
                raise ValueError('prediction/source zero-step stock flag mismatch')
            if row['stock_solved'] and row['already_in_stock']:
                raise ValueError('zero-step target mislabeled as nontrivial route')
    def group(subset):
        return {'n':len(subset),'conditions':{mode:rate_interval([indexed[mode][t['id']]['stock_solved'] for t in subset])
                                            for mode in indexed}}
    strata={label:group([target for target in targets if target.get('composition_stratum','Unknown')==label])
            for label in ['C0','C1','C2','Unknown']}
    depth={str(value):group([target for target in targets if target.get('reference_depth')==value])
           for value in sorted({target['reference_depth'] for target in targets if target.get('reference_depth') is not None})}
    summary={}
    for mode,rows in conditions.items():
        known=[r['work']['sampled_tokens'] for r in rows]
        summary[mode]={**rate_interval([r['stock_solved'] for r in rows]),
            'zero_step_targets':sum(t['already_in_stock'] for t in targets),
            'total_output_tokens':sum(known),'mean_output_tokens':sum(known)/len(targets),
            'structural_reference_graph_recovery':(sum(graph_contains_published_route(
                indexed[mode][target['id']].get('generated_reaction_graph',[]),target.get('reference_reactions',[]))
                for target in targets)/len(targets)) if any(t.get('reference_routes') for t in targets) else None,
            'unique_route_count_lower_bound':sum(r['unique_route_count_lower_bound'] for r in rows)
                                            if all('unique_route_count_lower_bound' in r for r in rows) else None,
            'route_count_is_exhaustive':False}
    paired=(paired_bootstrap([indexed['greedy'][identifier]['stock_solved'] for identifier in identifiers],
                             [indexed['tree'][identifier]['stock_solved'] for identifier in identifiers])
            if 'greedy' in indexed and 'tree' in indexed else None)
    return {'denominator':len(targets),'conditions':summary,'strata':strata,'reference_depth':depth,
            'paired_tree_minus_greedy':paired,'equal_actual_compute_claim':False,
            'target_ids':identifiers,'self_improvement_conditions':'not_run_user_paused_RSI'}


def complexity_rows(targets):
    """Primary source-reference route only, with per-dimension availability.

    Atom/changed-bond counts sum per reaction (not unique atoms over a route).
    Ring delta is source forward product minus the entire reactant mixture's
    RDKit ring count; its sign counts are separate so closure/opening cannot
    cancel. Zero ring delta does not rule out ring reorganization. Stereo counts
    count declared chiral atoms, not proof of complete stereochemical labeling.
    No missing source measurement or electronic program becomes a zero.
    """
    rows=[]
    for target in targets:
        reference=(target.get('reference_routes') or [{}])[0]
        relevant=[edge for edge in target.get('reference_reactions',[])
                  if edge['reference_route_id']==reference.get('reference_route_id')]
        def total(key):
            return sum(edge[key] for edge in relevant) if relevant and all(edge.get(key) is not None for edge in relevant) else None
        ring_available=bool(relevant) and all(edge.get('ring_count_change') is not None for edge in relevant)
        dimensions={
            'reference_depth':target.get('reference_depth'),'reaction_edge_count':reference.get('reaction_edge_count'),
            'convergent_branch_count':reference.get('convergent_branch_count'),
            'maximum_precursor_multiplicity':reference.get('maximum_precursor_multiplicity'),
            'changed_bonds_shared_mapped_atoms':total('changed_bonds_shared_mapped_atoms'),
            'mapped_reacting_atom_count_shared':total('mapped_reacting_atom_count_shared'),
            'ring_count_change':total('ring_count_change'),
            'ring_closure_edge_count':sum(edge['ring_count_change']>0 for edge in relevant) if ring_available else None,
            'ring_opening_edge_count':sum(edge['ring_count_change']<0 for edge in relevant) if ring_available else None,
            'stereo_atom_count':total('stereo_atom_count'),
            'explicit_hydrogen_atom_count':total('explicit_hydrogen_atom_count'),
            'reference_event_length':reference.get('reference_event_length'),
            'coupled_electronic_moves':reference.get('coupled_electronic_moves'),
            'imported_participants':reference.get('imported_participants')}
        rows.append({'id':target['id'],'reference_route_id':reference.get('reference_route_id'),
            **dimensions,**{f'{key}_available':value is not None for key,value in dimensions.items()},
            'mapped_reference_edges':sum(edge.get('mapping_valid') is True for edge in relevant),
            'composition_stratum':target.get('composition_stratum','Unknown')})
    return rows


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def plot_results(summaries,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],
                         'pdf.fonttype':42,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    figure,axis=plt.subplots(figsize=(7.2,3.5))
    names=list(summaries)
    labels=['URSA Expert','URSA Drugs/Clinicals','RetroCast market','PaRoutes-derived']
    positions=np.arange(len(names))
    for offset,mode,color,label in [(-.18,'greedy','#777777','Frozen policy (greedy)'),(.18,'tree','#416B82','Adaptive electron tree')]:
        rates=[summaries[name]['conditions'][mode]['solved'] for name in names]
        intervals=[summaries[name]['conditions'][mode]['ci95'] for name in names]
        axis.bar(positions+offset,100*np.asarray(rates),.34,label=label,color=color)
        axis.errorbar(positions+offset,100*np.asarray(rates),yerr=100*np.array([[r-i[0] for r,i in zip(rates,intervals)],
                        [i[1]-r for r,i in zip(rates,intervals)]]),fmt='none',ecolor='black',capsize=3,linewidth=.8)
    axis.set_xticks(positions,[f'{label}\n(n={summaries[name]["denominator"]})' for label,name in zip(labels,names)])
    axis.set_ylabel('Stock-connected targets (%)');axis.set_ylim(0,100)
    axis.legend(frameon=False);figure.tight_layout()
    figure.savefig(destination/'figure2_route_solved.pdf');plt.close(figure)
    r2=summaries['retrocast_mkt_cnv160']
    figure,axis=plt.subplots(figsize=(5.0,3.4))
    bins=list(r2['reference_depth'])
    for mode,color,offset,label in [('greedy','#777777',-.18,'Frozen policy (greedy)'),('tree','#416B82',.18,'Adaptive electron tree')]:
        values=[r2['reference_depth'][depth]['conditions'][mode] for depth in bins]
        rates=np.asarray([value['solved'] for value in values])
        axis.bar(np.arange(len(bins))+offset,100*rates,.34,color=color,label=label)
        axis.errorbar(np.arange(len(bins))+offset,100*rates,
            yerr=100*np.array([[rate-value['ci95'][0] for rate,value in zip(rates,values)],
                              [value['ci95'][1]-rate for rate,value in zip(rates,values)]]),fmt='none',ecolor='black',capsize=3,linewidth=.8)
    axis.set_xticks(np.arange(len(bins)),[f'{depth}\n(n={r2["reference_depth"][depth]["n"]})' for depth in bins])
    axis.set_xlabel('Reference synthesis depth');axis.set_ylabel('Stock-connected targets (%)');axis.set_ylim(0,100)
    axis.legend(frameon=False);figure.tight_layout()
    figure.savefig(destination/'figure3_compositional_complexity.pdf');plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--analysis-output',type=Path,help='separate destination preserving previous analysis')
    parser.add_argument('--mechanism-annotations',type=Path,help='sealed actual-SFT reference annotation manifest')
    parser.add_argument('--world',type=int,default=8)
    parser.add_argument('--allow-single-condition',action='store_true')
    args=parser.parse_args()
    assets=json.loads(args.assets.read_text())
    if not assets['stock_preparation_complete']:raise ValueError('stock preparation pending')
    from scripts.infer_dynamic_electron_trees import ADAPTER_SHA,REVISION
    destination=args.analysis_output or args.output
    annotations=None
    if args.mechanism_annotations:
        adapter=Path('/aaa/fionafyang/buddy1/whaleywang/MechET/outputs/agent/natural_language_event_history_v2_qwen3_0_6b_seed17')
        contract=json.loads((adapter/'data_contract.json').read_text())
        annotations=load_mechanism_index(args.mechanism_annotations,assets,model_sha=ADAPTER_SHA,
            train_sha=contract['train_file_sha256'],decisions=contract['n_rows'],reactions=contract['finish_trace_rows'])
    summaries={};provenance=[];all_conditions={};contracts=[];runtime_totals={}
    expected={'adapter_sha':ADAPTER_SHA,'revision':REVISION,'model':'Qwen/Qwen3-0.6B',
        'source_sha':assets['full_targets_sha256'],'route_assets_sha':file_sha256(args.assets),
        'code_sha':assets['preparation_code_sha'],
        'task_budget_mode':'renewable_checkpoints_no_token_or_time_cap_v1',
        'stock_canonical_shas':{name:stock['canonical_sha256'] for name,stock in assets['stocks'].items()}}
    all_source_ids={identifier for cohort in assets['cohorts'].values() for identifier in cohort['target_ids']}
    for mode in ['greedy','tree']:
        shard_paths=[args.output/mode/f'rank{rank:02d}'/'complete.json' for rank in range(args.world)]
        if not all(path.exists() for path in shard_paths):
            if args.allow_single_condition:continue
            raise ValueError(f'full route condition incomplete: {mode}')
        shards=[json.loads(path.read_text()) for path in shard_paths]
        for rank,shard in enumerate(shards):
            validate_shard_contract(shard['manifest'],expected,rank=rank,world=args.world,mode=mode)
            contracts.append(shard['manifest'])
        rows=[row for shard in shards for row in shard['rows']]
        if len(rows)!=len(all_source_ids) or {row['id'] for row in rows}!=all_source_ids:
            raise ValueError('shared full460 denominator mismatch')
        all_conditions[mode]=rows
        runtime_totals[mode]={'output_tokens':sum(shard['tokens'] for shard in shards),
                              'model_forward_calls':sum(shard['model_forward_calls'] for shard in shards)}
        provenance.extend({'path':str(path),'sha256':file_sha256(path)} for path in shard_paths)
    for key in ['seed','task_token_budget','max_model_len','wall_hours','task_budget_mode','code_sha']:
        if contracts and any(contract[key]!=contracts[0][key] for contract in contracts):
            raise ValueError(f'mixed frozen route conditions: {key}')
    for name,cohort in assets['cohorts'].items():
        path=Path(cohort['targets_path'])
        if file_sha256(path)!=cohort['targets_sha256']:raise ValueError('frozen target SHA mismatch')
        targets=[json.loads(line) for line in path.open() if line.strip()]
        if annotations:
            information=annotations['cohorts'][name]
            annotated_path=Path(information['annotations_path'])
            if file_sha256(annotated_path)!=information['annotations_sha256']:
                raise ValueError('mechanism annotation file SHA mismatch')
            targets=merge_reference_annotations(targets,[json.loads(line) for line in annotated_path.open() if line.strip()])
        cohort_ids=set(cohort['target_ids'])
        conditions={mode:[row for row in rows if row['id'] in cohort_ids] for mode,rows in all_conditions.items()}
        if not conditions:continue
        summary=aggregate_cohort(targets,conditions)
        summaries[name]=summary
        write_json(destination/name/'route_evaluation.json',summary)
        if name=='retrocast_mkt_cnv160':
            r2=destination/'r2';r2.mkdir(parents=True,exist_ok=True)
            complexity=complexity_rows(targets)
            with (r2/'r2_route_complexity.csv').open('w') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(complexity[0]));writer.writeheader();writer.writerows(complexity)
            write_json(r2/'r2_composition_strata.json',{'cohort':name,'denominator':160,'strata':summary['strata'],
                        'annotation_complete':all(t['composition_stratum']!='Unknown' for t in targets),
                        'actual_sft_index_provided':annotations is not None,'missing_is_not_zero':True})
            write_json(r2/'r2_paired_solved_vs_work.json',summary)
            write_json(r2/'r2_h2_historical_separate.json',{
                'status':'historical_H2_not_rerun','full_data_0_6B_is_not_H2_heldout':True,
                'original_train_valid_test':[223863,6200,27104],
                'reported_direct_generation_first':[4030,27104],
                'reported_direct_pass10':[7629,27104],'reported_direct_nll_top1':[5018,27104],
                'source':'https://github.com/wangyu-sd/MechET-paper/blob/ee96f346ccfc2c40038dd1c4f9b6e7f955915659/editorial/R1_R2_COMPLETE_ROUTE_SCENARIOS_AND_COMPOSITION_PROTOCOL_20261013.md',
                'newly_recomputed':False})
    write_json(destination/'r1_route_results.json',summaries)
    write_json(destination/'r1_runtime_work_totals.json',runtime_totals)
    if len(summaries)==4 and all({'greedy','tree'}<=set(summary['conditions']) for summary in summaries.values()):
        plot_results(summaries,destination)
    write_json(destination/'figure3_manifest.json',{'source_shards':provenance,'assets_sha256':file_sha256(args.assets),
        'mechanism_annotation_manifest':str(args.mechanism_annotations) if annotations else None,
        'mechanism_annotation_manifest_sha256':file_sha256(args.mechanism_annotations) if annotations else None,
        'inference_input_assets_unchanged':True,
        'actual_observed_results_only':True,'mechanism_annotation_complete':assets['composition_annotation_complete'],
        'C1_claim_allowed':False if not assets['composition_annotation_complete'] else None,
        'route_count_exhaustive':False,'figures_are_partial_panels_not_replacement_full_paper_figures':True})
    print(json.dumps({'phase':'full-route-analysis','cohorts':list(summaries),'rows':sum(s['denominator'] for s in summaries.values()),
                      'solved':{name:{mode:condition['solved'] for mode,condition in summary['conditions'].items()} for name,summary in summaries.items()}}),flush=True)


if __name__=='__main__':main()
