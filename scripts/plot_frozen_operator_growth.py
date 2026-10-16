#!/usr/bin/env python3
"""Render Figure3b's observed historical v1 census, not new model results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from mechet.frozen_route_assets import file_sha256


def render_growth(path,*,expected_sha,destination):
    path=Path(path);destination=Path(destination)
    if file_sha256(path)!=expected_sha:raise ValueError('historical report SHA mismatch')
    report=json.loads(path.read_text())
    expected={'artifact_type':'nmi_execution_operator_basis_audit_v1',
        'scope':'strict_executable_training_pool_only_no_final_heldout_test',
        'rows':257167,'unique_source_ids':257167,
        'primitive_basis':'source_to_sink_execution_moves_v1',
        'composition_basis':'execution_composition_signature_ordered_moves_v1'}
    for key,value in expected.items():
        if report.get(key)!=value:raise ValueError(f'historical census contract mismatch: {key}')
    if file_sha256(Path(report['source']))!=report['source_sha256']:
        raise ValueError('historical full source SHA mismatch')
    points=report.get('vocabulary_growth',[])
    previous=(0.,0,0)
    for point in points:
        current=(point['hash_sample_fraction'],point['unique_primitives'],point['unique_compositions'])
        if (not previous[0]<current[0]<=1 or any(type(n) is not int or n<1 for n in current[1:])
                or current[1]<previous[1] or current[2]<previous[2]):
            raise ValueError('historical growth must have increasing coverage and nondecreasing observed counts')
        previous=current
    if (not points or previous!=(1.,report['unique_local_primitives'],report['unique_complete_move_compositions'])):
        raise ValueError('historical growth lacks full observed census endpoint')
    if (destination/'figure3b_manifest.json').exists():
        raise ValueError('preserve previous figure; choose a new destination')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager,ticker
    font=Path('/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf')
    family='Liberation Serif'
    if font.exists():
        font_manager.fontManager.addfont(str(font));family='Times New Roman'
    with plt.rc_context({'font.family':family,'font.size':11,'pdf.fonttype':42,'svg.fonttype':'path',
            'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black',
            'axes.spines.top':False,'axes.spines.right':False}):
        figure,axis=plt.subplots(figsize=(3.9,3.1))
        x=[100*p['hash_sample_fraction'] for p in points]
        for key,label,color,marker in [('unique_compositions','Ordered programs (v1)','#476B82','o'),
                                      ('unique_primitives','Local primitives (v1)','#777777','s')]:
            y=[p[key] for p in points]
            axis.plot(x,y,color=color,marker=marker,markersize=4,linewidth=1.35,label=label)
            axis.annotate(f'{y[-1]:,}',(x[-1],y[-1]),xytext=(-7,7),textcoords='offset points',
                          ha='right',color='black',fontsize=11)
        axis.set_yscale('log');axis.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        axis.yaxis.set_minor_locator(ticker.NullLocator())
        all_counts=[p[key] for p in points for key in ['unique_primitives','unique_compositions']]
        axis.set_ylim(min(all_counts)*.65,max(all_counts)*1.7)
        axis.set_xlim(-2,104);axis.set_xticks([0,25,50,75,100])
        axis.set_xlabel('Training-pool coverage (%)')
        axis.set_ylabel('Unique execution signatures (log scale)')
        axis.legend(frameon=False,fontsize=9,loc='upper left')
        axis.grid(axis='y',alpha=.15,linewidth=.6)
        figure.tight_layout()
        destination.mkdir(parents=True,exist_ok=True)
        for extension in ['pdf','svg']:
            figure.savefig(destination/f'figure3b_historical_operator_growth.{extension}')
        plt.close(figure)
    provenance={'figure_panel':'historical_Figure3b_only_not_complete_Figure3',
        'report_path':str(path),'report_sha256':expected_sha,'source_path':report['source'],
        'source_sha256':report['source_sha256'],'rows':report['rows'],
        'primitive_basis':report['primitive_basis'],'composition_basis':report['composition_basis'],
        'coverage_definition':'nested_seed42_identifier_hash_subsets_not_optimizer_steps',
        'observed_points':points,'new_model_result':False,'full_route_result':False,
        'current_stage_II_training_exposure_statistics':False,
        'limitations':['v1 omits import identities','coupled-event ordering affects signatures',
                       'some BE_DELTA signatures are not ordinary source/sink primitives'],
        'font_family':family,
        'renderer_code_sha':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        'figure_sha256':{extension:file_sha256(destination/f'figure3b_historical_operator_growth.{extension}')
                         for extension in ['pdf','svg']}}
    (destination/'figure3b_manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--report-sha',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=render_growth(args.report,expected_sha=args.report_sha,destination=args.output)
    print(json.dumps({'phase':'observed-historical-growth','rows':result['rows'],
        'final_point':result['observed_points'][-1],'new_model_result':False,'output':str(args.output)}),flush=True)


if __name__=='__main__':main()
