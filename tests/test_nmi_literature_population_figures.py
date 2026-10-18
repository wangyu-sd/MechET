"""Population figures retain original denominators, missingness and units."""
import copy
import importlib
import gzip
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest


def implementation():
    try:return importlib.import_module('mechet.analysis.literature_population_figures')
    except ModuleNotFoundError:pytest.fail('literature population figure preparation is missing')


def fixture():
    names=['StockSolved@B','AnyExactStepPrecedent@B','AllStepExactPrecedentSolved@B','PrecedentCoveredSolved@B']
    def target(identifier,observed,values):
        return {'target_id':identifier,'report_status':observed,**dict(zip(names,values))}
    m=[target('t1','observed',[True,True,True,True]),target('t2','pending',[False]*4)]
    b=[target('t1','observed',[False]*4),target('t2','observed',[True,False,False,False])]
    zero={'program_occurrences':0,'unique_program_signatures':0,'unique_complete_reactions':0,
          'source_supported_unique_reactions':0,'targets':0,'source_supported_targets':0}
    composition={'denominator':2,'targets_with_programs':1,'seen_ordinary':zero,
        'unseen_ordinary':{**zero,'program_occurrences':2,'unique_program_signatures':1,
            'unique_complete_reactions':1,'source_supported_unique_reactions':1,'targets':1,'source_supported_targets':1},
        'nonordinary':zero,'unavailable':zero}
    metrics={'ranking_is_primary':False,'matched_compute_claim':False,'runs':{},'composition':{},
             'literature_tiers':{'A':1,'C':2},'r2_complete_program_join':'complete'}
    for run,rows,counts in [('r1_mcts_pr88',m,[1,1,1,1]),('r1_greedy_dynamic',b,[1,0,0,0]),('r1_greedy_shared',b,[1,0,0,0])]:
        metrics['runs'][run]={'denominator':2,'observed_reports':1 if run=='r1_mcts_pr88' else 2,
            'pending_reports':1 if run=='r1_mcts_pr88' else 0,'targets':rows,'counts':dict(zip(names,counts))}
        metrics['composition'][run]=copy.deepcopy(composition)
    return metrics,{'t1':'cohort','t2':'cohort'}


def test_panels_preserve_pending_targets_and_independent_program_reaction_units():
    metrics,cohorts=fixture()
    data=implementation().prepare_panels(metrics,cohorts)
    row=data['route_cohorts']['cohort']['r1_mcts_pr88']
    assert row['denominator']==2
    assert row['observed_reports']==1
    assert row['pending_reports']==1
    assert row['StockSolved@B']==1
    assert data['matched_compute_claim'] is False
    assert data['composition']['r1_mcts_pr88']['unseen_ordinary']['program_occurrences']==2
    assert data['composition']['r1_mcts_pr88']['unseen_ordinary']['unique_complete_reactions']==1


@pytest.mark.parametrize('mutation',['drop_target','pending_success','wrong_count','wrong_denominator','ranking'])
def test_population_panels_reject_silent_filtering_or_misreported_metrics(mutation):
    metrics,cohorts=fixture();run=metrics['runs']['r1_mcts_pr88']
    if mutation=='drop_target':run['targets'].pop()
    if mutation=='pending_success':run['targets'][1]['StockSolved@B']=True
    if mutation=='wrong_count':run['counts']['StockSolved@B']=2
    if mutation=='wrong_denominator':run['denominator']=1
    if mutation=='ranking':metrics['ranking_is_primary']=True
    with pytest.raises(ValueError):implementation().prepare_panels(metrics,cohorts)


def test_population_svg_keeps_source_evidence_and_actual_electron_arrows_separate():
    import yaml
    from test_nmi_literature_figures import case_fixture
    from mechet.analysis.literature_figures import prepare_case
    metrics,cohorts=fixture();data=implementation().prepare_panels(metrics,cohorts)
    route,occurrences,programs,tree=case_fixture()
    case=prepare_case(route,occurrences,programs,tree,[])
    style=yaml.safe_load((Path(__file__).parents[1]/'configs/figures/mechet_nmi_tree_style_v1.yaml').read_text())
    figures=implementation().render_population(data,case,style)
    assert set(figures)=={'fig2_r1_literature_supported_routes','fig3_r2_supported_composition'}
    ns={'s':'http://www.w3.org/2000/svg'}
    roots={name:ET.fromstring(svg) for name,svg in figures.items()}
    for root in roots.values():
        assert root.attrib['width']=='183mm'
        assert float(root.attrib['height'].removesuffix('mm'))<=170
        assert root.findall('.//s:path',ns)
        assert root.findall('.//s:text',ns)
        assert not root.findall('.//s:image',ns)
        text=' '.join(n.text or '' for n in root.iter())
        assert 'unequal work' in text
    arrows=roots['fig3_r2_supported_composition'].findall('.//*[@data-electron-flow]',ns)
    assert len(arrows)==1
    assert arrows[0].attrib['data-source-maps']=='1,2'
    assert arrows[0].attrib['data-sink-maps']=='2'
    text=' '.join(n.text or '' for n in roots['fig3_r2_supported_composition'].iter())
    assert 'Reactions with' in text and 'unseen programs' in text
    assert 'Unseen reactions' not in text


def test_population_delivery_rejects_incomplete_corpus_before_output(tmp_path):
    import json
    try:run=importlib.import_module('scripts.render_nmi_literature_population_figures').render_population_figures
    except ModuleNotFoundError:pytest.fail('full-population figure delivery runner is missing')
    analysis=tmp_path/'analysis';analysis.mkdir()
    (analysis/'source_corpus_manifest.json').write_text(json.dumps({'complete':False}))
    with pytest.raises(ValueError,match='corpus'):
        run(analysis,tmp_path/'freeze',tmp_path/'assets',tmp_path/'case',tmp_path/'exposure',[],
            tmp_path/'style',tmp_path/'config',tmp_path/'must-not-exist')
    assert not (tmp_path/'must-not-exist').exists()


def population_files(tmp_path):
    """Real-file software fixture; no model sampling or experimental result."""
    import shutil
    import yaml
    from test_nmi_literature_figures import frozen_case_files
    from mechet.analysis.literature_reaction_keys import make_reaction,stable_hash
    from mechet.analysis.literature_evidence import assess
    from mechet.analysis.literature_programs import join_report_programs,composition_metrics
    from mechet.analysis.literature_route_metrics import aggregate_routes
    from mechet.analysis.literature_sources import file_sha256
    spec,freeze,exposure,report=frozen_case_files(tmp_path)
    def load_rows(name):
        with gzip.open(freeze/(name+'.jsonl.gz'),'rt') as f:return [json.loads(l) for l in f]
    def write_rows(path,records):
        with gzip.open(path,'wt') as f:
            for r in records:f.write(json.dumps(r)+'\n')
    run='r1_mcts_pr88';runs=[run,'r1_greedy_dynamic','r1_greedy_shared']
    route=load_rows('frozen_routes')[0];occ=load_rows('generated_reaction_steps')[0]
    for r in [route,occ]:r['run_id']=run;r['source']['run_id']=run
    write_rows(freeze/'frozen_routes.jsonl.gz',[route])
    write_rows(freeze/'generated_reaction_steps.jsonl.gz',[occ])
    queries=[{k:v for k,v in occ.items() if k not in {'source','run_id','target_id','occurrence_id'}},
             make_reaction('CN',['CCl','N'])]
    write_rows(freeze/'unique_reaction_queries.jsonl.gz',queries)
    targets=[{'id':'t' if i==0 else f't{i}','cohort':cohort}
             for i,cohort in enumerate(['ursa_expert100']*100+['ursa_drugs100']*100+
                                      ['retrocast_mkt_cnv160']*160+['paroutes_derived_n5_100']*100)]
    target_file=tmp_path/'targets.jsonl';target_file.write_text(''.join(json.dumps(t)+'\n' for t in targets))
    assets=tmp_path/'assets.json';assets.write_text(json.dumps({'full_targets_path':str(target_file),
        'full_targets_sha256':file_sha256(target_file)}))
    population=[{'run_id':r,'target_id':t['id'],'report_status':
        'observed' if r!=run or t['id']=='t' else 'pending'} for r in runs for t in targets]
    write_rows(freeze/'target_population.jsonl.gz',population)
    frozen_path=freeze/'prediction_freeze_manifest.json';frozen=json.loads(frozen_path.read_text())
    frozen['source_reports'][0]['run_id']=run
    frozen['route_assets_manifest_sha256']=file_sha256(assets)
    frozen['output_files']={p.name:{'sha256':file_sha256(p)} for p in freeze.glob('*.jsonl.gz')}
    frozen_path.write_text(json.dumps(frozen))
    selected=json.loads(spec.read_text());selected.update(run_id=run,
        prediction_freeze_manifest_sha256=file_sha256(frozen_path),
        program_record_ids=[stable_hash([run,'t','CO','end'])])
    spec.write_text(json.dumps(selected))
    tree=json.loads((report.parent.parent/'trees/t.json').read_text())
    programs=join_report_programs(json.loads(report.read_text()),tree,[occ],json.loads(exposure.read_text()),actor_sha='b'*64)
    program_root=tmp_path/'programs';program_root.mkdir()
    links=program_root/'program_reaction_links.jsonl.gz';write_rows(links,programs)
    pm={'complete':True,'actor_sha256':'b'*64,'actor_exposure_manifest_sha256':file_sha256(exposure),
        'prediction_freeze_manifest_sha256':file_sha256(frozen_path),
        'output_files':{links.name:{'sha256':file_sha256(links)}}}
    (program_root/'program_join_manifest.json').write_text(json.dumps(pm))
    analysis=tmp_path/'analysis';analysis.mkdir();shutil.copyfile(links,analysis/links.name)
    shutil.copyfile(frozen_path,analysis/frozen_path.name)
    evidence={q['reaction_key']:assess(q,[],[]) for q in queries}
    write_rows(analysis/'reaction_evidence.jsonl.gz',list(evidence.values()))
    write_rows(analysis/'literature_hits.jsonl.gz',[])
    metrics={'ranking_is_primary':False,'matched_compute_claim':False,'literature_tiers':{'C':2},
        'runs':{},'composition':{},'r2_complete_program_join':'complete',
        'r2_program_join_manifest_sha256':file_sha256(program_root/'program_join_manifest.json')}
    universe=[t['id'] for t in targets]
    for r in runs:
        observed={p['target_id'] for p in population if p['run_id']==r and p['report_status']=='observed'}
        metrics['runs'][r]=aggregate_routes([route] if r==run else [],evidence,universe=universe,observed_ids=observed)
        metrics['runs'][r].pop('route_evidence')
        metrics['composition'][r]=composition_metrics(programs if r==run else [],evidence,
            [route] if r==run else [],universe=universe)
    (analysis/'metrics.json').write_text(json.dumps(metrics))
    (analysis/'evidence_coverage.json').write_text(json.dumps({'unique_reaction_queries':2,'tier_counts':{'C':2},
        'source_checked_reactions':0,'source_supported_reactions':0,'primary_registry_manifests':[],
        'remaining_work':['primary checking']}))
    source=tmp_path/'source';source.mkdir()
    catalogue={'declared_file_count':93,'download_complete':True,'files':[]}
    receipts=[]
    for i in range(93):
        raw=source/f'{i}.parquet';raw.write_text(f'fixture{i}')
        catalogue['files'].append({'path':raw.name,'sha256':file_sha256(raw)})
        receipts.append({'source_path':str(raw),'source_sha256':file_sha256(raw),'complete':True})
    raw_manifest=source/'catalogue.json';raw_manifest.write_text(json.dumps(catalogue))
    index=source/'reactions.sqlite';index.write_text('software fixture index identity')
    corpus={'complete':True,'files':receipts,'source_corpus':catalogue,'source_manifest':str(raw_manifest),
        'source_manifest_sha256':file_sha256(raw_manifest),'index_path':str(index),'index_sha256':file_sha256(index)}
    (analysis/'source_corpus_manifest.json').write_text(json.dumps(corpus))
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump({'r1':{'full_declared_target_denominator':460,
        'datasets':{'ursa_expert100':100,'ursa_drugs100':100,'retrocast_mkt_cnv160':160,'paroutes_derived_n5_100':100}},
        'actor':{'weight_sha256':'b'*64}}))
    style=Path(__file__).parents[1]/'configs/figures/mechet_nmi_tree_style_v1.yaml'
    arguments=[analysis,freeze,assets,spec,exposure,[],style,config,tmp_path/'output']
    return arguments,program_root


def seal_analysis(arguments,program_root):
    from mechet.analysis.literature_sources import file_sha256
    analysis,freeze=arguments[:2]
    payload={'complete':True,'actor_sha256':'b'*64,
        'prediction_freeze_manifest_sha256':file_sha256(freeze/'prediction_freeze_manifest.json'),
        'source_corpus_manifest_sha256':file_sha256(analysis/'source_corpus_manifest.json'),
        'retrieval_config_sha256':file_sha256(arguments[7]),'primary_registry_manifests':[],
        'program_join_manifest_sha256':file_sha256(program_root/'program_join_manifest.json'),
        'output_files':{p.name:{'sha256':file_sha256(p)} for p in analysis.iterdir()
                        if p.is_file() and p.name!='analysis_manifest.json'}}
    (analysis/'analysis_manifest.json').write_text(json.dumps(payload))


def test_population_delivery_exports_actual_unresolved_case_without_inventing_support(tmp_path):
    arguments,program_root=population_files(tmp_path);seal_analysis(arguments,program_root)
    run=importlib.import_module('scripts.render_nmi_literature_population_figures').render_population_figures
    result=run(*arguments,program_join_root=program_root)
    assert result['original_route_target_denominator']==460
    assert len(result['figures'])==2
    captions=(arguments[-1]/'captions.md').read_text()
    assert 'Fixture only' in captions
    assert 'WO2018104954A1 supports' not in captions
    assert 'US20160022662A1 supports' not in captions
    assert 'has603' not in captions
    assert '0 local primitive' in captions


@pytest.mark.parametrize('mutation',[
    'invent_route_support','invent_baseline_route','invent_observed_reports','exposure_mismatch',
    'duplicate_source_receipts','corpus_catalogue_hash','corpus_index_hash',
    'edited_evidence_table','invent_noncase_support','checked_coverage','missing_analysis_manifest'])
def test_population_delivery_rejects_self_consistent_but_unbound_claims(tmp_path,mutation):
    from mechet.analysis.literature_sources import file_sha256
    arguments,program_root=population_files(tmp_path)
    analysis=arguments[0];mp=analysis/'metrics.json';metrics=json.loads(mp.read_text())
    cp=analysis/'source_corpus_manifest.json';corpus=json.loads(cp.read_text())
    if mutation=='invent_route_support':
        record=metrics['runs']['r1_mcts_pr88']
        for k in ['AnyExactStepPrecedent@B','AllStepExactPrecedentSolved@B','PrecedentCoveredSolved@B']:
            record['targets'][0][k]=True;record['counts'][k]=1
    if mutation=='invent_baseline_route':
        record=metrics['runs']['r1_greedy_dynamic'];record['targets'][0]['StockSolved@B']=True
        record['counts']['StockSolved@B']=1
    if mutation=='invent_observed_reports':
        record=metrics['runs']['r1_mcts_pr88'];record.update(observed_reports=460,pending_reports=0)
        for r in record['targets']:r['report_status']='observed'
    if mutation=='exposure_mismatch':
        path=program_root/'program_join_manifest.json';p=json.loads(path.read_text())
        p['actor_exposure_manifest_sha256']='a'*64;path.write_text(json.dumps(p))
        metrics['r2_program_join_manifest_sha256']=file_sha256(path)
    if mutation=='duplicate_source_receipts':corpus['files']=[corpus['files'][0]]*93
    if mutation=='corpus_catalogue_hash':Path(corpus['source_manifest']).write_text('{}')
    if mutation=='corpus_index_hash':Path(corpus['index_path']).write_text('changed index')
    cp.write_text(json.dumps(corpus));mp.write_text(json.dumps(metrics))
    if mutation=='checked_coverage':
        path=analysis/'evidence_coverage.json';c=json.loads(path.read_text());c['source_checked_reactions']=2
        path.write_text(json.dumps(c))
    if mutation=='invent_noncase_support':
        path=analysis/'reaction_evidence.jsonl.gz'
        with gzip.open(path,'rt') as f:records=[json.loads(l) for l in f]
        records[1].update(tier='A',supported=True,verification_status='source_passage_checked',selected_source='invented')
        with gzip.open(path,'wt') as f:
            for r in records:f.write(json.dumps(r)+'\n')
        metrics['literature_tiers']={'C':1,'A':1};mp.write_text(json.dumps(metrics))
        path=analysis/'evidence_coverage.json';c=json.loads(path.read_text())
        c.update(tier_counts={'C':1,'A':1},source_checked_reactions=1,source_supported_reactions=1)
        path.write_text(json.dumps(c))
    if mutation!='missing_analysis_manifest':seal_analysis(arguments,program_root)
    if mutation=='edited_evidence_table':
        path=analysis/'reaction_evidence.jsonl.gz'
        with gzip.open(path,'rt') as f:records=[json.loads(l) for l in f]
        with gzip.open(path,'wt') as f:
            for r in records:f.write(json.dumps(r,sort_keys=True)+'\n')
    run=importlib.import_module('scripts.render_nmi_literature_population_figures').render_population_figures
    with pytest.raises(ValueError):run(*arguments,program_join_root=program_root)
    assert not arguments[-1].exists()
