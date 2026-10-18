"""Saved-program joins must not manufacture novelty or borrow another target."""
import copy
import importlib

import pytest


def implementation():
    try:
        return importlib.import_module('mechet.analysis.literature_programs')
    except ModuleNotFoundError:
        pytest.fail('saved complete-program literature join is not implemented')


def fixture():
    # Hand-checkable inverse heterolysis, not a chemical-precedent assertion.
    actions = [
        {'name':'apply_electron_flow','state_before':'[CH3:1][OH:2]',
         'arguments':{'direction':'retrosynthetic','electron_flow':[
             {'source':'the bond between atoms A01 and A02','destination':'atom A02'}]}},
        {'name':'finish_trace','state_before':'[CH3+:1].[OH-:2]','arguments':{}}]
    tree = {'case_id':'t','target':'CO','work':{'sampled_tokens':11},
            'reaction_trees':{'CO':{'root_context':'root','contexts':[
                {'id':'root','state':'[CH3:1][OH:2]','actions':[],'terminal':False},
                {'id':'one','state':'[CH3+:1].[OH-:2]','actions':actions,'terminal':True},
                {'id':'two','state':'[CH3+:1].[OH-:2]',
                 'actions':copy.deepcopy(actions),'terminal':True}]}}}
    report = {'id':'t','target':'CO','work':{'sampled_tokens':11},
              'generated_reaction_graph':[{'product':'CO','precursors':['[CH3+]','[OH-]']}]}
    from mechet.analysis.literature_reaction_keys import make_reaction
    reaction = make_reaction('CO',['[CH3+]','[OH-]'])
    occurrence = {**reaction,'occurrence_id':'o','target_id':'t','run_id':'r',
                  'source':{'report_sha256':'a'*64,'actor_sha256':'b'*64}}
    exposure = {'model_sha256':'b'*64,'signature_version':'source_to_sink_execution_moves_v1',
                'actual_train_exposure':{'primitive_counts':{},'program_counts':{}}}
    return report,tree,occurrence,exposure


def test_join_keeps_both_saved_programs_for_one_distinct_complete_reaction():
    report,tree,occurrence,exposure = fixture()
    rows = implementation().join_report_programs(report,tree,[occurrence],exposure,actor_sha='b'*64)
    assert len(rows) == 2
    assert {r['native_context_id'] for r in rows} == {'one','two'}
    assert {r['reaction_key'] for r in rows} == {occurrence['reaction_key']}
    assert all(r['exposure_class']=='unseen_ordinary' for r in rows)
    assert all(r['whole_program_unseen'] is True for r in rows)
    assert all(r['all_primitives_seen_at_least_five'] is False for r in rows)
    assert all(r['route_C1_claim'] is False for r in rows)
    assert all(r['mapping_status']=='mapped_center_available' for r in rows)
    assert all(r['complexity']['changed_bond_count']==1 for r in rows)
    assert all(r['complexity']['changed_atom_count']==2 for r in rows)
    assert all(r['complexity']['flow_events']==1 for r in rows)
    assert all(r['independent_chemical_precedent'] is False for r in rows)
    assert rows[0]['program_source_pointer']=='/reaction_trees/CO/contexts/1/actions'


@pytest.mark.parametrize('change', ['actor','signature_version','target','tree_work','duplicate_context'])
def test_program_join_rejects_wrong_lineage_or_tree_snapshot(change):
    report,tree,occurrence,exposure = fixture()
    if change=='actor': exposure['model_sha256']='c'*64
    if change=='signature_version': exposure['signature_version']='different'
    if change=='target': tree['case_id']='another'
    if change=='tree_work': tree['work']['sampled_tokens']=12
    if change=='duplicate_context': tree['reaction_trees']['CO']['contexts'].append(tree['reaction_trees']['CO']['contexts'][1])
    with pytest.raises(ValueError):
        implementation().join_report_programs(report,tree,[occurrence],exposure,actor_sha='b'*64)


def test_unmatched_or_incomplete_program_is_not_invented_or_called_seen():
    report,tree,occurrence,exposure = fixture()
    tree['reaction_trees']['CO']['contexts'][1]['actions'].pop()
    tree['reaction_trees']['CO']['contexts'][2]['state']='[CH2:1]=[O:2]'
    rows=implementation().join_report_programs(report,tree,[occurrence],exposure,actor_sha='b'*64)
    assert len(rows)==1
    assert rows[0]['exposure_class']=='unavailable'
    assert rows[0]['whole_program_unseen'] is None


def test_program_join_accepts_the_actual_single_reaction_native_tree_serializer():
    report,tree,occurrence,exposure=fixture()
    flat={**tree['reaction_trees']['CO'],'case_id':'t','target':'CO','work':tree['work']}
    rows=implementation().join_report_programs(report,flat,[occurrence],exposure,actor_sha='b'*64)
    assert len(rows)==2
    assert rows[0]['program_source_pointer']=='/contexts/1/actions'
    assert {r['exposure_class'] for r in rows}=={'unseen_ordinary'}


@pytest.mark.parametrize('bad_state', ['initial','terminal','continuity'])
def test_action_path_state_mismatch_remains_unavailable_not_unseen(bad_state):
    report,tree,occurrence,exposure=fixture()
    for context in tree['reaction_trees']['CO']['contexts'][1:]:
        if bad_state=='initial':context['actions'][0]['state_before']='[CH3:1][NH2:2]'
        if bad_state=='terminal':context['actions'][-1]['state_before']='[CH2:1]=[O:2]'
        if bad_state=='continuity':context['actions'][0]['result']={'current_state':'CN'}
    rows=implementation().join_report_programs(report,tree,[occurrence],exposure,actor_sha='b'*64)
    assert {r['exposure_class'] for r in rows}=={'unavailable'}
    assert all(r['whole_program_unseen'] is None for r in rows)
    assert all(r['program_state_binding']['valid'] is False for r in rows)
    assert all(r['program_complete'] is False for r in rows)
    assert all(r['complete_signature_training_count'] is None for r in rows)


def test_program_join_rejects_cross_method_and_cross_report_occurrences():
    report,tree,occurrence,exposure=fixture()
    for altered in [{**occurrence,'run_id':'other'},
                    {**occurrence,'source':{**occurrence['source'],'report_sha256':'f'*64}}]:
        with pytest.raises(ValueError,match='report or run'):
            implementation().join_report_programs(report,tree,[occurrence,altered],exposure,actor_sha='b'*64)


def test_composition_metrics_deduplicate_program_occurrences_and_keep_unknown_targets():
    m=implementation()
    programs=[{'run_id':'r','target_id':'t','reaction_key':'x','complete_signature':'s',
               'exposure_class':'unseen_ordinary','all_primitives_seen_at_least_five':True,
               'complexity':{'changed_bond_count':2}},
              {'run_id':'r','target_id':'t','reaction_key':'x','complete_signature':'s',
               'exposure_class':'unseen_ordinary','all_primitives_seen_at_least_five':True,
               'complexity':{'changed_bond_count':2}}]
    evidence={'x':{'tier':'A','supported':True,'verification_status':'source_passage_checked'}}
    routes=[{'run_id':'r','target_id':'t','reaction_keys':['x','y'],
             'stock_verified':True,'already_in_stock':False}]
    result=m.composition_metrics(programs,evidence,routes,universe=['t','missing'])
    assert result['denominator']==2
    assert result['unseen_ordinary']['program_occurrences']==2
    assert result['unseen_ordinary']['unique_program_signatures']==1
    assert result['unseen_ordinary']['unique_complete_reactions']==1
    assert result['UnseenProgramToSupportedReaction']['unique_reactions']==1
    assert result['UnseenProgramToSupportedReaction']['targets']==1
    assert result['NovelProgramPrecedentCoveredRoute']['targets']==0
    evidence['y']={'tier':'B','supported':True,'verification_status':'source_passage_checked'}
    result=m.composition_metrics(programs,evidence,routes,universe=['t','missing'])
    assert result['NovelProgramPrecedentCoveredRoute']['targets']==1
    assert result['reference_composition_claim'] is False
    assert result['targets_with_programs']==1


def test_composition_never_borrows_unseen_program_or_support_from_other_target():
    row={'run_id':'r','target_id':'t','reaction_key':'x','complete_signature':'s',
         'exposure_class':'unseen_ordinary','all_primitives_seen_at_least_five':False,
         'complexity':{}}
    evidence={'x':{'tier':'A','supported':True,'verification_status':'machine_structural_match'}}
    route={'run_id':'r','target_id':'u','reaction_keys':['x'],'stock_verified':True}
    result=implementation().composition_metrics([row],evidence,[route],universe=['t','u'])
    assert result['UnseenProgramToSupportedReaction']['targets']==0
    evidence['x']['verification_status']='source_passage_checked'
    result=implementation().composition_metrics([row],evidence,[route],universe=['t','u'])
    assert result['NovelProgramPrecedentCoveredRoute']['targets']==0
    assert result['unseen_combination_supported_primitives']['program_occurrences']==0


def test_composition_distinguishes_checked_contradiction_from_support_and_rejects_mixed_runs():
    row={'run_id':'r','target_id':'t','reaction_key':'x','complete_signature':'s',
         'exposure_class':'unseen_ordinary','complexity':{}}
    evidence={'x':{'tier':'D','supported':False,'verification_status':'source_passage_checked'}}
    result=implementation().composition_metrics([row],evidence,[],universe=['t'])
    assert result['unseen_ordinary']['source_checked_coverage_fraction']==1
    assert result['unseen_ordinary']['source_supported_program_fraction']==0
    with pytest.raises(ValueError,match='run'):
        implementation().composition_metrics([row,{**row,'run_id':'other'}],evidence,[],universe=['t'])


def test_full_freeze_annotation_persists_program_sources_and_empty_targets(tmp_path):
    import gzip, hashlib, json
    try:
        annotate=importlib.import_module('scripts.annotate_nmi_literature_programs').annotate
    except ModuleNotFoundError:
        pytest.fail('full frozen-population program annotation runner missing')
    report,tree,occurrence,exposure=fixture()
    report['tree_artifact']='trees/t.json'
    data=tmp_path/'rank00';(data/'trees').mkdir(parents=True);(data/'evaluations').mkdir()
    (data/'trees/t.json').write_text(json.dumps(tree))
    rp=data/'evaluations/t.json';rp.write_text(json.dumps(report))
    source={'report_path':str(rp),'report_sha256':hashlib.sha256(rp.read_bytes()).hexdigest(),
            'actor_sha256':'b'*64,'run_id':'r','target_id':'t'}
    occurrence['source']=source
    freeze=tmp_path/'freeze';freeze.mkdir()
    payloads={'generated_reaction_steps':[occurrence], 'frozen_routes':[],
              'target_population':[{'run_id':'r','target_id':'t','report_status':'observed'},
                                   {'run_id':'r','target_id':'pending','report_status':'pending'}]}
    files={}
    for name,rows in payloads.items():
        p=freeze/(name+'.jsonl.gz')
        with gzip.open(p,'wt') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        files[p.name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (freeze/'prediction_freeze_manifest.json').write_text(json.dumps({
        'complete':True,'actor_sha256':'b'*64,'source_reports':[source],'output_files':files}))
    ep=tmp_path/'exposure.json';ep.write_text(json.dumps(exposure))
    out=tmp_path/'out'
    manifest=annotate(freeze,ep,out,workers=1)
    assert manifest['complete'] is True
    assert manifest['unmatched_occurrence_count']==0
    assert manifest['counts']['program_records']==2
    with gzip.open(out/'program_reaction_links.jsonl.gz','rt') as f:rows=[json.loads(x) for x in f]
    assert {row['source']['report_sha256'] for row in rows}=={source['report_sha256']}
    assert all(row['source']['tree_sha256']==hashlib.sha256((data/'trees/t.json').read_bytes()).hexdigest() for row in rows)
    census=json.loads((out/'composition_census.json').read_text())
    assert census['r']['denominator']==2
    assert census['r']['unseen_ordinary']['unique_complete_reactions']==1
    work=json.loads((out/'observed_work.json').read_text())
    assert work['r']['observed_targets']==1
    assert work['r']['fields']['sampled_tokens']['observed_n']==1
    assert work['r']['per_target'][0]['values']['sampled_tokens']==11
    # Explicit cache adoption reuses only same-source, same-actor annotations;
    # source trees are re-hashed before reuse, never blindly trusted by path.
    second=annotate(freeze,ep,tmp_path/'reuse',workers=1,compatible_program_cache=out)
    assert second['counts']['reports_reused']==1
    assert second['counts']['program_records']==2
    # A changed cached classification must not be trusted just because its
    # source tree and row count remain unchanged.
    original=(out/'program_reaction_links.jsonl.gz').read_bytes()
    rows[0]['exposure_class']='seen_ordinary'
    with gzip.open(out/'program_reaction_links.jsonl.gz','wt') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='cache artifact checksum'):
        annotate(freeze,ep,tmp_path/'corruptcache',workers=1,compatible_program_cache=out)
    (out/'program_reaction_links.jsonl.gz').write_bytes(original)
    # Incomplete legacy artifacts need independent checksum pins. A trusted
    # exact blob plus complete report-prefix boundary can be safely reused.
    old_manifest=out/'program_join_manifest.json'
    partial=json.loads(old_manifest.read_text());partial['complete']=False
    old_manifest.write_text(json.dumps(partial))
    progress=json.loads((out/'progress.json').read_text());progress['counts']['reports']=1
    (out/'progress.json').write_text(json.dumps(progress))
    with pytest.raises(ValueError,match='external checksum pins'):
        annotate(freeze,ep,tmp_path/'unpinned',workers=1,compatible_program_cache=out)
    pinned=annotate(freeze,ep,tmp_path/'pinned',workers=1,compatible_program_cache=out,
        cache_manifest_sha256=hashlib.sha256(old_manifest.read_bytes()).hexdigest(),
        cache_links_sha256=hashlib.sha256(original).hexdigest())
    assert pinned['counts']['reports_reused']==1
    partial['complete']=True;old_manifest.write_text(json.dumps(partial))
    (data/'trees/t.json').write_text(json.dumps({**tree,'work':{'sampled_tokens':12}}))
    with pytest.raises(ValueError,match='cached native tree checksum'):
        annotate(freeze,ep,tmp_path/'wrongtree',workers=1,compatible_program_cache=out)
    # Source tampering must fail before publishing any mixed-snapshot program.
    rp.write_text('{}')
    with pytest.raises(ValueError,match='checksum'):
        annotate(freeze,ep,tmp_path/'tampered',workers=1)
