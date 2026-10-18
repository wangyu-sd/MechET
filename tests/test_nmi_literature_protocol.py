"""LGRV contracts: failures here must catch inflated chemical evidence."""
import importlib
import hashlib

import pytest


def module(name):
    # A missing implementation is reported as a behavioral assertion during RED.
    try:
        return importlib.import_module('mechet.analysis.' + name)
    except ModuleNotFoundError as error:
        pytest.fail(f'LGRV implementation missing: {error.name}')


def test_reaction_identity_ignores_maps_and_component_order_not_multiplicity():
    keys = module('literature_reaction_keys')
    one = keys.make_reaction('[CH3:7][OH:9]', ['[CH3:1][Br:2]', '[OH-:3]'])
    two = keys.make_reaction('CO', ['[OH-]', 'BrC'])
    assert one['reaction_key'] == two['reaction_key']
    assert one['reaction_key'] != keys.make_reaction('CO', ['CBr', '[OH-]', '[OH-]'])['reaction_key']


@pytest.mark.parametrize('a,b', [('C[C@H](O)F', 'C[C@@H](O)F'),
                               ('C[C@H](O)F', 'CC(O)F'),
                               ('[13CH3]O', 'CO'), ('C[NH3+]', 'CN'),
                               ('CO.[Na+]', 'CO')])
def test_no_silent_stereo_isotope_charge_or_salt_normalization(a, b):
    keys = module('literature_reaction_keys')
    assert keys.make_reaction(a, ['CC'])['reaction_key'] != keys.make_reaction(b, ['CC'])['reaction_key']


def test_auxiliary_roles_do_not_change_structural_reaction_key():
    keys = module('literature_reaction_keys')
    a = keys.make_reaction('CO', ['CBr', '[OH-]'], auxiliary=['O', '[Na+]'])
    b = keys.make_reaction('CO', ['CBr', '[OH-]'])
    assert a['reaction_key'] == b['reaction_key']
    assert a['auxiliary_components'] == ['O', '[Na+]']
    assert keys.make_reaction('CO', ['CBr', 'O'])['reaction_key'] != b['reaction_key']


def test_mapped_center_is_invariant_to_consistent_renumbering():
    keys = module('literature_reaction_keys')
    a = keys.make_reaction('[CH3:1][OH:2]', ['[CH3:1][Br:3]', '[OH-:2]'])
    b = keys.make_reaction('[CH3:9][OH:7]', ['[OH-:7]', '[CH3:9][Br:6]'])
    assert a['mapping_status'] == 'mapped_center_available'
    assert a['reaction_family_key'] == b['reaction_family_key']
    assert a['changed_bond_count'] == 2
    assert a['reaction_family_key'] != keys.make_reaction('[CH3:1][NH2:2]', ['[CH3:1][Br:3]', '[NH2-:2]'])['reaction_family_key']


def test_unmapped_or_duplicate_maps_never_manufacture_a_family():
    keys = module('literature_reaction_keys')
    for product, precursors in [('CO', ['CBr', '[OH-]']),
                                ('[CH3:1][OH:2]', ['[CH3:1][Br:1]', '[OH-:2]'])]:
        result = keys.make_reaction(product, precursors)
        assert result['mapping_status'] == 'mapping_unknown'
        assert result['reaction_family_key'] is None


def test_invalid_smiles_not_silently_discarded():
    with pytest.raises(ValueError, match='SMILES'):
        module('literature_reaction_keys').make_reaction('invalid', ['C'])


def test_index_matches_both_sides_and_deduplicates_known_patent_family(tmp_path):
    index = module('literature_index').ReactionIndex(tmp_path / 'sources.sqlite')
    for source_id, family, reactants in [('s1','family1',['CBr','[OH-]']),
                                       ('s2','family1',['CBr','[OH-]']),
                                       ('s3','family2',['CI','[OH-]'])]:
        index.add({'source_id': source_id, 'patent_family': family,
                   'product': 'CO', 'precursors': reactants, 'license': 'CC-BY-SA-4.0'})
    key = module('literature_reaction_keys').make_reaction('CO', ['CBr', '[OH-]'])
    hits = index.query(key, cap=5)
    assert [h['source_id'] for h in hits] == ['s1']
    assert hits[0]['match_type'] == 'exact'
    assert index.count() == 3
    index.close()


def primary(tmp_path, **changes):
    payload = b'Example 1. The isolated product was obtained from the stated substrates.'
    raw_path = tmp_path/'primary.html';raw_path.write_bytes(payload)
    cache = tmp_path/'primary_cache'
    receipt = module('literature_sources').snapshot_primary_document(raw_path.as_uri(),cache)
    return {**receipt,'source_receipt_path':str(cache/(hashlib.sha256(raw_path.as_uri().encode()).hexdigest()+'.json')),
            'source_id':'test-source',
            'source_product':'CO', 'source_precursors':['CBr','[OH-]'],
            'document_sha256':hashlib.sha256(payload).hexdigest(),
            'checked_passage_sha256':hashlib.sha256(payload).hexdigest(),
            'document_text':payload.decode(),
            'checked_passage':payload.decode(), 'locator':'Example 1',
            'performed_example':True, 'passage_checked':True,
            'checker':'test_fixture_not_empirical_evidence', **changes}


def test_exact_machine_hit_without_primary_passage_is_not_verified_a():
    evidence = module('literature_evidence')
    reaction = module('literature_reaction_keys').make_reaction('CO', ['CBr','[OH-]'])
    result = evidence.assess(reaction, [{'source_id':'s', 'product':'CO',
                                        'precursors':['CBr','[OH-]'], 'match_type':'exact'}])
    assert result['tier'] == 'C'
    assert result['verification_status'] == 'machine_structural_match'
    assert result['supported'] is False


@pytest.mark.parametrize('changes', [{'performed_example':False}, {'passage_checked':False},
                                    {'locator':''}, {'checked_passage':''},
                                    {'source_precursors':['CI','[OH-]']},
                                    {'checked_passage_sha256':'0'*64}])
def test_claims_missing_passage_or_wrong_structures_never_earn_a(changes,tmp_path):
    reaction = module('literature_reaction_keys').make_reaction('CO', ['CBr','[OH-]'])
    result = module('literature_evidence').assess(reaction, [], [primary(tmp_path,**changes)])
    assert result['tier'] == 'C'
    assert result['supported'] is False


def test_source_checked_exact_reaction_is_a_not_physical_arrow_proof(tmp_path):
    reaction = module('literature_reaction_keys').make_reaction('CO', ['CBr','[OH-]'])
    result = module('literature_evidence').assess(reaction, [], [primary(tmp_path)])
    assert result['tier'] == 'A'
    assert result['verification_status'] == 'source_passage_checked'
    assert result['supported'] is True
    assert result['physical_mechanism_verified'] is False


def test_multiple_primary_registries_preserve_seed_and_new_sources_without_id_conflicts(tmp_path):
    import json
    paths=[tmp_path/'seed.json',tmp_path/'extra.json']
    paths[0].write_text(json.dumps({'records':[{'source_id':'seed','source_product':'CO'}]}))
    paths[1].write_text(json.dumps({'records':[{'source_id':'extra','source_product':'CCl'}]}))
    evidence=module('literature_evidence')
    assert hasattr(evidence,'load_primary_registries'), 'cannot combine independently frozen source registries'
    registry=evidence.load_primary_registries(paths)
    assert [r['source_id'] for r in registry['records']]==['seed','extra']
    assert len(registry['source_manifests'])==2
    paths[1].write_text(json.dumps({'records':[{'source_id':'seed','source_product':'CCl'}]}))
    with pytest.raises(ValueError,match='source identity'):
        evidence.load_primary_registries(paths)


def test_no_hit_is_unresolved_not_contradiction():
    reaction = module('literature_reaction_keys').make_reaction('CO', ['CBr','[OH-]'])
    result = module('literature_evidence').assess(reaction, [])
    assert result['tier'] == 'C'
    assert result['retrieval_status'] == 'unresolved'


def test_stereo_unspecified_source_cannot_verify_stereospecific_product(tmp_path):
    reaction = module('literature_reaction_keys').make_reaction('C[C@H](O)F', ['CC(Br)F','[OH-]'])
    result = module('literature_evidence').assess(reaction, [], [primary(tmp_path,
        source_product='CC(O)F', source_precursors=['CC(Br)F','[OH-]'])])
    assert result['tier'] == 'C'


def test_source_download_requires_exact_bytes_and_preserves_existing_file(tmp_path):
    sources = module('literature_sources')
    source = tmp_path / 'origin'
    source.write_bytes(b'official source payload')
    output = tmp_path / 'download'
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    sources.fetch_verified(source.as_uri(), output, sha)
    assert output.read_bytes() == b'official source payload'
    source.write_bytes(b'corrupt new source')
    sources.fetch_verified(source.as_uri(), output, sha)  # verified cache, no rewrite
    assert output.read_bytes() == b'official source payload'
    with pytest.raises(ValueError, match='checksum'):
        sources.fetch_verified(source.as_uri(), tmp_path / 'bad', sha)
    assert not (tmp_path / 'bad').exists()


def test_complete_reaction_extraction_does_not_use_reference_or_auxiliary_substrates():
    predictions = module('literature_predictions')
    report = {'id':'t1','target':'CO','expected_structural_precursor':'POISON_GT',
              'ranked_candidates':[{'context':'c1','structural_precursor':'CBr.[OH-]',
                                    'structural_precursor_mapped':'[CH3:1][Br:3].[OH-:2]',
                                    'full_mixture':'CBr.[OH-].O.[Na+]'}]}
    records = list(predictions.single_reactions(report, run_id='run1', source={'report_sha256':'f'*64}))
    assert len(records) == 1
    assert records[0]['structural_components'] == ['CBr', '[OH-]']
    assert 'POISON_GT' not in str(records)
    assert records[0]['target_id'] == 't1'
    assert records[0]['terminal_record_id'] == 'c1'


def test_multistep_reaction_extraction_is_synthetic_edges_not_flow_operations():
    predictions = module('literature_predictions')
    report = {'id':'t','target':'CO','generated_reaction_graph':[
        {'product':'CO','precursors':['CBr','[OH-]']},
        {'product':'CBr','precursors':['C','BrBr']}],
        'complete_reaction_edges':[], 'enumerated_stock_connected_routes':[]}
    records = list(predictions.route_reactions(report, run_id='r', source={'report_sha256':'f'*64}))
    assert len(records) == 2
    assert {r['reaction_smiles'] for r in records} == {'CBr.[OH-]>>CO', 'BrBr.C>>CBr'}


def test_snapshot_selection_is_hash_bound_and_keeps_only_declared_reports(tmp_path):
    predictions = module('literature_predictions')
    import csv, json
    root = tmp_path / 'reports'; root.mkdir()
    (root / 'saved.json').write_text(json.dumps({'id':'saved','target':'CO'}))
    (root / 'new.json').write_text(json.dumps({'id':'new','target':'CC'}))
    manifest = tmp_path / 'r1_case_results.csv'
    with manifest.open('w') as handle:
        writer = csv.DictWriter(handle,fieldnames=['stable_id','report_relative_path','report_sha256'])
        writer.writeheader(); writer.writerow({'stable_id':'saved','report_relative_path':'saved.json',
                                               'report_sha256':hashlib.sha256((root/'saved.json').read_bytes()).hexdigest()})
    selected = predictions.frozen_reports(root,manifest)
    assert [r[0]['id'] for r in selected] == ['saved']
    (root/'saved.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum'):
        predictions.frozen_reports(root,manifest)


def test_official_catalog_ignores_mirror_test_fixtures_but_checks_git_lfs_identity():
    sources = module('literature_sources')
    path = 'data/ab/ord_dataset-abcdef.parquet'
    pointer = b'version https://git-lfs.github.com/spec/v1\noid sha256:'+b'1'*64+b'\nsize 10\n'
    git_sha = hashlib.sha1(b'blob '+str(len(pointer)).encode()+b'\0'+pointer).hexdigest()
    entry = {'path':path,'type':'file','size':10,'lfs':{'oid':'1'*64}}
    fixture = {'path':'data/ab/test.pb','type':'file','size':5}
    selected, excluded = sources.official_ord_catalog([entry,fixture],[{'path':path,'type':'blob','sha':git_sha}])
    assert selected == [entry]
    assert excluded == ['data/ab/test.pb']
    entry['lfs']['oid'] = '2'*64
    with pytest.raises(ValueError,match='official Git LFS'):
        sources.official_ord_catalog([entry],[{'path':path,'type':'blob','sha':git_sha}])


@pytest.mark.parametrize('n', [2,3,5])
def test_route_coverage_requires_every_edge_not_a_supported_average(n):
    metrics = module('literature_route_metrics')
    route = {'route_key':'r','reaction_keys':[str(i) for i in range(n)],'stock_verified':True,'already_in_stock':False}
    evidence = {str(i):{'tier':'A','supported':True,'verification_status':'source_passage_checked'} for i in range(n-1)}
    result = metrics.route_evidence(route,evidence)
    assert result['precedent_covered'] is False
    assert result['all_exact'] is False
    assert result['any_exact'] is True
    assert result['coverage'] == (n-1)/n
    assert result['unresolved_reaction_keys'] == [str(n-1)]
    evidence[str(n-1)] = {'tier':'B','supported':True,'verification_status':'source_passage_checked'}
    assert metrics.route_evidence(route,evidence)['precedent_covered'] is True
    assert metrics.route_evidence(route,evidence)['all_exact'] is False


def test_zero_step_and_false_stock_cannot_be_supported_routes():
    metrics = module('literature_route_metrics')
    for route in [{'reaction_keys':[],'stock_verified':True,'already_in_stock':True},
                  {'reaction_keys':['a'],'stock_verified':False,'already_in_stock':False}]:
        result = metrics.route_evidence(route,{'a':{'tier':'A','supported':True,'verification_status':'source_passage_checked'}})
        assert result['precedent_covered'] is False
        assert result['all_exact'] is False


def test_target_universe_is_retained_and_pairing_uses_ids_not_row_order():
    metrics = module('literature_route_metrics')
    paired = metrics.paired_targets(['a','b','c','d'], {'a','b'},{'b','c'},bootstrap_draws=2000)
    assert paired['both'] == 1 and paired['method_only'] == 1 and paired['baseline_only'] == 1 and paired['neither'] == 1
    assert paired['denominator'] == 4
    assert paired['difference'] == 0
    assert paired['interpretation'] == 'recorded-run descriptive; unequal work is not a causal search gain'


def test_paired_discovery_keeps_full_denominator_and_separate_observed_intersection():
    metrics=module('literature_route_metrics')
    assert hasattr(metrics,'paired_observed_targets'), 'pending reports are not explicit in paired results'
    result=metrics.paired_observed_targets(['a','b','c'],{'a'},{'b'},
        method_observed={'a'},baseline_observed={'a','b','c'},bootstrap_draws=2000)
    assert result['denominator']==3
    assert result['baseline_only']==1
    assert result['method_pending_reports']==2
    assert result['observed_both']['denominator']==1
    assert result['observed_both']['method_only']==1
    assert result['observed_both']['baseline_only']==0
    assert result['pending_is_failure'] is False


def test_machine_only_tier_a_flag_cannot_leak_into_route_support():
    metrics = module('literature_route_metrics')
    result = metrics.route_evidence({'reaction_keys':['a'],'stock_verified':True,'already_in_stock':False},
        {'a':{'tier':'A','supported':True,'verification_status':'machine_structural_match'}})
    assert result['precedent_covered'] is False


def test_same_target_multiple_programs_dont_inflate_supported_disconnections():
    metrics = module('literature_route_metrics')
    occurrences = [{'target_id':'a','reaction_key':'r1'}, {'target_id':'a','reaction_key':'r1'},
                   {'target_id':'b','reaction_key':'r1'}]
    evidence = {'r1':{'tier':'A','supported':True,'verification_status':'source_passage_checked'}}
    result = metrics.disconnection_counts(occurrences,evidence,universe=['a','b','c'])
    assert result['unique_supported_reactions'] == 1
    assert result['supported_per_target'] == {'a':1,'b':1,'c':0}
    assert result['any_supported_targets'] == 2


def test_ord_adapter_preserves_declared_roles_and_primary_source_locator():
    sources = module('literature_sources')
    raw = {'reaction_id':'ord-test', 'inputs':{
        'substrate':{'components':[{'reaction_role':'REACTANT','identifiers':[{'type':'SMILES','value':'CBr'}]}]},
        'nucleophile':{'components':[{'reaction_role':'REACTANT','identifiers':[{'type':'SMILES','value':'[OH-]'}]}]},
        'solvent':{'components':[{'reaction_role':'SOLVENT','identifiers':[{'type':'SMILES','value':'O'}]}]}},
        'outcomes':[{'products':[{'is_desired_product':True,'identifiers':[{'type':'SMILES','value':'CO'}]}]}],
        'notes':{'procedure_details':'Example 4. An isolated product was obtained.'},
        'provenance':{'patent':'US1234567A','is_mined':True}}
    records = sources.ord_records(raw,dataset={'dataset_id':'d','file_sha256':'f'*64},row_index=12)
    assert len(records) == 1
    assert sorted(records[0]['precursors']) == ['CBr','[OH-]']
    assert records[0]['agents'] == ['O']
    assert records[0]['product'] == 'CO'
    assert records[0]['patent_id'] == 'US1234567A'
    assert records[0]['source_locator']['parquet_row'] == 12
    assert records[0]['verification_status'] == 'machine_structural_match'
    assert records[0]['performed_example_verified'] is False


def test_ord_missing_structural_reactant_smiles_is_not_silently_removed():
    sources = module('literature_sources')
    raw = {'reaction_id':'r','inputs':{'a':{'components':[
        {'reaction_role':'REACTANT','identifiers':[{'type':'SMILES','value':'CBr'}]},
        {'reaction_role':'REACTANT','identifiers':[{'type':'NAME','value':'unknown essential substrate'}]}]}},
        'outcomes':[{'products':[{'identifiers':[{'type':'SMILES','value':'CO'}]}]}]}
    with pytest.raises(ValueError,match='structural reactant'):
        sources.ord_records(raw,dataset={'dataset_id':'d','file_sha256':'f'*64},row_index=0)


def test_passage_hash_alone_without_matching_original_document_cannot_verify_a(tmp_path):
    reaction = module('literature_reaction_keys').make_reaction('CO',['CBr','[OH-]'])
    for changes in [{'document_text':''},{'document_text':'unrelated document'}, {'document_sha256':'0'*64}]:
        result = module('literature_evidence').assess(reaction,[],[primary(tmp_path,**changes)])
        assert result['tier'] == 'C'


def test_parquet_adapter_preserves_absolute_rows_and_dataset_identity(tmp_path):
    import json
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')
    sources = module('literature_sources')
    path = tmp_path/'source.parquet'
    schema = pa.schema([('reaction_id',pa.string()),('reaction',pa.binary())],
                       metadata={b'ord.dataset_id':b'original-dataset',b'ord.name':b'unit-test'})
    table = pa.Table.from_pylist([{'reaction_id':f'r{i}','reaction':json.dumps({'value':i}).encode()} for i in range(3)],schema=schema)
    pq.write_table(table,path,row_group_size=1)
    rows = list(sources.iter_ord_messages(path,json.loads,batch_size=1))
    assert [row[0] for row in rows] == [0,1,2]
    assert [row[1]['dataset_id'] for row in rows] == ['original-dataset']*3
    assert [row[2]['value'] for row in rows] == [0,1,2]


def test_net_reaction_precedent_never_proves_internal_arrow_order():
    mechanisms = module('mechanism_consistency')
    row = {'occurrence_id':'o','reaction_key':'r','run_id':'run','target_id':'t','source':{},
           'formal_status':'recorded_executor_accepted_complete_reaction'}
    result = mechanisms.assess_mechanism(row,{'tier':'A','supported':True})
    assert result['mechanistic_consistency'] == 'formal_only'
    assert result['physical_intermediates_experimentally_verified'] is False


@pytest.mark.parametrize('native_mapping',[False,True])
def test_literature_cli_evaluates_every_fixture_target_without_promoting_index_hits(tmp_path,native_mapping):
    # Filesystem unit integration only: synthetic fixtures, no model or real
    # benchmark sampling. Catches CLI contract/joins and fabricated A metrics.
    import gzip,json,pathlib,subprocess,sys,yaml
    keys = module('literature_reaction_keys')
    sources = module('literature_sources')
    index = module('literature_index')
    freeze = tmp_path/'freeze';freeze.mkdir()
    analysis = tmp_path/'old_analysis';analysis.mkdir()
    (analysis/'source_manifest.json').write_text('{}')
    snapshot = tmp_path/'snapshot.json';snapshot.write_text('{}')
    assets = tmp_path/'assets.json';assets.write_text('{}')
    reaction = keys.make_reaction('CCCO',['CCCBr','[OH-]']) if native_mapping else keys.make_reaction('CO',['CBr','[OH-]'])
    runs = ['r1_mcts_pr88','r1_greedy_dynamic','r1_greedy_shared','mech31k_mcts_pr88',
            'mech31k_greedy_dynamic','mech31k_greedy_shared']
    def write(name,rows):
        with gzip.open(freeze/(name+'.jsonl.gz'),'wt') as handle:
            for row in rows:
                handle.write(json.dumps(row)+'\n')
    write('unique_reaction_queries',[reaction])
    write('generated_reaction_steps',[{**reaction,'occurrence_id':r,'run_id':r,'target_id':'a','source':{},
          'formal_status':'recorded_executor_accepted_complete_reaction'} for r in runs])
    write('target_population',[{'run_id':r,'target_id':t,'report_status':'observed' if t=='a' else 'pending'}
                               for r in runs for t in ['a','b']])
    write('frozen_routes',[{'run_id':r,'target_id':'a','reaction_keys':[reaction['reaction_key']],
                           'stock_verified':True,'already_in_stock':False} for r in runs[:3]])
    manifest = {'complete':True,'actor_sha256':'fixture-actor',
        'progress_snapshot_sha256':sources.file_sha256(snapshot),'route_assets_manifest_sha256':sources.file_sha256(assets),
        'prior_analysis_source_manifest_sha256':sources.file_sha256(analysis/'source_manifest.json'),
        'runs':{r:{'mode':'route' if r.startswith('r1') else 'single'} for r in runs},
        'output_files':{p.name:{'sha256':sources.file_sha256(p)} for p in freeze.glob('*.jsonl.gz')}}
    (freeze/'prediction_freeze_manifest.json').write_text(json.dumps(manifest))
    # A real frozen program-link artifact must be consumed by the evaluation,
    # rather than leaving R2 permanently marked pending after annotation.
    program_root=tmp_path/'programs';program_root.mkdir()
    with gzip.open(program_root/'program_reaction_links.jsonl.gz','wt') as f:
        for run in runs:
            row={'run_id':run,'target_id':'a','reaction_key':reaction['reaction_key'],
                'complete_signature':'unseen','exposure_class':'unseen_ordinary',
                'all_primitives_seen_at_least_five':True,'complexity':{}}
            if native_mapping:
                product='[CH3:1][CH2:2][CH2:3][OH:4]'
                precursors='[CH3:1][CH2:2][CH2:3][Br:5].[OH-:4]'
                center=keys.mapped_center(precursors,product)
                row.update(program_record_id=run,actor_sha256='fixture-actor',
                    actor_exposure_manifest_sha256='e'*64,program_state_binding={'valid':True},
                    mapped_product=product,mapped_structural_precursors=precursors,
                    mapping_status=center['mapping_status'],reaction_family_key=center['reaction_family_key'])
            f.write(json.dumps(row)+'\n')
    (program_root/'observed_work.json').write_text(json.dumps({}))
    (program_root/'program_join_manifest.json').write_text(json.dumps({
        'complete':True,'actor_sha256':'fixture-actor','unmatched_occurrence_count':0,
        'actor_exposure_manifest_sha256':'e'*64,
        'prediction_freeze_manifest_sha256':sources.file_sha256(freeze/'prediction_freeze_manifest.json'),
        'output_files':{name:{'sha256':sources.file_sha256(program_root/name)}
                        for name in ['program_reaction_links.jsonl.gz','observed_work.json']}}))
    db = tmp_path/'index';db.mkdir()
    source_index = index.ReactionIndex(db/'reactions.sqlite')
    if native_mapping:
        source_index.add({'source_id':'analogue-only',
            'product':'[CH3:1][CH2:2][CH2:3][CH2:4][OH:5]',
            'precursors':['[CH3:1][CH2:2][CH2:3][CH2:4][Br:6].[OH-:5]'],'license':'CC-BY-SA-4.0'})
    else:source_index.add({'source_id':'s','product':'CO','precursors':['CBr','[OH-]'],'license':'CC-BY-SA-4.0'})
    source_index.close()
    (db/'source_corpus_manifest.json').write_text(json.dumps({'complete':True,'index_sha256':sources.file_sha256(db/'reactions.sqlite')}))
    config = tmp_path/'retrieval.yaml'
    config.write_text(yaml.safe_dump({'literature':{'protocol':'LGRV-v1','primary_search_cap_per_unique_reaction':5},
        'actor':{'weight_sha256':'fixture-actor'},'output':{'pairwise_ci_bootstrap_draws':2000},
        'main_outcome':'literature_evidence_for_complete_synthetic_reaction_steps_and_stock_routes'}))
    output = tmp_path/'result'
    script = pathlib.Path(__file__).resolve().parents[1]/'scripts/verify_nmi_literature_reactions.py'
    command=[sys.executable,str(script),'--tree-analysis-root',str(analysis),
        '--progress-snapshot',str(snapshot),'--source-reaction-index',str(db/'reactions.sqlite'),
        '--route-assets-manifest',str(assets),'--output-root',str(output),'--retrieval-config',str(config),
        '--prediction-freeze-root',str(freeze),'--program-join-root',str(program_root)]
    if native_mapping:
        view=tmp_path/'native_view'
        module('literature_mapping').enrich_queries(freeze,program_root,view)
        command+=['--native-mapping-view',str(view)]
    completed = subprocess.run(command,capture_output=True,text=True)
    assert completed.returncode == 0,completed.stderr
    delivery=json.loads((output/'analysis_manifest.json').read_text())
    assert delivery['complete'] is True
    assert delivery['prediction_freeze_manifest_sha256']==sources.file_sha256(freeze/'prediction_freeze_manifest.json')
    for name in ['metrics.json','evidence_coverage.json','reaction_evidence.jsonl.gz','literature_hits.jsonl.gz']:
        assert delivery['output_files'][name]['sha256']==sources.file_sha256(output/name)
    metrics = json.loads((output/'metrics.json').read_text())
    assert metrics['literature_tiers'] == {'C':1}
    if native_mapping:
        hits=list(json.loads(l) for l in gzip.open(output/'literature_hits.jsonl.gz','rt'))
        assert [h['source_id'] for h in hits]==['analogue-only']
        assert hits[0]['match_type']=='local_edit_analogue_candidate'
        assert delivery['native_mapping_enrichment_manifest_sha256']==sources.file_sha256(view/'mapping_enrichment_manifest.json')
        assert delivery['output_files']['unique_reaction_queries_enriched.jsonl.gz']['sha256']==sources.file_sha256(view/'unique_reaction_queries_enriched.jsonl.gz')
    assert metrics['runs']['r1_mcts_pr88']['denominator'] == 2
    assert metrics['runs']['r1_mcts_pr88']['pending_reports'] == 1
    assert metrics['runs']['r1_mcts_pr88']['counts']['StockSolved@B'] == 1
    assert metrics['runs']['r1_mcts_pr88']['counts']['PrecedentCoveredSolved@B'] == 0
    assert metrics['runs']['mech31k_mcts_pr88']['AnySourceSupportedReaction'] == 0
    assert (output/'baseline_comparison.csv').exists()
    assert (output/'source_passages.jsonl.gz').exists()
    assert metrics['r2_complete_program_join']=='complete'
    assert metrics['composition']['r1_mcts_pr88']['unseen_ordinary']['unique_complete_reactions']==1
    assert metrics['composition']['r1_mcts_pr88']['UnseenProgramToSupportedReaction']['targets']==0
    assert (output/'program_reaction_links.jsonl.gz').exists()
    assert metrics['runs']['mech31k_mcts_pr88']['reaction_source_outcomes']['unique_generated_reactions']==1
    assert metrics['paired']['mech31k_greedy_dynamic']['AnySourceSupportedReaction']['denominator']==2
    assert metrics['paired_disconnections']['mech31k_greedy_shared']['target_reaction_pairs']['method_only']==0
    assert metrics['paired']['r1_greedy_dynamic']['StockSolved@B']['observed_both']['denominator']==1
    import csv
    with (output/'baseline_comparison.csv').open() as f:first=next(csv.DictReader(f))
    assert first['method_pending_reports']=='1'
    assert first['observed_both_denominator']=='1'
    assert 'lower bound' in first['full_population_outcome']
    # A checked contradiction counts as inspected, never as positive support;
    # a partial program join must remain on the outstanding-work list.
    contradiction=primary(tmp_path,contradicted_reaction_key=reaction['reaction_key'],
                          contradiction_class='fixture-only-conflict',contradiction_evidence='synthetic fixture',
                          source_precursors=['CCl','[OH-]'])
    registry=tmp_path/'contradiction.json';registry.write_text(json.dumps({'records':[contradiction]}))
    pm=program_root/'program_join_manifest.json';partial=json.loads(pm.read_text())
    partial['unmatched_occurrence_count']=1;pm.write_text(json.dumps(partial))
    command=list(completed.args)
    if native_mapping:
        i=command.index('--native-mapping-view');del command[i:i+2]
    command[command.index('--output-root')+1]=str(tmp_path/'checked_d')
    command+=['--primary-source-registry',str(registry)]
    second=subprocess.run(command,capture_output=True,text=True)
    assert second.returncode==0,second.stderr
    coverage=json.loads((tmp_path/'checked_d/evidence_coverage.json').read_text())
    assert coverage['source_checked_reactions']==1
    assert coverage['source_supported_reactions']==0
    assert 'authenticated full R2 join' in coverage['remaining_work']


def test_checked_in_retrieval_config_authenticates_original_snapshot_actor():
    import json
    import pathlib
    import yaml
    root=pathlib.Path(__file__).resolve().parents[1]
    config=yaml.safe_load((root/'configs/analysis/nmi_r1_r2_matched_baselines_20261017.yaml').read_text())
    snapshot=json.loads((root/config['source_progress_snapshot']).read_text())
    assert config['actor']['weight_sha256']==snapshot['mech31k']['actor_sha256']


def test_frozen_readonly_index_preserves_literal_path_and_never_writes(tmp_path):
    import sqlite3
    keys=module('literature_reaction_keys');sources=module('literature_sources')
    cls=module('literature_index').ReactionIndex
    path=tmp_path/'literal?#index.sqlite'
    writable=cls(path)
    writable.add({'source_id':'fixture-source','product':'CO','precursors':['CBr','[OH-]'],'license':'fixture'})
    writable.close();before=sources.file_sha256(path)
    frozen=module('literature_frozen_index').FrozenReactionIndex
    reader=frozen(path,index_sha256=before)
    hits=reader.query(keys.make_reaction('CO',['CBr','[OH-]']))
    assert [h['source_id'] for h in hits]==['fixture-source']
    with pytest.raises(sqlite3.OperationalError):
        reader.add({'source_id':'forbidden','product':'CN','precursors':['CCl','N'],'license':'fixture'})
    reader.close()
    assert sources.file_sha256(path)==before
    with pytest.raises(ValueError):frozen(tmp_path/'must-not-create.sqlite',index_sha256=before)
    assert not (tmp_path/'must-not-create.sqlite').exists()
    with pytest.raises(ValueError):frozen(path,index_sha256='0'*64)


def test_frozen_index_rejects_mutation_during_readonly_analysis(tmp_path):
    import os
    cls=module('literature_index').ReactionIndex
    path=tmp_path/'index.sqlite';writer=cls(path);writer.close()
    frozen=module('literature_frozen_index').FrozenReactionIndex
    reader=frozen(path,index_sha256=module('literature_sources').file_sha256(path))
    stat=path.stat();os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000))
    with pytest.raises(ValueError):reader.close()


def test_aggregate_tree_path_resolution_handles_resumed_absolute_and_new_relative(tmp_path):
    predictions = module('literature_predictions')
    tree = tmp_path/'rank00/trees/tree.json';tree.parent.mkdir(parents=True);tree.write_text('{}')
    source = {'report_path':str(tmp_path/'evaluation.json'),'report_row_index':233}
    assert predictions.resolve_tree_artifact({'tree_artifact':'trees/tree.json'},source) == tree
    assert predictions.resolve_tree_artifact({'tree_artifact':str(tree)},source) == tree
    duplicate = tmp_path/'rank01/trees/tree.json';duplicate.parent.mkdir(parents=True);duplicate.write_text('{}')
    with pytest.raises(ValueError,match='ambiguous'):
        predictions.resolve_tree_artifact({'tree_artifact':'trees/tree.json'},source)


def test_primary_document_cache_binds_source_bytes_and_normalized_text(tmp_path):
    sources = module('literature_sources')
    original = tmp_path/'source.html'
    original.write_text('<html><p id="p-2">Example 2. Product <b>isolated</b>.</p></html>')
    result = sources.snapshot_primary_document(original.as_uri(),tmp_path/'cache')
    assert result['raw_document_sha256'] == sources.file_sha256(original)
    text = __import__('pathlib').Path(result['document_path']).read_text()
    assert text == 'Example 2. Product isolated .'
    assert result['document_sha256'] == hashlib.sha256(text.encode()).hexdigest()
    original.write_text('changed remote source')
    cached = sources.snapshot_primary_document(original.as_uri(),tmp_path/'cache')
    assert cached == result


def test_real_ord_partial_stereo_failure_preserves_exact_reaction():
    keys = module('literature_reaction_keys')
    product = r'[Br-:20].[OH:23][CH2:21][CH2:22][N+:18]1[CH:17]=[CH:16][N:15]([CH3:19])[C:14]=1/[CH:13]=[N:12]\[N:11]=[C:3]1\[S:4][C:5]2[CH:10]=[CH:9][CH:8]=[CH:7][C:6]=2[N:2]\1[CH3:1]'
    precursor = r'[CH3:1][N:2]1[C:6]2[CH:7]=[CH:8][CH:9]=[CH:10][C:5]=2[S:4]/[C:3]/1=[N:11]/[N:12]=[CH:13][C:14]1[N:15]([CH3:19])[CH:16]=[CH:17][N:18]=1.[Br:20][CH:21]([OH:23])[CH3:22]'
    result = keys.make_reaction(product,[precursor])
    assert len(result['reaction_key']) == 64
    assert len(result['product_components']) == 2
    assert result['reaction_family_key'] is None
    assert result['mapping_status'] == 'mapping_unknown'
    assert result['family_error'] == 'local_stereochemical_fragment_serialization_failed'


def test_ord_alternate_representations_do_not_consume_independent_source_slots(tmp_path):
    index = module('literature_index').ReactionIndex(tmp_path/'i.sqlite')
    for suffix in ['outcome0:roles','mapped_identifier']:
        index.add({'source_id':'ORD:d:r:'+suffix,'source_corpus':'Open Reaction Database',
                   'dataset_id':'d','ord_reaction_id':'r','license':'CC-BY-SA-4.0',
                   'product':'CO','precursors':['CBr','[OH-]']})
    hits = index.query(module('literature_reaction_keys').make_reaction('CO',['CBr','[OH-]']))
    assert len(hits) == 1
    assert hits[0]['source_family_key'] == 'ORD:d:r'
    assert set(hits[0]['source_representation_ids']) == {'ORD:d:r:outcome0:roles','ORD:d:r:mapped_identifier'}
    index.close()


def test_route_edges_cannot_be_borrowed_from_another_target_or_method():
    predictions = module('literature_predictions')
    keys = module('literature_reaction_keys')
    recorded = keys.make_reaction('CO',['CBr','[OH-]'])
    route = {'complete_reaction_edges':[{'product':'CBr','precursors':['C','BrBr']}]}
    with pytest.raises(ValueError,match='current report'):
        predictions.route_reaction_keys(route,[recorded])


def test_completed_index_rerun_preserves_manifest_and_database(tmp_path):
    import json,pathlib,subprocess,sys,importlib.util
    sources=module('literature_sources')
    source=tmp_path/'source.json';source.write_text(json.dumps({'download_complete':True,'files':[]}))
    codec=tmp_path/'codec.py';codec.write_text('# fixture codec never loaded for completed output\n')
    root=tmp_path/'index';root.mkdir()
    index=module('literature_index').ReactionIndex(root/'reactions.sqlite');index.close()
    script=pathlib.Path(__file__).resolve().parents[1]/'scripts/build_nmi_literature_index.py'
    spec=importlib.util.spec_from_file_location('lgrv_cache_identity_unit',script)
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    manifest={'complete':True,'source_manifest_sha256':sources.file_sha256(source),
              'codec_sha256':sources.file_sha256(codec),'implementation_id':'lgrv_index_v1_1',
              'implementation':builder.implementation_identity(),
              'index_sha256':sources.file_sha256(root/'reactions.sqlite'),'files':[]}
    path=root/'source_corpus_manifest.json';path.write_text(json.dumps(manifest));original=path.read_bytes()
    command=[sys.executable,str(script),'--source-manifest',str(source),'--codec-file',str(codec),
        '--codec-sha256',sources.file_sha256(codec),'--codec-source-revision','fixture',
        '--output-root',str(root),'--workers','1']
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
    assert path.read_bytes() == original
    assert sources.file_sha256(root/'reactions.sqlite') == manifest['index_sha256']
    manifest.pop('implementation');path.write_text(json.dumps(manifest));missing_identity=path.read_bytes()
    rejected=subprocess.run(command,capture_output=True,text=True)
    assert rejected.returncode != 0
    assert path.read_bytes() == missing_identity


@pytest.mark.parametrize('changes',[{'raw_document_path':'/nonexistent/primary-document'},
                                  {'raw_document_sha256':'0'*64}, {'source_receipt_path':''},
                                  {'source_url':'https://example.org/unrelated-source'}])
def test_original_byte_receipt_and_url_are_required_for_verified_source(tmp_path,changes):
    reaction=module('literature_reaction_keys').make_reaction('CO',['CBr','[OH-]'])
    result=module('literature_evidence').assess(reaction,[],[primary(tmp_path,**changes)])
    assert result['tier'] == 'C'


def test_partial_index_merge_resumes_without_duplicating_committed_shard(tmp_path):
    import importlib.util,pathlib
    script=pathlib.Path(__file__).resolve().parents[1]/'scripts/build_nmi_literature_index.py'
    spec=importlib.util.spec_from_file_location('lgrv_builder_unit',script)
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    idx=module('literature_index');sources=module('literature_sources')
    shard=tmp_path/'shard.sqlite';source=idx.ReactionIndex(shard)
    source.add({'source_id':'s','product':'CO','precursors':['CBr','[OH-]'],'license':'CC-BY-SA-4.0'});source.close()
    root=tmp_path/'merged';root.mkdir()
    partial=idx.ReactionIndex(root/'reactions.building.sqlite')
    partial.connection.execute('ATTACH DATABASE ? AS source',(str(shard),))
    partial.connection.execute('INSERT INTO reactions SELECT * FROM source.reactions');partial.commit()
    partial.connection.execute('DETACH DATABASE source')
    partial.connection.execute('CREATE TABLE merged_shards(path TEXT PRIMARY KEY,sha256 TEXT NOT NULL)')
    partial.connection.execute('INSERT INTO merged_shards VALUES (?,?)',(str(shard),sources.file_sha256(shard)));partial.close()
    result=builder.merge_shards([{'index_path':str(shard),'index_sha256':sources.file_sha256(shard)}],root)
    assert result['record_count'] == 1
    assert (root/'reactions.sqlite').exists()
    assert not (root/'reactions.building.sqlite').exists()
    assert result['index_sha256'] == sources.file_sha256(root/'reactions.sqlite')


def test_index_worker_failure_is_saved_as_failure_not_silent_completion(tmp_path):
    import json,pathlib,subprocess,sys
    pa=pytest.importorskip('pyarrow');pq=pytest.importorskip('pyarrow.parquet')
    source=tmp_path/'bad_codec_input.parquet'
    schema=pa.schema([('reaction_id',pa.string()),('reaction',pa.binary())],metadata={b'ord.dataset_id':b'fixture'})
    pq.write_table(pa.Table.from_pylist([{'reaction_id':'r','reaction':b'fixture'}],schema=schema),source)
    sha=module('literature_sources').file_sha256
    source_manifest=tmp_path/'source.json'
    source_manifest.write_text(json.dumps({'download_complete':True,'files':[{'path':source.name,'sha256':sha(source)}]}))
    codec=tmp_path/'codec.py';codec.write_text('# Deliberately invalid unit-test codec: no Reaction class.\n')
    script=pathlib.Path(__file__).resolve().parents[1]/'scripts/build_nmi_literature_index.py'
    root=tmp_path/'output'
    completed=subprocess.run([sys.executable,str(script),'--source-manifest',str(source_manifest),
        '--codec-file',str(codec),'--codec-sha256',sha(codec),'--codec-source-revision','fixture',
        '--output-root',str(root),'--workers','1'],capture_output=True,text=True)
    assert completed.returncode != 0
    manifest=json.loads((root/'source_corpus_manifest.json').read_text())
    assert manifest['complete'] is False
    assert manifest['failed_sources'][0]['source'] == source.name
    assert 'AttributeError' in manifest['failed_sources'][0]['error']
    assert 'source_index_failed' in completed.stdout
    assert not (root/'reactions.sqlite').exists()


def test_source_index_identity_also_works_for_relative_runpy_entrypoint(monkeypatch):
    import pathlib,runpy
    monkeypatch.chdir(pathlib.Path(__file__).resolve().parents[1])
    builder=runpy.run_path('scripts/build_nmi_literature_index.py')
    identity=builder['implementation_identity']()
    assert identity['id'] == 'lgrv_index_v1_1'
    assert len(identity['producer_sha256']) == 64
