"""Source/normalization unit tests, not model-performance experiments."""
import pytest


def reference():
    return {'content_hash':'source-route','length':2,'target':{'smiles':'CC(O)O','synthesis_step':{
        'mapped_smiles':'[CH3:1][CH:2]=[O:3].[OH2:4]>>[CH3:1][CH:2]([OH:3])[OH:4]',
        'metadata':{'ID':'patent-reaction'},'is_convergent':False,
        'reactants':[{'smiles':'CC=O','synthesis_step':{
            'mapped_smiles':None,'metadata':{},'reactants':[{'smiles':'CC','synthesis_step':None}]}},
            {'smiles':'O','synthesis_step':None}]}}}


def test_reference_depth_and_all_and_branches_are_retained():
    from mechet.frozen_route_assets import extract_reference_route
    route, edges = extract_reference_route(reference(), 'target', 0)
    assert route['depth'] == 2
    assert route['reaction_edge_count'] == 2
    assert route['leaf_smiles'] == ['CC','O']
    assert route['maximum_precursor_multiplicity'] == 2
    assert len(edges) == 2
    assert edges[0]['source_mapping_id'] == 'patent-reaction'
    assert edges[0]['source_precursor_smiles'] == ['CC=O','O']


def test_missing_electron_programs_are_unknown_not_zero_or_primitive_unseen():
    from mechet.frozen_route_assets import extract_reference_route, classify_composition
    route, edges = extract_reference_route(reference(), 'target', 0)
    assert route['reference_event_length'] is None
    assert route['coupled_electronic_moves'] is None
    assert edges[0]['annotation_status'] == 'mapping_only'
    assert edges[1]['annotation_status'] == 'no_verified_match'
    assert classify_composition(edges, set(), {})['stratum'] == 'Unknown'


def test_verified_composition_classification_obeys_complete_training_support():
    from mechet.frozen_route_assets import classify_composition
    edge = {'annotation_status':'verified_mechanism','program_signature':'p','primitive_signatures':['a']}
    assert classify_composition([edge], {'p'}, {'a':20})['stratum'] == 'C0'
    assert classify_composition([edge], set(), {'a':5})['stratum'] == 'C1'
    assert classify_composition([edge], set(), {'a':0})['stratum'] == 'C2'
    assert classify_composition([edge], set(), {'a':2})['stratum'] == 'Unknown'


def test_stock_canonicalization_preserves_charge_stereo_and_salt_units():
    from mechet.frozen_route_assets import normalize_stock_smiles
    assert normalize_stock_smiles('[CH3:1][OH:2]') == 'CO'
    assert normalize_stock_smiles('O.[Na+]') == 'O.[Na+]'
    assert normalize_stock_smiles('[O-]') != normalize_stock_smiles('O')
    assert normalize_stock_smiles('F[C@H](Cl)Br') != normalize_stock_smiles('F[C@@H](Cl)Br')


def test_corrupt_source_checksum_stops_before_constructing_targets(tmp_path):
    from mechet.frozen_route_assets import verify_source
    path = tmp_path/'source'
    path.write_text('changed source')
    with pytest.raises(ValueError, match='SHA256'):
        verify_source(path, '0'*64)


def test_ring_change_and_mapped_scope_are_not_fabricated_electron_ground_truth():
    from mechet.frozen_route_assets import extract_reference_route
    _, edges = extract_reference_route(reference(), 'target', 0)
    assert edges[0]['changed_bonds_shared_mapped_atoms'] == 2
    assert edges[0]['mapped_reacting_atom_count_shared'] == 3
    assert edges[0]['program_signature'] is None
    assert edges[1]['changed_bonds_shared_mapped_atoms'] is None


def test_freeze_keeps_full_cohort_and_reports_zero_step_stock_separately(tmp_path):
    import hashlib,json
    from scripts.prepare_frozen_route_benchmarks import freeze_assets
    stock = tmp_path/'stock.csv'
    stock.write_text('smiles\nCO\nO\n')
    targets = tmp_path/'targets.csv'
    targets.write_text('id,smiles\nstock_target,CO\nunresolved,CCO\n')
    config = {'raw_root':str(tmp_path), 'stocks':{'s':{'path':'stock.csv','rows':2,'smiles_column':'smiles',
        'sha256':hashlib.sha256(stock.read_bytes()).hexdigest()}}, 'cohorts':{'c':{
        'path':'targets.csv','format':'csv','rows':2,'smiles_column':'smiles','id_column':'id','stock':'s',
        'sha256':hashlib.sha256(targets.read_bytes()).hexdigest()}}}
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    output = tmp_path/'frozen'
    manifest = freeze_assets(path,output,workers=1,normalize_stocks=True)
    rows = [json.loads(line) for line in (output/'c/targets.jsonl').read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]['already_in_stock'] is True
    assert rows[1]['already_in_stock'] is False
    assert rows[0]['reference_routes'] == []
    assert rows[0]['reference_depth'] is None
    assert manifest['cohorts']['c']['rows'] == 2
    assert manifest['cohorts']['c']['target_ids'] == ['c:stock_target','c:unresolved']
