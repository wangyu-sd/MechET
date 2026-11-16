"""Material availability is a declared workup convention, not electron execution."""
from copy import deepcopy
import json
import sqlite3

import pytest

EXACT = 'exact_material_stock_v1'
PROTIC = 'terminal_heteroanion_protonation_stock_v1'
PRODUCT = 'CSCCN1C(=O)c2ccccc2C1=O'
BROMIDE = 'O=C1c2ccccc2C(=O)N1CCBr'


def project(smiles, stock, contract=PROTIC):
    from mechet.route_materials import project_material_endpoints
    return project_material_endpoints(smiles, stock, contract)


def test_thiolate_is_available_as_exact_stock_thiol_without_mutating_mechanism():
    result = project('C[S-].'+BROMIDE, {'CS', BROMIDE})
    assert result['precursors'] == ['CS', BROMIDE]
    assert result['materials'] == [
        dict(mechanistic='C[S-]', material='CS', rule='terminal_S_minus_add_H'),
        dict(mechanistic=BROMIDE, material=BROMIDE, rule='exact')]


@pytest.mark.parametrize('smiles,stock,expected', [
    ('C[S-]', {'CS'}, 'C[S-]'),
    ('C[S-]', {'C[S-]', 'CS'}, 'C[S-]'),
    ('C[S-]', set(), 'C[S-]'),
])
def test_exact_contract_and_exact_stock_priority(smiles, stock, expected):
    contract = EXACT if expected=='C[S-]' and stock=={'CS'} else PROTIC
    assert project(smiles, stock, contract)['precursors'] == [expected]


@pytest.mark.parametrize('active,material', [
    ('C[O-]', 'CO'), ('CC(=O)[O-]', 'CC(=O)O'),
    ('[13CH3][S-]', '[13CH3]S'), ('[2H]C([2H])([2H])[S-]', '[2H]C([2H])([2H])S'),
    ('C[C@H](F)C[O-]', 'C[C@H](F)CO'),
])
def test_single_proton_convention_preserves_stereo_and_isotopes(active, material):
    assert project(active, {material})['precursors'] == [material]


@pytest.mark.parametrize('active,tempting', [
    ('[CH2-]C', 'CC'), ('C[N-]C', 'CNC'), ('C[NH3+]', 'CN'),
    ('C[N+](C)(C)C', 'CN(C)C'), ('[CH2][O-]', 'CO'),
    ('[O-]CC[O-]', 'OCCO'), ('[Na+]', '[Na]'),
    ('C[S-]', '[13CH3]S'),
    ('C[S-]', 'C[SH+]'), ('C[O-]', 'C=O'),
    ('F/C=C/C[O-]', 'F/C=C\\CO'),
])
def test_no_generic_uncharging_tautomerization_or_salt_stripping(active, tempting):
    from mechet.route_training_contract import chemical_identity
    result = project(active, {tempting})
    assert tempting not in result['precursors']
    assert sorted(result['precursors']) == sorted(chemical_identity(active).split('.'))


def test_whole_stock_salt_priority_and_duplicate_occurrences_are_retained():
    assert project('C[S-].[Na+]', {'C[S-].[Na+]', 'CS'})['precursors'] == ['C[S-].[Na+]']
    assert project('C[S-].C[S-]', {'CS'})['precursors'] == ['CS', 'CS']
    assert project('C[S-].C[S-]', {'CS', 'CS.CS'})['precursors'] == ['CS', 'CS']
    assert project('C[S-].[Na+]', {'CS'})['precursors'] == ['CS', '[Na+]']


def test_unknown_material_contract_fails_closed():
    with pytest.raises(ValueError, match='MATERIAL_CONTRACT'):
        project('C[S-]', {'CS'}, 'unversioned_neutralization')


def fixture_edge():
    # Real inverse substitution endpoint; atom maps distinguish contributors
    # from sodium/water/bromide auxiliaries.
    from mechet.route_policy_episode import NativeChemistry
    chemistry = NativeChemistry()
    initial = chemistry.root('CSC')
    from rdkit import Chem
    mol = Chem.MolFromSmiles(initial.state)
    sulfur = next(a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetSymbol()=='S')
    carbons = sorted(a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetSymbol()=='C')
    final = f'[CH3:{carbons[0]}][S-:{sulfur}].[CH3:{carbons[1]}][Br:4]'
    resolved = project('C[S-].CBr', {'CS','CBr'})
    return dict(product='CSC',precursors=resolved['precursors'],
                occurrence='0',depth=1,initial_state=initial.state,final_state=final,
                endpoint_materials=resolved['materials'])


def reward_database(tmp_path):
    from mechet.external_route_data import edge_signature
    path = tmp_path/'refs.sqlite'
    with sqlite3.connect(path) as db:
        db.executescript('CREATE TABLE tasks(id TEXT,target TEXT,split TEXT,reference_steps INT);'
                         'CREATE TABLE edges(task_id TEXT,signature TEXT);')
        db.execute('INSERT INTO tasks VALUES(?,?,?,?)',('t','CSC','training',1))
        db.execute('INSERT INTO edges VALUES(?,?)',('t',edge_signature('CSC',['CBr','CS'])))
    return path


def test_reward_verifies_projected_materials_but_does_not_upgrade_legacy(tmp_path):
    from mechet.external_route_rl import HiddenRouteReward
    edge = fixture_edge()
    report = dict(root='CSC',status='stock_solved',stock_material_contract=PROTIC,
                  stock_leaves=['CBr','CS'],reactions=[edge])
    with HiddenRouteReward(reward_database(tmp_path), {'CBr','CS'}, material_contract=PROTIC) as oracle:
        score = oracle.score('t', report, include_progress=True)
        assert score['verified_stock_closed'] and score['reward']==1.
        assert score['supported_reactions']==1
        forged = deepcopy(report); forged['reactions'][0]['final_state']='[CH4:1]'
        with pytest.raises(ValueError, match='MATERIAL'):
            oracle.score('t', forged)
        with pytest.raises(ValueError, match='MATERIAL_CONTRACT'):
            oracle.score('t', dict(report,stock_material_contract=EXACT))
    with HiddenRouteReward(tmp_path/'refs.sqlite', {'CBr','CS'}) as legacy:
        with pytest.raises(ValueError, match='MATERIAL_CONTRACT'):
            legacy.score('t', report)


def test_certificate_recomputes_mapping_and_rejects_forged_evidence():
    from scripts.eval_pr98_route_policy import validate_route_certificate
    edge = fixture_edge()
    report = dict(root='CSC',status='stock_solved',stock_material_contract=PROTIC,
                  stock_leaves=['CBr','CS'],reactions=[edge],depth=1,pending=[])
    assert validate_route_certificate(report, {'CBr','CS'})['reaction_count']==1
    report['reactions'][0]['endpoint_materials'][1]['mechanistic']='CN'
    with pytest.raises(ValueError, match='MATERIAL'):
        validate_route_certificate(report, {'CBr','CS'})


def substitution_episode(observation='route_sft_aligned_feedback_v9'):
    from mechet.external_route_rl import TargetOnlyRouteEpisode
    from test_route_native_observation import send
    from mechet.natural_language_electron_flow import build_inventory
    from rdkit import Chem
    cfg = dict(behavior_version='material-test', public_observation_contract=observation,
               stock_material_contract=PROTIC,max_consecutive_recoveries=8,max_action_tokens=2048)
    ep = TargetOnlyRouteEpisode(cfg,'CSC',{'CS','CBr'},'material',17)
    ep.next_decision()
    send(ep,'import_fragments',dict(fragments=[dict(smiles='[Br-]',count=1)]))
    inv = build_inventory(ep.node.state)
    mol = Chem.MolFromSmiles(ep.node.state)
    smap = next(a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetSymbol()=='S')
    cmap = min(a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetSymbol()=='C')
    bmap = next(a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetSymbol()=='Br')
    s,c,b = [inv.map_to_atom[m] for m in (smap,cmap,bmap)]
    send(ep,'apply_electron_flow',dict(direction='retrosynthetic',electron_flow=[
        dict(source=f'the bond between atoms {s} and {c}',destination='atom '+s,instruction='heterolytic cleavage'),
        dict(source='a lone pair on atom '+b,destination=f'the bond to form between atoms {b} and {c}',instruction='bond formation')],
        bond_order_changes=[],charge_changes=[]))
    assert ep.records[-1]['accepted']
    mechanism = ep.node.state
    send(ep,'finish_trace',{})
    assert ep.reactions[0]['final_state']==mechanism
    return ep


@pytest.mark.parametrize('observation',['route_sft_aligned_feedback_v9','full_route_history_feedback_v2'])
def test_native_adapter_closes_active_thiolate_and_restore_is_exact(observation):
    from mechet.external_route_rl import TargetOnlyRouteEpisode
    ep = substitution_episode(observation)
    assert ep.status=='stock_solved'
    report=ep.finish_report()
    assert report['reactions'][0]['precursors']==['CBr','CS']
    assert 'C[S-]' in [m['mechanistic'] for m in report['reactions'][0]['endpoint_materials']]
    restored = TargetOnlyRouteEpisode.restore(json.loads(json.dumps(ep.snapshot())))
    assert restored.finish_report()==report
    from scripts.eval_pr98_route_policy import validate_route_certificate,replay_complete_route
    validate_route_certificate(report,ep.stock)
    replay_complete_route(report)


@pytest.mark.parametrize('observation',['route_sft_aligned_feedback_v9','full_route_history_feedback_v2'])
def test_non_native_material_restore_rejects_forged_material_evidence(observation):
    from mechet.external_route_rl import TargetOnlyRouteEpisode
    ep=substitution_episode(observation)
    forged=json.loads(json.dumps(ep.snapshot()))
    forged['reactions'][0]['endpoint_materials'][0]['mechanistic']='FAKE'
    with pytest.raises(ValueError,match='MATERIAL|SFT_HISTORY'):
        TargetOnlyRouteEpisode.restore(forged)


@pytest.mark.parametrize('mutation',['element','isotope'])
def test_material_validation_preserves_element_and_isotope_per_atom_map(mutation):
    from mechet.route_materials import validate_material_edge
    edge=fixture_edge()
    stock={'CS','CBr'}
    if mutation=='element':
        # Same map set and molecular identities; two atom identities exchanged.
        edge['final_state']='[CH3:3][Br:4].[S-:1][CH3:2]'
    else:
        edge.update(product='[13CH3]SC',initial_state='[13CH3:1][S:2][CH3:3]',
            final_state='[CH3:1][S-:2].[13CH3:3][Br:4]',precursors=['CS','[13CH3]Br'],
            endpoint_materials=[dict(mechanistic='C[S-]',material='CS',rule='terminal_S_minus_add_H'),
                                dict(mechanistic='[13CH3]Br',material='[13CH3]Br',rule='exact')])
        stock={'CS','[13CH3]Br'}
    with pytest.raises(ValueError,match='MATERIAL'):
        validate_material_edge(edge,stock,PROTIC)


def test_material_projection_cannot_turn_acid_base_noop_into_synthesis():
    from mechet.route_policy_episode import RouteEpisode
    from test_route_policy_episode import Chemistry, act
    ep=RouteEpisode(dict(behavior_version='v0',episode_token_budget=100,
        public_observation_contract='full_route_history_feedback_v2',stock_material_contract=PROTIC),
        'CS', set(), 'cycle', 17, chemistry=Chemistry())
    # Simulate a stock entry becoming available before committing. The ancestor
    # check must see the projected child, not just its charged mechanism state.
    ep.next_decision();ep.stock.add('CS')
    act(ep,'apply_electron_flow','C[S-]');act(ep,'finish_trace')
    assert ep.records[-1]['accepted'] is False
    assert not ep.reactions
