"""Contract tests with toy mapped executor records; no teacher inference."""
import json
from copy import deepcopy
from hashlib import sha256

import pytest
from rdkit import Chem

from mechet.route_joint_inventory import extract_forward_input


def receipt_fixture():
    initial = '[H:41][O:20][CH3:10]'
    imported = initial + '.[Cl-:99].[Na+:165]'
    final = '[H:41][O-:20].[CH3:10][Cl:99].[Na+:165]'
    steps = [
        ('import_fragments', {'fragments': [{'smiles': '[Cl-]', 'count': 1},
                                            {'smiles': '[Na+]', 'count': 1}]}, initial, imported),
        ('apply_electron_flow', {'direction': 'retro'}, imported, final),
        ('finish_trace', {}, final, final),
    ]
    actions, records = [], []
    for index, (name, args, before, after) in enumerate(steps):
        actions.append(dict(name=name, arguments=args, state_before=before,
                            result={'ok': True, 'code': 'PASS'}))
        records.append(dict(
            accepted=True, request_id=f'episode/r0/a{index}', occurrence='0', reaction_id='episode/r0',
            behavior_version='policy-A', raw=json.dumps(dict(name=name, arguments=args)),
            state_before=before, state_after=after, error='', truncated=False,
        ))
    edge = dict(initial_state=initial, final_state=final, public_product='CO', occurrence='0',
                public_precursors=['deliberately-unused'], precursors=['deliberately-unused'],
                actions=actions)
    return dict(behavior_version='policy-A', reactions=[edge], records=records)


def test_full_inventory_preserves_graph_h_charge_import_auxiliary_and_map_scope():
    report = receipt_fixture()
    report['reward_evidence'] = {'reference_product': 'deliberately-unused'}
    snapshot = deepcopy(report)
    out = extract_forward_input(report, 0)
    assert out['status'] == 'eligible', out
    assert out['target_product'] == 'CO'
    assert out['finish_index'] == 2 and out['finish_request_id'] == 'episode/r0/a2'
    prm = Chem.SmilesParserParams()
    prm.removeHs = False
    full = Chem.MolFromSmiles(out['full_reactants_mapped'], prm)
    assert len(Chem.GetMolFrags(full)) == 3
    assert {a.GetAtomMapNum() for a in full.GetAtoms()} == {10, 20, 41, 99, 165}
    teacher = Chem.MolFromSmiles(out['teacher_reactants_mapped'], prm)
    assert sorted(a.GetAtomMapNum() for a in teacher.GetAtoms()) == list(range(1, 9))
    lut = out['executor_to_teacher_map']
    assert sorted(out['target_heavy_teacher_maps']) == sorted([lut[10], lut[20]])
    assert len(out['generated_hydrogen_parents']) == 3
    assert set(out['generated_hydrogen_parents'].values()) == {10}
    assert out['provenance']['imported_executor_maps'] == [99, 165]
    assert out['provenance']['final_state_sha256'] == sha256(report['reactions'][0]['final_state'].encode()).hexdigest()
    assert report == snapshot


@pytest.mark.parametrize('field,value', [
    ('accepted', False), ('error', 'rejected'), ('raw', '{"name":"apply_electron_flow"}'),
    ('state_before', 'tampered'), ('state_after', 'tampered'), ('truncated', True),
    ('occurrence', 'different'), ('behavior_version','policy-B'),
])
def test_rejected_or_unbound_finish_has_no_teacher_input(field, value):
    report = receipt_fixture()
    report['records'][-1][field] = value
    out = extract_forward_input(report, 0)
    assert out['status'] == 'unavailable' and 'teacher_reactants_mapped' not in out


@pytest.mark.parametrize('mistake', ['duplicate', 'lost', 'new', 'isotope', 'charge', 'import_count',
                                     'input_target', 'uncounted', 'duplicate_edge', 'net_noop'])
def test_corrupt_or_unsupported_executor_evidence_fails_closed(mistake):
    r = receipt_fixture()
    edge = r['reactions'][0]
    if mistake in ('duplicate','lost','new','isotope','charge','net_noop'):
        replacement = {'duplicate': '[Na+:41]', 'lost': '', 'new': '[Cl-:55]',
                       'charge': '[Na:165]', 'isotope': '[23Na+:165]'}
        value = (r['records'][1]['state_before'] if mistake == 'net_noop' else
                 edge['final_state'].replace('.[Na+:165]', '.' + replacement[mistake] if replacement[mistake] else ''))
        edge['final_state'] = edge['actions'][-1]['state_before'] = value
        r['records'][1]['state_after'] = value
        r['records'][2]['state_before'] = r['records'][2]['state_after'] = value
    elif mistake == 'import_count':
        args = edge['actions'][0]['arguments']
        args['fragments'][0]['count'] = 2
        r['records'][0]['raw'] = json.dumps({'name': 'import_fragments', 'arguments': args})
    elif mistake == 'input_target':
        edge['public_product'] = 'CC'
    elif mistake == 'uncounted':
        del edge['initial_state']
    else:
        r['reactions'].append(deepcopy(edge))
    out = extract_forward_input(r, 0)
    assert out['status'] == 'unavailable' and out['reason']


@pytest.mark.parametrize('initial,public', [
    ('[13CH3:10][OH:20]', '[13CH3]O'),
    ('[F:1][C@@H:2]([Cl:3])[Br:4]', 'F[C@@H](Cl)Br'),
    ('[F:1]/[CH:2]=[CH:3]/[F:4]', 'F/C=C/F'),
])
def test_nonisomeric_teacher_unsupported_features_are_named(initial, public):
    report = receipt_fixture()
    report['reactions'][0].update(initial_state=initial,public_product=public)
    assert extract_forward_input(report, 0)['reason'] == 'teacher_stereo_or_isotope_unsupported'
