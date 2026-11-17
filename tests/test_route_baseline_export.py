"""Software-only occurrence/identity tests; no planner or neural model execution."""
from types import SimpleNamespace

import pytest


def mol(smiles, children=None):
    result = {'type': 'mol', 'smiles': smiles}
    if children is not None:
        result['children'] = [{'type': 'reaction', 'children': children}]
    return result


def api():
    # Missing module is an explicit assertion failure in the initial RED run.
    import importlib.util
    assert importlib.util.find_spec('mechet.route_baseline_export') is not None, 'baseline adapter missing'
    from mechet import route_baseline_export
    return route_baseline_export


def test_duplicate_reactants_have_distinct_occurrences_and_leaf_obligations():
    result = api().normalize_route_tree(mol('CC', [mol('C'), mol('C')]))
    assert [v['smiles'] for v in result['molecules']] == ['CC', 'C', 'C']
    assert result['reactions'][0]['reactants'] == ['m1', 'm2']
    assert result['leaf_ids'] == ['m1', 'm2']
    assert result['electronic_replay_verified'] is False


def test_nested_identical_intermediates_keep_their_distinct_reaction_choices():
    raw = mol('CCCC', [mol('CC', [mol('C'), mol('C')]), mol('CC', [mol('[CH3-]'), mol('[CH3+]')])])
    result = api().normalize_route_tree(raw)
    assert len(result['reactions']) == 3
    assert [v['smiles'] for v in result['molecules']].count('CC') == 2
    assert result['reactions'][0]['reactants'] == ['m1', 'm4']
    assert api().route_stock_status(result, {'C', '[CH3-]', '[CH3+]'})['solved'] is True


@pytest.mark.parametrize('raw', [
    {'target': {'smiles': 'CO', 'product_of': {'reactants': [{'smiles': 'C'}, {'smiles': 'O'}]}}},
    {'route': {'target': {'smiles': 'CO', 'product_of': {'reactants': [{'smiles': 'C'}, {'smiles': 'O'}]}}}},
    {'smiles': 'CO', 'children': [{'smiles': 'C'}, {'smiles': 'O'}]},
    mol('CO', [mol('C'), mol('O')]),
])
def test_explicit_source_tree_shapes_normalize_to_same_and_obligations(raw):
    result = api().normalize_route_tree(raw)
    assert result['target'] == 'CO'
    assert [v['smiles'] for v in result['molecules']] == ['CO', 'C', 'O']
    assert result['reactions'] == [{'id': 'r0', 'product': 'm0', 'reactants': ['m1', 'm2']}]


def test_identity_removes_maps_but_retains_charge_isotope_and_stereo():
    a = api()
    assert a.normalize_route_tree(mol('[CH3:1][OH:2]'))['target'] == 'CO'
    assert a.normalize_route_tree(mol('[13CH3]O'))['target'] == '[13CH3]O'
    assert a.normalize_route_tree(mol('C[S-]'))['target'] == 'C[S-]'
    left = a.normalize_route_tree(mol('F[C@H](Cl)Br'))
    right = a.normalize_route_tree(mol('F[C@@H](Cl)Br'))
    assert left['target'] != right['target']
    assert a.route_stock_status(left, {right['target']})['stock_closed'] is False


def test_stock_membership_ignores_untrusted_native_in_stock_annotations():
    tree = mol('CO', [dict(mol('C'), in_stock=True), dict(mol('O'), in_stock=True)])
    status = api().route_stock_status(api().normalize_route_tree(tree), {'C'})
    assert status['stock_closed'] is False
    assert status['missing_leaf_ids'] == ['m2']
    assert status['solved'] is False


def test_zero_step_stock_target_is_closed_but_not_solved():
    a = api()
    status = a.route_stock_status(a.normalize_route_tree(mol('CO')), {'CO'})
    assert status == {'target_in_stock': True, 'stock_closed': True, 'nonzero_step': False,
                      'solved': False, 'missing_leaf_ids': []}
    expanded = a.normalize_route_tree(mol('CO', [mol('C'), mol('O')]))
    # Declared route leaves remain obligations even if its root is purchasable.
    assert a.route_stock_status(expanded, {'CO', 'C'})['solved'] is False


@pytest.mark.parametrize('raw,code', [
    (mol('CC', [mol('CC')]), 'ROUTE_CYCLE'),
    ({'smiles': 'CO', 'children': [{'type': 'reaction', 'children': [mol('C')]}, {'type': 'reaction', 'children': [mol('O')]}]}, 'AMBIGUOUS'),
    ({'smiles': 'CO', 'product_of': {'reactants': []}}, 'EMPTY_REACTION'),
    (mol('not valid smiles'), 'INVALID_SMILES'),
    ({'smiles': 'CO', 'product_of': {'reactants': [mol('C')]}, 'children': [mol('O')]}, 'AMBIGUOUS'),
])
def test_invalid_or_ambiguous_route_cannot_be_silently_pruned(raw, code):
    with pytest.raises(ValueError, match=code):
        api().normalize_route_tree(raw)


def test_target_alignment_retains_missing_and_empty_predictions_in_order():
    targets = [{'id': 'b', 'smiles': 'CO'}, {'id': 'a', 'smiles': 'CN'}, {'id': 'c', 'smiles': 'CC'}]
    result = api().align_predictions(targets, {'a': [], 'b': [mol('CO', [mol('C'), mol('O')])]}, {'C', 'O'})
    assert result['n_targets'] == 3 and result['n_solved'] == 1
    assert [v['target_id'] for v in result['targets']] == ['b', 'a', 'c']
    assert [v['status'] for v in result['targets']] == ['predicted', 'no_routes', 'missing_prediction']
    assert result['targets'][2]['solved'] is False


@pytest.mark.parametrize('targets,predictions,code', [
    ([{'id': 'a', 'smiles': 'CO'}], {'other': []}, 'UNKNOWN_TARGET_ID'),
    ([{'id': 'a', 'smiles': 'CO'}, {'id': 'a', 'smiles': 'CN'}], {}, 'DUPLICATE_TARGET_ID'),
    ([{'id': 'a', 'smiles': 'CO'}], {'a': [mol('CN')]}, 'TARGET_MISMATCH'),
])
def test_strict_target_id_alignment_rejects_bad_identity_assignments(targets, predictions, code):
    with pytest.raises(ValueError, match=code):
        api().align_predictions(targets, predictions, {'C', 'O'})


def test_invalid_candidate_keeps_rank_and_target_denominator():
    result = api().align_predictions([{'id': 'a', 'smiles': 'CO'}],
                                     {'a': [mol('CO', [mol('CO')]), mol('CO', [mol('C'), mol('O')])]}, {'C', 'O'})
    candidates = result['targets'][0]['candidates']
    assert [v['rank'] for v in candidates] == [1, 2]
    assert candidates[0]['status'] == 'invalid_route'
    assert 'ROUTE_CYCLE' in candidates[0]['error']
    assert candidates[1]['stock']['solved'] is True
    assert result['n_targets'] == 1 and result['n_solved'] == 1


class Reaction:
    def __init__(self, product, reactants):
        self.product = SimpleNamespace(smiles=product)
        # A real Syntheseus Bag is iterable and preserves duplicates.
        self.reactants = [SimpleNamespace(smiles=v) for v in reactants]


class SynthesisGraph:
    def __init__(self):
        self.root_node = Reaction('CCO', ['CC', 'CC', 'O'])
        self.child = Reaction('CC', ['C', 'C'])

    def nodes(self):
        return [self.root_node, self.child]

    def successors(self, reaction):
        return [self.child] if reaction is self.root_node else []


class AndOrGraph:
    root_node = SimpleNamespace(mol=SimpleNamespace(smiles='CCO'))

    def to_synthesis_graph(self, nodes):
        if set(nodes) != {'selected-root', 'selected-child'}:
            raise ValueError('wrong selected subgraph')
        return SynthesisGraph()


class MolSetGraph(AndOrGraph):
    root_node = SimpleNamespace(mols=frozenset({'CCO'}))


@pytest.mark.parametrize('graph', [AndOrGraph(), MolSetGraph()])
def test_graph_bridge_uses_selected_synthesis_graph_and_reinflates_bag_occurrences(graph):
    result = api().normalize_syntheseus_route(graph, ['selected-root', 'selected-child'], 'CCO')
    assert len(result['reactions']) == 3  # repeated CC obligation expands independently
    assert [v['smiles'] for v in result['molecules']].count('C') == 4
    assert api().route_stock_status(result, {'C', 'O'})['solved'] is True


def test_graph_bridge_rejects_wrong_target_and_disconnected_reactions():
    with pytest.raises(ValueError, match='TARGET_MISMATCH'):
        api().normalize_syntheseus_route(AndOrGraph(), ['selected-root', 'selected-child'], 'CO')
    graph = SynthesisGraph()
    extra = Reaction('N', ['[NH2-]'])
    graph.nodes = lambda: [graph.root_node, graph.child, extra]
    parent = SimpleNamespace(to_synthesis_graph=lambda nodes: graph)
    with pytest.raises(ValueError, match='DISCONNECTED'):
        api().normalize_syntheseus_route(parent, [], 'CCO')


@pytest.mark.parametrize('change', ['missing_leaf', 'orphan_molecule', 'shared_occurrence'])
def test_forged_normalized_tables_cannot_claim_stock_closure(change):
    a = api()
    result = a.normalize_route_tree(mol('CO', [mol('C'), mol('O')]))
    if change == 'missing_leaf':
        result['leaf_ids'] = []
    elif change == 'orphan_molecule':
        result['molecules'].append({'id': 'unused', 'smiles': 'N'})
    else:
        result['reactions'][0]['reactants'] = ['m1', 'm1']
    with pytest.raises(ValueError, match='NORMALIZED_ROUTE'):
        a.route_stock_status(result, {'C', 'O'})


def test_generator_stock_remains_available_for_every_target():
    result = api().align_predictions([{'id': 'a', 'smiles': 'CO'}, {'id': 'b', 'smiles': 'CC'}],
                                     {'a': [mol('CO', [mol('C'), mol('O')])],
                                      'b': [mol('CC', [mol('C'), mol('C')])]},
                                     iter(['C', 'O']))
    assert result['n_targets'] == 2 and result['n_solved'] == 2


def test_standalone_adapter_runs_with_neural_and_planner_imports_blocked():
    import subprocess
    import sys
    from pathlib import Path
    module = Path(__file__).resolve().parents[1] / 'src/mechet/route_baseline_export.py'
    source = '''import builtins, importlib.util, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'retrochimera', 'syntheseus', 'aizynthfinder'}:
        raise ImportError('NEURAL_IMPORT_FORBIDDEN:' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
spec = importlib.util.spec_from_file_location('standalone_baseline_adapter', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
route = module.normalize_route_tree({'smiles': '[CH3:1][OH:2]', 'children': [{'smiles': 'C'}, {'smiles': 'O'}]})
assert route['target'] == 'CO'
assert module.route_stock_status(route, {'C', 'O'})['solved']
print('standalone chemistry-only PASS')
'''
    result = subprocess.run([sys.executable, '-c', source, str(module)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'standalone chemistry-only PASS'
