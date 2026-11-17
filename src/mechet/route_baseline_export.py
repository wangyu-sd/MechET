"""Prepared structural baseline adapters; no planner/model execution or electronic proof.

Each molecule/reaction receives an occurrence ID. Equal molecule identities on
separate branches remain separate obligations and may have different reactions.
Stock closure refers to the declared route's leaves; this adapter never prunes a
route at an internal stock molecule or trusts producer ``in_stock`` annotations.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

SCHEMA_VERSION = 'mechet_route_baseline_occurrences_v1'


def _identity(smiles):
    # Mirror the existing route identity's per-component RDKit normalization
    # without importing mechet.__init__ (which eagerly imports Torch helpers).
    from rdkit import Chem
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError('INVALID_SMILES')
    normalized = []
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = True
    for part in smiles.split('.'):
        if not part.strip():
            raise ValueError('INVALID_SMILES')
        mol = Chem.MolFromSmiles(part, params)
        if mol is None:
            raise ValueError('INVALID_SMILES')
        try:
            mol = Chem.RemoveHs(mol)
        except Exception:
            # Same conservative endpoint fallback as proof_program identity.
            pass
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        normalized.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return '.'.join(sorted(normalized))


def _children(node):
    reaction = node.get('product_of')
    children = node.get('children', [])
    if reaction is not None:
        if children:
            raise ValueError('AMBIGUOUS_REFERENCE_SHAPE')
        if not isinstance(reaction, Mapping):
            raise ValueError('AMBIGUOUS_REFERENCE_OR_NODE')
        children = reaction.get('reactants')
        if not children:
            raise ValueError('EMPTY_REACTION')
    elif children:
        if not isinstance(children, list):
            raise ValueError('INVALID_CHILDREN')
        reactions = [child for child in children
                     if isinstance(child, Mapping) and child.get('type') == 'reaction']
        if reactions:
            if len(children) != 1:
                raise ValueError('AMBIGUOUS_REFERENCE_OR_NODE')
            children = reactions[0].get('children')
            if not children:
                raise ValueError('EMPTY_REACTION')
    if not isinstance(children, list):
        raise ValueError('INVALID_CHILDREN')
    return children


def normalize_route_tree(raw):
    """Convert explicit PaRoutes/DMS/RetroCast/bipartite AND trees, strictly.

    Alternative OR reactions must be separate candidates. No chemistry pruning,
    salt/tautomer/charge aliasing, stock-dependent rewriting or mechanism labels
    are introduced. Input annotations are intentionally not treated as evidence.
    """
    if not isinstance(raw, Mapping):
        raise ValueError('INVALID_ROUTE')
    root = raw.get('route', raw)
    if not isinstance(root, Mapping):
        raise ValueError('INVALID_ROUTE')
    root = root.get('target', root)
    molecules, reactions, leaves = [], [], []

    def visit(node, ancestors):
        if not isinstance(node, Mapping) or node.get('type', 'mol') != 'mol':
            raise ValueError('INVALID_MOLECULE_NODE')
        smiles = _identity(node.get('smiles'))
        if smiles in ancestors:
            raise ValueError('ROUTE_CYCLE')
        mid = f'm{len(molecules)}'
        molecules.append({'id': mid, 'smiles': smiles})
        children = _children(node)
        if children:
            reaction = {'id': f'r{len(reactions)}', 'product': mid, 'reactants': []}
            reactions.append(reaction)
            reaction['reactants'] = [visit(child, ancestors | {smiles}) for child in children]
        else:
            leaves.append(mid)
        return mid

    visit(root, frozenset())
    return {'schema_version': SCHEMA_VERSION, 'root': 'm0', 'target': molecules[0]['smiles'],
            'molecules': molecules, 'reactions': reactions, 'leaf_ids': leaves,
            'electronic_replay_verified': False}


def _validated_molecules(route):
    """Reject edited tables that drop, share, disconnect or cycle obligations."""
    try:
        if route['schema_version'] != SCHEMA_VERSION:
            raise ValueError('schema')
        molecules = {node['id']: node['smiles'] for node in route['molecules']}
        if len(molecules) != len(route['molecules']) or not molecules:
            raise ValueError('molecule IDs')
        if any(_identity(smiles) != smiles for smiles in molecules.values()):
            raise ValueError('noncanonical identity')
        if molecules[route['root']] != route['target']:
            raise ValueError('root identity')
        children, incoming, reaction_ids = {}, dict.fromkeys(molecules, 0), set()
        for reaction in route['reactions']:
            product, reactants = reaction['product'], reaction['reactants']
            if product not in molecules or product in children or not reactants:
                raise ValueError('reaction product')
            if reaction['id'] in reaction_ids:
                raise ValueError('reaction IDs')
            reaction_ids.add(reaction['id'])
            children[product] = reactants
            for mid in reactants:
                incoming[mid] += 1
        if incoming[route['root']] != 0 or any(
                count != 1 for mid, count in incoming.items() if mid != route['root']):
            raise ValueError('parent occurrences')
        leaves, seen = [], set()

        def visit(mid, ancestors):
            if mid in seen or molecules[mid] in ancestors:
                raise ValueError('cycle or shared occurrence')
            seen.add(mid)
            if mid in children:
                for child in children[mid]:
                    visit(child, ancestors | {molecules[mid]})
            else:
                leaves.append(mid)

        visit(route['root'], frozenset())
        if seen != set(molecules) or leaves != route['leaf_ids']:
            raise ValueError('disconnected or missing leaves')
        return molecules
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('INVALID_NORMALIZED_ROUTE:' + str(exc)) from exc


def _stock_status(route, membership):
    molecules = _validated_molecules(route)
    missing = [mid for mid in route['leaf_ids'] if molecules[mid] not in membership]
    nonzero = bool(route['reactions'])
    return {'target_in_stock': route['target'] in membership, 'stock_closed': not missing,
            'nonzero_step': nonzero, 'solved': nonzero and not missing,
            'missing_leaf_ids': missing}


def route_stock_status(route, stock):
    """Validate occurrence integrity, then score exact declared-leaf closure."""
    return _stock_status(route, {_identity(smiles) for smiles in stock})


def align_predictions(targets, predictions, stock):
    """Align ID-keyed candidate trees to every frozen target, preserving ranks.

    Unknown/duplicate IDs and target-identity mismatches are contract errors.
    Malformed individual routes retain their rank as invalid candidates. Missing
    predictions and empty lists remain unsolved in the complete denominator.
    """
    if not isinstance(predictions, Mapping):
        raise ValueError('PREDICTIONS_MUST_BE_ID_KEYED')
    ordered = []
    seen = set()
    for target in targets:
        tid = target.get('id')
        if not isinstance(tid, str) or not tid:
            raise ValueError('INVALID_TARGET_ID')
        if tid in seen:
            raise ValueError('DUPLICATE_TARGET_ID:' + tid)
        seen.add(tid)
        ordered.append((tid, _identity(target.get('smiles'))))
    unknown = set(predictions) - seen
    if unknown:
        raise ValueError('UNKNOWN_TARGET_ID:' + repr(sorted(unknown, key=str)))
    # Materialize iterables once; a generator stock must not vanish after target1.
    stock = {_identity(smiles) for smiles in stock}
    rows = []
    for tid, smiles in ordered:
        raw_routes = predictions.get(tid, [])
        if not isinstance(raw_routes, Sequence) or isinstance(raw_routes, (str, bytes)):
            raise ValueError('CANDIDATES_MUST_BE_ORDERED_LIST:' + tid)
        row = {'target_id': tid, 'target': smiles, 'solved': False, 'candidates': [],
               'status': 'missing_prediction' if tid not in predictions else
                         ('predicted' if raw_routes else 'no_routes')}
        for rank, raw in enumerate(raw_routes, 1):
            try:
                route = normalize_route_tree(raw)
            except ValueError as exc:
                row['candidates'].append({'rank': rank, 'status': 'invalid_route', 'error': str(exc)})
                continue
            if route['target'] != smiles:
                raise ValueError('TARGET_MISMATCH:' + tid)
            status = _stock_status(route, stock)
            row['candidates'].append({'rank': rank, 'status': 'normalized', 'route': route, 'stock': status})
            row['solved'] = row['solved'] or status['solved']
        rows.append(row)
    return {'schema_version': SCHEMA_VERSION, 'n_targets': len(rows),
            'n_solved': sum(row['solved'] for row in rows), 'targets': rows,
            'evidence': 'structural_source_stock_only_not_electronic_or_chemical_verification'}


def normalize_syntheseus_route(graph, route_nodes, target_smiles):
    """Bridge selected AndOrGraph/MolSetGraph routes via their common public API.

    Both source classes implement ``to_synthesis_graph(nodes)``. Only this
    duck-typed reaction graph is inspected; importing Syntheseus, Torch or a
    checkpoint is unnecessary. This function neither extracts routes nor runs
    search. Repeated reactant Bag entries are expanded as distinct occurrences.
    """
    synthesis = graph.to_synthesis_graph(route_nodes)
    root = synthesis.root_node
    target = _identity(target_smiles)
    if _identity(root.product.smiles) != target:
        raise ValueError('TARGET_MISMATCH')
    visited = set()

    def expand(reaction, ancestors):
        product = _identity(reaction.product.smiles)
        if product in ancestors:
            raise ValueError('ROUTE_CYCLE')
        visited.add(id(reaction))
        children = list(synthesis.successors(reaction))
        by_product = {}
        for child in children:
            identity = _identity(child.product.smiles)
            if identity in by_product:
                raise ValueError('AMBIGUOUS_REFERENCE_OR_NODE')
            by_product[identity] = child
        reactants = list(reaction.reactants)
        if not reactants:
            raise ValueError('EMPTY_REACTION')
        identities = [_identity(mol.smiles) for mol in reactants]
        if set(by_product) - set(identities):
            raise ValueError('DISCONNECTED_REACTION')
        result = []
        for mol, identity in zip(reactants, identities):
            if identity in by_product:
                result.append(expand(by_product[identity], ancestors | {product}))
            else:
                result.append({'smiles': mol.smiles})
        return {'smiles': reaction.product.smiles, 'children': result}

    tree = expand(root, frozenset())
    if visited != {id(reaction) for reaction in synthesis.nodes()}:
        raise ValueError('DISCONNECTED_REACTION')
    return normalize_route_tree(tree)
