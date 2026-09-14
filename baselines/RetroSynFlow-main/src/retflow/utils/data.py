import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_scatter import scatter_max
from torchdrug.data import Molecule

from retflow.utils.wrappers import GraphWrapper

bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = (
        torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    )
    E[diag] = 0
    return E


def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(  # type: ignore
        edge_index, edge_attr
    )
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)
    E = to_dense_adj(
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        max_num_nodes=max_num_nodes,
    )
    E = encode_no_edge(E)

    return GraphWrapper(X=X, E=E, y=None), node_mask


def build_molecule(atom_types, edge_types, atom_decoder, return_n_dummy_atoms=False):
    mol = Chem.RWMol()
    dummy_atoms = set()
    mapping = {}
    j = 0
    for i, atom in enumerate(atom_types):
        a = Chem.Atom(atom_decoder[atom.item()])
        if a.GetSymbol() == "*":
            dummy_atoms.add(i)
            continue

        mol.AddAtom(a)
        mapping[i] = j
        j += 1

    edge_types_up = torch.triu(edge_types)
    all_bonds = torch.nonzero(edge_types_up)
    for i, bond in enumerate(all_bonds):
        if bond[0].item() == bond[1].item():
            continue
        if bond[0].item() in dummy_atoms:
            continue
        if bond[1].item() in dummy_atoms:
            continue

        mol.AddBond(
            mapping[bond[0].item()],
            mapping[bond[1].item()],
            bond_dict[edge_types_up[bond[0], bond[1]].item()],
        )

    if return_n_dummy_atoms:
        return mol, len(dummy_atoms)
    else:
        return mol


def build_simple_molecule(molecule):
    mol = Chem.RWMol()
    mapping = {}
    for i, atom in enumerate(molecule.GetAtoms()):
        a = Chem.Atom(atom.GetSymbol())
        mol.AddAtom(a)
        mapping[i] = i

    adj_mat = GetAdjacencyMatrix(molecule)
    (rows, cols) = np.nonzero(adj_mat)

    for i, j in zip(rows, cols):
        try:
            bond = molecule.GetBondBetweenAtoms(int(i), int(j))
            mol.AddBond(int(i), int(j), bond.GetBondType())
        except Exception:  # bond already exists
            pass
    return mol


def get_graph_list(X, E, node_mask, onehot=False):
    graph_list = []
    n_nodes = node_mask.sum(-1)
    for i in range(len(X)):
        n = n_nodes[i]

        atom_types = X[i, :n].cpu()
        edge_types = E[i, :n, :n].cpu()

        if onehot:
            atom_types = atom_types.argmax(-1)
            edge_types = edge_types.argmax(-1)
        
        graph_list.append([atom_types, edge_types])
    return graph_list


def get_molecule_list(X, E, node_mask, atom_decoder, onehot=False):
    molecule_list = []
    n_nodes = node_mask.sum(-1)
    for i in range(len(X)):
        n = n_nodes[i]
        atom_types = X[i, :n].cpu()
        edge_types = E[i, :n, :n].cpu()

        if onehot:
            atom_types = atom_types.argmax(-1)
            edge_types = edge_types.argmax(-1)
        mol = build_molecule(atom_types, edge_types, atom_decoder)
        molecule_list.append(mol)
    return molecule_list

def get_molecule_smi_list(X, E, node_mask, atom_decoder, onehot=False):
    molecule_smi_list = []
    n_nodes = node_mask.sum(-1)
    for i in range(len(X)):
        n = n_nodes[i]
        atom_types = X[i, :n].cpu()
        edge_types = E[i, :n, :n].cpu()
        if onehot:
            atom_types = atom_types.argmax(-1)
            edge_types = edge_types.argmax(-1)
        mol = build_molecule(atom_types, edge_types, atom_decoder)
        molecule_smi_list.append(Chem.MolToSmiles(mol, canonical=True))
    return molecule_smi_list


# following functions are for dataset processing and used for synthon based retrosynthesis
def build_graph_from_mol(molecule, max_num_nodes, types, bonds):
    max_num_nodes = max(
        molecule.GetNumAtoms(), max_num_nodes
    )  # in case |reactants|-|product| > max_n_dummy_nodes
    type_idx = [len(types) - 1] * max_num_nodes
    for i, atom in enumerate(molecule.GetAtoms()):
        type_idx[atom.GetIdx()] = types[atom.GetSymbol()]

    num_classes = len(types)
    x = F.one_hot(torch.tensor(type_idx), num_classes=num_classes).float()

    adj = GetAdjacencyMatrix(molecule)
    (rows, cols) = np.nonzero(adj)

    torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
    torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([torch_rows, torch_cols], dim=0)

    edge_type = []
    for i, j in zip(rows, cols):
        bond = molecule.GetBondBetweenAtoms(int(i), int(j))
        edge_type += [bonds[bond.GetBondType()] + 1]

    edge_attr = F.one_hot(
        torch.tensor(edge_type, dtype=torch.long), num_classes=len(bonds) + 1
    ).to(torch.float)
    return x, edge_index, edge_attr


def build_graph_from_mol_with_mapping(molecule, mapping, max_num_nodes, types, bonds):
    max_num_nodes = max(
        molecule.GetNumAtoms(), max_num_nodes
    )  # in case |reactants|-|product| > max_n_dummy_nodes
    type_idx = [len(types) - 1] * max_num_nodes
    for i, atom in enumerate(molecule.GetAtoms()):
        type_idx[mapping[atom.GetAtomMapNum()]] = types[atom.GetSymbol()]

    num_classes = len(types)
    x = F.one_hot(torch.tensor(type_idx), num_classes=num_classes).float()

    row, col, edge_type = [], [], []
    for bond in molecule.GetBonds():
        start_atom_map_num = molecule.GetAtomWithIdx(
            bond.GetBeginAtomIdx()
        ).GetAtomMapNum()
        end_atom_map_num = molecule.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum()
        start, end = mapping[start_atom_map_num], mapping[end_atom_map_num]
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()] + 1]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

    return x, edge_index, edge_attr


def compute_nodes_order_mapping(molecule):
    # In case if atomic map numbers do not start from 1
    order = []
    for atom in molecule.GetAtoms():
        order.append(atom.GetAtomMapNum())
    order = {atom_map_num: idx for idx, atom_map_num in enumerate(sorted(order))}
    return order


def compute_joint_nodes_order_mapping(*molecules):
    """Build one atom-map coordinate system for every molecule in a reaction.

    The original dataset builder derived the coordinate system from the
    reactants alone.  That silently assumes that every product atom also
    occurs in the reactants and makes complete endpoint datasets fail on
    product-only mapped atoms.  A union mapping represents additions and
    deletions with the existing dummy-atom feature instead.
    """
    atom_map_numbers = set()
    for molecule in molecules:
        if molecule is None:
            raise ValueError("cannot build a node mapping for a missing molecule")
        current = [atom.GetAtomMapNum() for atom in molecule.GetAtoms()]
        if any(atom_map_number <= 0 for atom_map_number in current):
            raise ValueError("all graph atoms must have positive atom-map numbers")
        if len(current) != len(set(current)):
            raise ValueError("atom-map numbers must be unique within each molecule")
        atom_map_numbers.update(current)
    return {
        atom_map_number: index
        for index, atom_map_number in enumerate(sorted(atom_map_numbers))
    }


def compute_nodes_mapping(molecule):
    # In case if atomic map numbers do not start from 1
    order = []
    for atom in molecule.GetAtoms():
        order.append(atom.GetAtomMapNum())
    order = {atom_map_num: idx for idx, atom_map_num in enumerate(order)}
    return order


def reactants_with_partial_atom_mapping(
    reactant: Molecule, product: Molecule, atom_feature
):
    reactant = reactant.to_molecule()
    product_maps = set(product.atom_map.tolist())
    reactant_maps = {
        atom.GetAtomMapNum() for atom in reactant.GetAtoms()
    }
    missing_product_maps = product_maps - reactant_maps
    if missing_product_maps:
        raise ValueError(
            "published reaction-center mapping requires every product atom "
            "to occur in the reactants"
        )
    for atom in reactant.GetAtoms():
        if atom.GetAtomMapNum() not in product_maps:
            atom.SetAtomMapNum(0)
    new_rmol = Chem.MolToSmiles(reactant, canonical=True)
    return Molecule.from_molecule(Chem.MolFromSmiles(new_rmol), atom_feature)


def get_synthons(reactant: Molecule, product: Molecule):
    edge_added, edge_modified, prod2react = get_difference(reactant, product)

    reactants = []
    synthons = []

    if len(edge_added) > 0:
        if len(edge_added) == 1:  # add a single edge
            reverse_edge = edge_added.flip(1)
            any = -torch.ones(2, 1, dtype=torch.long)
            pattern = torch.cat([edge_added, reverse_edge])
            pattern = torch.cat([pattern, any], dim=-1)
            index, num_match = product.match(pattern)
            edge_mask = torch.ones(product.num_edge, dtype=torch.bool)
            edge_mask[index] = 0
            product = product.edge_mask(edge_mask)
            _reactants = reactant.connected_components()[0]
            _synthons = product.connected_components()[0]
            assert len(_synthons) >= len(
                _reactants
            )  # because a few samples contain multiple products

            h, t = edge_added[0]
            reaction_center = torch.tensor([product.atom_map[h], product.atom_map[t]])
            with _reactants.graph():
                _reactants.reaction_center = reaction_center.expand(len(_reactants), -1)
            with _synthons.graph():
                _synthons.reaction_center = reaction_center.expand(len(_synthons), -1)
            # reactant / sython can be uniquely indexed by their maximal atom mapping ID
            reactant_id = scatter_max(
                _reactants.atom_map, _reactants.node2graph, dim_size=len(_reactants)
            )[0]
            synthon_id = scatter_max(
                _synthons.atom_map, _synthons.node2graph, dim_size=len(_synthons)
            )[0]
            react2synthon = (
                (reactant_id.unsqueeze(-1) == synthon_id.unsqueeze(0)).long().argmax(-1)
            )
            react2synthon = react2synthon.tolist()
            for r, s in enumerate(react2synthon):
                reactants.append(_reactants[r])
                synthons.append(_synthons[s])
    else:
        num_cc = reactant.connected_components()[1]
        assert num_cc == 1

        if len(edge_modified) == 1:  # modify a single edge
            synthon = product
            h, t = edge_modified[0]
            if product.degree_in[h] == 1:
                reaction_center = torch.tensor([product.atom_map[h], 0])
            elif product.degree_in[t] == 1:
                reaction_center = torch.tensor([product.atom_map[t], 0])
            else:
                # pretend the reaction center is h
                reaction_center = torch.tensor([product.atom_map[h], 0])
            with reactant.graph():
                reactant.reaction_center = reaction_center
            with synthon.graph():
                synthon.reaction_center = reaction_center
            reactants.append(reactant)
            synthons.append(synthon)
        else:
            product_hs = torch.tensor(
                [atom.GetTotalNumHs() for atom in product.to_molecule().GetAtoms()]
            )
            reactant_hs = torch.tensor(
                [atom.GetTotalNumHs() for atom in reactant.to_molecule().GetAtoms()]
            )
            atom_modified = (product_hs != reactant_hs[prod2react]).nonzero().flatten()
            if len(atom_modified) == 1:  # modify single node
                synthon = product
                reaction_center = torch.tensor([product.atom_map[atom_modified[0]], 0])
                with reactant.graph():
                    reactant.reaction_center = reaction_center
                with synthon.graph():
                    synthon.reaction_center = reaction_center
                reactants.append(reactant)
                synthons.append(synthon)

    return reactants, synthons


def get_difference(reactant: Molecule, product: Molecule):
    product2id = product.atom_map
    id2reactant = torch.zeros(product2id.max() + 1, dtype=torch.long)
    id2reactant[reactant.atom_map] = torch.arange(reactant.num_node)
    prod2react = id2reactant[product2id]

    # check edges in the product
    product = product.directed()
    # O(n^2) brute-force match is faster than O(nlogn) data.Graph.match for small molecules
    mapped_edge = product.edge_list.clone()
    mapped_edge[:, :2] = prod2react[mapped_edge[:, :2]]
    is_same_index = mapped_edge.unsqueeze(0) == reactant.edge_list.unsqueeze(1)
    has_typed_edge = is_same_index.all(dim=-1).any(dim=0)
    has_edge = is_same_index[:, :, :2].all(dim=-1).any(dim=0)
    is_added = ~has_edge
    is_modified = has_edge & ~has_typed_edge
    edge_added = product.edge_list[is_added, :2]
    edge_modified = product.edge_list[is_modified, :2]

    return edge_added, edge_modified, prod2react
