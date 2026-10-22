"""Main-product-only source rebasing for inverse electron-flow supervision.

The source reference maps and electron moves are training-private. The actor
still starts from the unmapped single main product and has to generate IMPORT.
This code does not introduce a model, sampling, or a new chemical executor.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from rdkit import Chem


@dataclass(frozen=True)
class MainProductPartition:
    root: str
    withheld: tuple[str, ...]
    original_components: int


def select_main_product_from_mapped_target(target: str) -> MainProductPartition:
    """Select only a unique largest carbon-containing molecular component.

    No reference reactants or electron-event labels are consulted here.
    Atom maps are retained only for offline reference replay. Stereochemistry,
    formal charges and explicit graph hydrogens are preserved.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = False
    molecule = Chem.MolFromSmiles(target, params)
    if molecule is None:
        raise ValueError("MAIN_PRODUCT_TARGET_PARSE_FAILED")
    fragments = list(Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True))
    if not fragments:
        raise ValueError("MAIN_PRODUCT_TARGET_EMPTY")
    organics = [
        (i, fragment) for i, fragment in enumerate(fragments)
        if any(a.GetAtomicNum() == 6 for a in fragment.GetAtoms())
    ]
    if not organics:
        raise ValueError("MAIN_PRODUCT_NOT_FOUND")
    largest = max(fragment.GetNumHeavyAtoms() for _, fragment in organics)
    winners = [(i, fragment) for i, fragment in organics
               if fragment.GetNumHeavyAtoms() == largest]
    if len(winners) != 1:
        # Do not guess the intended product when ties are present. A tie of
        # identical components is still ambiguous as an atom-mapped instance.
        raise ValueError("MAIN_PRODUCT_AMBIGUOUS")
    chosen_index = winners[0][0]
    all_smiles = [Chem.MolToSmiles(f, canonical=True, isomericSmiles=True)
                  for f in fragments]
    root = all_smiles[chosen_index]
    others = tuple(all_smiles[i] for i in range(len(all_smiles))
                   if i != chosen_index)
    root_maps = set(a.GetAtomMapNum() for a in fragments[chosen_index].GetAtoms())
    all_maps = [a.GetAtomMapNum() for a in molecule.GetAtoms()]
    if any(k <= 0 for k in all_maps) or len(all_maps) != len(set(all_maps)):
        raise ValueError("MAIN_PRODUCT_SOURCE_MAPS_INVALID")
    if len(root_maps) != fragments[chosen_index].GetNumAtoms():
        raise ValueError("MAIN_PRODUCT_ROOT_MAPS_INVALID")
    return MainProductPartition(root=root, withheld=others,
                                original_components=len(fragments))


def rebase_flower_inverse_row(source: Mapping[str, Any]) -> dict[str, Any]:
    """Change only root inventory and reference IMPORT availability.

    The original source FLowER inverse moves, terminal GT, and state_after
    references remain untouched. The existing v2 converter independently
    remaps and replays all moves, including newly scheduled imports.
    """
    row = deepcopy(dict(source))
    target = str(row.get("target_smiles") or "")
    plan = deepcopy(dict((row.get("metadata") or {}).get("trace_plan") or {}))
    if not plan.get("steps"):
        raise ValueError("MAIN_PRODUCT_SOURCE_TRACE_MISSING")
    partition = select_main_product_from_mapped_target(target)
    if partition.original_components == 1:
        # Still flag the real public-root protocol for map remap parity.
        pass
    import_maps = set()
    for fragment in partition.withheld:
        mol = Chem.MolFromSmiles(fragment)
        if mol is None:
            raise ValueError("MAIN_PRODUCT_WITHHELD_PARSE_FAILED")
        for atom in mol.GetAtoms():
            k = atom.GetAtomMapNum()
            if k in import_maps:
                raise ValueError("MAIN_PRODUCT_DUPLICATE_IMPORTED_MAP")
            import_maps.add(k)

    existing_messages = list(row.get("messages") or [])
    existing_import_maps: set[int] = set()
    for message in existing_messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") != "import_fragment":
                continue
            frag = str((function.get("arguments") or {}).get("fragment_smiles") or "")
            mol = Chem.MolFromSmiles(frag)
            if mol is None:
                raise ValueError("MAIN_PRODUCT_EXISTING_IMPORT_PARSE_FAILED")
            existing_import_maps.update(a.GetAtomMapNum() for a in mol.GetAtoms())
    if existing_import_maps & import_maps:
        raise ValueError("MAIN_PRODUCT_PRELOADED_AND_IMPORTED_MAP_OVERLAP")

    synthetic_imports = [
        {
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": f"main_root_import_{index:04d}", "type": "function",
                "function": {"name": "import_fragment",
                             "arguments": {"fragment_smiles": fragment}},
            }],
        }
        for index, fragment in enumerate(partition.withheld)
    ]
    row["messages"] = synthetic_imports + existing_messages
    row["target_smiles"] = partition.root
    plan["target_smiles"] = partition.root
    plan["initial_imports"] = (
        list(partition.withheld) + list(plan.get("initial_imports") or [])
    )
    row["metadata"] = {
        **dict(row.get("metadata") or {}),
        "trace_plan": plan,
        "target_selection": {
            "input_condition": "main_product_only_v1",
            "original_product_component_count": partition.original_components,
            "withheld_product_side_components": len(partition.withheld),
            "reference_import_maps_added": len(import_maps),
        },
        "main_product_root_source": "frozen_source_target_smiles",
    }
    if "." in partition.root:
        raise ValueError("MAIN_PRODUCT_ROOT_NOT_ONE_COMPONENT")
    return row
