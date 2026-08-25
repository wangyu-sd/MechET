"""Strict atom-mapping contracts for reaction-level retrosynthesis data."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


_UNCHARGER = rdMolStandardize.Uncharger()


@dataclass(frozen=True)
class MappedReaction:
    reactants: str
    reagents: str
    products: str
    reactants_unmapped: str
    products_unmapped: str
    structural_precursor: str
    auxiliary_fragments: tuple[str, ...]
    product_atom_count: int

    @property
    def reaction_smiles(self) -> str:
        return f"{self.reactants}>{self.reagents}>{self.products}"


def parse_reaction_smiles(value: str) -> tuple[str, str, str]:
    text = str(value or "").strip()
    if text.count(">") == 2:
        reactants, reagents, products = text.split(">")
    elif text.count(">>") == 1:
        reactants, products = text.split(">>")
        reagents = ""
    else:
        raise ValueError("reaction must contain reactants>reagents>products or reactants>>products")
    if not reactants.strip() or not products.strip():
        raise ValueError("mapped reaction has an empty reactant or product side")
    return reactants.strip(), reagents.strip(), products.strip()


def _mol(smiles: str) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(smiles or "").strip(), params)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return mol


def _positive_maps(mol: Chem.Mol) -> list[int]:
    return [int(atom.GetAtomMapNum()) for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0]


def _canonical(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def remove_atom_maps(smiles: str, *, neutralize: bool = False) -> str:
    mol = _mol(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    if neutralize:
        mol = _UNCHARGER.uncharge(mol)
    return _canonical(mol)


def fragment_multiset_unmapped(
    smiles: str, *, neutralize: bool = False
) -> Counter[str]:
    output: Counter[str] = Counter()
    for fragment in str(smiles or "").split("."):
        if fragment.strip():
            output[remove_atom_maps(fragment, neutralize=neutralize)] += 1
    return output


def assert_product_contained_in_reference(product: str, reference: str) -> str:
    """Require an exact or neutralized desired product in the final mixture.

    The mechanism endpoint can retain an acid/base salt even when the original
    reaction-level desired product is neutral. Exact containment remains the
    first audit; neutralization is a named fallback and never changes the
    mapped product used as model input or evaluation reference.
    """
    wanted = fragment_multiset_unmapped(product)
    available = fragment_multiset_unmapped(reference)
    if not (wanted - available):
        return "exact"
    wanted_neutral = fragment_multiset_unmapped(product, neutralize=True)
    available_neutral = fragment_multiset_unmapped(reference, neutralize=True)
    missing = wanted_neutral - available_neutral
    if not missing:
        return "neutralized"
    raise ValueError(
        "mapped reaction product is absent from rxn_prod_min after exact and "
        f"neutralized audits: {dict(missing)}"
    )


def product_only_reindex_reaction(reaction: str) -> MappedReaction:
    """Reindex a mapped reaction without retaining source map-number information.

    Product atoms receive 1..N in deterministic canonical-rank order. The same
    permutation is transported to reactant atoms. Reactant-only or originally
    unmapped atoms receive fresh identifiers above N and are never visible in
    the product input.
    """
    reactant_text, reagents, product_text = parse_reaction_smiles(reaction)
    reactants = _mol(reactant_text)
    products = _mol(product_text)

    product_maps = _positive_maps(products)
    if len(product_maps) != products.GetNumAtoms():
        raise ValueError("every product atom must have a positive atom map")
    if len(product_maps) != len(set(product_maps)):
        raise ValueError("product atom maps must be unique")
    reactant_maps = _positive_maps(reactants)
    if len(reactant_maps) != len(set(reactant_maps)):
        raise ValueError("positive reactant atom maps must be unique")
    missing_on_left = set(product_maps) - set(reactant_maps)
    if missing_on_left:
        raise ValueError(
            "every product atom map must occur on the reactant side; missing "
            + ",".join(map(str, sorted(missing_on_left)))
        )

    rank_product = Chem.Mol(products)
    for atom in rank_product.GetAtoms():
        atom.SetAtomMapNum(0)
    ranks = list(Chem.CanonicalRankAtoms(rank_product, breakTies=True))
    product_order = sorted(range(products.GetNumAtoms()), key=lambda idx: (ranks[idx], idx))
    old_to_new: dict[int, int] = {}
    for new_map, atom_idx in enumerate(product_order, start=1):
        old_map = int(products.GetAtomWithIdx(atom_idx).GetAtomMapNum())
        old_to_new[old_map] = new_map
    for atom in products.GetAtoms():
        atom.SetAtomMapNum(old_to_new[int(atom.GetAtomMapNum())])

    reactant_only: list[int] = []
    for atom in reactants.GetAtoms():
        old_map = int(atom.GetAtomMapNum())
        if old_map in old_to_new:
            atom.SetAtomMapNum(old_to_new[old_map])
        else:
            reactant_only.append(atom.GetIdx())
            atom.SetAtomMapNum(0)

    rank_reactants = Chem.Mol(reactants)
    for atom in rank_reactants.GetAtoms():
        atom.SetAtomMapNum(0)
    reactant_ranks = list(Chem.CanonicalRankAtoms(rank_reactants, breakTies=True))
    next_map = products.GetNumAtoms() + 1
    for atom_idx in sorted(reactant_only, key=lambda idx: (reactant_ranks[idx], idx)):
        reactants.GetAtomWithIdx(atom_idx).SetAtomMapNum(next_map)
        next_map += 1

    remapped_reactants = _canonical(reactants)
    remapped_products = _canonical(products)
    final_product_maps = set(_positive_maps(products))
    final_reactant_maps = set(_positive_maps(reactants))
    expected_product_maps = set(range(1, products.GetNumAtoms() + 1))
    if final_product_maps != expected_product_maps:
        raise AssertionError("product-only reindexing did not produce contiguous maps")
    if not final_product_maps <= final_reactant_maps:
        raise AssertionError("product/reactant atom-map transport failed")

    structural: list[str] = []
    auxiliary: list[str] = []
    for fragment in remapped_reactants.split("."):
        maps = set(_positive_maps(_mol(fragment)))
        (structural if maps & final_product_maps else auxiliary).append(fragment)
    if not structural:
        raise ValueError("mapped reaction has no atom-contributing precursor fragment")

    return MappedReaction(
        reactants=remapped_reactants,
        reagents=reagents,
        products=remapped_products,
        reactants_unmapped=remove_atom_maps(remapped_reactants),
        products_unmapped=remove_atom_maps(remapped_products),
        structural_precursor=".".join(sorted(structural)),
        auxiliary_fragments=tuple(sorted(auxiliary)),
        product_atom_count=products.GetNumAtoms(),
    )
