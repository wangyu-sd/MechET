from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    left = text.index(start)
    right = text.index(end, left)
    return text[:left] + replacement.rstrip() + "\n\n" + text[right:]


def patch_forward_data() -> None:
    path = ROOT / "src/mechet/forward_data.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "from .forward_expert import ElectronContainer, ElectronMove",
        "from .forward_expert import (\n"
        "    ElectronContainer,\n"
        "    ElectronMove,\n"
        "    verify_electron_step,\n"
        ")",
    )
    if "MECHANISTIC_VARIANTS" not in text:
        anchor = "COMPETITOR_FIELDS = (\n    \"competitor_products\", \"negative_products\", \"side_products\", \"alternatives\",\n)"
        addition = anchor + "\nMECHANISTIC_VARIANTS = (\n" \
            "    (\"ori\", \"mech_smi_ori\", \"elem_reac_ori\", \"elem_prod_ori\"),\n" \
            "    (\"spe\", \"mech_smi_spe\", \"elem_reac_spe\", \"elem_prod_spe\"),\n" \
            "    (\"equ\", \"mech_smi_equ\", \"elem_reac_equ\", \"elem_prod_equ\"),\n" \
            "    (\"min\", \"mech_smi_min\", \"elem_reac_min\", \"elem_prod_min\"),\n" \
            ")"
        if anchor not in text:
            raise RuntimeError("COMPETITOR_FIELDS anchor not found")
        text = text.replace(anchor, addition)

    canonical_block = r'''
def _parse_mol(smiles: str) -> Chem.Mol:
    text = str(smiles or "").strip()
    if not text:
        raise ValueError("empty SMILES")
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(text, params)
    if mol is None:
        raise ValueError(f"invalid SMILES: {text}")
    return mol


def _canonical(smiles: str, *, require_maps: bool = True) -> str:
    text = str(smiles or "").strip()
    if not text:
        return ""
    mol = _parse_mol(text)
    maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    positive_maps = [value for value in maps if value > 0]
    if len(positive_maps) != len(set(positive_maps)):
        raise ValueError("positive atom maps must be unique")
    if require_maps and len(positive_maps) != len(maps):
        raise ValueError("unique positive atom maps are required")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _complete_atom_maps(smiles: str) -> tuple[str, dict[str, Any]]:
    """Fill missing maps without changing arrow-referenced positive map labels."""
    mol = _parse_mol(smiles)
    existing = [
        atom.GetAtomMapNum()
        for atom in mol.GetAtoms()
        if atom.GetAtomMapNum() > 0
    ]
    if len(existing) != len(set(existing)):
        raise ValueError("duplicate positive atom maps in mechanistic state")
    rank_mol = Chem.Mol(mol)
    for atom in rank_mol.GetAtoms():
        atom.SetAtomMapNum(0)
    ranks = list(Chem.CanonicalRankAtoms(rank_mol, breakTies=True))
    unmapped = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomMapNum() <= 0
    ]
    next_map = max(existing, default=0) + 1
    assigned: dict[str, int] = {}
    for atom_idx in sorted(unmapped, key=lambda idx: (ranks[idx], idx)):
        while next_map in existing:
            next_map += 1
        mol.GetAtomWithIdx(atom_idx).SetAtomMapNum(next_map)
        assigned[str(atom_idx)] = next_map
        existing.append(next_map)
        next_map += 1
    output = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    _canonical(output, require_maps=True)
    return output, {
        "completed": bool(unmapped),
        "assigned_maps": assigned,
        "original_mapped_atoms": len(existing) - len(unmapped),
        "completed_atoms": len(unmapped),
    }


def _unmapped_canonical(smiles: str) -> str:
    mol = _parse_mol(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def _same_structure_unmapped(left: str, right: str) -> bool:
    return _unmapped_canonical(left) == _unmapped_canonical(right)
'''
    text = replace_between(text, "def _canonical(", "def _stable_split", canonical_block)

    steps_block = r'''
def _mechanistic_step(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Recover a fully mapped elementary state pair from partial-map mech_smi."""
    failures: list[str] = []
    for variant, mech_key, reactant_key, product_key in MECHANISTIC_VARIANTS:
        raw = row.get(mech_key)
        if not raw:
            continue
        state_partial, moves = parse_mechanistic_smiles(str(raw))
        if not moves:
            failures.append(f"{variant}: no arrow pairs")
            continue
        try:
            state, map_info = _complete_atom_maps(state_partial)
            referenced_maps = {
                atom_map
                for move in moves
                for side in (move["source"], move["sink"])
                for atom_map in side["atoms"]
            }
            state_maps = {
                atom.GetAtomMapNum() for atom in _parse_mol(state).GetAtoms()
            }
            missing = referenced_maps - state_maps
            if missing:
                raise ValueError(
                    f"arrow atom maps missing from state: {sorted(missing)}"
                )
            executed = verify_electron_step(state, moves)
            if not executed.get("ok"):
                raise ValueError(
                    "formal step execution failed: "
                    + str(executed.get("message") or executed.get("code"))
                )
            target = _canonical(str(executed["state_smiles"]), require_maps=True)
            reactant_reference = str(row.get(reactant_key) or "")
            product_reference = str(row.get(product_key) or "")
            reactant_match = (
                not reactant_reference
                or _same_structure_unmapped(state, reactant_reference)
            )
            product_match = (
                not product_reference
                or _same_structure_unmapped(target, product_reference)
            )
            if not reactant_match or not product_match:
                raise ValueError(
                    "reference mismatch "
                    f"reactant={reactant_match} product={product_match}"
                )
            return {
                "variant": variant,
                "step_index": int(
                    row.get("step_idx_forward", row.get("step_index", 0)) or 0
                ),
                "state_smiles": state,
                "target_product": target,
                "moves": moves,
                "map_completion": map_info,
                "reference_reactant": reactant_reference,
                "reference_product": product_reference,
                "reference_reactant_match": reactant_match,
                "reference_product_match": product_match,
            }
        except Exception as exc:
            failures.append(f"{variant}: {exc}")
    if failures:
        raise ValueError(
            "mechanistic row could not be reconstructed; " + " | ".join(failures)
        )
    return None


def _steps(
    row: Mapping[str, Any],
    reactants: str,
    product: str,
) -> list[dict[str, Any]]:
    raw = _jsonish(_first(row, STEP_FIELDS, []))
    if isinstance(raw, dict):
        raw = raw.get("steps") or raw.get("trajectory") or [raw]
    output = []
    for index, item in enumerate(raw if isinstance(raw, (list, tuple)) else []):
        if not isinstance(item, Mapping):
            continue
        state = _first(
            item,
            ("state_smiles", "state_before", "reactants", "input_smiles"),
            reactants,
        )
        target = _first(
            item,
            ("target_product", "state_after", "product", "output_smiles"),
            product,
        )
        moves = _moves(item)
        if moves:
            output.append(
                {
                    "step_index": index,
                    "state_smiles": _canonical(str(state), require_maps=True),
                    "target_product": _canonical(str(target), require_maps=True),
                    "moves": moves,
                }
            )
    if not output:
        moves = _moves(row)
        if moves:
            output.append(
                {
                    "step_index": 0,
                    "state_smiles": reactants,
                    "target_product": product,
                    "moves": moves,
                }
            )
    return output
'''
    text = replace_between(text, "def _steps(", "def _adapt_ord_row", steps_block)

    normalize_block = r'''
def normalize_reaction_row(
    row: Mapping[str, Any],
    *,
    source: str,
    row_index: int,
    require_maps: bool = True,
) -> dict[str, Any]:
    row = _adapt_ord_row(row)
    aliases = {
        str(key).strip().lower().replace(" ", "_"): value
        for key, value in row.items()
    }
    row = {**dict(row), **aliases}
    metadata: dict[str, Any] = {
        "original_keys": sorted(str(key) for key in row)
    }
    mechanistic = _mechanistic_step(row)
    if mechanistic is not None:
        reactants = mechanistic["state_smiles"]
        products = mechanistic["target_product"]
        reagents = ""
        steps = [
            {
                key: mechanistic[key]
                for key in (
                    "step_index", "state_smiles", "target_product", "moves"
                )
            }
        ]
        metadata.update(
            {
                "mechanistic_variant": mechanistic["variant"],
                "atom_map_policy": "complete_partial_mechanistic_maps",
                "map_completion": mechanistic["map_completion"],
                "reference_reactant": mechanistic["reference_reactant"],
                "reference_product": mechanistic["reference_product"],
                "reference_reactant_match": mechanistic[
                    "reference_reactant_match"
                ],
                "reference_product_match": mechanistic[
                    "reference_product_match"
                ],
                "mapped_target_source": "formal_electron_step_execution",
            }
        )
    else:
        reaction = str(_first(row, REACTION_FIELDS, "") or "")
        if reaction:
            reactants, reagents, products = split_reaction_smiles(reaction)
        else:
            reactants = str(_first(row, REACTANT_FIELDS, "") or "")
            products = str(_first(row, PRODUCT_FIELDS, "") or "")
            reagents = str(_first(row, REAGENT_FIELDS, "") or "")
        reactants = _canonical(reactants, require_maps=require_maps)
        products = _canonical(products, require_maps=require_maps)
        if not reactants or not products:
            raise ValueError("reactants and products are required")
        reagents = _canonical(reagents, require_maps=False) if reagents else ""
        steps = _steps(row, reactants, products)
    reaction = f"{reactants}>{reagents}>{products}"
    identity = str(
        row.get("id")
        or row.get("reaction_id")
        or row.get("rxn_idx")
        or hashlib.sha1(reaction.encode()).hexdigest()[:16]
    )
    split = str(row.get("split") or row.get("set") or "").lower()
    if split in {"validation", "val", "dev"}:
        split = "valid"
    if split not in {"train", "valid", "test"}:
        split = _stable_split(f"{source}:{identity}")
    competitors = _jsonish(_first(row, COMPETITOR_FIELDS, []))
    if isinstance(competitors, str):
        competitors = [value for value in competitors.split("|") if value]
    normalized_competitors = []
    for value in competitors or []:
        try:
            normalized_competitors.append(
                _canonical(str(value), require_maps=require_maps)
            )
        except ValueError:
            continue
    return {
        "id": identity,
        "source": source,
        "source_row": int(row_index),
        "reaction_smiles": reaction,
        "reactants": reactants,
        "reagents": reagents,
        "products": products,
        "mechanism_class": str(_first(row, CLASS_FIELDS, "") or ""),
        "conditions": _jsonish(_first(row, CONDITION_FIELDS, {})),
        "competitor_products": normalized_competitors,
        "steps": steps,
        "split": split,
        "metadata": metadata,
    }
'''
    text = replace_between(
        text, "def normalize_reaction_row(", "def _iter_json", normalize_block
    )
    path.write_text(text, encoding="utf-8")


def patch_forward_expert() -> None:
    path = ROOT / "src/mechet/forward_expert.py"
    text = path.read_text(encoding="utf-8")
    replacement = r'''
def _mol(smiles: str) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    if any(value <= 0 for value in maps) or len(maps) != len(set(maps)):
        raise ValueError("all atoms require unique positive atom maps")
    return mol
'''
    text = replace_between(text, "def _mol(", "def _lp_electrons", replacement)
    path.write_text(text, encoding="utf-8")


def write_tests_and_docs() -> None:
    (ROOT / "docs/MECH_USPTO_31K_STANDARDIZATION.md").write_text(
        """# mech-USPTO-31k standardization

`mech_uspto_31k` stores atom maps primarily in `mech_smi_*`; those strings map only atoms referenced by the current arrow code. The accompanying `elem_reac_*` and `elem_prod_*` fields are usually unmapped.

The standardizer parses the mapped reactive state and arrows from `mech_smi_*`, deterministically fills maps for previously unmapped atoms, preserves explicit mapped hydrogens, executes the coupled moves, and uses the executor-derived mapped state as the target. It then audits heavy-atom reactant and product structures against the corresponding unmapped `elem_reac_*` and `elem_prod_*` fields. Inconsistent rows are quarantined.

```bash
python scripts/forward_expert_data.py standardize \\
  --input data/raw/mech_uspto_31k \\
  --output data/forward_expert/reactions.jsonl \\
  --source mech_uspto_31k

python scripts/forward_expert_data.py build \\
  --input data/forward_expert/reactions.jsonl \\
  --output-dir data/forward_expert/steps
```

RXNMapper is not required for these elementary-step rows. Original arrow-referenced map labels are retained; newly assigned maps are local deterministic identifiers recorded in metadata.
""",
        encoding="utf-8",
    )
    (ROOT / "tests/test_mech_uspto_standardization.py").write_text(
        """from rdkit import Chem

from mechet.forward_data import normalize_reaction_row
from mechet.forward_expert import ElectronMove, verify_electron_step


def _maps(smiles: str) -> list[int]:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    assert mol is not None
    return [atom.GetAtomMapNum() for atom in mol.GetAtoms()]


def test_partial_maps_are_completed_and_target_is_executed():
    row = normalize_reaction_row(
        {
            "rxn_idx": 7,
            "step_idx_forward": 0,
            "split": "train",
            "mech_smi_ori": "CC[CH2:1][Br:3].[OH-:2]|(2, 1);((1, 3), 3)",
            "elem_reac_ori": "CCCBr.[OH-]",
            "elem_prod_ori": "CCCO.[Br-]",
        },
        source="mech_uspto_31k",
        row_index=0,
    )
    step = row["steps"][0]
    assert row["metadata"]["mapped_target_source"] == "formal_electron_step_execution"
    assert row["metadata"]["map_completion"]["completed"]
    assert row["metadata"]["reference_reactant_match"]
    assert row["metadata"]["reference_product_match"]
    for smiles in (step["state_smiles"], step["target_product"]):
        maps = _maps(smiles)
        assert all(value > 0 for value in maps)
        assert len(maps) == len(set(maps))
    replay = verify_electron_step(step["state_smiles"], step["moves"])
    assert replay["ok"]
    assert replay["state_smiles"] == step["target_product"]


def test_explicit_mapped_hydrogen_is_preserved():
    row = normalize_reaction_row(
        {
            "rxn_idx": 8,
            "step_idx_forward": 0,
            "split": "train",
            "mech_smi_ori": "C=[O:1].[H:2][Br:3]|(1, 2);((2, 3), 3)",
            "elem_reac_ori": "C=O.Br",
            "elem_prod_ori": "C=[OH+].[Br-]",
        },
        source="mech_uspto_31k",
        row_index=0,
    )
    step = row["steps"][0]
    move_ids = [ElectronMove.parse(value).id for value in step["moves"]]
    assert "LP:1->BOND:1,2/2e" in move_ids
    assert "BOND:2,3->ATOM:3/2e" in move_ids
    assert 2 in _maps(step["state_smiles"])
    assert row["metadata"]["reference_product_match"]
""",
        encoding="utf-8",
    )


def patch_workflow() -> None:
    path = ROOT / ".github/workflows/forward-expert-tests.yml"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        '      - "tests/test_forward_expert.py"\n',
        '      - "tests/test_forward_expert.py"\n'
        '      - "tests/test_mech_uspto_standardization.py"\n',
    )
    text = text.replace(
        "        run: pytest -q tests/test_forward_expert.py",
        "        run: |\n"
        "          pytest -q \\\n"
        "            tests/test_forward_expert.py \\\n"
        "            tests/test_mech_uspto_standardization.py",
    )
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    patch_forward_data()
    patch_forward_expert()
    write_tests_and_docs()
    patch_workflow()
