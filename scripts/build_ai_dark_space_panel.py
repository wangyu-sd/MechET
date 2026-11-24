"""Elevation (5): Deterministically freeze and stratify a 250-molecule AI-Generated 'Synthesizability Dark Space' benchmark panel."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, GraphDescriptors, rdMolDescriptors


def _load_smiles(path: Path):
    rows = []
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        for idx, line in enumerate(path.read_text().splitlines()):
            if not line.strip():
                continue
            obj = json.loads(line)
            smi = str(obj.get("smiles") or obj.get("SMILES") or obj.get("target") or "").strip()
            tid = str(obj.get("task_id") or obj.get("id") or f"ai_mol_{idx:05d}")
            if smi:
                rows.append((tid, smi, str(obj.get("source") or path.stem)))
    elif suffix == ".csv":
        with path.open() as f:
            reader = csv.DictReader(f)
            for idx, obj in enumerate(reader):
                smi = str(obj.get("smiles") or obj.get("SMILES") or obj.get("canon_smiles") or "").strip()
                tid = str(obj.get("task_id") or obj.get("id") or obj.get("name") or f"ai_mol_{idx:05d}")
                if smi:
                    rows.append((tid, smi, str(obj.get("source") or path.stem)))
    else:
        for idx, line in enumerate(path.read_text().splitlines()):
            parts = line.strip().split()
            if not parts:
                continue
            smi = parts[0]
            tid = parts[1] if len(parts) > 1 else f"ai_mol_{idx:05d}"
            rows.append((tid, smi, path.stem))
    return rows


def freeze_ai_dark_space(source_paths, out_dir: Path, target_count: int = 250):
    """Deduplicate, RDKit-validate, compute complexity strata (Easy / Medium / Hard Dark Space), and SHA256-lock."""
    seen = set()
    valid = []
    for sp in source_paths:
        for tid, smi, src in _load_smiles(Path(sp)):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            heavy = int(mol.GetNumHeavyAtoms())
            if heavy < 12 or heavy > 45:
                continue
            canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            if canon in seen:
                continue
            seen.add(canon)
            bertz = round(float(GraphDescriptors.BertzCT(mol)), 2)
            rings = int(rdMolDescriptors.CalcNumRings(mol))
            spiro = int(rdMolDescriptors.CalcNumSpiroAtoms(mol))
            bridge = int(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
            chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            # Complexity tier proxy for Dark Space stratification
            if bertz < 450 and spiro == 0 and bridge == 0:
                tier = "tier1_standard_denovo"
            elif bertz < 800 and (spiro + bridge) <= 1:
                tier = "tier2_complex_denovo"
            else:
                tier = "tier3_hard_dark_space"
            h = hashlib.sha256(canon.encode("utf-8")).hexdigest()
            valid.append((h, {
                "task_id": tid,
                "smiles": canon,
                "source": src,
                "tier": tier,
                "bertz_ct": bertz,
                "heavy_atoms": heavy,
                "num_rings": rings,
                "spiro_atoms": spiro,
                "bridgehead_atoms": bridge,
                "chiral_centers": chiral,
                "mw": round(float(Descriptors.MolWt(mol)), 2),
            }))

    by_tier = {"tier1_standard_denovo": [], "tier2_complex_denovo": [], "tier3_hard_dark_space": []}
    for item in valid:
        by_tier[item[1]["tier"]].append(item)

    per_tier = max(1, target_count // 3)
    selected = []
    leftovers = []
    for tname in ("tier1_standard_denovo", "tier2_complex_denovo", "tier3_hard_dark_space"):
        ranked = sorted(by_tier[tname], key=lambda x: x[0])
        selected.extend(x[1] for x in ranked[:per_tier])
        leftovers.extend(x[1] for x in ranked[per_tier:])

    if len(selected) < target_count:
        leftovers.sort(key=lambda x: hashlib.sha256(x["smiles"].encode("utf-8")).hexdigest())
        selected.extend(leftovers[: target_count - len(selected)])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ai_dark_space_{len(selected)}.jsonl"
    out_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in selected) + ("\n" if selected else ""))
    tier_counts = {t: sum(1 for r in selected if r["tier"] == t) for t in by_tier}
    meta = {
        "path": str(out_path),
        "total_molecules": len(selected),
        "tier_counts": tier_counts,
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
    (out_dir / "ai_dark_space_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=250)
    args = parser.parse_args()
    print(json.dumps(freeze_ai_dark_space(args.source, args.out_dir, target_count=args.count), indent=2))


if __name__ == "__main__":
    main()
