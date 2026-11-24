"""Build and freeze the 40-molecule Clinical SAR Common-Intermediate Panel (Elevation 4) and 250-molecule AI-Generated Dark Space stratifier (Elevation 5)."""
import argparse
import hashlib
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# 5 canonical medicinal-chemistry scaffolds (derived from clinical kinase/GPCR/NSAID/sulfonamide/benzimidazole drug series),
# each expanded to 8 late-stage R-group analogs sharing one common advanced intermediate core.
SAR_SERIES_TEMPLATES = [
    {
        "series_id": "sar_01_anilinoquinazoline_egfr",
        "drug_class": "EGFR Kinase Inhibitor (Erlotinib/Gefitinib core)",
        "common_intermediate_smiles": "COc1cc2ncnc(Cl)c2cc1OC",
        "forbidden_reagents": ["CrO3", "OsO4", "Pd_cat"],
        "analogs": [
            ("egfr_01", "COc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OC"),
            ("egfr_02", "COc1cc2ncnc(Nc3cccc(Cl)c3)c2cc1OC"),
            ("egfr_03", "COc1cc2ncnc(Nc3cccc(F)c3)c2cc1OC"),
            ("egfr_04", "COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OC"),
            ("egfr_05", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OC"),
            ("egfr_06", "COc1cc2ncnc(Nc3cccc(C(F)(F)F)c3)c2cc1OC"),
            ("egfr_07", "COc1cc2ncnc(Nc3cccc(OC)c3)c2cc1OC"),
            ("egfr_08", "COc1cc2ncnc(Nc3ccc(C#N)cc3)c2cc1OC"),
        ],
    },
    {
        "series_id": "sar_02_biaryl_carboxamide",
        "drug_class": "Biaryl Carboxamide (Imatinib/Sonidegib-like core)",
        "common_intermediate_smiles": "O=C(O)c1ccc(-c2ccccc2)cc1",
        "forbidden_reagents": ["CrO3", "Pd_cat"],
        "analogs": [
            ("biaryl_01", "O=C(Nc1ccccc1)c1ccc(-c2ccccc2)cc1"),
            ("biaryl_02", "O=C(Nc1ccc(F)cc1)c1ccc(-c2ccccc2)cc1"),
            ("biaryl_03", "O=C(Nc1ccc(Cl)cc1)c1ccc(-c2ccccc2)cc1"),
            ("biaryl_04", "O=C(Nc1ccc(OC)cc1)c1ccc(-c2ccccc2)cc1"),
            ("biaryl_05", "O=C(Nc1ccc(C(F)(F)F)cc1)c1ccc(-c2ccccc2)cc1"),
            ("biaryl_06", "O=C(Nc1ccncc1)c1ccc(-c2ccccc2)cc1"),
            ("biaryl_07", "O=C(N1CCOCC1)c1ccc(-c2ccccc2)cc1"),
            ("biaryl_08", "O=C(NCc1ccccc1)c1ccc(-c2ccccc2)cc1"),
        ],
    },
    {
        "series_id": "sar_03_arylsulfonamide_cox2",
        "drug_class": "Arylsulfonamide (Celecoxib/Sildenafil sulfonamide core)",
        "common_intermediate_smiles": "Cc1ccc(S(=O)(=O)Cl)cc1",
        "forbidden_reagents": ["OsO4", "CrO3"],
        "analogs": [
            ("sulf_01", "Cc1ccc(S(=O)(=O)Nc2ccccc2)cc1"),
            ("sulf_02", "Cc1ccc(S(=O)(=O)Nc2ccc(F)cc2)cc1"),
            ("sulf_03", "Cc1ccc(S(=O)(=O)Nc2ccc(Cl)cc2)cc1"),
            ("sulf_04", "Cc1ccc(S(=O)(=O)Nc2ccc(OC)cc2)cc1"),
            ("sulf_05", "Cc1ccc(S(=O)(=O)N2CCN(C)CC2)cc1"),
            ("sulf_06", "Cc1ccc(S(=O)(=O)N2CCOCC2)cc1"),
            ("sulf_07", "Cc1ccc(S(=O)(=O)Nc2ccncc2)cc1"),
            ("sulf_08", "Cc1ccc(S(=O)(=O)NCc2ccccc2)cc1"),
        ],
    },
    {
        "series_id": "sar_04_benzimidazole_antagonist",
        "drug_class": "2-Arylbenzimidazole (Telmisartan/Candesartan core)",
        "common_intermediate_smiles": "Nc1ccc(C(=O)OC)cc1N",
        "forbidden_reagents": ["CrO3", "OsO4"],
        "analogs": [
            ("bzm_01", "COC(=O)c1ccc2[nH]c(-c3ccccc3)nc2c1"),
            ("bzm_02", "COC(=O)c1ccc2[nH]c(-c3ccc(F)cc3)nc2c1"),
            ("bzm_03", "COC(=O)c1ccc2[nH]c(-c3ccc(Cl)cc3)nc2c1"),
            ("bzm_04", "COC(=O)c1ccc2[nH]c(-c3ccc(OC)cc3)nc2c1"),
            ("bzm_05", "COC(=O)c1ccc2[nH]c(-c3ccc(C#N)cc3)nc2c1"),
            ("bzm_06", "COC(=O)c1ccc2[nH]c(-c3ccncc3)nc2c1"),
            ("bzm_07", "COC(=O)c1ccc2[nH]c(-c3ccc(C(F)(F)F)cc3)nc2c1"),
            ("bzm_08", "COC(=O)c1ccc2[nH]c(Cc3ccccc3)nc2c1"),
        ],
    },
    {
        "series_id": "sar_05_aminopyrimidine_kinase",
        "drug_class": "2,4-Diaminopyrimidine (Osimertinib/Ceritinib core)",
        "common_intermediate_smiles": "Clc1nccc(Nc2ccccc2)n1",
        "forbidden_reagents": ["Pd_cat", "OsO4"],
        "analogs": [
            ("apyr_01", "COc1ccc(Nc2nccc(Nc3ccccc3)n2)cc1"),
            ("apyr_02", "Fc1ccc(Nc2nccc(Nc3ccccc3)n2)cc1"),
            ("apyr_03", "Clc1ccc(Nc2nccc(Nc3ccccc3)n2)cc1"),
            ("apyr_04", "Cc1ccc(Nc2nccc(Nc3ccccc3)n2)cc1"),
            ("apyr_05", "N#Cc1ccc(Nc2nccc(Nc3ccccc3)n2)cc1"),
            ("apyr_06", "FC(F)(F)c1ccc(Nc2nccc(Nc3ccccc3)n2)cc1"),
            ("apyr_07", "CN1CCN(c2ccc(Nc3nccc(Nc4ccccc4)n3)cc2)CC1"),
            ("apyr_08", "O=C(C)Nc1ccc(Nc2nccc(Nc3ccccc3)n2)cc1"),
        ],
    },
]


def freeze_sar40_panel(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for series in SAR_SERIES_TEMPLATES:
        core_mol = Chem.MolFromSmiles(series["common_intermediate_smiles"])
        assert core_mol is not None, f"Invalid core SMILES: {series['common_intermediate_smiles']}"
        core_canon = Chem.MolToSmiles(core_mol, canonical=True)
        for tid, smi in series["analogs"]:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None, f"Invalid analog SMILES: {tid} {smi}"
            canon = Chem.MolToSmiles(mol, canonical=True)
            rows.append({
                "task_id": tid,
                "series_id": series["series_id"],
                "drug_class": series["drug_class"],
                "smiles": canon,
                "common_intermediate_smiles": core_canon,
                "forbidden_reagents": series["forbidden_reagents"],
                "heavy_atoms": int(mol.GetNumHeavyAtoms()),
                "num_rings": int(rdMolDescriptors.CalcNumRings(mol)),
                "mw": round(float(Descriptors.MolWt(mol)), 2),
            })

    out_path = out_dir / "steerable_sar_40.jsonl"
    out_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    meta = {
        "path": str(out_path),
        "total_molecules": len(rows),
        "num_series": len(SAR_SERIES_TEMPLATES),
        "analogs_per_series": 8,
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
    (out_dir / "steerable_sar_40_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_sar40_panel(args.out_dir), indent=2))


if __name__ == "__main__":
    main()
