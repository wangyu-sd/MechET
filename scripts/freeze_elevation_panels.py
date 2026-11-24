"""Freeze deterministic public benchmark panels for CbyD 240, Elevation (4) Steerable SAR Common-Intermediate (~40), and Elevation (5) AI Dark Space (200-300)."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception:  # pragma: no cover
    Chem = None
    MurckoScaffold = None


def _tree_lls(node: dict) -> int:
    children = node.get("children") or []
    if not children:
        return 0
    if node.get("type") == "reaction":
        return 1 + max((_tree_lls(c) for c in children), default=0)
    return max((_tree_lls(c) for c in children), default=0)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_cbyd_stratified(cbyd_jsonl: Path, out_dir: Path, per_stratum: int = 80, exclude_ids=None):
    """Deterministically select per_stratum targets from each LLS bucket (5-9, 10-14, >=15)."""
    exclude_ids = set(exclude_ids or [])
    buckets = {"lls_5_9": [], "lls_10_14": [], "lls_ge_15": []}
    with cbyd_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tid = str(obj.get("id") or "")
            smi = str(obj.get("target") or "")
            if not tid or not smi or tid in exclude_ids:
                continue
            lls = _tree_lls(obj)
            if 5 <= lls <= 9:
                bucket = "lls_5_9"
            elif 10 <= lls <= 14:
                bucket = "lls_10_14"
            elif lls >= 15:
                bucket = "lls_ge_15"
            else:
                continue
            # Deterministic hash order (zero cherry-picking)
            h = hashlib.sha256(tid.encode("utf-8")).hexdigest()
            buckets[bucket].append((h, {
                "task_id": tid,
                "smiles": smi,
                "reference_lls": lls,
                "stratum": bucket,
            }))

    selected = []
    counts = {}
    for bname in ("lls_5_9", "lls_10_14", "lls_ge_15"):
        items = sorted(buckets[bname], key=lambda x: x[0])[:per_stratum]
        counts[bname] = len(items)
        selected.extend(x[1] for x in items)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cbyd_stratified_{len(selected)}.jsonl"
    out_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in selected) + "\n")
    return {
        "path": str(out_path),
        "total": len(selected),
        "strata": counts,
        "sha256": _sha256_file(out_path),
    }


def _murcko_scaffold_smiles(smiles: str) -> str:
    if Chem is None or MurckoScaffold is None:
        return smiles[:12]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return ""
        gen = MurckoScaffold.MakeScaffoldGeneric(scaf)
        return Chem.MolToSmiles(gen, canonical=True)
    except Exception:
        return ""


def freeze_steerable_sar_panel(target_jsonls, out_dir: Path, num_series: int = 5, max_per_series: int = 8):
    """Elevation (4): Group public drug targets by generic Murcko scaffold for Common-Intermediate SAR + constraint steering."""
    clusters = defaultdict(list)
    seen = set()
    for p in target_jsonls:
        with Path(p).open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                smi = str(
                    obj.get("smiles")
                    or obj.get("product_unmapped")
                    or obj.get("source_product_smiles")
                    or obj.get("target")
                    or obj.get("root")
                    or ""
                ).strip()
                tid = str(obj.get("task_id") or obj.get("id") or obj.get("name") or smi)
                if not smi or smi in seen:
                    continue
                seen.add(smi)
                scaf = _murcko_scaffold_smiles(smi)
                if scaf:
                    clusters[scaf].append({"task_id": tid, "smiles": smi, "generic_scaffold": scaf})

    ranked = sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    selected_series = []
    flat_rows = []
    for idx, (scaf, members) in enumerate(ranked[:num_series]):
        chosen = sorted(members, key=lambda x: x["task_id"])[:max_per_series]
        series_id = f"sar_series_{idx + 1:02d}"
        selected_series.append({
            "series_id": series_id,
            "generic_scaffold": scaf,
            "size": len(chosen),
            "members": chosen,
        })
        for m in chosen:
            flat_rows.append({
                "series_id": series_id,
                "task_id": m["task_id"],
                "smiles": m["smiles"],
                "generic_scaffold": scaf,
                "constraints": {
                    "forbid_reagent_classes": ["CrO3", "OsO4", "Pd_cat"],
                    "seek_common_intermediate": True,
                },
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"steerable_sar_panel_{len(flat_rows)}.jsonl"
    out_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in flat_rows) + ("\n" if flat_rows else ""))
    (out_dir / "steerable_sar_series.json").write_text(json.dumps(selected_series, indent=2) + "\n")
    return {
        "path": str(out_path),
        "total_molecules": len(flat_rows),
        "num_series": len(selected_series),
        "series_sizes": [s["size"] for s in selected_series],
        "sha256": _sha256_file(out_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cbyd-jsonl", type=Path, required=True)
    parser.add_argument("--sar-source", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cbyd-per-stratum", type=int, default=80)
    args = parser.parse_args()

    cbyd_meta = freeze_cbyd_stratified(args.cbyd_jsonl, args.out_dir, per_stratum=args.cbyd_per_stratum)
    sar_meta = freeze_steerable_sar_panel(args.sar_source, args.out_dir)
    manifest = {"cbyd": cbyd_meta, "steerable_sar": sar_meta}
    (args.out_dir / "elevation_panels_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
