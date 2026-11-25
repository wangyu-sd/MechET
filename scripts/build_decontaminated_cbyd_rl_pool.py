"""Build a strictly decontaminated high-difficulty CbyD RL curriculum & milestone-reward pool on shared Ceph storage."""
import argparse
import hashlib
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import inchi as rd_inchi


def _canon_and_ik(smiles: str):
    smiles = str(smiles or "").strip()
    if not smiles:
        return "", ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "", ""
    canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    try:
        ik = rd_inchi.MolToInchiKey(mol)
    except Exception:
        ik = ""
    return canon, ik


def _tree_lls(node: dict) -> int:
    children = node.get("children") or []
    if not children:
        return 0
    if node.get("type") == "reaction":
        return 1 + max((_tree_lls(c) for c in children), default=0)
    return max((_tree_lls(c) for c in children), default=0)


def _extract_milestones(node: dict, depth: int = 0, out=None):
    if out is None:
        out = []
    if node.get("type") == "mol" and depth > 0:
        smi = node.get("smiles") or node.get("target") or ""
        if smi:
            c, _ = _canon_and_ik(smi)
            if c:
                out.append(c)
    for child in node.get("children") or []:
        _extract_milestones(child, depth + (1 if node.get("type") == "reaction" else 0), out)
    return out


def _collect_protected_identities(paths):
    protected_ids = set()
    protected_smiles = set()
    protected_iks = set()
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        if p.suffix == ".json":
            data = json.loads(p.read_text())
            items = data if isinstance(data, list) else (data.get("targets") or data.get("rows") or [])
        else:
            items = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        for obj in items:
            if not isinstance(obj, dict):
                continue
            for k in ("task_id", "id", "source_id", "name"):
                if obj.get(k):
                    protected_ids.add(str(obj[k]))
            for k in ("smiles", "product_unmapped", "source_product_smiles", "target", "root"):
                if obj.get(k):
                    c, ik = _canon_and_ik(str(obj[k]))
                    if c:
                        protected_smiles.add(c)
                    if ik:
                        protected_iks.add(ik)
    return protected_ids, protected_smiles, protected_iks


def build_cbyd_rl_pool(cbyd_jsonl: Path, protected_paths, out_dir: Path, min_lls: int = 2):
    protected_ids, protected_smiles, protected_iks = _collect_protected_identities(protected_paths)
    kept = []
    removed_overlap = 0
    removed_invalid = 0
    removed_shallow = 0
    seen_smiles = set()

    with cbyd_jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            tid = str(obj.get("id") or "")
            smi = str(obj.get("target") or "")
            canon, ik = _canon_and_ik(smi)
            if not tid or not canon:
                removed_invalid += 1
                continue
            if tid in protected_ids or canon in protected_smiles or (ik and ik in protected_iks):
                removed_overlap += 1
                continue
            if canon in seen_smiles:
                continue
            seen_smiles.add(canon)
            lls = _tree_lls(obj)
            if lls < min_lls:
                removed_shallow += 1
                continue
            milestones = list(dict.fromkeys(_extract_milestones(obj)))[:12]
            kept.append({
                "task_id": f"cbyd_rl:{tid}",
                "source_id": tid,
                "target": canon,
                "inchikey": ik,
                "reference_lls": lls,
                "stratum": min(max(lls, 1), 4),
                "milestone_intermediates": milestones,
            })

    kept.sort(key=lambda r: hashlib.sha256(r["task_id"].encode("utf-8")).hexdigest())
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cbyd_rl_train_decontaminated.jsonl"
    out_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in kept) + ("\n" if kept else ""))
    manifest = {
        "source": str(cbyd_jsonl),
        "output": str(out_path),
        "kept_targets": len(kept),
        "removed_overlap_with_public_or_valid": removed_overlap,
        "removed_invalid": removed_invalid,
        "removed_shallow": removed_shallow,
        "protected_sources": [str(p) for p in protected_paths],
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
    (out_dir / "cbyd_rl_train_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cbyd-jsonl", type=Path, required=True)
    parser.add_argument("--protect", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_cbyd_rl_pool(args.cbyd_jsonl, args.protect, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
