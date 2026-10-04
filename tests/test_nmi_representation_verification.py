import hashlib
import json

import pytest

from scripts.verify_nmi_matched_representations import verify


def test_verifier_rejects_product_drift_and_accepts_fixed_budget(tmp_path):
    root = tmp_path / "matched"
    for condition in ("direct", "open_flow", "closed_loop"):
        path = root / condition
        path.mkdir(parents=True)
        rows = {}
        hashes = {}
        for split in ("train", "valid", "test"):
            row = {
                "source_id": f"reaction_{split}",
                "target_smiles": "[CH3:1][OH:2]",
                "structural_precursor": "[CH3:1][Br:3].[OH-:2]",
                "metadata": {"nmi_split": split},
            }
            if condition == "closed_loop":
                row["messages"] = [{"role": "user", "content":
                    "TARGET: [CH3:1][OH:2]\nINITIAL ENVIRONMENT OBSERVATION:\n" +
                    json.dumps({"max_tool_calls": 40})}]
            data = (json.dumps(row) + "\n").encode()
            (path / f"{split}.jsonl").write_bytes(data)
            rows[split] = 1
            hashes[split] = hashlib.sha256(data).hexdigest()
        (path / "manifest.json").write_text(json.dumps({
            "parent_split_manifest_sha256": "same",
            "stable_reaction_ids": {"train": "a", "valid": "b", "test": "c"},
            "rows": rows,
            "output_sha256": hashes,
        }))
    report = verify(root, root / "verification.json")
    assert report["passed"] is True
    assert report["splits"]["test"]["mapped_product_exact_parity"] is True
    drift = root / "open_flow" / "test.jsonl"
    row = json.loads(drift.read_text())
    row["target_smiles"] = "[CH4:1]"
    drift.write_text(json.dumps(row) + "\n")
    manifest_path = root / "open_flow" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_sha256"]["test"] = hashlib.sha256(drift.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="product or endpoint mismatch"):
        verify(root, root / "verification2.json")
