import hashlib
import json

from scripts.build_nmi_reaction_covariates import build_covariates


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def test_covariates_use_train_frequencies_and_holdout_annotations(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [
        ("a", "A", ["p", "q"]),
        ("b", "A", ["p"]),
        ("c", "B", ["p", "q"]),
        ("d", "C", ["p"]),
    ]
    content = b"".join((json.dumps({
        "source_id": identifier,
        "metadata": {
            "execution_composition_signature": composition * 64,
            "execution_primitive_signatures": primitives,
            "n_trace_steps": 3,
            "n_trace_moves": 4,
            "n_trace_imports": 1,
        },
    }) + "\n").encode() for identifier, composition, primitives in rows)
    source.write_bytes(content)
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    ids = {"train": ["a", "b"], "valid": ["c"], "test": ["d"]}
    hashes = {}
    for split, values in ids.items():
        data = ("\n".join(values) + "\n").encode()
        (split_dir / f"{split}.ids.txt").write_bytes(data)
        hashes[split] = _sha(data)
    (split_dir / "manifest.json").write_text(json.dumps({
        "n_rows": 4,
        "source_sha256": _sha(content),
        "rows": {split: len(values) for split, values in ids.items()},
        "split_id_sha256": hashes,
    }))
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "artifact_type": "nmi_mechcomp_structural_overlap_v2",
        "scope": "full_frozen_split",
        "split_manifest_sha256": _sha((split_dir / "manifest.json").read_bytes()),
        "reaction_center_undefined_count": 0,
        "gates": {"test_zero_exact_reaction_overlap": True},
        "split": {
            "valid": {"annotations": {"c": {"scaffold_seen": True}}},
            "test": {"annotations": {"d": {"scaffold_seen": False}}},
        },
    }))
    output = tmp_path / "covariates.jsonl"
    report = build_covariates(source, split_dir, audit, output)
    assert report["rows"] == {"valid": 1, "test": 1}
    by_id = {row["source_id"]: row for row in map(json.loads, output.read_text().splitlines())}
    assert by_id["c"]["program_train_frequency"] == 0
    assert by_id["c"]["minimum_primitive_train_frequency"] == 1
    assert by_id["c"]["mean_primitive_train_frequency"] == 1.5
    assert by_id["c"]["fraction_primitives_seen_in_train"] == 1
    assert by_id["d"]["minimum_primitive_train_frequency"] == 2
    assert by_id["d"]["structural_overlap"] == {"scaffold_seen": False}
