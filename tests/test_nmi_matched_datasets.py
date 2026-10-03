import hashlib
import json

import pytest

from scripts.materialize_nmi_matched_datasets import (
    materialize, normalize_closed_loop_budget, strict_trace_to_direct,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_jsonl(path, rows):
    content = b"".join((json.dumps(row) + "\n").encode() for row in rows)
    path.write_bytes(content)
    return _sha(content)


def _split_fixture(tmp_path):
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    ids = {
        "train": ["flower_mech_proof_train_1"],
        "valid": ["flower_mech_proof_train_2"],
        "test": ["flower_mech_proof_train_3"],
    }
    hashes = {}
    for split, values in ids.items():
        content = ("\n".join(values) + "\n").encode()
        (split_dir / f"{split}.ids.txt").write_bytes(content)
        hashes[split] = _sha(content)
    (split_dir / "manifest.json").write_text(json.dumps({
        "rows": {split: len(values) for split, values in ids.items()},
        "split_id_sha256": hashes,
    }))
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "artifact_type": "nmi_mechcomp_structural_overlap_v2",
        "scope": "full_frozen_split",
        "split_manifest_sha256": _sha((split_dir / "manifest.json").read_bytes()),
        "reaction_center_undefined_count": 0,
        "gates": {"test_zero_exact_reaction_overlap": True,
                  "valid_zero_exact_reaction_overlap": True},
    }))
    return split_dir, audit


@pytest.mark.parametrize("condition", ["direct", "open_flow", "closed_loop"])
def test_materialize_matches_frozen_ids_and_provenance(tmp_path, condition):
    split_dir, audit = _split_fixture(tmp_path)
    source = tmp_path / "source.jsonl"
    source_ids = ["flower_mech_proof_train_1", "flower_mech_proof_train_2", "flower_mech_proof_train_3"]
    rows = [{"id": identifier, "source_id": identifier,
             "target_smiles": "[CH4:1]", "structural_precursor": "[CH3:1][OH:2]",
             "messages": [{"role": "user", "content": "same product"}],
             "metadata": {"original": True}} for identifier in source_ids]
    if condition == "closed_loop":
        for row in rows:
            row["messages"] = [
                {"role": "user", "content": "TARGET: X\nINITIAL ENVIRONMENT OBSERVATION:\n" +
                 json.dumps({"task": "trace_owned_inverse_electron_flow", "max_tool_calls": 12})},
                {"role": "assistant", "content": "", "tool_calls": []},
                {"role": "tool", "name": "apply_coupled_electron_moves",
                 "content": json.dumps({"ok": True, "remaining_tool_calls": 11})},
            ]
    sha = _write_jsonl(source, rows)
    output = tmp_path / "out"
    report = materialize(source, condition=condition, source_sha256=sha,
                         source_rows=len(rows), split_dir=split_dir, output_dir=output,
                         structural_audit=audit)
    assert report["rows"] == {"train": 1, "valid": 1, "test": 1}
    assert report["out_of_strict_universe_source_rows"] == 0
    assert report["training_allowed"] is True
    assert all(_sha((output / f"{split}.jsonl").read_bytes()) == report["output_sha256"][split]
               for split in ("train", "valid", "test"))
    assert json.loads((output / "test.jsonl").read_text())["source_id"] == "flower_mech_proof_train_3"
    assert json.loads((output / "test.jsonl").read_text())["metadata"]["nmi_split"] == "test"
    if condition == "closed_loop":
        result = json.loads((output / "test.jsonl").read_text())
        assert "\"max_tool_calls\": 40" in result["messages"][0]["content"]
        assert json.loads(result["messages"][2]["content"])["remaining_tool_calls"] == 39
    if condition == "direct":
        result = json.loads((output / "test.jsonl").read_text())
        assert result["messages"][-1]["content"] == "<answer>\n[CH3:1][OH:2]\n</answer>"
        assert result["metadata"]["mechanism_supervision"] is False
        assert "tools" not in result
    with pytest.raises(FileExistsError):
        materialize(source, condition=condition, source_sha256=sha,
                    source_rows=len(rows), split_dir=split_dir, output_dir=output,
                    structural_audit=audit)


def test_materialize_blocks_failed_structural_gate(tmp_path):
    split_dir, audit = _split_fixture(tmp_path)
    record = json.loads(audit.read_text())
    record["gates"]["test_zero_exact_reaction_overlap"] = False
    audit.write_text(json.dumps(record))
    source = tmp_path / "source.jsonl"
    sha = _write_jsonl(source, [{"source_id": "flower_mech_proof_train_1"}])
    with pytest.raises(ValueError, match="did not pass"):
        materialize(source, condition="direct", source_sha256=sha,
                    source_rows=1, split_dir=split_dir,
                    output_dir=tmp_path / "out", structural_audit=audit)
    assert not (tmp_path / "out").exists()


def test_fixed_budget_rejects_inconsistent_gold_tool_feedback():
    row = {"messages": [
        {"role": "user", "content": "TARGET: X\nINITIAL ENVIRONMENT OBSERVATION:\n" +
         json.dumps({"max_tool_calls": 12})},
        {"role": "tool", "content": json.dumps({"remaining_tool_calls": 3})},
    ]}
    with pytest.raises(ValueError, match="budget/count mismatch"):
        normalize_closed_loop_budget(row)


def test_direct_conversion_does_not_copy_trace_or_tools():
    row = {"id": "x", "source_id": "x", "target_smiles": "[CH4:1]",
           "structural_precursor": "[CH3:1][OH:2]", "tools": [{"secret": "tool"}],
           "metadata": {"trace_plan": [{"gold": "move"}]}}
    direct = strict_trace_to_direct(row)
    assert "tools" not in direct
    assert "trace_plan" not in direct["metadata"]
    assert direct["messages"][1]["content"] == "TARGET: [CH4:1]"
