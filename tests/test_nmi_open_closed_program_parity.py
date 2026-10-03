import hashlib
import json

import pytest

from scripts.audit_nmi_open_closed_program_parity import audit


def _fixture(tmp_path, *, mismatch=False, late_import=False, open_omits_last_import=False):
    root = tmp_path / "matched"
    imports = ["[OH:1]", "[CH3:2]"]
    moves = [{"source": {"kind": "LP", "atoms": [1]},
              "sink": {"kind": "BOND", "atoms": [1, 2]}, "electrons": 2}]
    for condition in ("open_flow", "closed_loop"):
        directory = root / condition
        directory.mkdir(parents=True)
        rows = {}
        hashes = {}
        for split in ("train", "valid", "test"):
            if condition == "open_flow":
                program = "<flow>\nOPEN_FLOW v1\n" + "\n".join(
                    f"IMPORT {fragment}" for fragment in imports[:1 if open_omits_last_import else None]
                ) + f"\nSTEP 0 {json.dumps(moves)}\nEXECUTE\n</flow>"
                messages = [{"role": "assistant", "content": program}]
            else:
                calls = [
                    {"name": "import_fragment", "arguments": {"fragment_smiles": imports[0]}},
                    {"name": "apply_coupled_electron_moves", "arguments": {
                        "moves": [{**moves[0], "electrons": 1 if mismatch and split == "test" else 2}]
                    }},
                    {"name": "import_fragment", "arguments": {"fragment_smiles": imports[1]}},
                    {"name": "finish_trace", "arguments": {}},
                ] if late_import else [
                    {"name": "import_fragment", "arguments": {"fragment_smiles": imports[0]}},
                    {"name": "import_fragment", "arguments": {"fragment_smiles": imports[1]}},
                    {"name": "apply_coupled_electron_moves", "arguments": {
                        "moves": [{**moves[0], "electrons": 1 if mismatch and split == "test" else 2}]
                    }},
                    {"name": "finish_trace", "arguments": {}},
                ]
                messages = [{"role": "assistant", "tool_calls": [
                    {"function": call} for call in calls
                ]}]
            row = {"source_id": f"reaction_{split}", "messages": messages}
            data = (json.dumps(row) + "\n").encode()
            (directory / f"{split}.jsonl").write_bytes(data)
            rows[split] = 1
            hashes[split] = hashlib.sha256(data).hexdigest()
        (directory / "manifest.json").write_text(json.dumps({
            "parent_split_manifest_sha256": "frozen-parent",
            "rows": rows, "output_sha256": hashes,
        }))
    return root


def test_open_closed_program_parity_and_late_import_accounting(tmp_path):
    root = _fixture(tmp_path, late_import=True)
    result = audit(root, tmp_path / "parity.json")
    assert result["passed"] is True
    assert result["splits"]["test"]["import_sequence_mismatch_count"] == 0
    assert result["splits"]["test"]["electron_step_sequence_mismatch_count"] == 0
    assert result["splits"]["test"]["closed_import_after_first_step_rows"] == 1


def test_open_closed_program_parity_rejects_move_drift(tmp_path):
    root = _fixture(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="electron steps differ"):
        audit(root, tmp_path / "parity.json")
    assert not (tmp_path / "parity.json").exists()


def test_open_closed_program_parity_reports_missing_late_import_without_filtering(tmp_path):
    root = _fixture(tmp_path, late_import=True, open_omits_last_import=True)
    result = audit(root, tmp_path / "parity.json")
    assert result["splits"]["test"]["import_sequence_mismatch_count"] == 1
    assert result["splits"]["test"]["import_multiset_mismatch_count"] == 1
    assert result["splits"]["test"]["electron_step_sequence_mismatch_count"] == 0
    assert result["splits"]["test"]["closed_minus_open_import_calls"] == 1
