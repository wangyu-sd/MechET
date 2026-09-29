import hashlib
import json
from pathlib import Path

from scripts.autoresearch.build_r1_multireference import build, disconnection_stratum


def test_disconnection_strata_are_explicit() -> None:
    assert disconnection_stratum([{(1, 2)}, {(1, 2)}]) == "similar"
    assert disconnection_stratum([{(1, 2)}, {(2, 3)}]) == "different"
    assert disconnection_stratum([set(), {(2, 3)}]) == "unavailable"


def test_r1_requires_two_distinct_heldout_records(tmp_path: Path) -> None:
    source = tmp_path / "test.jsonl"
    rows = [
        {"id": "a", "source_id": "a", "target_smiles": "[CH3:1][OH:2]",
         "structural_precursor": "[CH3:1][Br:3].[OH-:2]"},
        {"id": "b", "source_id": "b", "target_smiles": "[CH3:1][OH:2]",
         "structural_precursor": "[CH3:1][Cl:4].[OH-:2]"},
        {"id": "c", "source_id": "c", "target_smiles": "CC",
         "structural_precursor": "C.C"},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "official.json"
    manifest.write_text(json.dumps({"splits": {"test": {
        "output_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}}))
    report = build(source, manifest, tmp_path / "out")
    assert report["products"] == 1
    cohort = json.loads((tmp_path / "out/r1_multi_reference.jsonl").read_text())
    assert cohort["reference_count"] == 2
    assert {support for item in cohort["references"]
            for support in item["support_record_ids"]} == {"a", "b"}
