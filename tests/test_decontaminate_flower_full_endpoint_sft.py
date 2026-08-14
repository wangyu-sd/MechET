from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from decontaminate_flower_full_endpoint_sft import reaction_key, write_partition


def _row(identifier: str, product: str, precursor: str):
    return {
        "id": identifier,
        "target_smiles": product,
        "structural_precursor": precursor,
    }


def test_reaction_key_ignores_maps_and_fragment_order():
    left = _row("a", "[CH3:1][OH:2]", "[Na+:3].[CH3:1][Br:4]")
    right = _row("b", "[CH3:9][OH:8]", "[CH3:9][Br:7].[Na+:6]")
    assert reaction_key(left) == reaction_key(right)


def test_write_partition_quarantines_blocked_keys(tmp_path: Path):
    keep = _row("keep", "[CH4:1]", "[CH3:1][Br:2]")
    drop = _row("drop", "[NH3:3]", "[NH2:3][Cl:4]")
    report = write_partition(
        iter([keep, drop]),
        blocked_keys={reaction_key(drop)},
        kept_path=tmp_path / "train.jsonl",
        quarantine_path=tmp_path / "quarantine.jsonl",
    )
    assert report == {"source_rows": 2, "kept": 1, "removed": 1}
    assert '"keep"' in (tmp_path / "train.jsonl").read_text()
    assert '"drop"' in (tmp_path / "quarantine.jsonl").read_text()
