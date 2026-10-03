import hashlib
import json

from scripts.build_nmi_primitive_unseen_control import build_control


def test_official_valid_unseen_primitive_control_is_disjoint_from_h2_train(tmp_path):
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    train.write_text(json.dumps({"source_id": "train1", "metadata": {
        "execution_primitive_signatures": ["p1"],
        "execution_composition_signature": "program_a",
    }}) + "\n")
    valid.write_text("".join(json.dumps(row) + "\n" for row in [
        {"source_id": "valid1", "metadata": {"executor_replayed": True, "execution_primitive_signatures": ["p1"], "execution_composition_signature": "program_a"}},
        {"source_id": "valid2", "metadata": {"executor_replayed": True, "execution_primitive_signatures": ["p2"], "execution_composition_signature": "program_b"}},
        {"source_id": "valid3", "metadata": {"executor_replayed": True, "execution_primitive_signatures": ["p1"], "execution_composition_signature": "program_c"}},
    ]))
    split = tmp_path / "split"
    split.mkdir()
    ids = b"train1\n"
    (split / "train.ids.txt").write_bytes(ids)
    (split / "manifest.json").write_text(json.dumps({"split_id_sha256": {"train": hashlib.sha256(ids).hexdigest()}}))
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(json.dumps({"splits": {
        "train": {"rows": 1, "sha256": hashlib.sha256(train.read_bytes()).hexdigest()},
        "valid": {"rows": 3, "sha256": hashlib.sha256(valid.read_bytes()).hexdigest()},
    }}))
    out = tmp_path / "out"
    report = build_control(train, valid, split, source_manifest, out)
    assert report["negative_control_reactions"] == 1
    assert report["program_seen_reactions"] == 1
    assert report["program_unseen_primitive_seen_reactions"] == 1
    assert (out / "primitive_unseen.ids.txt").read_text() == "valid2\n"
    assert (out / "program_seen.ids.txt").read_text() == "valid1\n"
    assert (out / "program_unseen_primitive_seen.ids.txt").read_text() == "valid3\n"
