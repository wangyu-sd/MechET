import gzip
from hashlib import sha256
import json

import pytest

from scripts import pr117_reference_source as source


def released_files(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "PUBLIC_COHORTS", {"linear": 2, "long": 1})
    catalog = {}

    def save(relative, data):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        catalog[relative] = sha256(data).hexdigest()

    for cohort, rows, stock in (("linear", {"r0": "a", "r1": "b"}, "small"),
                                ("long", {"r0": "a-stereo"}, "wide")):
        definition = dict(name=cohort, default_constraints=[
            dict(kind="retrocast.stock_termination", stock=stock)], constraints=[],
            targets={key: dict(id=key, smiles=smiles, acceptable_routes=["private"])
                     for key, smiles in rows.items()})
        prefix = f"1-benchmarks/definitions/{cohort}"
        path = prefix + ".json.gz"
        save(path, gzip.compress(json.dumps(definition).encode()))
        manifest = dict(output_files=[{"file_hash": catalog[path]}],
                        statistics={"n_targets": len(rows)})
        save(prefix + ".manifest.json", json.dumps(manifest).encode())
    save("1-benchmarks/stocks/small.txt.gz", gzip.compress(b"a\n"))
    save("1-benchmarks/stocks/wide.txt.gz", gzip.compress(b"a-stereo\nc\n"))
    raw = "".join(f"{expected}  {path}\n" for path, expected in catalog.items()).encode()
    (tmp_path / "SHA256SUMS").write_bytes(raw)
    return sha256(raw).hexdigest()


def test_public_lock_keeps_exposed_and_already_stock_targets(tmp_path, monkeypatch):
    digest = released_files(tmp_path, monkeypatch)
    lock = source.build_source_lock(
        tmp_path, digest, normalize=str, remove_stereo=lambda s: s.removesuffix("-stereo"),
        train_origins={"a": {"stage2_history"}}, screen_targets={"b"})
    assert lock["total_targets"] == 3
    assert lock["cohorts"]["linear"]["targets"] == 2
    assert lock["cohorts"]["long"]["stock"] == "wide"
    assert [r["target_id"] for r in lock["targets"]] == ["r0", "r1", "r0"]
    assert all(set(row) == {"cohort", "target_id", "target", "stock", "seed"}
               for row in lock["targets"])
    assert lock["cohorts"]["linear"]["exposure_counts"] == dict(
        exact_train=1, connectivity_train=1, reused_development=1, already_stock=1)
    assert lock["cohorts"]["long"]["exposure_counts"] == dict(
        exact_train=0, connectivity_train=1, reused_development=0, already_stock=1)
    assert lock["target_annotations"]["long"]["r0"]["connectivity_train_sources"] == ["stage2_history"]


@pytest.mark.parametrize("relative", ["1-benchmarks/stocks/small.txt.gz",
                                      "1-benchmarks/definitions/long.manifest.json",
                                      "1-benchmarks/definitions/linear.json.gz",
                                      "SHA256SUMS"])
def test_changed_source_fails_admission(tmp_path, monkeypatch, relative):
    digest = released_files(tmp_path, monkeypatch)
    changed = tmp_path / relative
    changed.write_bytes(changed.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="Source digest changed"):
        source.build_source_lock(tmp_path, digest, normalize=str, remove_stereo=str,
                                 train_origins={}, screen_targets=set())
