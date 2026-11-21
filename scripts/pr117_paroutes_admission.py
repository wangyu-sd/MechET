"""Verify the complete frozen PaRoutes n5 inputs and estimate one-condition work."""
import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path


def digest(raw):
    return sha256(raw).hexdigest()


def admission(run, source, output):
    from mechet.route_training_contract import chemical_identity

    loaded = {}

    def read(path, expected=None):
        raw = path.read_bytes()
        value = digest(raw)
        if expected is not None and value != expected:
            raise ValueError(f"Source hash mismatch: {path.name}")
        loaded[str(path)] = value
        return raw

    reference = json.loads(read(run / "reference_1084_full_target_lock_20261121.json"))
    release = source / "hosted_frozen_20261116"
    routes = source / "paroutes/v2026-06-05/n5-routes"
    catalog = dict(line.split("  ", 1)[::-1] for line in
                   read(release / "SHA256SUMS", reference["source_catalog_sha256"]).decode().splitlines() if line)

    def hosted(relative):
        return read(release / relative, catalog[relative])

    definition = "1-benchmarks/definitions/paroutes-n5-full-pruned"
    stock_name = "n5-stock"
    stock_file = f"1-benchmarks/stocks/{stock_name}"
    target_manifest = json.loads(hosted(definition + ".manifest.json"))
    stock_manifest = json.loads(hosted(stock_file + ".manifest.json"))
    document = json.loads(gzip.decompress(hosted(definition + ".json.gz")))
    stock_raw = gzip.decompress(hosted(stock_file + ".txt.gz"))
    if (document["default_constraints"] != [{"kind": "retrocast.stock_termination", "stock": stock_name}]
            or target_manifest["statistics"]["n_targets"] != 10000
            or target_manifest["statistics"]["n_targets_with_pruning"] != 0
            or len(document["targets"]) != 10000
            or catalog[stock_file + ".txt.gz"] != reference["stocks"][stock_name]["source_sha256"]):
        raise ValueError("Expected the complete public n5 definition under its own stock")
    stock = {chemical_identity(line) for line in stock_raw.decode().splitlines() if line}
    if len(stock) != stock_manifest["statistics"]["unique_molecules"]:
        raise ValueError("Executor normalization changed the declared stock population")

    roots = {name: chemical_identity(row["smiles"]) for name, row in document["targets"].items()}
    route_manifest = json.loads(read(routes / "manifest.json"))
    source_hash = route_manifest["source_files"][0]["file_hash"]
    if source_hash != target_manifest["source_files"][0]["file_hash"]:
        raise ValueError("The frozen exports use different raw PaRoutes route inputs")
    rows = gzip.decompress(read(routes / "all.jsonl.gz", route_manifest["output_files"]["all"]["file_hash"]))
    raw_ids = set()
    for line in rows.splitlines():
        row = json.loads(line)
        origin, = row["sources"]
        target_id = f"n5-{origin['raw_index'] + 1:05d}"
        if (origin["dataset"] != "n5" or target_id in raw_ids
                or chemical_identity(row["route"]["target"]["smiles"]) != roots[target_id]):
            raise ValueError("A target differed across source routes and the benchmark definition")
        raw_ids.add(target_id)
    identities = set(roots.values())
    if raw_ids != set(roots) or len(identities) != 10000 or identities.intersection(stock):
        raise ValueError("Unexpected duplicate, missing or zero-reaction n5 target")

    exposure = gzip.decompress(read(run / "exact_training_target_index_20261121.jsonl.gz", reference["exposure_index_sha256"]))
    training = dict(json.loads(line) for line in exposure.splitlines())
    by_training_origin = Counter(origin for target in identities for origin in training.get(target, ()))
    overlap = {"reference1084": sum(row["target"] in identities for row in reference["targets"])}
    same_stock_overlap = sum(row["target"] in identities and row["stock"] == stock_name for row in reference["targets"])
    reference_identities = {row["target"] for row in reference["targets"]}
    for label, path in (("reference_heldout", "reference_confirmation_frozen_20261122/panel.json"),
                        ("reference_development", "reference_panel.json")):
        panel = json.loads(read(run / path))["targets"]
        overlap[label] = sum(row["target"] in identities for row in panel)

    freeze = json.loads(read(run / "stage_a_final_method_freeze_20261122.json"))
    phase_a_summary = "stage_a_final_verified/complete.json"
    development = json.loads(read(run / phase_a_summary, freeze["artifact_sha256"][phase_a_summary]))
    cap = freeze["confirmation"]["target_request_ceiling"]
    calls = len(roots) * cap
    rate = development["native_requests"] / development["GPU_hours"]
    evidence = dict(status="INPUTS_VERIFIED_GPU_ADMISSION_PENDING", complete_targets=len(roots),
        canonical_distinct_targets=len(identities), stock=stock_name, stock_identities=len(stock),
        stock_sha256=catalog[stock_file + ".txt.gz"], exact_matches_across_source_exports=len(raw_ids),
        already_stock_targets=0, raw_source_route_sha256=source_hash, overlap_with_reference=overlap,
        reference1084_overlap_at_same_stock=same_stock_overlap,
        exact_training_target_matches=len(identities.intersection(training)),
        training_target_matches_by_origin=dict(by_training_origin),
        training_scope="Exact identity of the target molecule in the audited training manifests",
        admission="One frozen condition if measured remaining independent accelerator capacity permits",
        projection=dict(assumed_call_cap=cap, max_native_calls=calls, observed_development_calls=development["native_requests"],
            observed_development_a100_worker_seconds=development["GPU_hours"] * 3600,
            observed_native_calls_per_a100_hour=rate, max_a100_hours_at_observed_rate=calls/rate,
            scope="Call ceiling at the measured Phase A rate; refresh from completed formal evaluations before launch"),
        model_calls=0, evaluated_targets=0, source_sha256=loaded)
    payload, annotations = [], []
    for target_id, target in sorted(roots.items()):
        seed = int(digest(f"PR117:public-paroutes:17:{target_id}".encode())[:8], 16)
        payload.append(json.dumps(dict(target_id=target_id, target=target, stock=stock_name, seed=seed)))
        annotations.append(json.dumps(dict(target_id=target_id, reference1084=target in reference_identities,
                                           exact_training_origins=training.get(target, []))))
    for name, lines in (("targets_only", payload), ("target_annotations", annotations)):
        compressed = gzip.compress(("\n".join(lines) + "\n").encode(), mtime=0)
        with (output / f"{name}.jsonl.gz").open("xb") as stream:
            stream.write(compressed)
        evidence[f"{name}_sha256"] = digest(compressed)
    with (output / "evidence.json").open("x") as file:
        json.dump(evidence, file, indent=2)
        file.write("\n")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "source", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if any(not path.resolve().is_relative_to("/aaa/fionafyang/buddy1") for path in vars(args).values()):
        parser.error("PaRoutes artifacts belong on shared storage")
    result = admission(args.run.resolve(), args.source.resolve(), args.output.resolve())
    print(json.dumps({k: result[k] for k in ("complete_targets", "stock_identities", "overlap_with_reference",
          "exact_training_target_matches", "training_target_matches_by_origin", "projection")}))


if __name__ == "__main__":
    main()
