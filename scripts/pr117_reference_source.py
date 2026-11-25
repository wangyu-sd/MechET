"""Lock the entire public Reference population before selecting a method."""

from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json


PUBLIC_COHORTS = {"ref-lin-600": 600, "ref-cnv-400": 400, "ref-lng-84": 84}


def source_bytes(path, expected):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"Source digest changed: {path}")
    return raw


def build_source_lock(root, catalog_sha256, *, normalize, remove_stereo, train_origins, screen_targets):
    root = root.resolve()
    catalog = {}
    for line in source_bytes(root / "SHA256SUMS", catalog_sha256).decode().splitlines():
        if line.strip():
            expected, path = line.split(maxsplit=1)
            path = path.lstrip("*")
            if path in catalog:
                raise ValueError(f"Duplicate release catalog path: {path}")
            catalog[path] = expected

    train_connectivity = defaultdict(set)
    for target, origins in train_origins.items():
        train_connectivity[remove_stereo(target)].update(origins)
    explored_connectivity = {remove_stereo(target) for target in screen_targets}
    stocks, inventory, definitions, annotations, targets = {}, {}, {}, {}, []
    for cohort, count in PUBLIC_COHORTS.items():
        definition_path = f"1-benchmarks/definitions/{cohort}.json.gz"
        manifest_path = f"1-benchmarks/definitions/{cohort}.manifest.json"
        definition = json.loads(gzip.decompress(source_bytes(root / definition_path, catalog[definition_path])))
        manifest = json.loads(source_bytes(root / manifest_path, catalog[manifest_path]))
        if (definition["name"] != cohort or len(definition["targets"]) != count
                or manifest["statistics"]["n_targets"] != count
                or [item["file_hash"] for item in manifest["output_files"]] != [catalog[definition_path]]):
            raise ValueError(f"Definition and release manifest disagree: {cohort}")
        constraint, = definition["default_constraints"]
        if definition["constraints"] or constraint["kind"] != "retrocast.stock_termination":
            raise ValueError(f"Unsupported release constraints: {cohort}")
        stock = constraint["stock"]
        if stock not in stocks:
            relative = f"1-benchmarks/stocks/{stock}.txt.gz"
            path = root / relative
            raw = source_bytes(path, catalog[relative])
            inventory[stock] = {normalize(molecule) for molecule in gzip.decompress(raw).decode().splitlines()
                                if molecule}
            stocks[stock] = dict(source_path=str(path), source_sha256=catalog[relative],
                                 size_after_executor_identity=len(inventory[stock]))

        annotations[cohort] = {}
        exposure = Counter(exact_train=0, connectivity_train=0, reused_development=0, already_stock=0)
        for target_id, item in sorted(definition["targets"].items()):
            if target_id != item["id"]:
                raise ValueError(f"Release target identifier changed: {cohort}/{target_id}")
            target = normalize(item["smiles"])
            connectivity = remove_stereo(target)
            exact = sorted(train_origins.get(target, ()))
            relaxed = sorted(train_connectivity.get(connectivity, ()))
            explored = connectivity in explored_connectivity
            already_stock = target in inventory[stock]
            exposure.update(exact_train=int(bool(exact)), connectivity_train=int(bool(relaxed)),
                            reused_development=int(explored), already_stock=int(already_stock))
            annotations[cohort][target_id] = dict(
                exact_train_sources=exact, connectivity_train_sources=relaxed,
                reused_development=explored, already_stock=already_stock)
            seed_hex = sha256(f"PR117:public-reference:17:{cohort}:{target_id}".encode()).hexdigest()[:8]
            targets.append(dict(cohort=cohort, target_id=target_id, target=target, stock=stock, seed=int(seed_hex, 16)))
        definitions[cohort] = dict(targets=count, stock=stock, definition_sha256=catalog[definition_path],
                                   manifest_sha256=catalog[manifest_path], exposure_counts=dict(exposure))

    return dict(status="FULL_PUBLIC_REFERENCE_TARGETS_LOCKED", model_calls=0, seed=17,
                source_catalog_sha256=catalog_sha256, total_targets=len(targets), cohorts=definitions,
                stocks=stocks, targets=targets, target_annotations=annotations)
