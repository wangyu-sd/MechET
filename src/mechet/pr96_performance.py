"""Exact sampler metadata and narrowly scoped performance-resume validation."""


def arrow_sequence_lengths(dataset):
    """Read Arrow list offsets, not Python token lists; retain selected row order."""
    import pyarrow.compute as pc
    column = dataset.data.column("input_ids")
    lengths = pc.list_value_length(column)
    indices = getattr(dataset, "_indices", None)
    if indices is not None:
        lengths = pc.take(lengths, indices.column(0))
    if len(lengths) != len(dataset) or lengths.null_count:
        raise ValueError("PR96_INVALID_ARROW_SEQUENCE_LENGTHS")
    if len(lengths) and pc.min(lengths).as_py() < 1:
        raise ValueError("PR96_EMPTY_ARROW_SEQUENCE_LENGTH")
    if getattr(dataset, "_format_type", None) == "custom":
        raise ValueError("PR96_TRANSFORMED_ARROW_LENGTHS_NOT_AUTHORIZED")
    return lengths.to_numpy(zero_copy_only=False)


def largest_cached_example(cache_dirs):
    """Inspect Arrow offsets, then decode only the longest real training row."""
    import json
    from pathlib import Path
    from datasets import Dataset
    longest, audit = None, {"maximum_input_tokens": 0, "cache_count": 0}
    for cache in map(Path, cache_dirs):
        record = json.loads((cache / "manifest.json").read_text())["splits"]["train"]
        maximum, rows = 0, 0
        for original in record["arrow_files"]:
            dataset = Dataset.from_file(str(cache / Path(original).name))
            lengths = arrow_sequence_lengths(dataset)
            rows += len(lengths)
            index = int(lengths.argmax())
            size = int(lengths[index])
            maximum = max(maximum, size)
            if size > audit["maximum_input_tokens"]:
                longest = dataset[index]
                audit.update(maximum_input_tokens=size, cache=str(cache), row_index=index,
                             arrow_file=str(cache / Path(original).name))
        if rows != record["n_rows"] or maximum != record["max_input_tokens"]:
            raise ValueError("PR96_PERFORMANCE_PROBE_CACHE_AUDIT_MISMATCH")
        audit["cache_count"] += 1
    if longest is None:
        raise ValueError("PR96_PERFORMANCE_PROBE_NO_TRAINING_ROWS")
    return longest, audit


PERFORMANCE_TRAINING_KEYS = frozenset({
    "per_device_train_batch_size", "gradient_accumulation_steps", "gradient_checkpointing",
    "dataloader_num_workers", "dataloader_pin_memory", "dataloader_persistent_workers",
    "dataloader_prefetch_factor",
})
PERFORMANCE_CODE_PATHS = frozenset({
    "src/mechet/pr96_trainer.py", "src/mechet/pr96_artifacts.py",
    "src/mechet/pr96_performance.py",
})


def validate_performance_identity_change(source, target):
    """Allow only explicit compute-layout changes, never objective/data changes."""
    fixed = lambda value: {k: v for k, v in value.items() if k not in ("training", "code_sha256")}
    if fixed(source) != fixed(target):
        raise ValueError("PR96_PERFORMANCE_SCIENTIFIC_IDENTITY_CHANGED")
    before, after = source["training"], target["training"]
    fixed_training = lambda value: {k: v for k, v in value.items() if k not in PERFORMANCE_TRAINING_KEYS}
    if fixed_training(before) != fixed_training(after):
        raise ValueError("PR96_PERFORMANCE_TRAINING_OBJECTIVE_CHANGED")
    def per_rank_batch(value):
        micro, accum = value.get("per_device_train_batch_size"), value.get("gradient_accumulation_steps")
        if type(micro) is not int or type(accum) is not int or min(micro, accum) < 1:
            raise ValueError("PR96_PERFORMANCE_INVALID_BATCH")
        return micro * accum
    if per_rank_batch(before) != per_rank_batch(after):
        raise ValueError("PR96_PERFORMANCE_GLOBAL_BATCH_CHANGED")
    old_code, new_code = source["code_sha256"], target["code_sha256"]
    differences = {name for name in set(old_code) | set(new_code)
                   if old_code.get(name) != new_code.get(name)}
    if not differences <= PERFORMANCE_CODE_PATHS or set(old_code) - set(new_code):
        raise ValueError("PR96_PERFORMANCE_UNAUTHORIZED_CODE_CHANGE")


def identity_sha256(value):
    import hashlib
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_performance_resume_authorization(cfg, checkpoint):
    """Bind one complete resume anchor and its existing validation-best ancestor."""
    import json
    from pathlib import Path
    from .pr96_artifacts import training_identity, _verify_checkpoint_components
    from .pr96_eval_runtime import sha
    checkpoint = Path(checkpoint).resolve()
    metadata_path = checkpoint / "pr96_checkpoint_manifest.json"
    source = json.loads(metadata_path.read_text())
    target_identity = training_identity(cfg)
    source_identity = source["lineage"]["training_identity"]
    validate_performance_identity_change(source_identity, target_identity)
    _verify_checkpoint_components(source, checkpoint)
    if (source["track"], source["stage"]) != (cfg["pr96"]["track"], cfg["pr96"]["stage"]):
        raise ValueError("PR96_PERFORMANCE_WRONG_TRACK_OR_STAGE")
    allowed = {str(checkpoint): sha(metadata_path)}
    best = source.get("best_checkpoint")
    if best and Path(best).resolve() != checkpoint:
        ancestor = Path(best).resolve()
        prior_path = ancestor / "pr96_checkpoint_manifest.json"
        prior = json.loads(prior_path.read_text())
        if (prior["lineage"]["training_identity"] != source_identity or prior["step"] > source["step"]
                or (prior["track"], prior["stage"]) != (source["track"], source["stage"])):
            raise ValueError("PR96_PERFORMANCE_BEST_ANCESTOR_MISMATCH")
        _verify_checkpoint_components(prior, ancestor)
        allowed[str(ancestor)] = sha(prior_path)
    return {"artifact_type": "pr96_performance_resume_authorization_v1",
            "source_checkpoint": str(checkpoint), "source_step": source["step"],
            "source_epoch": source["epoch"],
            "source_identity_sha256": identity_sha256(source_identity),
            "target_identity_sha256": identity_sha256(target_identity),
            "allowed_source_checkpoints": allowed,
            "training_changes": {k: {"before": source_identity["training"].get(k),
                                       "after": target_identity["training"].get(k)}
                                 for k in PERFORMANCE_TRAINING_KEYS
                                 if source_identity["training"].get(k) != target_identity["training"].get(k)},
            "preserves_optimizer_scheduler_rng_and_data_position": True,
            "requires_renewed_deployed_preflight": True}


def verify_performance_checkpoint_identity(cfg, checkpoint, metadata, target_identity):
    """No silent relaxation: require exact anchor hash plus a hashed authorization."""
    import json
    from pathlib import Path
    from .pr96_eval_runtime import sha
    authorization = cfg["pr96"].get("performance_resume_authorization") or {}
    path = Path(authorization.get("path", "/nonexistent-pr96-resume-authorization"))
    if not path.is_file() or sha(path) != authorization.get("sha256"):
        raise ValueError("PR96_RESUME_PERFORMANCE_AUTHORIZATION_REQUIRED")
    receipt = json.loads(path.read_text())
    source_identity = metadata["lineage"]["training_identity"]
    if (receipt.get("artifact_type") != "pr96_performance_resume_authorization_v1"
            or receipt.get("source_identity_sha256") != identity_sha256(source_identity)
            or receipt.get("target_identity_sha256") != identity_sha256(target_identity)
            or receipt.get("allowed_source_checkpoints", {}).get(str(Path(checkpoint).resolve()))
               != sha(Path(checkpoint) / "pr96_checkpoint_manifest.json")):
        raise ValueError("PR96_RESUME_PERFORMANCE_AUTHORIZATION_MISMATCH")
    validate_performance_identity_change(source_identity, target_identity)


def select_performance_resume_checkpoint(cfg):
    """Pin the handoff anchor; later recover only complete new-identity saves.

    A directory alone is not a completed checkpoint. The old process may have
    started another save before shutdown; do not let that override the exact
    authenticated anchor. Corrupted complete checkpoints still fail closed.
    """
    import json
    from pathlib import Path
    from .pr96_artifacts import training_identity, verify_resume_checkpoint
    directory = Path(cfg["output_dir"]).resolve()
    anchor = Path(cfg["pr96"]["performance_resume_anchor"]).resolve()
    if anchor.parent != directory:
        raise ValueError("PR96_PERFORMANCE_RESUME_ANCHOR_OUTSIDE_RUN")
    identity = training_identity(cfg)
    candidates = sorted(directory.glob("checkpoint-*/pr96_checkpoint_manifest.json"),
                        key=lambda p: int(p.parent.name.split("-")[-1]), reverse=True)
    anchor_step = int(anchor.name.split("-")[-1])
    for manifest in candidates:
        if int(manifest.parent.name.split("-")[-1]) <= anchor_step:
            break
        metadata = json.loads(manifest.read_text())
        if metadata.get("lineage", {}).get("training_identity") == identity:
            verify_resume_checkpoint(cfg, manifest.parent)
            return manifest.parent
    if not anchor.is_dir():
        raise ValueError("PR96_PERFORMANCE_RESUME_ANCHOR_MISSING")
    return anchor
