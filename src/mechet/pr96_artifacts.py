"""Fail-closed bindings between PR96 references, token caches and own parents."""
from .qwen35_protocol import OFFICIAL_MODEL, OFFICIAL_REVISION
from .pr96_validation import stage_one_gate


def training_identity(cfg):
    """Scientific contract, excluding node-local paths and renewed preflight."""
    from pathlib import Path
    from .pr96_eval_runtime import sha
    settings = cfg["pr96"]
    root = Path(__file__).resolve().parents[2]
    code = ["src/mechet/pr96_trainer.py", "src/mechet/qwen35_protocol.py",
            "src/mechet/collator.py", "scripts/train_tool_sft.py",
            "scripts/prepare_tool_sft_arrow.py", "src/mechet/pr96_evaluation.py",
            "src/mechet/pr96_eval_runtime.py", "src/mechet/pr96_artifacts.py"]
    identity = {"model": cfg["model_name_or_path"], "training": cfg["training"],
                "lora": cfg.get("lora", {}), "contract": cfg.get("contract", {}),
                "bindings": {k: settings[k] for k in ("track", "stage", "base_manifest_sha256",
                    "reference_approval_sha256", "cohorts_sha256", "token_cache_manifest_sha256")},
                "code_sha256": {name: sha(root / name) for name in code}}
    if settings["stage"] == "history":
        identity["parent_sha256"] = sha(Path(cfg["initial_adapter_path"]) / "adapter_model.safetensors")
    return identity


def verify_resume_checkpoint(cfg, checkpoint):
    import json
    from pathlib import Path
    from .pr96_eval_runtime import sha
    checkpoint = Path(checkpoint)
    value = json.loads((checkpoint / "pr96_checkpoint_manifest.json").read_text())
    if (value.get("track") != cfg["pr96"]["track"] or value.get("stage") != cfg["pr96"]["stage"]
            or value.get("lineage", {}).get("training_identity") != training_identity(cfg)):
        raise ValueError("PR96_RESUME_SCIENTIFIC_LINEAGE_MISMATCH")
    required = {"adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt", "trainer_state.json"}
    if set(value.get("files", {})) != required:
        raise ValueError("PR96_RESUME_COMPONENTS_MISSING")
    for key, names in (("rng_files", {f"rng_state_{r}.pth" for r in range(8)}),
                       ("counter_files", {f"pr96_counters.rank{r:02d}.json" for r in range(8)})):
        if set(value.get(key, {})) != names:
            raise ValueError("PR96_RESUME_RANK_STATE_MISSING")
    for key in ("files", "rng_files", "counter_files"):
        for name, digest in value[key].items():
            path = checkpoint / name
            if not path.is_file() or sha(path) != digest:
                raise ValueError(f"PR96_RESUME_COMPONENT_HASH_MISMATCH:{name}")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if state["global_step"] != value["step"] or state["epoch"] != value["epoch"]:
        raise ValueError("PR96_RESUME_TRAINER_POSITION_MISMATCH")
    return value


def verify_evaluation_bindings(cfg, adapter):
    import json
    from pathlib import Path
    from .pr96_eval_runtime import sha
    settings = cfg["pr96"]
    approval_path = Path(settings["reference_approval"])
    if sha(approval_path) != settings["reference_approval_sha256"]:
        raise ValueError("PR96_EVALUATION_APPROVAL_CHANGED")
    approval = json.loads(approval_path.read_text())
    manifest_path = Path(approval["reference_manifest"])
    if (approval.get("track") != settings["track"] or not approval.get("reference_training_authorized")
            or sha(manifest_path) != approval["reference_manifest_sha256"]
            or manifest_path.parent.resolve() != Path(settings["reference_dir"]).resolve()
            or sha(settings["cohorts_file"]) != settings["cohorts_sha256"]
            or settings["cohorts_sha256"] != approval["frozen_cohorts_sha256"]):
        raise ValueError("PR96_EVALUATION_REFERENCE_OR_COHORT_CHANGED")
    reference = json.loads(manifest_path.read_text())
    sources = {}
    for kind in ("inverse", "state_sft"):
        path = manifest_path.parent / kind / "valid.jsonl"
        digest = sha(path)
        if digest != reference["reports"]["valid"]["outputs"][kind]["sha256"]:
            raise ValueError(f"PR96_EVALUATION_VALID_SOURCE_CHANGED:{kind}")
        sources[kind] = digest
    adapter = Path(adapter)
    metadata = json.loads((adapter / "pr96_checkpoint_manifest.json").read_text())
    weight = sha(adapter / "adapter_model.safetensors")
    if (metadata.get("track") != settings["track"] or metadata.get("stage") != settings["stage"]
            or metadata.get("lineage", {}).get("training_identity") != training_identity(cfg)
            or metadata["files"].get("adapter_model.safetensors") != weight):
        raise ValueError("PR96_EVALUATION_CHECKPOINT_LINEAGE_CHANGED")
    return {"valid_source_sha256": sources, "reference_approval_sha256": sha(approval_path),
            "cohorts_sha256": sha(settings["cohorts_file"]), "adapter_model_sha256": weight,
            "training_identity": training_identity(cfg)}


def validate_preflight(value, track):
    required = ("finite_nonzero_gradients", "parameters_updated", "assistant_mask_verified",
                "tool_prefix_parity", "all_language_targets_present", "language_only", "generation_completed",
                "ddp_gradient_synchronization")
    ranks = value.get("ranks", [])
    if value.get("track") != track or value.get("base_revision") != OFFICIAL_REVISION:
        raise ValueError("PR96_PREFLIGHT_LINEAGE_MISMATCH")
    if len(ranks) != 8 or {r["rank"] for r in ranks} != set(range(8)):
        raise ValueError("PR96_PREFLIGHT_REQUIRES_ALL_EIGHT_RANKS")
    expected = "A100" if track == "mech31k" else "H20"
    for rank in ranks:
        if len(rank["gpu_names"]) != 8 or not all(expected in n for n in rank["gpu_names"]):
            raise ValueError("PR96_PREFLIGHT_WRONG_GPU_TYPE")
        if not all(rank.get(k) is True for k in required):
            raise ValueError("PR96_PREFLIGHT_FAILED_COMPATIBILITY_CHECK")


def validate_tokens(cache, expected):
    if (cache.get("model_name_or_path") != OFFICIAL_MODEL or cache.get("model_revision") != OFFICIAL_REVISION
            or cache.get("encoding_protocol") != "qwen35_tool_span_v1"):
        raise ValueError("PR96_TOKENIZER_LINEAGE_MISMATCH")
    for split, (digest, rows) in expected.items():
        record = cache["splits"][split]
        if cache["sources"][split]["sha256"] != digest or record["n_rows"] != rows:
            raise ValueError("PR96_TOKEN_SOURCE_OR_DENOMINATOR_MISMATCH")
        if (record.get("truncation_count") != 0 or record.get("n_windowed_source_rows") != 0
                or record.get("min_supervised_tokens", 0) < 1
                or record.get("assistant_mask_methods") != ["qwen35_owned_xml_tool_span_v1"]):
            raise ValueError("PR96_TOKEN_MASK_OR_STATE_CROPPING_ERROR")


def validate_parent(parent, gate, *, track, weight_sha, cohort_sha):
    if (parent.get("track") != track or parent.get("stage") != "state"
            or parent.get("lineage", {}).get("base_revision") != OFFICIAL_REVISION
            or parent.get("files", {}).get("adapter_model.safetensors") != weight_sha):
        raise ValueError("PR96_STAGE_TWO_WRONG_PARENT")
    if (gate.get("track") != track or gate.get("stage") != "state"
            or gate.get("adapter_model_sha256") != weight_sha
            or gate.get("cohorts_sha256") != cohort_sha or gate.get("base_revision") != OFFICIAL_REVISION):
        raise ValueError("PR96_STAGE_TWO_GATE_CHECKPOINT_MISMATCH")
    if not stage_one_gate(gate.get("metrics", {}))["go"]:
        raise ValueError("PR96_STAGE_TWO_GATE_NO_GO")


def verify_training_bindings(cfg, cache):
    import json
    from pathlib import Path
    from .pr96_eval_runtime import sha
    settings = cfg["pr96"]
    approval_path = Path(settings["reference_approval"])
    if sha(approval_path) != settings["reference_approval_sha256"]:
        raise ValueError("PR96_REFERENCE_APPROVAL_CHANGED")
    approval = json.loads(approval_path.read_text())
    if approval.get("track") != settings["track"] or not approval.get("reference_training_authorized"):
        raise ValueError("PR96_UNAUTHORIZED_DATA_OR_TRACK")
    for name in ("reference", "history"):
        if sha(approval[f"{name}_manifest"]) != approval[f"{name}_manifest_sha256"]:
            raise ValueError("PR96_REFERENCE_PARENT_CHANGED")
    reference = json.loads(Path(approval["reference_manifest"]).read_text())
    history = json.loads(Path(approval["history_manifest"]).read_text())
    expected_train = (reference["reports"]["train"]["outputs"]["state_sft"]["sha256"]
                      if settings["stage"] == "state" else history["reports"]["train"]["sha256"])
    if (sha(cfg["train_file"]) != expected_train
            or sha(cfg["validation_file"]) != approval["validation_views"][settings["stage"]]["sha256"]):
        raise ValueError("PR96_TRAINING_FILES_NOT_THE_AUTHORIZED_SUPERVISION")
    if sha(settings["cohorts_file"]) != settings["cohorts_sha256"] or settings["cohorts_sha256"] != approval["frozen_cohorts_sha256"]:
        raise ValueError("PR96_VALID_COHORTS_CHANGED")
    if sha(settings["base_manifest"]) != settings["base_manifest_sha256"]:
        raise ValueError("PR96_OFFICIAL_BASE_MANIFEST_CHANGED")
    if sha(settings["preflight_manifest"]) != settings["preflight_manifest_sha256"]:
        raise ValueError("PR96_PREFLIGHT_MANIFEST_CHANGED")
    preflight = json.loads(Path(settings["preflight_manifest"]).read_text())
    validate_preflight(preflight, settings["track"])
    if preflight["base_manifest_sha256"] != settings["base_manifest_sha256"]:
        raise ValueError("PR96_PREFLIGHT_BASE_MISMATCH")
    if cache is None:
        raise ValueError("PR96_COMPLETE_PRETOKENIZED_CACHE_REQUIRED")
    cache_dir = Path(cfg["pretokenized_cache_dir"])
    if sha(cache_dir / "manifest.json") != settings["token_cache_manifest_sha256"]:
        raise ValueError("PR96_TOKEN_MANIFEST_CHANGED")
    for split in cache["splits"].values():
        for name, digest in split["arrow_sha256"].items():
            if sha(cache_dir / name) != digest:
                raise ValueError("PR96_TOKEN_SHARD_CHANGED")
    validate_tokens(cache, {name:(sha(cfg[key]), cfg["contract"][count]) for name,key,count in
                            (("train", "train_file", "expected_train_rows"),
                             ("validation", "validation_file", "expected_validation_rows"))})
    if settings["stage"] == "history":
        adapter = Path(cfg["initial_adapter_path"])
        parent = json.loads((adapter / "pr96_checkpoint_manifest.json").read_text())
        if parent["lineage"].get("reference_approval_sha256") != settings["reference_approval_sha256"]:
            raise ValueError("PR96_STAGE_TWO_PARENT_DATA_MISMATCH")
        validate_parent(parent,
                        json.loads(Path(settings["stage_one_gate_report"]).read_text()),
                        track=settings["track"], weight_sha=sha(adapter / "adapter_model.safetensors"),
                        cohort_sha=settings["cohorts_sha256"])
