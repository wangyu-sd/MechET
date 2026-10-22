"""Fail-closed bindings between PR96 references, token caches and own parents."""
from .qwen35_protocol import OFFICIAL_MODEL, OFFICIAL_REVISION
from .pr96_validation import stage_one_gate


def validate_preflight(value, track):
    required = ("finite_nonzero_gradients", "parameters_updated", "assistant_mask_verified",
                "tool_prefix_parity", "all_language_targets_present", "language_only", "generation_completed")
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
    preflight = json.loads(Path(settings["preflight_manifest"]).read_text())
    validate_preflight(preflight, settings["track"])
    if preflight["base_manifest_sha256"] != settings["base_manifest_sha256"]:
        raise ValueError("PR96_PREFLIGHT_BASE_MISMATCH")
    if cache is None:
        raise ValueError("PR96_COMPLETE_PRETOKENIZED_CACHE_REQUIRED")
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
